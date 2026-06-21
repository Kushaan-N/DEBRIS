# -*- coding: utf-8 -*-
"""
training/modal_train.py — Modal GPU training entrypoint for DEBRIS

Two-phase recommended flow (local check then Modal):
  modal run --detach training/modal_train.py::start
  modal run --detach training/modal_train.py::run_all

Single stage on Modal:
  modal run training/modal_train.py::train_single --stage full_chaos --steps 1500000

All 4 stages on Modal (no local phase):
  modal run training/modal_train.py

Local dev (no Modal, no Newton sim — mock envs):
  python training/modal_train.py --local --stage static_only --steps 5000
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import modal
from modal import App, Volume, Image, Retries

# ---------------------------------------------------------------------------
# Modal app + infrastructure
# ---------------------------------------------------------------------------

app = App("debris-training")
volume = Volume.from_name("debris-checkpoints", create_if_missing=True)
VOLUME_PATH = Path("/checkpoints")

# In Modal 1.5.0 local code is bundled into the image via .add_local_dir()
# instead of a separate Mount passed to mounts=[].
image = (
    Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.3.0", "numpy", "tensorboard", "mujoco")
    .pip_install("hud-python[robot]")
    .add_local_dir(
        ".",
        remote_path="/debris",
        copy=True,  # required when run_commands follows add_local_dir
        # Check path *components* not substring — "data" in str(p) would
        # accidentally exclude "metadata.json" (contains "data").
        ignore=lambda p: any(
            x in p.parts for x in [".git", "__pycache__", "checkpoints", ".venv", "data"]
        ),
    )
    .run_commands(
        # Install the Newton wheel bundled with worldsim-template.
        "pip install /debris/wheels/newton-*.whl || echo 'Newton wheel install failed'",
        # Add /debris to Python path — use Python to find site-packages rather
        # than hardcoding the version string.
        "python3 -c \"import sys; open(next(p for p in sys.path if 'site-packages' in p) + '/debris.pth', 'w').write('/debris\\n')\"",
    )
)

# ---------------------------------------------------------------------------
# Async-safe reset helper
# ---------------------------------------------------------------------------

def _sync_reset(env, task_id: str = "", seed: int = 0):
    """
    Call env.reset() whether it is sync or async.
    DisasterBridgeV2.reset() is async (HUD RobotBridge interface).
    MockDisasterEnv.reset() is sync.
    Returns the obs dict in both cases.
    """
    if hasattr(env, "reset_sync"):
        result = env.reset_sync(task_id=task_id, seed=seed)
    else:
        result = env.reset()
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, result)
                    result = future.result()
            else:
                result = loop.run_until_complete(result)
        except RuntimeError:
            result = asyncio.run(result)
    return result


# ---------------------------------------------------------------------------
# Mock environment — used when Newton/HUD not available (local dev, CI)
# ---------------------------------------------------------------------------

class MockDisasterEnv:
    """Minimal stand-in for DisasterBridgeV2 that needs no Newton sim."""

    OBS_DIM = 71

    def __init__(self, stage: str = "static_only", seed: int = 0):
        import numpy as np
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self.last_obs: "np.ndarray" = np.zeros(self.OBS_DIM, dtype=np.float32)

    def reset(self) -> dict:
        import numpy as np
        self._step_count = 0
        self.last_obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        return {"obs": self.last_obs.copy()}

    def step(self, action: "np.ndarray") -> tuple[dict, float, bool, dict]:
        import numpy as np
        self._step_count += 1
        reward = float(self.rng.uniform(-0.1, 0.3))
        done = self._step_count >= 500
        self.last_obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        return {"obs": self.last_obs.copy()}, reward, done, {}

    def score(self) -> dict:
        return {"success": False, "collisions": 0, "steps": self._step_count}


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    policy, envs, buffer, device
) -> "tuple[np.ndarray, list[float], list[int]]":
    """
    Run rollout_len steps across all envs and fill buffer.

    Returns:
        last_values       — shape [N_ENVS], for GAE bootstrap
        completed_rewards — per-episode total rewards for episodes that finished
        completed_collisions — per-episode collision counts for finished episodes
    """
    import numpy as np
    import torch

    buffer.reset()
    rollout_len = buffer.rollout_len
    n_envs = len(envs)

    # Per-env episode accumulators — persist across the rollout loop
    ep_rewards    = [0.0] * n_envs
    ep_steps      = [0]   * n_envs
    ep_collisions = [0]   * n_envs

    completed_rewards: list[float] = []
    completed_collisions: list[int] = []

    for _ in range(rollout_len):
        obs_np = np.stack([e.last_obs for e in envs])  # [N, 71]
        obs_t = torch.from_numpy(obs_np).float().to(device)

        with torch.no_grad():
            actions, log_probs, values = policy.get_action(obs_t)

        actions_np = actions.cpu().numpy()  # [N, 3] — already scaled by policy
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy().flatten()

        rewards = np.zeros(n_envs, dtype=np.float32)
        dones   = np.zeros(n_envs, dtype=np.float32)

        for i, env in enumerate(envs):
            if hasattr(env, 'get_observation'):
                # DisasterBridgeV2: step() stores last_reward internally
                env.step(actions_np[i])
                obs_dict, done = env.get_observation()
                next_obs = obs_dict["observation/state"]
                reward = getattr(env, 'last_reward', 0.0)
                done = bool(done)
            else:
                # MockDisasterEnv: step() returns (obs, reward, done, info)
                obs_result, reward, done, _ = env.step(actions_np[i])
                next_obs = obs_result.get("obs", obs_result) if isinstance(obs_result, dict) else obs_result
                done = bool(done)

            rewards[i]     = reward
            dones[i]       = float(done)
            ep_rewards[i]  += reward
            ep_steps[i]    += 1
            env.last_obs    = next_obs

            if done:
                ep_coll = getattr(env, 'collision_count', ep_collisions[i])
                completed_rewards.append(ep_rewards[i])
                completed_collisions.append(ep_coll)
                print(
                    f"[episode] reward={ep_rewards[i]:.3f} "
                    f"collisions={ep_coll} steps={ep_steps[i]}"
                )
                ep_rewards[i]    = 0.0
                ep_steps[i]      = 0
                ep_collisions[i] = 0

                # Reset env — split by type to avoid double-reset
                if hasattr(env, 'get_observation'):
                    _sync_reset(env)
                    obs_dict, _ = env.get_observation()
                    env.last_obs = obs_dict["observation/state"]
                else:
                    obs_result = _sync_reset(env)
                    env.last_obs = (
                        obs_result.get("obs", obs_result)
                        if isinstance(obs_result, dict)
                        else obs_result
                    )

        buffer.add(
            obs=obs_np,
            action=actions_np,
            log_prob=log_probs_np,
            reward=rewards,
            done=dones,
            value=values_np,
        )

    last_obs_t = torch.from_numpy(np.stack([e.last_obs for e in envs])).float().to(device)
    with torch.no_grad():
        _, _, last_values = policy.get_action(last_obs_t)
    return last_values.cpu().numpy().flatten(), completed_rewards, completed_collisions


# ---------------------------------------------------------------------------
# Core training logic (shared between Modal and local paths)
# ---------------------------------------------------------------------------

def _run_training(
    curriculum_stage: str,
    total_steps: int,
    load_from_stage: str | None = None,
    volume_path: str | None = None,
    debris_root: str | None = None,
    use_mock: bool = True,
    start_step: int = 0,
) -> int:
    """
    Run PPO training for curriculum_stage up to total_steps.

    Returns the global step count reached (useful for two-phase handoff).
    Handles KeyboardInterrupt by saving a checkpoint and returning cleanly.
    """
    import sys
    import numpy as np
    import torch

    # Resolve paths: default to local project layout when not on Modal
    if debris_root is None:
        debris_root = str(Path(__file__).resolve().parent.parent)
    if volume_path is None:
        volume_path = str(Path(debris_root) / "checkpoints")

    sys.path.insert(0, debris_root)

    # -- choose env class --
    EnvClass = MockDisasterEnv
    if not use_mock:
        try:
            from environment.disaster_bridge_v2 import DisasterBridgeV2
            test_env = DisasterBridgeV2(stage=curriculum_stage, seed=0)
            if test_env._sim is None:
                raise RuntimeError("Newton sim not available — _sim is None")
            if hasattr(test_env, "close"):
                test_env.close()
            EnvClass = DisasterBridgeV2
            print("Using DisasterBridgeV2 with Newton sim ✓")
        except Exception as e:
            print(f"DisasterBridgeV2 not available ({e}), using MockDisasterEnv")

    from training.policy import PolicyNet
    from training.ppo import RolloutBuffer, PPOUpdater
    from training.curriculum import CurriculumManager

    N_ENVS = 16
    LOG_INTERVAL = 10_000
    SAVE_INTERVAL = 50_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Stage: {curriculum_stage} | Total steps: {total_steps:,}")

    policy = PolicyNet().to(device)
    updater = PPOUpdater(policy)

    metadata_path = str(Path(debris_root) / "scenes" / "disaster-corridor-v2" / "metadata.json")
    curriculum = CurriculumManager(metadata_path, volume_path)
    stage_cfg = curriculum.get_stage_config(curriculum_stage)
    print(f"Stage config: {stage_cfg}")

    # Load checkpoint and capture the step so the while loop resumes correctly
    global_step = start_step
    if load_from_stage is not None:
        try:
            loaded_step = curriculum.load_checkpoint(
                policy, updater.optimizer, load_from_stage
            )
            # Only carry over the step count if resuming the SAME stage
            # (mid-run interrupt recovery). Cross-stage loads just copy
            # weights — the new stage always starts its step count from 0.
            if load_from_stage == curriculum_stage:
                global_step = loaded_step
                print(f"Resumed same-stage checkpoint at step {global_step:,}")
            else:
                global_step = 0
                print(f"Loaded weights from stage={load_from_stage}, starting step count from 0")
        except FileNotFoundError:
            print(f"No checkpoint for '{load_from_stage}' — starting from scratch")
            global_step = 0
        except Exception as exc:
            print(f"Warning: checkpoint load failed ({exc}) — starting from scratch")
            global_step = 0

    # Create envs
    envs = [EnvClass(stage=curriculum_stage, seed=i) for i in range(N_ENVS)]
    for env in envs:
        _sync_reset(env)
        if hasattr(env, 'get_observation'):
            obs_dict, _ = env.get_observation()
            env.last_obs = obs_dict["observation/state"]
        else:
            obs_dict = _sync_reset(env)
            env.last_obs = obs_dict.get("obs", obs_dict) if isinstance(obs_dict, dict) else obs_dict

    buffer = RolloutBuffer(N_ENVS, 1024, obs_dim=71)

    all_ep_rewards: list[float] = []
    all_collisions: list[int] = []
    t_start = time.time()
    last_log_step = global_step

    try:
        while global_step < total_steps:
            last_values, ep_rews, ep_cols = collect_rollout(policy, envs, buffer, device)
            all_ep_rewards.extend(ep_rews)
            all_collisions.extend(ep_cols)

            # Anneal entropy coefficient from 0.05 → 0.005 over training
            ent_coef_current = max(
                updater.ent_coef_final,
                updater.ent_coef * (1.0 - global_step / max(total_steps, 1)),
            )
            loss_info = updater.update(buffer, last_values, device, ent_coef=ent_coef_current)
            global_step += N_ENVS * buffer.rollout_len

            if global_step - last_log_step >= LOG_INTERVAL:
                elapsed = time.time() - t_start
                fps = global_step / max(elapsed, 1e-6)
                mean_rew = float(np.mean(all_ep_rewards[-50:])) if all_ep_rewards else float("nan")
                mean_col = float(np.mean(all_collisions[-50:])) if all_collisions else float("nan")
                print(
                    f"step={global_step:>8d} | "
                    f"rew={mean_rew:+.3f} | "
                    f"collisions={mean_col:.2f} | "
                    f"pg_loss={loss_info.get('pg_loss', float('nan')):.4f} | "
                    f"vf_loss={loss_info.get('vf_loss', float('nan')):.4f} | "
                    f"ent_coef={ent_coef_current:.4f} | "
                    f"fps={fps:.0f}"
                )
                last_log_step = global_step

            if global_step % SAVE_INTERVAL < N_ENVS * buffer.rollout_len:
                curriculum.save_checkpoint(policy, updater.optimizer, global_step, curriculum_stage)
                print(f"Checkpoint saved at step {global_step:,}")

    except KeyboardInterrupt:
        print(f"\n[interrupted] Saving checkpoint at step {global_step:,}...")
        curriculum.save_checkpoint(
            policy, updater.optimizer, global_step, curriculum_stage,
            extra_meta={"tag": "interrupted", "step": global_step},
        )
        try:
            volume.commit()
        except Exception:
            pass  # volume.commit() is a no-op locally; fine to swallow
        print(f"[interrupted] Checkpoint saved. Stopped at step {global_step:,}.")
        return global_step

    # Normal completion — final checkpoint
    curriculum.save_checkpoint(
        policy, updater.optimizer, global_step, curriculum_stage,
        extra_meta={"tag": "final"},
    )
    print(f"Final checkpoint saved for stage: {curriculum_stage} at step {global_step:,}")
    return global_step


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    image=image,
    timeout=72000,
    retries=Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=0.0),
    volumes={str(VOLUME_PATH): volume},
    env={"DEBRIS_ROOT": "/debris"},
)
def train_stage(
    curriculum_stage: str,
    total_steps: int,
    load_from_stage: str | None = None,
) -> None:
    _run_training(
        curriculum_stage=curriculum_stage,
        total_steps=total_steps,
        load_from_stage=load_from_stage,
        volume_path=str(VOLUME_PATH),
        debris_root="/debris",
        use_mock=False,
    )
    volume.commit()


@app.function(
    gpu="A100",
    image=image,
    timeout=72000,
    retries=Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=0.0),
    volumes={str(VOLUME_PATH): volume},
    env={"DEBRIS_ROOT": "/debris"},
)
def train_all_stages() -> None:
    """Run all 4 curriculum stages sequentially on a single A100. ~12-16h total."""
    stages = [
        ("static_only",        None,                   600_000),
        ("debris_only",        "static_only",        1_000_000),
        ("debris_and_scatter", "debris_only",        1_200_000),
        ("full_chaos",         "debris_and_scatter", 1_500_000),
    ]
    for stage, load_from, steps in stages:
        print(f"\n{'='*50}\nStarting stage: {stage}\n{'='*50}")
        _run_training(
            curriculum_stage=stage,
            total_steps=steps,
            load_from_stage=load_from,
            volume_path=str(VOLUME_PATH),
            debris_root="/debris",
            use_mock=False,
        )
        volume.commit()
        print(f"Completed stage: {stage}")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    """Run all 4 curriculum stages sequentially on Modal A100s (blocking)."""
    stages = [
        ("static_only",        None,                   600_000),
        ("debris_only",        "static_only",        1_000_000),
        ("debris_and_scatter", "debris_only",        1_200_000),
        ("full_chaos",         "debris_and_scatter", 1_500_000),
    ]
    for stage, load_from, steps in stages:
        print(f"\n{'='*50}\nStarting stage: {stage}\n{'='*50}")
        train_stage.spawn(stage, steps, load_from).get()
        print(f"Completed stage: {stage}")


@app.local_entrypoint()
def train_single(
    stage: str = "full_chaos",
    steps: int = 1_500_000,
    load_from: str = None,
) -> None:
    """Run a single curriculum stage on a Modal A100 (blocking)."""
    train_stage.spawn(stage, steps, load_from).get()


@app.local_entrypoint()
def start(
    stage: str = "static_only",
    local_steps: int = 2000,
    total_steps: int = 400_000,
    load_from: str = "",
) -> None:
    """
    Two-phase training for a single curriculum stage.

    Phase 1: runs locally for local_steps so you can watch for errors.
             Press Ctrl+C when happy to hand off to Modal.
    Phase 2: saves checkpoint, submits remaining steps to Modal detached.
             Safe to close laptop after handoff message prints.

    Usage:
      modal run --detach training/modal_train.py::start
      modal run --detach training/modal_train.py::start --stage debris_only --local-steps 3000 --total-steps 800000

    Without --detach the entrypoint blocks for up to 120s until the Modal container
    boots, then exits. With --detach it exits immediately after spawn().
    """
    load = load_from if load_from else None

    print("=" * 60)
    print("DEBRIS Two-Phase Training — Single Stage")
    print("=" * 60)
    print(f"  Stage:       {stage}")
    print(f"  Local steps: {local_steps:,}  <- watch for errors here")
    print(f"  Total steps: {total_steps:,}  <- remainder runs on Modal A100")
    print(f"  Load from:   {load or 'scratch'}")
    print("=" * 60)
    print()
    print("PHASE 1 — Running locally. Watch for errors.")
    print("Press Ctrl+C at any time to hand off to Modal.")
    print()

    stopped_at = 0
    try:
        stopped_at = _run_training(
            curriculum_stage=stage,
            total_steps=local_steps,
            load_from_stage=load,
        )
        print(f"\nLocal phase completed {local_steps:,} steps cleanly.")
    except KeyboardInterrupt:
        print("\nCtrl+C received outside training loop.")

    remaining = total_steps - stopped_at
    if remaining <= 0:
        print(f"All {total_steps:,} steps completed locally. Nothing to hand off.")
        return

    print()
    print("=" * 60)
    print("PHASE 2 — Handing off to Modal A100")
    print(f"  Resuming from step: {stopped_at:,}")
    print(f"  Steps remaining:    {remaining:,}")
    print("=" * 60)
    print()

    # Pass stage as load_from_stage so Modal resumes from the checkpoint we
    # just saved. The while loop condition (global_step < total_steps) handles
    # running only the remaining steps.
    call = train_stage.spawn(stage, total_steps, stage)

    dashboard_url = call.get_dashboard_url()
    print(f"Job submitted to Modal.")
    print(f"  Call ID:   {call.object_id}")
    print(f"  Dashboard: {dashboard_url}")
    print()
    print("Waiting up to 120s for Modal container to boot...")
    print("Tip: add --detach to skip this wait: modal run --detach training/modal_train.py::start")
    print()
    try:
        call.get(timeout=120)
        print("Job finished (completed within 120s — unexpected for full training).")
    except modal.exception.TimeoutError:
        # Expected: training takes hours, not seconds. Container is running.
        print("Container is running. Detaching — safe to close your laptop.")
    except Exception as exc:
        print(f"Warning: {exc}")
    print()
    print("Stream logs:  modal app logs debris-training")


@app.local_entrypoint()
def run_all(local_steps: int = 2000) -> None:
    """
    Two-phase training across all 4 curriculum stages.

    Phase 1: runs static_only locally for local_steps so you can catch errors.
    Phase 2: hands all 4 stages to Modal as one detached job. ~12-16h total.

    Usage:
      modal run --detach training/modal_train.py::run_all
      modal run --detach training/modal_train.py::run_all --local-steps 5000

    Without --detach the entrypoint blocks for up to 120s until the Modal container
    boots, then exits. With --detach it exits immediately after spawn().
    """
    print("=" * 60)
    print("DEBRIS Full Curriculum — Two-Phase")
    print("=" * 60)
    print(f"  Local warmup: {local_steps:,} steps on static_only")
    print("  Then: all 4 stages submitted to Modal A100 as one detached job")
    print("=" * 60)
    print()
    print("PHASE 1 — Local warmup. Watch for errors.")
    print("Press Ctrl+C when ready to hand off, or wait for warmup to finish.")
    print()

    try:
        _run_training(
            curriculum_stage="static_only",
            total_steps=local_steps,
            load_from_stage=None,
        )
        print(f"\nWarmup complete ({local_steps:,} steps). Handing off to Modal.")
    except KeyboardInterrupt:
        print("\nCtrl+C — handing off all stages to Modal now.")

    print()
    print("=" * 60)
    print("PHASE 2 — Submitting all 4 stages to Modal as a single detached job")
    print("=" * 60)
    print()

    call = train_all_stages.spawn()

    dashboard_url = call.get_dashboard_url()
    print(f"All 4 stages submitted to Modal.")
    print(f"  Call ID:   {call.object_id}")
    print(f"  Dashboard: {dashboard_url}")
    print()
    print("Waiting up to 120s for Modal container to boot...")
    print("Tip: add --detach to skip this wait: modal run --detach training/modal_train.py::run_all")
    print()
    try:
        call.get(timeout=120)
        print("Job finished (completed within 120s — unexpected for full curriculum).")
    except modal.exception.TimeoutError:
        # Expected: full curriculum takes 12-16h. Container is running.
        print("Container is running. Detaching — safe to close your laptop.")
    except Exception as exc:
        print(f"Warning: {exc}")
    print()
    print("Stream logs:  modal app logs debris-training")


# ---------------------------------------------------------------------------
# Local dev path — no Modal, no Newton, uses MockDisasterEnv
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DEBRIS local training (no Modal)")
    parser.add_argument("--local", action="store_true", required=True)
    parser.add_argument("--stage", default="static_only")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument(
        "--load-from",
        default=None,
        dest="load_from",
        help="Stage name to load checkpoint from (optional)",
    )
    args = parser.parse_args()

    if args.local:
        import os

        debris_root = str(Path(__file__).resolve().parent.parent)
        volume_path = str(Path(debris_root) / "checkpoints")
        os.makedirs(volume_path, exist_ok=True)

        print(f"Local training | stage={args.stage} | steps={args.steps} | root={debris_root}")
        stopped = _run_training(
            curriculum_stage=args.stage,
            total_steps=args.steps,
            load_from_stage=args.load_from,
            volume_path=volume_path,
            debris_root=debris_root,
            use_mock=True,
        )
        print(f"Stopped at step {stopped:,}")
