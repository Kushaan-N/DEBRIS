"""
training/modal_train.py — Modal GPU training entrypoint for DEBRIS

Run all 4 curriculum stages:
  modal run training/modal_train.py

Run single stage:
  modal run training/modal_train.py::train_single --stage full_chaos --steps 1200000

Local dev (no Modal, no Newton sim — mock envs):
  python training/modal_train.py --local --stage static_only --steps 50000
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app + infrastructure
# ---------------------------------------------------------------------------

app = modal.App("debris-training")
volume = modal.Volume.from_name("debris-checkpoints", create_if_missing=True)
VOLUME_PATH = Path("/checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.3.0", "numpy", "tensorboard")
    .pip_install("hud-python[robot]")
)

project_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/debris",
    condition=lambda p: not any(
        x in p for x in [".git", "__pycache__", "checkpoints", ".venv", "data"]
    ),
)

# ---------------------------------------------------------------------------
# Mock environment — used when Newton/HUD not available (local dev, CI)
# ---------------------------------------------------------------------------

class MockDisasterEnv:
    """Minimal stand-in for DisasterBridgeV2 that needs no Newton sim."""

    OBS_DIM = 59

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

def collect_rollout(policy, envs, buffer, device) -> "np.ndarray":
    """
    Run rollout_len=512 steps across all envs and fill buffer.
    Returns last_values array for GAE bootstrap (shape: [N_ENVS]).
    """
    import numpy as np
    import torch

    rollout_len = buffer.rollout_len
    n_envs = len(envs)

    for _ in range(rollout_len):
        obs_np = np.stack([e.last_obs for e in envs])  # [N, 59]
        obs_t = torch.from_numpy(obs_np).float().to(device)

        with torch.no_grad():
            actions, log_probs, values = policy.get_action(obs_t)

        actions_np = actions.cpu().numpy()  # [N, 3] — already scaled by policy
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy().flatten()

        rewards = np.zeros(n_envs, dtype=np.float32)
        dones = np.zeros(n_envs, dtype=np.float32)
        next_obs_np = np.zeros_like(obs_np)

        for i, env in enumerate(envs):
            _, reward, done, _ = env.step(actions_np[i])
            rewards[i] = reward
            dones[i] = float(done)
            if done:
                obs_dict = env.reset()
                env.last_obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
            next_obs_np[i] = env.last_obs

        buffer.add(
            obs=obs_np,
            action=actions_np,
            log_prob=log_probs_np,
            reward=rewards,
            done=dones,
            value=values_np,
        )

    # Bootstrap values for last observation
    last_obs_t = torch.from_numpy(np.stack([e.last_obs for e in envs])).float().to(device)
    with torch.no_grad():
        _, _, last_values = policy.get_action(last_obs_t)
    return last_values.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Core training logic (shared between Modal and local paths)
# ---------------------------------------------------------------------------

def _run_training(
    curriculum_stage: str,
    total_steps: int,
    load_from_stage: str | None,
    volume_path: str,
    debris_root: str,
    use_mock: bool = False,
) -> None:
    import sys
    import numpy as np
    import torch

    sys.path.insert(0, debris_root)

    # -- choose env class --
    EnvClass = MockDisasterEnv
    if not use_mock:
        try:
            from environment.disaster_bridge_v2 import DisasterBridgeV2
            EnvClass = DisasterBridgeV2
            print("Using DisasterBridgeV2 (Newton sim)")
        except ImportError:
            print("DisasterBridgeV2 not found — falling back to MockDisasterEnv")

    from training.policy import PolicyNet, save_policy, load_policy
    from training.ppo import RolloutBuffer, PPOUpdater
    from training.curriculum import CurriculumManager

    N_ENVS = 16
    LOG_INTERVAL = 10_000
    SAVE_INTERVAL = 50_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Stage: {curriculum_stage} | Total steps: {total_steps}")

    policy = PolicyNet().to(device)
    updater = PPOUpdater(policy)

    metadata_path = str(Path(debris_root) / "scenes" / "disaster-corridor-v2" / "metadata.json")
    curriculum = CurriculumManager(metadata_path, volume_path)
    stage_cfg = curriculum.get_stage_config(curriculum_stage)
    print(f"Stage config: {stage_cfg}")

    if load_from_stage is not None:
        try:
            curriculum.load_checkpoint(policy, updater.optimizer, load_from_stage)
            print(f"Loaded checkpoint from stage: {load_from_stage}")
        except Exception as exc:
            print(f"Warning: could not load checkpoint from {load_from_stage}: {exc}")

    # Create envs
    envs = [EnvClass(stage=curriculum_stage, seed=i) for i in range(N_ENVS)]

    # Init last_obs
    for env in envs:
        obs_dict = env.reset()
        env.last_obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict

    buffer = RolloutBuffer(N_ENVS, 1024, obs_dim=71)

    global_step = 0
    all_ep_rewards: list[float] = []
    all_collisions: list[int] = []
    t_start = time.time()
    last_log_step = 0

    while global_step < total_steps:
        last_values = collect_rollout(policy, envs, buffer, device)
        # Anneal entropy coefficient from 0.05 → 0.005 over training
        ent_coef_current = max(
            updater.ent_coef_final,
            updater.ent_coef * (1.0 - global_step / max(total_steps, 1)),
        )
        loss_info = updater.update(buffer, last_values, device, ent_coef=ent_coef_current)
        global_step += N_ENVS * buffer.rollout_len

        # Collect episode stats from envs that finished this rollout
        for env in envs:
            ep_info = env.score() if hasattr(env, "_step_count") else {}
            if ep_info:
                all_ep_rewards.append(float(ep_info.get("reward", 0.0)))
                all_collisions.append(int(ep_info.get("collisions", 0)))

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
            print(f"Checkpoint saved at step {global_step}")

    # Final checkpoint
    curriculum.save_checkpoint(policy, updater.optimizer, global_step, curriculum_stage, tag="final")
    print(f"Final checkpoint saved for stage: {curriculum_stage}")


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    image=image,
    timeout=72000,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=0.0),
    volumes={str(VOLUME_PATH): volume},
    mounts=[project_mount],
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


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    """Run all 4 curriculum stages sequentially on Modal A100s."""
    stages = [
        ("static_only",        None,                   400_000),
        ("debris_only",        "static_only",          800_000),
        ("debris_and_scatter", "debris_only",        1_000_000),
        ("full_chaos",         "debris_and_scatter", 1_500_000),
    ]
    for stage, load_from, steps in stages:
        print(f"\n{'='*50}\nStarting stage: {stage}\n{'='*50}")
        train_stage.spawn(stage, steps, load_from).get()
        print(f"Completed stage: {stage}")


@app.local_entrypoint()
def train_single(
    stage: str = "full_chaos",
    steps: int = 1_200_000,
    load_from: str = None,
) -> None:
    """Run a single curriculum stage on a Modal A100."""
    train_stage.spawn(stage, steps, load_from).get()


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
        _run_training(
            curriculum_stage=args.stage,
            total_steps=args.steps,
            load_from_stage=args.load_from,
            volume_path=volume_path,
            debris_root=debris_root,
            use_mock=True,
        )
