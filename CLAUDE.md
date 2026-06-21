# DEBRIS — Disaster Environment for Blind-assistive Reinforcement learning & Intelligent Systems

## What this project is

DEBRIS trains a navigation policy entirely from physics reward signals with no human demonstrations to navigate a collapsing building corridor with falling debris, crumbling ceiling tiles, collapsing wall sections, scatter shrapnel, and fire hazard zones. The trained policy demonstrates that RL produces collision-aware navigation from pure sim experience, directly applicable to assistive devices for visually impaired individuals and autonomous vehicle hazard avoidance.

This is a HUD/YC RSI hackathon project built on top of hud-evals/worldsim-template. It runs in 18 real build hours and demos as: base policy vs trained policy side-by-side on the same disaster scene, held-out eval showing success rate and collision count curves across curriculum stages, and live judge-controlled novel scene injection.

---

## Platform Stack

### HUD (hud.ai) — eval harness and environment orchestration, NOT the trainer

Install: `uv add 'hud-python[robot]'`

HUD runs episodes, scores them, streams telemetry to hud.ai, and records LeRobot v3 datasets. It does not do gradient updates.

Key classes:

- `RobotBridge`: subclass with `reset()`, `step()`, `get_observation()`. Wraps the Newton sim. Framework owns the WebSocket serve loop.
- `RobotEndpoint`: control handle — start, reset, result, stop. Called from env.py initialize/template/shutdown hooks.
- `RobotAgent`: agent-side harness. Owns the observe/act loop. Subclass with Model and Adapter.
- `Model`: implement `infer(batch) -> np.ndarray` returning `[T, A]` float32 action chunk.
- `Adapter`: translates env obs space to policy input space from the JSON contract.
- `Contract`: `contract.json` — both env and agent read this. Defines `control_rate`, observation feature names/roles/types, action feature names. Agent wires itself purely from this file.
- `BatchedAgent`: wraps RobotAgent for N concurrent rollouts against one batched GPU forward. Use `batch_size=16`.
- Dataset recording: set `agent.save=True` and `RECORD_DIR=./data` to auto-record every (obs, action) tick to LeRobot v3 dataset.
- HUD CLI: `hud eval environment/tasks.py --all --group 5`

---

### Antim Labs — Newton Physics + Gizmo Scene Generator

- Scenes live at `scenes/<scene-name>/scene.xml` + `scenes/<scene-name>/metadata.json`
- Newton engine runs as a subprocess via `sim/server.py` inside worldsim-template.
- Scene XML is MuJoCo-compatible: `worldbody`, `body`, `geom`, `joint`, `actuator`, `sensor` tags.
- Free joints (`freejoint`) make bodies fully dynamic under gravity.
- `WORLDSIM_VIEWER=1` env var opens live 3D viewer.
- Sensor data via `sim.get_sensor_data()` returns dict of named numpy arrays.
- `SimHost` and `SimBridge` in worldsim-template handle the Newton subprocess.

**CRUMBLING PHYSICS — key technique:**
MuJoCo/Newton equality weld constraints can be toggled at runtime:
```python
model.eq_active[i] = 0  # breaks the weld → body falls under gravity
```
Bodies with `freejoint` are held up by weld constraints. Python bridge releases them on a schedule. This gives us real physics crumbling — 300-400kg concrete slabs fall, bounce, slide.
See: https://github.com/google-deepmind/mujoco/issues/525

**Gizmo** (gizmo.antimlabs.com): text prompt → SimReady Newton scene → folder with `scene.xml` + `metadata.json`. Drop directly into `scenes/`.

---

### Modal (modal.com) — GPU compute for PPO training

Modal is NOT involved in the Newton sim (CPU-bound) or HUD eval. It runs the PPO training loop on A100s.

Key patterns:

```python
import modal
app = modal.App("debris-training")
volume = modal.Volume.from_name("debris-checkpoints", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.3.0", "numpy", "tensorboard")
    .pip_install("hud-python[robot]")
)
project_mount = modal.Mount.from_local_dir(
    ".", remote_path="/debris",
    condition=lambda p: not any(x in p for x in [".git", "__pycache__", "checkpoints", ".venv"]),
)

@app.function(
    gpu="A100",
    image=image,
    timeout=72000,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=0.0),
    volumes={"/checkpoints": volume},
    mounts=[project_mount],
)
def train_stage(curriculum_stage: str, total_steps: int, load_from_stage: str | None = None):
    ...

@app.local_entrypoint()
def main():
    train_stage.spawn("static_only", 300_000, None).get()
```

Run with: `modal run training/modal_train.py`
Use `.spawn(...).get()` not `.remote()` — `.remote()` Function Calls expire after 24h.
Checkpoints save to `/checkpoints/` in Modal volume via `volume.commit()`.

---

## Repository Layout

Built ON TOP of hud-evals/worldsim-template. Clone that first, then add DEBRIS files.

```
worldsim-template/
├── CLAUDE.md                           <- THIS FILE
├── scenes/
│   └── disaster-corridor-v2/
│       ├── scene.xml                   <- Newton physics scene (MuJoCo XML)
│       └── metadata.json               <- reward config, curriculum, obs/act spec
├── environment/
│   ├── disaster_bridge_v2.py           <- RobotBridge subclass
│   ├── contract.json                   <- HUD contract (obs/act space)
│   └── tasks.py                        <- extend with disaster nav tasks
├── agents/
│   └── debris_agent.py                 <- RobotAgent with trained MLP policy
├── training/
│   ├── policy.py                       <- PolicyNet (MLP actor-critic, OBS_DIM=59)
│   ├── ppo.py                          <- PPO algorithm + rollout buffer
│   ├── curriculum.py                   <- curriculum stage manager
│   └── modal_train.py                  <- Modal GPU training entrypoint
├── eval/
│   ├── run_eval.py                     <- base vs trained comparison runner
│   └── plot_results.py                 <- demo plots
└── checkpoints/                        <- local checkpoint dir (gitignored)
```

---

## Disaster Physics — 4 Hazard Layers

### Layer 1: Primary Falling Debris

- 5 heavy bodies (18-30kg), freejoint, spawn at 6-9m height, fall under gravity.
- Respawn from random x/y position after landing (`respawn_delay_steps` from metadata).
- Body names: `debris_primary_0` through `debris_primary_4`

### Layer 2: Scatter Shrapnel

- 5 lightweight bodies (2-5kg), freejoint, spawn at 4.5-7m with random lateral velocity kicks.
- Simulate chip-off fragments bouncing off walls.
- Body names: `debris_scatter_0` through `debris_scatter_4`

### Layer 3: Ceiling Crumble Wave (weld-break physics)

5 ceiling tiles (180-420kg each), each held by a weld equality constraint.

Weld index map (must match scene.xml equality order):
- `weld_ceil_0`: index 0  (start zone, breaks at step 200 or agent_x > -6.0)
- `weld_ceil_1`: index 1  (breaks at step 400 or agent_x > -2.0)
- `weld_ceil_2`: index 2  (breaks at step 550 or agent_x > 2.0)
- `weld_ceil_3`: index 3  (breaks at step 700 or agent_x > 5.0)
- `weld_ceil_4`: index 4  (breaks at step 900 or agent_x > 9.0)

Wall chunks:
- `weld_wall_L0`: index 5 (breaks at step 350 or agent_x > -2.0)
- `weld_wall_L1`: index 6 (breaks at step 650 or agent_x > 5.0)
- `weld_wall_R0`: index 7 (breaks at step 500 or agent_x > 0.0)
- `weld_wall_R1`: index 8 (breaks at step 800 or agent_x > 7.0)

Total 9 weld constraints, all active at episode start, released by bridge.

### Layer 4: Fire Hazard Zones

3 static zones at: `[-6.0, 1.0, 0.5]`, `[-1.5, -1.5, 0.5]`, `[4.5, 1.8, 0.5]`
Radius 0.65m. No physics collision — reward function checks agent proximity.
Bridge sets `fire_zone_flag=1` in observation when agent is inside any zone.

---

## Observation Space (dim=71)

```
[0:3]    agent_pos               xyz world frame
[3:6]    agent_vel               vx vy vz
[6:9]    agent_facing            forward unit vector
[9:12]   goal_relative_pos       goal_pos - agent_pos
[12:27]  primary_debris_pos      5 × xyz flattened
[27:32]  primary_debris_vz       5 vertical velocities (negative = falling toward agent)
[32:47]  scatter_debris_pos      5 × xyz flattened
[47:52]  scatter_debris_vz       5 vertical velocities
[52:57]  primary_debris_vx       5 lateral x velocities — predicts landing zone
[57:62]  primary_debris_vy       5 lateral y velocities — predicts landing zone
[62]     time_remaining_norm     (max_steps - step) / max_steps, 1.0=start 0.0=end
[63]     heading_to_goal_angle   angle between facing and goal dir in radians, 0=facing goal
[64:69]  ceil_broken_flags       5 booleans, 1=tile broken and falling
[69]     fire_zone_flag          1 if agent in any fire zone
[70]     collision_flag          1 if debris contact force > threshold
```

No RGB camera. Policy navigates using position/velocity/proximity only — analogous to LIDAR/sonar. This IS the blind navigation framing.

## Action Space (dim=3)

```
[0]  vx_target   forward/back  [-1.5, 1.5] m/s
[1]  vy_target   left/right    [-1.0, 1.0] m/s
[2]  wz_target   yaw           [-1.0, 1.0] rad/s
```

Policy outputs in [-1,1] after tanh, then scaled to physical ranges.

---

## Reward Function

```python
reward = 0.0
progress = prev_dist_to_goal - current_dist_to_goal
reward += 0.1 * progress                          # dense progress shaping, always on

if dist_to_goal < 0.9:    reward += 25.0          # goal bonus — increased from 10
if collision (EDGE ONLY):  reward += -3.0          # only fires on first frame of contact
if fire_zone_flag:         reward += -2.0          # per step in fire zone
if overhead_debris_cone:   reward += -0.5          # debris above + falling fast + overhead
reward += -0.002                                   # time penalty — reduced from 0.005
```

COLLISION EDGE DETECTION: collision penalty fires only on the rising edge of contact
(first frame), not every frame. Prevents large sustained penalties from a single hit.
Tracked via `was_in_contact_last_step` boolean in episode state.

OVERHEAD CONE: replaces near-miss sphere. Only penalizes debris that is:
  - More than 1.0m above the agent
  - Within 1.2m horizontal radius
  - Falling faster than 2.0 m/s downward

Collision penalties disabled entirely in `static_only` stage so agent
learns navigation before learning avoidance.

All coefficients live in `metadata.json`. Never hardcode them in bridge code.

---

## Curriculum (4 stages)

| Stage | Name | Debris | Scatter | Crumble | Steps |
|-------|------|--------|---------|---------|-------|
| 1 | static_only | 0 | 0 | no | 300k |
| 2 | debris_only | 3 | 0 | no | 600k |
| 3 | debris_and_scatter | 5 | 3 | no | 900k |
| 4 | full_chaos | 5 | 5 | yes | 1.2M |

Each stage loads from the previous stage's checkpoint.

---

## Policy Architecture

```
PolicyNet:
  shared: Linear(71, 256) -> LayerNorm -> Tanh -> Linear(256, 256) -> LayerNorm -> Tanh
  actor_mean:    Linear(256, 3)
  actor_log_std: Parameter(3,)   learnable
  critic:        Linear(256, 1)
```

OBS_DIM=71, ACT_DIM=3. Orthogonal init. PPO: clip_eps=0.2, GAE-lambda=0.95, gamma=0.99.
16 parallel envs, rollout_len=1024, batch_size=256, n_epochs=10, lr=3e-4.
Entropy coefficient annealed from 0.05 → 0.005 over training.

---

## Key Invariants — Claude Code must never violate these

1. **OBS_DIM=71 everywhere.** `policy.py`, `ppo.py`, `contract.json`, `disaster_bridge_v2.py` must all agree.
2. **Never import from `sim/` directly in `training/`.** Training uses `DisasterCorridorEnv` which uses `SimBridge` internally.
3. **`contract.json` is the source of truth for obs/act feature names.** Names in `get_observation()` must match exactly.
4. **All reward coefficients live in `metadata.json`.** Read them at init. Never hardcode numbers in bridge code.
5. **Modal training does NOT import HUD.** `ppo.py` and `policy.py` are pure PyTorch + numpy.
6. **Curriculum stage is a parameter not a code branch.** `DisasterBridgeV2(stage=...)` reads config from `metadata.json`.
7. **Checkpoints must be curriculum-tagged:** `checkpoints/stage_{name}/policy_final.pt`
8. **`WORLDSIM_VIEWER=1` must never be set in Modal** — no display in container.
9. **Weld index map in bridge must match equality constraint order in `scene.xml` exactly.**

---

## Environment Variables

```
HUD_API_KEY=...          required for hud eval and trace streaming
WORLDSIM_VIEWER=1        optional: live 3D Newton viewer (local only, never Modal)
RECORD_DIR=./data        optional: record LeRobot v3 dataset during HUD eval
HF_REPO=your/repo        optional: push recorded dataset to HuggingFace
HF_TOKEN=...             required if HF_REPO set
MODAL_TOKEN_ID=...       set via: modal token new
MODAL_TOKEN_SECRET=...
```

---

## Install

```bash
git clone https://github.com/hud-evals/worldsim-template
cd worldsim-template
uv sync --extra robot --extra viewer
hud set HUD_API_KEY=your-key-here
python scripts/check_setup.py
pip install modal
modal token new
```

---

## What does NOT exist yet (the parallel terminals build these)

- [ ] `scenes/disaster-corridor-v2/scene.xml`
- [ ] `scenes/disaster-corridor-v2/metadata.json`
- [ ] `environment/disaster_bridge_v2.py`
- [ ] `environment/contract.json`
- [ ] `environment/tasks.py` extensions
- [ ] `agents/debris_agent.py`
- [ ] `training/policy.py`
- [ ] `training/ppo.py`
- [ ] `training/curriculum.py`
- [ ] `training/modal_train.py`
- [ ] `eval/run_eval.py`
- [ ] `eval/plot_results.py`
