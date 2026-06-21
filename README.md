# DEBRIS

Disaster Environment for Blind-assistive Reinforcement learning and Intelligent Systems.

DEBRIS is a reinforcement learning agent trained to navigate collapsing buildings using only spatial sensors. No cameras. No vision. It finds the exit and guides survivors out so firefighters don't have to enter a structure that might kill them.

The agent runs on a 71-dimensional sensor array: position, velocity, debris proximity, heading to goal, and contact flags. It trained entirely from physics reward signals with no human demonstrations across 4.3 million steps on an A100 GPU.

**Demo:** https://debris-mu.vercel.app

---

## What it does

A fire breaks out in a building. The structure starts to fail. Firefighters can't safely enter. DEBRIS deploys on a ground robot or drone, navigates through falling debris and fire zones using only proximity and inertial sensors, and leads survivors to the nearest exit.

It works in total darkness, zero visibility, and active collapse because it never needed vision to begin with. The policy learned purely from physics: did it get closer to the exit, did it get hit, did it walk into fire.

---

## Results

Evaluated on the `debris_and_scatter` stage across 10 seeds:

- 10/10 seeds reached the exit
- 1 average collision per episode
- Trained policy covers 97% of the corridor vs 3% for random baseline
- 514 sim steps per second on A100

---

## How training works

Training uses PPO with a 4-stage curriculum. Each stage loads weights from the previous one.

| Stage | Hazards | Steps | End reward |
|---|---|---|---|
| static_only | empty corridor | 600k | +8.5 |
| debris_only | 3 falling slabs | 1.0M | -2.8 |
| debris_and_scatter | 5 slabs + 3 shrapnel | 1.2M | +0.17 |
| full_chaos | 5+5 debris, ceiling crumble, fire zones, wall collapse | 1.5M | -10.9 |

The idea is to get the agent navigating correctly before it has to dodge anything. The static stage alone gets it to a +8.5 reward. Then debris is added and reward dips while it relearns. By the scatter stage it has recovered. Full chaos is still hard, which is expected given all 5 hazard layers firing simultaneously.

Physics runs in MuJoCo with the `implicitfast` integrator at 0.002s timestep. Ceiling tiles are held up by weld equality constraints that get released at runtime, so they fall under real gravity. Debris bodies are 18 to 420 kg.

---

## Observation space

The 71-dimensional input has no pixels. Every sensor is something a real robot can read without a camera.

| Indices | What it is | Dims |
|---|---|---|
| 0:3 | agent position (xyz) | 3 |
| 3:6 | agent velocity (vx vy vz) | 3 |
| 6:9 | forward facing vector | 3 |
| 9:12 | vector to goal | 3 |
| 12:27 | positions of 5 primary debris bodies | 15 |
| 27:32 | vertical fall speed of primary debris | 5 |
| 32:47 | positions of 5 scatter debris bodies | 15 |
| 47:52 | vertical fall speed of scatter debris | 5 |
| 52:62 | lateral velocities of primary debris (vx, vy) | 10 |
| 62 | normalized time remaining | 1 |
| 63 | angle between facing and goal direction | 1 |
| 64:69 | ceiling tile broken flags | 5 |
| 69 | fire zone flag | 1 |
| 70 | collision flag | 1 |

Action space is 3-dimensional: forward/back velocity, lateral velocity, yaw rate. Policy outputs tanh-scaled actions in physical units.

---

## Stack

- **RL:** PPO, GAE lambda 0.95, gamma 0.99, clip 0.2, 16 parallel envs, entropy coef annealed 0.05 to 0.005
- **Physics:** MuJoCo 3.x, implicitfast integrator, velocity actuators kv=4000
- **Training:** Modal A100-40GB, 72h timeout, curriculum checkpoint transfer via Modal Volume
- **Eval harness:** HUD (hud.ai)
- **Policy:** MLP actor-critic, Linear(71 to 256) with LayerNorm and Tanh, two layers

---

## Running locally

```bash
git clone https://github.com/Kushaan-N/DEBRIS
cd DEBRIS
uv sync --extra robot
```

Run a trained policy:

```bash
PYTHONPATH=. .venv/bin/python -c "
import numpy as np, torch
from environment.disaster_bridge_v2 import DisasterBridgeV2
from training.policy import load_policy

policy = load_policy('checkpoints/stage_debris_and_scatter/policy_final.pt', 'cpu')
env = DisasterBridgeV2(stage='debris_and_scatter', seed=0)
env.reset_sync()
obs, _ = env.get_observation()
for i in range(10000):
    with torch.no_grad():
        action, _, _ = policy.get_action(
            torch.from_numpy(obs['observation/state']).unsqueeze(0),
            deterministic=True)
    env.step(action.numpy().flatten())
    obs, done = env.get_observation()
    if done:
        print(f'Done at step {i}, success={env.success}, collisions={env.collision_count}')
        break
"
```

Run training (requires Modal):

```bash
modal token new
modal run --detach training/modal_train.py::run_all
```

---

Built at the HUD / YC RSI Hackathon.
