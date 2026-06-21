"""
Usage:
    python eval/run_eval.py \
        --trained checkpoints/stage_full_chaos/policy_final.pt \
        --n-seeds 20 \
        --stage full_chaos \
        --output eval/results.json

Runs each seed twice: trained policy + random/base policy.
Saves per-episode JSON and prints summary table.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MAX_STEPS = 2000

# Observation indices per contract (OBS_DIM=59)
IDX_GOAL_REL = slice(9, 12)
IDX_FIRE_FLAG = 57
IDX_COLLISION_FLAG = 58


def run_episode(bridge, agent, max_steps: int = MAX_STEPS) -> dict:
    obs_dict = bridge.reset()
    agent.model.reset()

    collisions = 0
    fire_hits = 0
    success = False
    step = 0

    for step in range(1, max_steps + 1):
        adapted = agent.adapter.adapt_observation(obs_dict, spaces=None)
        action_raw = agent.model.infer(adapted)
        action = agent.adapter.adapt_action(action_raw, spaces=None)

        obs_dict, _reward, done, _info = bridge.step(action.flatten())

        state = obs_dict["observation/state"]
        if state[IDX_COLLISION_FLAG] > 0.5:
            collisions += 1
        if state[IDX_FIRE_FLAG] > 0.5:
            fire_hits += 1

        goal_rel = state[IDX_GOAL_REL]
        dist_to_goal = float(np.linalg.norm(goal_rel))
        if dist_to_goal < 0.9:
            success = True
            done = True

        if done:
            break

    return {
        "success": success,
        "collisions": collisions,
        "fire_hits": fire_hits,
        "steps": step,
    }


def print_summary(trained_results: list, base_results: list, stage: str, n_seeds: int):
    def stats(results):
        successes = [r["success"] for r in results]
        cols = [r["collisions"] for r in results]
        steps = [r["steps"] for r in results]
        return (
            100.0 * sum(successes) / len(successes),
            sum(cols) / len(cols),
            sum(steps) / len(steps),
        )

    t_succ, t_col, t_steps = stats(trained_results)
    b_succ, b_col, b_steps = stats(base_results)

    header = f"DEBRIS Eval Results — stage={stage}, N={n_seeds} seeds"
    sep = "─" * 51
    print(f"\n  {header}")
    print(f"  {sep}")
    print(f"  {'Policy':<14} {'Success%':>9}   {'Avg Collisions':>14}   {'Avg Steps':>9}")
    print(f"  {sep}")
    print(f"  {'Trained':<14} {t_succ:>8.1f}%   {t_col:>14.1f}   {t_steps:>9.0f}")
    print(f"  {'Base/Random':<14} {b_succ:>8.1f}%   {b_col:>14.1f}   {b_steps:>9.0f}")
    print(f"  {sep}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", required=True, help="Path to trained checkpoint .pt")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--stage", default="full_chaos")
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from environment.disaster_bridge_v2 import DisasterBridgeV2
    except ImportError as e:
        print(
            f"ERROR: Could not import DisasterBridgeV2 from environment.disaster_bridge_v2.\n"
            f"Make sure you are running from the project root and environment/ is set up.\n"
            f"Details: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    from agents.debris_agent import DebrisAgent, RandomAgent

    trained_agent = DebrisAgent(args.trained, device=args.device)
    base_agent = RandomAgent()

    trained_results = []
    base_results = []

    for seed in range(args.n_seeds):
        print(f"Seed {seed:02d}/{args.n_seeds - 1} — trained ...", end=" ", flush=True)
        t_bridge = DisasterBridgeV2(stage=args.stage, seed=seed)
        t_ep = run_episode(t_bridge, trained_agent)
        t_ep["seed"] = seed
        trained_results.append(t_ep)
        print(f"{'OK' if t_ep['success'] else 'FAIL'} ({t_ep['steps']} steps, {t_ep['collisions']} col) | base ...", end=" ", flush=True)

        b_bridge = DisasterBridgeV2(stage=args.stage, seed=seed)
        b_ep = run_episode(b_bridge, base_agent)
        b_ep["seed"] = seed
        base_results.append(b_ep)
        print(f"{'OK' if b_ep['success'] else 'FAIL'} ({b_ep['steps']} steps, {b_ep['collisions']} col)")

    output_data = {
        "trained": trained_results,
        "base": base_results,
        "metadata": {
            "stage": args.stage,
            "n_seeds": args.n_seeds,
            "checkpoint": args.trained,
            "max_steps": MAX_STEPS,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {out_path}")

    print_summary(trained_results, base_results, args.stage, args.n_seeds)

    print("Per-seed breakdown:")
    print(f"  {'Seed':>4}  {'Trained':>8}  {'T-col':>5}  {'T-steps':>7}  {'Base':>8}  {'B-col':>5}  {'B-steps':>7}")
    for t, b in zip(trained_results, base_results):
        seed = t["seed"]
        print(
            f"  {seed:>4}  {'OK' if t['success'] else 'FAIL':>8}  {t['collisions']:>5}  {t['steps']:>7}"
            f"  {'OK' if b['success'] else 'FAIL':>8}  {b['collisions']:>5}  {b['steps']:>7}"
        )


if __name__ == "__main__":
    main()
