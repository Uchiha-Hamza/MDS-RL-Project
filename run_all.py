"""
One-click script: train all agents and generate all result plots.

Usage:
    python run_all.py
    python run_all.py --episodes 1000    (quick run)
    python run_all.py --episodes 5000    (full run)
"""

import os
import sys
import time
import argparse

from train import train_agent, save_results
from evaluate import generate_all_plots
from config import AGENT_CONFIGS
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Run complete RL experiment pipeline")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count for all agents (default: use config values)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only generate plots from existing results")
    parser.add_argument("--agents", type=str, nargs="+", default=None,
                        help="Train only specific agents (e.g. --agents q_learning dqn)")
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    agent_names = args.agents if args.agents else list(AGENT_CONFIGS.keys())

    # ─── Phase 1: Training ───────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("  PHASE 1: TRAINING ALL AGENTS")
        print("=" * 60)
        start_time = time.time()

        for i, agent_name in enumerate(agent_names, 1):
            print(f"\n{'-' * 50}")
            print(f"  [{i}/{len(agent_names)}] Training: {agent_name}")
            print(f"{'-' * 50}")

            history = train_agent(
                agent_name,
                episodes=args.episodes,
                viewport_pattern="linear",
                seed=args.seed,
                verbose=True,
            )
            save_results(history, results_dir=results_dir)
            print(f"  > Best avg reward: {history['best_avg_reward']:.2f}")

        elapsed = time.time() - start_time
        print(f"\n  Total training time: {elapsed / 60:.1f} minutes")

    # ─── Phase 2: Evaluation & Plots ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: GENERATING VISUALIZATIONS")
    print("=" * 60)

    generate_all_plots(results_dir)

    # ─── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n  Results saved to: {os.path.abspath(results_dir)}/")
    print(f"\n  Generated files:")

    for f in sorted(os.listdir(results_dir)):
        fpath = os.path.join(results_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"    [F] {f} ({size:.0f} KB)")
        elif os.path.isdir(fpath):
            n_files = len(os.listdir(fpath))
            print(f"    [D] {f}/ ({n_files} files)")

    print()


if __name__ == "__main__":
    main()
