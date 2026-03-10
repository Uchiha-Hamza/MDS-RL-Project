"""
Unified training script for all RL agents.

Usage:
    python train.py --agent q_learning --episodes 5000
    python train.py --agent dqn --episodes 3000
    python train.py --agent all   (trains all agents)
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm

from env.ad_stealth_env import AdStealthEnv
from agents import AGENT_REGISTRY
from config import AGENT_CONFIGS, ENV_CONFIG, REWARD_CONFIG, TRAIN_CONFIG
from utils import set_seed


def train_agent(agent_name, episodes=None, viewport_pattern="linear", seed=42, verbose=True):
    """
    Train a single agent and return training history.

    Returns:
        dict with keys: rewards, steps, agent_name, config
    """
    set_seed(seed)

    # Build config
    agent_config = AGENT_CONFIGS[agent_name].copy()
    if episodes is not None:
        agent_config["episodes"] = episodes
    n_episodes = agent_config["episodes"]

    env_config = {**ENV_CONFIG, **REWARD_CONFIG, "viewport_pattern": viewport_pattern}
    env = AdStealthEnv(config=env_config)

    # Create agent
    AgentClass = AGENT_REGISTRY[agent_name]
    agent = AgentClass(agent_config)

    # Training loop
    all_rewards = []
    all_steps = []
    best_avg = -float("inf")

    log_interval = TRAIN_CONFIG["log_interval"]
    smoothing = TRAIN_CONFIG["smoothing_window"]

    desc = f"Training {agent.name}"
    pbar = tqdm(range(1, n_episodes + 1), desc=desc, disable=not verbose)

    for ep in pbar:
        ep_reward, ep_steps = agent.train_episode(env)
        all_rewards.append(ep_reward)
        all_steps.append(ep_steps)

        # Logging
        if ep % log_interval == 0 or ep == n_episodes:
            recent = all_rewards[-smoothing:]
            avg_r = np.mean(recent)
            avg_s = np.mean(all_steps[-smoothing:])
            eps_str = ""
            if hasattr(agent, "epsilon"):
                eps_str = f" | ε={agent.epsilon:.3f}"
            pbar.set_postfix_str(f"avg_R={avg_r:.1f} | avg_steps={avg_s:.0f}{eps_str}")

            if avg_r > best_avg:
                best_avg = avg_r

    env.close()

    return {
        "rewards": all_rewards,
        "steps": all_steps,
        "agent_name": agent_name,
        "display_name": agent.name,
        "config": agent_config,
        "best_avg_reward": best_avg,
        "agent": agent,
    }


def save_results(history, results_dir="results"):
    """Save training history to CSV and agent checkpoint."""
    agent_name = history["agent_name"]
    agent_dir = os.path.join(results_dir, agent_name)
    os.makedirs(agent_dir, exist_ok=True)

    # Save rewards and steps as CSV
    import pandas as pd
    df = pd.DataFrame({
        "episode": list(range(1, len(history["rewards"]) + 1)),
        "reward": history["rewards"],
        "steps": history["steps"],
    })
    csv_path = os.path.join(agent_dir, "training_log.csv")
    df.to_csv(csv_path, index=False)

    # Save config
    config_path = os.path.join(agent_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(history["config"], f, indent=2)

    # Save agent checkpoint
    import torch
    checkpoint_path = os.path.join(agent_dir, "checkpoint.pt")
    state_dict = history["agent"].get_state_dict()
    torch.save(state_dict, checkpoint_path)

    print(f"  > Results saved to {agent_dir}/")
    return agent_dir


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for Ad Stealth Env")
    parser.add_argument("--agent", type=str, default="q_learning",
                        choices=list(AGENT_REGISTRY.keys()) + ["all"],
                        help="Agent to train (or 'all')")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    parser.add_argument("--pattern", type=str, default="linear",
                        choices=["linear", "random_walk", "erratic"],
                        help="Viewport movement pattern")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    print("=" * 60)
    print("  RL Stealthy Ad Placement — Training")
    print("=" * 60)

    agents_to_train = list(AGENT_REGISTRY.keys()) if args.agent == "all" else [args.agent]

    for agent_name in agents_to_train:
        print(f"\n{'-' * 40}")
        print(f"  Agent: {agent_name}")
        print(f"{'-' * 40}")

        history = train_agent(
            agent_name,
            episodes=args.episodes,
            viewport_pattern=args.pattern,
            seed=args.seed,
        )
        save_results(history, results_dir=args.results_dir)
        print(f"  Best avg reward: {history['best_avg_reward']:.2f}")

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
