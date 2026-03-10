"""
Comprehensive evaluation and visualization script.

Generates all plots for the report and poster:
  1. Training curves (smoothed reward vs episodes)
  2. Algorithm comparison (bar chart + table)
  3. Risk sensitivity analysis (behavior as τ → 0)
  4. Agent position heatmaps
  5. Trajectory visualization
  6. Generalization test across viewport patterns
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from env.ad_stealth_env import AdStealthEnv
from agents import AGENT_REGISTRY
from config import AGENT_CONFIGS, ENV_CONFIG, REWARD_CONFIG, TRAIN_CONFIG
from utils import set_seed, smooth, normalize_state
import torch


# ─── Style Setup ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.2)
COLORS = {
    "q_learning":  "#FF6B6B",
    "sarsa":       "#4ECDC4",
    "dqn":         "#45B7D1",
    "double_dqn":  "#96CEB4",
    "reinforce":   "#FFEAA7",
    "a2c":         "#DDA0DD",
}
DISPLAY_NAMES = {
    "q_learning":  "Q-Learning",
    "sarsa":       "SARSA",
    "dqn":         "DQN",
    "double_dqn":  "Double DQN",
    "reinforce":   "REINFORCE",
    "a2c":         "A2C",
}


def load_training_logs(results_dir="results"):
    """Load all training CSVs from results directory."""
    logs = {}
    for agent_name in AGENT_REGISTRY.keys():
        csv_path = os.path.join(results_dir, agent_name, "training_log.csv")
        if os.path.exists(csv_path):
            logs[agent_name] = pd.read_csv(csv_path)
    return logs


# ═══════════════════════════════════════════════════════════════════
# 1. Training Curves
# ═══════════════════════════════════════════════════════════════════
def plot_training_curves(logs, results_dir="results", window=50):
    """Plot smoothed reward curves for all agents on one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Individual curves
    ax = axes[0]
    for agent_name, df in logs.items():
        rewards = df["reward"].values
        smoothed = smooth(rewards, window)
        color = COLORS.get(agent_name, "#999999")
        ax.plot(smoothed, label=DISPLAY_NAMES[agent_name], color=color, linewidth=2)
        # Add shaded region
        raw_smooth = pd.Series(rewards).rolling(window, min_periods=1)
        ax.fill_between(range(len(smoothed)),
                        raw_smooth.mean() - raw_smooth.std(),
                        raw_smooth.mean() + raw_smooth.std(),
                        alpha=0.15, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Training Curves — All Algorithms")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Episode length
    ax = axes[1]
    for agent_name, df in logs.items():
        steps = df["steps"].values
        smoothed = smooth(steps, window)
        color = COLORS.get(agent_name, "#999999")
        ax.plot(smoothed, label=DISPLAY_NAMES[agent_name], color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Episode Length — All Algorithms")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 2. Algorithm Comparison
# ═══════════════════════════════════════════════════════════════════
def plot_algorithm_comparison(logs, results_dir="results", last_n=500):
    """Bar chart comparing final performance of all algorithms."""
    names = []
    means = []
    stds = []
    colors = []

    for agent_name, df in logs.items():
        rewards = df["reward"].values[-last_n:]
        names.append(DISPLAY_NAMES[agent_name])
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
        colors.append(COLORS.get(agent_name, "#999999"))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, means, yerr=stds, color=colors, edgecolor="white",
                  linewidth=1.5, capsize=5, alpha=0.85)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel("Mean Reward (last 500 episodes)")
    ax.set_title("Algorithm Comparison — Final Performance")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "algorithm_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {path}")

    # Also save a markdown table
    table_path = os.path.join(results_dir, "comparison_table.md")
    with open(table_path, "w") as f:
        f.write("| Algorithm | Mean Reward | Std Reward | Mean Steps |\n")
        f.write("|-----------|------------|------------|------------|\n")
        for agent_name, df in logs.items():
            rews = df["reward"].values[-last_n:]
            stp = df["steps"].values[-last_n:]
            f.write(f"| {DISPLAY_NAMES[agent_name]} | {np.mean(rews):.2f} | "
                    f"{np.std(rews):.2f} | {np.mean(stp):.1f} |\n")
    print(f"  [OK] Saved: {table_path}")


# ═══════════════════════════════════════════════════════════════════
# 3. Risk Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════
def plot_risk_sensitivity(results_dir="results", n_episodes=200):
    """
    Analyze agent behavior as scan timer τ approaches 0.
    Shows what action the agent takes depending on τ value.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    env_config = {**ENV_CONFIG, **REWARD_CONFIG, "viewport_pattern": "linear"}

    for idx, agent_name in enumerate(AGENT_REGISTRY.keys()):
        ax = axes[idx]
        checkpoint_path = os.path.join(results_dir, agent_name, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            ax.set_title(f"{DISPLAY_NAMES[agent_name]} (no checkpoint)")
            continue

        # Load agent
        agent_config = AGENT_CONFIGS[agent_name].copy()
        AgentClass = AGENT_REGISTRY[agent_name]
        agent = AgentClass(agent_config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(state_dict)

        # Collect actions by timer value
        env = AdStealthEnv(config=env_config)
        action_by_tau = {tau: [] for tau in range(5)}  # tau: 0,1,2,3,4

        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.select_action(state, training=False)
                tau = int(state[5])
                action_by_tau[tau].append(action)
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        env.close()

        # Plot action distribution per timer value
        action_names = ["Up", "Down", "Left", "Right", "Stay"]
        tau_vals = sorted(action_by_tau.keys(), reverse=True)
        data = np.zeros((5, 5))  # (n_actions, n_tau_values)

        for i, tau in enumerate(tau_vals):
            if len(action_by_tau[tau]) > 0:
                counts = np.bincount(action_by_tau[tau], minlength=5)
                data[:, i] = counts / counts.sum()

        sns.heatmap(data, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=[f"τ={t}" for t in tau_vals],
                    yticklabels=action_names, cbar=False, vmin=0, vmax=1)
        ax.set_title(f"{DISPLAY_NAMES[agent_name]}")
        ax.set_xlabel("Scan Timer (τ)")
        ax.set_ylabel("Action")

    plt.suptitle("Risk Sensitivity — Action Distribution vs. Scan Timer τ", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(results_dir, "risk_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 4. Agent Position Heatmaps
# ═══════════════════════════════════════════════════════════════════
def plot_position_heatmaps(results_dir="results", n_episodes=200):
    """Heatmap of where the agent spends time on the grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    env_config = {**ENV_CONFIG, **REWARD_CONFIG, "viewport_pattern": "linear"}

    for idx, agent_name in enumerate(AGENT_REGISTRY.keys()):
        ax = axes[idx]
        checkpoint_path = os.path.join(results_dir, agent_name, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            ax.set_title(f"{DISPLAY_NAMES[agent_name]} (no checkpoint)")
            continue

        agent_config = AGENT_CONFIGS[agent_name].copy()
        AgentClass = AGENT_REGISTRY[agent_name]
        agent = AgentClass(agent_config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(state_dict)

        env = AdStealthEnv(config=env_config)
        heatmap = np.zeros((10, 10))

        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.select_action(state, training=False)
                heatmap[int(state[1]), int(state[0])] += 1
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        env.close()

        # Normalize
        heatmap = heatmap / heatmap.sum() * 100

        # Draw quadrant lines
        sns.heatmap(heatmap, ax=ax, cmap="viridis", annot=True, fmt=".1f",
                    cbar_kws={"label": "% time"}, vmin=0)
        ax.axhline(5, color="red", linewidth=2, linestyle="--")
        ax.axvline(5, color="red", linewidth=2, linestyle="--")
        ax.set_title(f"{DISPLAY_NAMES[agent_name]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.suptitle("Agent Position Heatmaps (% of time in each cell)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(results_dir, "position_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# 5. Trajectory Visualization
# ═══════════════════════════════════════════════════════════════════
def plot_trajectories(results_dir="results", n_episodes=3):
    """Plot sample episode trajectories with viewport and blocker overlay."""
    for agent_name in AGENT_REGISTRY.keys():
        checkpoint_path = os.path.join(results_dir, agent_name, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            continue

        agent_config = AGENT_CONFIGS[agent_name].copy()
        AgentClass = AGENT_REGISTRY[agent_name]
        agent = AgentClass(agent_config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(state_dict)

        env_config = {**ENV_CONFIG, **REWARD_CONFIG, "viewport_pattern": "linear"}
        env = AdStealthEnv(config=env_config)

        fig, axes = plt.subplots(1, n_episodes, figsize=(6 * n_episodes, 6))
        if n_episodes == 1:
            axes = [axes]

        for ep_idx in range(n_episodes):
            ax = axes[ep_idx]
            state, _ = env.reset()
            positions = [(int(state[0]), int(state[1]))]
            vp_centers = [(int(state[2]), int(state[3]))]
            danger_sectors = [int(state[4])]
            done = False

            while not done:
                action = agent.select_action(state, training=False)
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                positions.append((int(state[0]), int(state[1])))
                vp_centers.append((int(state[2]), int(state[3])))
                danger_sectors.append(int(state[4]))

            # Draw grid
            ax.set_xlim(-0.5, 9.5)
            ax.set_ylim(9.5, -0.5)
            ax.set_aspect("equal")
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.grid(True, alpha=0.3)

            # Draw quadrant boundaries
            ax.axhline(4.5, color="red", linewidth=1.5, linestyle="--", alpha=0.5)
            ax.axvline(4.5, color="red", linewidth=1.5, linestyle="--", alpha=0.5)

            # Draw trajectory
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            ax.plot(xs, ys, 'b-', alpha=0.4, linewidth=1)
            ax.scatter(xs[0], ys[0], c="green", s=100, zorder=5, label="Start", marker="^")
            ax.scatter(xs[-1], ys[-1], c="red", s=100, zorder=5, label="End", marker="v")

            # Color trajectory points by time
            colors_t = plt.cm.plasma(np.linspace(0, 1, len(xs)))
            ax.scatter(xs, ys, c=colors_t, s=15, zorder=4, alpha=0.7)

            # Draw final viewport
            vp_x, vp_y = vp_centers[-1]
            vp_rect = patches.Rectangle((vp_x - 1.5, vp_y - 1.5), 3, 3,
                                        linewidth=2, edgecolor="green",
                                        facecolor="green", alpha=0.15)
            ax.add_patch(vp_rect)

            ax.set_title(f"Episode {ep_idx + 1} ({len(positions)} steps)")
            ax.legend(fontsize=8, loc="upper right")

        fig.suptitle(f"Trajectories — {DISPLAY_NAMES[agent_name]}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(results_dir, f"trajectory_{agent_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  [OK] Saved trajectory plots")


# ═══════════════════════════════════════════════════════════════════
# 6. Generalization Test
# ═══════════════════════════════════════════════════════════════════
def plot_generalization_test(results_dir="results", n_episodes=200):
    """
    Test trained agents (trained on linear) against all 3 viewport patterns.
    """
    patterns = ["linear", "random_walk", "erratic"]
    pattern_labels = ["Linear", "Random Walk", "Erratic"]

    data = {p: {} for p in patterns}

    for agent_name in AGENT_REGISTRY.keys():
        checkpoint_path = os.path.join(results_dir, agent_name, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            continue

        agent_config = AGENT_CONFIGS[agent_name].copy()
        AgentClass = AGENT_REGISTRY[agent_name]
        agent = AgentClass(agent_config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(state_dict)

        for pattern in patterns:
            env_config = {**ENV_CONFIG, **REWARD_CONFIG, "viewport_pattern": pattern}
            env = AdStealthEnv(config=env_config)

            rewards = []
            for _ in range(n_episodes):
                state, _ = env.reset()
                total_r = 0
                done = False
                while not done:
                    action = agent.select_action(state, training=False)
                    state, r, terminated, truncated, _ = env.step(action)
                    total_r += r
                    done = terminated or truncated
                rewards.append(total_r)
            env.close()

            data[pattern][agent_name] = {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
            }

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(patterns))
    n_agents = len(AGENT_REGISTRY)
    width = 0.12
    offset = -(n_agents - 1) / 2 * width

    for i, agent_name in enumerate(AGENT_REGISTRY.keys()):
        means = [data[p].get(agent_name, {}).get("mean", 0) for p in patterns]
        stds = [data[p].get(agent_name, {}).get("std", 0) for p in patterns]
        color = COLORS.get(agent_name, "#999999")
        ax.bar(x + offset + i * width, means, width, yerr=stds,
               label=DISPLAY_NAMES[agent_name], color=color, edgecolor="white",
               capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(pattern_labels)
    ax.set_xlabel("Viewport Pattern")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Generalization — Performance Across Viewport Patterns\n(Trained on Linear)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "generalization_test.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def generate_all_plots(results_dir="results"):
    """Generate all visualization plots."""
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60)

    logs = load_training_logs(results_dir)
    if not logs:
        print("  [ERR] No training logs found. Run train.py first.")
        return

    print(f"\n  Found logs for: {', '.join(DISPLAY_NAMES[k] for k in logs.keys())}\n")

    print("  [1/6] Training curves...")
    plot_training_curves(logs, results_dir)

    print("  [2/6] Algorithm comparison...")
    plot_algorithm_comparison(logs, results_dir)

    print("  [3/6] Risk sensitivity analysis...")
    plot_risk_sensitivity(results_dir)

    print("  [4/6] Position heatmaps...")
    plot_position_heatmaps(results_dir)

    print("  [5/6] Trajectory plots...")
    plot_trajectories(results_dir)

    print("  [6/6] Generalization test...")
    plot_generalization_test(results_dir)

    print(f"\n  All plots saved to: {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    generate_all_plots(args.results_dir)
