# Reinforcement Learning for Stealthy Advertisement Placement

## Project Overview

This project implements an autonomous ad agent that navigates a 10×10 website grid to maximize user engagement while dynamically evading a scanning ad-blocker. It models a **Stochastic Stealth Pathfinding** problem in a non-stationary environment.

## Project Structure

```
Projet RL/
├── env/
│   ├── __init__.py
│   └── ad_stealth_env.py       # Custom Gymnasium environment
├── agents/
│   ├── __init__.py
│   ├── q_learning.py           # Tabular Q-Learning
│   ├── sarsa.py                # Tabular SARSA
│   ├── dqn.py                  # Deep Q-Network
│   ├── double_dqn.py           # Double DQN
│   ├── reinforce.py            # REINFORCE (Policy Gradient)
│   └── a2c.py                  # Advantage Actor-Critic
├── train.py                    # Unified training script
├── evaluate.py                 # Visualization & evaluation
├── config.py                   # Hyperparameters
├── utils.py                    # Shared utilities
├── run_all.py                  # One-click pipeline
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Everything (Train + Visualize)

```bash
python run_all.py
```

For a quick test run:
```bash
python run_all.py --episodes 500
```

### 3. Train Individual Agents

```bash
python train.py --agent q_learning --episodes 5000
python train.py --agent dqn --episodes 3000
python train.py --agent a2c --episodes 3000
python train.py --agent all   # Train all agents
```

### 4. Generate Plots Only (from existing results)

```bash
python run_all.py --skip-train
# or
python evaluate.py
```

## Environment Details

- **Grid**: 10×10 discrete grid
- **Agent**: 1×1 block (display ad)
- **User Viewport**: 3×3 window, moves every 5 steps
- **Ad-Blocker**: Cycles through 4 sectors (5×5 quadrants), 4-step warning + 1-step scan

### State Space

`(xa, ya, xt, yt, σ, τ)` — agent position, viewport center, blocker sector, scan timer

### Action Space

`{Up, Down, Left, Right, Stay}`

### Reward Structure

| Event | Reward |
|-------|--------|
| Inside viewport | +1.0 |
| Near viewport | +0.3 |
| Step penalty | -0.1 |
| Caught by scanner | -10.0 |
| Survival bonus (every 50 steps) | +5.0 |

## Algorithms Implemented

| Algorithm | Type | Key Features |
|-----------|------|-------------|
| **Q-Learning** | Tabular, Off-policy | ε-greedy, decaying ε |
| **SARSA** | Tabular, On-policy | ε-greedy, on-policy updates |
| **DQN** | Deep, Off-policy | Replay buffer, target network |
| **Double DQN** | Deep, Off-policy | Decoupled selection/evaluation |
| **REINFORCE** | Policy Gradient | Monte-Carlo returns, baseline |
| **A2C** | Actor-Critic | GAE, entropy bonus, n-step |

## Generated Visualizations

After running `run_all.py`, the `results/` folder contains:

1. **`training_curves.png`** — Smoothed reward & episode length curves
2. **`algorithm_comparison.png`** — Bar chart of final performance
3. **`risk_sensitivity.png`** — Action distributions vs. scan timer τ
4. **`position_heatmaps.png`** — Where agents spend time on the grid
5. **`trajectory_*.png`** — Sample episode trajectories per agent
6. **`generalization_test.png`** — Performance across viewport patterns
7. **`comparison_table.md`** — Markdown table of results

## Configuration

All hyperparameters are in `config.py`. Key parameters:

- `ENV_CONFIG`: Grid size, viewport movement interval, max steps
- `REWARD_CONFIG`: Reward values for each event
- `*_CONFIG`: Per-algorithm hyperparameters (lr, gamma, epsilon, etc.)
