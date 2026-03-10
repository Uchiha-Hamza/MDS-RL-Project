"""
Centralized hyperparameter configuration for all RL algorithms.
"""

# ─── Environment ────────────────────────────────────────────────────
ENV_CONFIG = {
    "grid_size": 10,
    "viewport_size": 3,
    "sector_size": 5,
    "warning_phase_length": 4,   # steps before scan
    "scan_phase_length": 1,      # scan lasts 1 step
    "viewport_move_interval": 5, # viewport shifts every N steps
    "max_steps": 300,            # max episode length
    "viewport_pattern": "linear", # "linear", "random_walk", "erratic"
}

# ─── Reward Shaping ─────────────────────────────────────────────────
REWARD_CONFIG = {
    "in_viewport": 1.0,         # inside the 3x3 viewport
    "near_viewport": 0.3,       # adjacent to viewport (1 cell away)
    "step_penalty": -0.1,       # small cost per step
    "caught_penalty": -10.0,    # caught by ad-blocker
    "survival_bonus": 5.0,      # bonus every survival_interval steps
    "survival_interval": 50,    # how often to give survival bonus
}

# ─── Q-Learning ──────────────────────────────────────────────────────
Q_LEARNING_CONFIG = {
    "lr": 0.1,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.9995,
    "episodes": 5000,
}

# ─── SARSA ───────────────────────────────────────────────────────────
SARSA_CONFIG = {
    "lr": 0.1,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.9995,
    "episodes": 5000,
}

# ─── DQN ─────────────────────────────────────────────────────────────
DQN_CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.999,
    "batch_size": 64,
    "buffer_capacity": 50000,
    "target_update_freq": 100,  # steps between target net updates
    "hidden_dim": 128,
    "episodes": 3000,
}

# ─── Double DQN ──────────────────────────────────────────────────────
DOUBLE_DQN_CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.999,
    "batch_size": 64,
    "buffer_capacity": 50000,
    "target_update_freq": 100,
    "hidden_dim": 128,
    "episodes": 3000,
}

# ─── REINFORCE ───────────────────────────────────────────────────────
REINFORCE_CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "hidden_dim": 128,
    "episodes": 5000,
}

# ─── A2C ─────────────────────────────────────────────────────────────
A2C_CONFIG = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "hidden_dim": 128,
    "n_steps": 20,          # steps before update (n-step returns)
    "episodes": 3000,
}

# ─── Training ────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "seed": 42,
    "log_interval": 100,      # episodes between console logs
    "smoothing_window": 50,   # window for smoothing reward curves
    "save_checkpoints": True,
    "checkpoint_interval": 500,
}

# ─── Mapping agent name → config ────────────────────────────────────
AGENT_CONFIGS = {
    "q_learning": Q_LEARNING_CONFIG,
    "sarsa": SARSA_CONFIG,
    "dqn": DQN_CONFIG,
    "double_dqn": DOUBLE_DQN_CONFIG,
    "reinforce": REINFORCE_CONFIG,
    "a2c": A2C_CONFIG,
}
