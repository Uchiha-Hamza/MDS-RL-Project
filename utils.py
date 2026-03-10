"""
Shared utilities: Replay Buffer, state encoding, seed management.
"""

import random
import numpy as np
import torch
from collections import deque


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """Experience replay buffer for DQN-based agents."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def state_to_index(state, grid_size=10, viewport_range=8, n_sectors=4, timer_range=5):
    """
    Convert continuous state tuple to a flat index for tabular methods.
    State: (xa, ya, xt, yt, sigma, tau)
    """
    xa, ya, xt, yt, sigma, tau = [int(s) for s in state]
    # xt, yt are offset by 1 (valid range is 1..8)
    xt_idx = xt - 1
    yt_idx = yt - 1
    index = xa
    index = index * grid_size + ya
    index = index * viewport_range + xt_idx
    index = index * viewport_range + yt_idx
    index = index * n_sectors + sigma
    index = index * timer_range + tau
    return index


def get_total_states(grid_size=10, viewport_range=8, n_sectors=4, timer_range=5):
    """Return the total number of discrete states."""
    return grid_size * grid_size * viewport_range * viewport_range * n_sectors * timer_range


def normalize_state(state, grid_size=10, n_sectors=4, timer_max=4):
    """Normalize state to [0, 1] range for neural network input."""
    state = np.array(state, dtype=np.float32)
    # xa, ya in [0, grid_size-1]
    state[0] /= (grid_size - 1)
    state[1] /= (grid_size - 1)
    # xt, yt in [1, grid_size-2]
    state[2] = (state[2] - 1) / (grid_size - 3)
    state[3] = (state[3] - 1) / (grid_size - 3)
    # sigma in [0, n_sectors-1]
    state[4] /= (n_sectors - 1)
    # tau in [0, timer_max]
    state[5] /= timer_max
    return state


def smooth(values, window=50):
    """Compute smoothed values using a moving average."""
    if len(values) < window:
        window = max(1, len(values))
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start:i + 1]))
    return smoothed
