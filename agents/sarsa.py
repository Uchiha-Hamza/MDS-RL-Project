"""
Tabular SARSA Agent — On-policy TD control.
"""

import numpy as np
from utils import state_to_index, get_total_states


class SARSAAgent:
    """On-policy tabular SARSA."""

    name = "SARSA"

    def __init__(self, config):
        self.lr = config.get("lr", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.9995)
        self.n_actions = 5

        self.n_states = get_total_states()
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def select_action(self, state, training=True):
        """ε-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        idx = state_to_index(state)
        return int(np.argmax(self.q_table[idx]))

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]"""
        idx = state_to_index(state)
        next_idx = state_to_index(next_state)

        target = reward
        if not done:
            target += self.gamma * self.q_table[next_idx, next_action]

        self.q_table[idx, action] += self.lr * (target - self.q_table[idx, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self, env):
        """Train for one episode. Returns (total_reward, steps)."""
        state, _ = env.reset()
        action = self.select_action(state, training=True)
        total_reward = 0
        steps = 0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = self.select_action(next_state, training=True)

            self.update(state, action, reward, next_state, next_action, done)

            total_reward += reward
            steps += 1
            state = next_state
            action = next_action

            if done:
                break

        self.decay_epsilon()
        return total_reward, steps

    def get_state_dict(self):
        return {"q_table": self.q_table.copy(), "epsilon": self.epsilon}

    def load_state_dict(self, state_dict):
        self.q_table = state_dict["q_table"]
        self.epsilon = state_dict["epsilon"]
