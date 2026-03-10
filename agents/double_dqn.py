"""
Double DQN Agent — Reduces overestimation bias by decoupling
action selection from action evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import ReplayBuffer, normalize_state


class QNetwork(nn.Module):
    """3-layer MLP Q-network."""

    def __init__(self, state_dim=6, n_actions=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DoubleDQNAgent:
    """Double DQN: uses online net for action selection, target net for evaluation."""

    name = "Double DQN"

    def __init__(self, config):
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.999)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 100)
        hidden_dim = config.get("hidden_dim", 128)
        lr = config.get("lr", 1e-3)
        buffer_cap = config.get("buffer_capacity", 50000)

        self.n_actions = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(hidden_dim=hidden_dim).to(self.device)
        self.target_net = QNetwork(hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)
        self.step_count = 0

    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        state_norm = normalize_state(state.copy())
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def _update_network(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_norm = np.array([normalize_state(s) for s in states])
        next_states_norm = np.array([normalize_state(s) for s in next_states])

        states_t = torch.FloatTensor(states_norm).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states_norm).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: select action with online net, evaluate with target net
        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1)
            next_q_values = self.target_net(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self, env):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.buffer.push(state.copy(), action, reward, next_state.copy(), float(done))
            self._update_network()

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        self.decay_epsilon()
        return total_reward, steps

    def get_state_dict(self):
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state_dict):
        self.q_net.load_state_dict(state_dict["q_net"])
        self.target_net.load_state_dict(state_dict["target_net"])
        self.epsilon = state_dict["epsilon"]
