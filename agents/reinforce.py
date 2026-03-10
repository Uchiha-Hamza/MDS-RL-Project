"""
REINFORCE (Monte-Carlo Policy Gradient) Agent with baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from utils import normalize_state


class PolicyNetwork(nn.Module):
    """Policy network: state → action probabilities."""

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
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class REINFORCEAgent:
    """REINFORCE with mean-return baseline."""

    name = "REINFORCE"

    def __init__(self, config):
        self.gamma = config.get("gamma", 0.99)
        hidden_dim = config.get("hidden_dim", 128)
        lr = config.get("lr", 1e-3)

        self.n_actions = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode storage
        self.log_probs = []
        self.rewards = []
        self.baseline = 0  # running average return

    def select_action(self, state, training=True):
        state_norm = normalize_state(state.copy())
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        probs = self.policy(state_t)

        if training:
            dist = Categorical(probs)
            action = dist.sample()
            self.log_probs.append(dist.log_prob(action))
            return action.item()
        else:
            return int(probs.argmax(dim=1).item())

    def _compute_returns(self):
        """Compute discounted returns G_t for each timestep."""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def _update_policy(self):
        """Update policy using REINFORCE with baseline."""
        returns = self._compute_returns()
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Update baseline (running average)
        episode_return = returns_t[0].item()
        self.baseline = 0.99 * self.baseline + 0.01 * episode_return

        # Normalize returns with baseline
        advantages = returns_t - self.baseline
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        loss = 0
        for log_prob, advantage in zip(self.log_probs, advantages):
            loss -= log_prob * advantage

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()

        # Clear episode data
        self.log_probs.clear()
        self.rewards.clear()

    def train_episode(self, env):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        self.log_probs.clear()
        self.rewards.clear()

        while True:
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.rewards.append(reward)
            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        self._update_policy()
        return total_reward, steps

    def get_state_dict(self):
        return {
            "policy": self.policy.state_dict(),
            "baseline": self.baseline,
        }

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict["policy"])
        self.baseline = state_dict["baseline"]
