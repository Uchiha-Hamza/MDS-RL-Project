"""
Advantage Actor-Critic (A2C) Agent with GAE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from utils import normalize_state


class ActorCritic(nn.Module):
    """Shared backbone with separate actor (policy) and critic (value) heads."""

    def __init__(self, state_dim=6, n_actions=5, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        policy = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value


class A2CAgent:
    """Advantage Actor-Critic with Generalized Advantage Estimation (GAE)."""

    name = "A2C"

    def __init__(self, config):
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.n_steps = config.get("n_steps", 20)
        hidden_dim = config.get("hidden_dim", 128)
        lr = config.get("lr", 3e-4)

        self.n_actions = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state, training=True):
        state_norm = normalize_state(state.copy())
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        policy, _ = self.model(state_t)

        if training:
            dist = Categorical(policy)
            action = dist.sample()
            return action.item()
        else:
            return int(policy.argmax(dim=1).item())

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE advantages and returns."""
        advantages = []
        gae = 0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def _update(self, states, actions, advantages, returns):
        """Perform A2C update."""
        states_norm = np.array([normalize_state(s) for s in states])
        states_t = torch.FloatTensor(states_norm).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Forward pass
        policy, values = self.model(states_t)
        values = values.squeeze(1)
        dist = Categorical(policy)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # Losses
        actor_loss = -(log_probs * advantages_t).mean()
        critic_loss = nn.MSELoss()(values, returns_t)
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

    def train_episode(self, env):
        """Train for one full episode using n-step rollouts."""
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        # Collect trajectory buffers
        states_buf, actions_buf, rewards_buf, dones_buf, values_buf = [], [], [], [], []

        while True:
            # Collect n steps
            state_norm = normalize_state(state.copy())
            state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.model(state_t)

            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states_buf.append(state.copy())
            actions_buf.append(action)
            rewards_buf.append(reward)
            dones_buf.append(float(done))
            values_buf.append(value.item())

            total_reward += reward
            steps += 1
            state = next_state

            # Update every n_steps or at episode end
            if len(states_buf) >= self.n_steps or done:
                # Bootstrap value for incomplete trajectory
                if done:
                    next_value = 0.0
                else:
                    ns_norm = normalize_state(next_state.copy())
                    ns_t = torch.FloatTensor(ns_norm).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        _, nv = self.model(ns_t)
                    next_value = nv.item()

                advantages, returns = self._compute_gae(
                    rewards_buf, values_buf, dones_buf, next_value
                )
                self._update(states_buf, actions_buf, advantages, returns)

                states_buf.clear()
                actions_buf.clear()
                rewards_buf.clear()
                dones_buf.clear()
                values_buf.clear()

            if done:
                break

        return total_reward, steps

    def get_state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
