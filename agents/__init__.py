from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.reinforce import REINFORCEAgent
from agents.a2c import A2CAgent

AGENT_REGISTRY = {
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "reinforce": REINFORCEAgent,
    "a2c": A2CAgent,
}

__all__ = [
    "QLearningAgent", "SARSAAgent", "DQNAgent",
    "DoubleDQNAgent", "REINFORCEAgent", "A2CAgent",
    "AGENT_REGISTRY",
]
