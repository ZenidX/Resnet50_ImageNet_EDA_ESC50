"""
Core - Implementaciones de agentes y utilidades de RL
"""

from .agentes import (
    ReplayBuffer,
    QLearningAgent,
    SARSAAgent,
    entrenar_qlearning,
    entrenar_dqn,
)

from .utils import (
    plot_training_curves,
    plot_q_values,
    evaluar_agente,
)

# DQN solo si PyTorch está disponible
try:
    from .agentes import DQNAgent, DQNetwork
except ImportError:
    pass

__all__ = [
    'ReplayBuffer',
    'QLearningAgent',
    'SARSAAgent',
    'DQNAgent',
    'DQNetwork',
    'entrenar_qlearning',
    'entrenar_dqn',
    'plot_training_curves',
    'plot_q_values',
    'evaluar_agente',
]
