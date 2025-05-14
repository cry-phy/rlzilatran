"""Module containing the agents.

- GUI agent
- Minimax agent
- CNN agent
"""

from .base import BaseAgent, Tool, agent_class_dict
from .human import HumanAgent
from .minimax import MinimaxAgent

__all__ = [
    "BaseAgent",
    "HumanAgent",
    "MinimaxAgent",
    "Tool",
    "agent_class_dict",
]
