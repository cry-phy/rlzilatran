from agents.base import BaseAgent
from game.game import Game
import torch


class BaseObserver:
    """Base Observer as parent for Observers.

    Not intended for direct use.
    """

    def __init__(
        self, workers: tuple[int, ...], game: Game, agents: tuple[BaseAgent, BaseAgent]
    ) -> None:
        """Initialize Observer with ownership over threads."""
        self._workers: tuple[int, ...] = workers
        self._root = None
        self._game = game
        self._agents = agents

    @classmethod
    def self_play(
        cls, agent_cls: type[BaseAgent], workers: tuple[int, ...]
    ) -> "BaseObserver":
        """Generate a game between two of the same agents."""
        game = Game.new_game()
        return cls(workers, game, (agent_cls(workers, game), agent_cls(workers, game)))

    @classmethod
    def against(
        cls,
        agent_cls1: type[BaseAgent],
        agent_cls2: type[BaseAgent],
        workers: tuple[int, ...],
    ) -> "BaseObserver":
        """Pit two agents against each other."""
        game = Game.new_game()
        return cls(
            workers, game, (agent_cls1(workers, game), agent_cls2(workers, game))
        )

    def move(self) -> None:
        """Make a single move of the current Agent."""
        raise NotImplementedError

    def generate_episode(self) -> None:
        """Generate an entire episode."""
        raise NotImplementedError

    def get_data(self) -> torch.Tensor:
        """Get a segment of data."""
        raise NotImplementedError
