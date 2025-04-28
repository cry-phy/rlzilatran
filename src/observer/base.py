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
        self._game_over = False

    @classmethod
    def self_play(
        cls, agent_cls: type[BaseAgent], workers: tuple[int, ...]
    ) -> "BaseObserver":
        """Generate a game between two of the same agents."""
        game = Game.new_game()
        agent1 = agent_cls(game, workers)
        agent2 = agent_cls(game, workers)
        agent1.player = True
        agent1._clear_tool()
        agent2.player = agent1.player
        agent2._clear_tool()
        return cls(workers, game, (agent1, agent2))

    @classmethod
    def any_vs_any(
        cls,
        agent_cls1: type[BaseAgent],
        agent_cls2: type[BaseAgent],
        workers: tuple[int, ...],
    ) -> "BaseObserver":
        """Pit two agents against each other."""
        game = Game.new_game()
        agent1 = agent_cls1(game, workers)
        agent2 = agent_cls2(game, workers)
        agent1.player = True
        agent2.player = agent1.player
        return cls(workers, game, (agent1, agent2))

    def move(self) -> bool:
        """Make a single move of the current Agent."""
        raise NotImplementedError

    def generate_episode(self) -> None:
        """Generate an entire episode."""
        while not self._game_over:
            self.game_over = self.move()

    def get_data(self) -> torch.Tensor:
        """Get a segment of data."""
        raise NotImplementedError

    def run(self) -> None:
        """Run indefinitely."""
        raise NotImplementedError
