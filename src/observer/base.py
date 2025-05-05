from agents.base import BaseAgent
from game.game import Game
from game.utils import Board
from mpi.handler import MPIHandler
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
        self._state_data: list[torch.Tensor] = []
        self._value_data: list[float] = []
        self._move_data: list[torch.Tensor] = []
        self._num_data: int = 0
        self.mpi = MPIHandler()

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

    def current_agent(self) -> BaseAgent:
        """Return the currently at play agent."""
        return self._agents[self._game.state.player]

    def move(self) -> bool:
        """Make a single move of the current Agent."""
        raise NotImplementedError

    def generate_episode(self) -> None:
        """Generate an entire episode."""
        while not self._game_over:
            self.game_over = self.move()

    def add_data(
        self,
        state_tensor: torch.Tensor,
        value: float,
        move_tensor: torch.Tensor,
    ) -> None:
        """Add move data to the history."""
        self._state_data.append(state_tensor)
        self._value_data.append(value)
        self._move_data.append(move_tensor)
        self._num_data += 1

    def batch_ready(self) -> bool:
        """Check if batch is full."""
        return self._num_data >= Board.NUM_LEAVES

    def reset_data(self) -> None:
        """Remove Data and reset counter."""
        self._state_data = []
        self._value_data = []
        self._move_data = []
        self._num_data = 0

    def send_epsiode(self) -> None:
        """Send batch of move data to Training."""
        states = torch.stack(self._state_data)
        values = torch.Tensor(self._value_data)
        moves = torch.stack(self._move_data)
        self.mpi.send_episode(states, values, moves)

    def run(self) -> None:
        """Run indefinitely."""
        raise NotImplementedError
