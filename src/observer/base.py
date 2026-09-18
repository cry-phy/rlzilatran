from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agents import BaseAgent
from game import Game
from game.utils import Constants, Translator
from mpi import MPIHandler
import torch


class BaseObserver:
    """Base Observer as parent for Observers.

    Not intended for direct use.
    """

    def __init__(self, game: Game, agents: tuple[BaseAgent, BaseAgent]) -> None:
        """Initialize Observer with ownership over threads."""
        self.game = game
        self.agents = agents
        self._game_over = False
        self._state_data: list[torch.Tensor] = []
        self._value_data: list[float] = []
        self._move_data: list[torch.Tensor] = []
        self._num_data: int = 0
        self.mpi = MPIHandler()

    @classmethod
    def any_vs_any(
        cls,
        game: Game,
        agent_cls1: type[BaseAgent],
        agent_cls2: type[BaseAgent],
    ) -> BaseObserver:
        """Pit two agents against each other."""
        agent1 = agent_cls1(game)
        agent2 = agent_cls2(game)
        agent1.player = True
        agent2.player = not agent1.player
        return cls(game, (agent1, agent2))

    def current_agent(self) -> BaseAgent:
        """Return the currently at play agent."""
        return self.agents[self.game.state.player]

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
        return self._num_data >= Constants.BATCH_SIZE

    def reset_data(self) -> None:
        """Remove Data and reset counter."""
        self._state_data = []
        self._value_data = []
        self._move_data = []
        self._num_data = 0

    def reset(self) -> None:
        """Reset all componenents."""
        self.game.reset()
        agent_cls1 = type(self.agents[0])
        agent_cls2 = type(self.agents[1])
        agent1 = agent_cls1(self.game)
        agent2 = agent_cls2(self.game)
        agent1.player = True
        agent2.player = not agent1.player
        self.agents = (agent1, agent2)

    def make_random_moves(self, num: int) -> None:
        """Make random moves."""
        for _ in range(num):
            moves = list(zip(*self.game.all_moves(), strict=True))
            i = torch.randint(0, len(moves), (1,))
            x1, y1, index = moves[i]
            x2, y2, flag = Translator.move_from_index(index)
            self.game.make_move(x1, y1, x2, y2, flag)

    def send_episode(self) -> None:
        """Send batch of move data to Training."""
        states = torch.stack(self._state_data, dim=0)
        values = torch.Tensor(self._value_data)
        moves = torch.stack(self._move_data, dim=0)
        self.mpi.send_episode(states, values, moves)
