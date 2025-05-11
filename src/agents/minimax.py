from typing import Callable
from game.game import Game
from agents.base import BaseAgent
from game.utils import Board, MoveFlag
import torch


class MinimaxAgent(BaseAgent):
    """Simple Minimax agent as benchmark. Runs only single threaded."""

    def __init__(self, game: Game) -> None:
        """Initialize Minimax agent."""
        super().__init__(game)
        self._depth: int = 3
        self._weights: dict[str, float] = {
            "material": 1,
            "capture": 1,
            "mobility": 1,
            "win_lose": 10,
        }
        self._metrics: dict[str, Callable[[Game], float]] = {
            "material": self.mobility,
            "capture": self.capture,
            "mobility": self.mobility,
            "win_lose": self.win_lose,
        }

    @classmethod
    def new(cls, game: Game, depth: int) -> "MinimaxAgent":
        """Generate Minimax agent with depth only."""
        instance = cls(game)
        instance.depth = depth
        return instance

    @property
    def depth(self) -> int:
        """Depth property."""
        return self.depth

    @depth.setter
    def depth(self, depth: int) -> None:
        """Depth property setter."""
        self._depth = depth

    def material(self, game: Game) -> float:
        """Get material of current player."""
        return game.state.player_material(game.state.player)

    def capture(self, game: Game) -> float:
        """Get Capture of current player."""
        return game.state.captured_value[self._player]

    def mobility(self, game: Game) -> torch.Tensor:
        """Get mobility score of current player."""
        return game.move_mask().to(torch.int32).sum()

    def win_lose(self, game: Game) -> float:
        """Game score if game state is terminal."""
        if game.game_over():
            if game.winner() == self._player:
                return 1
            else:
                return -1
        else:
            return 0

    def single_state_valuation(self, game: Game) -> torch.Tensor:
        """Calculate total valuation with functions and weights from dicts."""
        valuations = torch.empty(len(self._weights))
        for index, key in enumerate(self._weights.keys()):
            valuations[index] = self._weights[key] * self._metrics[key](game)
        return valuations.sum()

    def minimax(self, depth: int, game: Game) -> torch.Tensor:
        """Actual minimax algorithm in recursion format."""
        if depth == self._depth or self._game.game_over():
            return self.single_state_valuation(game)

        state_value: torch.Tensor = torch.tensor([], dtype=torch.float32)
        if self._player == self._game.state.player:
            best: float = -float("inf")
            for x, y, index in game.all_moves():
                new_game: Game = game.copy()
                new_game.make_move(
                    x, y, *Board.move_from_index(index), MoveFlag.STANDARD
                )
                val = self.minimax(depth - 1, new_game)
                best = max(best, val)
        else:
            best: float = float("inf")
            for x, y, index in game.all_moves():
                new_game: Game = game.copy()
                new_game.make_move(x, y, *Board.move_from_index(index), 0)
                val = self.minimax(depth - 1, new_game)
                best = min(best, val)
        return state_value

    # -- MPI compatability --
    def generate_move(
        self, move_tensor: torch.Tensor
    ) -> tuple[
        int | torch.Tensor, int | torch.Tensor, int | torch.Tensor, int | torch.Tensor
    ]:
        """Generate a move for the current game."""
        move_indices = torch.argmax(move_tensor)
        x1: torch.Tensor = move_indices[0]
        y1: torch.Tensor = move_indices[1]
        x2, y2 = Board.move_from_index(move_indices[2])
        return x1, y1, x2, y2

    def generate_move_tensor(self) -> None:
        """Generate move tensor."""
        move_valuations = torch.zeros(Board.move_tensor_shape)
        for x, y, index in self._game.all_moves():
            game: Game = self._game.copy()
            move_valuations[x, y, index] = self.minimax(1, game)
        self._value = float(torch.max(move_valuations))
        self._move_tensor = move_valuations
