from __future__ import annotations
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Callable
    from game.game import Game

from agents.base import BaseAgent
from game.utils import Board
from time import time
import torch
import logging

logger: logging.Logger = logging.getLogger(__name__)


class MinimaxAgent(BaseAgent):
    """Simple Minimax agent as benchmark. Runs only single threaded."""

    def __init__(self, game: Game) -> None:
        """Initialize Minimax agent."""
        super().__init__(game)
        self._depth: int = 5
        self._weights: dict[str, float] = {
            "material": 0,
            "capture": 1,
            "mobility": 0,
            "win_lose": 10,
        }
        self._metrics: dict[str, Callable[[Game], float | torch.Tensor]] = {
            "material": self.material,
            "capture": self.capture,
            "mobility": self.mobility,
            "win_lose": self.win_lose,
        }
        self.counter = 0

    @classmethod
    def new(cls, game: Game, depth: int) -> MinimaxAgent:
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
        return (
            game.state.captured_value[not game.state.player]
            - game.state.captured_value[game.state.player]
        )

    def mobility(self, game: Game) -> torch.Tensor:
        """Get mobility score of current player."""
        return game.move_mask().to(torch.int32).sum()

    def win_lose(self, game: Game) -> float:
        """Game score if game state is terminal."""
        if game.game_over():
            if game.winner() == game.state.player:
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

    def minimax(
        self, depth: int, alpha: float, beta: float, game: Game
    ) -> torch.Tensor:
        """Depth-limited minimax with alpha beta pruning.

        Returns a single-element tensor with the best value for the current player.
        """
        self.counter += 1

        if depth >= self._depth or game.game_over():
            return self.single_state_valuation(game)

        if game.state.player == self._player:
            value = -1e9
            for x, y, index in zip(*game.all_moves(), strict=True):
                new_game = game.copy()
                new_game.make_move(x, y, *Board.move_from_index(index))
                child_val = float(self.minimax(depth + 1, alpha, beta, new_game))
                value = max(value, child_val)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return torch.tensor([value], dtype=torch.float32)
        else:
            value = 1e9
            for x, y, index in zip(*game.all_moves(), strict=True):
                new_game = game.copy()
                new_game.make_move(x, y, *Board.move_from_index(index))
                child_val = float(self.minimax(depth + 1, alpha, beta, new_game))
                value = min(value, child_val)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return torch.tensor([value], dtype=torch.float32)

    # -- MPI compatability --
    def generate_move(
        self, move_tensor: torch.Tensor
    ) -> tuple[
        int | torch.Tensor,
        int | torch.Tensor,
        int | torch.Tensor,
        int | torch.Tensor,
        int | torch.Tensor,
    ]:
        """Generate a move for the current game."""
        shifted_flat = move_tensor.flatten() - move_tensor.min()
        move_propabilities = shifted_flat / shifted_flat.sum()
        move_flat_index = torch.multinomial(move_propabilities, 1)
        move_indices = torch.unravel_index(move_flat_index, Board.move_tensor_shape)
        logger.debug("Suggesting move: %s", move_indices)
        x1: torch.Tensor = move_indices[0]
        y1: torch.Tensor = move_indices[1]
        x2, y2, flag = Board.move_from_index(move_indices[2])
        return x1, y1, x2, y2, flag

    def generate_move_tensor(self) -> None:
        """Generate move tensor for the subset of tasks for this rank."""
        move_list = self._game.all_moves()
        # allocate moves to specific ranks
        all_moves = torch.stack(move_list).T
        num_tasks = len(all_moves)
        min_tasks = num_tasks // cast(int, self.mpi._osize)
        remainder = num_tasks % cast(int, self.mpi._osize)
        tasks_per_rank = torch.full((cast(int, self.mpi._osize),), min_tasks)
        tasks_per_rank[:remainder] += 1
        end_indices_per_rank = torch.cumsum(tasks_per_rank, dim=0)
        start_indices_per_rank = end_indices_per_rank - tasks_per_rank

        # Generate which moves are evaluated on this rank
        start_index = start_indices_per_rank[self.mpi._orank]
        end_index = end_indices_per_rank[self.mpi._orank]
        task_moves = all_moves[start_index:end_index]

        logger.debug("Running moves %i:%i.", start_index, end_index)

        # Evaluate
        move_valuations = torch.zeros(Board.move_tensor_shape)
        start_time = time()
        self.counter = 1
        for x, y, index in task_moves:
            game: Game = self._game.copy()
            game.make_move(x, y, *Board.move_from_index(index))
            value = self.minimax(1, -1e9, 1e9, game)
            move_valuations[x, y, index] = value + 1e-9
        runtime = time() - start_time
        logger.info(
            "Ran through %i Nodes in %fs, average: %fms",
            self.counter,
            runtime,
            runtime / self.counter * 1e6,
        )
        self.counter = 0
        self._value = float(torch.max(move_valuations))
        self._move_tensor = move_valuations + torch.min(move_valuations)
