from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Callable
    from game import State

from game.utils import Constants
import torch
import logging

logger: logging.Logger = logging.getLogger(__name__)


class __Rules:
    """Holds functions related to game Rules (available moves, termination states, ...).

    This is a module level singleton, i. e. being loaded from config at import time.
    """

    def __init__(self) -> None:
        """Empty initializer use Factory methods instead."""
        # from config
        self._piece_move_dict = {
            "P": self.generate_p_moves,
            "S": self.generate_s_moves,
            "O": self.generate_o_moves,
        }
        self._move_list = Constants.convert_dict_list(self._piece_move_dict)

    def valid_position(self, x: int, y: int) -> bool:
        return (x >= 0) and (x < Constants.x) and (y >= 0) and (y < Constants.y)

    def valid_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < Constants.x)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < Constants.y)
        )

    def moves_from_piece(self, piece: str) -> Callable[[int, int, State], torch.Tensor]:
        """Get moves from piece."""
        return self._piece_move_dict[piece]

    def generate_p_moves(self, x: int, y: int, state: State) -> torch.Tensor:
        """Pregenerate moves for Private."""
        # benchmarked against ray casting.
        move_targets = Constants.PRIVATE_MOVES + torch.tensor([x, y], dtype=torch.int)
        in_out_mask = self.valid_positions(move_targets)
        real_move_targets = move_targets[in_out_mask]
        own_mask = state.get_player_mask(state.player)
        keep = ~own_mask[real_move_targets[:, 0], real_move_targets[:, 1]]
        return real_move_targets[keep]

    def generate_s_moves(self, x: int, y: int, state: State) -> torch.Tensor:
        """Pregenerate moves for Sergeant."""
        # benchmarked against ray casting.
        moves = []
        origin = torch.tensor(
            [
                [x, y],
            ],
            dtype=torch.int,
        )
        own_mask = state.get_player_mask(state.player)
        other_mask = state.get_player_mask(not state.player)
        for direction in Constants.SERGEANT_DIRECTIONS:
            for i in range(1, 3):
                pos = origin + i * direction
                if not self.valid_positions(pos):
                    break
                if own_mask[pos[0, 0], pos[0, 1]]:
                    break
                moves.append(pos)
                if other_mask[pos[0, 0], pos[0, 1]]:
                    break
        if not moves:
            return torch.empty((0, 2), dtype=torch.int32)
        return torch.concatenate(moves, dim=0)

    def generate_o_moves(self, x: int, y: int, state: State) -> torch.Tensor:
        """Pregenerate moves for Officer."""
        # benchmarked against ray casting.
        moves = []
        origin = torch.tensor(
            [
                [x, y],
            ],
            dtype=torch.int,
        )
        own_mask = state.get_player_mask(state.player)
        other_mask = state.get_player_mask(not state.player)
        max_steps = max(Constants.BOARD_SIZE) - 1
        for direction in Constants.OFFICER_DIRECTIONS:
            for i in range(1, max_steps + 1):
                pos = origin + i * direction
                if not self.valid_positions(pos):
                    break
                if own_mask[pos[0, 0], pos[0, 1]]:
                    break
                moves.append(pos)
                if other_mask[pos[0, 0], pos[0, 1]]:
                    break
        if not moves:
            return torch.empty((0, 2), dtype=torch.int32)
        return torch.concatenate(moves, dim=0)

    def check_half_empty(self, player, tensor: torch.Tensor) -> bool:
        """Termination state this half of the board is empty."""
        i1 = int(player) * Constants.num_pieces
        i2 = (1 + int(player)) * Constants.num_pieces
        return not torch.any(tensor[:, :, i1:i2].to(dtype=torch.bool))

    def check_captured_value_termination(self, captured_values: list[float]) -> bool:
        """Check whether game has ended due to captures."""
        return (
            captured_values[0] >= Constants.max_capture_value / 2
            or captured_values[1] >= Constants.max_capture_value / 2
        )

    def check_capture_call_termination(self, since_capture_call: int) -> bool:
        """Check game status by capture call."""
        return since_capture_call >= Constants.capture_clock

    # Rules
    def check_capture_termination(self, since_capture: int) -> bool:
        """Check if the game has ended due to no captures.

        Needed to get non infinite Episodes.
        """
        return since_capture >= 100

    # Rules


Rules = __Rules()
