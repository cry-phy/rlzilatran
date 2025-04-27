import torch
from copy import deepcopy
from typing import TYPE_CHECKING, Callable
import logging

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from game.state import State

__all__ = ["Board"]

PRIVATE_POSITIONS = torch.IntTensor(
    [
        (2, 1),
        (1, 2),
        (2, 2),
        (-3, -2),
        (-2, -3),
        (-3, -3),
    ]
).T
SERGEANT_POSITIONS = torch.IntTensor(
    [
        (0, 2),
        (1, 1),
        (2, 0),
        (-1, -3),
        (-2, -2),
        (-3, -1),
    ]
).T
OFFICER_POSITIONS = torch.IntTensor(
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (-1, -1),
        (-1, -2),
        (-2, -1),
    ]
).T


class __BoardMaster:
    """Generator for board tensors.

    This is a module level singleton, i. e. being loaded from config at import time.
    TODO: Extend to read config Files.
    """

    _PRIVATE_MOVES = torch.IntTensor(
        [
            (-1, -1),
            (-1, 1),
            (1, 1),
            (1, -1),
        ]
    )
    _SERGEANT_DIRECTIONS = torch.IntTensor(
        [
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
        ]
    )
    _OFFICER_DIRECTIONS = torch.IntTensor(
        [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]
    )

    def __init__(self) -> None:
        """Empty initializer use Factory methods instead."""
        self._pieces = ["P", "S", "O"]
        self.BOARD_SIZE = (4, 8)
        self.NUM_LAYERS = 2 * len(self._pieces)
        self.BOARD_DIMENSIONS = (self.NUM_LAYERS, *self.BOARD_SIZE)
        self.MAX_MOVES = self.x * self.y + 1
        self._piece_layer_dict = {
            "P": 0,
            "S": 1,
            "O": 2,
        }
        self._piece_value_dict = {
            "P": 1,
            "S": 2,
            "O": 3,
        }
        self._piece_move_dict = {
            "P": self.generate_p_moves,
            "S": self.generate_s_moves,
            "O": self.generate_o_moves,
        }
        self._initial_piece_dictionary = {
            "P": PRIVATE_POSITIONS,
            "S": SERGEANT_POSITIONS,
            "O": OFFICER_POSITIONS,
        }
        self._board: torch.IntTensor = torch.zeros(self.BOARD_DIMENSIONS).to(
            dtype=torch.int32
        )

    def index_from_move(self, x, y) -> int:
        return y * self.x + x

    def move_from_index(self, index: int) -> tuple[int, int]:
        return index % self.x, index // self.x

    def generate_p_moves(self, x: int, y: int, state: "State") -> torch.IntTensor:
        """Pregenerate moves for Private."""
        # benchmarked against ray casting.
        move_targets = self._PRIVATE_MOVES + torch.IntTensor([x, y])
        in_out_mask = self.valid_positions(move_targets)
        real_move_targets = move_targets[in_out_mask]
        own_mask = state.get_player_mask(state.player)
        keep = ~own_mask[real_move_targets[:, 0], real_move_targets[:, 1]]
        return real_move_targets[keep]

    def generate_s_moves(self, x: int, y: int, state: "State") -> torch.IntTensor:
        """Pregenerate moves for Sergeant."""
        # benchmarked against ray casting.
        moves = []
        origin = torch.IntTensor([x, y])
        own_mask = state.get_player_mask(state.player)
        other_mask = state.get_player_mask(not state.player)
        for direction in self._SERGEANT_DIRECTIONS:
            for i in range(1, 3):
                xi, yi = origin + i * direction
                if not self.valid_position(xi, yi):
                    break
                if own_mask[xi, yi]:
                    break
                moves.append((xi, yi))
                if other_mask[xi, yi]:
                    break
        if not moves:
            return torch.empty((0, 2), dtype=torch.int32)
        return torch.IntTensor(moves)

    def generate_o_moves(self, x: int, y: int, state: "State") -> torch.IntTensor:
        """Pregenerate moves for Officer."""
        # benchmarked against ray casting.
        moves = []
        origin = torch.IntTensor([x, y])
        own_mask = state.get_player_mask(state.player)
        other_mask = state.get_player_mask(not state.player)
        max_steps = max(self.BOARD_SIZE) - 1
        for direction in self._OFFICER_DIRECTIONS:
            for i in range(1, max_steps):
                xi, yi = origin + i * direction
                if not self.valid_position(xi, yi):
                    break
                if own_mask[xi, yi]:
                    break
                moves.append((xi, yi))
                if other_mask[xi, yi]:
                    break
        if not moves:
            return torch.empty((0, 2), dtype=torch.int32)
        return torch.IntTensor(moves)

    def _generate_move_tensor(self) -> torch.Tensor:
        """Generate empty output tensor."""
        torch.zeros()

    def _set_piece_position(self, key: str) -> None:
        """Set all piece positions."""
        x, y = self._initial_piece_dictionary[key]
        self._board[self._piece_layer_dict[key]][x, y] = 1

    def _set_pieces(self) -> None:
        for key in self._pieces:
            self._set_piece_position(key)
        self._board[self.num_pieces :, :, self.num_pieces :] = self._board[
            : self.num_pieces, :, self.num_pieces :
        ]
        self._board[: self.num_pieces, :, self.num_pieces :] = 0

    def empty_board(self) -> torch.Tensor:
        """Get empty board."""
        return torch.zeros(self.BOARD_DIMENSIONS)

    def initial_board(self) -> torch.Tensor:
        """Get initial board state."""
        self._set_pieces()
        return self._board.clone()

    @property
    def num_pieces(self) -> int:
        """Return number of pieces."""
        return len(self._pieces)

    @property
    def board_size(self) -> tuple[int, int]:
        """Allow access to the board size."""
        return deepcopy(self.BOARD_SIZE)

    @property
    def x(self) -> int:
        return self.BOARD_SIZE[0]

    @property
    def y(self) -> int:
        return self.BOARD_SIZE[1]

    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        return self.NUM_LAYERS

    @property
    def pieces(self) -> list[str]:
        return self._pieces.copy()

    def player_half(self, y: int, player: bool) -> bool:
        """Return if y is on the player side of the board."""
        return (
            int(player) * Board.board_size[1] // 2 <= y
            and y < (1 + int(player)) * Board.board_size[1] // 2
        )

    def half_indices(self, y: int) -> tuple[int, int]:
        """Get indices equating to player layers."""
        jbool = not (y < self.num_pieces)
        j1 = int(jbool) * self.num_pieces
        j2 = (int(jbool) + 1) * self.num_pieces
        return j1, j2

    def layer_from_piece(self, piece: str) -> int:
        """Get layer from piece."""
        return self._piece_layer_dict[piece]

    def value_from_piece(self, piece: str) -> float:
        """Get value from piece."""
        return self._piece_value_dict[piece]

    def moves_from_piece(self, piece: str) -> Callable[[int, int, "State"], int]:
        """Get moves from piece."""
        return self._piece_move_dict[piece]

    def piece_from_layer(self, layer: int) -> str:
        return self._pieces[layer % len(self._pieces)]

    def valid_position(self, x: int, y: int) -> bool:
        return (x >= 0) and (x < self.x) and (y >= 0) and (y < self.y)

    def valid_positions(self, positions: torch.Tensor) -> torch.BoolTensor:
        return (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < self.x)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < self.y)
        )

    @property
    def move_tensor_shape(self) -> tuple[int, int, int]:
        """Get shape for move tensor."""
        return (*self.board_size, self.MAX_MOVES)


Board = __BoardMaster()
