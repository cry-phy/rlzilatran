import torch
from copy import deepcopy
from functools import cache

__all__ = ["board_generator"]


class __BoardGenerator:
    """Generator for board tensors.

    This is a module level singleton, i. e. being loaded from config at import time.
    TODO: Extend to read config Files.
    """

    def __init__(self) -> None:
        """Empty initializer use Factory methods instead."""
        PRIVATE_LAYER = 2
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
        SERGEANT_LAYER = 1
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
        OFFICER_LAYER = 0
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
        self._pieces = ["P", "S", "O"]
        self.BOARD_SIZE = (4, 8)
        self.NUM_LAYERS = 2 * len(self._pieces) + 1
        self.BOARD_DIMENSIONS = (self.NUM_LAYERS, *self.BOARD_SIZE)
        self._piece_layer_dict = {
            "P": PRIVATE_LAYER,
            "S": SERGEANT_LAYER,
            "O": OFFICER_LAYER,
        }
        self._piece_value_dict = {
            "P": 1,
            "S": 2,
            "O": 3,
        }
        self._piece_move_dict = {
            "P": None,
            "S": None,
            "O": None,
        }
        self._initial_piece_dictionary = {
            "P": PRIVATE_POSITIONS,
            "S": SERGEANT_POSITIONS,
            "O": OFFICER_POSITIONS,
        }
        self._board: torch.Tensor = torch.zeros(self.BOARD_DIMENSIONS)
        self._layer_values: torch.Tensor = torch.zeros(self.NUM_LAYERS)
        for piece in self._pieces:
            self._layer_values[self._piece_layer_dict[piece]] = self._piece_value_dict[
                piece
            ]
            self._layer_values[-(self._piece_layer_dict[piece] + 1)] = (
                self._piece_value_dict[piece]
            )

    def empty_board(self) -> torch.Tensor:
        """Get empty board."""
        return torch.zeros(self.BOARD_DIMENSIONS)

    def initial_board(self) -> torch.Tensor:
        """Get initial board state."""
        self._set_pieces()
        return self._board.clone()

    def board_size(self) -> tuple[int, int]:
        """Allow access to the board size."""
        return deepcopy(self.BOARD_SIZE)

    def num_layers(self) -> int:
        """Get number of layers."""
        return self.NUM_LAYERS

    def player_half(self, y: int) -> bool:
        """Return if y is on the player side of the board."""
        return y < int(self.BOARD_SIZE[1] / 2)

    def pieces(self) -> list[str]:
        return deepcopy(self._pieces)

    def piece_layer_dict(self) -> dict[str, int]:
        return self._piece_layer_dict

    def layer_value(self, index: int) -> float:
        """Return value of layer piece."""
        return self._layer_values[index]

    @cache
    def layer_values(self) -> torch.Tensor:
        return self._layer_values.clone()

    def _set_piece_position(self, key: str) -> None:
        """Get Private Board."""
        x, y = self._initial_piece_dictionary[key]
        self._board[self._piece_layer_dict[key]][x, y] = 1

    def _set_pieces(self) -> None:
        for key in self._pieces:
            self._set_piece_position(key)
        self._board[len(self._pieces) + 1 :, :, int(self.BOARD_SIZE[1] / 2) :] = (
            self._board[: len(self._pieces), :, int(self.BOARD_SIZE[1] / 2) :]
        )
        self._board[: len(self._pieces), :, int(self.BOARD_SIZE[1] / 2) :] = 0


board_generator = __BoardGenerator()
