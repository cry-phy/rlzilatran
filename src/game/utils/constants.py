from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Iterable

import numpy as np
from copy import copy
import torch
import json
import logging

logger: logging.Logger = logging.getLogger(__name__)


class __Constants:
    """Holds Constants for the game.

    This is a module level singleton, i. e. being loaded from config at import time.
    """

    def __init__(self, config_path="constants.json"):
        """Generate constants needed for computation."""
        # Read Variables from files.
        with open(config_path) as f:
            config = json.load(f)
            self.PRIVATE_MOVES = torch.tensor(
                config.get("PRIVATE_MOVES", []), dtype=torch.int
            )
            self.SERGEANT_DIRECTIONS = torch.tensor(
                config.get("SERGEANT_DIRECTIONS", []), dtype=torch.int
            )
            self.OFFICER_DIRECTIONS = torch.tensor(
                config.get("OFFICER_DIRECTIONS", []), dtype=torch.int
            )
            self.PRIVATE_POSITIONS = torch.tensor(
                config.get("PRIVATE_POSITIONS", []), dtype=torch.int
            ).T
            self.SERGEANT_POSITIONS = torch.tensor(
                config.get("SERGEANT_POSITIONS", []), dtype=torch.int
            ).T
            self.OFFICER_POSITIONS = torch.tensor(
                config.get("OFFICER_POSITIONS", []), dtype=torch.int
            ).T
            self._pieces = np.array(config.get("PIECES", []))
            self.BOARD_SIZE = tuple(config.get("BOARD_SIZE", ()))
            self.HISTORY_LENGTH = config.get("HISTORY_LENGTH", 0)
            self.BATCH_SIZE = config.get("BATCH_SIZE", 0)
            self._capture_clock = config.get("CAPTURE_CLOCK", 0)
            self._piece_value_dict = config.get("PIECE_VALUE_DICT", {})
            self._initial_piece_dictionary = {
                piece: torch.tensor(positions, dtype=torch.int).T
                for piece, positions in config.get(
                    "INITIAL_PIECE_DICTIONARY", {}
                ).items()
            }

        # Generate the rest from read variables
        self._max_capture_value = self._generate_max_capture_value()
        self.NUM_LAYERS = 2 * len(self._pieces)
        self.BOARD_DIMENSIONS = (self.NUM_LAYERS, *self.BOARD_SIZE)
        self.MAX_MOVES = self.x * self.y + 1
        self.CHANNEL_NUMBER = (len(self._pieces) * 2) * self.HISTORY_LENGTH + 4
        self._piece_unique_dict: dict[str, int] = {
            key: i for i, key in enumerate(self._pieces)
        }

        self._layer_list = torch.tensor(self.convert_dict_list(self._piece_unique_dict))
        self._value_list = torch.tensor(self.convert_dict_list(self._piece_value_dict))

        self._init_board = self._get_init_board()

    def _generate_max_capture_value(self) -> float:
        """Pregenerate sum of all captures."""
        total_values_piece = [
            self._piece_value_dict[piece]
            * self._initial_piece_dictionary[piece].shape[1]
            for piece in self._pieces
        ]
        return sum(total_values_piece)

    def convert_dict_list(self, piece_any_dict) -> list[Any]:
        """Convert a piece to any dictionary into a unique identifiers dict."""
        converter = [None] * len(piece_any_dict)
        for key in piece_any_dict.keys():
            converter[self._piece_unique_dict[key]] = piece_any_dict[key]
        return converter

    def _set_piece_position(self, board: torch.Tensor, key: str) -> None:
        """Set all piece positions."""
        x, y = self._initial_piece_dictionary[key]
        board[self._piece_unique_dict[key]][x, y] = 1

    def _get_init_board(self) -> torch.Tensor:
        board = torch.zeros(self.BOARD_DIMENSIONS, dtype=torch.int32)
        for key in self._pieces:
            self._set_piece_position(board, key)
        board[self.num_pieces :, :, self.num_pieces :] = board[
            : self.num_pieces, :, self.num_pieces :
        ]
        board[: self.num_pieces, :, self.num_pieces :] = 0
        return board

    def empty_board(self) -> torch.Tensor:
        """Get empty board."""
        return torch.zeros(self.BOARD_DIMENSIONS)

    def initial_board(self) -> torch.Tensor:
        """Get initial board state."""
        return self._init_board.clone()

    @property
    def num_pieces(self) -> int:
        """Return number of pieces."""
        return len(self._pieces)

    @property
    def board_size(self) -> tuple[int, int]:
        """Allow access to the board size."""
        return copy(self.BOARD_SIZE)

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
    def pieces(self) -> Iterable[str]:
        return self._pieces.copy()

    @property
    def max_capture_value(self) -> float:
        return self._max_capture_value

    @property
    def capture_clock(self) -> int:
        return self._capture_clock

    @property
    def move_tensor_shape(self) -> tuple[int, int, int]:
        """Get shape for move tensor."""
        return (*self.board_size, self.MAX_MOVES)

    @property
    def state_tensor_shape(self) -> tuple[int, int, int]:
        return (self.CHANNEL_NUMBER, *self.board_size)

    @property
    def state_tensor_batch_shape(self) -> tuple[int, int, int, int]:
        """Get shape for move tensor."""
        return (self.BATCH_SIZE, *self.state_tensor_shape)

    @property
    def move_tensor_batch_shape(self) -> tuple[int, int, int, int]:
        """Get shape for move tensor."""
        return (self.BATCH_SIZE, *self.move_tensor_shape)

    @property
    def leaf_batch_shape(self) -> tuple[int, int, int, int]:
        return (self.BATCH_SIZE, *self.state_tensor_shape)

    @property
    def value_batch_shape(self) -> tuple[int]:
        return (self.BATCH_SIZE,)

    @property
    def leaf_batch_dtype(self) -> str:
        return "f4"

    @property
    def value_batch_dtype(self) -> str:
        return "f4"


__all__ = ["Constants"]
Constants = __Constants()
