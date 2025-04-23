import torch
from copy import deepcopy

__all__ = ["board_generator"]


class __BoardGenerator:
    """Generator for board tensors.

    This is a module level singleton, i. e. being loaded from config at import time.
    TODO: Extend to read config Files.
    """

    def __init__(self) -> None:
        """Empty initializer use Factory methods instead."""
        self.BOARD_SIZE = (4, 8)
        self.NUM_LAYERS = 7
        self.BOARD_DIMENSIONS = (self.NUM_LAYERS, *self.BOARD_SIZE)
        self.PRIVATE_LAYER = 2
        self.PRIVATE_POSITIONS = [
            (2, 1),
            (1, 2),
            (2, 2),
            (-3, -2),
            (-2, -3),
            (-3, -3),
        ]
        self.SERGEANT_LAYER = 1
        self.SERGEANT_POSITIONS = [
            (0, 2),
            (1, 1),
            (2, 0),
            (-1, -3),
            (-2, -2),
            (-3, -1),
        ]
        self.OFFICER_LAYER = 0
        self.OFFICER_POSITIONS = [
            (0, 0),
            (0, 1),
            (1, 0),
            (-1, -1),
            (-1, -2),
            (-2, -1),
        ]
        self._board: torch.Tensor = torch.empty(self.BOARD_DIMENSIONS)

    def empty_board(self) -> torch.Tensor:
        """Get empty board."""
        return torch.empty(self.BOARD_DIMENSIONS)

    def initial_board(self) -> torch.Tensor:
        """Get initial board state."""
        self.set_privates()
        self.set_sergeants()
        self.set_officers()
        return self._board.clone()

    def board_size(self) -> tuple[int, int]:
        """Allow access to the board size."""
        return deepcopy(self.BOARD_SIZE)

    def num_layers(self) -> int:
        """Get number of layers."""
        return self.NUM_LAYERS

    def set_privates(self) -> None:
        """Get Private Board."""
        self._board[self.PRIVATE_LAYER, self.PRIVATE_POSITIONS] = 1

    def set_sergeants(self) -> None:
        """Get Sergeant Board."""
        self._board[self.SERGEANT_LAYER, self.SERGEANT_POSITIONS] = 1

    def set_officers(self) -> None:
        """Get Officer Board."""
        self._board[self.OFFICER_LAYER, self.OFFICER_POSITIONS] = 1


board_generator = __BoardGenerator()
