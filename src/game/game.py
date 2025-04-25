from game.state import State
from game.utils import Board
from typing import Callable, Generic, TypeVar
import torch

G = TypeVar("G")


class Game(Generic[G]):
    """Game/States/Rules."""

    def __init__(self) -> None:
        """Generate Game instance.

        Should not be used directly! Use factory Methods instead.
        """
        self.state: State = State.initial_state()

    @classmethod
    def new_game(cls: type[G]) -> G:
        """Generate Initial game state."""
        return cls()

    def reset(self) -> None:
        """Reset game state."""
        self.state = State.initial_state()

    def save(self, path: str) -> None:
        """Save game state to torch tensor file."""
        self.state.save(path)

    def load(self, path: str) -> None:
        """Load game state from torch tensor file."""
        self.state.load(path)

    def get_piece_targets(self, x: int, y: int) -> torch.Tensor:
        """Get all move target squares of a single piece."""
        layers: torch.IntTensor = self.state.get_layer(x, y)
        if layers.nelement() != 0:
            piece: str = Board.piece_from_layer(layers)
            move_generator: Callable = Board.moves_from_piece(piece)
            return move_generator(x, y, self.state)
