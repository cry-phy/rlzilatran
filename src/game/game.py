from typing import Generic, TypeVar

from game.state import State

G = TypeVar("G")


class Game(Generic[G]):
    """Game/States/Rules."""

    def __init__(self) -> None:
        """Generate Game instance.

        Should not be used directly! Use factory Methods instead.
        """
        self.state: State = State.initial_state()
        ...

    @classmethod
    def new_game(cls: type[G]) -> G:
        """Generate Initial game state."""
        ...
        return cls()

    def reset(self) -> None:
        """Reset game state."""
        ...

    def seed(self, seed: int) -> None:
        """Set seed for reproducability."""
        ...

    def render(self) -> None:
        """Render game state for debugging."""

    def save(self, path: str) -> None:
        """Save game state to torch tensor file."""
        self.state.save(path)

    def load(self, path: str) -> None:
        """Load game state from torch tensor file."""
        self.state.load(path)
