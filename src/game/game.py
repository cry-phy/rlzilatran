from typing import Generic, TypeVar

G = TypeVar("G")


class Game(Generic[G]):
    """Game/States/Rules."""

    def __init__(self) -> None:
        """Generate Game instance.

        Should not be used directly! Use factory Methods instead.
        """
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
        """Render game stae for debugging."""
