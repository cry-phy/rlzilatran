from game.game import Game


class BaseAgent:
    """Base class for all agents.

    Used for polymorphism and interface considerations.
    """

    def __init__(self, game: Game, workers: tuple[int]) -> None:
        """Initialize agent with ownership over threads."""
        self._game = game
        self._workers = workers
        self._root = None
        self._player = game.state.player

    @property
    def player(self):
        """The player property."""
        return self._player

    @player.setter
    def player(self, value):
        self._player = value

    def generate_move(self) -> tuple[int, int, int, int]:
        """Generate a move for the current game."""
        raise NotImplementedError
