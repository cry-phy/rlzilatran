from tkinter import Event
from game.game import Game
from enum import Enum


class Tool(Enum):
    """Enum for handling selected tool."""

    NONE = 0


class BaseAgent:
    """Base class for all agents.

    Used for polymorphism and interface considerations.
    """

    clickable: bool = False

    def __init__(self, game: Game, workers: tuple[int, ...]) -> None:
        """Initialize agent with ownership over threads."""
        self._game = game
        self._workers = workers
        self._root = None
        self._player = game.state.player
        self._move_ready = False

    @property
    def player(self):
        """The player property."""
        return self._player

    @player.setter
    def player(self, value):
        self._player = value

    @property
    def selected_var(self):
        """The selected_var property."""
        return 0

    @selected_var.setter
    def selected_var(self, value):
        self._selected_var = value

    @property
    def move_ready(self) -> bool:
        """Move ready flag."""
        return self._move_ready

    def generate_move(self) -> tuple[int, int, int, int]:
        """Generate a move for the current game."""
        raise NotImplementedError

    def handle_click(self, event: Event) -> None:
        """Handle a click event.

        Mainly for human player but might be used to limit thinking
        time. Left empty on purpose as only implemented when its needed.
        """
        pass

    # -- Functions for GUI --
    def _select_tool(self, tool: Tool) -> None:
        """Centralize tool selection logic."""
        pass

    def _clear_tool(self) -> None:
        """Clear tool selection."""
        pass

    def selected_positions(self) -> list[tuple[int, int]]:
        """Get currently propable moves."""
        raise NotImplementedError

    def possible_moves(self, x: int, y: int) -> list[tuple[int, int]]:
        """Get currently propable moves."""
        raise NotImplementedError
