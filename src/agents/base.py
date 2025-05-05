from tkinter import Event
from game.game import Game
from enum import Enum, auto
from game.utils import Board
from mpi.handler import MPIHandler
import torch


class Tool(Enum):
    """Enum for handling selected tool."""

    MOVE = auto()
    INMOVE = auto()
    ADDP = auto()
    ADDS = auto()
    ADDO = auto()
    REMOVE = auto()
    FORCE_SELECT = auto()
    FORCE_MOVE = auto()


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
        self._move_tensor = torch.zeros(Board.move_tensor_shape)
        self._value = 0
        self.mpi = MPIHandler()

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
        raise NotImplementedError

    @selected_var.setter
    def selected_var(self, value):
        raise NotImplementedError

    @property
    def move_ready(self) -> bool:
        """Move ready flag."""
        return self._move_ready

    def generate_move(self, move_tensor: torch.Tensor) -> tuple[int, int, int, int]:
        """Generate a move for the current game."""
        raise NotImplementedError

    def generate_move_tensor(self) -> None:
        """Generate move tensor and value."""
        raise NotImplementedError

    def get_state_value(self) -> float:
        """Get evaluation of current State."""
        raise NotImplementedError

    def make_move(self, x1, y1, x2, y2) -> None:
        """Make specific move."""
        self._game.make_move(x1, y1, x2, y2)
        self._move_ready = False

    # -- MPI compatability --
    def bcast_move(
        self, move: tuple[int, int, int, int] | None
    ) -> tuple[int, int, int, int]:
        """Broadcast the next move to all node ranks."""
        bcast_move = self.mpi.bcast_decided_move(move)
        return bcast_move

    def reduce_move_tensors(self) -> torch.Tensor:
        """Reduce the move tensor to observer rank."""
        sum_move_tensor = self.mpi.reduce_move_tensors(self._move_tensor)
        return sum_move_tensor

    def reduce_values(self) -> float | None:
        """Reduce values from all ocomm."""
        red_value: float | None = self.mpi.reduce_values(self._value)
        return red_value

    # -- Functions for GUI --
    @property
    def cell_size(self):
        """The CELL_SIZE property."""
        raise NotImplementedError

    @cell_size.setter
    def cell_size(self, value):
        raise NotImplementedError

    def handle_click(self, event: Event) -> None:
        """Handle a click event.

        Mainly for human player but might be used to limit thinking
        time. Left empty on purpose as only implemented when its needed.
        """
        pass

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
