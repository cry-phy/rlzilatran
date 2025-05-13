from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.game import Game
    import tkinter as tk

from enum import Enum, auto
from game.utils import Board, MoveFlag
from mpi.handler import MPIHandler
import torch
import importlib
import pkgutil
import agents
import logging

logger = logging.getLogger(__name__)


class Tool(Enum):
    """Enum for handling selected tool."""

    NONE = auto()
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

    def __init__(self, game: Game) -> None:
        """Initialize agent with ownership over threads."""
        self.mpi: MPIHandler = MPIHandler()
        self._game: Game = game
        self._root: tk.Tk | None = None
        self._player: bool = game.state.player
        self._move_ready: bool = True
        self._move_flag: MoveFlag = MoveFlag.STANDARD
        self._move_tensor: torch.Tensor = torch.zeros(Board.move_tensor_shape)
        self._value: float = 0

        # For clickable Agents.
        self.selected: Tool = Tool.MOVE
        self._selected_var: tk.IntVar | None = None

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
        return self._selected_var

    @selected_var.setter
    def selected_var(self, value):
        self._selected_var = value

    @property
    def move_ready(self) -> bool:
        """Move ready flag."""
        return self._move_ready

    @property
    def move_flag(self) -> MoveFlag:
        """The move_flag property."""
        return self._move_flag

    @move_flag.setter
    def move_flag(self, value):
        self._move_flag = value

    def generate_move(
        self, move_tensor: torch.Tensor
    ) -> tuple[int, int, int, int, int]:
        """Generate a move for the current game."""
        raise NotImplementedError

    def generate_move_tensor(self) -> None:
        """Generate move tensor and value."""
        raise NotImplementedError

    def get_state_value(self) -> float:
        """Get evaluation of current State."""
        raise NotImplementedError

    def make_move(self, x1: int, y1: int, x2: int, y2: int, flag: int) -> None:
        """Make specific move."""
        self._game.make_move(x1, y1, x2, y2, flag)
        self.move_flag = MoveFlag.STANDARD

    # -- MPI compatability --
    def bcast_move(
        self, move: tuple[int, int, int, int, int] | None
    ) -> tuple[int, int, int, int, int]:
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

    def handle_click(self, event: tk.Event) -> None:
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


def __import_all_modules(package):
    """Import all submodules in certain module."""
    for _, module_name, _ in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        importlib.import_module(module_name)


def __get_all_subclasses(base_class):
    """Glob all subclasses of a base class in the namespace."""
    subclasses = set()

    def recurse(cls):
        for subclass in cls.__subclasses__():
            subclasses.add(subclass)
            recurse(subclass)

    recurse(base_class)
    return subclasses


def agent_class_dict() -> dict[str, type[BaseAgent]]:
    """Build full dictionary of agent classes."""
    __import_all_modules(agents)
    subclasses = __get_all_subclasses(BaseAgent)
    return {cls.__name__: cls for cls in subclasses}
