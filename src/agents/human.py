from copy import copy
from tkinter import Event
from typing import Callable
from agents.base import BaseAgent, Tool
from game.game import Game
import logging
import torch
from game.utils import Board
import functools
import inspect

from mpi.handler import MPIHandler

logger: logging.Logger = logging.getLogger(__name__)


def on_grid(fn):
    """Wrap functions only intended for Grid usage."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        x = bound.arguments["x"]
        y = bound.arguments["y"]
        if Board.valid_position(x, y):
            return fn(*args, **kwargs)
        else:
            logger.debug("Clicked position (%i,%i) outside %s", x, y, Board.board_size)

    return wrapper


PIECE_TOOL_DICT = {
    Tool.ADDP: "P",
    Tool.ADDS: "S",
    Tool.ADDO: "O",
}


class DummyAgent(BaseAgent):
    """Class for the non Observer ranks.

    Does not return any usefull data  but is able to act as a fully implemented Agent.
    """

    def generate_move(self, move_tensor: torch.Tensor) -> tuple[int, int, int, int]:
        """Generate a move.

        The Human player bypasses the Move selection by only reading on the observer rank.
        """
        return (-1, -1, -1, -1)

    def generate_move_tensor(self) -> None:
        """Generate empty move tensor and value.

        The Human player bypasses the Move selection by only reading on the observer rank.
        """
        self._move_tensor = torch.zeros(Board.move_tensor_shape)
        self._value = -1

    def reduce_move_tensors(self) -> torch.Tensor:
        """Reduce the move tensor to observer rank."""
        return self._move_tensor

    def reduce_values(self) -> float | None:
        """Reduce values from all ocomm."""
        return self._value


class HumanAgent(BaseAgent):
    """Handle GUI input from human.

    Needs to run only on the observer rank.
    """

    clickable: bool = True

    def __new__(cls, game: Game) -> "HumanAgent | BaseAgent":
        """Decide beween Human Agent and dummy agent."""
        mpi = MPIHandler()
        if mpi._orank == mpi.OBSERVER_ORANK:
            return DummyAgent(game)
        else:
            return super().__new__(cls)

    def __init__(self, game: Game) -> None:
        """Initialize Human interactible agent."""
        super().__init__(game)
        self._selected_position: tuple[int, int] = (0, 0)
        self.tool_function_dict: dict[Tool, Callable[[int, int], None]] = {
            Tool.MOVE: self.tool_move,
            Tool.INMOVE: self.tool_inmove,
            Tool.ADDP: self.tool_add,
            Tool.ADDS: self.tool_add,
            Tool.ADDO: self.tool_add,
            Tool.REMOVE: self.tool_remove,
            Tool.FORCE_SELECT: self.tool_force_select,
            Tool.FORCE_MOVE: self.tool_force_move,
        }
        self._CELL_SIZE = 80

    @property
    def cell_size(self):
        """The CELL_SIZE property."""
        return self._CELL_SIZE

    @cell_size.setter
    def cell_size(self, value):
        self._CELL_SIZE = value

    def selected_positions(self) -> list[tuple[int, int]]:
        """Return selected positions. Overriding BaseAgent."""
        if self._selected_position and self.selected == Tool.INMOVE:
            return [self._selected_position]
        else:
            return []

    def _select_tool(self, tool: Tool) -> None:
        """Centralize tool selection logic."""
        self.selected = tool
        if self._selected_var:
            self._selected_var.set(tool.value)

    def _clear_tool(self) -> None:
        """Clear tool selection."""
        self._select_tool(Tool.MOVE)

    def handle_click(self, event: Event) -> None:
        """Handle click event. Overriding Base."""
        x: int = event.x // self._CELL_SIZE
        y: int = event.y // self._CELL_SIZE
        logger.debug("Click registered: (%f, %f) -> (%i, %i)", event.x, event.y, x, y)

        logger.debug("Running %s", self.tool_function_dict[self.selected].__name__)
        self.tool_function_dict[self.selected](x, y)
        return super().handle_click(event)

    # -- MPI compatability --
    def generate_move(self, move_tensor: torch.Tensor) -> tuple[int, int, int, int]:
        """Generate a move.

        The Human player bypasses the Move selection by only reading on the observer rank.
        """
        return copy(self._move)

    def generate_move_tensor(self) -> None:
        """Generate empty move tensor and value.

        The Human player bypasses the Move selection by only reading on the observer rank.
        """
        winner = self._game.winner()
        if winner is None:
            self._value = 0
        elif winner == self._player:
            self._value = 1
        else:
            self._value = -1
        self._move_tensor = torch.zeros(Board.move_tensor_shape)

    # Overwrite MPI interactions since communication is not necessary
    def reduce_move_tensors(self) -> torch.Tensor:
        """Reduce the move tensor to observer rank."""
        return self._move_tensor

    def reduce_values(self) -> float | None:
        """Reduce values from all ocomm."""
        return self._value

    # -- Tools --
    @on_grid
    def tool_move(self, x: int, y: int) -> None:
        """Separate tool actions."""
        logger.debug(
            "Attempting to select square (%i, %i): %s",
            x,
            y,
            self.first_click_validity(x, y),
        )
        if self.first_click_validity(x, y):
            logger.debug("Selecting square (%i, %i)", x, y)
            self.select_square(x, y)
            self._select_tool(Tool.INMOVE)

    @on_grid
    def tool_inmove(self, x: int, y: int) -> None:
        """Separate tool actions."""
        logger.debug(
            "Attempting to move to square (%i, %i): %s",
            x,
            y,
            self.second_click_validity(x, y),
        )
        if self.second_click_validity(x, y):
            self.prepare_move(x, y)
        self._select_tool(Tool.MOVE)

    @on_grid
    def tool_add(self, x: int, y: int) -> None:
        """Separate tool actions."""
        logger.debug("Checking to add piece on (%i, %i)", x, y)
        if not self._game.state.square_filled(x, y):
            self.place_piece(x, y)

    @on_grid
    def tool_remove(self, x: int, y: int) -> None:
        """Separate tool actions."""
        self.remove_piece(x, y)

    @on_grid
    def tool_force_select(self, x: int, y: int) -> None:
        """Separate tool actions."""
        if self._game.state.square_filled(x, y):
            self.select_square(x, y)
            if not Board.player_half(y, self._game.state.player):
                self._game.state.player = not self._game.state.player
            self._select_tool(Tool.FORCE_MOVE)

    @on_grid
    def tool_force_move(self, x: int, y: int) -> None:
        """Separate tool actions."""
        if not (
            (x == self._selected_position[0]) and (y == self._selected_position[1])
        ):
            self.prepare_move(x, y)
        self._select_tool(Tool.FORCE_SELECT)

    @on_grid
    def select_square(self, x: int, y: int):
        """Select certain square."""
        self._selected_position = (x, y)
        self._possible_moves = self._game.get_piece_targets(x, y)
        self._select_tool(Tool.INMOVE)

    # -- Click validity --
    def first_click_validity(self, x: int, y: int) -> bool:
        """Check whether square is valid for selection."""
        return self._game.state.square_filled(x, y) and Board.player_half(
            y, self._game.state.player
        )

    def second_click_validity(self, x: int, y: int) -> bool:
        """Check whether square is valid for move."""
        return self._game.state.check_move(
            self._selected_position[0], self._selected_position[1], x, y
        )

    # -- Make moves --
    def prepare_move(self, x: int, y: int) -> None:
        """Make move to piece."""
        logger.debug("Setting move and readiness.")
        self._move = (self._selected_position[0], self._selected_position[1], x, y)
        self._move_ready = True

    def place_piece(self, x: int, y: int) -> None:
        """Place piece according to current tool on square."""
        logger.debug("Adding %s on (%i, %i)", PIECE_TOOL_DICT[self.selected], x, y)
        self._game.state.add_piece(x, y, PIECE_TOOL_DICT[self.selected])

    def remove_piece(self, x: int, y: int) -> None:
        """Clear square."""
        self._game.state.clear_square(x, y)

    # -- Visuals in GUI --
    def possible_moves(self, x: int, y: int) -> torch.Tensor:
        """Generate a move."""
        if self.selected == Tool.INMOVE:
            return self._possible_moves
        else:
            torch.empty(0, 2)
