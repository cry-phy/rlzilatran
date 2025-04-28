from copy import copy
from tkinter import Event
from typing import Callable
from agents.base import BaseAgent
from enum import Enum, auto
from game.game import Game
import logging
import torch
from game.utils import Board
import tkinter as tk
import functools
import inspect

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


PIECE_TOOL_DICT = {
    Tool.ADDP: "P",
    Tool.ADDS: "S",
    Tool.ADDO: "O",
}


class HumanAgent(BaseAgent):
    """Handle GUI input."""

    clickable: bool = True

    def __init__(self, game: Game, workers: tuple[int]) -> None:
        """Initialize Human interactible agent."""
        super().__init__(game, workers)
        self.selected: Tool | None = None
        self.selected_var: tk.IntVar | None = None
        self._selected_position: tuple[int, int] = []
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
        self.CELL_SIZE = 80

    @property
    def cell_size(self):
        """The CELL_SIZE property."""
        return self._CELL_SIZE

    @cell_size.setter
    def cell_size(self, value):
        self._CELL_SIZE = value

    def selected_positions(self) -> list[tuple[int, int]]:
        """Return selected positions. Overriding BaseAgent."""
        return [self._selected_position]

    def _select_tool(self, tool: Tool) -> None:
        """Centralize tool selection logic."""
        self.selected = tool
        if self.selected_var:
            self.selected_var.set(tool.value)

    def _clear_tool(self) -> None:
        """Clear tool selection."""
        self._select_tool(Tool.MOVE)

    def handle_click(self, event: Event) -> None:
        """Handle click event. Overriding Base."""
        x: int = event.x // self.CELL_SIZE
        y: int = event.y // self.CELL_SIZE
        logger.debug("Click registered: (%f, %f) -> (%i, %i)", event.x, event.y, x, y)
        if self.selected:
            self.tool_function_dict[self.selected](x, y)
        return super().handle_click(event)

    def generate_move(self) -> tuple[int, int, int, int]:
        """Generate a move."""
        return copy(self._move)

    def possible_moves(self, x: int, y: int) -> torch.IntTensor:
        """Generate a move."""
        if self.selected == Tool.INMOVE:
            return self._possible_moves
        else:
            torch.empty(0, 2)

    # -- Tools --
    @on_grid
    def tool_move(self, x: int, y: int) -> None:
        """Separate tool actions."""
        if self.first_click_validity(x, y):
            self.select_square(x, y)
            self._select_tool(Tool.INMOVE)

    @on_grid
    def tool_inmove(self, x: int, y: int) -> None:
        """Separate tool actions."""
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
        if not ((x == self.selected_position[0]) and (y == self.selected_position[1])):
            self.prepare_move(x, y)
        self._select_tool(Tool.FORCE_SELECT)

    @on_grid
    def select_square(self, x: int, y: int):
        """Select certain square."""
        self.selected_position = (x, y)
        self._possible_moves = self._game.get_piece_targets(x, y)
        self._select_tool(Tool.INMOVE)

    # -- Click validity --
    def first_click_validity(self, x: int, y: int) -> bool:
        """Check whether square is valid for selection."""
        return self._game.state.state[:, x, y].sum(axis=0) and Board.player_half(
            y, self._game.state.player
        )

    def second_click_validity(self, x: int, y: int) -> bool:
        """Check whether square is valid for move."""
        return self._game.state.check_move(
            self.selected_position[0], self.selected_position[1], x, y
        )

    # -- Make moves --
    def prepare_move(self, x: int, y: int) -> None:
        """Make move to piece."""
        self._move = (self.selected_position[0], self.selected_position[1], x, y)
        self._move_ready = True

    def place_piece(self, x: int, y: int) -> None:
        """Place piece according to current tool on square."""
        if self.selected:
            logger.debug("Adding %s on (%i, %i)", PIECE_TOOL_DICT[self.selected], x, y)
            self._game.state.add_piece(x, y, PIECE_TOOL_DICT[self.selected])

    def remove_piece(self, x: int, y: int) -> None:
        """Clear square."""
        self._game.state.clear_square(x, y)
