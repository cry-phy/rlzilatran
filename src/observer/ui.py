from typing import Callable
from game.game import Game
from game.state import State
from game.utils import Board
import tkinter as tk
from tkinter import Event, Menu, filedialog, messagebox
from observer.base import BaseObserver
import torch
from enum import Enum, auto
import logging
import functools
import inspect

logger: logging.Logger = logging.getLogger(__name__)


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


class UIObserver(BaseObserver):
    """Game UI for debugging and later play."""

    def __init__(self, root, game: Game) -> None:
        """Initialize window."""
        self.root = root
        self.game = game
        self.selected: Tool = Tool.MOVE
        self.selected_var: tk.IntVar = tk.IntVar(value=self.selected.value)
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

        # --- Menu Bar ---
        self.menubar = Menu(root)
        self._add_save_menu()
        self._add_tool_menu()
        self.root.config(menu=self.menubar)

        # --- Main Frame ---
        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # --- Canvas for Board ---
        self.COLUMNS, self.ROWS = Board.board_size
        self.CELL_SIZE: int = 80
        self.FONT_SIZE: int = 15
        self.SMALL_FONT: int = 10
        self.initial_resize_done = False

        canvas_width = self.COLUMNS * self.CELL_SIZE
        canvas_height = self.ROWS * self.CELL_SIZE
        self.canvas = tk.Canvas(
            self.main_frame,
            width=canvas_width,
            height=canvas_height,
            bg="white",
        )

        # -- Placement and resize --
        self.canvas.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self._on_resize)
        root.after(0, lambda: self._on_resize(self.canvas))

        # -- Add Binds --
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-2>", self._clear_tool)
        self.canvas.bind("<Button-3>", self._clear_tool)
        self.root.bind_all("<Control-s>", lambda _: self.on_save())
        self.root.bind_all("<Control-o>", lambda _: self.on_load())
        self.root.bind_all("<Control-n>", lambda _: self.new_empty())

        self.draw_board()

    def _add_save_menu(self) -> None:
        filemenu = Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Save", accelerator="Ctrl+S", command=self.on_save)
        filemenu.add_command(label="Load", accelerator="Ctrl+O", command=self.on_load)
        filemenu.add_separator()
        filemenu.add_command(
            label="Empty", accelerator="Ctrl+N", command=self.new_empty
        )
        filemenu.add_command(label="Restart", command=self.on_restart)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=filemenu)

    def _add_tool_menu(self) -> None:
        """Separate menubar logic."""
        toolmenu = tk.Menu(self.menubar, tearoff=0)
        for label, tool in [
            ("P", Tool.ADDP),
            ("S", Tool.ADDS),
            ("O", Tool.ADDO),
            ("X", Tool.REMOVE),
        ]:
            toolmenu.add_radiobutton(
                label=label,
                variable=self.selected_var,
                value=tool.value,
                command=lambda t=tool: self._select_tool(t),
            )
        toolmenu.add_separator()
        toolmenu.add_radiobutton(
            label="MOVE",
            variable=self.selected_var,
            value=Tool.MOVE.value,
            command=lambda: self._select_tool(Tool.MOVE),
        )
        toolmenu.add_radiobutton(
            label="FORCE",
            variable=self.selected_var,
            value=Tool.FORCE_SELECT.value,
            command=lambda: self._select_tool(Tool.FORCE_SELECT),
        )
        self.menubar.add_cascade(label="Tools", menu=toolmenu)

    def _on_resize(self, event_or_widget) -> None:
        """Resize canvas and scale/redraw contents proportionally."""
        if hasattr(event_or_widget, "width"):
            w, h = event_or_widget.width, event_or_widget.height
        else:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
        if not self.initial_resize_done:
            self.width, self.height = w, h
            self.initial_resize_done = True
            return
        if w <= 1 or h <= 1:
            return
        real_w, real_h = w * 0.96, h * 0.96
        self.CELL_SIZE = int(real_h / self.ROWS)
        self.FONT_SIZE = int(15 * self.CELL_SIZE / 80)
        self.SMALL_FONT = int(2 * self.FONT_SIZE / 3)
        logger.debug(
            "Resizing (CELL_SIZE, FONT_SIZE, SMALL_FONT) -> (%i, %i,%i,)",
            self.CELL_SIZE,
            self.FONT_SIZE,
            self.SMALL_FONT,
        )
        wscale = real_w / self.width
        hscale = real_h / self.height
        self.canvas.scale("all", 0, 0, wscale, hscale)
        self.width = self.COLUMNS * self.CELL_SIZE
        self.height = self.ROWS * self.CELL_SIZE
        self.canvas.config(width=real_w, height=real_h)
        self.redraw_board()

    def _select_tool(self, tool: Tool) -> None:
        """Centralize tool selection logic."""
        self.selected = tool
        self.selected_var.set(tool.value)
        self.redraw_board()

    def _clear_tool(self, event: Event) -> None:
        """Clear tool selection."""
        self._select_tool(Tool.MOVE)

    def _get_cell_boundaries(self, x: int, y: int) -> tuple[int, int, int, int]:
        """Find boundary of cell for drawing Board."""
        x1: float = x * self.CELL_SIZE
        y1: float = y * self.CELL_SIZE
        x2: float = x1 + self.CELL_SIZE
        y2: float = y1 + self.CELL_SIZE
        return x1, y1, x2, y2

    def draw_grid(self) -> None:
        """Draw grid on which the Game is rendered."""
        for y in range(self.ROWS):
            for x in range(self.COLUMNS):
                self.canvas.create_rectangle(
                    *self._get_cell_boundaries(x, y), outline="black", fill="lightgray"
                )

    def draw_grid_coords(self) -> None:
        """Draw coordinates for debugging."""
        for y in range(self.ROWS):
            for x in range(self.COLUMNS):
                xf: float = (x + 0.5) * self.CELL_SIZE
                yf: float = (y + 0.9) * self.CELL_SIZE
                self.canvas.create_text(
                    xf,
                    yf,
                    text=f"{x} {y}",
                    fill="gray",
                    font=(f"Helvetica {self.SMALL_FONT}"),
                )

    def draw_pieces(self) -> None:
        """Draw pieces."""
        current_state = self.game.state.state
        for piece in Board.pieces:
            piece_layer = (
                current_state[Board.layer_from_piece(piece)]
                + current_state[Board.layer_from_piece(piece) + Board.num_pieces]
            )
            positions: tuple[torch.Tensor, torch.Tensor] = torch.where(piece_layer)
            xs: torch.Tensor = (positions[0] + 0.5) * self.CELL_SIZE
            ys: torch.Tensor = (positions[1] + 0.5) * self.CELL_SIZE
            for x, y in zip(xs, ys, strict=False):
                self.canvas.create_text(
                    float(x),
                    float(y),
                    text=piece,
                    fill="black",
                    font=(f"Helvetica {self.FONT_SIZE} bold"),
                )

    def draw_selected(self) -> None:
        """Draw selected square."""
        if self.selected in (Tool.INMOVE, Tool.FORCE_MOVE):
            self.canvas.create_rectangle(
                *self._get_cell_boundaries(*self.selected_position),
                outline="black",
                fill="lightblue",
            )
            for move in self.possible_moves:
                self.canvas.create_rectangle(
                    *self._get_cell_boundaries(int(move[0]), int(move[1])),
                    outline="black",
                    fill="orchid1",
                )

    def on_save(self) -> None:
        """Save current game as json."""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("State files", "*.json"), ("All Files", "*.*")],
        )
        if path:
            self.game.save(path)
            messagebox.showinfo("Save", f"Game saved to {path}")

    def on_load(self) -> None:
        """Load game from pickled state."""
        path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("State files", "*.json"), ("All Files", "*.*")],
        )
        if path:
            self.game.load(path)
            self.redraw_board()

    def new_empty(self) -> None:
        """Generate empty board."""
        self.game.state = State.empty()
        self.redraw_board()

    def on_restart(self) -> None:
        """Reset game to initial."""
        self.game.reset()
        self.redraw_board()

    def on_click(self, event: Event) -> None:
        """Recognize cell click."""
        x: int = event.x // self.CELL_SIZE
        y: int = event.y // self.CELL_SIZE
        logger.debug("Click registered: (%f, %f) -> (%i, %i)", event.x, event.y, x, y)
        # to agent
        self.tool_function_dict[self.selected](x, y)
        self.redraw_board()
        print(self.game.state.player)

    # to agent
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
            self.move_piece(x, y)
        self._select_tool(Tool.MOVE)

    @on_grid
    def tool_add(self, x: int, y: int) -> None:
        """Separate tool actions."""
        logger.debug("Checking to add piece on (%i, %i)", x, y)
        if not self.game.state.square_filled(x, y):
            self.place_piece(x, y)

    @on_grid
    def tool_remove(self, x: int, y: int) -> None:
        """Separate tool actions."""
        self.remove_piece(x, y)

    @on_grid
    def tool_force_select(self, x: int, y: int) -> None:
        """Separate tool actions."""
        if self.game.state.square_filled(x, y):
            self.select_square(x, y)
            if not Board.player_half(y, self.game.state.player):
                self.game.state.player = not self.game.state.player
            self._select_tool(Tool.FORCE_MOVE)

    @on_grid
    def tool_force_move(self, x: int, y: int) -> None:
        """Separate tool actions."""
        if not ((x == self.selected_position[0]) and (y == self.selected_position[1])):
            self.move_piece(x, y)
        self._select_tool(Tool.FORCE_SELECT)

    @on_grid
    def select_square(self, x: int, y: int):
        """Select certain square."""
        self.selected_position = (x, y)
        self.possible_moves = self.game.get_piece_targets(x, y)
        self._select_tool(Tool.INMOVE)

    def first_click_validity(self, x: int, y: int) -> bool:
        """Check whether square is valid for selection."""
        return self.game.state.state[:, x, y].sum(axis=0) and Board.player_half(
            y, self.game.state.player
        )

    def second_click_validity(self, x: int, y: int) -> bool:
        """Check whether square is valid for move."""
        return self.game.state.check_move(
            self.selected_position[0], self.selected_position[1], x, y
        )

    def move_piece(self, x: int, y: int) -> None:
        """Make move to piece."""
        self.game.make_move(self.selected_position[0], self.selected_position[1], x, y)

    def place_piece(self, x: int, y: int) -> None:
        """Place piece according to current tool on square."""
        logger.debug("Adding %s on (%i, %i)", PIECE_TOOL_DICT[self.selected], x, y)
        self.game.state.add_piece(x, y, PIECE_TOOL_DICT[self.selected])

    def remove_piece(self, x: int, y: int) -> None:
        """Clear square."""
        self.game.state.clear_square(x, y)

    # stays
    def draw_board(self) -> None:
        """Wrap common draw functions."""
        self.draw_grid()
        self.draw_selected()
        self.draw_pieces()
        self.draw_grid_coords()

    def redraw_board(self) -> None:
        """Redraw game state."""
        self.canvas.delete("all")
        self.draw_board()
