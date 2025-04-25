from game.game import Game
from game.state import State
from game.utils import Board
import tkinter as tk
from tkinter import Event, Menu, filedialog, messagebox
import torch
from enum import Enum, auto


class Tool(Enum):
    """Enum for handling selected tool."""

    NONE = auto()
    INMOVE = auto()
    ADDP = auto()
    ADDS = auto()
    ADDO = auto()
    REMOVE = auto()


PIECE_TOOL_DICT = {
    Tool.ADDP: "P",
    Tool.ADDS: "S",
    Tool.ADDO: "O",
}


class GameUI:
    """Game UI for debugging and later play."""

    CELL_SIZE = 80

    def __init__(self, root, game: Game) -> None:
        """Initialize window."""
        self.game = game
        self.selected: Tool = Tool.NONE
        self.selected_var: tk.IntVar = tk.IntVar(value=self.selected.value)

        # --- Menu Bar ---
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save", accelerator="Ctrl+S", command=self.on_save)
        filemenu.add_command(label="Load", accelerator="Ctrl+O", command=self.on_load)
        filemenu.add_separator()
        filemenu.add_command(
            label="Empty", accelerator="Ctrl+N", command=self.new_empty
        )
        filemenu.add_command(label="Restart", command=self.on_restart)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        toolmenu = tk.Menu(menubar, tearoff=0)
        toolmenu.add_radiobutton(
            label="P",
            variable=self.selected_var,
            value=Tool.ADDP.value,
            command=lambda: self._select_tool(Tool.ADDP),
        )
        toolmenu.add_radiobutton(
            label="S",
            variable=self.selected_var,
            value=Tool.ADDS.value,
            command=lambda: self._select_tool(Tool.ADDS),
        )

        toolmenu.add_radiobutton(
            label="O",
            variable=self.selected_var,
            value=Tool.ADDO.value,
            command=lambda: self._select_tool(Tool.ADDO),
        )
        toolmenu.add_radiobutton(
            label="X",
            variable=self.selected_var,
            value=Tool.REMOVE.value,
            command=lambda: self._select_tool(Tool.REMOVE),
        )
        toolmenu.add_separator()
        toolmenu.add_radiobutton(
            label="None",
            variable=self.selected_var,
            value=Tool.NONE.value,
            command=lambda: self._select_tool(Tool.NONE),
        )
        menubar.add_cascade(label="Tools", menu=toolmenu)
        root.config(menu=menubar)

        self.selected_position: tuple[int, int] = (-1, -1)
        self.selected_piece: str = ""
        self.possible_moves: torch.IntTensor = None

        # --- Canvas for Board ---
        self.COLUMNS, self.ROWS = Board.board_size
        canvas_width = self.COLUMNS * self.CELL_SIZE
        canvas_height = self.ROWS * self.CELL_SIZE
        self.canvas = tk.Canvas(
            root, width=canvas_width, height=canvas_height, bg="white"
        )
        self.canvas.pack(padx=10, pady=10)

        self.draw_board()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-2>", self._clear_tool)
        root.bind_all("<Control-s>", lambda e: self.on_save())
        root.bind_all("<Control-o>", lambda e: self.on_load())
        root.bind_all("<Control-n>", lambda e: self.new_empty())

    def _select_tool(self, tool: Tool) -> None:
        """Centralize tool selection logic."""
        self.selected = tool
        self.selected_var.set(tool.value)
        self.redraw_board()

    def _clear_tool(self, event: Event) -> None:
        """Clear tool selection."""
        self._select_tool(Tool.NONE)

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
                yf: float = (y + 0.1) * self.CELL_SIZE
                self.canvas.create_text(
                    xf,
                    yf,
                    text=f"{x}   {y}",
                    fill="gray",
                    font=("Helvetica 10"),
                )

    def draw_pieces(self) -> None:
        """Draw pieces."""
        current_state = self.game.state.get_state()
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
                    font=("Helvetica 15 bold"),
                )

    def draw_selected(self) -> None:
        """Draw selected square."""
        if self.selected == Tool.INMOVE:
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
        """Save current game as pickle."""
        path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch files", "*.pt"), ("All Files", "*.*")],
        )
        if path:
            self.game.save(path)
            messagebox.showinfo("Save", f"Game saved to {path}")

    def on_load(self) -> None:
        """Load game from pickled state."""
        path = filedialog.askopenfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch files", "*.pt"), ("All Files", "*.*")],
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
        match self.selected:
            case Tool.NONE:
                if self.first_click_validity(x, y):
                    self.select_square(x, y)
            case Tool.INMOVE:
                if (x, y) == self.selected_position:
                    self._select_tool(Tool.NONE)
                if self.second_click_validity(x, y):
                    self.move_piece(x, y)
            case Tool.ADDP | Tool.ADDS | Tool.ADDO:
                if not self.game.state.square_filled(x, y):
                    self.place_piece(x, y)
            case Tool.REMOVE:
                self.remove_piece(x, y)
        self.redraw_board()

    def select_square(self, x: int, y: int):
        """Select certain square."""
        if self.selected == Tool.NONE:
            self.selected_position = (x, y)
            self.possible_moves = self.game.get_piece_targets(x, y)
            self._select_tool(Tool.INMOVE)

    def first_click_validity(self, x: int, y: int) -> bool:
        """Check whether square is valid for selection."""
        return (
            self.game.state.get_state()[:, x, y].sum(axis=0)
            and Board.player_half(y)
            and self.selected == Tool.NONE
        )

    def second_click_validity(self, x: int, y: int) -> bool:
        """Check whether square is valid for move."""
        return self.selected == Tool.INMOVE and self.game.state.check_move(
            self.selected_position[0], self.selected_position[1], x, y
        )

    def move_piece(self, x: int, y: int) -> None:
        """Make move to piece."""
        self.game.state.make_move(
            self.selected_position[0], self.selected_position[1], x, y
        )
        self._select_tool(Tool.NONE)

    def place_piece(self, x: int, y: int) -> None:
        """Place piece according to current tool on square."""
        self.game.state.add_piece(x, y, PIECE_TOOL_DICT[self.selected])

    def remove_piece(self, x: int, y: int) -> None:
        """Clear square."""
        self.game.state.clear_square(x, y)

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
