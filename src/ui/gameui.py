from game.game import Game
from game.utils import board_generator
import tkinter as tk
from tkinter import Menu, filedialog, messagebox
import torch


class GameUI:
    """Game UI for debugging and later play."""

    CELL_SIZE = 80

    def __init__(self, root, game: Game):
        self.game = game

        # --- Menu Bar ---
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save", command=self.on_save)
        filemenu.add_command(label="Load", command=self.on_load)
        filemenu.add_separator()
        filemenu.add_command(label="Restart", command=self.on_restart)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)

        self.COLUMNS, self.ROWS = board_generator.board_size()
        self.selected: bool = False
        self.selected_position: tuple[int, int] = (-1, -1)

        # --- Canvas for Board ---
        canvas_width = self.COLUMNS * self.CELL_SIZE
        canvas_height = self.ROWS * self.CELL_SIZE
        self.canvas = tk.Canvas(
            root, width=canvas_width, height=canvas_height, bg="white"
        )
        self.canvas.pack(padx=10, pady=10)

        self.draw_board()

        self.canvas.bind("<Button-1>", self.on_click)

    def _get_cell_boundaries(self, x: int, y: int) -> tuple[int, int, int, int]:
        x1 = x * self.CELL_SIZE
        y1 = y * self.CELL_SIZE
        x2 = x1 + self.CELL_SIZE
        y2 = y1 + self.CELL_SIZE
        return x1, y1, x2, y2

    def draw_grid(self) -> None:
        """Draw grid on which the Game is rendered."""
        for y in range(self.ROWS):
            for x in range(self.COLUMNS):
                self.canvas.create_rectangle(
                    *self._get_cell_boundaries(x, y), outline="black", fill="lightgray"
                )

    def draw_pieces(self):
        """Draw pieces."""
        current_state = self.game.state.get_state()
        piece_layer_dict = board_generator.piece_layer_dict()
        for piece in board_generator.pieces():
            piece_layer = (
                current_state[piece_layer_dict[piece]]
                + current_state[-(piece_layer_dict[piece] + 1)]
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
        if self.selected:
            self.canvas.create_rectangle(
                *self._get_cell_boundaries(*self.selected_position),
                outline="black",
                fill="lightblue",
            )

    def on_save(self):
        """Save current game as pickle."""
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if path:
            self.game.save(path)
            messagebox.showinfo("Save", f"Game saved to {path}")

    def on_load(self):
        """Load game from pickled state."""
        path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if path:
            self.game.load(path)
            self.redraw_board()

    def on_restart(self):
        """Reset game to initial."""
        self.game.reset()
        self.redraw_board()

    def on_click(self, event):
        """Recognize cell click."""
        x = event.x // self.CELL_SIZE
        y = event.y // self.CELL_SIZE
        if self.first_click_validity(x, y):
            self.select_square(x, y)
        elif self.second_click_validity(x, y):
            self.move_piece(x, y)
        self.redraw_board()

    def select_square(self, x, y):
        """Select certain square."""
        if not self.selected:
            self.selected_position = (x, y)
            self.selected = True
        else:
            self.selected = False

    def first_click_validity(self, x, y) -> bool:
        """Check whether square is valid for selection."""
        return (
            self.game.state.get_state()[:, x, y].sum(axis=0)
            and y < int(board_generator.board_size()[1] / 2)
            and not self.selected
        )

    def second_click_validity(self, x, y) -> bool:
        """Check whether square is valid for move."""
        return self.selected and self.game.state.check_move(
            self.selected_position[0], self.selected_position[1], x, y
        )

    def move_piece(self, x, y) -> None:
        """Make move to piece.

        TODO: Only to generate and test till move logic is valid.
        """
        self.game.state.make_move(
            self.selected_position[0], self.selected_position[1], x, y
        )
        self.selected = False

    def draw_board(self) -> None:
        """Wrap common draw functions."""
        self.draw_grid()
        self.draw_selected()
        self.draw_pieces()

    def redraw_board(self):
        """Redraw game state."""
        self.canvas.delete("all")
        self.draw_board()
