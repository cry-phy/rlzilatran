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

        # --- Canvas for Board ---
        canvas_width = self.COLUMNS * self.CELL_SIZE
        canvas_height = self.ROWS * self.CELL_SIZE
        self.canvas = tk.Canvas(
            root, width=canvas_width, height=canvas_height, bg="white"
        )
        self.canvas.pack(padx=10, pady=10)

        self.draw_grid()
        self.draw_pieces()

        self.canvas.bind("<Button-1>", self.on_click)

    def draw_grid(self):
        """Draw grid on which the Game is rendered."""
        for row in range(self.ROWS):
            for col in range(self.COLUMNS):
                x1 = col * self.CELL_SIZE
                y1 = row * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="black", fill="lightgray"
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

    def on_save(self):
        """Save current game as pickle.

        TODO!
        """
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
        col = event.x // self.CELL_SIZE
        row = event.y // self.CELL_SIZE
        print(f"Clicked on row {row}, col {col}")
        self.redraw_board()

    def redraw_board(self):
        """Redraw game state."""
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_pieces()
