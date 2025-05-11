from typing import Callable
from agents.base import BaseAgent
from game.game import Game
from game.state import State
from game.utils import Board, MoveFlag
from mpi.handler import gracefull_error
from observer.base import BaseObserver
from agents.human import HumanAgent, Tool
import tkinter as tk
from tkinter import Event, Menu, filedialog, messagebox
import torch
import logging

logger: logging.Logger = logging.getLogger(__name__)


class UIObserver(BaseObserver):
    """Game UI for debugging and later play."""

    @classmethod
    def any_vs_any(
        cls,
        game: Game,
        agent_cls1: type[BaseAgent],
        agent_cls2: type[BaseAgent],
    ) -> "UIObserver":
        """Pit two agents against each other."""
        agent1 = agent_cls1(game)
        agent2 = agent_cls2(game)
        agent1.player = True
        agent2.player = agent1.player
        return cls(game, (agent1, agent2))

    def __init__(self, game: Game, agents: tuple[BaseAgent, BaseAgent]) -> None:
        """Initialize window."""
        super().__init__(game, agents)
        self.root = tk.Tk()
        self.root.title("Katruji / Zilatran Chess")

        self.selected_positions: list[tuple[int, int]] = []
        self.possible_moves: torch.Tensor = torch.empty(0, 2)

        # --- Menu Bar ---
        self.menubar = Menu(self.root)
        self._add_save_menu()
        self._add_tool_menu()
        self.root.config(menu=self.menubar)

        # --- Main Frame ---
        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=0)

        # --- Canvas for Board (in column 0) ---
        self.canvas = tk.Canvas(self.main_frame, width=400, height=800, bg="white")
        self.canvas.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.side_panel = tk.Frame(self.main_frame)
        self.side_panel.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        self.side_panel.grid_rowconfigure(0, weight=1)
        self.side_panel.grid_rowconfigure(1, weight=0)
        self.side_panel.grid_rowconfigure(2, weight=1)
        self.side_panel.grid_columnconfigure(0, weight=1)

        # --- Canvas for Board ---
        self.COLUMNS, self.ROWS = Board.board_size
        self.CELL_SIZE: int = 80
        self.FONT_SIZE: int = 15
        self.SMALL_FONT: int = 10
        self.initial_resize_done = False
        self.WAIT_TIME = 100

        self._side_panel()

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
        self.root.after(0, lambda: self._on_resize(self.canvas))

        # -- Add Binds --
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-2>", lambda _: self.current_agent()._clear_tool)
        self.canvas.bind("<Button-3>", lambda _: self.current_agent()._clear_tool)
        self.root.bind_all("<Control-s>", lambda _: self.on_save())
        self.root.bind_all("<Control-o>", lambda _: self.on_load())
        self.root.bind_all("<Control-n>", lambda _: self.new_empty())

        self.root.protocol("WM_DELETE_WINDOW", self.exit_mpi)

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
        filemenu.add_command(label="Quit", command=self.exit_mpi)
        self.menubar.add_cascade(label="File", menu=filemenu)

    def _side_panel(self) -> None:
        """Set up side panel."""
        self.capture_var = tk.StringVar(value="Game not started.")
        self.capture_label = tk.Label(
            self.side_panel,
            textvariable=self.capture_var,
            justify="center",
            font=(f"Helvetica {self.FONT_SIZE}"),
        )
        self.capture_label.grid(row=1, column=0, pady=(0, 1))

        # --- Centered Button ---
        self.extra_button = tk.Button(
            self.side_panel,
            text="Capture Call",
            command=lambda: self.set_move_flag(MoveFlag.CAPTURE_CALL),
        )
        self.extra_button.grid(row=2, column=0)

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
                variable=self.current_agent().selected_var,
                value=tool.value,
                command=lambda t=tool: self.current_agent()._select_tool(t),
            )
        toolmenu.add_separator()
        toolmenu.add_radiobutton(
            label="MOVE",
            variable=self.current_agent().selected_var,
            value=Tool.MOVE.value,
            command=lambda: self.current_agent()._select_tool(Tool.MOVE),
        )
        toolmenu.add_radiobutton(
            label="FORCE",
            variable=self.current_agent().selected_var,
            value=Tool.FORCE_SELECT.value,
            command=lambda: self.current_agent()._select_tool(Tool.FORCE_SELECT),
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
        if isinstance(self.agents[0], HumanAgent):
            self.agents[0]._CELL_SIZE = self.CELL_SIZE
        if isinstance(self.agents[1], HumanAgent):
            self.agents[1]._CELL_SIZE = self.CELL_SIZE
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
            positions: tuple[torch.Tensor, ...] = torch.where(piece_layer)
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
        print(self.selected_positions)
        if self.selected_positions:
            for position in self.selected_positions:
                self.canvas.create_rectangle(
                    *self._get_cell_boundaries(*position),
                    outline="black",
                    fill="lightblue",
                )
                moves = self.current_agent().possible_moves(*position)
                for move in moves:
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

    def exit_mpi(self):
        """Force quitting all Ranks."""
        logger.info("Quitting GUI.")
        self.root.destroy()
        self.mpi.force_exit()

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
        agent = self.current_agent()
        if agent.clickable:
            logger.debug("Handing Event to clickable %s.", type(agent))
            agent.handle_click(event)
        self.selected_positions = agent.selected_positions()
        self.redraw_board()

    def update_values(self) -> None:
        """Update common values."""
        self.capture_var.set(
            f"{self.game.state.captured_value[0]} {self.game.state.captured_value[0]}"
        )

    def draw_board(self) -> None:
        """Wrap common draw functions."""
        self.draw_grid()
        self.draw_selected()
        self.draw_pieces()
        self.draw_grid_coords()

    def redraw_board(self) -> None:
        """Redraw game state."""
        self.update_values()
        self.canvas.delete("all")
        self.draw_board()

    def set_move_flag(self, flag: MoveFlag) -> None:
        """Set a certain game flag for the current agent."""
        agent = self.current_agent()
        agent.move_flag = flag

    def pre_clickable_input(self):
        """Let human select move."""
        self.canvas.bind("<Button-1>", self.on_click)

    def post_clickable_input(self):
        """Remove interaction."""
        self.canvas.unbind("<Button-1>")

    def wait(self, func: Callable) -> None:
        """Wait for refresh."""
        self.root.after(self.WAIT_TIME, gracefull_error(func))
