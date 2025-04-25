from collections import deque
from game.state import State
from game.utils import Board
from typing import Callable
import torch
import os


class Game:
    """Game/States/Rules."""

    HIST_LEN = 8

    def __init__(self, state: State, last_moves: deque[torch.IntTensor]) -> None:
        """Generate Game instance.

        Should not be used directly! Use factory Methods instead.
        """
        self.state: State = state
        self.last_moves: deque[torch.IntTensor] = last_moves

    @classmethod
    def new_game(cls) -> "Game":
        """Generate Initial game state."""
        last_moves = deque(maxlen=cls.HIST_LEN)
        last_moves.appendleft(Board.empty_board())
        return cls(State.initial_state(), last_moves)

    @classmethod
    def from_record_game(cls, record_game: "RecordGame") -> "Game":
        """Copy from RecordGame."""
        return cls(record_game.state, record_game.last_moves)

    def reset(self) -> None:
        """Reset game state."""
        self.state = State.initial_state()

    def save(self, path: str) -> None:
        """Save game state to torch tensor file."""
        self.state.save(path)

    def load(self, path: str) -> None:
        """Load game state from torch tensor file."""
        self.state.load(path)

    def make_move(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Make move."""
        self.last_moves.appendleft(self.state.get_state())
        self.state.make_move(x1, y1, x2, y2)

    def get_piece_targets(self, x: int, y: int) -> torch.Tensor:
        """Get all move target squares of a single piece."""
        layers: torch.IntTensor = self.state.get_layer(x, y)
        if layers.nelement() != 0:
            piece: str = Board.piece_from_layer(layers)
            move_generator: Callable = Board.moves_from_piece(piece)
            return move_generator(x, y, self.state)


class RecordGame(Game):
    """Game with recording of past moves into files."""

    def __init__(
        self, state: State, last_moves: deque[torch.Tensor], folder_name: str
    ) -> None:
        """Generate RecordGame instance.

        Should not be used directly! Use factory Methods instead.
        """
        super().__init__(state, last_moves)
        self.folder_name: str = folder_name
        self.recent_file_names: deque[str] = deque(maxlen=self.HIST_LEN)

    def _prep(self):
        """Prepare recording Folder."""
        for tensor in self.last_moves:
            self.recent_file_names.appendleft(f"{hex(hash(tensor))}.pt")
        if not self.folder_name.endswith(os.path.sep):
            self.folder_name += os.path.sep
        os.makedirs(self.folder_name, exist_ok=False)
        with open(self.folder_name + "moves.csv") as f:
            f.write("STARTBOARD,ENDBOARD,MOVEX1,MOVEY1,MOVEX2,MOVEY2,VALIDITY\n")

    @classmethod
    def copy_game(cls, game: Game, folder_name: str) -> "RecordGame":
        """Copy content from other game."""
        return cls(game.state, game.last_moves, folder_name)

    @classmethod
    def new_game(cls, folder_name: str) -> "RecordGame":
        """Generate new empty game."""
        last_moves = deque(maxlen=cls.HIST_LEN)
        last_moves.appendleft(Board.empty_board())
        return cls(Board.empty_board(), last_moves, folder_name)

    def record(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Record made move."""
        with open(self.folder_name + "moves.csv") as f:
            f.write(
                f"{self.recent_file_names[0]},{self.recent_file_names[0]},{x1},{y1},{x2},{y2}\n"
            )

    def make_move(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Make move while recording."""
        super().make_move(x1, y1, x2, y2)
        self.record(x1, y1, x2, y2)
