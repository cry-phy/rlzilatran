from copy import deepcopy

from game.state import State
from game.utils import Board
from typing import Callable
from collections import deque
import torch
import os
import logging

logger: logging.Logger = logging.getLogger(__name__)


class Game:
    """Game/States/Rules."""

    HIST_LEN = Board.HISTORY_LENGTH

    def __init__(self, state: State, last_moves: deque[torch.Tensor]) -> None:
        """Generate Game instance.

        Should not be used directly! Use factory Methods instead.
        """
        logger.info("Generating Game.")
        self.state: State = state
        self.last_moves: deque[torch.Tensor] = last_moves

    @classmethod
    def new_game(cls) -> "Game":
        """Generate Initial game state."""
        last_moves = deque(maxlen=cls.HIST_LEN)
        [last_moves.appendleft(Board.empty_board()) for _ in range(cls.HIST_LEN)]
        return cls(State.initial_state(), last_moves)

    @classmethod
    def from_game(cls, game: "Game") -> "Game":
        """Copy from RecordGame."""
        return cls(game.state.copy(), deepcopy(game.last_moves))

    @classmethod
    def from_record_game(cls, record_game: "RecordGame") -> "Game":
        """Copy from RecordGame."""
        return cls(record_game.state.copy(), deepcopy(record_game.last_moves))

    def copy(self) -> "Game":
        """Copy operation."""
        return self.from_game(self)

    def reset(self) -> None:
        """Reset game state."""
        self.state = State.initial_state()

    def save(self, path: str) -> None:
        """Save game state to torch tensor file."""
        self.state.save(path)

    def load(self, path: str) -> None:
        """Load game state from torch tensor file."""
        self.state = State.from_file(path)

    def make_move(
        self,
        x1: int | torch.Tensor,
        y1: int | torch.Tensor,
        x2: int | torch.Tensor,
        y2: int | torch.Tensor,
        flag: int,
    ) -> None:
        """Make move."""
        logger.debug(
            "Making move: (%i, %i), (%i, %i), flag: %i",
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            int(flag),
        )
        self.last_moves.appendleft(self.state.state)
        self.state.make_move(x1, y1, x2, y2, flag)

    def get_piece_targets(self, x: int, y: int) -> torch.Tensor:
        """Get all move target squares of a single piece."""
        layers: torch.Tensor = self.state.get_layer(x, y)
        if layers.nelement() != 0:
            piece: str = Board.piece_from_layer(layers)
            move_generator: Callable = Board.moves_from_piece(piece)
            return move_generator(x, y, self.state)
        else:
            return torch.empty((0, 2), dtype=torch.int)

    @property
    def state_tensor(self) -> torch.Tensor:
        """Stack the history and add additional information."""
        return torch.cat(
            [*self.last_moves, *self.state.get_additional_channels()], dim=0
        )

    def full_move_tensor(self, value: float) -> torch.Tensor:
        """Get a move tensor with all moves equally likely."""
        return torch.full(Board.move_tensor_shape, value)

    def move_mask(self) -> torch.Tensor:
        """Get mask for viable moves in move tensor."""
        own_positions = torch.where(self.state.get_player_mask(self.state.player))
        mask = torch.full(Board.move_tensor_shape, False)
        for x, y in zip(*own_positions, strict=True):
            move_x, move_y = self.get_piece_targets(x, y).T
            mask[x, y, Board.index_from_move(move_x, move_y)] = True
        return mask

    def all_moves(self) -> tuple[torch.Tensor, ...]:
        """Get list of all moves."""
        return torch.where(self.move_mask())

    def game_over(self) -> bool:
        """Check game status."""
        return self.state.game_over()

    def winner(self) -> bool | None:
        """Game winner."""
        if (
            self.state.captured_value[int(False)]
            == self.state.captured_value[int(True)]
        ):
            return None
        else:
            return (
                self.state.captured_value[int(False)]
                <= self.state.captured_value[int(True)]
            )


class RecordGame(Game):
    """Game with recording of past moves into files."""

    def __init__(
        self, state: State, last_moves: deque[torch.Tensor], folder_name: str
    ) -> None:
        """Generate RecordGame instance.

        Should not be used directly! Use factory Methods instead.
        """
        super().__init__(state, last_moves)
        logger.info("Recording.")
        self.folder_name: str = folder_name
        self.recent_file_names: deque[str] = deque(maxlen=self.HIST_LEN)

    def _prep(self):
        """Prepare recording Folder."""
        for _ in range(self.HIST_LEN):
            self.recent_file_names.appendleft(self.get_state_name())
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

    def get_state_name(self) -> str:
        """Hash the state to a file name."""
        return f"{hex(hash(self.state.tensor_reference))}.pt"

    def record(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Record made move."""
        self.recent_file_names.appendleft(self.get_state_name())
        self.state.save(self.recent_file_names[0])
        with open(self.folder_name + "moves.csv") as f:
            f.write(
                f"{self.recent_file_names[0]},{self.recent_file_names[1]},{x1},{y1},{x2},{y2}\n"
            )

    def make_move(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Make move while recording."""
        super().make_move(x1, y1, x2, y2)
        self.record(x1, y1, x2, y2)
