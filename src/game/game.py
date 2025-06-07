from __future__ import annotations
from typing import TYPE_CHECKING

from game.utils import Constants, Rules, Translator

if TYPE_CHECKING:
    from typing import Callable

from game.state import State
from collections import deque
import torch
import logging

logger: logging.Logger = logging.getLogger(__name__)


class Game:
    """Game/States/Rules."""

    HIST_LEN = Constants.HISTORY_LENGTH

    def __init__(self, state: State, last_moves: deque[torch.Tensor]) -> None:
        """Generate Game instance.

        Should not be used directly! Use factory Methods instead.
        """
        logger.debug("Generating Game.")
        self.state: State = state
        self.last_moves: deque[torch.Tensor] = deque(
            (last_move.detach().clone() for last_move in last_moves),
            maxlen=self.HIST_LEN,
        )

    @classmethod
    def new_game(cls) -> Game:
        """Generate Initial game state."""
        state = State.initial_state()
        last_moves = deque(maxlen=cls.HIST_LEN)
        [last_moves.appendleft(state.state) for _ in range(cls.HIST_LEN)]
        return cls(state, last_moves)

    @classmethod
    def from_game(cls, game: Game) -> Game:
        """Copy from RecordGame."""
        return cls(game.state.copy(), game.last_moves)

    def copy(self) -> Game:
        """Copy operation."""
        return self.from_game(self)

    def save(self, path: str) -> None:
        """Save game state to torch tensor file."""
        self.state.save(path)

    def load(self, path: str) -> None:
        """Load game state from torch tensor file."""
        self.state = State.from_file(path)

    def reset(self) -> None:
        """Reset game state."""
        self.state = State.initial_state()
        [self.last_moves.appendleft(self.state.state) for _ in range(self.HIST_LEN)]

    # State
    def make_move(
        self,
        x1: int | torch.Tensor,
        y1: int | torch.Tensor,
        x2: int | torch.Tensor,
        y2: int | torch.Tensor,
        flag: int | torch.Tensor,
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

    # State
    def get_piece_targets(self, x: int, y: int) -> torch.Tensor:
        """Get all move target squares of a single piece."""
        layers: torch.Tensor = self.state.get_layer(x, y)
        if layers.nelement() != 0:
            piece: str = Translator.piece_from_layer(layers)
            move_generator: Callable = Rules.moves_from_piece(piece)
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
        return torch.full(Constants.move_tensor_shape, value, dtype=torch.float32)

    # State
    def move_mask(self) -> torch.Tensor:
        """Get mask for viable moves in move tensor."""
        own_positions = torch.where(self.state.get_player_mask(self.state.player))
        mask = self.full_move_tensor(False).to(dtype=torch.bool)
        for x, y in zip(*own_positions, strict=True):
            move_x, move_y = self.get_piece_targets(x, y).T
            mask[x, y, Translator.index_from_move(move_x, move_y)] = True
        print("MASKMASK", mask.dtype)
        return mask

    # State
    def all_moves(self) -> tuple[torch.Tensor, ...]:
        """Get list of all moves."""
        return torch.where(self.move_mask())

    def game_over(self) -> bool:
        """Check game status."""
        return self.state.game_over()

    # State
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

    def move_from_index(
        self, index: int | torch.Tensor
    ) -> tuple[int | torch.Tensor, int | torch.Tensor, int | torch.Tensor]:
        """Pass through translator."""
        return Translator.move_from_index(index)
