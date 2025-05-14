from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from game.utils.constants import Constants
from enum import IntEnum, auto
import torch
import logging

logger: logging.Logger = logging.getLogger(__name__)


class MoveFlag(IntEnum):
    """Encodes actions outside the board to flags for moves."""

    STANDARD = auto()
    CAPTURE_CALL = auto()

    # The following are only for human players for now as
    # opposed to the computer they have limited time.
    RESIGN = auto()
    DRAW_OFFER = auto()


class __Translator:
    """Holds functions able to translate between different representations/data for the game.

    This is a module level singleton, i. e. being loaded from config at import time.
    """

    def index_from_move(self, x, y) -> int:
        return y * Constants.x + x

    def move_from_index(
        self, index: int | torch.Tensor
    ) -> tuple[int | torch.Tensor, int | torch.Tensor, int | torch.Tensor]:
        x = index % Constants.x
        y = index // Constants.x
        if torch.is_tensor(index):
            flag = torch.where(
                index == 33,
                torch.tensor(int(MoveFlag.CAPTURE_CALL)),
                torch.tensor(int(MoveFlag.STANDARD)),
            )
        else:
            flag = (
                int(MoveFlag.STANDARD)
                if not index == 33
                else int(MoveFlag.CAPTURE_CALL)
            )
        return x, y, flag

    def player_half(self, y: int, player: bool) -> bool:
        """Return if y is on the player side of the board."""
        return (
            int(player) * Constants.y // 2 <= y
            and y < (1 + int(player)) * Constants.y // 2
        )

    def half_indices(self, y: int | torch.Tensor) -> tuple[int, int]:
        """Get indices equating to player layers."""
        jbool = not (y <= Constants.num_pieces)
        j1 = int(jbool) * Constants.num_pieces
        j2 = (int(jbool) + 1) * Constants.num_pieces
        return j1, j2

    def layer_from_piece(self, piece: str) -> int:
        """Get layer from piece."""
        return Constants._piece_unique_dict[piece]

    def value_from_piece(self, piece: str) -> float:
        """Get value from piece."""
        return Constants._piece_unique_dict[piece]

    def unqiue_from_layer(self, layer: int | torch.Tensor) -> int | torch.Tensor:
        return layer % len(Constants._pieces)

    def value_from_layer(self, layer: int | torch.Tensor) -> torch.Tensor:
        return Constants._value_list[self.unqiue_from_layer(layer)]

    def piece_from_layer(self, layer: int | torch.Tensor) -> str:
        return Constants._pieces[layer % len(Constants._pieces)]


__all__ = ["Translator"]
Translator = __Translator()
