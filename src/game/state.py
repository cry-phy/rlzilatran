from copy import copy
import torch
import json
import logging

from game.utils import Constants, MoveFlag, Translator, Rules

logger: logging.Logger = logging.getLogger(__name__)


class State:
    """Wrapper around game state pytorch tensor.

    Contains:
        - Pieces State
        - Game End Logic
        - Reward
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        player: bool = True,
        captured_value: list[float] | None = None,
        since_capture: int = 0,
        since_capture_call: int = 0,
    ) -> None:
        """Generate State instance.

        Should not be used directly! Use factory Methods instead.
        """
        if captured_value is None:
            captured_value = [0, 0]

        self._tensor: torch.Tensor = tensor
        self._player: bool = player
        self._captured_value: list[float] = captured_value
        self._active_capture_call: bool = False
        self._since_capture: int = since_capture
        self._since_capture_call: int = since_capture_call

    @property
    def tensor_reference(self) -> torch.Tensor:
        """Provide access to base."""
        return self._tensor

    @property
    def player(self) -> bool:
        """Current player."""
        return self._player

    @property
    def captured_value(self) -> list[float]:
        """Value of captured pieces."""
        return self._captured_value

    @property
    def since_capture(self) -> int:
        """Moves since last capture."""
        return self._since_capture

    @property
    def since_capture_call(self) -> int:
        """Moves since last capture."""
        return self._since_capture_call

    @property
    def state(self) -> torch.Tensor:
        """Return state Tensor for training."""
        return self._tensor.clone()

    @player.setter
    def player(self, player: bool) -> None:
        """Set current player."""
        self._player = player

    @captured_value.setter
    def captured_value(self, captured_value: list[float]) -> None:
        """Value of captured pieces by player False."""
        self._captured_value = captured_value

    @since_capture.setter
    def since_capture(self, since_capture: int) -> None:
        """Set moves since last capture."""
        self._since_capture = since_capture

    @since_capture_call.setter
    def since_capture_call(self, since_capture_call: int) -> None:
        """Set moves since last capture."""
        self._since_capture_call = since_capture_call

    @state.setter
    def state(self, tensor: torch.Tensor) -> None:
        """Set state Tensor."""
        self._tensor = tensor

    @classmethod
    def initial_state(cls) -> "State":
        """Generate Initial game State."""
        return cls(Constants.initial_board())

    @classmethod
    def empty(cls) -> "State":
        """Generate empty state."""
        return cls(Constants.empty_board())

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "State":
        """Generate State from state Tensor."""
        return cls(tensor)

    @classmethod
    def from_state(cls, state: "State") -> "State":
        """Create copy from state for copy operations."""
        return cls(
            state.state.clone(),
            state.player,
            copy(state.captured_value),
            state.since_capture,
        )

    # --- Dunder Methods ---
    def __eq__(self, other: "State") -> bool:
        """Compare states to each other."""
        return (
            torch.allclose(self._tensor, other.tensor_reference)
            and self._player == other.player
            and self._captured_value == other.captured_value
            and self._since_capture == other.since_capture
            and self._since_capture_call == other.since_capture_call
        )

    def __repr__(self) -> str:
        """Repr method for debugging."""
        string = "State:"
        string += "\n\t"
        string += "player:              " + str(self._player)
        string += "\n\t"
        string += "captured_value:      " + str(self._captured_value)
        string += "\n\t"
        string += "since_capture:       " + str(self._since_capture)
        string += "\n\t"
        string += "since_capture_call:  " + str(self._since_capture_call)
        string += "\n\t"
        string += self._tensor.__repr__()
        return string

    def __str__(self) -> str:
        """Str method same as Repr for now."""
        return self.__repr__()

    # --- File IO ---
    def state_dict(self) -> dict:
        """Return a plain dict of only the data fields."""
        return {
            "player": self._player,
            "captured_value": self._captured_value,
            "since_capture": self._since_capture,
            "since_capture_call": self._since_capture_call,
        }

    def save(self, path: str) -> None:
        """Save only the data to a JSON file in human readable spacing."""
        data = self.state_dict().copy()

        logger.debug("Dumping data %s", data)
        meta_str = json.dumps(data, indent=2).rstrip()
        if meta_str.endswith("}"):
            meta_str = meta_str[:-1]

        logger.debug("Dumping tensor and board size.")
        with open(path, "w") as f:
            f.write(meta_str)
            # This incurs overhead but provides human readable data
            # Should not be called often anyways.
            f.write(',\n  "tensor": [\n')
            for s_idx, slice2d in enumerate(self._tensor.tolist()):
                f.write("    [\n")
                for r_idx, row in enumerate(slice2d):
                    f.write("      ")
                    json.dump(row, f, separators=(",", ":"))
                    f.write(",\n" if r_idx < len(slice2d) - 1 else "\n")
                f.write("    ]")
                f.write(",\n" if s_idx < len(self._tensor) - 1 else "\n")
            f.write("  ]\n")
            f.write("}\n")

    @classmethod
    def from_file(cls, path: str) -> "State":
        """Load data from JSON and rebuild a GameState instance."""
        with open(path) as f:
            data = json.load(f)
        tensor = torch.tensor(
            data["tensor"],
            dtype=torch.int32,
        )
        instance = cls(
            tensor=tensor,
            player=data["player"],
            captured_value=data["captured_value"],
            since_capture=data["since_capture"],
            since_capture_call=data["since_capture_call"],
        )
        return instance

    # --- Simple operations ---
    def copy(self) -> "State":
        """Generate copy of self."""
        return self.from_state(self)

    def reset(self) -> None:
        """Set to initial state."""
        self._tensor = Constants.initial_board()

    def square_filled(self, x: int, y: int) -> bool | torch.Tensor:
        """Return if th square has pieces on it."""
        return torch.any(self._tensor[:, x, y])

    def get_player_mask(self, player: bool) -> torch.Tensor:
        """Get mask of player."""
        return (
            self._tensor[
                int(player) * Constants.num_pieces : (int(player) + 1)
                * Constants.num_pieces,
                :,
                :,
            ]
            .sum(dim=0)
            .to(dtype=torch.bool)
        )

    def player_material(self, player: bool) -> float | torch.Tensor:
        """Get the material of player."""
        x, y = torch.where(self.get_player_mask(player))
        positions: torch.Tensor = torch.stack((x, y), dim=1).T
        layer, _ = torch.where(self.tensor_reference[:, positions[0], positions[1]])
        value: torch.Tensor = Translator.value_from_layer(layer)
        return value.sum()

    def check_piece_move(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> bool | torch.Tensor:
        """Check whether attempted move is in valid moves for piece."""
        target = torch.tensor([x2, y2])
        layers = torch.where(self._tensor[:, x1, y1])[0]
        piece = Translator.piece_from_layer(layers)
        moves = Rules.moves_from_piece(piece)(x1, y1, self)
        return torch.any(torch.all(moves == target, dim=1))

    def game_over(self) -> bool:
        """Check all termination status."""
        return (
            Rules.check_half_empty(self._player, self._tensor)
            or Rules.check_captured_value_termination(self._captured_value)
            or Rules.check_capture_call_termination(self._since_capture_call)
            or Rules.check_capture_termination(self._since_capture)
        )

    # --- Board Manipulations ---
    def make_move(
        self,
        x1: int | torch.Tensor,
        y1: int | torch.Tensor,
        x2: int | torch.Tensor,
        y2: int | torch.Tensor,
        flag: int | torch.Tensor,
    ) -> None:
        """Make move corresponding to index."""
        match int(flag) + 1:
            case int(MoveFlag.STANDARD):
                self.standard_move(x1, y1, x2, y2)
            case int(MoveFlag.CAPTURE_CALL):
                self.make_capture_call()
            case int(MoveFlag.RESIGN):
                self.resign()
            case int(MoveFlag.DRAW_OFFER):
                self.offer_draw()
            case _:
                raise RuntimeError(f"Flag {flag} does not exist.")
        self._player = not self._player

    def standard_move(
        self,
        x1: int | torch.Tensor,
        y1: int | torch.Tensor,
        x2: int | torch.Tensor,
        y2: int | torch.Tensor,
    ) -> None:
        """Move a piece to another square."""
        # Capture logic handling.
        capture_layer: torch.Tensor = torch.where(self._tensor[:, x2, y2])[0]
        if capture_layer.nelement() != 0:
            capture_value = Translator.value_from_layer(capture_layer)
            logger.debug(
                "Capturing %s with value %i",
                Translator.piece_from_layer(capture_layer),
                Translator.value_from_layer(capture_layer),
            )
            self._captured_value[int(self._player)] += float(capture_value)
            self._active_capture_call = False
            self._since_capture_call = 0
        else:
            self._since_capture += 1
            if self._active_capture_call:
                self.since_capture_call += 1

        # Moving pieces
        i1, i2 = Translator.half_indices(y1)
        j1, j2 = Translator.half_indices(y2)
        logger.debug(
            "Moving %s to %s",
            str(self._tensor[i1:i2, x1, y1]),
            str(self._tensor[j1:j2, x2, y2]),
        )
        self._tensor[j1:j2, x2, y2] = self._tensor[i1:i2, x1, y1]
        self._tensor[i1:i2, x1, y1] = 0

    def make_capture_call(self) -> None:
        """Set flags for capture call."""
        self._active_capture_call = True

    def resign(self) -> None:
        """Resign, ending game immidatly."""
        raise NotImplementedError

    def offer_draw(self) -> None:
        """Set/resolve draw offer flag."""
        raise NotImplementedError

    def add_piece(self, x: int, y: int, piece: str) -> None:
        """Add Piece to certain layer."""
        layer: int = Translator.layer_from_piece(piece)
        i1, _ = Translator.half_indices(y)
        self._tensor[i1 + layer, x, y] = 1

    def remove_piece(self, x: int, y: int, piece: str) -> None:
        """Remove certain piece."""
        layer = Translator.layer_from_piece(piece)
        self._tensor[layer, x, y] = 0
        self._tensor[layer + Constants.num_pieces, x, y] = 0

    def clear_square(self, x: int, y: int) -> None:
        """Clear square entirely."""
        self._tensor[:, x, y] = 0

    def get_layer(self, x: int, y: int):
        """Get layer of pieces on x, y."""
        return torch.where(self._tensor[:, x, y])[0]

    # --- Game Rules ---
    def check_move(self, x1: int, y1: int, x2: int, y2: int) -> bool | torch.Tensor:
        """Check whether move is valid."""
        return (
            Rules.valid_position(x1, y1)
            and Rules.valid_position(x2, y2)
            and Translator.player_half(y1, self._player)
            and self.check_piece_move(x1, y1, x2, y2)
        )

    # -- Retrieve Channels --
    def player_channel(self) -> torch.Tensor:
        """Generate entire layer of player values."""
        return torch.full(Constants.board_size, int(self._player))[None, :, :]

    def captured_value_channel(self) -> torch.Tensor:
        """Generate entire layer of player values."""
        layer: torch.Tensor = torch.empty(Constants.board_size)
        layer[:, : Constants.y // 2] = self._captured_value[0]
        layer[:, Constants.y // 2 :] = self._captured_value[1]
        return layer[None, :, :]

    def since_capture_channel(self) -> torch.Tensor:
        """Generate entire layer of player values."""
        return torch.full(Constants.board_size, int(self._since_capture))[None, :, :]

    def since_capture_call_channel(self) -> torch.Tensor:
        """Generate entire layer of player values."""
        return torch.full(Constants.board_size, int(self._since_capture_call))[
            None, :, :
        ]

    def get_additional_channels(self) -> list[torch.Tensor]:
        """Get list of additional channels."""
        return [
            self.player_channel(),
            self.captured_value_channel(),
            self.since_capture_channel(),
            self.since_capture_call_channel(),
        ]

    def check_flag(self, flag: MoveFlag) -> bool:
        """Check move flag for applicability."""
        match flag:
            case MoveFlag.CAPTURE_CALL:
                return self.since_capture >= 7
            case _:
                return True
