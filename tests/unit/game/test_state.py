from game.state import State
from game.utils.constants import Constants
from testlib import FIXTURES, TEST_CASES, IDS
import torch
import pytest

SEED = 0


def test_state_load():
    """Test loading of state."""
    state = State.from_file(FIXTURES + "boards/init_board.json")
    reference = torch.load(FIXTURES + "boards/init_board.pt").to(dtype=torch.int32)
    assert torch.allclose(state.state, reference)


def test_state_saving():
    """Test state saving."""
    reference = torch.load(FIXTURES + "boards/init_board.pt").to(dtype=torch.int32)
    state = State.from_tensor(reference)
    state.save("temp/test_state_saving.json")
    state_saved = State.from_file("temp/test_state_saving.json")
    assert torch.allclose(state.state, state_saved.state)


def test_state_copy():
    """Test if copy from another state keeps information."""
    state1 = State.from_file(FIXTURES + "boards/init_board.json")
    state2 = State.from_state(state1)
    assert state1 == state2
    state3 = state1.copy()
    assert state1 == state3


def test_state_initial_state():
    """Test if initial state is set correctly."""
    state1 = State.initial_state()
    state2 = State.from_file(FIXTURES + "boards/init_board.json")
    assert state1 == state2
    state3 = state1.copy()
    state3.reset()
    assert state1 == state3


def test_state_empty_board():
    """Test if Empty board is set correctly."""
    state = State.empty()
    assert torch.allclose(state.tensor_reference, Constants.empty_board())


def test_state_properties():
    """Test properties."""
    state = State.from_file(FIXTURES + "boards/init_board.json")
    torch.manual_seed(SEED)
    random_state = torch.randint(0, 2, Constants.empty_board().shape)
    random_values = torch.randint(0, 10, (3, 1))
    state.state = random_state
    state.player = True
    state.captured_value = random_values[0]
    state.since_capture = random_values[1]
    state.since_capture_call = random_values[2]


def test_state_move_piece():
    """Test getting available moves."""


def test_state_player_mask():
    """Test if mask works."""
    false_mask, true_mask = torch.load(FIXTURES + "game_fixtures/player_masks.pt")
    state = State.initial_state()
    assert torch.allclose(state.get_player_mask(False), false_mask)
    assert torch.allclose(state.get_player_mask(True), true_mask)


@pytest.mark.parametrize(
    "name,start_board,end_board,x1,y1,x2,y2,validity", TEST_CASES, ids=IDS
)
def test_move_validity(
    name: str,
    start_board: State,
    end_board: State,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    validity: bool,
):
    """Run batch tests for move in states."""
    start_board.make_move(x1, y1, x2, y2)
    assert torch.allclose(start_board.state, end_board.state) is validity
