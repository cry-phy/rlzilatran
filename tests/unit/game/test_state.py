from game.state import State
import torch


def test_state_load():
    """Test loading of state."""
    state = State.initial_state()
    state.load("tests/fixtures/init_board.pt")
    reference = torch.load("tests/fixtures/init_board.pt")
    assert torch.allclose(state.get_state(), reference)


def test_state_saving():
    """Test state saving."""
    reference = torch.load("tests/fixtures/init_board.pt")
    state = State.from_tensor(reference)
    state.save("temp/state_saving_test.pt")
    state_saved = torch.load("tests/fixtures/init_board.pt")
    assert torch.allclose(state.get_state(), state_saved)
