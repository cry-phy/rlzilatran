from game.utils.constants import Constants
from game.utils.translator import Translator
from testlib import FIXTURES
import torch


def test_empty_board_size():
    """Test whether board is correct sizes."""
    size = Constants.board_size
    layers = Constants.num_layers
    board = Constants.empty_board()
    assert (layers, *size) == board.size()


def test_init_board_size():
    """Test whether oard is correct sizes."""
    size = Constants.board_size
    layers = Constants.num_layers
    board = Constants.initial_board()
    assert (layers, *size) == board.size()


def test_init_board():
    """Test whether initial board is set correctly."""
    assert (
        torch.allclose(
            torch.load(FIXTURES + "boards/init_board.pt").to(torch.int32),
            Constants.initial_board(),
        )
        is True
    )


def test_indexing():
    """Test indexing from move to move_tensor."""
    x, y = torch.meshgrid(torch.arange(4), torch.arange(8))
    indices = Translator.index_from_move(x, y)
    x2, y2, flag = Translator.move_from_index(indices)
    assert torch.allclose(x, x2)
    assert torch.allclose(y, y2)
