from game.utils import Board
from testlib import FIXTURES
import torch
import numpy as np


def test_empty_board_size():
    """Test whether board is correct sizes."""
    size = Board.board_size
    layers = Board.num_layers
    board = Board.empty_board()
    assert (layers, *size) == board.size()


def test_init_board_size():
    """Test whether oard is correct sizes."""
    size = Board.board_size
    layers = Board.num_layers
    board = Board.initial_board()
    assert (layers, *size) == board.size()


def test_init_board():
    """Test whether initial board is set correctly."""
    assert (
        torch.allclose(
            torch.load(FIXTURES + "boards/init_board.pt").to(torch.int32),
            Board.initial_board(),
        )
        is True
    )


def test_indexing():
    """Test indexing from move to move_tensor."""
    x, y = np.meshgrid(np.arange(4), np.arange(8))
    indices = Board.index_from_move(x, y)
    x2, y2 = Board.move_from_index(indices)
    assert np.allclose(x, x2)
    assert np.allclose(y, y2)
