from game.utils import board_generator


def test_empty_board_size():
    """Test whether oard is correct sizes."""
    size = board_generator.board_size()
    layers = board_generator.num_layers()
    board = board_generator.empty_board()
    assert (layers, *size) == board.size()


def test_init_board_size():
    """Test whether oard is correct sizes."""
    size = board_generator.board_size()
    layers = board_generator.num_layers()
    board = board_generator.initial_board()
    assert (layers, *size) == board.size()
