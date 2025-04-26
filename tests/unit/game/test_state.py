from game.state import State
import torch
import pytest
import pandas as pd


def test_state_load():
    """Test loading of state."""
    state = State.from_file("tests/fixtures/boards/init_board.json")
    reference = torch.load("tests/fixtures/boards/init_board.pt").to(dtype=torch.int32)
    assert torch.allclose(state.state, reference)


def test_state_saving():
    """Test state saving."""
    reference = torch.load("tests/fixtures/boards/init_board.pt").to(dtype=torch.int32)
    state = State.from_tensor(reference)
    state.save("temp/test_state_saving.json")
    state_saved = State.from_file("temp/test_state_saving.json")
    assert torch.allclose(state.state, state_saved.state)


def state_move_validity(
    csv_path: str,
) -> list[tuple[str, State, State, int, int, int, int, bool]]:
    """Make list of states moves and validity available to tests."""
    tests = pd.read_csv(csv_path)
    boards_path = "tests/fixtures/boards/"
    names: list[str] = [
        f"{tests.STARTBOARD[i]}-{tests.ENDBOARD[i]}-{bool(tests.VALIDITY[i])}"
        for i in tests.index
    ]
    start_board: list[State] = [
        State.from_file(boards_path + file) for file in tests.STARTBOARD
    ]
    end_board: list[State] = [
        State.from_file(boards_path + file) for file in tests.ENDBOARD
    ]
    x1s: list[int] = tests.MOVEX1.tolist()
    y1s: list[int] = tests.MOVEY1.tolist()
    x2s: list[int] = tests.MOVEX2.tolist()
    y2s: list[int] = tests.MOVEY2.tolist()
    validity: list[bool] = tests.VALIDITY.astype(bool).tolist()
    return list(
        zip(names, start_board, end_board, x1s, y1s, x2s, y2s, validity, strict=True)
    )


TEST_CASES: list[tuple[str, State, State, int, int, int, int, bool]] = (
    state_move_validity("tests/fixtures/game_fixtures/p_state_move_validity.csv")
    + state_move_validity("tests/fixtures/game_fixtures/s_state_move_validity.csv")
    + state_move_validity("tests/fixtures/game_fixtures/o_state_move_validity.csv")
)

IDS: list[str] = [test_case[0] for test_case in TEST_CASES]


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
