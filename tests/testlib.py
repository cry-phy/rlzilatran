from game.state import State
import pandas as pd

FIXTURES: str = "tests/fixtures/"


def state_move_validity(
    csv_path: str,
) -> list[tuple[str, State, State, int, int, int, int, bool]]:
    """Make list of states moves and validity available to tests."""
    tests = pd.read_csv(csv_path)
    boards_path = FIXTURES + "boards/"
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
    x0s: list[int] = tests.MOVEX1.tolist()
    y0s: list[int] = tests.MOVEY1.tolist()
    x1s: list[int] = tests.MOVEX2.tolist()
    y1s: list[int] = tests.MOVEY2.tolist()
    validity: list[bool] = tests.VALIDITY.astype(bool).tolist()
    return list(
        zip(names, start_board, end_board, x0s, y0s, x1s, y1s, validity, strict=True)
    )


TEST_CASES: list[tuple[str, State, State, int, int, int, int, bool]] = (
    state_move_validity(FIXTURES + "game_fixtures/p_state_move_validity.csv")
    + state_move_validity(FIXTURES + "game_fixtures/s_state_move_validity.csv")
    + state_move_validity(FIXTURES + "game_fixtures/o_state_move_validity.csv")
)

IDS: list[str] = [test_case[0] for test_case in TEST_CASES]
