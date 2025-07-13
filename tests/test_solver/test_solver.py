"""Pytest suite for the Flow Free board logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flowzero_src.flowfree.game import EMPTY_CODE, Coordinate, FlowFree, body, terminal
from flowzero_src.flowfree.solver import FlowFreeSATSolver, brute_force_solve

# ────────────────────────────── Shared test data ────────────────────────────── #

TERMINALS_5x5 = {
    1: (Coordinate(0, 0), Coordinate(4, 4)),
    2: (Coordinate(0, 4), Coordinate(4, 0)),
}

VALID_SOLVABLE_EASY = np.array(
    [
        [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
        [terminal(3), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(3)],
    ]
)

INCOMPLETE_BUT_VALID = np.array(
    [
        [terminal(1), body(1), body(1), body(1), terminal(1)],
        [body(1), body(1), body(1), body(1), EMPTY_CODE],
    ]
)

COMPLETED_GAME_1 = np.array(
    [
        [terminal(1), body(1), body(1), body(1), terminal(1)],
        [terminal(2), body(2), body(2), body(2), terminal(2)],
        [terminal(3), body(3), body(3), body(3), terminal(3)],
    ]
)

# Same as completed_Game_1 but vertical
COMPLETED_GAME_2 = np.array(
    [
        [terminal(1), terminal(2), terminal(3)],
        [body(1), body(2), body(3)],
        [body(1), body(2), body(3)],
        [body(1), body(2), body(3)],
        [terminal(1), terminal(2), terminal(3)],
    ]
)

GAME_1 = np.array(
    [
        [terminal(1), EMPTY_CODE, terminal(2), EMPTY_CODE, EMPTY_CODE],
        [terminal(3), EMPTY_CODE, terminal(1), EMPTY_CODE, EMPTY_CODE],
        [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        [terminal(3), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(2)],
    ]
)

GAME_2 = np.array(
    [
        [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(2)],
        [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        [terminal(2), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
    ]
)


@pytest.mark.parametrize(
    "board, is_solvable",
    [
        (INCOMPLETE_BUT_VALID, False),
        (COMPLETED_GAME_1, True),
        (COMPLETED_GAME_2, True),
        (VALID_SOLVABLE_EASY, True),
        (GAME_1, False),
        (GAME_2, False),
    ],
)
def test_game_solvability(board: np.ndarray, is_solvable: bool) -> None:
    """Test if the game is solvable."""
    ff = FlowFree.from_board(board)
    solver = FlowFreeSATSolver(ff)
    assert solver.is_solvable() is is_solvable
    if is_solvable:
        assert brute_force_solve(ff, 10) is not None, (
            "Expected a solution, but brute_force_solve returned None"
        )
    else:
        assert brute_force_solve(ff, 10) is None, (
            "Expected None for unsolvable board, but brute_force_solve returned a solution"
        )


BASE = Path(__file__).resolve().parent.parent.parent

# Handcrafted boards were found at https://github.com/mzucker/flow_solver/tree/master/puzzles
HANDCRAFTED_BOARD_PATHS = BASE / "flowzero_src" / "data" / "handcrafted"
HANDCRAFTED_BOARDS = [file.resolve() for file in HANDCRAFTED_BOARD_PATHS.glob("*.txt")]


def path_to_ff(path: Path) -> FlowFree:
    """Convert an ASCII board from a path to the numpy array."""
    with open(path) as f:
        lines = f.readlines()
        lines = "".join(lines)
        # print(lines)
        ff = FlowFree.from_ascii_board(lines)
        return ff


@pytest.mark.parametrize(
    "ff, is_solvable, file_name",
    [(path_to_ff(p), True, p.name) for p in HANDCRAFTED_BOARDS],
)
def test_prebuilt_boards(ff: FlowFree, is_solvable: bool, file_name: str) -> None:
    """Test if the game is solvable."""
    assert ff.is_valid_board(ff._board), "Board must be valid"

    solver = FlowFreeSATSolver(ff)
    assert solver.is_solvable() is is_solvable, (
        f"{file_name} -- Handcrafted ascii boards should be solvable"
    )
