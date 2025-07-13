"""Pytest suite for the Flow Free board logic."""

from __future__ import annotations

import numpy as np

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

INVALID_SINGLE_TERMINAL = np.array(
    [
        [terminal(3), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
        [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        [terminal(2), body(2), body(2), body(2), terminal(2)],
    ]
)

INVALID_MISSING_HEAD = np.array(
    [
        [terminal(1), body(1), EMPTY_CODE, EMPTY_CODE, terminal(1)],
        [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        [terminal(2), body(2), body(2), body(2), terminal(2)],
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


def test_game():
    """Test the flow free game logic."""
    ff = FlowFree.from_board(GAME_1)
    print(ff.board_str())

    solver = FlowFreeSATSolver(ff)
    assert solver.is_solvable() is False and brute_force_solve(ff) is None


def test_solvable_game():
    """Test solving a solvable game."""
    ff = FlowFree.from_board(VALID_SOLVABLE_EASY)
    solved_game = brute_force_solve(ff)
    assert solved_game is not None

    print(solved_game.board_str())

    solver = FlowFreeSATSolver(ff)
    solved_game = solver.solve(strict=False)
    assert solved_game.is_solved()
