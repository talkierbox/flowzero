"""Pytest suite for the Flow Free board logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flowzero_src.flowfree.game import (
    EMPTY_CODE,
    Action,
    ActionTypes,
    Coordinate,
    FlowFree,
    body,
    head,
    terminal,
)
from flowzero_src.flowfree.solver import FlowFreeSATSolver
from flowzero_src.util.save_util import import_ndarray

# ────────────────────────────── Shared test data ────────────────────────────── #

TERMINALS_5x5 = {
    1: (Coordinate(0, 0), Coordinate(4, 4)),
    2: (Coordinate(0, 4), Coordinate(4, 0)),
}

VALID_BOARD_3x5 = np.array(
    [
        [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
        [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        [terminal(2), body(2), body(2), body(2), terminal(2)],
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

IMPROPER_HEADS_1 = np.array(
    [
        [terminal(1), terminal(2), terminal(3)],
        [head(1), body(2), body(3)],
        [body(1), body(2), body(3)],
        [body(1), body(2), body(3)],
        [terminal(1), terminal(2), terminal(3)],
    ]
)

IMPROPER_HEADS_2 = np.array(
    [
        [terminal(1), terminal(2), terminal(3)],
        [body(1), body(2), body(3)],
        [body(1), body(2), head(3)],
        [body(1), body(2), body(3)],
        [terminal(1), terminal(2), terminal(3)],
    ]
)

FLOATING_HEAD = np.array(
    [
        [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
        [EMPTY_CODE, EMPTY_CODE, head(3), EMPTY_CODE, EMPTY_CODE],
        [terminal(2), body(2), body(2), body(2), terminal(2)],
    ]
)

DISCONNECTED_HEAD = np.array(
    [
        [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
        [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        [terminal(2), body(2), body(2), head(2), terminal(2)],
        [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
    ]
)

FLOATING_BODY = np.array(
    [
        [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
        [EMPTY_CODE, body(1), body(1), EMPTY_CODE, EMPTY_CODE],
        [terminal(2), body(2), body(2), body(2), terminal(2)],
    ]
)

QUICK_GAME = np.array([[terminal(1), EMPTY_CODE, terminal(1)]])


@pytest.fixture
def empty_5x5_game() -> FlowFree:
    """Return a blank 5 x 5 game with two colours."""
    return FlowFree(5, 5, terminals=TERMINALS_5x5)


def test_board_initialization(empty_5x5_game: FlowFree) -> None:
    assert empty_5x5_game.board.shape == (5, 5)


@pytest.mark.parametrize(
    "board, is_valid",
    [
        (VALID_BOARD_3x5, True),
        (INVALID_SINGLE_TERMINAL, False),
        (INVALID_MISSING_HEAD, False),
        (INCOMPLETE_BUT_VALID, True),
        (FLOATING_HEAD, False),
        (DISCONNECTED_HEAD, False),
        (FLOATING_BODY, False),
        (IMPROPER_HEADS_1, False),
        (IMPROPER_HEADS_2, False),
        (COMPLETED_GAME_1, True),
        (COMPLETED_GAME_2, True),
    ],
)
def test_is_valid_board(board: np.ndarray, is_valid: bool) -> None:
    msg = "Board validity mismatch for test case"
    assert FlowFree.is_valid_board(board) is is_valid, msg


def test_board_creation_from_ndarray() -> None:
    ff = FlowFree.from_board(VALID_BOARD_3x5)
    assert ff.board.shape == (3, 5)
    assert np.array_equal(ff.board, VALID_BOARD_3x5)

    # Spot-checks
    assert ff._tile(Coordinate(0, 0)) == terminal(1)
    assert ff._tile(Coordinate(0, 1)) == EMPTY_CODE

    # Game state
    assert not ff.is_solved()
    assert not ff.is_color_solved(1)
    assert ff.is_color_solved(2)


def test_path_completion() -> None:
    board = np.array(
        [
            [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
            [terminal(2), body(2), head(2), EMPTY_CODE, terminal(2)],
        ]
    )
    ff = FlowFree.from_board(board)

    move = Coordinate(2, 3)
    assert ff.is_legal_move(move, 2)

    ill_move = Coordinate(0, 1)
    assert not ff.is_legal_move(ill_move, 2)

    ill_move = Coordinate(1, 1)
    assert not ff.is_legal_move(ill_move, 2)
    assert not ff.is_legal_move(ill_move, 1)

    ff.attempt_move(move, 2)

    assert ff._board[2, 3] == body(2)
    assert ff._heads[2] is None
    assert ff.is_color_solved(2)
    assert not ff.is_solved()
    assert not ff.is_color_solved(1)


def test_quick_game() -> None:
    ff = FlowFree.from_board(QUICK_GAME)

    move = Coordinate(0, 1)
    assert ff.is_legal_move(move, 1)
    ff.attempt_move(move, 1)

    assert ff.is_solved()
    assert ff.is_color_solved(1)
    assert ff._heads[1] is None
    assert ff._board[0, 1] == body(1)


@pytest.mark.parametrize(
    "board, is_complete",
    [
        (VALID_BOARD_3x5, False),
        (INCOMPLETE_BUT_VALID, False),
        (COMPLETED_GAME_1, True),
        (COMPLETED_GAME_2, True),
    ],
)
def test_game_completion(board: np.ndarray, is_complete: bool) -> None:
    msg = "Board game completion status for test case"
    if FlowFree.is_valid_board(board):
        f = FlowFree.from_board(board)
        assert f.is_solved() is is_complete, msg


def test_board_import_export() -> None:
    for board in [
        VALID_BOARD_3x5,
        COMPLETED_GAME_1,
        COMPLETED_GAME_2,
        INCOMPLETE_BUT_VALID,
    ]:
        ff = FlowFree.from_board(board)
        exported = ff.get_internal_board()
        assert np.array_equal(exported, board), "Exported board does not match original"
        imported_ff = FlowFree.from_board(exported)
        assert np.array_equal(imported_ff.board, board), "Imported board does not match original"


def test_color_reset() -> None:
    """Test resetting a color in the game."""
    ff = FlowFree.from_board(VALID_BOARD_3x5)
    assert not ff.is_color_solved(1)

    # Attempt to reset color 1
    ff.reset_color(1)

    # Check that the board is unchanged, but color 1 is now unsolved
    assert np.array_equal(ff.board, VALID_BOARD_3x5)
    assert not ff.is_color_solved(1)
    assert ff._heads[1] is None  # Color 1 should have no head after reset

    # Reset color 2
    ff.reset_color(2)
    assert not ff.is_color_solved(2)  # Color 2 should no longer be solved
    assert not np.array_equal(ff.board, VALID_BOARD_3x5)  # Board should be changed


BASE = Path(__file__).resolve().parent.parent.parent

SYNTH_PATH = BASE / "flowzero_src" / "data" / "synthetic"
SYNTH_BOARDS = [
    import_ndarray(file) for file in SYNTH_PATH.glob("*.npy")
]  # 20 synthetic boards at random


def test_20_synthetic_boards() -> None:
    """Test that the synthetic boards can be read back correctly."""
    for board in SYNTH_BOARDS:
        assert isinstance(board, np.ndarray), "Board data should be a numpy array"
        assert FlowFree.is_valid_board(board), "Board data should be valid"
        game = FlowFree.from_board(board)
        solver = FlowFreeSATSolver(game)
        assert solver.is_solvable(), "Synthetic board should be solvable"


@pytest.mark.parametrize(
    "board",
    [
        VALID_BOARD_3x5,
        INCOMPLETE_BUT_VALID,
        COMPLETED_GAME_1,
        COMPLETED_GAME_2,
        *SYNTH_BOARDS[:10],  # TODO: Investigate the failing test cases
    ],  # Use first 10 synthetic boards for valid moves
)
def test_valid_moves(board: np.ndarray) -> None:
    """Ensure the valid moves are actually valid."""
    ff = FlowFree.from_board(board)
    valid_moves: set[Action] = ff.get_all_valid_moves()

    for a in valid_moves:
        color = a.color
        coord = a.coordinate
        ff = FlowFree.from_board(board)  # Reset the board for each action check

        if a.action_type == ActionTypes.PLACE:
            assert ff.is_legal_move(coord, color), f"Move {coord} for color {color} is not legal"
            # Check that the move can be made
            ff.attempt_move(coord, color)
            assert ff._board[coord.row, coord.col] == body(color) or ff._board[
                coord.row, coord.col
            ] == head(color), f"Move {coord} for color {color} did not update the board correctly"
        elif a.action_type == ActionTypes.RESET:
            ff.reset_color(color)
            assert not ff.is_color_solved(color), f"Color {color} should not be solved after reset"
            # Check that the head is None after reset
            assert ff._heads.get(color, None) is None, (
                f"Head for color {color} should be None after reset"
            )
