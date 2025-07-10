"""Tests for the Flow Free game board specifically."""

import numpy as np

from flowzero_src.flowfree.game import EMPTY_CODE, Coordinate, FlowFree, body, head, terminal


def test_board_initialization():
    terminals = {1: (Coordinate(0, 0), Coordinate(4, 4)), 2: (Coordinate(0, 4), Coordinate(4, 0))}
    game = FlowFree(5, 5, terminals=terminals)
    # print(game.board_str())
    assert game.board.shape == (5, 5), "Board should be initialized with shape (5, 5)"


def test_board_creation():
    board: np.ndarray = np.array(
        [
            [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
            [terminal(2), body(2), body(2), body(2), terminal(2)],
        ]
    )

    assert FlowFree.is_valid_board(board), "The board should be valid according to FlowFree rules"

    ff = FlowFree.from_board(board)
    assert ff.board.shape == (3, 5), "Board should be created with shape (3, 5)"
    assert np.array_equal(ff.board, board), "The created board should match the expected board"
    # print(ff.board_str())

    assert ff._tile(Coordinate(0, 0)) == (terminal(1)), (
        "Tile at (0, 0) should be a terminal for color 1"
    )
    assert ff._tile(Coordinate(0, 1)) == EMPTY_CODE, "Tile at (0, 1) should be empty"

    invalid_board = np.array(
        [
            [terminal(3), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
            [terminal(2), body(2), body(2), body(2), terminal(2)],
        ]
    )

    assert not FlowFree.is_valid_board(invalid_board), (
        "The board should be invalid due to single terminals"
    )

    valid_board = np.array(
        [
            [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        ]
    )
    assert FlowFree.is_valid_board(valid_board), (
        "The board should be valid with a single color having terminals"
    )

    invalid_board = np.array(
        [
            [terminal(1), body(1), EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
            [terminal(2), body(2), body(2), body(2), terminal(2)],
        ]
    )

    assert not FlowFree.is_valid_board(invalid_board), (
        "The board should be invalid due to color 1 not having a head"
    )

    valid_board = np.array(
        [
            [terminal(1), body(1), body(1), body(1), terminal(1)],
            [body(1), body(1), body(1), body(1), EMPTY_CODE],
        ]
    )

    assert FlowFree.is_valid_board(invalid_board), (
        "The board should be valid, despite being incomplete"
    )

    ff = FlowFree.from_board(valid_board)
    assert ff.board.shape == (2, 5), "Board should be created with shape (2, 5)"
    assert np.array_equal(ff.board, valid_board), (
        "The created board should match the expected board"
    )
    assert not ff.is_solved(), "Game should not be solved yet, not all board squares are filled"
    assert ff.is_color_solved(1), "Color 1 should be solved after placing all segments"


def test_floating():
    # Test for a floating head (no body)
    board: np.ndarray = np.array(
        [
            [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, EMPTY_CODE, head(3), EMPTY_CODE, EMPTY_CODE],
            [terminal(2), body(2), body(2), body(2), terminal(2)],
        ]
    )

    assert not FlowFree.is_valid_board(board), "The board should be invalid due to a floating head"

    board: np.ndarray = np.array(
        [
            [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
            [terminal(2), body(2), body(2), head(2), terminal(2)],
            [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
        ]
    )

    assert not FlowFree.is_valid_board(board), (
        "The board should be valid with a head that could complete the path but is not connected"
    )

    board: np.ndarray = np.array(
        [
            [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, body(1), body(1), EMPTY_CODE, EMPTY_CODE],
            [terminal(2), body(2), body(2), body(2), terminal(2)],
        ]
    )

    assert not FlowFree.is_valid_board(board), "The board should not be valid with floating body"


def test_path_completion():
    # Test for a path that can be completed
    board: np.ndarray = np.array(
        [
            [terminal(1), EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, terminal(1)],
            [EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE, EMPTY_CODE],
            [terminal(2), body(2), head(2), EMPTY_CODE, terminal(2)],
        ]
    )
    ff = FlowFree.from_board(board)
    assert ff.is_legal_move(Coordinate(2, 3), 2), "Legal move to place a 2 at (2, 3) "
    ff.attempt_move(Coordinate(2, 3), 2)  # Completes the path for color 2

    # print(ff._board)

    assert ff._board[2, 3] == body(2), "Tile at (2, 3) should now be a body segment for color 2"
    assert ff._heads[2] is None, "Head for color 2 should be None after completing the path"
    assert ff.is_color_solved(2), "Color 2 should be solved after completing the path"
    assert not ff.is_solved(), "Game should not be solved yet, color 1 is not completed"
    assert not ff.is_color_solved(1), "Color 1 should not be solved yet, path is incomplete"


def test_quick_game():
    board: np.ndarray = np.array([[terminal(1), EMPTY_CODE, terminal(1)]])

    ff = FlowFree.from_board(board)
    assert ff.is_legal_move(Coordinate(0, 1), 1), "Legal move to place a 1 at (0, 1)"
    ff.attempt_move(Coordinate(0, 1), 1)  # Completes the path for color 1
    assert ff.is_solved(), "Game should be solved after completing the path for color 1"
    assert ff.is_color_solved(1), "Color 1 should be solved after completing the path"
    assert ff._heads[1] is None, "Head for color 1 should be None after completing the path"
    assert ff._board[0, 1] == body(1), (
        f"Tile at (0, 1) should now be a body segment for color 1. Got {ff._board[0, 1]} instead."
    )
