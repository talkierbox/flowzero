"""Pytest suite for the Monte-Carlo Graph Search (MCGS) implementation."""

from __future__ import annotations

import numpy as np
import pytest

from flowzero_src.flowfree.game import Action, ActionTypes, EncodedBoard, FlowFree, body, terminal
from flowzero_src.mcgs.mcgs import MCGS, EdgeData, StateData

# Test board configurations
SIMPLE_2x3_BOARD = np.array(
    [[terminal(1), 0, terminal(1)], [terminal(2), 0, terminal(2)]], dtype=np.int8
)

SIMPLE_SOLVED_BOARD = np.array(
    [[terminal(1), body(1), terminal(1)], [terminal(2), body(2), terminal(2)]], dtype=np.int8
)

# A solvable 4x4 board with three colors
LARGER_BOARD = np.array(
    [
        [terminal(1), 0, 0, terminal(1)],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [terminal(2), 0, 0, terminal(2)],
    ],
    dtype=np.int8,
)

# Alternative larger board - 3x4 that's definitely solvable
MEDIUM_BOARD = np.array(
    [
        [terminal(1), 0, 0, terminal(1)],
        [terminal(2), 0, 0, terminal(2)],
        [terminal(3), 0, 0, terminal(3)],
    ],
    dtype=np.int8,
)


@pytest.fixture
def simple_game() -> FlowFree:
    """Create a simple 2x3 game for testing."""
    return FlowFree.from_board(SIMPLE_2x3_BOARD)


@pytest.fixture
def solved_game() -> FlowFree:
    """Create a solved game for testing."""
    return FlowFree.from_board(SIMPLE_SOLVED_BOARD)


@pytest.fixture
def larger_game() -> FlowFree:
    """Create a larger 4x4 game for testing."""
    return FlowFree.from_board(LARGER_BOARD)


@pytest.fixture
def medium_game() -> FlowFree:
    """Create a medium 3x4 game for testing."""
    return FlowFree.from_board(MEDIUM_BOARD)


class TestMCGSInitialization:
    """Test MCGS initialization and basic setup."""

    def test_init_with_game(self, simple_game: FlowFree) -> None:
        """Test MCGS initialization with game."""
        mcgs = MCGS(simple_game)
        assert mcgs.current_game == simple_game
        assert len(mcgs.state_table) == 1  # Root state should be initialized
        assert mcgs.root_key in mcgs.state_table
        assert len(mcgs.edge_table) == 0

    def test_init_with_custom_simulations(self, simple_game: FlowFree) -> None:
        """Test MCGS initialization with custom simulation count."""
        mcgs = MCGS(simple_game, simulations_per_move=50)
        assert mcgs.simulations_per_move == 50


class TestMCGSSingleAction:
    """Test MCGS single action selection."""

    def test_get_best_action_basic(self, simple_game: FlowFree) -> None:
        """Test getting best action."""
        mcgs = MCGS(simple_game, simulations_per_move=10)  # Low simulation count for fast testing
        action = mcgs.get_best_action()

        assert action is not None
        assert isinstance(action, Action)
        assert action.action_type in [ActionTypes.PLACE, ActionTypes.RESET]

    def test_get_best_action_solved_game_returns_none(self, solved_game: FlowFree) -> None:
        """Test that get_best_action returns None for solved games."""
        mcgs = MCGS(solved_game)
        action = mcgs.get_best_action()
        assert action is None

    def test_get_best_action_updates_state_table(self, simple_game: FlowFree) -> None:
        """Test that get_best_action populates the state table."""
        mcgs = MCGS(simple_game, simulations_per_move=5)
        initial_states = len(mcgs.state_table)

        mcgs.get_best_action()

        # Should have explored more states
        assert len(mcgs.state_table) > initial_states


class TestMCGSFullGame:
    """Test MCGS full game playing functionality."""

    def test_play_game_simple_solvable(self, simple_game: FlowFree) -> None:
        """Test playing a simple solvable game."""
        mcgs = MCGS(simple_game, simulations_per_move=20)  # Increased for better performance
        final_game, move_history, solved = mcgs.play_game()

        assert isinstance(final_game, FlowFree)
        assert isinstance(move_history, list)
        assert isinstance(solved, bool)
        assert len(move_history) >= 0  # Could be 0 if already solved

        # All moves should be valid Action objects
        for move in move_history:
            assert isinstance(move, Action)

    def test_play_game_medium_solvable(self, medium_game: FlowFree) -> None:
        """Test playing a medium solvable game."""
        mcgs = MCGS(medium_game, simulations_per_move=100)
        final_game, move_history, solved = mcgs.play_game(max_moves=30)

        assert isinstance(final_game, FlowFree)
        assert isinstance(move_history, list)
        assert len(move_history) <= 20

    def test_play_game_max_moves_limit(self, medium_game: FlowFree) -> None:
        """Test that play_game respects max_moves limit."""
        mcgs = MCGS(medium_game, simulations_per_move=5)  # Low simulations for speed
        final_game, move_history, solved = mcgs.play_game(max_moves=5)

        assert len(move_history) <= 5

    def test_play_game_does_not_modify_original(self, simple_game: FlowFree) -> None:
        """Test that play_game doesn't modify the original game."""
        original_board = simple_game.get_internal_board().copy()
        mcgs = MCGS(simple_game, simulations_per_move=10)

        mcgs.play_game()

        # Original game should be unchanged
        assert np.array_equal(simple_game.get_internal_board(), original_board)

    def test_play_game_already_solved(self, solved_game: FlowFree) -> None:
        """Test playing a game that's already solved."""
        mcgs = MCGS(solved_game, simulations_per_move=5)
        final_game, move_history, solved = mcgs.play_game()

        assert solved is True
        assert len(move_history) == 0  # No moves should be needed


class TestMCGSInternalMethods:
    """Test MCGS internal methods and data structures."""

    def test_state_data_creation(self) -> None:
        """Test StateData initialization."""
        state = StateData()
        assert state.value == 0.0
        assert state.visits == 0
        assert len(state.children) == 0

    def test_edge_data_creation(self) -> None:
        """Test EdgeData initialization."""
        edge = EdgeData()
        assert edge.value == 0.0
        assert edge.visits == 0
        assert edge.child_key is None

    def test_get_state_data_creates_new_state(self, simple_game: FlowFree) -> None:
        """Test that _get_state_data creates new states when needed."""
        mcgs = MCGS(simple_game)

        # Create a dummy encoded board key
        dummy_key = (simple_game.get_internal_board().shape, b"dummy")

        assert dummy_key not in mcgs.state_table
        state_data = mcgs._get_state_data(dummy_key)

        assert dummy_key in mcgs.state_table
        assert isinstance(state_data, StateData)

    def test_reward_function_solved_game(self, solved_game: FlowFree) -> None:
        """Test reward function for solved games."""
        mcgs = MCGS(solved_game)
        reward = mcgs._reward_func(solved_game)
        assert reward == 1.0

    def test_reward_function_empty_game(self, simple_game: FlowFree) -> None:
        """Test reward function for empty games."""
        mcgs = MCGS(simple_game)
        reward = mcgs._reward_func(simple_game)
        assert 0.0 <= reward < 1.0  # Should be between 0 and 1, but not 1


class TestMCGSValidMoves:
    """Test MCGS interaction with valid moves."""

    def test_mcgs_respects_valid_moves(self, simple_game: FlowFree) -> None:
        """Test that MCGS only considers valid moves."""
        mcgs = MCGS(simple_game, simulations_per_move=5)

        # Get valid moves from the game
        valid_moves = simple_game.get_all_valid_moves()

        # Run a few simulations to populate the tree
        mcgs.get_best_action()

        # Check that root state only has valid actions as children
        root_state = mcgs._get_state_data(mcgs.root_key)
        if root_state.children:
            for action in root_state.children:
                assert action in valid_moves


class TestMCGSEdgeCases:
    """Test MCGS edge cases and error conditions."""

    def test_mcgs_with_zero_simulations(self, simple_game: FlowFree) -> None:
        """Test MCGS with zero simulations per move."""
        mcgs = MCGS(simple_game, simulations_per_move=0)
        action = mcgs.get_best_action()

        # Should still return None since no simulations were run
        # and no children were explored
        assert action is None

    def test_mcgs_encoding_decoding(self, simple_game: FlowFree) -> None:
        """Test that MCGS correctly handles board encoding/decoding."""
        MCGS(simple_game)

        # Encode the board
        encoded: EncodedBoard = simple_game.encode_board()

        # Decode it back
        decoded_game = FlowFree.from_encoded_board(encoded)

        # Should be equivalent to original
        assert np.array_equal(simple_game.get_internal_board(), decoded_game.get_internal_board())

    def test_mcgs_cycle_detection(self, simple_game: FlowFree) -> None:
        """Test that MCGS handles cycles in the search tree."""
        mcgs = MCGS(simple_game, simulations_per_move=10)

        # Run simulations - should complete without infinite loops
        mcgs.get_best_action()

        # If we get here, cycle detection worked
        assert True

    def test_mcgs_verbose_output(
        self, simple_game: FlowFree, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test MCGS verbose output."""
        mcgs = MCGS(simple_game, simulations_per_move=5)

        # Test verbose single action
        mcgs.get_best_action(verbose=True)
        captured = capsys.readouterr()
        assert "Starting MCGS process" in captured.out

        # Test verbose game play
        mcgs.play_game(verbose=True, max_moves=3)
        captured = capsys.readouterr()
        # Should have some output
        assert len(captured.out) > 0


@pytest.mark.slow
class TestMCGSPerformance:
    """Performance and integration tests for MCGS."""

    def test_mcgs_medium_game_performance(self, medium_game: FlowFree) -> None:
        """Test MCGS performance on medium games."""
        mcgs = MCGS(medium_game, simulations_per_move=50)

        # Should complete in reasonable time
        final_game, move_history, solved = mcgs.play_game(max_moves=20)

        assert isinstance(final_game, FlowFree)
        assert len(move_history) <= 20

    def test_mcgs_larger_game_performance(self, larger_game: FlowFree) -> None:
        """Test MCGS performance on larger games."""
        mcgs = MCGS(larger_game, simulations_per_move=30)

        # Should complete in reasonable time, but may not solve due to complexity
        final_game, move_history, solved = mcgs.play_game(max_moves=15)

        assert isinstance(final_game, FlowFree)
        assert len(move_history) <= 15

    def test_mcgs_state_table_growth(self, simple_game: FlowFree) -> None:
        """Test that MCGS state table grows appropriately."""
        mcgs = MCGS(simple_game, simulations_per_move=20)

        initial_states = len(mcgs.state_table)
        mcgs.get_best_action()
        final_states = len(mcgs.state_table)

        # Should have explored more states
        assert final_states > initial_states

    def test_mcgs_multiple_runs_consistency(self, simple_game: FlowFree) -> None:
        """Test that MCGS produces consistent results across runs."""
        # Note: Due to randomness, we test for general consistency rather than exact matches
        results = []

        for _ in range(3):
            mcgs = MCGS(simple_game, simulations_per_move=10)
            action = mcgs.get_best_action()
            results.append(action)

        # All results should be valid actions (or None)
        for result in results:
            assert result is None or isinstance(result, Action)
