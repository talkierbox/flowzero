"""Monte-Carlo Graph Search (MCGS) implementation."""

# No longer a naive tree search
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from flowzero_src.flowfree.game import Action, ActionTypes, FlowFree
from flowzero_src.util.config import get_key

if TYPE_CHECKING:
    from flowzero_src.flowfree.game import EncodedBoard

random.seed(get_key("mcgs.seed", 42))  # Seed for reproducibility


@dataclass(frozen=False, slots=True)
class StateData:
    """Data structure for MCGS state data."""

    value: float = 0.0
    visits: int = 0
    children: dict[Action, EdgeData] = field(default_factory=dict)


@dataclass(frozen=False, slots=True)
class EdgeData:
    """Data structure for MCGS edge data."""

    value: float = 0.0
    child_key: EncodedBoard | None = None
    visits: int = 0


class MCGS:
    """Monte-Carlo Graph Search (MCGS) class for Flow Free puzzles."""

    def __init__(
        self,
        game: FlowFree,
        simulations_per_move: int = get_key("mcgs.sims_per_move", 100),
    ):
        """Initialize the MCGS with a game state and number of simulations per move."""
        self.current_game: FlowFree = game
        self.simulations_per_move = simulations_per_move
        self.state_table: dict[EncodedBoard, StateData] = {}  # Maps board hash to state data
        self.edge_table: dict[
            tuple[EncodedBoard, Action], EdgeData
        ] = {}  # Maps (board hash, action) to edge data

        # Initialize root state
        self.root_key: EncodedBoard = game.encode_board()
        self.state_table[self.root_key] = StateData()

    def get_best_action(self, verbose: bool = get_key("mcgs.verbose", False)) -> Action | None:
        """Get the best action for the current game state."""
        if self.current_game.is_solved():
            return None

        if verbose:
            print("Starting MCGS process...")
            for _ in tqdm(range(self.simulations_per_move), desc="Simulating moves", unit="simuls"):
                self._simulate(self.root_key)
        else:
            for _ in range(self.simulations_per_move):
                self._simulate(self.root_key)

        # Select the best action based on visits
        root_state = self._get_state_data(self.root_key)
        if not root_state.children:
            return None

        best_action = max(
            root_state.children.items(),
            key=lambda item: item[1].visits,
            default=(None, None),
        )[0]

        if verbose:
            print(f"Best action: {best_action}")

        return best_action

    def play_game(
        self,
        max_moves: int = get_key("mcgs.max_game_moves", 1000),
        verbose: bool = get_key("mcgs.verbose", False),
    ) -> tuple[FlowFree, list[Action], bool]:
        """Play a complete game using MCGS. Returns (final_game, move_history, solved)."""
        # Make a copy of the game to avoid modifying the original
        current_game = FlowFree.from_board(self.current_game.get_internal_board())
        move_history: list[Action] = []

        for move_count in range(max_moves):
            if current_game.is_solved():
                if verbose:
                    print(f"Game solved in {move_count} moves!")
                return current_game, move_history, True

            # Create a new MCGS instance for the current state
            mcgs_for_move = MCGS(current_game, self.simulations_per_move)
            action = mcgs_for_move.get_best_action(
                verbose=verbose and move_count < 5
            )  # Only verbose for first few moves

            if action is None:
                if verbose:
                    print(f"No valid actions available at move {move_count}")
                break

            # Apply the action
            current_game.attempt_action(action, in_place=True)
            move_history.append(action)

            if verbose and move_count < 10:  # Show first 10 moves
                print(f"Move {move_count + 1}: {action}")

        if verbose:
            print(f"Game ended after {len(move_history)} moves. Solved: {current_game.is_solved()}")

        return current_game, move_history, current_game.is_solved()

    def _simulate(self, start_key: EncodedBoard) -> None:
        """Run a single simulation of the MCGS."""
        path_states: list[EncodedBoard] = []
        path_edges: list[tuple[EncodedBoard, Action]] = []

        cur_key = start_key
        cur_board = FlowFree.from_encoded_board(start_key)

        # Selection and expansion
        while True:
            cur_stats: StateData = self._get_state_data(cur_key)
            path_states.append(cur_key)

            if len(cur_stats.children) == 0:  # No children -> First visit -> Expand all actions
                for action in cur_board.get_all_valid_moves():
                    e = EdgeData()
                    cur_stats.children[action] = e
                    self.edge_table[(cur_key, action)] = e

            # Select an unexplored edge, otherwise use UCT to select the best edge
            unexplored: list[Action] = [
                action for action, edge in cur_stats.children.items() if edge.visits == 0
            ]

            selected_action: Action = None

            if unexplored:
                selected_action = random.choice(unexplored)  # noqa: S311
            else:
                selected_action = max(
                    cur_stats.children.keys(),
                    key=lambda a: self._uct(parent_state=cur_stats, edge=cur_stats.children[a]),
                )

            selected_edge = cur_stats.children[selected_action]

            # Compute the next board state
            next_board: FlowFree = cur_board.attempt_action(selected_action, in_place=False)
            next_key: EncodedBoard = next_board.encode_board()

            if next_key in path_states:  # Avoid a cycle!
                # Mark edge as a dead-end
                selected_edge.visits += 1
                selected_edge.value += 0.0
                path_edges.append((cur_key, selected_action))
                break

            if (
                selected_edge.child_key is None
            ):  # This is the first time this edge has been explored
                selected_edge.child_key = next_key

                if next_key not in self.state_table:  # Ensure the next state is initialized
                    self.state_table[next_key] = StateData()

                path_edges.append((cur_key, selected_action))
                cur_board, cur_key = next_board, next_key  # Move to the next state
                path_states.append(cur_key)
                break  # Expansion done
            else:
                path_edges.append((cur_key, selected_action))
                cur_board, cur_key = next_board, next_key  # continue down the graph
        # Rollout
        reward = self._rollout(cur_board)

        # Backpropagation
        for pk, act in path_edges:
            e = self.edge_table[(pk, act)]
            e.visits += 1
            e.value += reward

        # Update states (once per unique state)
        for state_key in set(path_states):
            s = self.state_table[state_key]
            s.visits += 1
            s.value += reward
        # Done
        return

    """ 
    TODO: Fix looping issue
    Action(action_type=<ActionTypes.PLACE: 'place'>, color=3, coordinate=Coordinate(row=2, col=1))
    Action(action_type=<ActionTypes.PLACE: 'place'>, color=3, coordinate=Coordinate(row=1, col=1))
    Action(action_type=<ActionTypes.PLACE: 'place'>, color=3, coordinate=Coordinate(row=0, col=1))
    Action(action_type=<ActionTypes.PLACE: 'place'>, color=3, coordinate=Coordinate(row=0, col=2))
    Action(action_type=<ActionTypes.PLACE: 'place'>, color=3, coordinate=Coordinate(row=1, col=2))
    Action(action_type=<ActionTypes.RESET: 'reset'>, color=3, coordinate=None)

    x 10
    """

    def _rollout(
        self, board: FlowFree, max_steps: int = get_key("mcgs.max_rollout_steps", 2000)
    ) -> float:
        """Perform a rollout from the given board state and return the reward."""
        cur_board = board
        steps = 0
        consecutive_resets = 0
        max_consecutive_resets = 0

        while not cur_board.is_solved() and steps < max_steps:
            valid_moves: set[Action] = cur_board.get_all_valid_moves()
            if not valid_moves:
                break

            pl_actions = [move for move in valid_moves if move.action_type == ActionTypes.PLACE]
            rs_actions = [move for move in valid_moves if move.action_type == ActionTypes.RESET]

            # Action selection
            if not pl_actions:
                action = random.choice(list(valid_moves))  # noqa: S311
            elif not rs_actions:
                action = random.choice(pl_actions)  # noqa: S311
            else:
                # Avoid too many consecutive resets
                if (
                    consecutive_resets >= 3 or random.random() < 0.7  # noqa: S311
                ):  # Force a place action
                    action = random.choice(pl_actions)  # noqa: S311
                else:
                    action = random.choice(rs_actions)  # noqa: S311

            # Track consecutive resets
            if action.action_type == ActionTypes.RESET:
                consecutive_resets += 1
                max_consecutive_resets = max(max_consecutive_resets, consecutive_resets)
            else:
                consecutive_resets = 0

            cur_board.attempt_action(action, in_place=True)
            steps += 1

        return self._reward_func(cur_board, max_consecutive_resets)

    def _reward_func(self, board: FlowFree, total_resets: int) -> float:
        """Calculate the reward for the given board state."""
        colors_in_board: frozenset[int] = board._colors

        if board.is_solved():
            return 1.0

        num_solved_colors = sum(1 for color in colors_in_board if board.is_color_solved(color))
        num_path_tiles = int(np.sum(board._board > 0))
        total_non_terminal_tiles = (board.rows * board.cols) - (2 * len(colors_in_board))

        tile_completion = (
            num_path_tiles / total_non_terminal_tiles if total_non_terminal_tiles > 0 else 0
        )
        color_completion = num_solved_colors / len(colors_in_board) if colors_in_board else 0

        # Stronger reset penalty
        reset_penalty = min(0.2 * total_resets, 0.5)  # Cap penalty at 0.5

        return max(0.2 * tile_completion + 0.6 * color_completion - reset_penalty, 0)

    def _get_state_data(self, board_state: EncodedBoard) -> StateData:
        """Get or create the state data for a given board state."""
        if board_state not in self.state_table:
            self.state_table[board_state] = StateData()
        return self.state_table[board_state]

    def _uct(
        self,
        parent_state: StateData,
        edge: EdgeData,
        c: float = get_key("mcgs.uct_constant", 1.414),
    ) -> float:
        """Calculate the Upper Confidence Bound for the given state and edge."""
        assert edge.visits > 0, "Edge visits must be greater than zero for UCT calculation"
        return (edge.value / edge.visits) + c * (
            math.log(parent_state.visits + 1e-9) / edge.visits
        ) ** 0.5
