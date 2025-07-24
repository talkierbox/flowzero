"""Monte-Carlo Graph Search (MCGS) implementation."""

# No longer a naive tree search
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from flowzero_src.flowfree.game import ActionTypes, FlowFree
from flowzero_src.util.config import get_key

if TYPE_CHECKING:
    from flowzero_src.flowfree.game import Action, EncodedBoard


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


# TODO: Testing
class MCGS:
    """Monte-Carlo Graph Search (MCGS) class for Flow Free puzzles."""

    def __init__(
        self, game: FlowFree, simulations_per_move: int = get_key("mcgs.sims_per_move", 100)
    ):
        """Initialize the MCGS with a game instance and number of simulations per move."""
        self.root_key: EncodedBoard = game.encode_board()
        self.simulations_per_move = simulations_per_move
        self.state_table: dict[EncodedBoard, StateData] = {}  # Maps board hash to state data
        self.edge_table: dict[
            tuple[EncodedBoard, Action], EdgeData
        ] = {}  # Maps (board hash, action) to edge data

        self.state_table[self.root_key] = StateData()  # Initialize the root state data

    def start(self, verbose: bool = get_key("mcgs.verbose", False)) -> Action:
        """Start the MCGS process and return the best action."""
        if verbose:
            print("Starting MCGS process...")
            for _ in tqdm(range(self.simulations_per_move), desc="Simulating moves", unit="simuls"):
                self._simulate(self.root_key)
        else:
            for _ in range(self.simulations_per_move):
                self._simulate(self.root_key)

        # Select the best action based on visits
        root_state = self._get_state_data(self.root_key)
        best_action = max(
            root_state.children.items(),
            key=lambda item: item[1].visits,
            default=(None, None),
        )[0]

        if verbose:
            print(f"Best action: {best_action}")

        return best_action

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

    def _rollout(
        self, board: FlowFree, max_steps: int = get_key("mcgs.max_rollout_steps", 2000)
    ) -> float:
        """Perform a rollout from the given board state and return the reward."""
        cur_board = board
        steps = 0

        while not cur_board.is_solved() and steps < max_steps:
            valid_moves: set[Action] = cur_board.get_all_valid_moves()
            if not valid_moves:
                break
            else:
                action: Action = None
                pl_actions = {move for move in valid_moves if move.action_type == ActionTypes.PLACE}
                rs_actions = {move for move in valid_moves if move.action_type == ActionTypes.RESET}
                # 90% chance to place, 10% chance to reset
                if len(pl_actions) > 0 and len(rs_actions) > 0:
                    action = random.choices(  # noqa: S311
                        population=[*pl_actions, *rs_actions],
                        weights=[0.9] * len(pl_actions) + [0.1] * len(rs_actions),
                        k=1,
                    )[0]
                else:  # Only one type of action available
                    action = random.choice(list(valid_moves))  # noqa: S311
                cur_board.attempt_action(action, in_place=True)
            steps += 1

        # Calculate the reward function
        return self._reward_func(cur_board)

    def _reward_func(self, board: FlowFree) -> float:
        """Calculate the reward for the given board state. Returns a number between 0 and 1."""
        colors_in_board: set[int] = set(board._heads.keys())

        if board.is_solved():
            return 1.0

        # return 0.2 * (# tiles nonempty / # tiles) + 0.6 * (# colors solved / # colors)
        num_solved_colors = sum(1 for color in colors_in_board if board.is_color_solved(color))
        num_nonempty_tiles = int(np.count_nonzero(board._board))
        total_tiles = board.rows * board.cols
        return 0.2 * (num_nonempty_tiles / total_tiles) + 0.50 * (
            num_solved_colors / len(colors_in_board) if colors_in_board else 0
        )

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
