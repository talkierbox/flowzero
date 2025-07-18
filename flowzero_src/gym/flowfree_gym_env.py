"""Flow Free Gymnasium Environment."""

import gymnasium as gym

from flowzero_src.flowfree.game import VALID_TILE_INTS, FlowFree


# TODO: Add more environment metadata, such as action space, reward structure, etc.
class FlowFreeEnv(gym.Env):
    """Flow Free environment for gymnasium."""

    def __init__(self, board: FlowFree):
        """Initialize the Flow Free environment with a given board."""
        super().__init__()
        self.board = board

        assert self.board.is_valid_board(self.board.get_internal_board()), (
            "Invalid Flow Free board."
        )

        self.observation_space = gym.spaces.Box(
            low=min(VALID_TILE_INTS),
            high=max(VALID_TILE_INTS),
            shape=(self.board.rows, self.board.cols),
            dtype=int,
        )

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_solved()
