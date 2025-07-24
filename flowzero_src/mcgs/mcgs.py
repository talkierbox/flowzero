"""Monte-Carlo Graph Search (MCGS) implementation."""

# No longer a naive tree search
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flowzero_src.util.config import get_key

if TYPE_CHECKING:
    from flowzero_src.flowfree.game import Action, EncodedBoard, FlowFree


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


# TODO: Finish this implementation
class MCGS:
    """Monte-Carlo Graph Search (MCGS) class for Flow Free puzzles."""

    def __init__(self, game: FlowFree, simulations_per_move: int = get_key("mcgs.sims_per_move")):
        """Initialize the MCGS with a game instance and number of simulations per move."""
        self.game = game
        self.simulations_per_move = simulations_per_move
        self.state_table: dict[EncodedBoard, StateData] = {}  # Maps board hash to state data
        self.edge_table: dict[
            tuple[EncodedBoard, Action], EdgeData
        ] = {}  # Maps (board hash, action) to edge data
