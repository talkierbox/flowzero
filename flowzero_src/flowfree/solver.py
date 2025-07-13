"""
Solves a Flow Free board using a SAT solver via PySAT.

This is used to generate synthetic boards for RL experiments."
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Minisat22

from flowzero_src.flowfree.game import (
    Coordinate,
    FlowFree,
    body,
)

if TYPE_CHECKING:
    import numpy as np

# Edge is represented by a tuple of two Coordinates
Edge = tuple[Coordinate, Coordinate]


class FlowFreeSATSolver:
    """Compile a FlowFree puzzle to CNF and solve it with PySAT."""

    def __init__(self, game: FlowFree) -> None:
        """Initialize the solver with a FlowFree game instance."""
        game = FlowFree.from_board(game._board)
        game.reset_board()

        self.game = game
        self.rows, self.cols = game.rows, game.cols
        # map color -> (terminal1, terminal2)
        self.terminals: dict[int, tuple[Coordinate, Coordinate]] = game._terminals
        # actual color palette: 0 == unused plus terminal colors
        self.colors: list[int] = [0, *sorted(self.terminals.keys())]
        self.id_pool = IDPool()
        self.edges: list[Edge] = []
        self.incident: dict[Coordinate, list[int]] = {}
        self._enumerate_edges()

    def solve(self, strict: bool = True) -> FlowFree:
        """Return a newly solved FlowFree instance, or raise ValueError."""
        cnf = self._build_cnf()
        with Minisat22(bootstrap_with=cnf) as sat:
            if not sat.solve():
                raise ValueError("Puzzle is unsatisfiable - no solution exists.")
            model = set(sat.get_model())

        solved_arr = self._model_to_board(model)
        if not FlowFree.is_valid_board(solved_arr):
            raise ValueError("SAT model did not yield a valid Flow Free board.")

        solved = FlowFree.from_board(solved_arr)
        if strict and not solved.is_solved():
            raise ValueError("SAT model did not yield a valid Flow Free solution.")
        return solved

    def is_solvable(self) -> bool:
        """Check if the puzzle has any SAT solution (not necessarily valid final)."""
        try:
            return self.solve(strict=False).is_solved()
        except ValueError:
            return False

    def _enumerate_edges(self) -> None:
        """Build the list of undirected grid edges and per-cell incident lists."""

        def add(e1: Coordinate, e2: Coordinate) -> None:
            idx = len(self.edges)
            self.edges.append((e1, e2))
            self.incident.setdefault(e1, []).append(idx)
            self.incident.setdefault(e2, []).append(idx)

        for r in range(self.rows):
            for c in range(self.cols):
                cur = Coordinate(r, c)
                if r + 1 < self.rows:
                    add(cur, Coordinate(r + 1, c))
                if c + 1 < self.cols:
                    add(cur, Coordinate(r, c + 1))

    def _var(self, edge_idx: int, color: int) -> int:
        """Return a unique SAT variable ID for edge 'edge_idx' having 'color'. Color=0 means unused."""
        return self.id_pool.id((edge_idx, color))

    def _build_cnf(self) -> CNF:
        cnf = CNF()
        # 1) Edge exclusivity: each edge picks exactly one colour
        for e_idx in range(len(self.edges)):
            lits = [self._var(e_idx, c) for c in self.colors]
            cnf.extend(CardEnc.equals(lits=lits, bound=1, vpool=self.id_pool))

        # 2) Cell constraints
        for cell, incident in self.incident.items():
            if any(cell == term for pair in self.terminals.values() for term in pair):
                tcol = next(col for col, pair in self.terminals.items() if cell in pair)
                self._encode_terminal(cnf, incident, tcol)
            else:
                self._encode_body(cnf, incident)
        return cnf

    def _encode_terminal(
        self,
        cnf: CNF,
        incident: list[int],
        color: int,
    ) -> None:
        """Terminal cell: exactly one edge of 'color', no other pipe colors."""
        # exactly one edge of the given color
        vars_c = [self._var(e, color) for e in incident]
        cnf.extend(CardEnc.equals(lits=vars_c, bound=1, vpool=self.id_pool))
        # forbid other pipe colors, allow unused (0)
        for e in incident:
            for c in self.colors:
                if c != 0 and c != color:
                    cnf.append([-self._var(e, c)])

    def _encode_body(
        self,
        cnf: CNF,
        incident: list[int],
    ) -> None:
        """Non-terminal cell: exactly two coloured edges of the same colour."""
        # gather all pipe-color vars (exclude 0)
        colored_vars = [self._var(e, c) for e in incident for c in self.colors if c != 0]
        cnf.extend(CardEnc.equals(lits=colored_vars, bound=2, vpool=self.id_pool))
        # forbid two edges having different colours
        for e1, e2 in combinations(incident, 2):
            for c1 in self.colors:
                for c2 in self.colors:
                    if c1 != 0 and c2 != 0 and c1 != c2:
                        cnf.append([-self._var(e1, c1), -self._var(e2, c2)])

    def _model_to_board(self, model: set[int]) -> np.ndarray:
        """Decode SAT model back into a board array."""
        board = self.game.board.copy()
        for cell, incident in self.incident.items():
            # skip terminal cells
            if any(cell == term for pair in self.terminals.values() for term in pair):
                continue
            # find the pipe colour used
            colour = None
            for e in incident:
                for c in self.colors:
                    if c != 0 and self._var(e, c) in model:
                        colour = c
                        break
                if colour is not None:
                    break
            if colour is None:
                raise AssertionError(f"No colour for cell {cell}")
            board[cell.row, cell.col] = body(colour)
        return board


def brute_force_solve(game: FlowFree, depth_limit: int | None = None) -> FlowFree | None:
    """Recursive backtracking fallback."""
    if game.is_solved():
        return game
    if depth_limit is not None and depth_limit <= 0:
        return None

    # pick next unsolved color
    for color in game._terminals:
        if not game.is_color_solved(color):
            break
    else:
        return None

    for r in range(game.rows):
        for c in range(game.cols):
            coord = Coordinate(r, c)
            if not game.is_legal_move(coord, color):
                continue
            snapshot = game.get_internal_board()
            new_game = FlowFree.from_board(snapshot)
            if not new_game.attempt_move(coord, color):
                continue
            next_limit = None if depth_limit is None else depth_limit - 1
            sol = brute_force_solve(new_game, next_limit)
            if sol is not None:
                return sol
    return None
