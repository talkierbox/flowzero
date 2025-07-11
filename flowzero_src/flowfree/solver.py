"""Solves a flow free board using a SAT solver --- This is implemented in order to generate synthetic boards for the RL algorihtm."""

from __future__ import annotations

from dataclasses import dataclass
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

Edge = tuple[Coordinate, Coordinate]


@dataclass(slots=True)
class FlowFreeSATSolver:
    """Compile a `FlowFree` puzzle to CNF and solve it with PySAT."""

    game: FlowFree

    def __post_init__(self) -> None:
        """Initialize the SAT solver with the game board."""
        self.rows, self.cols = self.game.rows, self.game.cols
        self.terminals: dict[int, tuple[Coordinate, Coordinate]] = self.game._terminals
        self.k = len(self.terminals)
        self.id_pool = IDPool()
        self.edges: list[Edge] = []
        self.edge_of: dict[tuple[int, int, int], int] = {}
        self.incident: dict[Coordinate, list[int]] = {}
        self._enumerate_edges()

    def solve(self, strict: bool = True) -> FlowFree:
        """Return a *new* solved `FlowFree` instance or raise ``ValueError``."""
        cnf = self._build_cnf()
        with Minisat22(bootstrap_with=cnf) as sat:
            if not sat.solve():
                raise ValueError("Puzzle is unsatisfiable - no solution exists.")
            model = set(sat.get_model())
        solved_board = self._model_to_board(model)
        solved_game = FlowFree.from_board(solved_board)
        if strict and not solved_game.is_solved():
            raise ValueError("SAT model did not yield a valid Flow Free solution.")
        return solved_game

    def _enumerate_edges(self) -> None:
        """Populate `self.edges` and `self.incident` dicts."""

        def add_edge(a: Coordinate, b: Coordinate) -> None:
            idx = len(self.edges)
            self.edges.append((a, b))
            self.incident.setdefault(a, []).append(idx)
            self.incident.setdefault(b, []).append(idx)

        for r in range(self.rows):
            for c in range(self.cols):
                cur = Coordinate(r, c)
                if r + 1 < self.rows:  # vertical edge v
                    add_edge(cur, Coordinate(r + 1, c))
                if c + 1 < self.cols:  # horizontal edge ->
                    add_edge(cur, Coordinate(r, c + 1))

    def _var(self, edge_idx: int, color: int) -> int:
        """Return SAT var-ID for (edge, color). color 0 ⇒ unused."""
        return self.id_pool.id((edge_idx, color))

    def _build_cnf(self) -> CNF:
        cnf = CNF()
        colors = list(range(self.k + 1))  # 0 to k, 0 is unused

        # (1) edge exclusivity: each edge takes exactly one color
        for e_idx in range(len(self.edges)):
            lits = [self._var(e_idx, c) for c in colors]
            cnf.extend(CardEnc.equals(lits=lits, bound=1, vpool=self.id_pool))

        # (2) cell-wise constraints
        for cell, incident_edges in self.incident.items():
            if any(cell == term for pair in self.terminals.values() for term in pair):
                # terminal cell
                t_color = next(k for k, pair in self.terminals.items() if cell in pair)
                self._encode_terminal(cnf, cell, incident_edges, t_color, colors)
            else:
                self._encode_body(cnf, incident_edges, colors)
        return cnf

    # ------------------------------------------------------------------
    #  Encoding per-cell cases
    # ------------------------------------------------------------------

    def _encode_terminal(
        self,
        cnf: CNF,
        # cell: Coordinate,
        incident_edges: list[int],
        color: int,
        colors: list[int],
    ) -> None:
        # Exactly *one* edge of `color` is used
        good = [self._var(e, color) for e in incident_edges]
        cnf.extend(CardEnc.equals(lits=good, bound=1, vpool=self.id_pool))
        # All *other* colors (incl. 0) are disallowed on incident edges
        for e in incident_edges:
            for c in colors:
                if c != color:
                    cnf.append([-self._var(e, c)])

    def _encode_body(
        self,
        cnf: CNF,
        incident_edges: list[int],
        colors: list[int],
    ) -> None:
        # Gather all non-empty color literals for this cell
        colored_lits = [self._var(e, c) for e in incident_edges for c in colors if c != 0]
        # Exactly 2 colored edges overall
        cnf.extend(CardEnc.equals(lits=colored_lits, bound=2, vpool=self.id_pool))
        # Pairwise "same-color" constraint: two colored edges must share color
        for e1, e2 in combinations(incident_edges, 2):
            for c1 in range(1, len(colors)):
                for c2 in range(1, len(colors)):
                    if c1 != c2:
                        cnf.append([-self._var(e1, c1), -self._var(e2, c2)])

    def _model_to_board(self, model: set[int]) -> np.ndarray:
        board = self.game.board  # copy of starting board (terminals set)
        # Resolve color for every non-terminal cell
        for cell, incident_edges in self.incident.items():
            if any(cell == t for pair in self.terminals.values() for t in pair):
                # keep terminal marking
                continue
            # fetch color (> 0) that appears on incident edges
            color = None
            for e in incident_edges:
                for c in range(1, self.k + 1):
                    if self._var(e, c) in model:
                        color = c
                        break
                if color is not None:
                    break
            if color is None:
                raise AssertionError("Model decoding error: no color found for cell")
            board[cell.row, cell.col] = body(color)
        return board
