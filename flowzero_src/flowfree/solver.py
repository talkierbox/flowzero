"""Solves a Flow Free board using a SAT solver via PySAT. Built for completeness sake."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Minisat22

from .game import (
    Coordinate,
    FlowFree,
    body,
)

if TYPE_CHECKING:
    import numpy as np

# Edge is represented by a tuple of two Coordinates
Edge = tuple[Coordinate, Coordinate]


def find_cycle(edges: list[Edge], used_edges: set[int]) -> list[int]:
    """Return one simple cycle (list of edge indices) in the subgraph induced by used_edges."""
    # build adjacency of nodes to (neighbor, edge_idx)
    adj: dict[Coordinate, list[tuple[Coordinate, int]]] = {}
    for idx in used_edges:
        u, v = edges[idx]
        adj.setdefault(u, []).append((v, idx))
        adj.setdefault(v, []).append((u, idx))
    # DFS to find a cycle
    visited: set[Coordinate] = set()
    parent_edge: dict[Coordinate, tuple[Coordinate, int]] = {}

    def dfs(u: Coordinate) -> list[int] | None:
        visited.add(u)
        for v, eidx in adj.get(u, []):
            if v not in visited:
                parent_edge[v] = (u, eidx)
                cycle = dfs(v)
                if cycle:
                    return cycle
            else:
                pu, _ = parent_edge.get(u, (None, None))
                # found back-edge u->v where v is not parent of u
                if v != pu:
                    # reconstruct cycle from u back to v
                    path_edges: list[int] = [eidx]
                    cur = u
                    while cur != v:
                        prev, edge_idx = parent_edge[cur]
                        path_edges.append(edge_idx)
                        cur = prev
                    return path_edges
        return None

    for node in adj:
        if node not in visited:
            parent_edge[node] = (node, -1)
            cycle = dfs(node)
            if cycle:
                return cycle
    return []


class FlowFreeSATSolver:
    """Compile a FlowFree puzzle to CNF and solve it with PySAT, banning loops incrementally."""

    def __init__(self, game: FlowFree) -> None:
        """Initialize the solver with a FlowFree game instance."""
        # preserve only terminals in board
        game = FlowFree.from_board(game._board)
        game.reset_board()
        self.game = game
        self.rows, self.cols = game.rows, game.cols
        # map color -> (terminal1, terminal2)
        self.terminals: dict[int, tuple[Coordinate, Coordinate]] = game._terminals
        # color palette including 0 for unused
        self.colors: list[int] = [0, *sorted(self.terminals.keys())]
        self.id_pool = IDPool()
        self.edges: list[Edge] = []
        self.incident: dict[Coordinate, list[int]] = {}
        self._enumerate_edges()

    def solve(self, strict: bool = True) -> FlowFree:
        """Return a newly solved FlowFree instance, or raise ValueError."""
        # build initial CNF
        base_cnf = self._build_cnf()
        with Minisat22(bootstrap_with=base_cnf.clauses) as solver:
            while True:
                if not solver.solve():
                    raise ValueError("Puzzle is unsatisfiable—no solution exists.")
                model = set(solver.get_model())

                # For each colour, ban any edges not connected back to its terminal
                # This is to prevent free cycles in the solution that are not prevented by the SAT encoding.
                # Special thanks to Matt Zucker for this idea https://mzucker.github.io/2016/09/02/eating-sat-flavored-crow.html
                banned_any = False
                for colour, (t1, _t2) in self.terminals.items():
                    # collect all edges chosen for this colour
                    chosen = [e for e in range(len(self.edges)) if self._var(e, colour) in model]

                    # build adjacency map for chosen edges
                    adj: dict[Coordinate, list[tuple[Coordinate, int]]] = {}
                    for e in chosen:
                        a, b = self.edges[e]
                        adj.setdefault(a, []).append((b, e))
                        adj.setdefault(b, []).append((a, e))

                    # flood-fill from terminal t1
                    seen = {t1}
                    stack = [t1]
                    while stack:
                        cur = stack.pop()
                        for nbr, _eidx in adj.get(cur, ()):
                            if nbr not in seen:
                                seen.add(nbr)
                                stack.append(nbr)

                    # any chosen edge whose BOTH endpoints are outside 'seen' is in a free cycle
                    unreachable = [
                        e
                        for e in chosen
                        if self.edges[e][0] not in seen and self.edges[e][1] not in seen
                    ]
                    if unreachable:
                        # ban that whole component at once
                        clause = [-self._var(e, colour) for e in unreachable]
                        solver.add_clause(clause)
                        banned_any = True

                if banned_any:
                    # re-solve with the new bans
                    continue

                # no more free cycles — decode and verify
                arr = self._model_to_board(model)
                if not FlowFree.is_valid_board(arr):
                    raise ValueError("SAT model did not yield a valid Flow Free board.")
                solved = FlowFree.from_board(arr)
                if strict and not solved.is_solved():
                    raise ValueError("SAT model did not yield a valid Flow Free solution.")
                return solved

    def is_solvable(self) -> bool:
        """Check if the puzzle has any valid SAT solution."""
        try:
            return self.solve(strict=True).is_solved()
        except ValueError:
            return False

    def _enumerate_edges(self) -> None:
        """Build list of undirected edges and incident lists."""

        def add(u: Coordinate, v: Coordinate) -> None:
            idx = len(self.edges)
            self.edges.append((u, v))
            self.incident.setdefault(u, []).append(idx)
            self.incident.setdefault(v, []).append(idx)

        for r in range(self.rows):
            for c in range(self.cols):
                cur = Coordinate(r, c)
                if r + 1 < self.rows:
                    add(cur, Coordinate(r + 1, c))
                if c + 1 < self.cols:
                    add(cur, Coordinate(r, c + 1))

    def _var(self, edge_idx: int, color: int) -> int:
        """SAT var for edge 'edge_idx' having 'color'."""
        return self.id_pool.id((edge_idx, color))

    def _build_cnf(self) -> CNF:
        """Construct CNF with local constraints only."""
        cnf = CNF()
        # edge exclusivity: each edge picks exactly one colour
        for idx in range(len(self.edges)):
            lits = [self._var(idx, c) for c in self.colors]
            cnf.extend(CardEnc.equals(lits=lits, bound=1, vpool=self.id_pool))
        # cell rules
        for cell, inc in self.incident.items():
            if cell in [t for pair in self.terminals.values() for t in pair]:
                col = next(c for c, pair in self.terminals.items() if cell in pair)
                self._encode_terminal(cnf, inc, col)
            else:
                self._encode_body(cnf, inc)
        return cnf

    def _encode_terminal(self, cnf: CNF, inc: list[int], color: int) -> None:
        """Terminal cell: one edge of 'color', no other pipes."""
        # exactly one incident edge of this colour
        cnf.extend(
            CardEnc.equals(lits=[self._var(e, color) for e in inc], bound=1, vpool=self.id_pool)
        )
        # forbid other pipe colours on those edges
        for e in inc:
            for c in self.colors:
                if c != 0 and c != color:
                    cnf.append([-self._var(e, c)])

    def _encode_body(self, cnf: CNF, inc: list[int]) -> None:
        """Non-terminal cell: two edges same colour."""
        # pick exactly two coloured edges
        lits = [self._var(e, c) for e in inc for c in self.colors if c != 0]
        cnf.extend(CardEnc.equals(lits=lits, bound=2, vpool=self.id_pool))
        # forbid pairs of different colours
        for e1, e2 in combinations(inc, 2):
            for c1 in self.colors:
                for c2 in self.colors:
                    if c1 != 0 and c2 != 0 and c1 != c2:
                        cnf.append([-self._var(e1, c1), -self._var(e2, c2)])

    def _model_to_board(self, model: set[int]) -> np.ndarray:
        """Decode SAT model back into a board; terminals remain, others via body()."""
        arr = self.game.board.copy()
        for cell, inc in self.incident.items():
            if cell in [t for pair in self.terminals.values() for t in pair]:
                continue
            col = None
            for e in inc:
                for c in self.colors:
                    if c != 0 and self._var(e, c) in model:
                        col = c
                        break
                if col is not None:
                    break
            if col is None:
                raise AssertionError(f"No colour for {cell}")
            arr[cell.row, cell.col] = body(col)
        return arr


def brute_force_solve(game: FlowFree, depth_limit: int | None = None) -> FlowFree | None:
    """Recursive backtracking fallback."""
    if game.is_solved():
        return game
    if depth_limit is not None and depth_limit <= 0:
        return None
    for col in game._terminals:
        if not game.is_color_solved(col):
            break
    else:
        return None
    for r in range(game.rows):
        for c in range(game.cols):
            coord = Coordinate(r, c)
            if not game.is_legal_move(coord, col):
                continue
            snap = game.get_internal_board()
            ng = FlowFree.from_board(snap)
            if not ng.attempt_move(coord, col):
                continue
            nxt = None if depth_limit is None else depth_limit - 1
            sol = brute_force_solve(ng, nxt)
            if sol:
                return sol
    return None
