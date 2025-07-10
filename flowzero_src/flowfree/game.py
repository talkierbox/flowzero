"""
Head-aware internal board representation for the Flow Free puzzle.

Each tile is stored as a single signed int8, so an in-progress board can be saved or loaded without extra metadata:

0 represents an empty square

+c (1 to 15) represents a body segment of color c

-c (-1 to -15) represents a terminal of color c

HEAD_OFFSET + c represents the current head (path frontier) of color c

HEAD_OFFSET is chosen to keep these values within the valid int8 range (-128 to 127) and avoid overlap with other codes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

MAX_COLORS: int = 15  # vanilla Flow Free palette
HEAD_OFFSET: int = 32  # 33…47 become head codes

EMPTY_CODE: np.int8 = np.int8(0)


def body(c: int) -> np.int8:
    """Return the int8 code for a body segment of color *c*."""
    return np.int8(c)


def terminal(c: int) -> np.int8:
    """Return the int8 code for a terminal of color *c*."""
    return np.int8(-c)


def head(c: int) -> np.int8:
    """Return the int8 code for the head of color *c*."""
    return np.int8(HEAD_OFFSET + c)


def is_empty(v: int) -> bool:
    """Return ``True`` iff *v* encodes an empty tile."""
    return v == 0


def is_body(v: int) -> bool:
    """Return ``True`` iff *v* encodes a body segment."""
    return 0 < v <= MAX_COLORS


def is_terminal(v: int) -> bool:
    """Return ``True`` iff *v* encodes a terminal."""
    return -MAX_COLORS <= v < 0


def is_head(v: int) -> bool:
    """Return ``True`` iff *v* encodes a head segment."""
    return HEAD_OFFSET < v <= HEAD_OFFSET + MAX_COLORS


def color_of(v: int) -> int:
    """Return the color index (1…15) associated with tile code *v*."""
    if is_body(v):
        return v
    if is_terminal(v):
        return -v
    if is_head(v):
        return v - HEAD_OFFSET
    if v == EMPTY_CODE:
        return EMPTY_CODE
    raise ValueError("Invalid Flow Free tile code")


VALID_TILE_INTS: set[int] = (
    set(range(-MAX_COLORS, 0))
    | {0}
    | set(range(1, MAX_COLORS + 1))
    | set(range(HEAD_OFFSET + 1, HEAD_OFFSET + MAX_COLORS + 1))
)


@dataclass(frozen=True, slots=True)
class Coordinate:
    """Immutable (row, col) grid coordinate."""

    row: int
    col: int

    def neighbors4(self) -> list[Coordinate]:
        """Return the four cardinal neighbours of the coordinate."""
        return [
            Coordinate(self.row - 1, self.col),
            Coordinate(self.row + 1, self.col),
            Coordinate(self.row, self.col - 1),
            Coordinate(self.row, self.col + 1),
        ]

    def manhattan(self, other: Coordinate) -> int:
        """Return L1 distance to *other*."""
        return abs(self.row - other.row) + abs(self.col - other.col)

    def __add__(self, other: Coordinate) -> Coordinate:
        """Return a new coordinate that is the sum of this and *other*."""
        return Coordinate(self.row + other.row, self.col + other.col)

    def __subtract__(self, other: Coordinate) -> Coordinate:
        """Return a new coordinate that is the difference of this and *other*."""
        return Coordinate(self.row - other.row, self.col - other.col)


class FlowFree:
    """Mutable in-memory game state for a Flow Free board."""

    #
    def __init__(
        self,
        rows: int,
        cols: int,
        terminals: dict[int, tuple[Coordinate, Coordinate]],
    ) -> None:
        """Create a fresh board with given *terminals* mapping color→(t1, t2)."""
        self.rows, self.cols = rows, cols
        self._board: np.ndarray = np.zeros((rows, cols), dtype=np.int8)
        self._heads: dict[int, Coordinate | None] = dict.fromkeys(terminals)
        self._terminals = terminals

        # place terminal codes
        for color, (a, b) in terminals.items():
            self._place(a, terminal(color))
            self._place(b, terminal(color))

    def _in_bounds(self, c: Coordinate) -> bool:
        """Return ``True`` iff *c* lies on the board."""
        return 0 <= c.row < self.rows and 0 <= c.col < self.cols

    def _tile(self, c: Coordinate) -> int:
        """Return raw int8 code stored at *c*."""
        if not self._in_bounds(c):
            raise ValueError("Coordinate out of bounds")
        return int(self._board[c.row, c.col])

    def _place(self, c: Coordinate, code: np.int8) -> None:
        """Write *code* to board at *c* without checks."""
        self._board[c.row, c.col] = code

    def is_legal_move(self, c: Coordinate, color: int) -> bool:
        """Return ``True`` if placing next segment of *color* at *c* is legal."""
        if not self._in_bounds(c) or not is_empty(self._tile(c)):
            return False

        if color_of(self._tile(c)) == color or is_terminal(self._tile(c)):
            # cannot place on a tile of the same color or on any terminal
            return False

        if color == terminal(color) or color == EMPTY_CODE:
            # cannot place a terminal or empty tile
            return False

        if color not in VALID_TILE_INTS:
            raise ValueError("Invalid color code")

        head_pos = self._heads.get(color, None)
        if head_pos is None:
            # must touch one of its terminals
            return any(
                self._in_bounds(n)
                and is_terminal(self._tile(n))
                and color_of(self._tile(n)) == color
                for n in c.neighbors4()
            )
        # must extend from current head
        return c.manhattan(head_pos) == 1

    def reset_color(self, color: int) -> None:
        """Reset all paths of a specific color."""
        if color == EMPTY_CODE or color not in VALID_TILE_INTS:
            raise ValueError("Invalid color code")

        color = color_of(color)
        # Remove all tiles of the color, do not remove the terminals. Also remove the head if it exists
        for c in self._generate_board_coords():
            tile = self._tile(c)
            if (is_body(tile) or is_head(tile)) and color_of(tile) == color:
                self._place(c, EMPTY_CODE)

        # Reset the head position for this color
        self._heads[color] = None
        return

    def reset_board(self) -> None:
        """Reset the board to just the terminals."""
        for color in range(1, MAX_COLORS + 1):
            self.reset_color(color)  # Reset all colors

    def attempt_move(self, c: Coordinate, color: int) -> bool:
        """Place next segment of *color* at *c* if legal, updating heads and connecting terminals (also resetting interferences)."""
        color = color_of(color)

        if (
            color not in VALID_TILE_INTS
            or color == EMPTY_CODE
            or color not in range(1, MAX_COLORS + 1)
        ):
            raise ValueError("Invalid color code")

        if not self.is_legal_move(c, color):
            return False

        # If this new coordinate contains a different color, reset that color
        existing_tile = self._tile(c)
        if not is_empty(existing_tile) and color_of(existing_tile) != color:
            existing_color = color_of(existing_tile)
            self.reset_color(existing_color)

        head_pos = self._heads.get(color, None)
        if head_pos is None:
            # Place the first segment of a new path as a head
            self._place(c, head(color))
        else:
            # demote old head to body
            self._place(head_pos, body(color))
        # place new head
        self._place(c, head(color))
        self._heads[color] = c

        # Check if head connects two terminals
        adj_terminals = [
            n
            for n in c.neighbors4()
            if self._in_bounds(n)
            and is_terminal(self._tile(n))
            and color_of(self._tile(n)) == color
        ]
        if len(adj_terminals) == 2:
            # If the head connects two terminals, remove the head and mark color as solved
            self._place(c, body(color))
            self._heads[color] = None
            return True

        # check closure into opposite terminal iff this is not the terminal that we started from (i.e. not the first segment and no same color body)
        # get the positions of the two terminals for this color
        term1, term2 = self._terminals[color]

        # Theoretical board
        t_b = np.copy(self._board)
        t_b[c.row, c.col] = head(color)  # Place the head temporarily
        if self._connected(term1, term2, color, t_b):
            # Place as a body segment and remove the head
            self._place(c, body(color))
            self._heads[color] = None
            return True

        # Everything passes
        return True

    def _connected(
        self, a: Coordinate, b: Coordinate, color: int, board: np.ndarray | None = None
    ) -> bool:
        """Return ``True`` iff *a* and *b* are connected by tiles of *color*."""
        q: deque[Coordinate] = deque([a])
        seen: set[Coordinate] = {a}

        if board is None:
            board = self._board

        while q:
            cur = q.popleft()
            if cur == b:
                return True
            for n in cur.neighbors4():
                if self._in_bounds(n) and n not in seen and color_of(self._tile(n)) == color:
                    seen.add(n)
                    q.append(n)
        return False

    def is_color_solved(self, color: int) -> bool:
        """Return ``True`` iff color *color*'s terminals are connected."""
        a, b = self._terminals[color]
        return self._heads[color] is None and self._connected(a, b, color)

    def is_solved(self) -> bool:
        """Return ``True`` iff every color is solved and no empties remain."""
        head_vals = set(range(HEAD_OFFSET + 1, HEAD_OFFSET + MAX_COLORS + 1))
        return all(self.is_color_solved(c) for c in self._terminals) and all(
            item not in head_vals and item != 0 for item in self._board.flatten()
        )

    def _generate_board_coords(self) -> Iterator[Coordinate]:
        """Yield all coordinates on the board."""
        for r in range(self.rows):
            for c in range(self.cols):
                yield Coordinate(r, c)

    @property
    def board(self) -> np.ndarray:
        """Return a *copy* of the current board array."""
        return self._board.copy()

    @classmethod
    def is_valid_board(cls, arr: np.ndarray) -> bool:
        """Return ``True`` iff *arr* is a syntactically valid Flow-Free board."""
        # ───────────────────────────── basic shape / dtype / values ─────────────────────────────
        if (
            not isinstance(arr, np.ndarray)
            or arr.dtype != np.int8
            or arr.ndim != 2
            or not {int(v) for v in arr.flat}.issubset(VALID_TILE_INTS)
        ):
            return False

        n_rows, n_cols = arr.shape

        def in_bounds(c: Coordinate) -> bool:
            """True iff *c* is inside the array."""
            return 0 <= c.row < n_rows and 0 <= c.col < n_cols

        # ───────────────────────────── first pass: BFS over every colour ─────────────────────────
        visited: set[Coordinate] = set()
        # per-colour bookkeeping
        col_components: dict[
            int, list[dict[str, int]]
        ] = {}  # each dict: {'heads', 'terms', 'bodies'}

        for r in range(n_rows):
            for c in range(n_cols):
                start = Coordinate(r, c)
                v = int(arr[r, c])
                if is_empty(v) or start in visited:
                    continue

                colour = color_of(v)
                comp = {"heads": 0, "terms": 0, "bodies": 0}

                # BFS for this component
                q: deque[Coordinate] = deque([start])
                while q:
                    cur = q.popleft()
                    if cur in visited:
                        continue
                    visited.add(cur)

                    cur_val = int(arr[cur.row, cur.col])
                    if is_head(cur_val):
                        comp["heads"] += 1
                    elif is_terminal(cur_val):
                        comp["terms"] += 1
                    else:  # body segment
                        comp["bodies"] += 1

                    for n in cur.neighbors4():
                        if (
                            in_bounds(n)
                            and n not in visited
                            and color_of(int(arr[n.row, n.col])) == colour
                        ):
                            q.append(n)

                # store
                col_components.setdefault(colour, []).append(comp)

        # ───────────────────────────── second pass: per-colour rules ─────────────────────────────
        for comps in col_components.values():
            heads_total = sum(c["heads"] for c in comps)
            bodies_total = sum(c["bodies"] for c in comps)
            terminals_total = sum(c["terms"] for c in comps)  # should be 2 by design

            # 1) quick structural rejects
            if heads_total > 1 or terminals_total != 2:
                return False
            # body-only island?
            if any(c["bodies"] and c["heads"] == 0 and c["terms"] == 0 for c in comps):
                return False

            # 2) connectivity of the two terminals
            terminals_connected = any(c["terms"] == 2 for c in comps)

            if bodies_total == 0:
                # only valid start-state: two isolated terminals, no body, no head
                if heads_total != 0:
                    return False
                continue

            # bodies present from here on
            if terminals_connected:
                # finished path - must have NO head
                if heads_total != 0:
                    return False
            else:
                # unfinished path - must have EXACTLY ONE head
                if heads_total != 1:
                    return False

        # If every colour passes the checks the board is syntactically valid
        return True

    @classmethod
    def from_board(cls, arr: np.ndarray) -> FlowFree:
        """Create a game instance from a saved numpy ``int8`` board."""
        if not cls.is_valid_board(arr):
            raise ValueError("Invalid Flow Free board array")
        # gather terminals
        terms: dict[int, list[Coordinate]] = {}
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v = int(arr[r, c])
                if is_terminal(v):
                    terms.setdefault(color_of(v), []).append(Coordinate(r, c))
        starting = {k: (v[0], v[1]) for k, v in terms.items()}
        game = cls(arr.shape[0], arr.shape[1], starting)
        game._board = arr.copy()
        # rebuild head positions
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v = int(arr[r, c])
                if is_head(v):
                    game._heads[color_of(v)] = Coordinate(r, c)
        return game

    def get_internal_board(self) -> np.ndarray:
        """Return the internal board representation."""
        return self._board.copy()

    def board_str(self) -> str:
        """Pretty string view of the board with borders."""

        def cell(token: str) -> str:
            """Pad/align each cell to width 3."""
            return f"{token:>3}"

        top_bottom = "+" + "-" * (self.cols * 4 - 1) + "+"  # 3 chars + 1 space per cell
        rows_out: list[str] = [top_bottom]

        for r in range(self.rows):
            pieces: list[str] = []
            for c in range(self.cols):
                v = self._tile(Coordinate(r, c))
                if is_empty(v):
                    pieces.append(cell("."))
                elif is_body(v):
                    pieces.append(cell(str(color_of(v))))
                elif is_terminal(v):
                    pieces.append(cell(f"{color_of(v)}T"))
                elif is_head(v):
                    pieces.append(cell(f"{color_of(v)}H"))
                else:  # should never occur
                    raise ValueError("Invalid tile code in board.")
            rows_out.append("|" + " ".join(pieces) + "|")

        rows_out.append(top_bottom)
        return "\n".join(rows_out)

    def __str__(self) -> str:
        """Return a string representation of the board."""
        return self.board_str()

    def __repr__(self) -> str:
        """Return a string representation of the FlowFree instance."""
        return f"FlowFree(rows={self.rows}, cols={self.cols})"
