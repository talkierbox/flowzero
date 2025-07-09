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

import numpy as np

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


def colour_of(v: int) -> int:
    """Return the color index (1…15) associated with tile code *v*."""
    if is_body(v):
        return v
    if is_terminal(v):
        return -v
    if is_head(v):
        return v - HEAD_OFFSET
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
        """Create a fresh board with given *terminals* mapping colour→(t1, t2)."""
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
        head_pos = self._heads[color]
        if head_pos is None:
            # must touch one of its terminals
            return any(
                self._in_bounds(n)
                and is_terminal(self._tile(n))
                and colour_of(self._tile(n)) == color
                for n in c.neighbors4()
            )
        # must extend from current head
        return c.manhattan(head_pos) == 1

    def attempt_move(self, c: Coordinate, color: int) -> bool:
        """Place next segment of *color* at *c* if legal, updating heads."""
        if not self.is_legal_move(c, color):
            return False
        head_pos = self._heads[color]
        if head_pos is None:
            # starting: find the terminal touched and mark it as body
            for n in c.neighbors4():
                if (
                    self._in_bounds(n)
                    and is_terminal(self._tile(n))
                    and colour_of(self._tile(n)) == color
                ):
                    self._place(n, body(color))
                    break
        else:
            # demote old head to body
            self._place(head_pos, body(color))
        # place new head
        self._place(c, head(color))
        self._heads[color] = c

        # check closure into opposite terminal
        for n in c.neighbors4():
            if (
                self._in_bounds(n)
                and is_terminal(self._tile(n))
                and colour_of(self._tile(n)) == color
            ):
                # close path: demote both and clear head
                self._place(n, body(color))
                self._place(c, body(color))
                self._heads[color] = None
                break
        return True

    def _connected(self, a: Coordinate, b: Coordinate, color: int) -> bool:
        """Return ``True`` iff *a* and *b* are connected by tiles of *color*."""
        q: deque[Coordinate] = deque([a])
        seen: set[Coordinate] = {a}
        while q:
            cur = q.popleft()
            if cur == b:
                return True
            for n in cur.neighbors4():
                if self._in_bounds(n) and n not in seen and colour_of(self._tile(n)) == color:
                    seen.add(n)
                    q.append(n)
        return False

    def is_colour_solved(self, color: int) -> bool:
        """Return ``True`` iff color *color*'s terminals are connected."""
        a, b = self._terminals[color]
        return self._heads[color] is None and self._connected(a, b, color)

    def is_solved(self) -> bool:
        """Return ``True`` iff every color is solved and no empties remain."""
        return all(self.is_colour_solved(c) for c in self._terminals) and not np.any(
            self._board == EMPTY_CODE
        )

    @property
    def board(self) -> np.ndarray:
        """Return a *copy* of the current board array."""
        return self._board.copy()

    @classmethod
    def is_valid_board(cls, arr: np.ndarray) -> bool:
        """Return ``True`` iff *arr* is a syntactically valid Flow Free board."""
        if not isinstance(arr, np.ndarray) or arr.dtype != np.int8 or arr.ndim != 2:
            return False
        if not {int(v) for v in arr.flatten()}.issubset(VALID_TILE_INTS):
            return False
        # terminals: exactly two per color present
        counts: dict[int, int] = {}
        for v in arr.flatten():
            if is_terminal(int(v)):
                c = colour_of(int(v))
                counts[c] = counts.get(c, 0) + 1
        return all(v == 2 for v in counts.values())

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
                    terms.setdefault(colour_of(v), []).append(Coordinate(r, c))
        starting = {k: tuple(v) for k, v in terms.items()}
        game = cls(arr.shape[0], arr.shape[1], starting)
        game._board = arr.copy()
        # rebuild head positions
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v = int(arr[r, c])
                if is_head(v):
                    game._heads[colour_of(v)] = Coordinate(r, c)
        return game

    def get_internal_board(self) -> np.ndarray:
        """Return the internal board representation."""
        return self._board.copy()
