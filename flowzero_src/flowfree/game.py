from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

MAX_COLORS: int = 15  # standard Flow Free palette
HEAD_OFFSET: int = 32  # codes from 33 to 47 mark head positions

EMPTY_CODE: np.int8 = np.int8(0)


def body(c: int) -> np.int8:
    """Return the int8 code for a body segment of the given color."""
    return np.int8(c)


def terminal(c: int) -> np.int8:
    """Return the int8 code for a terminal of the given color."""
    return np.int8(-c)


def head(c: int) -> np.int8:
    """Return the int8 code for the head (current frontier) of the given color."""
    return np.int8(HEAD_OFFSET + c)


def is_empty(v: int) -> bool:
    """Return True if v represents an empty tile."""
    return v == 0


def is_body(v: int) -> bool:
    """Return True if v represents a body segment."""
    return 0 < v <= MAX_COLORS


def is_terminal(v: int) -> bool:
    """Return True if v represents a terminal."""
    return -MAX_COLORS <= v < 0


def is_head(v: int) -> bool:
    """Return True if v represents a head segment."""
    return HEAD_OFFSET < v <= HEAD_OFFSET + MAX_COLORS


def color_of(v: int) -> int:
    """Return the color index (1 to 15) for the tile code v."""
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
    """Immutable grid coordinate (row, col)."""

    row: int
    col: int

    def neighbors4(self) -> list[Coordinate]:
        """Return the four adjacent coordinates (up, down, left, right)."""
        return [
            Coordinate(self.row - 1, self.col),
            Coordinate(self.row + 1, self.col),
            Coordinate(self.row, self.col - 1),
            Coordinate(self.row, self.col + 1),
        ]

    def manhattan(self, other: Coordinate) -> int:
        """Return the Manhattan distance to another coordinate."""
        return abs(self.row - other.row) + abs(self.col - other.col)

    def __add__(self, other: Coordinate) -> Coordinate:
        """Return a new coordinate that is the sum of this and other."""
        return Coordinate(self.row + other.row, self.col + other.col)

    def __subtract__(self, other: Coordinate) -> Coordinate:
        """Return a new coordinate that is the difference of this and other."""
        return Coordinate(self.row - other.row, self.col - other.col)


def parse_ascii_board(board_str: str) -> dict[int, tuple[Coordinate, Coordinate]]:
    """Convert an ASCII board string into a mapping from color IDs to their terminal coordinates."""
    rows: list[str] = [
        line.rstrip()  # keep any leading spaces
        for line in board_str.strip("\n").splitlines()
        if line.strip()
    ]

    if any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("All rows must have the same length")

    # assign each new symbol a unique color ID
    symbol_to_id: dict[str, int] = {}
    next_id = 1

    occurrences: dict[int, list[Coordinate]] = defaultdict(list)

    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            cid = symbol_to_id.setdefault(ch, next_id)
            if cid == next_id:
                next_id += 1
            occurrences[cid].append(Coordinate(r, c))

    # convert occurrences into terminal pairs and check counts
    terminals: dict[int, tuple[Coordinate, Coordinate]] = {}
    for cid, coords in occurrences.items():
        if len(coords) != 2:
            sym = next(k for k, v in symbol_to_id.items() if v == cid)
            raise ValueError(
                f"Symbol '{sym}' appears {len(coords)} times (should appear exactly twice)"
            )
        terminals[cid] = (coords[0], coords[1])

    return terminals


class FlowFree:
    """Mutable game state for a Flow Free board."""

    def __init__(
        self,
        rows: int,
        cols: int,
        terminals: dict[int, tuple[Coordinate, Coordinate]],
    ) -> None:
        """Create a new board with given terminals mapping color to terminal coordinates."""
        self.rows, self.cols = rows, cols
        self._board: np.ndarray = np.zeros((rows, cols), dtype=np.int8)
        self._heads: dict[int, Coordinate | None] = dict.fromkeys(terminals)
        self._terminals = terminals

        # place the terminal codes on the board
        for color, (a, b) in terminals.items():
            self._place(a, terminal(color))
            self._place(b, terminal(color))

    def _in_bounds(self, c: Coordinate) -> bool:
        """Return True if coordinate c is inside the board bounds."""
        return 0 <= c.row < self.rows and 0 <= c.col < self.cols

    def _tile(self, c: Coordinate) -> int:
        """Return the raw int8 code at coordinate c."""
        if not self._in_bounds(c):
            raise ValueError("Coordinate out of bounds")
        return int(self._board[c.row, c.col])

    def _place(self, c: Coordinate, code: np.int8) -> None:
        """Write code to the board at coordinate c without validation."""
        self._board[c.row, c.col] = code

    def is_legal_move(self, c: Coordinate, color: int) -> bool:
        """Return True if placing the next segment of the given color at c is allowed."""
        if not self._in_bounds(c) or not is_empty(self._tile(c)):
            return False

        if color_of(self._tile(c)) == color or is_terminal(self._tile(c)):
            # cannot place on a tile of the same color or on a terminal
            return False

        if color == terminal(color) or color == EMPTY_CODE:
            # cannot place a terminal or empty tile here
            return False

        if color not in VALID_TILE_INTS:
            raise ValueError("Invalid color code")

        head_pos = self._heads.get(color)
        if head_pos is None:
            # new path must touch one of its terminals
            return any(
                self._in_bounds(n)
                and is_terminal(self._tile(n))
                and color_of(self._tile(n)) == color
                for n in c.neighbors4()
            )
        # move must extend from current head position
        return c.manhattan(head_pos) == 1

    def reset_color(self, color: int) -> None:
        """Clear all path segments of a specific color but leave the terminals."""
        if color == EMPTY_CODE or color not in VALID_TILE_INTS:
            raise ValueError("Invalid color code")

        color = color_of(color)
        # remove all body and head tiles for this color
        for coord in self._generate_board_coords():
            tile = self._tile(coord)
            if (is_body(tile) or is_head(tile)) and color_of(tile) == color:
                self._place(coord, EMPTY_CODE)

        # clear the head tracking for this color
        self._heads[color] = None

    def reset_board(self) -> None:
        """Clear all paths and return the board to only the terminals."""
        for color in range(1, MAX_COLORS + 1):
            self.reset_color(color)

    def attempt_move(self, c: Coordinate, color: int) -> bool:
        """Try placing the next segment of the given color at c. Returns True if the move was applied (including completing a path)."""
        if self.is_solved():
            return False

        color = color_of(color)
        if (
            color not in VALID_TILE_INTS
            or color == EMPTY_CODE
            or color not in range(1, MAX_COLORS + 1)
        ):
            raise ValueError("Invalid color code")

        if not self.is_legal_move(c, color):
            return False

        # if this spot has another color's path, clear that path first
        existing_tile = self._tile(c)
        if not is_empty(existing_tile) and color_of(existing_tile) != color:
            self.reset_color(color_of(existing_tile))

        head_pos = self._heads.get(color)
        if head_pos is None:
            # first segment of a new path becomes a head
            self._place(c, head(color))
        else:
            # turn the old head into a regular body
            self._place(head_pos, body(color))
        # place the new head and track it
        self._place(c, head(color))
        self._heads[color] = c

        # check if this head now touches two terminals (completing the path)
        adj_terminals = [
            n
            for n in c.neighbors4()
            if self._in_bounds(n)
            and is_terminal(self._tile(n))
            and color_of(self._tile(n)) == color
        ]
        if len(adj_terminals) == 2:
            # complete the path and clear head tracking
            self._place(c, body(color))
            self._heads[color] = None
            return True

        # check if placing here closes the path between the two terminals
        term1, term2 = self._terminals[color]
        temp_board = np.copy(self._board)
        temp_board[c.row, c.col] = head(color)
        if self._connected(term1, term2, color, temp_board):
            self._place(c, body(color))
            self._heads[color] = None
            return True

        return True

    def _connected(
        self, a: Coordinate, b: Coordinate, color: int, board: np.ndarray | None = None
    ) -> bool:
        """Return True if coordinates a and b are connected by tiles of the given color."""
        queue: deque[Coordinate] = deque([a])
        seen: set[Coordinate] = {a}

        if board is None:
            board = self._board

        while queue:
            current = queue.popleft()
            if current == b:
                return True
            for n in current.neighbors4():
                if self._in_bounds(n) and n not in seen and color_of(self._tile(n)) == color:
                    seen.add(n)
                    queue.append(n)
        return False

    def is_color_solved(self, color: int) -> bool:
        """Return True if the two terminals for the given color are connected."""
        t1, t2 = self._terminals[color]
        return self._heads[color] is None and self._connected(t1, t2, color)

    def is_solved(self) -> bool:
        """Return True if every color path is complete and there are no empty tiles."""
        head_codes = set(range(HEAD_OFFSET + 1, HEAD_OFFSET + MAX_COLORS + 1))
        return all(self.is_color_solved(c) for c in self._terminals) and all(
            item not in head_codes and item != 0 for item in self._board.flat
        )

    def _generate_board_coords(self) -> Iterator[Coordinate]:
        """Yield every coordinate on the board in row-major order."""
        for r in range(self.rows):
            for c in range(self.cols):
                yield Coordinate(r, c)

    @property
    def board(self) -> np.ndarray:
        """Return a copy of the current board array."""
        return self._board.copy()

    @classmethod
    def from_ascii_board(cls, data: str) -> FlowFree:
        """Create a FlowFree instance from an ASCII board string."""
        lines = [line for line in data.strip().splitlines() if line.strip()]
        rows, cols = len(lines), len(lines[0])
        return FlowFree(rows, cols, parse_ascii_board(data))

    @classmethod
    def is_valid_board(cls, arr: np.ndarray) -> bool:
        """Return True if arr is a valid Flow Free board."""
        # check basic shape, dtype and valid tile codes
        if (
            not isinstance(arr, np.ndarray)
            or arr.dtype != np.int8
            or arr.ndim != 2
            or not {int(v) for v in arr.flat}.issubset(VALID_TILE_INTS)
        ):
            return False

        n_rows, n_cols = arr.shape

        def in_bounds(c: Coordinate) -> bool:
            """Return True if coordinate c is inside arr bounds."""
            return 0 <= c.row < n_rows and 0 <= c.col < n_cols

        visited: set[Coordinate] = set()
        components: dict[int, list[dict[str, int]]] = {}

        for r in range(n_rows):
            for c in range(n_cols):
                start = Coordinate(r, c)
                val = int(arr[r, c])
                if is_empty(val) or start in visited:
                    continue

                color = color_of(val)
                stats = {"heads": 0, "terms": 0, "bodies": 0}
                queue: deque[Coordinate] = deque([start])

                while queue:
                    cur = queue.popleft()
                    if cur in visited:
                        continue
                    visited.add(cur)

                    tile_val = int(arr[cur.row, cur.col])
                    if is_head(tile_val):
                        stats["heads"] += 1
                    elif is_terminal(tile_val):
                        stats["terms"] += 1
                    else:
                        stats["bodies"] += 1

                    for n in cur.neighbors4():
                        if (
                            in_bounds(n)
                            and n not in visited
                            and color_of(int(arr[n.row, n.col])) == color
                        ):
                            queue.append(n)

                components.setdefault(color, []).append(stats)

        for comps in components.values():
            total_heads = sum(c["heads"] for c in comps)
            total_terms = sum(c["terms"] for c in comps)
            total_bodies = sum(c["bodies"] for c in comps)

            # each color must have exactly two terminals and at most one head
            if total_terms != 2 or total_heads > 1:
                return False
            # no isolated body without a head or terminals
            if any(s["bodies"] and s["heads"] == 0 and s["terms"] == 0 for s in comps):
                return False

            path_closed = any(s["terms"] == 2 for s in comps)
            if total_bodies > 0:
                if path_closed:
                    # finished path should have no head
                    if total_heads != 0:
                        return False
                else:
                    # unfinished path should have exactly one head
                    if total_heads != 1:
                        return False

        return True

    @classmethod
    def from_board(cls, arr: np.ndarray) -> FlowFree:
        """Create a game instance from a saved numpy int8 board."""
        if not cls.is_valid_board(arr):
            raise ValueError("Invalid Flow Free board array")
        terms: dict[int, list[Coordinate]] = {}
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v = int(arr[r, c])
                if is_terminal(v):
                    terms.setdefault(color_of(v), []).append(Coordinate(r, c))
        starting = {k: (v[0], v[1]) for k, v in terms.items()}
        game = cls(arr.shape[0], arr.shape[1], starting)
        game._board = arr.copy()
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v = int(arr[r, c])
                if is_head(v):
                    game._heads[color_of(v)] = Coordinate(r, c)
        return game

    def get_internal_board(self) -> np.ndarray:
        """Return the internal board array after verifying it is valid."""
        assert self.is_valid_board(self._board), (
            "Internal board state is invalid. Something went wrong!"
        )
        return self._board.copy()

    def board_str(self) -> str:
        """Return a string view of the board with borders."""

        def cell(token: str) -> str:
            """Pad each cell token to width 3 for alignment."""
            return f"{token:>3}"

        top_line = "+" + "-" * (self.cols * 4 - 1) + "+"
        lines = [top_line]

        for r in range(self.rows):
            row_cells: list[str] = []
            for c in range(self.cols):
                v = self._tile(Coordinate(r, c))
                if is_empty(v):
                    row_cells.append(cell("."))
                elif is_body(v):
                    row_cells.append(cell(str(color_of(v))))
                elif is_terminal(v):
                    row_cells.append(cell(f"{color_of(v)}T"))
                elif is_head(v):
                    row_cells.append(cell(f"{color_of(v)}H"))
                else:
                    raise ValueError("Invalid tile code on board")
            lines.append("|" + " ".join(row_cells) + "|")

        lines.append(top_line)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a string representation of the board."""
        return self.board_str()

    def __repr__(self) -> str:
        """Return a string representation of the FlowFree instance."""
        return f"FlowFree(rows={self.rows}, cols={self.cols})"
