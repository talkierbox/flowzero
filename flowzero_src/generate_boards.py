"""Generate Flow Free puzzles using CLI with A* carving and progress bars."""

import argparse
import heapq
import random
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from flowfree.game import Coordinate, FlowFree
from flowfree.solver import FlowFreeSATSolver
from util.config import get_key

# Config keys:
#   output.dir: directory for puzzles
#   generation.max_pairs: max terminal pairs per puzzle
#   generation.method: "stochastic" or "algorithmic" --- Algorithmic uses A* to carve paths while stochastic generates random pairs and checks solvability
BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(get_key("output.dir", "./puzzles"))
MAX_PAIRS = int(get_key("generation.max_pairs", 3))


def generate_hash() -> str:
    """Generate a unique hash based on the current datetime."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for puzzle generation."""
    parser = argparse.ArgumentParser(description="Generate Flow Free puzzles.")
    parser.add_argument("-r", "--rows", type=int, required=True)
    parser.add_argument("-c", "--cols", type=int, required=True)
    parser.add_argument("-n", "--num-puzzles", type=int, required=True)
    parser.add_argument("-w", "--workers", type=int, default=cpu_count())
    return parser.parse_args()


# TODO: Refactor common algorithms like BFS, DFS, and A* into a separate module
def carve_path(
    rows: int, cols: int, start: Coordinate, end: Coordinate, occupied: set[Coordinate]
) -> list[Coordinate]:
    """Carve a path between two coordinates using A* algorithm."""
    open_set = []  # heap of (f, g, coord)
    heapq.heappush(open_set, (start.manhattan(end), 0, start))
    came_from: dict[Coordinate, Coordinate] = {}
    g_score: dict[Coordinate, int] = {start: 0}
    visited: set[Coordinate] = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current == end:
            # reconstruct
            path = []
            node = end
            while node != start:
                path.append(node)
                node = came_from[node]
            path.reverse()
            for p in path:
                occupied.add(p)
            return path
        visited.add(current)
        for nbr in current.neighbors4():
            if not (0 <= nbr.row < rows and 0 <= nbr.col < cols):
                continue
            if nbr in occupied:
                continue
            tentative_g = g + 1
            if nbr in visited and tentative_g >= g_score.get(nbr, float("inf")):
                continue
            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f_score = tentative_g + nbr.manhattan(end)
                heapq.heappush(open_set, (f_score, tentative_g, nbr))
    return []


def generate_one(args: object) -> dict[int, tuple[Coordinate, Coordinate]] | None:
    """Generate a single Flow Free puzzle with random terminal pairs."""
    rows, cols = args.rows, args.cols
    cells = [Coordinate(r, c) for r in range(rows) for c in range(cols)]
    pts = random.sample(cells, 2 * MAX_PAIRS)
    # random pairing
    random.shuffle(pts)
    pairs = [(pts[i], pts[i + MAX_PAIRS]) for i in range(MAX_PAIRS)]
    # sort by distance descending to carve hardest first
    pairs.sort(key=lambda ab: ab[0].manhattan(ab[1]), reverse=True)

    occupied: set[Coordinate] = set()
    for a, b in pairs:
        path = carve_path(rows, cols, a, b, occupied)
        if not path:
            return None
    return dict(enumerate(pairs, 1))


def generate_one_stochastic(args: object) -> dict[int, tuple[Coordinate, Coordinate]] | None:
    """Generate a single Flow Free puzzle through generating random terminal pairs on boards and checking solvability."""
    # Choose a random number of terminal pairs from cols // 2 to cols + 3
    rows, cols = args.rows, args.cols
    num_pairs = random.randint(cols // 2, cols + 3)  # noqa: S311

    if num_pairs >= (rows * cols) - 3:
        return None

    # Randomly choose a location for each terminal pair
    cells = [Coordinate(r, c) for r in range(rows) for c in range(cols)]
    pts = random.sample(cells, 2 * num_pairs)

    # Create the terminal dict
    terminals = {i: (pts[i], pts[num_pairs + i]) for i in range(num_pairs)}

    try:
        solver = FlowFreeSATSolver(FlowFree(rows, cols, terminals))
    except Exception:
        return None  # Not solvablae or invalid board config

    if not solver.is_solvable():
        # print("Generated board is unsolvable.")
        return None

    return terminals


def main() -> None:
    """Main function to generate Flow Free puzzles."""
    args = parse_args()
    ROWS, COLS, NUM_PUZZLES, WORKERS = args.rows, args.cols, args.num_puzzles, args.workers  # noqa: N806

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pad = len(str(NUM_PUZZLES))

    puzzles = []

    if get_key("generation.method", "stochastic") == "stochastic":
        generate_one = generate_one_stochastic
    else:
        generate_one = generate_one
    print("Using generation method:", get_key("generation.method", "stochastic"))

    with tqdm(total=NUM_PUZZLES, desc="Generating puzzles", unit="puzzle") as pbar:
        i = 0
        while len(puzzles) < NUM_PUZZLES:
            need = NUM_PUZZLES - len(puzzles)
            with Pool(WORKERS) as pool:
                results = pool.map(generate_one, [args] * need)
                i += 1
            pbar.set_postfix({"attempts": i})
            good = [r for r in results if r]
            puzzles.extend(good)
            pbar.update(len(good))

    for i, terms in enumerate(tqdm(puzzles[:NUM_PUZZLES], desc="Writing puzzles", unit="file"), 1):
        game = FlowFree(ROWS, COLS, terms)
        out = (
            OUTPUT_DIR / f"synth_puzzle_{generate_hash()}_{i:0{pad}d}_{args.rows}x{args.cols}_.txt"
        )
        out.write_bytes(game.get_internal_board().tobytes())

    print(f"Done: {NUM_PUZZLES} puzzles -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
