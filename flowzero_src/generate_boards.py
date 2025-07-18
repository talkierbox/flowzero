"""Generate Flow Free puzzles using CLI. WARNING: Do not seed the random number generator."""

import argparse
import heapq
import itertools
import random
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from flowfree.game import Coordinate, FlowFree
from flowfree.solver import FlowFreeSATSolver
from util.config import get_key
from util.save_util import export_ndarray

# Base output directory and defaults
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(get_key("generation.output.dir", "./puzzles"))


def generate_hash() -> str:
    """Generate a unique timestamp string."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Flow Free puzzles.")
    parser.add_argument(
        "-r", "--rows", type=int, required=True, help="Number of rows in each puzzle"
    )
    parser.add_argument(
        "-c", "--cols", type=int, required=True, help="Number of columns in each puzzle"
    )
    parser.add_argument(
        "-n", "--num-puzzles", type=int, required=True, help="Total number of puzzles to generate"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=cpu_count(), help="Number of parallel worker processes"
    )
    parser.add_argument(
        "-t", "--terminals", type=int, default=5, help="Maximum terminal pairs per puzzle"
    )
    return parser.parse_args()


def carve_path(
    rows: int, cols: int, start: Coordinate, end: Coordinate, occupied: set[Coordinate]
) -> list[Coordinate]:
    """Carve a path between two points using A* search."""
    open_set = []
    heapq.heappush(open_set, (start.manhattan(end), 0, start))
    came_from: dict[Coordinate, Coordinate] = {}
    g_score: dict[Coordinate, int] = {start: 0}
    visited: set[Coordinate] = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current == end:
            # reconstruct path
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
            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f_score = tentative_g + nbr.manhattan(end)
                heapq.heappush(open_set, (f_score, tentative_g, nbr))

    return []


def generate_one(args: object) -> dict[int, tuple[Coordinate, Coordinate]] | None:
    """Generate one puzzle by carving paths algorithmically."""
    rows, cols = args.rows, args.cols
    cells = [Coordinate(r, c) for r in range(rows) for c in range(cols)]
    pts = random.sample(cells, 2 * args.terminals)
    random.shuffle(pts)
    pairs = [(pts[i], pts[i + args.terminals]) for i in range(args.terminals)]
    pairs.sort(key=lambda ab: ab[0].manhattan(ab[1]), reverse=True)

    occupied: set[Coordinate] = set()
    for a, b in pairs:
        path = carve_path(rows, cols, a, b, occupied)
        if not path:
            return None

    return dict(enumerate(pairs, 1))


def generate_one_stochastic(args: object) -> dict[int, tuple[Coordinate, Coordinate]] | None:
    """Generate one puzzle by random terminals plus SAT solvability check."""
    rows, cols = args.rows, args.cols
    num_pairs = random.randint(cols // 2, args.terminals)  # noqa: S311

    # avoid boards that are too dense
    if num_pairs * 2 >= (rows * cols) - 3:
        return None

    cells = [Coordinate(r, c) for r in range(rows) for c in range(cols)]
    pts = random.sample(cells, 2 * num_pairs)
    terminals = {i: (pts[i], pts[num_pairs + i]) for i in range(num_pairs)}

    try:
        solver = FlowFreeSATSolver(FlowFree(rows, cols, terminals))
    except Exception:
        return None

    if not solver.is_solvable():
        return None

    return terminals


def main() -> None:
    """Main function to generate Flow Free puzzles."""
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, cols = args.rows, args.cols
    num_puzzles, workers = args.num_puzzles, args.workers

    method = get_key("generation.method", "algorithmic")
    gen = generate_one_stochastic if method == "stochastic" else generate_one

    pbar = tqdm(total=num_puzzles, desc="Generating puzzles", unit="puzzle", ncols=100)
    chunksize = max(1, num_puzzles // (workers * 4))
    saved = 0

    with Pool(workers) as pool:
        for result in pool.imap_unordered(gen, itertools.repeat(args), chunksize=chunksize):
            if not result:
                continue

            saved += 1
            game = FlowFree(rows, cols, result)
            fname = (
                f"synth_puzzle_{generate_hash()}_"
                f"{saved:0{len(str(num_puzzles))}d}_"
                f"{rows}x{cols}.txt"
            )
            out_path = OUTPUT_DIR / fname
            export_ndarray(game.get_internal_board(), out_path, compressed=False)

            pbar.update(1)
            pbar.set_postfix(saved=saved)

            if saved >= num_puzzles:
                pool.terminate()  # stop all workers
                break

    pbar.close()
    print(f"Done: {saved} puzzles → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
