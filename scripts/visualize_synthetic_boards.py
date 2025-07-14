"""Script to visualize the solution of a Flow Free puzzle."""

from pathlib import Path

from flowzero_src.flowfree.game import FlowFree
from flowzero_src.flowfree.solver import FlowFreeSATSolver
from flowzero_src.util.save_util import import_ndarray

BASE = Path(__file__).resolve().parent.parent

SYNTH_PATH = BASE / "flowzero_src" / "data" / "synthetic"
SYNTH_BOARDS = [(import_ndarray(file), file.name) for file in SYNTH_PATH.glob("*.npy")]


def visualize_board(board: FlowFree) -> None:
    """Visualize a Flow Free board and its solution."""
    board.display()
    solver = FlowFreeSATSolver(board)
    print("\n\n")
    solver.solve().display()


if __name__ == "__main__":
    for board, name in SYNTH_BOARDS:
        print(name)
        ff = FlowFree.from_board(board)
        visualize_board(ff)
        print("\n" + "=" * 40 + "\n")
