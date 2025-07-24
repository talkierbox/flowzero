"""Script to quickly delete invalid synthetic boards. This should not need to be run."""

from pathlib import Path

from flowzero_src.flowfree.game import FlowFree
from flowzero_src.util.save_util import import_ndarray

BASE = Path(__file__).resolve().parent.parent

SYNTH_PATH = BASE / "flowzero_src" / "data" / "synthetic"
SYNTH_BOARDS = [(import_ndarray(file), file.name) for file in SYNTH_PATH.glob("*.npy")]


if __name__ == "__main__":
    del_count = 0
    tot = 0
    for board, name in SYNTH_BOARDS:
        if not FlowFree.is_valid_board(board):
            # Delete the file
            file_path = SYNTH_PATH / name
            print(f"Deleting invalid board: {name}")
            file_path.unlink()
            del_count += 1
        tot += 1

    print(f"Total boards checked: {tot}")
    print(f"Deleted {del_count} invalid synthetic boards.")
