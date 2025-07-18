"""Utilities for saving and loading files."""

from pathlib import Path

import numpy as np


def export_ndarray(arr: np.ndarray, filepath: str | Path, compressed: bool = False) -> None:
    """
    Save a NumPy array to disk.

    If compressed is False, saves as .npy using numpy.save.
    If compressed is True, saves as .npz using numpy.savez_compressed
    under the key 'arr'.

    Args:
        arr:      The array to save.
        filepath: Path or filename (can include or omit extension).
        compressed: Whether to use .npz compression.
    """
    path = Path(filepath)
    if compressed:
        path = path.with_suffix(".npz")
        np.savez_compressed(path, arr=arr)
    else:
        path = path.with_suffix(".npy")
        np.save(path, arr)


def import_ndarray(filepath: str | Path) -> np.ndarray:
    """
    Load a NumPy array from disk, handling both .npy and .npz formats.

    Args:
        filepath: Path or filename to load.

    Returns:
        The loaded NumPy array.
    """
    path = Path(filepath)
    if path.suffix == ".npz":
        data = np.load(path)
        # assumes saved under key 'arr'
        return data["arr"]
    else:
        return np.load(path)
