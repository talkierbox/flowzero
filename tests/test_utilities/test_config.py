"""Unit tests for flowzero_src.util.config helper module."""

from flowzero_src.util.config import get_key


def test_get_key():
    assert get_key("mcgs.sims_per_move") > 0, (
        "Expected to successfully read 'mcgs.sims_per_move' to be a positive integer."
    )
