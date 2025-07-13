"""Unit tests for flowzero_src.util.config helper module."""

from flowzero_src.util.config import get_key, is_verbose


def test_get_key():
    assert get_key("logging.verbose") is True or get_key("logging.verbose") is False, (
        "Expected to successfully read 'logging.verbose' to be a boolean value."
    )


def test_is_verbose():
    assert is_verbose() is True or is_verbose() is False, (
        "Expected 'is_verbose' to return a boolean value."
    )
