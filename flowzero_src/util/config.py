"""Utility file for parsing yaml configuration file."""
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE.parent / "config.yaml"

config: dict = yaml.safe_load(CONFIG_PATH.read_text())

def get_key(key: str, default: str | None = None) -> str | None:
    """
    Get a configuration value by key. Search through nested keys using dot notation.

    Args:
        key (str): The key to look up in the configuration.
        default: The default value to return if the key is not found.

    Returns:
        The value associated with the key, or the default value if the key is not found.
    """
    keys = key.split('.')
    value = config
    for k in keys: 
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default  # Return default if key is not found
    return value

def print_config() -> None:
    """Print the entire configuration."""
    for key, value in config.items():
        print(f"{key}: {value}")
        
def is_verbose() -> bool:
    """
    Check if verbose logging is enabled.

    Returns:
        bool: True if verbose logging is enabled, False otherwise.
    """
    return get_key("logging.verbose", False)
        
if __name__ == "__main__":
    print_config()