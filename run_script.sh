#!/usr/bin/env bash

# Resolve this script’s directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure ’scripts’ is a package
if [ ! -d "./scripts" ] || [ ! -f "./scripts/__init__.py" ]; then
  echo "Error: ./scripts/ is not a Python package. Ensure scripts/__init__.py exists."
  exit 1
fi

# Find all .py files under ./scripts/ (excluding __init__.py), then strip directory
files=($(find ./scripts -maxdepth 1 -type f -name "*.py" ! -name "__init__.py" -exec basename {} \;))

if [ ${#files[@]} -eq 0 ]; then
  echo "No Python files found in ./scripts/."
  exit 1
fi

echo "Select a script to run:"
PS3="Enter number: "
select fname in "${files[@]}"; do
  if [ -n "$fname" ]; then
    mod="${fname%.py}"
    python -m scripts."$mod"
    exit $?
  else
    echo "Invalid selection. Try again."
  fi
done
