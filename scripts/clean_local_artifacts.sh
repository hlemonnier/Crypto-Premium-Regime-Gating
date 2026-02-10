#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Cleaning local temporary artifacts under: $ROOT_DIR"

find "$ROOT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$ROOT_DIR" -type d -name ".pytest_cache" -prune -exec rm -rf {} +
find "$ROOT_DIR" -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
find "$ROOT_DIR" -type f -name "*.pyc" -delete
find "$ROOT_DIR" -type f -name ".DS_Store" -delete

echo "Done."
