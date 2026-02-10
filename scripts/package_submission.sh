#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ZIP="${1:-$ROOT_DIR/submission_clean.zip}"
STAGE_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo "Preparing clean submission package..."
echo "Repo root: $ROOT_DIR"
echo "Output: $OUTPUT_ZIP"

rsync -a \
  --exclude=".git/" \
  --exclude=".venv/" \
  --exclude=".idea/" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  --exclude=".DS_Store" \
  --exclude="data/raw/" \
  --exclude="data/processed/" \
  --exclude="reports/episodes/" \
  --exclude="reports/tables/" \
  --exclude="reports/figures/" \
  "$ROOT_DIR/" "$STAGE_DIR/repo/"

(cd "$STAGE_DIR/repo" && zip -rq "$OUTPUT_ZIP" .)

echo "Created clean package: $OUTPUT_ZIP"
