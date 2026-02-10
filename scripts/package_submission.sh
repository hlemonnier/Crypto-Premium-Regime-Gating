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

mkdir -p "$STAGE_DIR/repo"
rm -f "$OUTPUT_ZIP"

copy_if_exists() {
  local path="$1"
  if [[ -e "$ROOT_DIR/$path" ]]; then
    local parent
    parent="$(dirname "$path")"
    mkdir -p "$STAGE_DIR/repo/$parent"
    rsync -a "$ROOT_DIR/$path" "$STAGE_DIR/repo/$parent/"
  fi
}

# Allowlist payload for recruiter-facing package.
copy_if_exists "src"
copy_if_exists "configs"
copy_if_exists "notebooks"
copy_if_exists "tests"
copy_if_exists "scripts"
copy_if_exists "reports/final"
copy_if_exists "README.md"
copy_if_exists "requirements.txt"
copy_if_exists ".gitignore"
copy_if_exists "AGENT.md"
copy_if_exists "Notice.pdf"
copy_if_exists "Notice + Hawkes.pdf"

# Final scrub in staging.
find "$STAGE_DIR/repo" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$STAGE_DIR/repo" -type d -name ".pytest_cache" -prune -exec rm -rf {} +
find "$STAGE_DIR/repo" -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
find "$STAGE_DIR/repo" -type f -name "*.pyc" -delete
find "$STAGE_DIR/repo" -type f -name ".DS_Store" -delete

(cd "$STAGE_DIR/repo" && zip -rq "$OUTPUT_ZIP" .)

echo "Created clean package: $OUTPUT_ZIP"
