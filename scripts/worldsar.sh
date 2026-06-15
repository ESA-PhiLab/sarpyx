#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/worldsar.sh PRODUCT_PATH [OUTPUT_DIR] [CUTS_DIR] [extra worldsar args...]

Defaults:
  OUTPUT_DIR  <PRODUCT_PATH parent>/output
  CUTS_DIR    <OUTPUT_DIR>/tiles
  DB_DIR      <OUTPUT_DIR>/db

Environment:
  CONDA_ENV   conda environment to activate (default: sarpyx-snap)
  DB_DIR      override database directory
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

PRODUCT_PATH="$1"
shift

INPUT_PARENT="$(cd "$(dirname "$PRODUCT_PATH")" && pwd)"
OUTPUT_DIR="${INPUT_PARENT}/output"
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  OUTPUT_DIR="$1"
  shift
fi

CUTS_DIR=""
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  CUTS_DIR="$1"
  shift
fi
if [[ -z "$CUTS_DIR" ]]; then
  CUTS_DIR="${OUTPUT_DIR}/tiles"
fi
DB_DIR="${DB_DIR:-${OUTPUT_DIR}/db}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sarpyx-snap}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
source "$SCRIPT_DIR/setvar.sh"
source "$SCRIPT_DIR/snap_userdir.sh"
worldsar_configure_snap_userdir "$(basename "$PRODUCT_PATH")" "$OUTPUT_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found after activating conda env: $CONDA_ENV" >&2
  exit 1
fi

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"
LOCK_TIMEOUT="${LOCK_TIMEOUT:-0}"

CMD=(
  python
  -m
  sarpyx.cli.worldsar
  --input "$PRODUCT_PATH"
  --output "$OUTPUT_DIR"
  --cuts-outdir "$CUTS_DIR"
  --db-dir "$DB_DIR"
  --lock-timeout "$LOCK_TIMEOUT"
)

CMD+=("$@")

echo "Conda env: $CONDA_ENV"
echo "SNAP GPT:  $GPT_PATH"
echo "SNAP user: $SNAP_USERDIR"
echo "SNAP seed: ${SNAP_USERDIR_SEED_MODE:-light}"
echo "Running:   ${CMD[*]}"
exec "${CMD[@]}"
