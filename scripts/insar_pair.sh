#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/insar_pair.sh [--preflight]

Downloads the fixed Sentinel-1 pair when missing, runs the local checkout's
s1_insar pipeline in full-swath mode, and validates expected output artifacts.

Options:
  --preflight         validate tooling, paths, and free space without running

Environment overrides:
  CONDA_ENV          conda environment to activate (default: sarpyx-snap)
  PHIDOWN_REPO       phidown repo with .s5cfg (default: /Users/roberto.delprete/Downloads/phidown)
  S5CFG              s5cmd config path (default: $PHIDOWN_REPO/.s5cfg)
  INPUT_DIR          SAFE input directory (default: <repo>/input_data)
  OUTPUT_DIR         pipeline output directory (default: $INPUT_DIR/out_test)
  CUTS_OUTDIR        Zarr tile directory (default: $OUTPUT_DIR/tiles)
  GRID_PATH          grid GeoJSON path (default: <repo>/grid/grid_10km.geojson)
  GPT_PATH           SNAP GPT executable path (default: conda env SNAP GPT)
  REQUIRED_FREE_GIB  minimum free space before download/run (default: 60)
  SKIP_DOWNLOAD      set to 1 to require existing SAFE directories
  SKIP_RUN           set to 1 to skip pipeline execution and only validate
USAGE
}

PREFLIGHT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --preflight)
      PREFLIGHT_ONLY=1
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-sarpyx-snap}"
PHIDOWN_REPO="${PHIDOWN_REPO:-/Users/roberto.delprete/Downloads/phidown}"
S5CFG="${S5CFG:-${PHIDOWN_REPO}/.s5cfg}"
INPUT_DIR="${INPUT_DIR:-${REPO_DIR}/input_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/out_test}"
CUTS_OUTDIR="${CUTS_OUTDIR:-${OUTPUT_DIR}/tiles}"
GRID_PATH="${GRID_PATH:-${REPO_DIR}/grid/grid_10km.geojson}"
GPT_PATH="${GPT_PATH:-/opt/miniconda3/envs/${CONDA_ENV}/opt/esa-snap/bin/gpt}"
REQUIRED_FREE_GIB="${REQUIRED_FREE_GIB:-60}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_RUN="${SKIP_RUN:-0}"

MASTER_NAME="S1A_IW_SLC__1SDV_20240807T154929_20240807T154956_055109_06B710_F274.SAFE"
SLAVE_NAME="S1A_IW_SLC__1SDV_20240819T154929_20240819T154956_055284_06BD6F_5908.SAFE"
MASTER_PATH="${INPUT_DIR}/${MASTER_NAME}"
SLAVE_PATH="${INPUT_DIR}/${SLAVE_NAME}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: ${label} not found: ${path}" >&2
    exit 2
  fi
}

free_gib_at() {
  python - "$1" <<'PY'
import shutil
import sys

usage = shutil.disk_usage(sys.argv[1])
print(usage.free // (1024 ** 3))
PY
}

require_free_space() {
  mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$CUTS_OUTDIR"
  local free_gib
  free_gib="$(free_gib_at "$INPUT_DIR")"
  if (( free_gib < REQUIRED_FREE_GIB )); then
    echo "ERROR: only ${free_gib} GiB free at ${INPUT_DIR}; need at least ${REQUIRED_FREE_GIB} GiB." >&2
    echo "Set REQUIRED_FREE_GIB lower only if you have independently confirmed enough space." >&2
    if [[ -d "${REPO_DIR}/data/full_products/out" ]]; then
      echo "Large local generated-output candidate:" >&2
      du -sh "${REPO_DIR}/data/full_products/out" >&2 || true
    fi
    exit 2
  fi
}

activate_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 2
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

check_tooling() {
  require_path "$PHIDOWN_REPO" "phidown repo"
  require_path "$S5CFG" "s5cmd config"
  require_path "$GPT_PATH" "SNAP GPT"
  require_path "$GRID_PATH" "grid GeoJSON"
  if ! command -v phidown >/dev/null 2>&1; then
    echo "ERROR: phidown not found in conda environment: ${CONDA_ENV}" >&2
    exit 2
  fi
  if ! command -v s5cmd >/dev/null 2>&1; then
    echo "ERROR: s5cmd not found in conda environment: ${CONDA_ENV}" >&2
    exit 2
  fi
  echo "Conda env: ${CONDA_ENV}"
  echo "phidown: $(command -v phidown)"
  echo "s5cmd:   $(command -v s5cmd)"
  echo "GPT:     ${GPT_PATH}"
  echo "S5CFG:   ${S5CFG}"
  echo "Grid:    ${GRID_PATH}"
}

download_product() {
  local name="$1"
  local target="$2"
  if [[ -d "$target" ]]; then
    echo "Product already present: $target"
    return
  fi
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    echo "ERROR: missing product with SKIP_DOWNLOAD=1: $target" >&2
    exit 2
  fi
  require_path "$S5CFG" "s5cmd config"
  echo "Downloading $name into $INPUT_DIR"
  (
    cd "$PHIDOWN_REPO"
    phidown --name "$name" --output-dir "$INPUT_DIR" --config-file "$S5CFG" --mode safe
  )
  require_path "$target" "downloaded product"
}

run_pipeline() {
  if [[ "$SKIP_RUN" == "1" ]]; then
    echo "Skipping pipeline execution because SKIP_RUN=1."
    return
  fi
  require_path "$GPT_PATH" "SNAP GPT"
  require_path "$GRID_PATH" "grid GeoJSON"
  export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"
  python -m sarpyx.cli.main pipeline s1_insar \
    --master "$MASTER_PATH" \
    --slave "$SLAVE_PATH" \
    --output "$OUTPUT_DIR" \
    --grid-path "$GRID_PATH" \
    --cuts-outdir "$CUTS_OUTDIR" \
    --gpt-path "$GPT_PATH" \
    --tile-writer zarr \
    --param use_esd=false
}

validate_outputs() {
  echo "Validating output artifacts under:"
  echo "  output: $OUTPUT_DIR"
  echo "  tiles:  $CUTS_OUTDIR"
  require_path "$OUTPUT_DIR" "output directory"
  require_path "$CUTS_OUTDIR" "tile directory"
  python - "$OUTPUT_DIR" "$CUTS_OUTDIR" <<'PY'
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
cuts_dir = Path(sys.argv[2])
dims = sorted(output_dir.rglob("*.dim"))
zarrs = sorted(path for path in cuts_dir.rglob("*.zarr") if path.is_dir())
reports = sorted(
    path
    for root in (output_dir, cuts_dir)
    for path in root.rglob("*")
    if path.is_file() and (path.suffix.lower() in {".json", ".pdf"} or "cut_report" in path.name)
)
for label, paths in (("DIM", dims), ("ZARR", zarrs), ("REPORT", reports)):
    for path in paths:
        print(f"{label}: {path}")
if not dims:
    raise SystemExit("ERROR: no BEAM-DIMAP .dim product found.")
if not zarrs:
    raise SystemExit("ERROR: no Zarr tile stores found.")
if not reports:
    raise SystemExit("ERROR: no cut report or validation report found.")
PY
}

activate_env
check_tooling
require_free_space
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Preflight passed."
  exit 0
fi
download_product "$MASTER_NAME" "$MASTER_PATH"
download_product "$SLAVE_NAME" "$SLAVE_PATH"
run_pipeline
validate_outputs
