#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pipeline.sh single
  scripts/run_pipeline.sh pair

Environment:
  PIPELINE       Pipeline recipe name or pipeline file (default: s1_tops)
  INPUT          Input product for single-product pipelines
  MASTER         Master product for pair pipelines
  SLAVE          Slave product for pair pipelines
  OUTPUT         Output directory (default: outputs/make/$PIPELINE)
  CUTS_OUTDIR    Tile output directory (default: $OUTPUT/tiles)
  GRID_PATH      Optional grid GeoJSON path
  GPT_PATH       Optional SNAP GPT executable path
  PARAMS         Space-separated NAME=VALUE items for --param
  EXTRA_ARGS     Extra arguments appended as a shell-style string
  ENV_PREFIX     Local conda environment prefix (default: <repo>/.conda/sarpyx)
  PYTHON         Python executable used to run the repo CLI
EOF
}

mode="${1:-}"
if [[ "${mode}" != "single" && "${mode}" != "pair" ]]; then
  usage >&2
  exit 2
fi

pipeline="${PIPELINE:-s1_tops}"
output="${OUTPUT:-outputs/make/${pipeline}}"
cuts_outdir="${CUTS_OUTDIR:-${output}/tiles}"
grid_path="${GRID_PATH:-}"
gpt_path="${GPT_PATH:-}"
params="${PARAMS:-}"
extra_args="${EXTRA_ARGS:-}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${ENV_PREFIX:-${repo_dir}/.conda/sarpyx}"
python_bin="${PYTHON:-${env_prefix}/bin/python}"
if [[ -z "${gpt_path}" ]]; then
  gpt_path="${env_prefix}/opt/esa-snap/bin/gpt"
fi

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${python_bin}" >&2
  echo "Hint: run 'make install' or set ENV_PREFIX/PYTHON explicitly." >&2
  exit 2
fi

cmd=("${python_bin}" -m sarpyx.cli.main pipeline "${pipeline}" --output "${output}")

case "${mode}" in
  single)
    input="${INPUT:-}"
    [[ -n "${input}" ]] || { echo "ERROR: INPUT is required for single-product pipelines." >&2; exit 2; }
    cmd+=(--input "${input}")
    ;;
  pair)
    master="${MASTER:-}"
    slave="${SLAVE:-}"
    [[ -n "${master}" ]] || { echo "ERROR: MASTER is required for pair pipelines." >&2; exit 2; }
    [[ -n "${slave}" ]] || { echo "ERROR: SLAVE is required for pair pipelines." >&2; exit 2; }
    cmd+=(--master "${master}" --slave "${slave}")
    ;;
esac

[[ -n "${cuts_outdir}" ]] && cmd+=(--cuts-outdir "${cuts_outdir}")
[[ -n "${grid_path}" ]] && cmd+=(--grid-path "${grid_path}")
[[ -n "${gpt_path}" ]] && cmd+=(--gpt-path "${gpt_path}")

for param in ${params}; do
  cmd+=(--param "${param}")
done

if [[ -n "${extra_args}" ]]; then
  read -r -a split_extra <<< "${extra_args}"
  cmd+=("${split_extra[@]}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${cmd[@]}"
