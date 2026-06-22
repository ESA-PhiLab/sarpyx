#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${ENV_FILE:-${repo_dir}/environment.yml}"
env_prefix="${ENV_PREFIX:-${repo_dir}/.conda/sarpyx}"
gpt_bin="${env_prefix}/opt/esa-snap/bin/gpt"
sarpyx_bin="${env_prefix}/bin/sarpyx"
phidown_bin="${env_prefix}/bin/phidown"

require_executable() {
  local path="$1"
  local label="$2"
  if [[ ! -x "${path}" ]]; then
    echo "ERROR: ${label} executable not found in local conda env: ${path}" >&2
    exit 2
  fi
  echo "${label}: ${path}"
}

if [[ ! -f "${env_file}" ]]; then
  echo "ERROR: conda environment file not found: ${env_file}" >&2
  exit 2
fi

if [[ -x "${env_prefix}/bin/python" ]]; then
  echo "Updating local conda environment from ${env_file}: ${env_prefix}"
  conda env update -p "${env_prefix}" -f "${env_file}" --prune
else
  echo "Creating local conda environment from ${env_file}: ${env_prefix}"
  mkdir -p "$(dirname "${env_prefix}")"
  conda env create -p "${env_prefix}" -f "${env_file}"
fi

echo "Installing sarpyx and phidown into ${env_prefix}"
conda run -p "${env_prefix}" python -m pip install -e "${repo_dir}[copernicus]"

echo "Verifying SNAP GPT, sarpyx, and phidown in ${env_prefix}"
require_executable "${gpt_bin}" "SNAP GPT"
require_executable "${sarpyx_bin}" "sarpyx"
require_executable "${phidown_bin}" "phidown"

echo "Installed. Activate with: conda activate ${env_prefix}"
