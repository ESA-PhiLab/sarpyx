#!/usr/bin/env bash
set -euo pipefail

product_name="${PRODUCT_NAME:-${NAME:-}}"
download_dir="${DOWNLOAD_DIR:-input_data}"
download_mode="${DOWNLOAD_MODE:-safe}"
config_file="${CONFIG_FILE:-}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${ENV_PREFIX:-${repo_dir}/.conda/sarpyx}"
phidown_bin="${PHIDOWN:-${env_prefix}/bin/phidown}"

if [[ -z "${product_name}" ]]; then
  echo "ERROR: PRODUCT_NAME is required." >&2
  echo "Example: make download PRODUCT_NAME=S1A_...SAFE" >&2
  exit 2
fi

if [[ ! -x "${phidown_bin}" ]]; then
  echo "ERROR: phidown executable not found in local conda env: ${phidown_bin}" >&2
  echo "Hint: run 'make install' or set ENV_PREFIX/PHIDOWN explicitly." >&2
  exit 2
fi

mkdir -p "${download_dir}"

args=(--name "${product_name}" --output-dir "${download_dir}" --mode "${download_mode}")
if [[ -n "${config_file}" ]]; then
  args+=(--config-file "${config_file}")
fi

echo "Downloading ${product_name} into ${download_dir}"
"${phidown_bin}" "${args[@]}"
