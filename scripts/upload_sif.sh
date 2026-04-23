#!/usr/bin/env bash
set -euo pipefail

LOCAL_FILE="${1:-${SIF:-}}"
REMOTE_HOST="SpaceHPC"
REMOTE_PATH="/lustre/projects/1001/rdelprete/WORLDSAR/sarpyx.sif"

if [[ -z "${LOCAL_FILE}" ]]; then
  echo "Usage: $0 /path/to/sarpyx.sif" >&2
  echo "Or set SIF=/path/to/sarpyx.sif in the environment." >&2
  exit 1
fi

if [[ ! -f "${LOCAL_FILE}" ]]; then
  echo "Error: local SIF not found: ${LOCAL_FILE}" >&2
  exit 1
fi

rsync -avh --progress "${LOCAL_FILE}" "${REMOTE_HOST}:${REMOTE_PATH}"
