#!/usr/bin/env bash
set -euo pipefail

LOCAL_FILE="/shared/home/rdelprete/PythonProjects/srp/sarpyx.sif"
REMOTE_HOST="SpaceHPC"
REMOTE_PATH="/lustre/projects/1001/rdelprete/WORLDSAR/sarpyx.sif"

rsync -avh --progress "${LOCAL_FILE}" "${REMOTE_HOST}:${REMOTE_PATH}"
