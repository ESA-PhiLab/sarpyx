#!/usr/bin/env bash
# Source this after activating the sarpyx-snap conda environment.

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is not set. Activate the conda environment first." >&2
  return 1 2>/dev/null || exit 1
fi

export SNAP_HOME="${SNAP_HOME:-${CONDA_PREFIX}/opt/esa-snap}"
export GPT_PATH="${GPT_PATH:-${SNAP_HOME}/bin/gpt}"
export gpt_path="$GPT_PATH"
export SNAP_USERDIR="${SNAP_USERDIR:-${CONDA_PREFIX}/.snap}"
export snap_userdir="$SNAP_USERDIR"

if [[ ! -x "$GPT_PATH" ]]; then
  echo "SNAP GPT not found or not executable: $GPT_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${SNAP_HOME}/bin:${CONDA_PREFIX}/bin:$PATH"
