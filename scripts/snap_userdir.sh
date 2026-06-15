#!/usr/bin/env bash

worldsar_configure_snap_userdir() {
  local product_name="$1"
  local output_root="$2"
  local current_userdir="${SNAP_USERDIR:-${SNAP_USER_DIR:-}}"
  local mode="${SNAP_USERDIR_MODE:-isolated}"

  if [[ -z "${current_userdir}" ]]; then
    echo "ERROR: SNAP_USERDIR/SNAP_USER_DIR is not configured." >&2
    return 2
  fi

  export SNAP_USER_BASE_DIR="${SNAP_USER_BASE_DIR:-${current_userdir}}"

  if [[ "${mode}" == "shared" ]]; then
    mkdir -p "${current_userdir}"
    export SNAP_USERDIR="${current_userdir}"
    export SNAP_USER_DIR="${current_userdir}"
    export snap_userdir="${current_userdir}"
    return 0
  fi
  if [[ "${mode}" != "isolated" ]]; then
    echo "ERROR: SNAP_USERDIR_MODE must be either 'isolated' or 'shared': ${mode}" >&2
    return 2
  fi
  local seed_mode="${SNAP_USERDIR_SEED_MODE:-light}"
  local shared_entries=" ${SNAP_USERDIR_SHARED_ENTRIES:-auxdata} "

  local job_id="${PBS_JOBID:-${SLURM_JOB_ID:-${LSB_JOBID:-manual-$$}}}"
  local safe_product="${product_name//[^A-Za-z0-9_.-]/_}"
  local safe_job="${job_id//[^A-Za-z0-9_.-]/_}"
  local run_userdir="${SNAP_RUN_USERDIR:-${SNAP_RUN_USER_DIR:-${output_root}/snap_userdirs/${safe_product}.${safe_job}}}"
  mkdir -p "${run_userdir}"

  local seed_source=""
  if [[ -d "${SNAP_USER_BASE_DIR}" ]]; then
    seed_source="$(cd "${SNAP_USER_BASE_DIR}" && pwd -P)"
  fi
  local run_resolved="$(cd "${run_userdir}" && pwd -P)"
  shopt -s nullglob dotglob
  local existing=("${run_userdir}"/*)
  shopt -u nullglob dotglob
  if [[ -n "${seed_source}" && "${seed_source}" != "${run_resolved}" && "${#existing[@]}" -eq 0 ]]; then
    case "${seed_mode}" in
      none)
        ;;
      copy)
        cp -a "${SNAP_USER_BASE_DIR}/." "${run_userdir}/"
        ;;
      light)
        shopt -s nullglob dotglob
        local entry name
        for entry in "${SNAP_USER_BASE_DIR}"/*; do
          name="$(basename "${entry}")"
          if [[ "${shared_entries}" == *" ${name} "* ]]; then
            ln -s "${entry}" "${run_userdir}/${name}"
          elif [[ -L "${entry}" || -f "${entry}" ]]; then
            cp -a "${entry}" "${run_userdir}/${name}"
          elif [[ -d "${entry}" ]]; then
            mkdir -p "${run_userdir}/${name}"
          fi
        done
        shopt -u nullglob dotglob
        ;;
      *)
        echo "ERROR: SNAP_USERDIR_SEED_MODE must be 'light', 'copy', or 'none': ${seed_mode}" >&2
        return 2
        ;;
    esac
  fi

  export SNAP_USERDIR="${run_userdir}"
  export SNAP_USER_DIR="${run_userdir}"
  export snap_userdir="${run_userdir}"
}
