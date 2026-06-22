#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./pbs_caller.sh [options] -- command [args...]
  ./pbs_caller.sh [options] command [args...]

Options:
  --size small|medium|large  Resource profile (default: large)
  --name NAME                PBS job name (default: worldsar)
  --queue QUEUE              PBS queue (default: cpu_std)
  --walltime HH:MM:SS        Override profile walltime
  --ncpus N                  Override profile CPU count
  --mem SIZE                 Override profile memory, e.g. 128g
  --qsub PATH                qsub executable (default: qsub)
  --dry-run                  Print the generated PBS script instead of submitting
  -h, --help                 Show this help

Profiles:
  small   02:00:00  32 CPUs   32g
  medium  04:00:00  96 CPUs   64g
  large   06:00:00  192 CPUs  128g
EOF
}

quote_command() {
  local quoted=()
  local arg
  for arg in "$@"; do
    if [[ "${arg}" =~ ^[A-Za-z0-9_./:=,+@%-]+$ ]]; then
      quoted+=("${arg}")
      continue
    fi
    arg="${arg//\'/\'\\\'\'}"
    arg="'${arg}'"
    quoted+=("${arg}")
  done
  printf "%s " "${quoted[@]}"
}

size="large"
job_name="${PBS_CALLER_JOB_NAME:-worldsar}"
queue="${PBS_CALLER_QUEUE:-cpu_std}"
walltime=""
ncpus=""
mem=""
qsub_bin="${PBS_CALLER_QSUB:-qsub}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)
      [[ $# -ge 2 ]] || { echo "ERROR: --size requires a value." >&2; exit 2; }
      size="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "ERROR: --name requires a value." >&2; exit 2; }
      job_name="$2"
      shift 2
      ;;
    --queue)
      [[ $# -ge 2 ]] || { echo "ERROR: --queue requires a value." >&2; exit 2; }
      queue="$2"
      shift 2
      ;;
    --walltime)
      [[ $# -ge 2 ]] || { echo "ERROR: --walltime requires a value." >&2; exit 2; }
      walltime="$2"
      shift 2
      ;;
    --ncpus)
      [[ $# -ge 2 ]] || { echo "ERROR: --ncpus requires a value." >&2; exit 2; }
      ncpus="$2"
      shift 2
      ;;
    --mem)
      [[ $# -ge 2 ]] || { echo "ERROR: --mem requires a value." >&2; exit 2; }
      mem="$2"
      shift 2
      ;;
    --qsub)
      [[ $# -ge 2 ]] || { echo "ERROR: --qsub requires a value." >&2; exit 2; }
      qsub_bin="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -gt 0 ]] || { echo "ERROR: command is required." >&2; usage >&2; exit 2; }

case "${size}" in
  small)
    profile_walltime="02:00:00"
    profile_ncpus="32"
    profile_mem="32g"
    ;;
  medium)
    profile_walltime="04:00:00"
    profile_ncpus="96"
    profile_mem="64g"
    ;;
  large)
    profile_walltime="06:00:00"
    profile_ncpus="192"
    profile_mem="128g"
    ;;
  *)
    echo "ERROR: unknown size '${size}'. Use small, medium, or large." >&2
    exit 2
    ;;
esac

walltime="${walltime:-${profile_walltime}}"
ncpus="${ncpus:-${profile_ncpus}}"
mem="${mem:-${profile_mem}}"
submit_dir="$(pwd -P)"
command_line="$(quote_command "$@")"
job_script="$(mktemp "${TMPDIR:-/tmp}/pbs_caller.XXXXXX.pbs")"

cleanup() {
  rm -f "${job_script}"
}
trap cleanup EXIT

{
  printf '#!/bin/bash\n'
  printf '#PBS -N %s\n' "${job_name}"
  printf '#PBS -S /bin/bash\n'
  printf '#PBS -q %s\n' "${queue}"
  printf '#PBS -l walltime=%s\n' "${walltime}"
  printf '#PBS -l select=1:ncpus=%s:mem=%s\n' "${ncpus}" "${mem}"
  printf '\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "${submit_dir}"
  printf 'exec %s\n' "${command_line}"
} > "${job_script}"

if [[ "${dry_run}" -eq 1 ]]; then
  cat "${job_script}"
else
  "${qsub_bin}" "${job_script}"
fi
