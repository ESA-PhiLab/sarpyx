#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Sentinel-1 burst InSAR from two extracted .SAFE burst products.

Quick start:
  pipelines/sentinel_insar/run_sentinel_insar.sh \
    --master data/bursts/extracted/master/<MASTER_ID>/product.SAFE \
    --slave  data/bursts/extracted/slave/<SLAVE_ID>/product.SAFE

Required inputs:
  --master PATH       Master extracted single-burst .SAFE directory.
  --slave PATH        Slave extracted single-burst .SAFE directory.

Common options:
  --pipeline NAME     ifg (default), gslc, both, or a full YAML pipeline name.
  --outdir PATH       Output directory. Default: data/output/sentinel_insar.
  --gpt PATH          SNAP GPT executable. Also read from SARPYX_SNAP_GPT or GPT_PATH.
  --dry-run           Show planned steps without running SNAP.
  --resume            Reuse matching existing outputs.
  --overwrite         Replace existing outputs.
  --json              Print sarpyx-pipeline JSON output.
  --help              Show this help.

Examples:
  # Preview exactly what will run.
  pipelines/sentinel_insar/run_sentinel_insar.sh --master master.SAFE --slave slave.SAFE --dry-run

  # Run the default IFG branch.
  pipelines/sentinel_insar/run_sentinel_insar.sh --master master.SAFE --slave slave.SAFE

  # Run both GSLC and IFG branches.
  pipelines/sentinel_insar/run_sentinel_insar.sh --master master.SAFE --slave slave.SAFE --pipeline both

If preflight fails, fix the listed item and rerun the same command.
EOF
}

info() {
  printf '%s\n' "$*" >&2
}

fail() {
  info "ERROR: $*"
  exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
config="${script_dir}/sentinel_insar.yaml"
outdir="${SARPYX_INSAR_OUTDIR:-${repo_root}/data/output/sentinel_insar}"
pipeline="${SARPYX_INSAR_PIPELINE:-ifg}"
gpt_path="${SARPYX_SNAP_GPT:-${GPT_PATH:-}}"
preflight=true
dry_run=false
extra_args=()
positional=()
master=""
slave=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --master)
      [[ $# -ge 2 ]] || fail "--master requires a path"
      master="$2"
      shift 2
      ;;
    --slave)
      [[ $# -ge 2 ]] || fail "--slave requires a path"
      slave="$2"
      shift 2
      ;;
    --outdir|-o)
      [[ $# -ge 2 ]] || fail "--outdir requires a path"
      outdir="$2"
      shift 2
      ;;
    --pipeline|-p)
      [[ $# -ge 2 ]] || fail "--pipeline requires ifg, gslc, both, or a YAML pipeline name"
      pipeline="$2"
      shift 2
      ;;
    --gpt)
      [[ $# -ge 2 ]] || fail "--gpt requires the SNAP GPT executable path"
      gpt_path="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      extra_args+=("$1")
      shift
      ;;
    --resume|--overwrite|--json)
      extra_args+=("$1")
      shift
      ;;
    --no-preflight)
      preflight=false
      shift
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    -*)
      extra_args+=("$1")
      shift
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${master}" && ${#positional[@]} -ge 1 ]]; then
  master="${positional[0]}"
fi
if [[ -z "${slave}" && ${#positional[@]} -ge 2 ]]; then
  slave="${positional[1]}"
fi
if [[ ${#positional[@]} -gt 2 ]]; then
  fail "Too many positional arguments. Use --help for examples."
fi

case "${pipeline}" in
  ifg|default)
    pipeline="sentinel_insar_ifg"
    ;;
  gslc)
    pipeline="sentinel_insar_gslc"
    ;;
  both|all)
    pipeline="sentinel_insar"
    ;;
esac

if [[ -z "${master}" || -z "${slave}" ]]; then
  usage >&2
  fail "Provide both --master and --slave extracted .SAFE paths."
fi

runner=()
python_cmd=()
if [[ -x "${repo_root}/.venv/bin/python" ]]; then
  export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
  runner=("${repo_root}/.venv/bin/python" -m sarpyx.cli.pipeline)
  python_cmd=("${repo_root}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  runner=(uv run sarpyx-pipeline)
  python_cmd=(uv run python)
elif command -v sarpyx-pipeline >/dev/null 2>&1; then
  runner=(sarpyx-pipeline)
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    python_cmd=(python3)
  fi
elif command -v python3 >/dev/null 2>&1; then
  if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    fail "Python >=3.11 is required. The system python3 is too old."
  fi
  export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
  runner=(python3 -m sarpyx.cli.pipeline)
  python_cmd=(python3)
else
  fail "sarpyx-pipeline runner not found. Install uv, sarpyx-pipeline, or Python >=3.11."
fi

is_snap_gpt() {
  local candidate="$1"
  [[ -x "${candidate}" ]] || return 1
  local help_text
  help_text="$("${candidate}" -h 2>&1 || true)"
  printf '%s' "${help_text}" | grep -Eiq 'snap|graph processing|operator|graph-file|target'
}

detect_gpt() {
  local candidate
  for candidate in \
    "${gpt_path}" \
    "/usr/local/snap/bin/gpt" \
    "/Applications/snap/bin/gpt" \
    "${HOME}/esa-snap/bin/gpt" \
    "/opt/snap/bin/gpt"; do
    [[ -n "${candidate}" ]] || continue
    if is_snap_gpt "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  if command -v gpt >/dev/null 2>&1; then
    candidate="$(command -v gpt)"
    if is_snap_gpt "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi
  return 1
}

errors=0
if [[ "${preflight}" == true ]]; then
  info "Sentinel-1 InSAR preflight"
  info "  master:   ${master}"
  info "  slave:    ${slave}"
  info "  pipeline: ${pipeline}"
  info "  outdir:   ${outdir}"

  if [[ ! -d "${master}" || "${master}" != *.SAFE ]]; then
    info ""
    info "Missing or invalid master input:"
    info "  ${master}"
    info "Expected an extracted single-burst .SAFE directory."
    errors=$((errors + 1))
  fi
  if [[ ! -d "${slave}" || "${slave}" != *.SAFE ]]; then
    info ""
    info "Missing or invalid slave input:"
    info "  ${slave}"
    info "Expected an extracted single-burst .SAFE directory."
    errors=$((errors + 1))
  fi

  if resolved_gpt="$(detect_gpt)"; then
    gpt_path="${resolved_gpt}"
    info "  SNAP GPT: ${gpt_path}"
  else
    info ""
    info "SNAP GPT was not found."
    info "Install SNAP, then rerun with one of:"
    info "  export SARPYX_SNAP_GPT=/Applications/snap/bin/gpt"
    info "  pipelines/sentinel_insar/run_sentinel_insar.sh --master master.SAFE --slave slave.SAFE --gpt /Applications/snap/bin/gpt"
    info "Do not use /usr/sbin/gpt; that is the macOS disk partition tool."
    errors=$((errors + 1))
  fi

  if [[ ${errors} -gt 0 && "${dry_run}" == true ]]; then
    info ""
    info "Preflight found ${errors} issue(s), but --dry-run will continue because no SNAP processing starts."
    info "A real run will stop until these issues are fixed."
  elif [[ ${errors} -gt 0 ]]; then
    info ""
    info "Preflight failed with ${errors} issue(s). No processing was started."
    info ""
    info "For the snapflow_v2 notebook pair, the selected burst IDs were:"
    info "  master: 8ff4f2b3-64d8-4852-8c3b-4b2b8f729b03"
    info "  slave:  2404a519-5e05-4dcc-95e5-b3e4e8a79127"
    info "Download and extract those bursts first, then rerun this command."
    info "For notebook downloads, set CDSE_USERNAME and CDSE_PASSWORD first."
    exit 2
  fi
fi

config_to_run="${config}"
tmp_config=""
cleanup() {
  [[ -z "${tmp_config}" ]] || rm -f "${tmp_config}"
}
trap cleanup EXIT

if [[ -n "${gpt_path}" ]]; then
  if [[ ${#python_cmd[@]} -eq 0 ]]; then
    fail "A Python >=3.11 interpreter is required to apply --gpt to the runtime config."
  fi
  tmp_config="$(mktemp "${TMPDIR:-/tmp}/sentinel_insar.XXXXXX.yaml")"
  "${python_cmd[@]}" - "${config}" "${tmp_config}" "${gpt_path}" <<'PY'
from pathlib import Path
import sys
import yaml

source, target, gpt_path = map(Path, sys.argv[1:4])
data = yaml.safe_load(source.read_text(encoding="utf-8"))
data.setdefault("defaults", {})["gpt_path"] = gpt_path.as_posix()
target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY
  config_to_run="${tmp_config}"
fi

cmd=(
  "${runner[@]}"
  run "${config_to_run}"
  --pipeline "${pipeline}"
  --set-input "master=${master}"
  --set-input "slave=${slave}"
  --outdir "${outdir}"
)
if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

info "Starting sarpyx-pipeline. Use --dry-run to preview without running SNAP."
exec "${cmd[@]}"
