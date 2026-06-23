# Changes

- Remote `Makefilev2` now defaults to `WORKER_ENV_MODE=shared` and `WORKER_SNAP_MODE=hardlink`.
- `setup-workers` now uses `set -euo pipefail`, validates setup modes, supports shared/clone/hardlink env modes, and uses hardlinked SNAP worker state with shared auxdata symlinks.
- Added `ensure-workers-ready` and `ensure-inode-headroom`.
- Added `PYTHONDONTWRITEBYTECODE=1` to submitted qsub job scripts.
- `pipeline-sen` now depends on worker readiness and inode headroom before submitting qsub jobs.
