# Changes

- Created remote `/lustre/projects/1001/rdelprete/WORLDSAR-v2/Makefilev2`.
- Added `WORLDSAR_MAX_WORKERS ?= 6` and worker ids `0..5`.
- Added worker env routing to `envs/worldsar-py312-worker<N>`.
- Added worker SNAP routing to `snap_workers/worker<N>/.snap`.
- Added `setup-workers` to clone the source env, remove cloned `opt/snap13/.snap`, seed worker SNAP dirs, and symlink shared auxdata paths.
- Added worker-level `flock` in generated PBS scripts.
- Added `pipeline-sen` round-robin routing and a `batch_size <= WORLDSAR_MAX_WORKERS` guard.
