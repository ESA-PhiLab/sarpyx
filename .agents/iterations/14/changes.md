# Changes

- Remote `Makefilev2` now routes logs, outputs, locks, runtime temp, and worker SNAP seeds under `/lustre/scratch/1001/rdelprete/worldsar_engine_tmp`.
- Added remote `scripts/gpt_isolated.sh` for per-GPT `TMPDIR`, `_JAVA_OPTIONS`, and runtime `snap.userdir` isolation.
- Added remote `scripts/pipeline_sen_dispatch.sh` for rolling qsub with `batch_size` as max active jobs.
- `setup-workers` now creates a SNAP symlink manifest, seeds six scratch worker userdirs, relinks shared auxdata, and validates manifest/shared symlinks.
