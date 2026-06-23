# Changes

- Created branch `1.0.1`.
- Added `sarpyx.snapflow.locks` and wired `sarpyx worldsar` through a product/output lock with `--lock-timeout`.
- Added fail-fast lock validation and parser coverage.
- Changed Zarr tile writes to build hidden staged stores and promote only after writer success.
- Added `scripts/snap_userdir.sh`; wired `scripts/worldsar.sh` and `scripts/main.sh` to isolate SNAP userdir per product/job by default.
- Changed SNAP userdir seeding from whole-directory copy to light seeding: shared entries such as `auxdata` are symlinked, top-level symlinks/files are preserved, and ordinary directories are created empty unless `SNAP_USERDIR_SEED_MODE=copy` is requested.
- Kept the earlier BEAM-DIMAP subap payload validation/retry hardening in this branch.
