# Changes

- Added `sarpyx.snapflow.stamps` with StaMPS export folder validation, master-date inference, and `mt_prep_snap` / `mt_prep_gamma` command execution.
- Added InSAR runtime support for `StampsPrep`.
- Appended `StampsPrep` to the `2stamps` recipe after `StampsExport`; callers can disable it with `run_stamps_prep=false` or pass command/config overrides.
- Added tests for the new recipe tail, runtime dispatch, and exact `mt_prep_snap` command construction.
