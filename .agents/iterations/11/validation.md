# Validation

- `bash -n scripts/snap_userdir.sh`
- `bash -n scripts/worldsar.sh`
- `bash -n scripts/main.sh`
- Shell probe with base `.snap/auxdata` as a real directory: run userdir contains an `auxdata` symlink, top-level files copy, other dirs are empty.
- Shell probe with base `.snap/auxdata` as a symlink: run userdir contains a symlink to the base auxdata entry.
- `python -m pytest tests/test_cli_worldsar.py` -> 8 passed.
- `python -m pytest tests/test_worldsar_zarr_hooks.py` -> 7 passed.
- `python -m pytest tests/test_subap_feature_pipeline.py` -> 16 passed.
- `python -m pytest tests/test_public_imports.py` -> 5 passed.
- `python -m pytest tests/test_sentinel_subap_pipeline.py -k subap` -> 22 passed.
- `python - <<'PY' ... print(sarpyx.__version__) ... PY` -> `1.0.1`.
- `git diff --check`
