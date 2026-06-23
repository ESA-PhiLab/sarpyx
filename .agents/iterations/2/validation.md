# Validation

- `uv run pytest tests/test_makefile_surface.py::test_install_uses_environment_file_and_verifies_snap_sarpyx_phidown -q`
- `uv run pytest tests/test_makefile_surface.py tests/test_pbs_caller.py -q`
- `bash -n scripts/install_conda.sh scripts/download.sh scripts/run_pipeline.sh scripts/pbs_caller.sh`
- `make help`

Real conda environment creation/update was not run.
