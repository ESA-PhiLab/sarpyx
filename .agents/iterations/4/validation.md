# Validation

- `uv run pytest tests/test_makefile_surface.py tests/test_pbs_caller.py -q`
- `bash -n scripts/install_conda.sh scripts/download.sh scripts/run_pipeline.sh scripts/pbs_caller.sh`
- `make help`
- `make -n pipeline PIPELINE=s1_tops INPUT=/tmp/input.SAFE`

Real conda solving and SNAP-heavy processing were not run.
