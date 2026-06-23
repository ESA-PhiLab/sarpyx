# Validation

- `uv run pytest tests/test_makefile_surface.py::test_pipeline_helper_uses_repo_dispatcher_not_stale_console_script -q`
- `make pipeline PIPELINE=s1_tops INPUT=/Users/roberto.delprete/Downloads/sarpyx/input_data/S1A_IW_SLC__1SDV_20240807T154929_20240807T154956_055109_06B710_F274.SAFE EXTRA_ARGS=--list`
- `bash -n scripts/run_pipeline.sh`

The full SAR processing command was not run because it would start SNAP-heavy
processing.
