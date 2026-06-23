# Validation

- `python -m compileall -q sarpyx/pipelines/double_product sarpyx/snapflow/insar.py sarpyx/snapflow/stamps.py`
- `uv run pytest -q tests/test_snap2stamps_double_product_pipeline.py tests/test_worldsar_pipeline_definitions.py tests/test_cli_pipeline.py tests/test_snap2stamps_pipelines.py tests/test_public_imports.py`
- Live prep boundary check validated the existing export folder and inferred master date `20250217`.
- Live prep execution is blocked locally because `mt_prep_snap` is not installed or sourced on `PATH`.
- Fresh end-to-end CLI attempt completed SNAP split/orbit/coreg/IFG/deburst/elevation/lat-lon/export to `data/2stamps_out_end2end/stamps`, then failed at `StampsPrep` because `mt_prep_snap` is unavailable.
- Temporary StaMPS source build succeeded under `/tmp/sarpyx-stamps-probe`, but a real local prep still lacks `matlab` and `gawk`.
