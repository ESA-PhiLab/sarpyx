`uv run pytest tests/test_worldsar_zarr_hooks.py tests/test_worldsar_h5.py::test_h5_raster_quality_counts_nan_and_all_band_zero_pixels tests/test_worldsar_h5.py::test_validate_tile_group_fails_all_nodata_tile tests/test_worldsar_h5.py::test_validate_tile_result_skips_and_removes_partial_nodata_tile -q` passed: 8 tests.

`uv run pytest tests/test_cli_pipeline.py tests/test_worldsar_pipeline_definitions.py tests/test_worldsar_zarr_hooks.py -q` passed: 17 tests.

`uv run pytest tests/test_worldsar_pipeline_definitions.py::test_insar_pipeline_declares_snapflow_v2_steps tests/test_worldsar_pipeline_definitions.py::test_declared_insar_runtime_executes_recipe tests/test_cli_pipeline.py::test_pipeline_cli_dispatches_builtin_double_product -q` passed: 3 tests.

`uv run pytest tests/test_worldsar_pipeline_definitions.py::test_insar_pipeline_declares_snapflow_v2_steps tests/test_worldsar_pipeline_definitions.py::test_declared_insar_runtime_executes_recipe tests/test_cli_pipeline.py::test_parse_params_coerces_json_values tests/test_cli_pipeline.py::test_pipeline_cli_dispatches_builtin_double_product -q` passed: 4 tests.

`uv run pytest tests/test_worldsar_pipeline_definitions.py::test_insar_pipeline_declares_snapflow_v2_steps tests/test_worldsar_pipeline_definitions.py::test_declared_insar_runtime_executes_recipe tests/test_cli_pipeline.py::test_pipeline_cli_dispatches_builtin_double_product tests/test_worldsar_zarr_hooks.py -q` passed: 8 tests.

Direct TOPS tile validation for `data/full_products/validation_zarr_fix_out_network/tiles/S1A_IW_SLC__1SDV_20250217T170559_20250217T170626_057939_072642_A1AB.SAFE` passed: 6 tiles, all `status=success`, `valid_fraction=1.0`, `nodata_fraction=0.0`.

Direct DB fallback check passed: `_run_db_indexing(..., cuts_outdir=<tmp>/cuts)` created `<tmp>/cuts/_db/product_core_metadata.parquet` when `config.DB_DIR` was unset.

`uv run pytest tests/test_sentinel_subap_pipeline.py -q` passed: 20 tests.
