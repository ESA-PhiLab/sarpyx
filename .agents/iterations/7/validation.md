# Validation

- `python -m pytest tests/test_subap_feature_pipeline.py` passed: 16 tests.
- `python -m pytest tests/test_worldsar_h5.py -k prepare_products_by_epsg` passed: 1 selected test.
- `python -m pytest tests/test_sentinel_subap_pipeline.py -k subap` passed: 22 tests.
- `git diff --check` passed.
- `uv run pytest tests/test_subap_feature_pipeline.py` was blocked by dependency resolution: `requires-python >=3.11,<4` conflicts with `rasterio==1.5.0` requiring Python >=3.12.
