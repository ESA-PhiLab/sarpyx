# Validation

- `python docs/generate_static_site.py`
- `uv run --group dev python docs/build_site.py`
- Internal docs link check: `MISSING_COUNT=0`
- Version/import check: package and subpackages report `1.0.2`
- `uv run --group dev pytest -q`: 211 passed, 5 skipped
- `uv build`: built `dist/sarpyx-1.0.2.tar.gz` and `dist/sarpyx-1.0.2-py3-none-any.whl`
