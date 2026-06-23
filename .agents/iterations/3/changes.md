# Changes

- Changed `scripts/run_pipeline.sh` to call
  `python -m sarpyx.cli.main pipeline` with the repo root on `PYTHONPATH`.
- Added a regression test that uses fake `python` and verifies the helper does
  not call the stale console script.
- Updated Makefile help and docs to prefer the repo module command in PBS
  examples.
