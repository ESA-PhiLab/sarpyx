# Changes

- Added `ENV_PREFIX` defaulting to `.conda/sarpyx`.
- Changed `make install` to create/update conda with `conda env ... -p`.
- Changed pipeline targets to pass `.conda/sarpyx/bin/python` and
  `.conda/sarpyx/opt/esa-snap/bin/gpt`.
- Changed `scripts/run_pipeline.sh` to default Python and GPT to `ENV_PREFIX`.
- Ignored ambient `PYTHON` and `GPT_PATH` in Makefile unless passed explicitly
  on the make command line.
