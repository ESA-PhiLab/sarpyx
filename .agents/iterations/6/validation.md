# Validation

- `python -m py_compile .agents/skills/sarpyx-processing/scripts/snap_operator.py`
- `python .agents/skills/sarpyx-processing/scripts/snap_operator.py --repo /Users/roberto.delprete/Downloads/sarpyx --list | wc -l`: 217.
- Dry-run checks for `Calibration`, `Terrain-Correction`, and `Multi-size Mosaic`.
- Invalid operator check returns exit code 2.
- Skill metadata and file length gates pass.
