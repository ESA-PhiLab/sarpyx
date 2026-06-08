# Validation

- `python -m py_compile .agents/skills/sarpyx-processing/scripts/preflight.py`
- `python .agents/skills/sarpyx-processing/scripts/preflight.py --repo /Users/roberto.delprete/Downloads/sarpyx`
- Metadata gate: required skill fields, filename/name parity, UI fields, positive route, negative route.
- Scope gate: destructive and operational actions require explicit user request.
- File length gate: all newly written skill files are under 300 lines.
- Publish gate: no local publish skill/script found; only GitHub publish workflows exist, not run.
