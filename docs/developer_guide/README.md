# Developer Guide

Development documentation for sarpyx maintainers and contributors.

## Topics

- [Architecture](architecture.md): package boundaries, data flow, and runtime
  components.
- [Contributing](contributing.md): setup, coding standards, tests, and review
  expectations.
- [Code of conduct](code_of_conduct.md): collaboration expectations for project
  spaces.

## Release Checks

Before publishing a release, verify the package metadata, documentation, and
tests from the repository root:

```bash
uv sync --group dev
uv run pytest -q
uv build
python docs/generate_static_site.py
python docs/build_site.py
```
