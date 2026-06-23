# Evidence

- Package version was declared only in `pyproject.toml`.
- Local `import sarpyx; sarpyx.__version__` returned installed metadata `1.0.0` after the manifest bump, because `_resolve_version()` preferred distribution metadata over the source-tree manifest.
