# Validation

- `uv run pytest tests/test_makefile_surface.py::test_install_uses_local_prefix_environment_and_verifies_snap_sarpyx_phidown tests/test_makefile_surface.py::test_download_helper_uses_local_prefix_phidown -q`
- `uv run pytest tests/test_makefile_surface.py::test_make_download_ignores_ambient_phidown_env -q`

Real conda solving, phidown downloads, and SNAP processing were not run.
