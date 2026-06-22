# Changes

- `scripts/install_conda.sh` now verifies exact executable paths under
  `ENV_PREFIX` and no longer uses `command -v`.
- `scripts/download.sh` now defaults to `ENV_PREFIX/bin/phidown`.
- Makefile adds `PHIDOWN` defaulting to `ENV_PREFIX/bin/phidown` and ignores
  ambient `PHIDOWN` unless passed explicitly as a make variable.
- Tests cover exact install verification and local phidown resolution.
