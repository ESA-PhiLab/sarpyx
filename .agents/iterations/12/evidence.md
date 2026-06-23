# Evidence

- Remote root Makefile used one `ENV_PREFIX` and one `HPC_SNAP_USER_DIR` for all PBS submissions.
- Root `scripts/main.sh` consumes `ENV_PREFIX` and `SNAP_USER_DIR` from the Makefile-generated qsub environment.
- Source `.snap` on SpaceHPC is about 6.1G; top-level `auxdata` holds the size, especially `auxdata/Orbits`.
- Existing DEM auxdata is a symlink under `.snap/auxdata/dem/Copernicus 30m Global DEM`.
