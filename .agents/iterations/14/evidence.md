# Evidence

- SpaceHPC root `Makefilev2` previously used fixed batch sleep and project-tree worker state.
- Source SNAP userdir symlink manifest currently contains `auxdata/dem/Copernicus 30m Global DEM -> /lustre/scratch/1000/WorldSAR/Copernicus 30m Global DEM`.
- Project-to-scratch `rsync --link-dest` failed with `Invalid cross-device link`; default seed mode was changed to plain `rsync -a` with large shared paths excluded and relinked.
- Existing job `104817.pbs` was already running with old-style direct GPT and shared source `.snap`; it was not touched.
