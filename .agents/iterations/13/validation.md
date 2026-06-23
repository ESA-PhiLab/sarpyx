# Validation

- `make -f Makefilev2 -n help` and `_run-hpc PRODUCT=input_data/FAKE.SAFE WORKER_ID=2` parsed on SpaceHPC.
- `make -f Makefilev2 setup-workers WORLDSAR_SETUP_MIN_FREE_INODES=0` completed.
- `make -f Makefilev2 ensure-workers-ready` completed.
- Verified workers `0..5` have `.snap` dirs and shared links for `auxdata/Orbits` and `auxdata/dem/Copernicus 30m Global DEM`.
- `make -f Makefilev2 pipeline-sen /lustre/projects/1001/rdelprete/WORLDSAR-v2/product_list.txt batch_size=6 wait_time=6h` stopped before qsub with `ensure-inode-headroom`: 272 free inodes versus 200000 required.
- `qstat -u "$USER"` showed no new queued jobs after the guarded launch attempt.
