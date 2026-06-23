# Validation

- `make -f Makefilev2 help` parses and shows worker defaults.
- `make -f Makefilev2 -n setup-workers` shows six worker setup loop, clone source env, remove cloned `.snap`, rsync SNAP seed, and symlink shared auxdata.
- `make -f Makefilev2 -n _run-hpc PRODUCT=input_data/FAKE.SAFE WORKER_ID=2` shows qsub with worker2 env, worker2 SNAP dir, and worker lock.
- `make -f Makefilev2 ensure-worker WORKER_ID=6` fails out of range.
- `make -f Makefilev2 ensure-worker WORKER_ID=5` passes.
- `make -f Makefilev2 pipeline-sen ... batch_size=7` fails because max workers is 6.
