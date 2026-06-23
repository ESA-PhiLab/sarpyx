# Validation

- Remote `bash -n scripts/gpt_isolated.sh scripts/pipeline_sen_dispatch.sh` passed.
- Remote `make -f Makefilev2 -n help`, `setup-workers`, `_run-hpc PRODUCT=input_data/FAKE.SAFE WORKER_ID=2`, and `pipeline-sen ... batch_size=6 poll_interval=60` passed.
- Remote `make -f Makefilev2 setup-workers WORLDSAR_MAX_WORKERS=6` completed.
- Remote `make -f Makefilev2 ensure-workers-ready` and `ensure-inode-headroom` passed; output scratch had about 198M free inodes.
- Remote fake-GPT wrapper test confirmed unique `TMPDIR`, `_JAVA_OPTIONS`, rewritten `-J-Dsnap.userdir`, worker seed symlink validation, and runtime cleanup.
