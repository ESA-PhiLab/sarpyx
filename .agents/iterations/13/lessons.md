# Lessons

For SpaceHPC, per-worker SNAP userdirs can be created without conda cloning by using a shared conda env plus hardlinked SNAP state and shared auxdata symlinks. The pipeline cannot be operational while Lustre object inodes are exhausted, because logs, qsub scripts, and directory-backed zarr chunks all need file objects.
