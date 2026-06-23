# Next

- Free Lustre object inodes or move zarr/log output to storage with real inode headroom.
- Re-run `make -f Makefilev2 pipeline-sen /lustre/projects/1001/rdelprete/WORLDSAR-v2/product_list.txt batch_size=6 wait_time=6h` after `df -Pi OUT/worldsar_output` reports enough free inodes.
- Do not bypass `WORLDSAR_MIN_FREE_INODES` for production unless the output store is changed away from directory-backed zarr.
