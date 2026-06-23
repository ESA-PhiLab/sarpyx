# Evidence

- Full worker cloning failed on SpaceHPC with `No space left on device` during conda clone and `.snap` rsync.
- `df -Pi` / `lfs df -i` showed Lustre object inode exhaustion: output path had hundreds of free inodes, and several OSTs had zero free object inodes.
- Partial worker dirs from the failed clone were removed.
- Patched remote `/lustre/projects/1001/rdelprete/WORLDSAR-v2/Makefilev2` in place because creating a replacement file also failed with `No space left on device`.
- Remote patched Makefile checksum: `d96a64eb22cb7f71523314e570488893e0d0373dbebb3d89c627627eb742b6ce`.
