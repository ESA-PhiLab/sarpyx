# Evidence

- `make install` verification printed `/usr/sbin/gpt`,
  `/opt/miniconda3/bin/sarpyx`, and `/opt/miniconda3/bin/phidown`.
- The intended paths are under `.conda/sarpyx`, especially
  `.conda/sarpyx/opt/esa-snap/bin/gpt`.
- `scripts/download.sh` still used `phidown` from `PATH`.
