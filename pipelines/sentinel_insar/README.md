# Sentinel-1 Burst InSAR

Download the same burst pair selected by `notebooks/snapflow_v2.ipynb`:

```bash
.venv/bin/python pipelines/sentinel_insar/download_snapflow_bursts.py
```

The downloader uses `phidown`, reads `CDSE_USERNAME` / `CDSE_PASSWORD` from the
environment or `.env`, extracts both ZIPs, and writes stable `master.SAFE` and
`slave.SAFE` links.

Then run this pipeline directly:

```bash
sarpyx-pipeline pipelines/sentinel_insar/sentinel_insar_A.yaml \
  --master data/bursts/extracted/master/8ff4f2b3-64d8-4852-8c3b-4b2b8f729b03/master.SAFE \
  --slave data/bursts/extracted/slave/2404a519-5e05-4dcc-95e5-b3e4e8a79127/slave.SAFE \
  --output data/output/sentinel_insar
```

Preview without running SNAP:

```bash
sarpyx-pipeline pipelines/sentinel_insar/sentinel_insar_A.yaml \
  --master /path/to/master-burst.SAFE \
  --slave /path/to/slave-burst.SAFE \
  --output data/output/sentinel_insar \
  --dry-run
```

Inputs must be extracted single-burst `.SAFE` directories. Full Sentinel-1
products must be split or downloaded as bursts first.

Pipeline choices:

```bash
--pipeline ifg    # default interferogram workflow
--pipeline gslc   # terrain-corrected complex stack
--pipeline both   # GSLC and IFG branches
```

If SNAP is not in the default location:

```bash
pipelines/sentinel_insar/run_sentinel_insar.sh \
  --master master.SAFE \
  --slave slave.SAFE \
  --gpt /Applications/snap/bin/gpt
```

The runner performs preflight checks and prints exactly what is missing before
starting a real run.
