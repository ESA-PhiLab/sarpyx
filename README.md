<div align="center">

<img src="src/sarpyx_logo.png" width="1400px" alt="sarpyx">

<br />

<a href="docs/user_guide/README.md">
  <img alt="User Manual" src="https://img.shields.io/badge/Read-User%20Manual-111827?style=for-the-badge" />
</a>
<a href="docs/user_guide/getting_started.md">
  <img alt="Quick Start" src="https://img.shields.io/badge/Start-Quick%20Start-0f766e?style=for-the-badge" />
</a>
<a href="LICENSE">
  <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-374151?style=for-the-badge" />
</a>
</div>

##

**sarpyx** is a specialized Python toolkit for **Synthetic Aperture Radar (SAR)** processing with tight integration to ESA **SNAP**. The supported surface is centered on WorldSAR preprocessing and tiling, SNAP GPT orchestration, sub-aperture decomposition, H5/Zarr tile validation, and SAR science utilities.

## Highlights

- WorldSAR CLI for mission-specific preprocessing, tiling, validation reports, and H5-to-Zarr conversion.
- SNAP GPT integration with configurable graphs and operator chaining.
- Sub-aperture decomposition for Sentinel-style BEAM-DIMAP products.
- Geocoded H5/Zarr outputs ready for GIS and downstream ML.
- Utilities compatible with `rasterio`, `geopandas`, `pyproj`, `h5py`, `zarr`, and `dask`.

## Commands

Installed console scripts:

```bash
sarpyx --help          # WorldSAR preprocessing, tiling, validation, H5->Zarr
sarpyx-pipeline --help # YAML-configured SNAP pipelines
sarpyx-decode --help   # Sentinel-1 Level-0 decode wrapper
sarpyx-unzip --help    # Extract Sentinel-1 ZIP products
sarpyx-upload --help   # Upload artifacts to Hugging Face Hub
```

Reserved placeholder commands such as `sarpyx-focus` and `sarpyx-shipdet` are not shipped.

## Sentinel-1 burst InSAR

First download and extract the same burst pair selected by
`notebooks/snapflow_v2.ipynb`:

```bash
.venv/bin/python pipelines/sentinel_insar/download_snapflow_bursts.py
```

Then use the wrapper. It checks the inputs and SNAP GPT before any processing
starts, then runs the default interferogram branch.

```bash
pipelines/sentinel_insar/run_sentinel_insar.sh \
  --master data/bursts/extracted/master/8ff4f2b3-64d8-4852-8c3b-4b2b8f729b03/master.SAFE \
  --slave data/bursts/extracted/slave/2404a519-5e05-4dcc-95e5-b3e4e8a79127/slave.SAFE
```

Preview the exact steps without running SNAP:

```bash
pipelines/sentinel_insar/run_sentinel_insar.sh \
  --master /data/master-burst.SAFE \
  --slave /data/slave-burst.SAFE \
  --dry-run
```

Useful options:

```bash
--pipeline ifg    # default: coregister, deburst, interferogram, topo removal, terrain correction
--pipeline gslc   # coregister, deburst, terrain-correct complex stack
--pipeline both   # run GSLC and IFG branches
--gpt /path/to/gpt
```

The inputs must be extracted single-burst `.SAFE` directories, not full
Sentinel-1 products. The `snapflow_v2.ipynb` notebook selects and extracts the
burst pair.

## YAML-configured pipelines

For reusable SNAP workflows, use `sarpyx-pipeline` with versioned YAML configs:

```bash
uv run sarpyx-pipeline validate pipelines/sentinel_insar/sentinel_insar_A.yaml
uv run sarpyx-pipeline list pipelines/sentinel_insar/sentinel_insar_A.yaml
uv run sarpyx-pipeline pipelines/sentinel_insar/sentinel_insar_A.yaml \
  --master /data/master-burst.SAFE \
  --slave /data/slave-burst.SAFE \
  --output data/output/sentinel_insar
```

Example pipeline configs live under [`pipelines/`](pipelines/), including
[`pipelines/sentinel_insar/sentinel_insar_A.yaml`](pipelines/sentinel_insar/sentinel_insar_A.yaml).
See [docs/user_guide/config_pipelines.md](docs/user_guide/config_pipelines.md)
for the YAML schema, nested pipelines, pair inputs, dry-run, resume, and
overwrite behavior.

## Install

For container workflows, use the Docker Compose CLI plugin (`docker compose`) with full commands:

```bash
docker compose version
make recreate
```

<details open>
<summary><strong>Using uv (recommended)</strong></summary>

```bash
uv sync
```

For development, testing, and optional Copernicus tooling:

```bash
uv sync --group dev
uv sync --group dev --extra copernicus
uv run pytest -q
uv build
```
</details>

<details>
<summary><strong>Using conda with SNAP Engine</strong></summary>

This is the fastest local path when you need `snap-engine` ready:

```bash
conda create -n sarpyx python=3.12
conda activate sarpyx
conda install -c sirbastiano/label/dev -c conda-forge snap13=13.0.0
```
</details>

<details>
<summary><strong>Using pip (editable)</strong></summary>

```bash
python -m pip install -e .
```
</details>


## Docs

See [docs/user_guide/README.md](docs/user_guide/README.md) for usage and workflows, and [docs/developer_guide/contributing.md](docs/developer_guide/contributing.md) for contributor commands.

## Container grid configuration

At startup the container checks for grid files in this order:

1. `GRID_PATH` (or `grid_path`) if it points to an existing in-container `*.geojson`
2. First `*.geojson` found in `/workspace/grid`

If neither exists, the container exits with an error. Automatic grid generation
on startup has been removed.

To use a mounted grid:

```bash
mkdir -p ./grid
# put any grid GeoJSON here, e.g. ./grid/my_region.geojson
docker compose up
```

For direct `docker run`, pass an explicit in-container path when needed:

```bash
docker run --rm \
  -v "$PWD/grid:/workspace/grid:ro" \
  -e GRID_PATH=/workspace/grid/my_region.geojson \
  sirbastiano94/sarpyx:latest \
  /usr/local/bin/start-jupyter.sh
```

You can also pass `--grid-path` to the `worldsar` CLI command. For Sentinel preprocessing, pass `--sentinel-subaps N` to override the subaperture count; defaults are `2` for TOPS products and `3` for STRIP products, and `N` must be at least `2`.

##
<div align="center">

**With Love By:** Roberto Del Prete

</div>
