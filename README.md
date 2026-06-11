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
<a href="https://github.com/ESA-PhiLab/sarpyx/releases/tag/v1.0.0">
  <img alt="Version" src="https://img.shields.io/badge/Version-1.0.0-2563eb?style=for-the-badge" />
</a>
</div>

##

**sarpyx** is a specialized Python toolkit for **Synthetic Aperture Radar (SAR)** processing with tight integration to ESA **SNAP**. It focuses on reproducible SAR workflows, SNAP GPT orchestration, fast tiling, validation, and research features such as **sub-aperture decomposition**.

## Highlights

- SNAP v13 ported into a conda package no needing additional installations.
- SNAP GPT integration with configurable graphs and operator chaining.
- WorldSAR preprocessing, tiling, validation, and H5-to-Zarr conversion.
- Generic pipeline CLI with built-in recipes for Sentinel-1, TSX, CSG, Biomass, NISAR, and Sentinel-1 InSAR.
- Sub-aperture decomposition for Sentinel-style BEAM-DIMAP products.
- Geocoded outputs ready for GIS and downstream ML.
- Utilities compatible with `rasterio`, `geopandas`, `pyproj`, `h5py`, `zarr`, and `dask`.

## Commands

```bash
sarpyx --help           # Top-level command dispatcher
sarpyx worldsar --help  # WorldSAR preprocessing, tiling, validation, H5-to-Zarr
sarpyx pipeline --help  # Explicit built-in or external pipeline recipes
sarpyx-worldsar --help  # Compatibility WorldSAR entry point
sarpyx-pipeline --help  # Compatibility pipeline entry point
```

## Documentation

- [Documentation site](https://esa-philab.github.io/sarpyx/)
- [Installation guide](docs/user_guide/installation.md)
- [CLI usage examples](docs/user_guide/cli_examples.md)
- [User guide](docs/user_guide/README.md)
- [API reference](docs/api/README.md)

## Install

The recommended installation uses **conda first** to provide ESA SNAP and `gpt`, then installs `sarpyx` with pip from this checkout. This keeps SNAP/native dependencies managed by conda while keeping the Python package editable.

```bash
conda create -n sarpyx -c sirbastiano/label/dev -c conda-forge \
  python=3.12 pip snap13=13.0.0

conda activate sarpyx
python -m pip install -e .
```

Verify the installation:

```bash
gpt -h
sarpyx --help
sarpyx worldsar --help
sarpyx pipeline --help
```

For development and tests:

```bash
python -m pip install -e ".[copernicus]"
python -m pip install pytest
pytest -q
```

<details>
<summary><strong>Using uv for repository maintenance</strong></summary>

```bash
uv sync
uv sync --group dev
uv sync --group dev --extra copernicus
uv run pytest -q
uv build
```

</details>

<details>
<summary><strong>Published pip package</strong></summary>

```bash
python -m pip install sarpyx
```

The pip package is suitable for Python-side usage, but SNAP GPT workflows require a working SNAP installation available in the environment.

</details>

## Container usage

For container workflows, use the Docker Compose CLI plugin:

```bash
docker compose version
make recreate
```

At startup, the container expects a mounted grid file. Provide either `GRID_PATH` or place a `*.geojson` file under `/workspace/grid`.

```bash
mkdir -p ./grid
# put any grid GeoJSON here, e.g. ./grid/my_region.geojson
docker compose up
```

You can also pass `--grid-path` to `sarpyx worldsar`.

## Project Links

- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Citation metadata](CITATION.cff)
- [JOSS paper draft](paper.md)
- [License](LICENSE)

##

<div align="center">

**Maintainers:** Roberto Del Prete, Gabriele Daga, Sebastian Fieldhouse, Juanfrancisco Amieva, Cedric Leonard, Valerio Marsocci, Eva Gmelich Mejling

</div>
