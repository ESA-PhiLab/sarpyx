# Installation Guide

This page is the detailed setup reference for local development, SNAP-backed processing, and containers.

## Requirements

- Python `>=3.11, <4`.
- `uv` for the repository-managed environment.
- Git for editable installs.
- ESA SNAP and Java for SNAP GPT workflows.
- Docker Engine and the Docker Compose CLI plugin for container workflows.
- A grid `*.geojson` file for WorldSAR tiling and container startup.

## Local Development with uv

Use this path for development, tests, and Python workflows that do not need a system SNAP install.

```bash
git clone https://github.com/ESA-PhiLab/sarpyx.git
cd sarpyx
uv sync --group dev
uv run sarpyx --help
uv run sarpyx pipeline --list
uv run pytest -q
```

Install the optional Copernicus tooling when you need phidown-backed data acquisition:

```bash
uv sync --group dev --extra copernicus
```

## Editable pip Install

Use this path when integrating `sarpyx` into an existing virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
sarpyx --help
```

For tests:

```bash
python -m pip install pytest
pytest -q
```

## Conda with ESA SNAP

Use this path for SNAP GPT processing outside Docker. The conda environment isolates Python, native geospatial dependencies, Java, and SNAP.

```bash
conda create -n sarpyx-snap -c sirbastiano/label/dev -c conda-forge \
  python=3.12 pip snap13=13.0.0

conda activate sarpyx-snap
python -m pip install -e .
```

Set SNAP runtime variables after activating the environment:

```bash
source scripts/setvars.sh
"$GPT_PATH" --help
sarpyx worldsar --help
```

`scripts/setvars.sh` exports `SNAP_HOME`, `GPT_PATH`, `gpt_path`, `SNAP_USERDIR`, `snap_userdir`, and updates `PATH`.

## Manual SNAP Install

If SNAP is installed outside conda, set `GPT_PATH` explicitly. It must point to SNAP's `gpt` executable.

```bash
export GPT_PATH=/Applications/esa-snap/bin/gpt
export SNAP_USERDIR="$PWD/.snap"
sarpyx worldsar --help
"$GPT_PATH" --help
```

Typical `gpt` paths:

| Setup | Path |
| --- | --- |
| Conda SNAP package | `$CONDA_PREFIX/opt/esa-snap/bin/gpt` |
| Docker image | `/workspace/snap13/bin/gpt` or `/usr/local/bin/gpt` |
| macOS manual install | `/Applications/snap/bin/gpt` or `/Applications/esa-snap/bin/gpt` |
| Linux manual install | `$HOME/ESA-STEP/snap/bin/gpt`, `/opt/snap/bin/gpt`, or `/usr/local/snap/bin/gpt` |
| Windows manual install | `C:\Program Files\snap\bin\gpt.exe` |

WorldSAR resolves GPT in this order:

1. Explicit `--gpt-path`.
2. `gpt_path` or `GPT_PATH`.
3. `$CONDA_PREFIX/opt/esa-snap/bin/gpt`.
4. `$SNAP_HOME/bin/gpt`.
5. A SNAP-looking `gpt` on `PATH`.

## Docker

The Docker image includes SNAP and links `gpt` into the runtime path. Container startup requires an existing grid file; it does not generate one automatically.

```bash
docker compose version
mkdir -p grid
# place a grid GeoJSON in ./grid, for example ./grid/my_region.geojson
make check-grid
make recreate
```

To select a specific mounted grid:

```bash
GRID_PATH="$PWD/grid/my_region.geojson" make recreate
```

For direct Docker use, pass an in-container grid path:

```bash
docker run --rm \
  -v "$PWD/grid:/workspace/grid:ro" \
  -e GRID_PATH=/workspace/grid/my_region.geojson \
  sirbastiano94/sarpyx:latest \
  sarpyx worldsar --help
```

## Verify the Installation

Run the checks that match your setup:

```bash
uv run python -c "import sarpyx; print(sarpyx.__version__)"
uv run sarpyx --help
uv run sarpyx worldsar --help
uv run sarpyx pipeline --list
```

For SNAP-backed runs:

```bash
test -x "$GPT_PATH"
"$GPT_PATH" --help
sarpyx worldsar \
  --input /data/product.SAFE \
  --output /tmp/sarpyx-check \
  --grid-path /data/grid.geojson \
  --gpt-path "$GPT_PATH" \
  --help
```

## Troubleshooting

- `CONDA_PREFIX is not set`: activate the conda environment before `source scripts/setvars.sh`.
- `SNAP GPT not found or not executable`: install SNAP, export `GPT_PATH`, or pass `--gpt-path`.
- Wrong system `gpt` is found: pass an absolute `--gpt-path`; WorldSAR rejects common non-SNAP executables.
- SNAP cache or permission errors: set `SNAP_USERDIR` to a writable run-local directory and pass `--snap-userdir`.
- Grid errors in Docker: set `GRID_PATH` or mount a `*.geojson` file under `/workspace/grid`.
- Large product failures: lower `--gpt-parallelism`, increase `--gpt-memory` or `--gpt-timeout`, and keep `SNAP_USERDIR` on fast local storage.

## Next Steps

- [CLI usage examples](cli_examples.md)
- [Generic Pipeline CLI](pipeline_cli.md)
- [SNAP Integration](snap_integration.md)
- [Troubleshooting](troubleshooting.md)
