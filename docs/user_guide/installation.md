# Installation Guide

This guide will help you install sarpyx and its dependencies on your system.

## Prerequisites

Before installing sarpyx, ensure you have:

- Conda or Mamba
- Python 3.11 or higher
- pip package manager
- Git (for development installation)

### System-specific Requirements

#### For SNAP Integration (Optional)
- ESA SNAP Desktop (version 9.0 or higher)
- Java Runtime Environment (JRE) 11 or higher
- At least 8GB of RAM recommended

#### For Scientific Computing
- NumPy compatible system
- GDAL libraries (for geospatial data handling)

## Installation Methods

### 1. Conda environment, then pip install sarpyx

This is the primary local installation method. Conda manages the Python runtime
and native/SNAP dependencies; pip installs the `sarpyx` Python package from the
repository checkout.

```bash
git clone https://github.com/ESA-PhiLab/sarpyx.git
cd sarpyx

conda create -n sarpyx -c sirbastiano/label/dev -c conda-forge \
  python=3.12 pip snap13=13.0.0
conda activate sarpyx
python -m pip install -e .
```

Verify the console scripts:

```bash
sarpyx --help
sarpyx-pipeline --help
```

For Copernicus burst downloads and development checks:

```bash
python -m pip install -e ".[copernicus]"
python -m pip install pytest
pytest -q
```

### 2. Published pip package

For environments that already provide the native geospatial and SNAP
dependencies:

```bash
python -m pip install sarpyx
```

### 3. Using uv for repository maintenance

`uv` is still useful for CI-style syncs, test runs, and builds:

```bash
uv sync --group dev
uv sync --group dev --extra copernicus
uv run pytest -q
uv build
```

The conda plus pip flow above remains the default user-facing install path.

### 4. Containerized execution

If you run `sarpyx` via the provided container, the entrypoint uses this order:

1. `GRID_PATH` (or `grid_path`) if it points to an existing in-container
   `*.geojson`
2. first `*.geojson` found under `/workspace/grid`

If neither is available, the container exits with an error. Startup-time grid
generation has been removed.

To use your own mounted grid, place any GeoJSON in `./grid`:

```bash
mkdir -p ./grid
# Example: cp my_region.geojson ./grid/
docker compose up
```

If you use `docker-compose`, mount the grid directory and optionally set
`GRID_PATH` to choose a specific file:

```bash
- ./grid:/workspace/grid:ro
- GRID_PATH=/workspace/grid/my_region.geojson
```

## Verifying Installation

To verify your installation works correctly:

```python
import sarpyx
print(f"sarpyx version: {sarpyx.__version__}")

# Test basic functionality
from sarpyx.utils import show_image
from sarpyx.sla import SubLookAnalysis
print("Installation successful!")
```

For the primary conda/pip environment:

```bash
python -c "import sarpyx; print(sarpyx.__version__)"
sarpyx --help
```

## Optional Dependencies

### SNAP Integration
For full SNAP functionality, install ESA SNAP:

1. Download from [ESA SNAP website](https://step.esa.int/main/download/snap-download/)
2. Follow platform-specific installation instructions
3. Ensure `gpt` command is available in your PATH

The primary conda command installs the packaged SNAP engine:

```bash
conda create -n sarpyx -c sirbastiano/label/dev -c conda-forge \
  python=3.12 pip snap13=13.0.0
conda activate sarpyx
```

### Jupyter Support
For interactive notebooks:
```bash
python -m pip install jupyter matplotlib ipywidgets
```

### Visualization
For enhanced plotting capabilities:
```bash
python -m pip install matplotlib seaborn plotly
```

## Troubleshooting

### Common Issues

#### GDAL Installation Problems
On some systems, GDAL can be challenging to install:

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
python -m pip install gdal
```

**macOS (using Homebrew):**
```bash
brew install gdal
python -m pip install gdal
```

**Windows:**
Consider using conda:
```bash
conda install -c conda-forge gdal
```

#### Memory Issues
For large SAR datasets, ensure sufficient memory:
- Minimum 8GB RAM
- 16GB+ recommended for large-scale processing

#### Permission Errors
On Unix systems, you might need:
```bash
python -m pip install --user sarpyx
```

### Getting Help

If you encounter installation issues:

1. Check [GitHub Issues](https://github.com/ESA-PhiLab/sarpyx/issues)
2. Create a new issue with:
   - Your operating system
   - Python version
   - Complete error message
   - Installation method used

## Next Steps

After successful installation:

1. Read [Getting Started](getting_started.md)
2. Explore [Basic Concepts](basic_concepts.md)
3. Try the [Examples](../examples/README.md)

## Environment Setup

### Virtual Environment

Create an isolated environment for sarpyx:

```bash
conda create -n sarpyx -c sirbastiano/label/dev -c conda-forge \
  python=3.12 pip snap13=13.0.0
conda activate sarpyx
python -m pip install -e .
```

### Development Environment

For development work, add test tooling in the same conda environment:

```bash
python -m pip install -e ".[copernicus]"
python -m pip install pytest
pytest -q
```
