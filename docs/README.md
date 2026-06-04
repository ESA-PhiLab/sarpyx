# sarpyx Documentation

Welcome to the sarpyx documentation. sarpyx is a Python package for Synthetic Aperture Radar (SAR) processing focused on the currently implemented WorldSAR/SNAP workflow surface: product preprocessing, tiling, sub-aperture generation, H5/Zarr validation, and selected science utilities.

<p align="center">
    <img src="../assets/sarpyx_logo.png" alt="sarpyx logo" width="1200"/>
</p>

## Table of Contents

### 📚 [User Guide](user_guide/README.md)
Complete guide for getting started with sarpyx, including installation, basic concepts, and common workflows.

### 🎯 [Tutorials](tutorials/README.md)
Step-by-step tutorials covering various SAR processing techniques and real-world applications.

### 💻 [Examples](examples/README.md)
Ready-to-run code examples demonstrating key features and processing workflows.

### 🔧 [API Reference](api/README.md)
Comprehensive API documentation for all modules, classes, and functions.

### 👩‍💻 [Developer Guide](developer_guide/README.md)
Information for developers contributing to sarpyx, including architecture, coding standards, and contribution guidelines.

### 📖 [MetaParams Reference](metaParams.md)
SAR metadata parameter documentation and external resources.

## Quick Start

```python
import sarpyx

# Example: Calculate vegetation indices from Sentinel-1 data
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll

# Load your SAR data (VV and VH polarizations)
sigma_vv = your_vv_data  # Linear scale backscatter coefficients
sigma_vh = your_vh_data  # Linear scale backscatter coefficients

# Calculate Radar Vegetation Index
rvi = calculate_rvi(sigma_vv, sigma_vh)

# Calculate Normalized Difference Polarization Index
ndpoll = calculate_ndpoll(sigma_vv, sigma_vh)
```

## Key Features

- **WorldSAR CLI**: mission-specific preprocessing, tiling, validation reports, and H5-to-Zarr conversion
- **SNAP Engine Integration**: direct interface with SNAP Graph Processing Tool (GPT)
- **Sub-Aperture Decomposition**: implemented Sentinel-style BEAM-DIMAP sub-aperture workflows
- **Data Compatibility**: Sentinel-1, TerraSAR-X, NISAR helper paths, H5, Zarr, BEAM-DIMAP, GeoTIFF
- **Geospatial Utilities**: helpers built around rasterio, geopandas, shapely, pyproj, h5py, zarr, and dask

## Installation

```bash
pip install sarpyx
```

For development installation:
```bash
git clone https://github.com/ESA-PhiLab/sarpyx.git
cd sarpyx
pip install -e .
```

For the fastest local setup with `snap-engine` ready:

```bash
conda create -n sarpyx python=3.12
conda activate sarpyx
conda install -c sirbastiano/label/dev -c conda-forge snap13=13.0.0
```

### Container note

For container workflows, mount or otherwise provide any `*.geojson` grid in `/workspace/grid` (or set `GRID_PATH` to a specific in-container `*.geojson`). The entrypoint loads an existing grid and now fails fast if none is available; it no longer generates one automatically.

## Support and Community

- 📧 **Issues**: [GitHub Issues](https://github.com/ESA-PhiLab/sarpyx/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ESA-PhiLab/sarpyx/discussions)
- 📖 **Documentation**: [Full Documentation](https://sarpyx.readthedocs.io)

## License

sarpyx is released under the Apache 2.0 License. See the [LICENSE](../LICENSE) file for details.
