# Tutorials

Step-by-step tutorials for learning sarpyx capabilities and SAR processing
techniques.

## Available Tutorials

### Getting Started

- [Tutorial 1: Your First Sub-Look Analysis](01_first_sublook_analysis.md)
  introduces sub-look decomposition with Sentinel-1 style inputs.
- [Tutorial 2: SNAP Integration Basics](02_snap_integration_basics.md)
  introduces SNAP GPT automation through sarpyx.
- [Tutorial 3: Visualization and Quality Assessment](03_visualization_quality.md)
  covers visual checks, plots, and quality review.

### Intermediate Workflows

- [Tutorial 4: Multi-temporal Analysis](04_multitemporal_analysis.md)
  demonstrates time-series analysis for change detection.
- [Tutorial 5: Polarimetric Analysis](05_polarimetric_analysis.md)
  covers dual-pol and polarimetric analysis concepts.
- [Tutorial 6: Custom Processing Workflows](06_custom_workflows.md)
  shows how to compose custom SAR processing chains.

### Advanced Workflows

- [Tutorial 7: Ship Detection](07_ship_detection.md)
  presents CFAR-based maritime detection algorithms and workflow structure.
- [Tutorial 8: Interferometric Analysis](08_interferometric_analysis.md)
  covers InSAR concepts, deformation analysis, and time-series patterns.

Tutorials 7 and 8 are advanced methodological guides. They include algorithm
sketches that may need adaptation to local helper classes and data products.

## Required Data

Most tutorials use placeholders for SAR products. Replace them with local
Sentinel-1, TerraSAR-X, COSMO-SkyMed, Biomass, NISAR, or compatible products
before running the snippets.

SNAP tutorials require ESA SNAP GPT:

```bash
gpt -h
```

Repository commands can be run through `uv` without activating the environment:

```bash
uv run sarpyx --help
uv run sarpyx worldsar --help
uv run sarpyx pipeline --list
```

## Learning Paths

### New SAR Users

1. [Tutorial 1](01_first_sublook_analysis.md)
2. [Tutorial 3](03_visualization_quality.md)
3. [Tutorial 2](02_snap_integration_basics.md)
4. [Tutorial 5](05_polarimetric_analysis.md)

### SNAP Users

1. [Tutorial 2](02_snap_integration_basics.md)
2. [Tutorial 6](06_custom_workflows.md)
3. [Tutorial 7](07_ship_detection.md)
4. [Pipeline CLI guide](../user_guide/pipeline_cli.md)

### Research Workflows

1. [Tutorial 4](04_multitemporal_analysis.md)
2. [Tutorial 5](05_polarimetric_analysis.md)
3. [Tutorial 8](08_interferometric_analysis.md)
4. [Science applications](../user_guide/science_applications.md)

## Interactive Formats

Notebook versions and companion notebooks live in [../notebooks](../notebooks):

- [01_getting_started.ipynb](../notebooks/01_getting_started.ipynb)
- [02_snap_integration.ipynb](../notebooks/02_snap_integration.ipynb)
- [03_insar_processing.ipynb](../notebooks/03_insar_processing.ipynb)
- [04_time_series_analysis.ipynb](../notebooks/04_time_series_analysis.ipynb)

## See Also

- [Examples](../examples/README.md)
- [API reference](../api/README.md)
- [User guide](../user_guide/README.md)
- [Developer guide](../developer_guide/README.md)
