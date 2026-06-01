# sarpyx Examples

This directory contains example scripts that mirror the files currently shipped
in the repository. Some examples are exploratory reference material and may need
local data paths or dependency adjustments before execution.

## Files

```text
docs/examples/
├── basic_sublook_analysis.py
├── basic/
│   ├── data_io_examples.py
│   ├── snap_integration.py
│   └── visualization_gallery.py
├── intermediate/
│   ├── polarimetric_analysis.py
│   ├── quality_assessment.py
│   └── vegetation_monitoring.py
└── advanced/
    ├── batch_processing.py
    ├── custom_processing_chains.py
    ├── insar_time_series.py
    └── performance_optimization.py
```

## Current Public Entry Points

Use these package APIs when adapting examples:

```python
from sarpyx.sla import SubLookAnalysis
from sarpyx.snapflow.engine import GPT
from sarpyx.science.indices import calculate_rvi, calculate_ndpoll
from sarpyx.processor.data import read_tif, read_zarr_file
```

## Running Examples

Run scripts from the repository root so relative imports resolve against the
local checkout:

```bash
uv run python docs/examples/basic_sublook_analysis.py
uv run python docs/examples/basic/snap_integration.py
```

SNAP examples require a working ESA SNAP installation and a configured GPT
executable. Data-intensive examples require local Sentinel-1 or compatible SAR
products; sample datasets are not committed to this repository.
