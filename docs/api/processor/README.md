# Processor Module API

`sarpyx.processor` exposes low-level SAR processing primitives and implemented
data readers. It no longer exports placeholder high-level Range-Doppler,
backprojection, writer, formatter, or autofocus packages.

## Module Structure

```text
sarpyx.processor/
├── core/          # Focus, decode, transforms, constants, and sub-aperture helpers
├── algorithms/    # Implemented algorithm support modules only
├── data/          # Implemented readers
└── utils/         # Processing utilities
```

## Quick Start

```python
from sarpyx.processor import core, data, utils
from sarpyx.processor.data import read_tif, read_zarr_file

array = read_tif("amplitude.tif")
store = read_zarr_file("decoded.zarr")
```

## Core

`sarpyx.processor.core` lazily exposes implemented source modules:

- `focus`: coarse focusing utilities and `CoarseRDA`
- `decode`: Sentinel-1 decode and Zarr persistence helpers
- `transforms`: FFT and transform helpers
- `constants`: processing constants

```python
from sarpyx.processor.core import transforms

spectrum = transforms.perform_fft_custom(data, axis=-1)
```

## Algorithms

`sarpyx.processor.algorithms` currently keeps only implemented support modules,
including `constants` and `mbautofocus`. Removed placeholder modules such as
`rda` and `backprojection` are intentionally not exported.

```python
from sarpyx.processor.algorithms import mbautofocus
```

## Data

`sarpyx.processor.data` exports implemented reader helpers:

- `read_tif(path, verbose=False)`
- `read_zarr_file(path, array_or_group_key=None)`

Writer and formatter placeholders were removed. Use project-specific utilities
or external libraries such as `rasterio`, `h5py`, or `zarr` directly when you
need output formats that are not implemented here.

## Utilities

`sarpyx.processor.utils` contains processor-specific helper modules such as
memory reporting, summaries, unzip helpers, metrics, and visualization support.

## See Also

- [SLA Module](../sla/README.md): Sub-Look Analysis capabilities
- [Snapflow Module](../snapflow/README.md): SNAP GPT integration
- [Science Module](../science/README.md): Scientific index calculations
- [Utils Module](../utils/README.md): General utilities and visualization
