# Processor Module API

The `sarpyx.processor` package contains SAR processing primitives, algorithms,
data helpers, and utility functions. It is imported lazily so lightweight CLI
commands do not load optional scientific dependencies until needed.

## Module Structure

```text
sarpyx.processor/
├── core/          # focus, decode, transforms, metadata, sub-aperture helpers
├── algorithms/    # RDA, back-projection, and model-based autofocus modules
├── data/          # readers, writers, and formatters
└── utils/         # memory, metrics, summaries, unzip, and visualization helpers
```

## Quick Imports

```python
from sarpyx import processor
from sarpyx.processor import core, algorithms, data, utils
from sarpyx.processor.core import focus, transforms, subaperture
from sarpyx.processor.utils import metrics, summary, viz
```

## Core Submodules

### `core`

Core processing routines for decoding, focusing, signal transforms, metadata
handling, and sub-aperture operations.

Common modules:

- `focus`: range and azimuth focusing helpers.
- `decode`: Sentinel-1 Level-0 packet/header decoding helpers.
- `transforms`: Fourier and mathematical transforms.
- `subaperture` and `subaperture_full_img`: sub-aperture decomposition helpers.
- `code2physical`: packet-field extraction and validation helpers.

### `algorithms`

Higher-level processing algorithms and research implementations:

- `rda`: Range-Doppler Algorithm entry points.
- `backprojection`: back-projection algorithm entry points.
- `mbautofocus`: model-based autofocus components.

### `data`

Data I/O and format helpers:

- `readers`: reader utilities.
- `writers`: writer utilities.
- `formatters`: format conversion helpers.

### `utils`

Processing support utilities:

- `metrics`: image-similarity metrics such as SSIM and PSNR.
- `summary`: array summary helpers.
- `mem`: memory cleanup helpers.
- `unzip`: archive extraction helpers.
- `viz`: plotting helpers.

## Examples

### Image Similarity Metrics

```python
import numpy as np
from sarpyx.processor.utils import metrics

reference = np.random.random((256, 256))
candidate = reference + np.random.normal(0, 0.01, reference.shape)

ssim_value = metrics.ssim(reference, candidate)
psnr_value = metrics.psnr(reference, candidate)

print(f"SSIM: {float(ssim_value):.3f}")
print(f"PSNR: {float(psnr_value):.3f}")
```

### Sub-Aperture Processing Entry Points

```python
from sarpyx.processor.core import subaperture

# See module API pages for function signatures and input product requirements.
print(subaperture.__name__)
```

### Range-Doppler and Back-Projection Modules

```python
from sarpyx.processor.algorithms import rda, backprojection

print(rda.__all__)
print(backprojection.__all__)
```

## Related Documentation

- [SLA module](../sla/README.md): Sub-Look Analysis capabilities.
- [Snapflow module](../snapflow/README.md): SNAP GPT preprocessing and pipeline
  orchestration.
- [Science module](../science/README.md): Scientific analysis tools.
- [Utils module](../utils/README.md): General utilities and visualization.
