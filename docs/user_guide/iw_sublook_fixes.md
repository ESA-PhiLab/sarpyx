# Selecting SAFE annotations for full-image IW sublooks

Full-image IW processing now matches annotation files to the processed swath and polarization, rather than relying on filesystem enumeration order. The public `do_subaps` function and `CombinedSublooking` constructor accept optional `safe_swath` and `safe_polarization` arguments:

```python
from sarpyx.processor.core.subaperture_full_img import do_subaps

do_subaps(
    dim_path="IW2_VV_PRE.dim",
    safe_path="acquisition.SAFE",
    numberofLooks=3,
    tops_iw_mode=True,
    safe_swath="IW2",
    safe_polarization="VV",
)
```

Selectors are inferred from recognizable I/Q filenames such as `i_IW2_VV.img` and `q_IW2_VV.img`. Explicit selectors must agree with any information identified in either filename. A `safe_polarization` argument selects the matching polarization from a multi-polarization product; without that argument each detected pair uses its own inferred polarization. The product must already represent the requested subswath: a conflicting swath is an error. Names such as `i_VV.img` do not identify a swath, so a multi-swath SAFE generally requires an explicit `safe_swath`.

All selected pairs are checked for a unique annotation before sublook outputs are written. Missing or ambiguous matches raise an error. `CombinedSublooking.annotation_file` records the selected XML and it is logged during processing.

The shared metadata `Handler` also accepts selectors and `require_unique=True`. Full-image processing always requires uniqueness. Legacy callers that supply no selectors retain the previous first-file behavior, with a warning if multiple annotations are present; callers should migrate to explicit selectors for reproducible metadata association.

Two spectral implementation corrections accompany this selection change: Dask inverse shifts operate on the complete azimuth axis, and residual Doppler rolls move the support bounds with the spectrum, preserving support across a circular boundary. The residual Doppler estimator and the final azimuth flip retain their definitions. These fixes do not establish the physical accuracy of the estimated residual Doppler shift or point-target cross-look coherence.

Focused regressions can be run from the source root with:

```bash
python -m unittest discover -s tests -p 'test_iw_regressions.py' -v
```
