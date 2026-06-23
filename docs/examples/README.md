# sarpyx Examples

This directory contains runnable examples and larger workflow sketches for the
current sarpyx package layout. Paths below are relative to `docs/examples/`.

## Available Examples

### Basic

- [basic_sublook_analysis.py](basic_sublook_analysis.py): minimal sub-look
  analysis workflow.
- [basic/snap_integration.py](basic/snap_integration.py): SNAP GPT automation
  basics.
- [basic/visualization_gallery.py](basic/visualization_gallery.py): plotting and
  image-display helpers.
- [basic/data_io_examples.py](basic/data_io_examples.py): data loading and
  format examples.

### Intermediate

- [intermediate/vegetation_monitoring.py](intermediate/vegetation_monitoring.py):
  dual-polarization vegetation metrics and time-series examples.
- [intermediate/polarimetric_analysis.py](intermediate/polarimetric_analysis.py):
  polarimetric feature extraction and decomposition examples.
- [intermediate/ship_detection_cfar.py](intermediate/ship_detection_cfar.py):
  CFAR-style maritime target detection.
- [intermediate/quality_assessment.py](intermediate/quality_assessment.py):
  processing quality checks and metrics.

### Advanced

- [advanced/insar_time_series.py](advanced/insar_time_series.py): InSAR
  time-series analysis patterns.
- [advanced/custom_processing_chains.py](advanced/custom_processing_chains.py):
  custom workflow composition.
- [advanced/batch_processing.py](advanced/batch_processing.py): batch execution
  and reporting.
- [advanced/performance_optimization.py](advanced/performance_optimization.py):
  memory, chunking, and acceleration patterns.

## Related Notebooks

Notebook tutorials live in [../notebooks](../notebooks):

- [01_getting_started.ipynb](../notebooks/01_getting_started.ipynb)
- [02_snap_integration.ipynb](../notebooks/02_snap_integration.ipynb)
- [03_insar_processing.ipynb](../notebooks/03_insar_processing.ipynb)
- [04_time_series_analysis.ipynb](../notebooks/04_time_series_analysis.ipynb)

## Running Examples

Install sarpyx with the extras needed by the example you want to run:

```bash
python -m pip install -e ".[geo,io,processing,viz]"
```

SNAP-backed examples also require ESA SNAP GPT on `PATH` or a configured
`--gpt-path`:

```bash
gpt -h
```

Run examples from the repository root:

```bash
python docs/examples/basic_sublook_analysis.py
python docs/examples/basic/snap_integration.py
python docs/examples/intermediate/quality_assessment.py
```

Some examples use synthetic data when no input is provided. Examples that need
real SAR products document the expected input path in the script docstring or
command-line help.

## Learning Path

1. Start with [basic_sublook_analysis.py](basic_sublook_analysis.py).
2. Review [basic/visualization_gallery.py](basic/visualization_gallery.py).
3. Move to [basic/snap_integration.py](basic/snap_integration.py) if SNAP GPT is
   installed.
4. Use the intermediate and advanced examples as workflow templates for your
   own data and runtime environment.

## See Also

- [User guide](../user_guide/README.md)
- [Tutorials](../tutorials/README.md)
- [API reference](../api/README.md)
