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

**sarpyx** is a specialized Python toolkit for **Synthetic Aperture Radar (SAR)** processing with tight integration to ESA **SNAP**. It focuses on reproducible pipelines, fast tiling workflows, and advanced research features like **sub-aperture decomposition**.

## Documentation

- [Documentation site](https://esa-philab.github.io/sarpyx/)
- [Installation guide](docs/user_guide/installation.md)
- [CLI usage examples](docs/user_guide/cli_examples.md)
- [User guide](docs/user_guide/README.md)
- [API reference](docs/api/README.md)

## Quick Reference

<summary><strong>Using conda (preferred, avoids global SNAP installs)</strong></summary>

The recommended installation uses conda first to provide ESA SNAP and `gpt`, then installs `sarpyx` with `pip` from this checkout. This keeps SNAP/native dependencies isolated in the environment while keeping the Python package editable.

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
sarpyx-pipeline --help
```

## Project Links

- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Citation metadata](CITATION.cff)
- [License](LICENSE)
