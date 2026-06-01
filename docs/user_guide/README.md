# User Guide

This guide provides comprehensive information for users of sarpyx, from installation to advanced processing workflows.

## Table of Contents

1. [Installation](installation.md)
2. [Getting Started](getting_started.md)
3. [Basic Concepts](basic_concepts.md)
4. [Data Formats](data_formats.md)
5. [Processing Workflows](processing_workflows.md)
6. [Science Applications](science_applications.md)
7. [SNAP Integration](snap_integration.md)
8. [Troubleshooting](troubleshooting.md)
9. [SNAP2StaMPS Guide](snap2stamps_guide.md)

## Overview

sarpyx is designed to provide researchers and developers with powerful tools for SAR data processing. The package is organized into several main modules:

- **`sarpyx.cli.worldsar`**: WorldSAR preprocessing, tiling, validation, H5-to-Zarr conversion
- **`sarpyx.snapflow`**: ESA SNAP GPT integration and SNAP2StaMPS-style pipeline orchestration
- **`sarpyx.processor.core`**: implemented decode, focus primitives, signal helpers, and sub-aperture processing
- **`sarpyx.science`**: implemented SAR index formulas
- **`sarpyx.utils`**: geospatial, H5/Zarr, DEM, plotting, metrics, and upload utilities

## Quick Navigation

- **New to SAR processing?** Start with [Basic Concepts](basic_concepts.md)
- **Want to dive in quickly?** Check out [Getting Started](getting_started.md)
- **Need specific processing workflows?** See [Processing Workflows](processing_workflows.md)
- **Working with vegetation indices?** Explore [Science Applications](science_applications.md)

## Prerequisites

- Basic knowledge of Python programming
- Understanding of SAR imaging principles (recommended)
- Familiarity with NumPy and scientific Python ecosystem

## Support

If you encounter any issues or have questions, please:

1. Check the [Troubleshooting](troubleshooting.md) section
2. Search existing [GitHub Issues](https://github.com/ESA-PhiLab/sarpyx/issues)
3. Create a new issue if your problem isn't covered
