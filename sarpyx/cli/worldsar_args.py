"""Argument definitions shared by the WorldSAR CLI entry points."""

from __future__ import annotations

import argparse


DEFAULT_ZARR_CHUNK_SIZE = (32, 32)
DEFAULT_ORBIT_TYPE = 'Sentinel Precise (Auto Download)'


def add_worldsar_arguments(parser: argparse.ArgumentParser) -> None:
    """Add WorldSAR arguments to *parser* without importing processing dependencies."""
    parser.add_argument(
        '--input',
        '-i',
        dest='product_path',
        type=str,
        required=True,
        help='Path to the input SAR product.'
    )
    parser.add_argument(
        '--output',
        '-o',
        dest='output_dir',
        type=str,
        required=False,
        default=None,
        help='Processed output directory, or target .zarr path in --h5-to-zarr-only mode.'
    )
    parser.add_argument(
        '--cuts-outdir',
        '--cuts_outdir',
        dest='cuts_outdir',
        type=str,
        required=False,
        default=None,
        help='Where to store the tiles after extraction.'
    )
    parser.add_argument(
        '--product-wkt',
        '--product_wkt',
        dest='product_wkt',
        type=str,
        required=False,
        default=None,
        help='WKT string defining the product region of interest.'
    )
    parser.add_argument(
        '--h5-to-zarr-only',
        dest='h5_to_zarr_only',
        action='store_true',
        help='Skip preprocessing/tiling and convert an existing .h5 tile into a Zarr v3 store.'
    )
    parser.add_argument(
        '--zarr-chunk-size',
        dest='zarr_chunk_size',
        type=int,
        nargs=2,
        metavar=('ROWS', 'COLS'),
        default=DEFAULT_ZARR_CHUNK_SIZE,
        help='Chunk size for H5-to-Zarr conversion (default: 32 32).'
    )
    parser.add_argument(
        '--overwrite-zarr',
        dest='overwrite_zarr',
        action='store_true',
        help='Replace an existing output Zarr store when converting H5 tiles.'
    )
    parser.add_argument(
        '--gpt-path',
        dest='gpt_path',
        type=str,
        default=None,
        help='Override GPT executable path (default: gpt_path env var).'
    )
    parser.add_argument(
        '--grid-path',
        dest='grid_path',
        type=str,
        default=None,
        help='Override grid GeoJSON path (default: grid_path env var).'
    )
    parser.add_argument(
        '--db-dir',
        dest='db_dir',
        type=str,
        default=None,
        help='Override database output directory (default: db_dir env var).'
    )
    parser.add_argument(
        '--gpt-memory',
        dest='gpt_memory',
        type=str,
        default=None,
        help='Override GPT Java heap (e.g., 24G).'
    )
    parser.add_argument(
        '--gpt-parallelism',
        dest='gpt_parallelism',
        type=int,
        default=None,
        help='Override GPT parallelism (number of tiles).'
    )
    parser.add_argument(
        '--gpt-timeout',
        dest='gpt_timeout',
        type=int,
        default=None,
        help='Override GPT timeout in seconds for a single invocation.'
    )
    parser.add_argument(
        '--snap-userdir',
        dest='snap_userdir',
        type=str,
        default=None,
        help='Override SNAP user directory.'
    )
    parser.add_argument(
        '--orbit-type',
        dest='orbit_type',
        type=str,
        default=DEFAULT_ORBIT_TYPE,
        help='SNAP Apply-Orbit-File orbitType.'
    )
    parser.add_argument(
        '--orbit-continue-on-fail',
        dest='orbit_continue_on_fail',
        action='store_true',
        help='Continue if orbit file cannot be applied.'
    )
    parser.add_argument(
        '--skip-preprocessing',
        dest='skip_preprocessing',
        action='store_true',
        help='Skip TC preprocessing and reuse existing BEAM-DIMAP intermediate products for tiling.'
    )
