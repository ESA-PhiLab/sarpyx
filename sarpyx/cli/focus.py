#!/usr/bin/env python3
"""
Focus CLI Tool for SARPyX.

This module provides a command-line interface for focusing SAR data using
the CoarseRDA (Range-Doppler Algorithm) processor.
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging

from sarpyx.processor.core.focus import CoarseRDA
from sarpyx.utils.zarr_utils import ZarrManager, dask_slice_saver, concatenate_slices_efficient
from sarpyx.utils.io import calculate_slice_indices
from .utils import validate_path, create_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__BUFFER_SLICE_HEIGHT__ = 15000  # Buffer size for each slice in rows


def _parse_chunk_shape(chunk_shape: str) -> Optional[Union[str, Tuple[int, ...]]]:
    """
    Parse a chunk-shape CLI value.

    Supported values:
      - 'auto'
      - 'none' (or empty) to use default behavior in saver
      - comma-separated positive integers, e.g. "2048,2048"
    """
    if chunk_shape is None:
        return 'auto'

    normalized = chunk_shape.strip().lower()
    if normalized == 'auto':
        return 'auto'
    if normalized in {'none', ''}:
        return None

    try:
        parsed = tuple(int(v.strip()) for v in chunk_shape.split(','))
    except ValueError as exc:
        raise ValueError(
            f'Invalid --chunk-shape value "{chunk_shape}". '
            'Use "auto", "none", or comma-separated integers like "2048,2048".'
        ) from exc

    if len(parsed) == 0 or any(v <= 0 for v in parsed):
        raise ValueError(
            f'Invalid --chunk-shape value "{chunk_shape}". '
            'Chunk dimensions must be positive integers.'
        )

    return parsed


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for focus command.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Focus SAR data using Range-Doppler Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Focus a zarr file
  sarpyx focus --input /path/to/data.zarr --output /path/to/output
  
  # Focus with custom processing chunk height
  sarpyx focus --input /path/to/data.zarr --output /path/to/output --slice-height 20000

  # Force single-pass processing with no slicing
  sarpyx focus --input /path/to/data.zarr --output /path/to/output --no-slicing

  # Save output with custom Zarr chunk shape
  sarpyx focus --input /path/to/data.zarr --output /path/to/output --chunk-shape 2048,2048

  # Disable output compression
  sarpyx focus --input /path/to/data.zarr --output /path/to/output --no-compression
  
  # Keep temporary files for debugging
  sarpyx focus --input /path/to/data.zarr --output /path/to/output --keep-tmp
"""
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input zarr file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./focused_data',
        help='Output directory (default: ./focused_data)'
    )
    
    parser.add_argument(
        '--slice-height',
        type=int,
        default=__BUFFER_SLICE_HEIGHT__,
        help=f'Slice height for processing (default: {__BUFFER_SLICE_HEIGHT__})'
    )

    parser.add_argument(
        '--no-slicing',
        action='store_true',
        help='Disable slicing and process the full input as a single slice'
    )

    parser.add_argument(
        '--chunk-shape',
        type=str,
        default='auto',
        help='Output Zarr chunk shape: "auto", "none", or comma-separated ints (e.g. 2048,2048)'
    )

    parser.add_argument(
        '--compression-level',
        type=int,
        default=5,
        help='Output compression level (0-9, where 0 means no compression)'
    )

    parser.add_argument(
        '--no-compression',
        action='store_true',
        help='Disable output compression (equivalent to --compression-level 0)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--keep-tmp',
        action='store_true',
        help='Keep temporary files after processing'
    )
    
    return parser


def setup_directories(output_dir: Path) -> tuple[Path, Path]:
    """
    Setup output and temporary directories.
    
    Args:
        output_dir: Base output directory path
        
    Returns:
        Tuple of (output_dir, tmp_dir) paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, tmp_dir


def focalize_slice(raw_data: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Process a single slice of raw SAR data.
    
    Args:
        raw_data: Raw data dictionary containing 'echo', 'metadata', and 'ephemeris'
        verbose: Enable verbose output
        
    Returns:
        Processed data dictionary containing 'raw', 'rc', 'rcmc', 'az', 'metadata', and 'ephemeris'
    """
    logger.info('Processing slice...')
    
    # Initialize processor
    processor = CoarseRDA(
        raw_data=raw_data,
        verbose=verbose,
    )
    logger.info(f'🛠️ Processor initialized with raw data of shape: {raw_data["echo"].shape}')
    
    # Focus the data
    processor.data_focus()
    
    # Extract processed data
    raw = processor.raw_data
    rc = processor.range_compressed_data
    rcmc = processor.rcmc_data
    az = processor.azimuth_compressed_data
    
    # Extract metadata and ephemeris from raw_data
    metadata = raw_data['metadata']
    ephemeris = raw_data['ephemeris']
    
    # Prepare result dictionary
    result = {
        'raw': raw, 
        'rc': rc, 
        'rcmc': rcmc, 
        'az': az, 
        'metadata': metadata, 
        'ephemeris': ephemeris
    }
    return result


def process_sar_slice(
    handler: ZarrManager,
    slice_idx: int, 
    slice_info: Dict[str, Any],
    tmp_dir: Path,
    verbose: bool = True,
    unique_slice: bool = True,
    output_chunks: Optional[Union[str, Tuple[int, ...]]] = 'auto',
    compression_level: int = 5,
) -> Path:
    """
    Process a single SAR data slice.
    
    Args:
        handler: Zarr data handler
        slice_idx: Index of the slice to process
        slice_info: Slice information dictionary
        tmp_dir: Temporary directory for saving results
        verbose: Enable verbose output
        unique_slice: Whether this is the only slice to process
        output_chunks: Chunking strategy passed to Zarr saver
        compression_level: Compression level passed to Zarr saver
        
    Returns:
        Path to saved slice file
    """
    # Extract slice information
    start_row = int(slice_info['original_start'])
    end_row = int(slice_info['original_end'])
    drop_start = int(slice_info['drop_start'])
    drop_end = int(slice_info['drop_end'])
    
    # Get slice data
    echo_data, metadata, ephemeris = handler.get_slice(rows=slice(start_row, end_row), cols=None)
    filename = handler.filename 
    raw_data = {'echo': echo_data, 'metadata': metadata, 'ephemeris': ephemeris}
    logger.info(f'📊 Sliced raw data shape: {raw_data["echo"].shape}')
   
    # Focus slice data
    result = focalize_slice(raw_data=raw_data, verbose=verbose)
    logger.info('✅ Slice focused successfully.')
    
    # Drop overlapping data
    result['raw'] = result['raw'][drop_start:-drop_end] if drop_end > 0 else result['raw'][drop_start:]
    result['rc'] = result['rc'][drop_start:-drop_end] if drop_end > 0 else result['rc'][drop_start:]
    result['rcmc'] = result['rcmc'][drop_start:-drop_end] if drop_end > 0 else result['rcmc'][drop_start:]
    result['az'] = result['az'][drop_start:-drop_end] if drop_end > 0 else result['az'][drop_start:]
    
    # Handle metadata and ephemeris slicing
    if hasattr(result['metadata'], 'iloc'):
        if drop_end > 0:
            result['metadata'] = result['metadata'].iloc[drop_start:-drop_end]
            result['ephemeris'] = result['ephemeris'].iloc[drop_start:-drop_end]
        else:
            result['metadata'] = result['metadata'].iloc[drop_start:]
            result['ephemeris'] = result['ephemeris'].iloc[drop_start:]
    
    logger.info(f'📉 Dropped overlapping data: start={drop_start}, end={drop_end}')
    logger.info(f'📊 Focused data shape: {result["raw"].shape}')

    if not unique_slice:
        # Save slice in tmp folder
        zarr_path = tmp_dir / f'processor_slice_{slice_idx}.zarr'
        logger.info(f'💾 Saving slice {slice_idx + 1} to: {zarr_path}')
        dask_slice_saver(result, zarr_path, chunks=output_chunks, clevel=compression_level)
        logger.info(f'📂 Slice {slice_idx + 1} saved successfully.')
    else:
        # Save the unique slice directly
        zarr_path = tmp_dir.parent / f'{filename}.zarr'
        logger.info(f'💾 Saving entire product to: {zarr_path}')
        dask_slice_saver(result, zarr_path, chunks=output_chunks, clevel=compression_level)
        logger.info('📂 Product saved successfully.')
    
    # Clean up memory
    del raw_data, result
    
    return zarr_path


def cleanup_tmp_directory(tmp_dir: Path) -> None:
    """
    Clean up temporary directory and all its contents recursively.
    
    Args:
        tmp_dir: Temporary directory to clean up
    """
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        logger.info(f'🗑️  Temporary directory {tmp_dir} cleaned up.')


def main() -> None:
    """
    Main entry point for focus CLI.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate input file
    input_path = validate_path(args.input, must_exist=True)
    
    if not input_path.suffix == '.zarr':
        logger.error(f'❌ Input file must be a .zarr file, got: {input_path.suffix}')
        sys.exit(1)
    
    # Setup directories
    output_dir = create_output_directory(args.output)
    output_dir, tmp_dir = setup_directories(output_dir)
    
    # Extract product name
    product_name = input_path.stem
    
    logger.info('🚀 Starting SAR data focusing...')
    logger.info(f'📁 Input file: {input_path}')
    logger.info(f'📁 Output directory: {output_dir}')
    logger.info(f'📁 Temporary directory: {tmp_dir}')
    
    try:
        if not args.no_slicing and args.slice_height <= 0:
            raise ValueError(f'--slice-height must be > 0, got {args.slice_height}')

        if not (0 <= args.compression_level <= 9):
            raise ValueError(f'--compression-level must be in [0, 9], got {args.compression_level}')

        output_chunks = _parse_chunk_shape(args.chunk_shape)
        compression_level = 0 if args.no_compression else args.compression_level

        logger.info(f'🧩 Output chunking: {output_chunks}')
        logger.info(f'🗜️ Output compression level: {compression_level}')

        # Initialize handler
        handler = ZarrManager(str(input_path))
        H = handler.load().shape[0]  # Get total height
        
        # Calculate slice indices
        slice_height = args.slice_height

        if args.no_slicing:
            logger.info('🧩 Slicing disabled. Processing full input as a single slice.')
            slice_indices = [
                {
                    'slice_index': 0,
                    'original_start': 0,
                    'original_end': H,
                    'actual_start': 0,
                    'actual_end': H,
                    'is_first': True,
                    'is_last': True,
                    'drop_start': 0,
                    'drop_end': 0,
                    'original_height': H,
                    'actual_height': H
                }
            ]
        elif (H // slice_height) > 1:
            slice_indices = calculate_slice_indices(
                array_height=H,
                slice_height=slice_height
            )
        else:
            slice_indices = [
                {
                    'slice_index': 0,
                    'original_start': 0,
                    'original_end': H,
                    'actual_start': 0,
                    'actual_end': H,
                    'is_first': True,
                    'is_last': True,
                    'drop_start': 0,
                    'drop_end': 0,
                    'original_height': H,
                    'actual_height': H
                }
            ]
        
        n_slices = len(slice_indices)
        
        # Display slice breakdown
        if args.verbose:
            logger.info(f'📊 Slice breakdown for {n_slices} slices:')
            logger.info('┌─────────┬─────────────┬───────────┬─────────────┬───────────┬─────────────┬─────────────┐')
            logger.info('│ Slice # │ Orig Start  │ Orig End  │ Actual Start│ Actual End│ Drop Start  │ Drop End    │')
            logger.info('├─────────┼─────────────┼───────────┼─────────────┼───────────┼─────────────┼─────────────┤')
            for i, slice_info in enumerate(slice_indices):
                logger.info(f'│ {i+1:^7} │ {slice_info["original_start"]:^11} │ {slice_info["original_end"]:^9} │ {slice_info["actual_start"]:^11} │ {slice_info["actual_end"]:^9} │ {slice_info["drop_start"]:^11} │ {slice_info["drop_end"]:^11} │')
            logger.info('└─────────┴─────────────┴───────────┴─────────────┴───────────┴─────────────┴─────────────┘')
        
        # Process each slice
        tmp_files = []
        for slice_idx in range(n_slices):
            logger.info(f'🔍 Processing slice {slice_idx + 1}/{n_slices}...')
            zarr_path = process_sar_slice(
                handler=handler,
                slice_idx=slice_idx,
                slice_info=slice_indices[slice_idx],
                tmp_dir=tmp_dir,
                verbose=args.verbose,
                unique_slice=(n_slices == 1),
                output_chunks=output_chunks,
                compression_level=compression_level,
            )
            tmp_files.append(zarr_path)
            logger.info(f'✅ Slice {slice_idx + 1} processed successfully.')
        
        # Concatenate slices if more than one
        if n_slices > 1:
            logger.info(f'🔗 Concatenating {len(tmp_files)} slices...')
            output_file = output_dir / f'{product_name}.zarr'
            concatenate_slices_efficient(
                tmp_files,
                output_file,
                chunks=output_chunks,
                clevel=compression_level
            )
            logger.info(f'✅ Concatenated data saved to: {output_file}')
        
        # Cleanup temporary files
        if not args.keep_tmp:
            cleanup_tmp_directory(tmp_dir)
        else:
            logger.info(f'📁 Temporary files kept in: {tmp_dir}')
        
        print('✅ Focusing completed successfully!')
        sys.exit(0)
        
    except Exception as e:
        logger.error(f'❌ Error during processing: {str(e)}')
        if not args.keep_tmp:
            cleanup_tmp_directory(tmp_dir)
        sys.exit(1)


if __name__ == '__main__':
    main()
