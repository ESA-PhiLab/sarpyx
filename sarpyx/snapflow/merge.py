"""Merge source IQ bands into polarimetric decomposition DIMAP products."""

from __future__ import annotations

from pathlib import Path

from sarpyx.snapflow.dimap import (
    clone_source_raster_metadata,
    copy_materialized_src_files,
    copy_src_files,
    detect_suffixes_from_src_data,
    edit_pdec_dim,
    get_data_dir_from_dim,
    materialized_band_names,
    validate_same_dimensions,
)


def merge_iq_into_pdec(
    src_dim: str | Path,
    pdec_dim: str | Path,
    is_tops: bool = False,
    overwrite_copied_files: bool = False,
    backup: bool = True,
    preserve_source_rasters: bool = True,
) -> None:
    src_dim = Path(src_dim).resolve()
    pdec_dim = Path(pdec_dim).resolve()
    if src_dim == pdec_dim:
        raise ValueError(f"Source DIM and PDEC DIM resolve to the same DIM product: {src_dim}")
    if not src_dim.exists():
        raise FileNotFoundError(f"Source DIM file does not exist: {src_dim}")
    if not pdec_dim.exists():
        raise FileNotFoundError(f"PDEC DIM file does not exist: {pdec_dim}")
    src_data_dir = get_data_dir_from_dim(src_dim)
    pdec_data_dir = get_data_dir_from_dim(pdec_dim)
    if not src_data_dir.exists():
        raise FileNotFoundError(f"Source data directory does not exist: {src_data_dir}")
    if not pdec_data_dir.exists():
        raise FileNotFoundError(f"PDEC data directory does not exist: {pdec_data_dir}")
    suffixes = detect_suffixes_from_src_data(src_data_dir)
    if not suffixes:
        raise RuntimeError(f"No valid suffixes were detected in {src_data_dir}.")
    validate_same_dimensions(src_dim, pdec_dim)
    if preserve_source_rasters:
        band_names = materialized_band_names(src_dim)
        copy_materialized_src_files(src_data_dir, pdec_data_dir, band_names, overwrite=overwrite_copied_files)
        clone_source_raster_metadata(src_dim=src_dim, pdec_dim=pdec_dim, band_names=band_names, backup=backup)
        backup = False
    copy_src_files(src_data_dir, pdec_data_dir, suffixes, overwrite=overwrite_copied_files)
    edit_pdec_dim(pdec_dim=pdec_dim, suffixes=suffixes, is_tops=is_tops, backup=backup)
