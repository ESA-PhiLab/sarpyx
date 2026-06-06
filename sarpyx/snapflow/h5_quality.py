"""H5 raster no-data quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def summarize_h5_raster_quality(tile_path: Path | str) -> dict[str, Any]:
    tile_path = Path(tile_path)
    with h5py.File(tile_path, "r") as h5_file:
        bands_group = h5_file.get("bands")
        if not isinstance(bands_group, h5py.Group):
            return _empty_summary("missing /bands group")
        arrays = [
            obj[...]
            for obj in bands_group.values()
            if isinstance(obj, h5py.Dataset)
            and len(obj.shape) == 2
            and np.issubdtype(obj.dtype, np.number)
        ]
    if not arrays:
        return _empty_summary("no 2D numeric band datasets")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        return _empty_summary("band shapes differ")

    any_nan = np.zeros(shape, dtype=bool)
    all_zero = np.ones(shape, dtype=bool)
    for array in arrays:
        any_nan |= np.isnan(array)
        all_zero &= array == 0

    zero_nodata = all_zero if len(arrays) > 1 else np.zeros(shape, dtype=bool)
    nodata = any_nan | zero_nodata
    total_pixels = int(nodata.size)
    nodata_pixels = int(nodata.sum())
    valid_pixels = total_pixels - nodata_pixels
    raster_data_ok = total_pixels > 0 and nodata_pixels == 0
    return {
        "band_count": len(arrays),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nodata_pixels": nodata_pixels,
        "zero_all_pixels": int(all_zero.sum()),
        "nan_pixels": int(any_nan.sum()),
        "valid_fraction": valid_pixels / total_pixels if total_pixels else 0.0,
        "nodata_fraction": nodata_pixels / total_pixels if total_pixels else 1.0,
        "raster_data_ok": raster_data_ok,
        "raster_quality_reason": None
        if raster_data_ok
        else f"incomplete raster coverage; nodata_fraction={nodata_pixels / total_pixels if total_pixels else 1.0:.6f}",
    }


def _empty_summary(reason: str) -> dict[str, Any]:
    return {
        "band_count": 0,
        "total_pixels": 0,
        "valid_pixels": 0,
        "nodata_pixels": 0,
        "zero_all_pixels": 0,
        "nan_pixels": 0,
        "valid_fraction": 0.0,
        "nodata_fraction": 1.0,
        "raster_data_ok": False,
        "raster_quality_reason": reason,
    }


def summarize_zarr_raster_quality(tile_path: Path | str) -> dict[str, Any]:
    import zarr

    tile_path = Path(tile_path)
    root = zarr.open(tile_path.as_posix(), mode="r")
    bands_group = root.get("bands") if hasattr(root, "get") else None
    if bands_group is None:
        return _empty_summary("missing /bands group")
    arrays = []
    for name in bands_group.keys():
        array = bands_group[name]
        if len(array.shape) == 2 and np.issubdtype(array.dtype, np.number):
            arrays.append(array[:])
    return _summarize_numeric_band_arrays(arrays)


def _summarize_numeric_band_arrays(arrays: list[np.ndarray]) -> dict[str, Any]:
    if not arrays:
        return _empty_summary("no 2D numeric band datasets")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        return _empty_summary("band shapes differ")

    any_nan = np.zeros(shape, dtype=bool)
    all_zero = np.ones(shape, dtype=bool)
    for array in arrays:
        any_nan |= np.isnan(array)
        all_zero &= array == 0

    zero_nodata = all_zero if len(arrays) > 1 else np.zeros(shape, dtype=bool)
    nodata = any_nan | zero_nodata
    total_pixels = int(nodata.size)
    nodata_pixels = int(nodata.sum())
    valid_pixels = total_pixels - nodata_pixels
    raster_data_ok = total_pixels > 0 and nodata_pixels == 0
    return {
        "band_count": len(arrays),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nodata_pixels": nodata_pixels,
        "zero_all_pixels": int(all_zero.sum()),
        "nan_pixels": int(any_nan.sum()),
        "valid_fraction": valid_pixels / total_pixels if total_pixels else 0.0,
        "nodata_fraction": nodata_pixels / total_pixels if total_pixels else 1.0,
        "raster_data_ok": raster_data_ok,
        "raster_quality_reason": None
        if raster_data_ok
        else f"incomplete raster coverage; nodata_fraction={nodata_pixels / total_pixels if total_pixels else 1.0:.6f}",
    }
