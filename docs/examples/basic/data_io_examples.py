#!/usr/bin/env python3
"""Basic data I/O examples using currently implemented sarpyx readers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sarpyx.processor.data import read_tif, read_zarr_file


def summarize_array(name: str, array) -> None:
    data = np.asarray(array)
    print(f"{name}: shape={data.shape}, dtype={data.dtype}")
    if data.size:
        print(f"{name}: min={np.nanmin(data):.6g}, max={np.nanmax(data):.6g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif", type=Path, help="GeoTIFF path to read")
    parser.add_argument("--zarr", type=Path, help="Zarr store path to inspect")
    parser.add_argument("--zarr-key", help="Optional array or group key inside the Zarr store")
    parser.add_argument("--out-npy", type=Path, help="Optional .npy output for the GeoTIFF array")
    args = parser.parse_args()

    if args.tif:
        tif_array = read_tif(args.tif, verbose=True)
        summarize_array("GeoTIFF band 1", tif_array)
        if args.out_npy:
            args.out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.out_npy, tif_array)
            print(f"Saved NumPy array: {args.out_npy}")

    if args.zarr:
        zarr_obj = read_zarr_file(args.zarr, args.zarr_key)
        print(f"Zarr object: {zarr_obj}")
        if hasattr(zarr_obj, "shape"):
            summarize_array("Zarr array", zarr_obj[:])

    if not args.tif and not args.zarr:
        parser.error("Provide --tif and/or --zarr")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
