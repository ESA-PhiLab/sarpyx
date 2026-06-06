"""ENVI writers used by sub-aperture processing."""
from __future__ import annotations

import errno
import os

import numpy as np
import rasterio


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _free_bytes(path: str) -> int | None:
    try:
        stats = os.statvfs(os.path.dirname(os.path.abspath(path)) or ".")
        return stats.f_bavail * stats.f_frsize
    except OSError:
        return None


def _write_float32_stream(path_img: str, arr_out: np.ndarray, band_name: str, chunk_rows: int = 256) -> None:
    expected_bytes = int(arr_out.size * arr_out.dtype.itemsize)
    free_bytes = _free_bytes(path_img)
    if free_bytes is not None and free_bytes < expected_bytes:
        raise OSError(
            errno.ENOSPC,
            (
                f"Not enough free space to write ENVI band {band_name!r}: "
                f"need {_format_bytes(expected_bytes)}, available {_format_bytes(free_bytes)}, target={path_img}"
            ),
        )

    written = 0
    try:
        with open(path_img, "wb") as dst:
            for start in range(0, arr_out.shape[0], chunk_rows):
                block = np.ascontiguousarray(arr_out[start : start + chunk_rows])
                data = block.tobytes(order="C")
                dst.write(data)
                written += len(data)
    except OSError as exc:
        free_after = _free_bytes(path_img)
        try:
            os.unlink(path_img)
        except OSError:
            pass
        raise OSError(
            exc.errno or errno.EIO,
            (
                f"Failed writing ENVI band {band_name!r}: wrote {_format_bytes(written)} of "
                f"{_format_bytes(expected_bytes)}, available now {_format_bytes(free_after)}, target={path_img}. "
                f"Original error: {exc}"
            ),
        ) from exc

    if written != expected_bytes:
        raise OSError(
            errno.EIO,
            f"Incomplete ENVI band write for {band_name!r}: wrote {_format_bytes(written)} of {_format_bytes(expected_bytes)}, target={path_img}",
        )


def _subap_output_name(prefix: str, component: str, pol: str, sa: int) -> str:
    return f"{prefix}{component}_{pol}_SA{sa}.img"


def _estimate_subap_output_bytes(data_dir: str, base_pairs: dict, decomps: list[int], prefix: str = "") -> int:
    required = 0
    for nlooks in decomps:
        prefix_n = f"{prefix}L{nlooks}_" if len(decomps) > 1 else prefix
        for pol, (i_fp, _q_fp) in base_pairs.items():
            with rasterio.open(i_fp) as src:
                band_bytes = int(src.height * src.width * np.dtype(np.float32).itemsize)
            for sa in range(1, nlooks + 1):
                for component in ("i", "q"):
                    output_path = os.path.join(data_dir, _subap_output_name(prefix_n, component, pol, sa))
                    existing_bytes = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    required += max(0, band_bytes - existing_bytes)
    return required


def _ensure_subap_output_space(data_dir: str, required_bytes: int) -> None:
    free_bytes = _free_bytes(os.path.join(data_dir, ".sarpyx-subap-space-check"))
    if free_bytes is not None and free_bytes < required_bytes:
        raise OSError(
            errno.ENOSPC,
            (
                "Not enough free space for Sentinel sub-aperture ENVI outputs: "
                f"need {_format_bytes(required_bytes)}, available {_format_bytes(free_bytes)}, target_dir={data_dir}"
            ),
        )


def write_envi_bsq_float32(path_img, path_hdr, arr2d, band_name, byte_order=1, type_="real"):
    """Write a single-band ENVI BSQ float32 image and header."""
    arr = np.asarray(arr2d, dtype=np.float32)
    arr_out = arr.astype(">f4", copy=False) if byte_order == 1 else arr.astype("<f4", copy=False)
    _write_float32_stream(str(path_img), arr_out, band_name=band_name)

    lines, samples = arr.shape
    hdr = f"""ENVI
description = {{Sentinel-1 SM Level-1 SLC Product - Unit: {type_}}}
samples = {samples}
lines = {lines}
bands = 1
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = {byte_order}
band names = {{ {band_name} }}
data gain values = {{1.0}}
data offset values = {{0.0}}
"""
    with open(path_hdr, "w", encoding="ascii") as f:
        f.write(hdr)
