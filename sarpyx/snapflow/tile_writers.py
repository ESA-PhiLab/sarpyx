"""Tile artifact writers and pre-write hook support."""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np


PreWriteHook = Callable[["TilePayload"], "TilePayload | None"]


@dataclass
class TilePayload:
    tile_name: str
    output_path: Path
    arrays: dict[str, np.ndarray]
    abstract_attrs: dict[str, Any]
    band_attrs: dict[str, dict[str, Any]]
    region: str | None = None
    crs_wkt: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    root_attrs: dict[str, Any] | None = None
    writer_options: dict[str, Any] = field(default_factory=dict)


WRITER_SUFFIXES = {
    "h5": ".h5",
    "zarr": ".zarr",
    "npz": ".npz",
    "npy": ".npy",
    "geotiff": ".tif",
    "pickle": ".pkl",
}

WRITER_ALIASES = {
    "hdf5": "h5",
    "h5": "h5",
    "zarr": "zarr",
    "numpy": "npz",
    "np": "npz",
    "npz": "npz",
    "npy": "npy",
    "tif": "geotiff",
    "tiff": "geotiff",
    "geotiff": "geotiff",
    "gtiff": "geotiff",
    "pkl": "pickle",
    "pickle": "pickle",
}


def normalize_tile_writer(writer: str | None) -> str:
    normalized = WRITER_ALIASES.get((writer or "h5").strip().lower())
    if normalized is None:
        choices = ", ".join(sorted(WRITER_ALIASES))
        raise ValueError(f"Unsupported tile writer {writer!r}. Supported writers: {choices}")
    return normalized


def tile_output_path(cuts_dir: Path, tile_name: str, writer: str | None = None) -> Path:
    normalized = normalize_tile_writer(writer)
    return cuts_dir / f"{tile_name}{WRITER_SUFFIXES[normalized]}"


def tile_glob_pattern(writer: str | None = None) -> str:
    return f"*{WRITER_SUFFIXES[normalize_tile_writer(writer)]}"


def write_tile_payloads(payloads: list[TilePayload], writer: str | None = None, pre_write_hook: PreWriteHook | None = None) -> list[Path]:
    normalized = normalize_tile_writer(writer)
    paths = []
    for payload in payloads:
        prepared = pre_write_hook(payload) if pre_write_hook else None
        payload = prepared or payload
        payload.output_path.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing(payload.output_path)
        _WRITERS[normalized](payload)
        paths.append(payload.output_path)
    return paths


def _remove_existing(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    if path.suffix == ".npy":
        path.with_suffix(path.suffix + ".json").unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _stack_arrays(payload: TilePayload) -> tuple[list[str], np.ndarray]:
    band_names = list(payload.arrays)
    if not band_names:
        raise ValueError(f"No arrays available for tile {payload.tile_name}")
    return band_names, np.stack([payload.arrays[band] for band in band_names], axis=0)


def _sidecar_metadata(payload: TilePayload, band_names: list[str]) -> dict[str, Any]:
    return {
        "tile": payload.tile_name,
        "band_names": band_names,
        "region": payload.region,
        "crs_wkt": payload.crs_wkt,
        "transform": payload.transform,
        "metadata": payload.abstract_attrs,
        "band_attrs": payload.band_attrs,
    }


def _write_h5(payload: TilePayload) -> None:
    with h5py.File(payload.output_path, "w") as h5_file:
        bands_group = h5_file.create_group("bands")
        abstract_group = h5_file.create_group("metadata").create_group("Abstracted_Metadata")
        for key, value in payload.abstract_attrs.items():
            abstract_group.attrs[key] = value
        for band_name, data in payload.arrays.items():
            dataset = bands_group.create_dataset(band_name, data=data, compression="gzip", compression_opts=4, chunks=True)
            for key, value in payload.band_attrs.get(band_name, {}).items():
                dataset.attrs[key] = value


def _write_zarr(payload: TilePayload) -> None:
    import zarr

    writer_options = payload.writer_options or {}
    root = zarr.create_group(store=payload.output_path.as_posix(), zarr_format=3, overwrite=True)
    root.attrs.update(_json_safe(payload.root_attrs if payload.root_attrs is not None else _sidecar_metadata(payload, list(payload.arrays))))
    bands = root.create_group("bands")
    chunks = writer_options.get("zarr_chunks") or writer_options.get("chunks")
    for band_name, data in payload.arrays.items():
        create_kwargs = {"data": data, "overwrite": False}
        normalized_chunks = _normalize_zarr_chunks(data, chunks)
        if normalized_chunks is not None:
            create_kwargs["chunks"] = normalized_chunks
        array = bands.create_array(band_name, **create_kwargs)
        array.attrs.update(_json_safe(payload.band_attrs.get(band_name, {})))
    if not writer_options.get("minimal_metadata"):
        metadata = root.create_group("metadata").create_group("Abstracted_Metadata")
        metadata.attrs.update(_json_safe(payload.abstract_attrs))


def _normalize_zarr_chunks(data: np.ndarray, chunks: Any) -> tuple[int, ...] | None:
    if chunks is None:
        return None
    chunk_values = tuple(int(value) for value in chunks)
    if len(chunk_values) != 2 or any(value <= 0 for value in chunk_values):
        raise ValueError(f"Zarr chunks must be two positive integers, got {chunks!r}")
    if data.ndim == 0:
        return None
    if data.ndim == 1:
        return (min(chunk_values[-1], int(data.shape[0])),)
    prefix = tuple(1 for _ in data.shape[:-2])
    return prefix + (
        min(chunk_values[0], int(data.shape[-2])),
        min(chunk_values[1], int(data.shape[-1])),
    )


def _write_npz(payload: TilePayload) -> None:
    band_names = list(payload.arrays)
    arrays = {band_name: payload.arrays[band_name] for band_name in band_names}
    arrays["__metadata__"] = np.array(json.dumps(_json_safe(_sidecar_metadata(payload, band_names))))
    np.savez_compressed(payload.output_path, **arrays)


def _write_npy(payload: TilePayload) -> None:
    band_names, data = _stack_arrays(payload)
    np.save(payload.output_path, data)
    payload.output_path.with_suffix(payload.output_path.suffix + ".json").write_text(
        json.dumps(_json_safe(_sidecar_metadata(payload, band_names)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_geotiff(payload: TilePayload) -> None:
    import rasterio
    from rasterio.transform import Affine

    band_names, data = _stack_arrays(payload)
    kwargs: dict[str, Any] = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": data.dtype,
    }
    if payload.transform:
        kwargs["transform"] = Affine(*payload.transform)
    if payload.crs_wkt:
        kwargs["crs"] = payload.crs_wkt
    with rasterio.open(payload.output_path, "w", **kwargs) as dst:
        for index, band_name in enumerate(band_names, start=1):
            dst.write(data[index - 1], index)
            dst.set_band_description(index, band_name)
        dst.update_tags(**{key: str(value) for key, value in _json_safe(payload.abstract_attrs).items()})


def _write_pickle(payload: TilePayload) -> None:
    with payload.output_path.open("wb") as file_obj:
        pickle.dump(_sidecar_metadata(payload, list(payload.arrays)) | {"arrays": payload.arrays}, file_obj, protocol=pickle.HIGHEST_PROTOCOL)


_WRITERS = {
    "h5": _write_h5,
    "zarr": _write_zarr,
    "npz": _write_npz,
    "npy": _write_npy,
    "geotiff": _write_geotiff,
    "pickle": _write_pickle,
}
