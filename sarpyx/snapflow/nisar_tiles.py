"""NISAR tile payload adapters for WorldSAR writers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import pyproj

from sarpyx.utils.meta import normalize_sar_timestamp
from sarpyx.utils.nisar_utils import NISARCutter, NISARReader
from sarpyx.snapflow.tile_writers import PreWriteHook, TilePayload, write_tile_payloads

CORE_METADATA_DEFAULTS = {
    "MISSION": "NISAR",
    "ACQUISITION_MODE": "GSLC",
    "PRODUCT_TYPE": "GSLC",
    "radar_frequency": -1.0,
    "pulse_repetition_frequency": -1.0,
    "range_spacing": -1.0,
    "azimuth_spacing": -1.0,
    "range_bandwidth": -1.0,
    "azimuth_bandwidth": -1.0,
    "antenna_pointing": "unknown",
    "PASS": "unknown",
    "avg_scene_height": 0.0,
    "PRODUCT": "unknown",
    "mds1_tx_rx_polar": "unknown",
    "mds2_tx_rx_polar": "unknown",
    "first_line_time": "1970-01-01T00:00:00.000000Z",
}


def select_nisar_tile_polarizations(product_path: Path | str) -> list[str]:
    available = NISARReader(product_path).get_available_polarizations()
    selected = [pol for pol in ("HH", "HV") if pol in available]
    return selected or available


def write_nisar_bbox_tile(product_path: Path, output_path: Path, tile_name: str, bbox: tuple, tile_writer: str, pre_write_hook: PreWriteHook | None = None) -> Path:
    product_path = Path(product_path)
    output_path = Path(output_path)
    reader = NISARReader(str(product_path))
    pols = select_nisar_tile_polarizations(product_path)
    if not pols:
        raise ValueError(f"No NISAR polarizations available in {product_path}")
    subset = NISARCutter(reader).cut_by_bbox(*bbox, pols, apply_mask=False)
    payload = _payload_from_subset(product_path, output_path, tile_name, bbox, subset, pols)
    return write_tile_payloads([payload], tile_writer, pre_write_hook)[0]


def _payload_from_subset(product_path: Path, output_path: Path, tile_name: str, bbox: tuple, subset: dict[str, Any], pols: list[str]) -> TilePayload:
    data = subset["data"]
    metadata = subset["metadata"]
    arrays = {pols[0]: data} if data.ndim == 2 else {pol: data[index] for index, pol in enumerate(pols)}
    abstract_attrs = _abstract_attrs_from_nisar(product_path, metadata, pols, bbox)
    band_attrs = {name: _band_attrs(array) for name, array in arrays.items()}
    transform = subset["transform"]
    return TilePayload(
        tile_name,
        output_path,
        arrays,
        abstract_attrs,
        band_attrs,
        crs_wkt=pyproj.CRS.from_epsg(metadata.epsg).to_wkt() if metadata.epsg else None,
        transform=(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
    )


def _band_attrs(array) -> dict[str, Any]:
    return {
        "CLASS": "org.esa.snap.core.datamodel.Band",
        "IMAGE_VERSION": "1.0",
        "log10_scaled": "false",
        "raster_height": int(array.shape[-2]),
        "raster_width": int(array.shape[-1]),
        "scaling_factor": "1.0",
        "scaling_offset": "0.0",
        "unit": "complex" if getattr(array.dtype, "kind", "") == "c" else "linear",
    }


def _abstract_attrs_from_nisar(product_path: Path, metadata, pols: list[str], bbox: tuple) -> dict[str, Any]:
    attrs = dict(CORE_METADATA_DEFAULTS)
    for key, value in asdict(metadata).items():
        if value is None:
            continue
        attrs[key] = ",".join(str(item) for item in value) if isinstance(value, (list, tuple)) else value
    attrs["azimuth_spacing"] = abs(metadata.y_spacing)
    attrs["range_spacing"] = metadata.x_spacing
    attrs["mds1_tx_rx_polar"] = pols[0]
    attrs["mds2_tx_rx_polar"] = pols[1] if len(pols) > 1 else pols[0]
    attrs.update(_corner_attrs(bbox, metadata.epsg))
    _update_attrs_from_source(product_path, metadata.frequency, attrs)
    return attrs


def _decode(value) -> str:
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


def _read_scalar(src: h5py.File, path: str):
    return src[path][()] if path in src else None


def _update_attrs_from_source(product_path: Path, frequency: str, attrs: dict[str, Any]) -> None:
    with h5py.File(product_path, "r") as src:
        ident = src.get("science/LSAR/identification")
        if ident is not None:
            mapping = {
                "missionId": "MISSION",
                "productType": "PRODUCT_TYPE",
                "granuleId": "PRODUCT",
                "orbitPassDirection": "PASS",
                "lookDirection": "antenna_pointing",
            }
            for source_name, target_name in mapping.items():
                if source_name in ident:
                    attrs[target_name] = _decode(ident[source_name][()])
            attrs["ACQUISITION_MODE"] = attrs["PRODUCT_TYPE"]
            if "zeroDopplerStartTime" in ident:
                attrs["first_line_time"] = normalize_sar_timestamp(_decode(ident["zeroDopplerStartTime"][()])) or CORE_METADATA_DEFAULTS["first_line_time"]
        freq_base = f"science/LSAR/GSLC/grids/{frequency}"
        for source_name, target_name in (("centerFrequency", "radar_frequency"), ("slantRangeSpacing", "range_spacing"), ("processedRangeBandwidth", "range_bandwidth"), ("processedAzimuthBandwidth", "azimuth_bandwidth")):
            value = _read_scalar(src, f"{freq_base}/{source_name}")
            if value is not None:
                attrs[target_name] = float(value)


def _corner_attrs(bbox: tuple, epsg: int | None) -> dict[str, float]:
    x_min, y_min, x_max, y_max = bbox
    if epsg is None:
        return {}
    transformer = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    bl_lon, bl_lat = transformer.transform(x_min, y_min)
    tl_lon, tl_lat = transformer.transform(x_min, y_max)
    tr_lon, tr_lat = transformer.transform(x_max, y_max)
    br_lon, br_lat = transformer.transform(x_max, y_min)
    cx_lon, cx_lat = transformer.transform((x_min + x_max) / 2, (y_min + y_max) / 2)
    return {
        "last_near_long": bl_lon,
        "last_near_lat": bl_lat,
        "first_near_long": tl_lon,
        "first_near_lat": tl_lat,
        "first_far_long": tr_lon,
        "first_far_lat": tr_lat,
        "last_far_long": br_lon,
        "last_far_lat": br_lat,
        "centre_lon": cx_lon,
        "centre_lat": cx_lat,
    }
