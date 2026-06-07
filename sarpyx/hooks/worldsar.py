"""WorldSAR tiling hooks for compact Zarr training tiles."""
from __future__ import annotations
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sarpyx.hooks.subap_features import SubapFeatureConfig, add_subap_features
from sarpyx.snapflow.h5_quality import summarize_zarr_raster_quality
from sarpyx.snapflow.tile_writers import TilePayload, tile_glob_pattern
DEFAULT_WORLDSAR_ZARR_CHUNKS = (128, 128)
POLARIZATION_RE = re.compile(r"(?<![A-Z0-9])(HH|HV|VH|VV)(?![A-Z0-9])", re.IGNORECASE)
_IW_PREFIX_RE = re.compile(r"(?i)(^|_)IW\d+_")
def product_output_name(product_path: str | Path) -> str:
    return Path(product_path).name.rstrip("/")
@dataclass(frozen=True)
class WorldSARZarrTileHook:
    product_path: Path
    product_mode: str | None = None
    product_name: str | None = None
    chunk_size: tuple[int, int] = DEFAULT_WORLDSAR_ZARR_CHUNKS
    subap_features: SubapFeatureConfig = SubapFeatureConfig()

    def __call__(self, payload: TilePayload) -> TilePayload:
        arrays, feature_band_attrs, feature_names = add_subap_features(payload.arrays, payload.band_attrs, self.subap_features)
        if payload.output_path.suffix.lower() == ".zarr":
            arrays, feature_band_attrs = _strip_iw_prefixes(arrays, feature_band_attrs)
            feature_names = [_strip_iw_prefix(name) for name in feature_names]
        attrs = _minimal_training_attrs(
            payload,
            product_name=self.product_name or product_output_name(self.product_path),
            product_mode=self.product_mode,
            chunk_size=self.chunk_size,
            arrays=arrays,
            subap_feature_names=feature_names,
        )
        band_attrs = {
            band_name: _minimal_band_attrs(band_name, feature_band_attrs.get(band_name, {}), attrs["polarizations"])
            for band_name in arrays
        }
        options = dict(payload.writer_options)
        options.update({"zarr_chunks": self.chunk_size, "minimal_metadata": True})
        return replace(
            payload,
            arrays=arrays,
            abstract_attrs={},
            band_attrs=band_attrs,
            root_attrs=attrs,
            writer_options=options,
        )
def make_worldsar_zarr_tile_hook(
    product_path: str | Path,
    *,
    product_mode: str | None = None,
    product_name: str | None = None,
    chunk_size: tuple[int, int] = DEFAULT_WORLDSAR_ZARR_CHUNKS,
    subap_features: SubapFeatureConfig | None = None,
) -> WorldSARZarrTileHook:
    return WorldSARZarrTileHook(
        product_path=Path(product_path),
        product_mode=product_mode,
        product_name=product_name,
        chunk_size=tuple(chunk_size),
        subap_features=subap_features or SubapFeatureConfig(enabled=bool(product_mode and product_mode.startswith("S1"))),
    )
def validate_worldsar_zarr_tile_group(cuts_dir, tiling_result: dict[str, Any], intermediate_product, swath: str | None = None) -> dict[str, Any]:
    cuts_dir = Path(cuts_dir)
    actual_tiles = sorted(tiling_result.get("actual_tiles") or [])
    if actual_tiles:
        tile_files = [cuts_dir / f"{tile}.zarr" for tile in actual_tiles]
        tile_files = [path for path in tile_files if path.exists()]
    else:
        tile_files = sorted(cuts_dir.glob(tile_glob_pattern("zarr")))
    expected_geometries = tiling_result.get("expected_tile_geometries", {}) or {}
    results = [
        validate_worldsar_zarr_tile(tile_file, swath=swath, tile_geometry=expected_geometries.get(tile_file.stem))
        for tile_file in tile_files
    ]
    return {
        "name": tiling_result["name"],
        "swath": swath,
        "cuts_dir": str(cuts_dir),
        "intermediate_product": str(intermediate_product),
        "expected_bands": _expected_bands_from_results(results),
        "expected_array_paths": ["bands"],
        "expected_metadata_paths": [],
        "expected_metadata_attr_paths": ["root@pass_direction", "root@polarizations"],
        "results": results,
        "rows": [result["quickinfo_row"] for result in results],
        "tile_writer": "zarr",
        "candidate_tiles": sorted(tiling_result.get("candidate_tiles", tiling_result.get("candidate_tile_geometries", {}).keys())),
        "expected_tiles": sorted(tiling_result.get("expected_tiles", expected_geometries.keys())),
        "actual_tiles": sorted(tiling_result.get("actual_tiles", [result["tile"] for result in results])),
        "missing_tiles": sorted(tiling_result.get("missing_tiles", [])),
        "extra_tiles": sorted(tiling_result.get("extra_tiles", [])),
        "skipped_tiles": sorted(tiling_result.get("skipped_tiles", [])),
        "partial_tiles": sorted(tiling_result.get("partial_tiles", [])),
        "failed_tiles": sorted(set(tiling_result.get("failed_tiles", [])) | {result["tile"] for result in results if result["status"] != "success"}),
        "skipped_tile_reasons": dict(tiling_result.get("skipped_tile_reasons", {})),
        "partial_tile_reasons": dict(tiling_result.get("partial_tile_reasons", {})),
        "candidate_tile_geometries": tiling_result.get("candidate_tile_geometries", {}),
        "expected_tile_geometries": expected_geometries,
        "cut_failed": tiling_result.get("cut_failed", False),
        "cut_report_path": str(tiling_result["report_path"]) if tiling_result.get("report_path") else None,
        "pre_tc_wkt": tiling_result.get("pre_tc_wkt"),
        "post_tc_wkt": tiling_result.get("post_tc_wkt") or tiling_result.get("source_wkt"),
        "source_wkt": tiling_result.get("source_wkt"),
        "report_source_wkt": tiling_result.get("report_source_wkt") or tiling_result.get("source_wkt"),
    }

def validate_worldsar_zarr_tile(tile_path: str | Path, swath: str | None = None, tile_geometry=None) -> dict[str, Any]:
    import zarr
    tile_path = Path(tile_path)
    issues: list[str] = []
    try:
        root = zarr.open(tile_path.as_posix(), mode="r")
        root_attrs = dict(root.attrs)
        bands_group = root.get("bands") if hasattr(root, "get") else None
        actual_bands = sorted(str(name) for name in bands_group.keys()) if bands_group is not None else []
        array_paths = [f"bands/{band_name}" for band_name in actual_bands]
        chunks_ok = True
        for band_name in actual_bands:
            array = bands_group[band_name]
            chunks_ok = chunks_ok and _array_chunks_ok(tuple(array.shape), tuple(array.chunks))
        bands_ok = bool(actual_bands)
        metadata_ok = _metadata_ok(root_attrs)
        band_attrs_ok = chunks_ok
        structure_ok = bands_group is not None
        raster_quality = summarize_zarr_raster_quality(tile_path)
    except Exception as exc:
        root_attrs = {}
        actual_bands = []
        array_paths = []
        bands_ok = metadata_ok = band_attrs_ok = structure_ok = False
        raster_quality = {}
        issues.append(f"{type(exc).__name__}: {exc}")

    raster_data_ok = raster_quality.get("raster_data_ok", False)
    status = "success" if bands_ok and metadata_ok and band_attrs_ok and structure_ok and raster_data_ok else "failed"
    quickinfo_row = _quickinfo_row(tile_path, root_attrs, swath)
    return {
        "tile": tile_path.stem,
        "swath": swath,
        "output_path": str(tile_path),
        "status": status,
        "bands_ok": bands_ok,
        "metadata_ok": metadata_ok,
        "band_attrs_ok": band_attrs_ok,
        "structure_ok": structure_ok,
        "missing_bands": [],
        "extra_bands": [],
        "actual_bands": actual_bands,
        "missing_metadata_section": False,
        "empty_metadata_fields": [],
        "missing_core_metadata_fields": [] if metadata_ok else ["pass_direction", "polarizations"],
        "empty_core_metadata_fields": [],
        "band_attr_issues": {} if band_attrs_ok else {"chunks": {"invalid_shape": False, "missing_attrs": ["128x128 chunks"], "empty_attrs": []}},
        "shape_summary": [],
        "array_paths": array_paths,
        "metadata_paths": [],
        "metadata_attr_paths": sorted(f"root@{key}" for key in root_attrs),
        "missing_array_paths": [],
        "missing_metadata_paths": [],
        "missing_metadata_attrs": [],
        "quickinfo_row": quickinfo_row,
        "tile_polygon_coords": tile_geometry,
        "tile_center_coords": None,
        "issues": issues,
        **raster_quality,
    }

def _minimal_training_attrs(
    payload: TilePayload,
    *,
    product_name: str,
    product_mode: str | None,
    chunk_size: tuple[int, int],
    arrays: dict[str, Any],
    subap_feature_names: list[str],
) -> dict[str, Any]:
    abstract = payload.abstract_attrs or {}
    band_names = list(arrays)
    polarizations = _polarizations(abstract, band_names)
    attrs = {
        "metadata_profile": "worldsar_training_minimal_v1",
        "tile": payload.tile_name,
        "product": product_name,
        "product_mode": product_mode,
        "pass_direction": _pass_direction(abstract),
        "polarizations": polarizations,
        "band_names": band_names,
        "subap_feature_bands": subap_feature_names,
        "chunk_size": [int(chunk_size[0]), int(chunk_size[1])],
        "mission": _first_nonblank(abstract, "MISSION", "mission"),
        "acquisition_time": _first_nonblank(abstract, "first_line_time", "FIRST_LINE_TIME", "start_time"),
        "look_direction": _first_nonblank(abstract, "antenna_pointing", "look_direction"),
        "epsg": _epsg(payload.crs_wkt),
        "transform": list(payload.transform) if payload.transform else None,
        "pixel_spacing": _pixel_spacing(payload.transform, abstract),
    }
    return {key: value for key, value in attrs.items() if not _blank(value)}

def _minimal_band_attrs(band_name: str, attrs: dict[str, Any], polarizations: list[str]) -> dict[str, Any]:
    band_pol = _polarization_from_text(band_name) or (polarizations[0] if len(polarizations) == 1 else None)
    return {
        key: value
        for key, value in {
            "polarization": band_pol,
            "unit": attrs.get("unit"),
        }.items()
        if not _blank(value)
    }


def _strip_iw_prefixes(
    arrays: dict[str, Any],
    band_attrs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    normalized_arrays: dict[str, Any] = {}
    normalized_band_attrs: dict[str, dict[str, Any]] = {}
    for name, data in arrays.items():
        normalized_name = _strip_iw_prefix(name)
        normalized_arrays[normalized_name] = data
        if name in band_attrs:
            normalized_band_attrs[normalized_name] = band_attrs[name]
    return normalized_arrays, normalized_band_attrs


def _strip_iw_prefix(name: str) -> str:
    return _IW_PREFIX_RE.sub(r"\1", name)

def _polarizations(attrs: dict[str, Any], band_names: list[str]) -> list[str]:
    values: list[str] = []
    for key in ("mds1_tx_rx_polar", "mds2_tx_rx_polar", "polarization", "polarisation", "polarisations", "polarizations"):
        value = attrs.get(key)
        if value is None:
            continue
        values.extend(_polarization_from_text(part) for part in re.split(r"[,;/\s]+", str(value)))
    values.extend(_polarization_from_text(band_name) for band_name in band_names)
    return sorted({value.upper() for value in values if value})

def _polarization_from_text(value: Any) -> str | None:
    match = POLARIZATION_RE.search(str(value or ""))
    return match.group(1).upper() if match else None

def _pass_direction(attrs: dict[str, Any]) -> str:
    value = _first_nonblank(attrs, "PASS", "pass", "pass_direction", "PASS_DIRECTION", "orbitPassDirection")
    normalized = str(value or "").strip().upper()
    if normalized.startswith("ASC"):
        return "ASC"
    if normalized.startswith("DESC"):
        return "DESC"
    return normalized or "UNKNOWN"

def _first_nonblank(attrs: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = attrs.get(key)
        if not _blank(value):
            return value
    return None

def _epsg(crs_wkt: str | None) -> int | None:
    if not crs_wkt:
        return None
    try:
        import pyproj

        return pyproj.CRS.from_wkt(crs_wkt).to_epsg()
    except Exception:
        return None

def _pixel_spacing(transform: tuple[float, float, float, float, float, float] | None, attrs: dict[str, Any]) -> list[float] | None:
    if transform:
        return [abs(float(transform[0])), abs(float(transform[4]))]
    range_spacing = _first_nonblank(attrs, "range_spacing")
    azimuth_spacing = _first_nonblank(attrs, "azimuth_spacing")
    if range_spacing is None or azimuth_spacing is None:
        return None
    try:
        return [float(range_spacing), float(azimuth_spacing)]
    except (TypeError, ValueError):
        return None

def _array_chunks_ok(shape: tuple[int, ...], chunks: tuple[int, ...]) -> bool:
    if len(shape) < 2 or len(chunks) < 2:
        return False
    return chunks[-2:] == (min(DEFAULT_WORLDSAR_ZARR_CHUNKS[0], shape[-2]), min(DEFAULT_WORLDSAR_ZARR_CHUNKS[1], shape[-1]))

def _metadata_ok(attrs: dict[str, Any]) -> bool:
    base_ok = attrs.get("pass_direction") in {"ASC", "DESC"} and bool(attrs.get("polarizations"))
    if str(attrs.get("product_mode") or "") in {"S1TOPS", "S1STRIP"}:
        return base_ok and bool(attrs.get("subap_feature_bands"))
    return base_ok

def _quickinfo_row(tile_path: Path, attrs: dict[str, Any], swath: str | None) -> dict[str, Any]:
    row = {
        "ID": tile_path.stem,
        "PRODUCT": attrs.get("product"),
        "PRODUCT_TYPE": attrs.get("product_mode"),
        "ACQUISITION_MODE": attrs.get("product_mode"),
        "MISSION": attrs.get("mission"),
        "PASS": attrs.get("pass_direction"),
        "mds1_tx_rx_polar": ",".join(attrs.get("polarizations") or []),
        "first_line_time": attrs.get("acquisition_time"),
        "epsg": attrs.get("epsg"),
        "tile_writer": "zarr",
    }
    if swath is not None:
        row["SWATH"] = swath
    return {key: value for key, value in row.items() if not _blank(value)}

def _expected_bands_from_results(results: list[dict[str, Any]]) -> list[str]:
    return sorted({band for result in results for band in result.get("actual_bands", [])})

def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False
