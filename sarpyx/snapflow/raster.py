"""Raster and BEAM-DIMAP geometry helpers for WorldSAR tiling."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

import h5py
import pyproj

from sarpyx.snapflow.dimap import get_data_dir_from_dim
from sarpyx.snapflow.tile_writers import PreWriteHook, TilePayload, tile_output_path, write_tile_payloads
from sarpyx.utils.worldsar_h5 import resolve_expected_band_names_from_dim_product


def _read_geotransform(dim_path: Path) -> tuple:
    root = ET.parse(dim_path).getroot()
    elem = root.find(".//IMAGE_TO_MODEL_TRANSFORM")
    if elem is not None and elem.text is not None:
        m00, m10, m01, m11, m02, m12 = [float(x.strip()) for x in elem.text.split(",")]
        return (m02, m00, m01, m12, m10, m11)
    ulx = root.find(".//ULXMAP")
    uly = root.find(".//ULYMAP")
    xdim = root.find(".//XDIM")
    ydim = root.find(".//YDIM")
    if ulx is not None and uly is not None and xdim is not None and ydim is not None:
        return (float(ulx.text), float(xdim.text), 0.0, float(uly.text), 0.0, -float(ydim.text))  # type: ignore[arg-type]
    raise RuntimeError(f"Could not extract geotransform from {dim_path}")


def _read_raster_size(dim_path: Path) -> tuple[int, int]:
    root = ET.parse(dim_path).getroot()
    raster_dimensions = root.find(".//Raster_Dimensions")
    if raster_dimensions is None:
        raise RuntimeError(f"Could not extract raster dimensions from {dim_path}")
    ncols = raster_dimensions.findtext("NCOLS")
    nrows = raster_dimensions.findtext("NROWS")
    if ncols is None or nrows is None:
        raise RuntimeError(f"Raster dimensions are incomplete in {dim_path}")
    return int(ncols), int(nrows)


def _read_crs_wkt(dim_path: Path) -> str:
    crs_wkt = ET.parse(dim_path).getroot().findtext(".//Coordinate_Reference_System/WKT")
    if crs_wkt is None or not crs_wkt.strip():
        raise RuntimeError(f"Could not extract CRS WKT from {dim_path}")
    return crs_wkt


def _dim_footprint_wkt(dim_path: Path) -> str:
    geotransform = _read_geotransform(dim_path)
    ncols, nrows = _read_raster_size(dim_path)
    transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_wkt(_read_crs_wkt(dim_path)), 4326, always_xy=True)
    origin_x, px_w, rot_x, origin_y, rot_y, px_h = geotransform
    corners = [
        (origin_x, origin_y),
        (origin_x + ncols * px_w, origin_y + ncols * rot_y),
        (origin_x + ncols * px_w + nrows * rot_x, origin_y + ncols * rot_y + nrows * px_h),
        (origin_x + nrows * rot_x, origin_y + nrows * px_h),
    ]
    lonlat_corners = [transformer.transform(x, y) for x, y in corners]
    lonlat_corners.append(lonlat_corners[0])
    return "POLYGON ((" + ", ".join(f"{lon} {lat}" for lon, lat in lonlat_corners) + "))"


def _feature_with_coords(template: dict, lon: float, lat: float) -> dict:
    import copy

    feature = copy.deepcopy(template)
    feature["geometry"]["coordinates"] = [lon, lat]
    return feature


def _rectangle_from_bl_anchor(bl_feature: dict) -> dict:
    from sarpyx.utils.geos import grid_cell_utm_bbox

    epsg = int(bl_feature["properties"]["epsg"].split(":")[1])
    x_min, y_min, x_max, y_max = grid_cell_utm_bbox({"BL": bl_feature}, epsg)
    transformer = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    bl_lon, bl_lat = transformer.transform(x_min, y_min)
    tl_lon, tl_lat = transformer.transform(x_min, y_max)
    tr_lon, tr_lat = transformer.transform(x_max, y_max)
    br_lon, br_lat = transformer.transform(x_max, y_min)
    return {
        "TL": _feature_with_coords(bl_feature, tl_lon, tl_lat),
        "TR": _feature_with_coords(bl_feature, tr_lon, tr_lat),
        "BR": _feature_with_coords(bl_feature, br_lon, br_lat),
        "BL": _feature_with_coords(bl_feature, bl_lon, bl_lat),
    }


def _fallback_rectangles_from_contained_points(contained: list[dict]) -> list[dict]:
    return [_rectangle_from_bl_anchor(feature) for feature in contained]


def _utm_bbox_to_pixel_region(utm_bbox: tuple, geotransform: tuple) -> str:
    x_min, y_min, x_max, y_max = utm_bbox
    orig_x, px_w, _, orig_y, _, px_h = geotransform
    col_start = int(round((x_min - orig_x) / px_w))
    row_start = int(round((y_max - orig_y) / px_h))
    width = int(round((x_max - x_min) / px_w))
    height = int(round((y_max - y_min) / abs(px_h)))
    return f"{col_start},{row_start},{width},{height}"


def _pixel_region_is_within_bounds(region: str, raster_size: tuple[int, int]) -> bool:
    col_start, row_start, width, height = (int(value) for value in region.split(","))
    ncols, nrows = raster_size
    return width > 0 and height > 0 and col_start >= 0 and row_start >= 0 and col_start + width <= ncols and row_start + height <= nrows


def _read_abstract_metadata_attrs(dim_path: Path) -> dict[str, str]:
    root = ET.parse(dim_path).getroot()
    abstract_metadata = root.find(".//MDElem[@name='Abstracted_Metadata']")
    if abstract_metadata is None:
        return {}
    return {
        attr.get("name"): (attr.text or "").strip()
        for attr in abstract_metadata.findall("MDATTR")
        if attr.get("name") and (attr.text or "").strip()
    }


def _read_band_metadata(dim_path: Path, band_name: str) -> dict[str, str]:
    root = ET.parse(dim_path).getroot()
    for spectral_band in root.findall("./Image_Interpretation/Spectral_Band_Info"):
        if (spectral_band.findtext("BAND_NAME") or "").strip() == band_name:
            return {
                "unit": (spectral_band.findtext("PHYSICAL_UNIT") or "unknown").strip() or "unknown",
                "scaling_factor": (spectral_band.findtext("SCALING_FACTOR") or "1.0").strip() or "1.0",
                "scaling_offset": (spectral_band.findtext("SCALING_OFFSET") or "0.0").strip() or "0.0",
                "log10_scaled": (spectral_band.findtext("LOG10_SCALED") or "false").strip() or "false",
            }
    return {"unit": "unknown", "scaling_factor": "1.0", "scaling_offset": "0.0", "log10_scaled": "false"}


def _write_h5_subset_from_dim(dim_path: Path, region: str, output_path: Path) -> Path:
    return _write_tile_subsets_from_dim(dim_path, [(region, output_path)], tile_writer="h5")[0]


def _write_h5_subsets_from_dim(dim_path: Path, subsets: list[tuple[str, Path]], max_open_outputs: int = 32) -> list[Path]:
    return _write_tile_subsets_from_dim(dim_path, subsets, tile_writer="h5", max_open_outputs=max_open_outputs)


def _write_tile_subsets_from_dim(dim_path: Path, subsets: list[tuple], tile_writer: str = "h5", pre_write_hook: PreWriteHook | None = None, max_open_outputs: int = 32) -> list[Path]:
    if max_open_outputs <= 0:
        raise ValueError(f"max_open_outputs must be > 0, got {max_open_outputs}")
    band_names = resolve_expected_band_names_from_dim_product(dim_path)
    if not band_names:
        raise RuntimeError(f"No materialized bands found in {dim_path}")
    normalized_subsets = [_normalize_tile_subset(subset) for subset in subsets]
    output_paths = [output_path for _, output_path, _ in normalized_subsets]
    abstract_attrs = _read_abstract_metadata_attrs(dim_path)
    band_metadata = {band_name: _read_band_metadata(dim_path, band_name) for band_name in band_names}
    data_dir = get_data_dir_from_dim(dim_path)
    geotransform = _read_geotransform(dim_path)
    try:
        crs_wkt = _read_crs_wkt(dim_path)
    except Exception:
        crs_wkt = None
    try:
        for output_path in output_paths:
            _remove_tile_output(output_path)
        for offset in range(0, len(normalized_subsets), max_open_outputs):
            chunk = normalized_subsets[offset : offset + max_open_outputs]
            payloads = _read_tile_payload_chunk(data_dir, band_names, band_metadata, abstract_attrs, geotransform, crs_wkt, chunk)
            write_tile_payloads(payloads, tile_writer, pre_write_hook)
    except Exception:
        for output_path in output_paths:
            _remove_tile_output(output_path)
        raise
    return output_paths


def _normalize_tile_subset(subset: tuple) -> tuple[str, Path, str]:
    region, output_path, *rest = subset
    output_path = Path(output_path)
    return region, output_path, str(rest[0]) if rest else output_path.stem


def _remove_tile_output(output_path: Path) -> None:
    if output_path.is_dir():
        shutil.rmtree(output_path)
    else:
        output_path.unlink(missing_ok=True)
    if output_path.suffix == ".npy":
        output_path.with_suffix(output_path.suffix + ".json").unlink(missing_ok=True)


def _tile_transform(geotransform: tuple, region: str) -> tuple[float, float, float, float, float, float]:
    col_start, row_start, _, _ = (int(value) for value in region.split(","))
    origin_x, px_w, rot_x, origin_y, rot_y, px_h = geotransform
    return (px_w, rot_x, origin_x + col_start * px_w + row_start * rot_x, rot_y, px_h, origin_y + col_start * rot_y + row_start * px_h)


def _band_attrs(metadata: dict[str, str], data) -> dict[str, str | int]:
    return {
        "CLASS": "org.esa.snap.core.datamodel.Band",
        "IMAGE_VERSION": "1.0",
        "log10_scaled": metadata["log10_scaled"],
        "raster_height": int(data.shape[0]),
        "raster_width": int(data.shape[1]),
        "scaling_factor": metadata["scaling_factor"],
        "scaling_offset": metadata["scaling_offset"],
        "unit": metadata["unit"],
    }


def _read_tile_payload_chunk(data_dir: Path, band_names: list[str], band_metadata: dict[str, dict[str, str]], abstract_attrs: dict[str, str], geotransform: tuple, crs_wkt: str | None, subsets: list[tuple[str, Path, str]]) -> list[TilePayload]:
    import rasterio
    from rasterio.windows import Window

    payloads = []
    windows = []
    for region, output_path, tile_name in subsets:
        col_start, row_start, width, height = (int(value) for value in region.split(","))
        windows.append(Window(col_start, row_start, width, height))
        payloads.append(TilePayload(tile_name, output_path, {}, dict(abstract_attrs), {}, region=region, crs_wkt=crs_wkt, transform=_tile_transform(geotransform, region)))
    for band_name in band_names:
        hdr_path = data_dir / f"{band_name}.hdr"
        raster_path = data_dir / f"{band_name}.img"
        if not hdr_path.exists() or not raster_path.exists():
            raise FileNotFoundError(f"Band files not found for tile export: {band_name}")
        metadata = band_metadata[band_name]
        with rasterio.open(raster_path) as src:
            for payload, window in zip(payloads, windows, strict=True):
                data = src.read(1, window=window)
                payload.arrays[band_name] = data
                payload.band_attrs[band_name] = _band_attrs(metadata, data)
    return payloads


def _write_h5_subsets_from_dim_rectangles(dim_path: Path, rectangles: list[dict], cuts_dir: Path) -> set[Path]:
    return _write_tile_subsets_from_dim_rectangles(dim_path, rectangles, cuts_dir, tile_writer="h5")


def _write_tile_subsets_from_dim_rectangles(dim_path: Path, rectangles: list[dict], cuts_dir: Path, tile_writer: str = "h5", pre_write_hook: PreWriteHook | None = None) -> set[Path]:
    from sarpyx.utils.geos import grid_cell_utm_bbox

    geotransform = _read_geotransform(dim_path)
    raster_size = _read_raster_size(dim_path)
    subsets = []
    for rect in rectangles:
        tile_name = rect["BL"]["properties"]["name"]
        epsg = int(rect["BL"]["properties"]["epsg"].split(":")[1])
        region = _utm_bbox_to_pixel_region(grid_cell_utm_bbox(rect, epsg), geotransform)
        if _pixel_region_is_within_bounds(region, raster_size):
            subsets.append((region, tile_output_path(cuts_dir, tile_name, tile_writer), tile_name))
    return set(_write_tile_subsets_from_dim(dim_path, subsets, tile_writer=tile_writer, pre_write_hook=pre_write_hook)) if subsets else set()


def _update_h5_corners(h5_path: Path, utm_bbox: tuple, epsg: int) -> None:
    x_min, y_min, x_max, y_max = utm_bbox
    transformer = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    bl_lon, bl_lat = transformer.transform(x_min, y_min)
    tl_lon, tl_lat = transformer.transform(x_min, y_max)
    tr_lon, tr_lat = transformer.transform(x_max, y_max)
    br_lon, br_lat = transformer.transform(x_max, y_min)
    cx_lon, cx_lat = transformer.transform((x_min + x_max) / 2, (y_min + y_max) / 2)
    with h5py.File(h5_path, "r+") as h5_file:
        attrs = h5_file["metadata/Abstracted_Metadata"].attrs
        attrs["last_near_long"], attrs["last_near_lat"] = bl_lon, bl_lat
        attrs["first_near_long"], attrs["first_near_lat"] = tl_lon, tl_lat
        attrs["first_far_long"], attrs["first_far_lat"] = tr_lon, tr_lat
        attrs["last_far_long"], attrs["last_far_lat"] = br_lon, br_lat
        attrs["centre_lon"], attrs["centre_lat"] = cx_lon, cx_lat
