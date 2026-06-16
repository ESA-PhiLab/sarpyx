"""Mission-aware footprint WKT resolution for WorldSAR tiling."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from math import atan2
from pathlib import Path

import pyproj
from shapely import force_2d, wkt as shapely_wkt
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.polygon import orient

from sarpyx.snapflow.raster import _dim_footprint_wkt
from sarpyx.utils.wkt_utils import sentinel1_swath_wkt_extractor_safe


def resolve_tiling_wkt(product_wkt, source_product, intermediate_product, product_mode, swath=None) -> str | None:
    """Resolve the footprint used for grid-cell selection during tiling."""
    intermediate_wkt = _raster_footprint_wkt(Path(intermediate_product))
    if intermediate_wkt:
        return intermediate_wkt
    if str(product_mode).upper() == "S1TOPS" and swath:
        derived_wkt = sentinel1_swath_wkt_extractor_safe(source_product, swath, display_results=False, verbose=False)
        if derived_wkt:
            return derived_wkt
    if product_wkt:
        return product_wkt
    return resolve_source_product_wkt(source_product, product_mode)


def resolve_source_product_wkt(product_path, product_mode) -> str | None:
    """Resolve a source-product metadata footprint for missions that expose one."""
    mode = str(product_mode).upper()
    product_path = Path(product_path)
    if mode == "NISAR":
        return _nisar_wkt(product_path)
    if mode in {"TSX", "TDX", "TERRASAR", "TANDEMX"}:
        return _tsx_wkt(product_path)
    if mode in {"BM", "BIOMASS"}:
        return _raster_footprint_wkt(product_path)
    return None


def _raster_footprint_wkt(product_path: Path) -> str | None:
    if product_path.suffix.lower() == ".dim":
        try:
            return _dim_footprint_wkt(product_path)
        except Exception as exc:
            print(f"[WARN] Failed to derive raster footprint from {product_path}: {type(exc).__name__}: {exc}")
            return None
    if product_path.suffix.lower() not in {".tif", ".tiff", ".vrt", ".img", ".jp2"}:
        return None
    try:
        import rasterio
    except Exception:
        return None
    try:
        with rasterio.open(product_path) as dataset:
            if dataset.crs is None:
                return None
            corners = [
                dataset.transform * (0, 0),
                dataset.transform * (dataset.width, 0),
                dataset.transform * (dataset.width, dataset.height),
                dataset.transform * (0, dataset.height),
            ]
            transformer = pyproj.Transformer.from_crs(dataset.crs, 4326, always_xy=True)
            lonlat = [transformer.transform(x, y) for x, y in corners]
    except Exception:
        return None
    lonlat.append(lonlat[0])
    return "POLYGON ((" + ", ".join(f"{lon} {lat}" for lon, lat in lonlat) + "))"


def _nisar_wkt(product_path: Path) -> str:
    import h5py

    with h5py.File(product_path, "r") as h5_file:
        raw = h5_file["science/LSAR/identification/boundingPolygon"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return force_2d(shapely_wkt.loads(str(raw))).wkt


def _tsx_wkt(product_path: Path) -> str:
    candidates = _tsx_wkt_candidates(product_path)
    if not candidates:
        raise ValueError(f"Could not find TSX/TanDEM-X sceneCornerCoord footprint in {product_path}.")
    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise ValueError(f"Multiple TSX/TanDEM-X footprint candidates found in {product_path}.")
    return unique[0]


def _tsx_wkt_candidates(product_path: Path) -> list[str]:
    if product_path.is_file():
        if product_path.suffix.lower() == ".zip":
            return _tsx_wkts_from_zip(product_path)
        if product_path.suffix.lower() == ".xml":
            return _tsx_wkt_from_xml_bytes(product_path.read_bytes()) or []
        return []
    if not product_path.is_dir():
        raise FileNotFoundError(f"TSX/TanDEM-X product path does not exist: {product_path}")
    candidates: list[str] = []
    for xml_path in sorted(product_path.rglob("*.xml")) + sorted(product_path.rglob("*.XML")):
        candidates.extend(_tsx_wkt_from_xml_bytes(xml_path.read_bytes()) or [])
    for zip_path in sorted(product_path.rglob("*.zip")) + sorted(product_path.rglob("*.ZIP")):
        candidates.extend(_tsx_wkts_from_zip(zip_path))
    return candidates


def _tsx_wkts_from_zip(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return _tsx_wkts_from_archive(archive)


def _tsx_wkts_from_archive(archive: zipfile.ZipFile) -> list[str]:
    candidates: list[str] = []
    for member in sorted(archive.namelist()):
        lower = member.lower()
        if lower.endswith(".xml"):
            try:
                candidates.extend(_tsx_wkt_from_xml_bytes(archive.read(member)) or [])
            except ET.ParseError:
                continue
        elif lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as nested:
                candidates.extend(_tsx_wkts_from_archive(nested))
    return candidates


def _tsx_wkt_from_xml_bytes(raw_xml: bytes) -> list[str]:
    root = ET.fromstring(raw_xml)
    corners: list[tuple[float, float]] = []
    for element in root.iter():
        if _xml_local_name(element.tag) != "sceneCornerCoord":
            continue
        lon_text = _xml_child_text(element, "lon")
        lat_text = _xml_child_text(element, "lat")
        if lon_text is not None and lat_text is not None:
            corners.append((float(lon_text), float(lat_text)))
    if len(corners) < 3:
        return []
    ordered = _ordered_polygon_coords(corners)
    ordered.append(ordered[0])
    return [_format_polygon(ordered)]


def _ordered_polygon_coords(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique_points = list(dict.fromkeys(points))
    if len(unique_points) < 3:
        return unique_points
    center_lon = sum(lon for lon, _lat in unique_points) / len(unique_points)
    center_lat = sum(lat for _lon, lat in unique_points) / len(unique_points)
    ordered = sorted(unique_points, key=lambda point: atan2(point[1] - center_lat, point[0] - center_lon))
    polygon = Polygon(ordered)
    if not polygon.is_valid or polygon.area == 0:
        polygon = MultiPoint(unique_points).convex_hull
    if polygon.geom_type == "Polygon":
        polygon = orient(polygon, sign=1.0)
        return [(float(lon), float(lat)) for lon, lat in polygon.exterior.coords[:-1]]
    return ordered


def _format_polygon(points: list[tuple[float, float]]) -> str:
    return f"POLYGON(({', '.join(f'{_format_number(lon)} {_format_number(lat)}' for lon, lat in points)}))"


def _format_number(value: float) -> str:
    return str(float(value))


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_child_text(element: ET.Element, local_name: str) -> str | None:
    for child in element:
        if _xml_local_name(child.tag) == local_name and child.text is not None:
            value = child.text.strip()
            if value:
                return value
    return None
