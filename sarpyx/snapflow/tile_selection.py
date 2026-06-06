"""WorldSAR grid-cell selection for tiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyproj
from shapely import wkt
from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep

from sarpyx.utils.geos import grid_cell_utm_bbox

GRID_MARGIN_DEGREES = 1.0


def select_intersecting_grid_rectangles(product_wkt: str, grid_geojson_path: str | Path) -> list[dict[str, Any]]:
    product_geom = wkt.loads(product_wkt)
    prepared_product = prep(product_geom)
    minx, miny, maxx, maxy = product_geom.bounds
    candidate_bounds = box(minx - GRID_MARGIN_DEGREES, miny - GRID_MARGIN_DEGREES, maxx + GRID_MARGIN_DEGREES, maxy + GRID_MARGIN_DEGREES)
    rectangles = []
    for feature in _grid_features(grid_geojson_path):
        if not _feature_has_tile_metadata(feature):
            continue
        point = feature["geometry"]["coordinates"]
        if not candidate_bounds.intersects(Point(point[0], point[1])):
            continue
        rectangle = _grid_cell_rectangle(feature)
        if prepared_product.intersects(_rectangle_polygon(rectangle)):
            rectangles.append(rectangle)
    return _dedupe_rectangles(rectangles)


def _grid_features(grid_geojson_path: str | Path) -> list[dict[str, Any]]:
    with Path(grid_geojson_path).open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    return list(payload.get("features") or [])


def _feature_has_tile_metadata(feature: dict[str, Any]) -> bool:
    props = feature.get("properties") or {}
    coords = (feature.get("geometry") or {}).get("coordinates") or []
    return bool(props.get("name") and props.get("epsg") and len(coords) >= 2)


def _grid_cell_rectangle(feature: dict[str, Any]) -> dict[str, Any]:
    epsg = int(str(feature["properties"]["epsg"]).split(":")[-1])
    x_min, y_min, x_max, y_max = grid_cell_utm_bbox({"BL": feature}, epsg)
    transformer = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    corners = {
        "TL": transformer.transform(x_min, y_max),
        "TR": transformer.transform(x_max, y_max),
        "BR": transformer.transform(x_max, y_min),
        "BL": transformer.transform(x_min, y_min),
    }
    return {
        corner: {
            "type": "Feature",
            "properties": dict(feature["properties"]),
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        }
        for corner, (lon, lat) in corners.items()
    }


def _rectangle_polygon(rectangle: dict[str, Any]) -> Polygon:
    return Polygon([rectangle[corner]["geometry"]["coordinates"] for corner in ("TL", "TR", "BR", "BL")])


def _dedupe_rectangles(rectangles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for rectangle in rectangles:
        name = rectangle["BL"]["properties"]["name"]
        if name in seen:
            continue
        seen.add(name)
        unique.append(rectangle)
    return sorted(unique, key=lambda item: item["BL"]["properties"]["name"])
