"""Persisted metadata for regenerating WorldSAR validation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sarpyx.utils.worldsar_h5 import normalize_expected_tile_geometries
from sarpyx.snapflow import config
from sarpyx.snapflow.tile_selection import select_intersecting_grid_rectangles

TILING_MANIFEST_KEYS = (
    "name",
    "cuts_dir",
    "report_path",
    "grid_path",
    "cut_failed",
    "tile_writer",
    "source_wkt",
    "pre_tc_wkt",
    "post_tc_wkt",
    "report_source_wkt",
    "candidate_tile_geometries",
    "expected_tile_geometries",
    "candidate_tiles",
    "expected_tiles",
    "actual_tiles",
    "skipped_tiles",
    "partial_tiles",
    "failed_tiles",
    "missing_tiles",
    "extra_tiles",
    "crs_groups",
    "skipped_tile_reasons",
    "partial_tile_reasons",
)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _tiling_manifest_path(report_path: Path) -> Path:
    return report_path.with_suffix(".json")


def write_tiling_manifest(report_path, tiling_result: dict[str, Any]):
    report_path = Path(report_path)
    manifest = {
        key: _json_safe(tiling_result[key])
        for key in TILING_MANIFEST_KEYS
        if key in tiling_result
    }
    manifest["cut_report_path"] = str(report_path)
    path = _tiling_manifest_path(report_path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_tiling_manifest(report_path) -> dict[str, Any]:
    if not report_path:
        return {}
    path = _tiling_manifest_path(Path(report_path))
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_count(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _parse_section_tile(line: str) -> tuple[str, str] | None:
    if not line.startswith("- ") or ": " not in line:
        return None
    tile, detail = line[2:].split(": ", 1)
    return tile.strip(), detail.strip()


def _parse_cut_report_text(report_path) -> dict[str, Any]:
    if not report_path:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {}
    parsed: dict[str, Any] = {"cut_report_path": str(path)}
    labels = {
        "Product WKT": "source_wkt",
        "Candidate intersecting tiles": "candidate_tile_count",
        "Expected full-data tiles": "expected_tile_count",
        "Expected tiles": "legacy_expected_tile_count",
        "Actual tiles on disk": "actual_tile_count",
        "Partial edge tiles (not deliverables)": "partial_tile_count",
        "Skipped tiles (outside raster bounds)": "skipped_tile_count",
        "Skipped tiles (incomplete raster coverage)": "skipped_incomplete_tile_count",
        "Failed tiles (this run)": "failed_tile_count",
        "Missing tiles": "missing_tile_count",
        "Unexpected tiles": "extra_tile_count",
    }
    section = None
    skipped_reasons: dict[str, str] = {}
    partial_reasons: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line in {"Skipped tiles:", "Partial edge tiles:", "Failed tiles:"}:
            section = line.rstrip(":")
            continue
        item = _parse_section_tile(line)
        if item and section == "Skipped tiles":
            skipped_reasons[item[0]] = item[1]
            continue
        if item and section == "Partial edge tiles":
            partial_reasons[item[0]] = item[1]
            continue
        if ": " not in line:
            continue
        label, value = line.split(": ", 1)
        key = labels.get(label)
        if not key:
            continue
        if key.endswith("_count"):
            count = _parse_count(value)
            if count is not None:
                parsed[key] = count
        else:
            parsed[key] = value
    if skipped_reasons:
        parsed["skipped_tiles"] = sorted(skipped_reasons)
        parsed["skipped_tile_reasons"] = skipped_reasons
    if partial_reasons:
        parsed["partial_tiles"] = sorted(partial_reasons)
        parsed["partial_tile_reasons"] = partial_reasons
    if parsed.get("source_wkt"):
        parsed.setdefault("report_source_wkt", parsed["source_wkt"])
        parsed.setdefault("pre_tc_wkt", parsed["source_wkt"])
        parsed.setdefault("post_tc_wkt", parsed["source_wkt"])
    if "legacy_expected_tile_count" in parsed:
        legacy_count = parsed.pop("legacy_expected_tile_count")
        parsed.setdefault("candidate_tile_count", legacy_count)
        skipped_total = parsed.get("skipped_tile_count", 0) + parsed.pop("skipped_incomplete_tile_count", 0)
        if skipped_total:
            parsed["skipped_tile_count"] = skipped_total
        parsed.setdefault("expected_tile_count", max(legacy_count - skipped_total, 0))
    return parsed


def _default_grid_path() -> Path | None:
    candidates = []
    if config.GRID_PATH:
        candidates.append(Path(config.GRID_PATH).expanduser())
    candidates.extend([config.PROJECT_ROOT / "grid" / "grid_10km.geojson", config.PROJECT_ROOT.parent / "grid" / "grid_10km.geojson"])
    return next((path for path in candidates if path.exists()), None)


def _recover_grid_geometries(tiling_result: dict[str, Any]) -> None:
    if tiling_result.get("candidate_tile_geometries") or not tiling_result.get("source_wkt"):
        return
    grid_path = Path(tiling_result["grid_path"]).expanduser() if tiling_result.get("grid_path") else _default_grid_path()
    if not grid_path or not grid_path.exists():
        return
    rectangles = select_intersecting_grid_rectangles(tiling_result["source_wkt"], grid_path)
    candidate_geometries = normalize_expected_tile_geometries(rectangles)
    skipped = set(tiling_result.get("skipped_tiles") or [])
    partial = set(tiling_result.get("partial_tiles") or [])
    expected = {
        tile: coords
        for tile, coords in candidate_geometries.items()
        if tile not in skipped and tile not in partial
    }
    tiling_result.setdefault("grid_path", str(grid_path))
    tiling_result["candidate_tile_geometries"] = candidate_geometries
    tiling_result.setdefault("candidate_tiles", sorted(candidate_geometries))
    tiling_result["expected_tile_geometries"] = expected
    tiling_result.setdefault("expected_tiles", sorted(expected))


def hydrate_tiling_result(tiling_result):
    if tiling_result is None:
        return None
    merged = dict(tiling_result)
    report_path = merged.get("report_path") or merged.get("cut_report_path")
    persisted = _parse_cut_report_text(report_path)
    persisted.update(_load_tiling_manifest(report_path))
    for key, value in persisted.items():
        if key not in merged or merged[key] in (None, [], {}, ""):
            merged[key] = value
    if "report_path" not in merged and merged.get("cut_report_path"):
        merged["report_path"] = merged["cut_report_path"]
    _recover_grid_geometries(merged)
    if report_path and merged.get("candidate_tile_geometries"):
        manifest_path = _tiling_manifest_path(Path(report_path))
        if not manifest_path.exists():
            write_tiling_manifest(report_path, merged)
    return merged
