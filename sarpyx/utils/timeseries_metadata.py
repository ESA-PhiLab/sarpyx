"""Tabular export of Sentinel annotation and applied SNAP metadata.

No SNAP invocation, downloads, or inferred orbit/geometry values.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


def product_id(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".zip", ".SAFE"):
        if name.lower().endswith(suffix.lower()):
            name = name[:-len(suffix)]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError(f"Unsafe product ID: {name!r}")
    return name


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def flatten_xml(root, source: str) -> list[dict]:
    """Keep indexed paths, source units, text and numeric values, including models."""
    records = []

    def visit(node, path):
        name = node.get("name") or _tag(node)
        path = f"{path}/{name}"
        text = (node.text or "").strip()
        if text and not len(node):
            records.append({
                "source": source, "path": path, "name": name,
                "value": text, "numeric_value": _number(text),
                "unit": node.get("unit"), "data_type": node.get("type"),
            })
        counts = Counter()
        for child in node:
            key = child.get("name") or _tag(child)
            index = counts[key]
            counts[key] += 1
            visit(child, f"{path}[{index}]")

    visit(root, "")
    return records


def annotation_documents(product: Path):
    if product.is_dir():
        manifest = product / "manifest.safe"
        if manifest.is_file():
            yield "manifest.safe", ET.parse(manifest).getroot()
        for path in sorted((product / "annotation").glob("*.xml")):
            yield path.name, ET.parse(path).getroot()
    elif product.suffix.lower() == ".zip":
        with ZipFile(product) as archive:
            for name in sorted(archive.namelist()):
                if re.search(r"(?:^|/)(?:annotation/[^/]+\.xml|manifest\.safe)$", name):
                    yield Path(name).name, ET.fromstring(archive.read(name))
    else:
        raise ValueError(f"Expected SAFE directory or ZIP: {product}")


def _child_text(node, name):
    for child in node.iter():
        if _tag(child) == name:
            return (child.text or "").strip()
    return None


def orbit_records(root, source):
    """Read source orbit samples and the orbit state vectors actually applied."""
    records = []
    for node in root.iter():
        snap = (node.get("name") or "").startswith("orbit_vector")
        if _tag(node) != "orbit" and not snap:
            continue
        if snap:
            values = {c.get("name"): (c.text or "").strip() for c in node}
            time = values.get("orbit_vector_time")
            components = [values.get(f"{axis}_{kind}") for kind in ("pos", "vel") for axis in "xyz"]
        else:
            time = _child_text(node, "time")
            components = []
            for kind in ("position", "velocity"):
                parent = next((c for c in node if _tag(c) == kind), None)
                components.extend([_child_text(parent, axis) if parent is not None else None for axis in "xyz"])
        if not time:
            continue
        records.append(dict(
            source=source, orbit_time=time, frame="ECEF",
            orbit_kind="applied_snap" if snap else "source_annotation",
            **dict(zip(("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"),
                       map(_number, components))),
        ))
    return records


def doppler_records(root, source):
    records = []
    for node in root.iter():
        if _tag(node) not in {"dcEstimate", "azimuthFmRate"}:
            continue
        for polynomial in node:
            if "polynomial" not in _tag(polynomial).lower():
                continue
            text = (polynomial.text or "").strip()
            records.append({
                "source": source, "model": _tag(node), "polynomial": _tag(polynomial),
                "azimuth_time": _child_text(node, "azimuthTime"),
                "reference_slant_range_time_s": _number(_child_text(node, "t0")),
                "coefficients": [float(v) for v in text.split()],
            })
    return records


FIELD_SCHEMA = pa.schema([
    ("source", pa.string()), ("path", pa.string()), ("name", pa.string()),
    ("value", pa.string()), ("numeric_value", pa.float64()),
    ("unit", pa.string()), ("data_type", pa.string()),
])
ORBIT_SCHEMA = pa.schema([
    ("source", pa.string()), ("orbit_time", pa.string()), ("frame", pa.string()),
    ("orbit_kind", pa.string()),
    *[(name, pa.float64()) for name in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")],
])
DOPPLER_SCHEMA = pa.schema([
    ("source", pa.string()), ("model", pa.string()), ("polynomial", pa.string()),
    ("azimuth_time", pa.string()), ("reference_slant_range_time_s", pa.float64()),
    ("coefficients", pa.list_(pa.float64())),
])
LOOK_SCHEMA = pa.schema([
    ("source", pa.string()), ("azimuth_time", pa.string()),
    ("line", pa.float64()), ("pixel", pa.float64()),
    ("latitude_deg", pa.float64()), ("longitude_deg", pa.float64()),
    ("height_m", pa.float64()), ("east", pa.float64()), ("north", pa.float64()),
    ("up", pa.float64()), ("orbit_source", pa.string()), ("orbit_kind", pa.string()),
])


def look_vector_records(documents, orbits):
    """Ground-to-sensor ENU vectors at existing annotation tie points.

    No raster interpolation, pixel-grid change, or orbit extrapolation. Prefer
    applied orbit samples from the matching swath; label annotation-only fallback.
    """
    import pandas as pd
    from pyproj import Transformer

    def seconds(text):
        if not text:
            return None
        try:
            # SNAP's textual UTC dates can use DD-Mon-YYYY as well as ISO8601.
            return pd.to_datetime(text, utc=True).timestamp()
        except (ValueError, TypeError):
            return None

    transformer = Transformer.from_crs(4979, 4978, always_xy=True)
    grouped = {}
    for row in orbits:
        time = seconds(row["orbit_time"])
        position = [row[key] for key in ("x_m", "y_m", "z_m")]
        if time is not None and all(value is not None for value in position):
            grouped.setdefault((row["source"], row["orbit_kind"]), {})[time] = position
    result = []
    for source, root in documents:
        match = re.search(r"iw[123]", source, re.I)
        strip_match = re.search(r"(?:^|[-_/])(s[1-6])(?:[-_/]|$)", source, re.I)
        swath = match[0].lower() if match else (strip_match[1].lower() if strip_match else None)
        candidates = [(key, samples) for key, samples in grouped.items()
                      if key[0] == source or (key[1] == "applied_snap" and
                         swath is not None and swath in key[0].lower())]
        candidates.sort(key=lambda item: item[0][1] != "applied_snap")
        for point in root.iter():
            if _tag(point) != "geolocationGridPoint":
                continue
            time_text = _child_text(point, "azimuthTime")
            time = seconds(time_text)
            lat, lon, height = [_number(_child_text(point, key)) for key in ("latitude", "longitude", "height")]
            if time is None or any(value is None for value in (lat, lon, height)):
                continue
            for (orbit_source, orbit_kind), samples in candidates:
                times = sorted(samples)
                if len(times) < 2 or not times[0] <= time <= times[-1]:
                    continue
                positions = np.asarray([samples[t] for t in times])
                satellite = np.array([np.interp(time, times, positions[:, axis]) for axis in range(3)])
                direction = satellite - np.array(transformer.transform(lon, lat, height))
                norm = np.linalg.norm(direction)
                if not np.isfinite(norm) or norm == 0:
                    continue
                x, y, z = direction / norm
                longitude, latitude = np.deg2rad([lon, lat])
                slon, clon, slat, clat = np.sin(longitude), np.cos(longitude), np.sin(latitude), np.cos(latitude)
                result.append({
                    "source": source, "azimuth_time": time_text,
                    "line": _number(_child_text(point, "line")),
                    "pixel": _number(_child_text(point, "pixel")),
                    "latitude_deg": lat, "longitude_deg": lon, "height_m": height,
                    "east": -slon * x + clon * y,
                    "north": -slat * clon * x - slat * slon * y + clat * z,
                    "up": clat * clon * x + clat * slon * y + slat * z,
                    "orbit_source": orbit_source, "orbit_kind": orbit_kind,
                })
                break
    return result


def _summary(records, acquisition_id):
    # Source Sentinel annotation values are Hz. SNAP Abstracted_Metadata uses
    # MHz for radar_frequency/range_bandwidth; never silently treat those as Hz.
    aliases = {
        "radar_frequency_hz": (("radarFrequency", 1), ("radar_frequency", 1e6)),
        "range_bandwidth_hz": (("processingBandwidth", 1), ("range_bandwidth", 1e6)),
        "azimuth_bandwidth_hz": (("azimuth_bandwidth", 1),),
        "pulse_repetition_frequency_hz": (("prf", 1), ("pulse_repetition_frequency", 1)),
        "range_spacing_m": (("range_spacing", 1), ("rangePixelSpacing", 1)),
        "azimuth_spacing_m": (("azimuth_spacing", 1), ("azimuthPixelSpacing", 1)),
        "absolute_orbit": (("ABS_ORBIT", 1), ("absoluteOrbitNumber", 1), ("orbitNumber", 1)),
        "relative_orbit": (("REL_ORBIT", 1), ("relativeOrbitNumber", 1)),
    }
    rows = []
    # Values can vary by annotation/swath/polarization: preserve a row per source.
    for source in sorted({r["source"] for r in records}):
        fields = [r for r in records if r["source"] == source]
        row = {"acquisition_id": acquisition_id, "source": source}
        for target, names in aliases.items():
            row[target] = None
            for name, factor in names:
                candidates = [r for r in fields if r["name"] == name and r["numeric_value"] is not None]
                if name == "processingBandwidth":
                    candidates = [r for r in candidates if "/rangeProcessing" in r["path"]]
                values = {r["numeric_value"] for r in candidates}
                if len(values) == 1:
                    unit = candidates[0]["unit"]
                    if target.endswith("_hz") and unit:
                        factor = {"hz": 1, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(unit.lower(), factor)
                    row[target] = values.pop() * factor
                    break
        azimuth_values = {r["numeric_value"] for r in fields
                          if r["name"] == "processingBandwidth"
                          and "/azimuthProcessing" in r["path"] and r["numeric_value"] is not None}
        if row["azimuth_bandwidth_hz"] is None and len(azimuth_values) == 1:
            row["azimuth_bandwidth_hz"] = azimuth_values.pop()
        for target, names in {
            "pass_direction": ("pass", "PASS"),
            "swath": ("swath", "SWATH"),
            "polarization": ("polarisation", "polarization"),
            "start_time": ("startTime", "first_line_time"),
            "stop_time": ("stopTime", "last_line_time"),
            "orbit_file": ("orbit_state_vector_file",),
            "mission": ("missionId", "MISSION"),
            "mode": ("mode", "ACQUISITION_MODE"),
            "antenna_pointing": ("antenna_pointing",),
        }.items():
            row[target] = next((r["value"] for name in names for r in fields if r["name"] == name), None)
        rows.append(row)
    schema = pa.schema([
        ("acquisition_id", pa.string()), ("source", pa.string()),
        *[(name, pa.float64()) for name in aliases],
        *[(name, pa.string()) for name in ("pass_direction", "swath", "polarization", "start_time", "stop_time", "orbit_file", "mission", "mode", "antenna_pointing")],
    ])
    return pa.Table.from_pylist(rows, schema=schema)


def export_metadata(dataset: Path, identity: str, documents, *, pair=False,
                    processing=False, geometry_documents=()) -> Path:
    documents = list(documents)
    fields, orbits, doppler = [], [], []
    for source, root in documents:
        fields.extend(flatten_xml(root, source))
        orbits.extend(orbit_records(root, source))
        doppler.extend(doppler_records(root, source))
    if not fields:
        raise ValueError(f"No annotation/processed metadata found for {identity}")
    looks = look_vector_records([*documents, *geometry_documents], orbits)
    digest = hashlib.sha256(json.dumps({
        "fields": fields, "looks": looks,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()[:20]
    parent = dataset / "metadata"
    if processing or pair:
        parent = parent / "processing"
    parent = parent / ("pairs" if pair else "acquisitions") / identity
    target = parent / digest
    if target.exists():
        return target
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".metadata-", dir=parent) as temp:
        stage = Path(temp) / digest
        stage.mkdir()
        tables = {
            "fields": pa.Table.from_pylist(fields, schema=FIELD_SCHEMA),
            "orbit_states": pa.Table.from_pylist(orbits, schema=ORBIT_SCHEMA),
            "doppler": pa.Table.from_pylist(doppler, schema=DOPPLER_SCHEMA),
            "summary": _summary(fields, identity),
            "look_vectors": pa.Table.from_pylist(looks, schema=LOOK_SCHEMA),
            "baselines": pa.Table.from_pylist(
                [row for row in fields if "baseline" in row["path"].lower()],
                schema=FIELD_SCHEMA),
        }
        for name, table in tables.items():
            table = table.replace_schema_metadata({
                b"worldsar_table": name.encode(),
                b"schema": b"worldsar-timeseries-v1",
                b"notes": (
                    b"Field suffixes specify SI units; fields/baselines preserve source units and paths. "
                    b"Look vectors: ground-to-sensor ENU at annotation tie points, linear orbit interpolation, no extrapolation."
                ),
            })
            pq.write_table(table, stage / f"{name}.parquet", compression="zstd")
        try:
            stage.rename(target)
        except OSError:
            if not target.exists():
                raise
    return target


def processed_documents(run: Path):
    """Use final products only, including per-swath applied orbit metadata."""
    for path in sorted(run.rglob("*_TC.dim")):
        # Per-EPSG reruns need not replicate large metadata tables.
        if any(part in {"worldsar_tc_epsg", "worldsar_reprojected", "worldsar_subap_tc"}
               for part in path.relative_to(run).parts[:-1]):
            continue
        root = ET.parse(path).getroot()
        # Exclude raster inventory/geocoding; preserve the metadata tree.
        parents = {child: parent for parent in root.iter() for child in parent}
        for node in root.iter("MDElem"):
            if parents.get(node) is not None and parents[node].tag == "MDElem":
                continue
            if node.get("name") in {"Original_Product_Metadata", "Processing_Graph"}:
                continue
            # Original annotation is already exported once from the SAFE/ZIP.
            # Preserve applied Abstracted_Metadata, slave orbits, ESD and baseline
            # records, not another embedded copy of original product metadata.
            for parent in list(node.iter()):
                for child in list(parent):
                    if child.tag == "MDElem" and child.get("name") in {"Original_Product_Metadata", "Processing_Graph"}:
                        parent.remove(child)
            # Random job-private directory names are execution provenance, not
            # part of acquisition identity or scientific metadata.
            source = "/".join(part for part in path.relative_to(run).parts
                              if not part.startswith("worldsar_tmp_"))
            yield source, node
