"""Small, content-addressed WorldSAR dataset writer.

Consumes validated native tiles. Does not change sampling or feature computation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

SCHEMA = "worldsar-timeseries-v1"
POL_RE = re.compile(r"(?:^|_)(HH|HV|VH|VV)(?:_|$)", re.I)
IQ_RE = re.compile(r"^([iq])_(HH|HV|VH|VV)$", re.I)
PAIR_RE = re.compile(r"^(i_ifg|q_ifg|phase_ifg|coh|coherence)_(.+)$", re.I)


def normalize_pair_band_name(name):
    """Match SNAP aliases without changing polarization/date identity.

    TopoPhaseRemoval leaves materialized I/Q names unchanged but declares their
    virtual phase as Phase_ifg_srd_<pol>_<dates>. SRD is not another observation.
    """
    name = re.sub(r"(^|_)IW\d+_", r"\1", name, flags=re.I)
    return re.sub(r"^(phase_ifg)_srd_", r"\1_", name, flags=re.I)


def pair_source_bands(names):
    """Select complex IFG + coherence before TC, phase only as a fallback."""
    matched = {}
    for name in sorted(names):
        match = PAIR_RE.match(normalize_pair_band_name(name))
        if match:
            parts = matched.setdefault(match[2].lower(), {})
            component = match[1].lower()
            component = "coh" if component == "coherence" else component
            if component in parts:
                raise ValueError(f"Ambiguous pair bands: {parts[component]} and {name}")
            parts[component] = name
    selected = []
    for suffix, parts in sorted(matched.items()):
        coherence = parts.get("coh") or parts.get("coherence")
        if not coherence:
            raise ValueError(f"No coherence for interferogram {suffix}")
        if ("i_ifg" in parts) != ("q_ifg" in parts):
            raise ValueError(f"Incomplete interferogram I/Q for {suffix}")
        if "i_ifg" in parts and "q_ifg" in parts:
            selected.extend([parts["i_ifg"], parts["q_ifg"]])
        elif "phase_ifg" in parts:
            selected.append(parts["phase_ifg"])
        else:
            raise ValueError(f"No complete interferogram for coherence {suffix}")
        selected.append(coherence)
    if not selected:
        raise ValueError("No interferogram/coherence bands found")
    return selected


def fingerprint(value, arrays=None):
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    for name, data in sorted((arrays or {}).items()):
        data = np.ascontiguousarray(data)
        digest.update(name.encode())
        digest.update(str((data.shape, data.dtype.str)).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()[:24]


def relative_asset(target: Path, dataset: Path) -> str:
    return target.relative_to(dataset).as_posix()


def sensing_time_ns(identity, fallback=None):
    match = re.search(r"(?:^|_)(\d{8}T\d{6})(?:_|$)", identity)
    if match:
        value = datetime.strptime(match[1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return int(value.timestamp()) * 1_000_000_000
    if fallback:
        import pandas as pd
        value = pd.Timestamp(fallback)
        value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return value.value
    raise ValueError(f"No sensing time for {identity}")


def write_store(target, attrs, fields):
    """Idempotent immutable store; stage beside target, then rename."""
    target = Path(target)
    if target.exists():
        if zarr.open_group(str(target), mode="r").attrs.get("asset_id") != attrs["asset_id"]:
            raise ValueError(f"Conflicting existing asset: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=target.parent) as temp:
        stage = Path(temp) / target.name
        group = zarr.create_group(str(stage), zarr_format=3)
        group.attrs.update(attrs)
        for name, (data, dims, field_attrs) in fields.items():
            parent, _, leaf = name.rpartition("/")
            owner = group.require_group(parent) if parent else group
            chunks = tuple(1 if dim not in {"x", "y"} else min(128, size)
                           for dim, size in zip(dims, data.shape))
            array = owner.create_array(
                leaf, data=data, chunks=chunks or "auto",
                dimension_names=dims, config={"write_empty_chunks": True},
            )
            array.attrs.update(field_attrs)
        # Read every field back before deleting any corresponding native tile.
        for name, (data, _, _) in fields.items():
            actual = group[name][...]
            equal = np.array_equal(actual, data, equal_nan=True) if data.dtype.kind in "fc" else np.array_equal(actual, data)
            if not equal:
                raise ValueError(f"Write verification failed: {name}")
        try:
            stage.rename(target)
        except OSError:
            if not target.exists():
                raise


def _pol(name):
    match = POL_RE.search(name)
    if not match:
        raise ValueError(f"Missing polarization in {name}")
    return match[1].upper()


def _finite(data, attrs):
    valid = np.isfinite(data)
    nodata = attrs.get("nodata")
    if nodata is not None:
        valid &= data != nodata
    return valid


def normalized_fields(group, kind):
    """Return normalized arrays and optional elevation. Preserve feature names."""
    bands = group["bands"]
    names = set(bands.keys())
    shapes = {bands[name].shape for name in names}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("Native tile must have uniformly shaped 2D bands")
    shape = next(iter(shapes))
    fields = {}
    elevation = None
    if kind == "pair":
        selected = pair_source_bands(names)
        by_pol = {}
        for name in selected:
            pol = _pol(name)
            parts = by_pol.setdefault(pol, {})
            match = PAIR_RE.match(normalize_pair_band_name(name))
            key = match[1].lower()
            if key in parts:
                raise ValueError(f"Multiple pair observations for {pol}: preserve segment identity")
            parts[key] = np.asarray(bands[name][:], dtype=np.float32)
        pols = sorted(by_pol)
        complex_ifg = all("i_ifg" in by_pol[p] and "q_ifg" in by_pol[p] for p in pols)
        if not complex_ifg and not all("phase_ifg" in by_pol[p] for p in pols):
            raise ValueError("Mixed complex/phase pair representations are not supported")
        ifg, coh = [], []
        for pol in pols:
            parts = by_pol[pol]
            ifg.append(parts["i_ifg"] + 1j * parts["q_ifg"] if complex_ifg else parts["phase_ifg"])
            coh.append(parts["coh"] if "coh" in parts else parts["coherence"])
        dims = ("pair", "pol", "y", "x")
        signal = np.stack(ifg)[None]
        coherence = np.stack(coh)[None]
        fields["interferogram" if complex_ifg else "wrapped_phase"] = (
            signal, dims, {"unit": "complex_amplitude" if complex_ifg else "rad",
                           "phase_convention": "SNAP reference/secondary convention; unchanged"},
        )
        fields["coherence"] = (coherence, dims, {"unit": "1", "long_name": "pairwise interferometric coherence magnitude"})
        valid = np.isfinite(signal) & np.isfinite(coherence) & (coherence >= 0) & (coherence <= 1)
        fields["finite_support"] = (valid, dims, {"description": "Finite IFG and coherence in [0,1]; zero coherence is allowed, not a coverage mask"})
    else:
        pairs = {}
        consumed = set()
        for name in sorted(names):
            match = IQ_RE.match(name)
            if match:
                pairs.setdefault(match[2].upper(), {})[match[1].lower()] = name
                consumed.add(name)
        pols = sorted(pairs)
        if not pols or any(set(parts) != {"i", "q"} for parts in pairs.values()):
            raise ValueError("Acquisition requires complete full-aperture I/Q per polarization")
        iq, support = [], []
        for pol in pols:
            i_name, q_name = pairs[pol]["i"], pairs[pol]["q"]
            i, q = np.asarray(bands[i_name][:]), np.asarray(bands[q_name][:])
            iq.append((i + 1j * q).astype(np.complex64))
            support.append(_finite(i, bands[i_name].attrs) & _finite(q, bands[q_name].attrs))
        dims = ("time", "pol", "y", "x")
        fields["gslc"] = (np.stack(iq)[None], dims, {"description": "Calibrated full-aperture complex GSLC; original I/Q sampling"})
        fields["finite_support"] = (np.stack(support)[None], dims, {"description": "Finite source I/Q, excluding declared nodata; not a geometric coverage mask"})
        feature_names = set(group.attrs.get("subap_feature_bands", []))
        geometry = {
            "localincidenceangle": "local_incidence_angle",
            "incidenceanglefromellipsoid": "incidence_angle_ellipsoid",
            "projectedlocalincidenceangle": "projected_local_incidence_angle",
            "layovershadowmask": "layover_shadow_mask",
        }
        for name in sorted(names - consumed):
            compact = re.sub(r"[^a-z]", "", name.lower())
            attrs = dict(bands[name].attrs)
            if name not in feature_names:
                attrs.pop("polarization", None)
            data = np.asarray(bands[name][:])
            if compact == "elevation":
                elevation = (data.astype(np.float32), attrs)
            elif name in feature_names:
                fields[f"features/subaperture/{name}"] = (data[None], ("time", "y", "x"), attrs)
            elif compact in {"entropy", "alpha", "anisotropy"}:
                fields[f"features/polarimetry/{compact}"] = (data[None], ("time", "y", "x"), attrs)
            elif compact in geometry:
                fields[f"geometry/{geometry[compact]}"] = (data[None], ("time", "y", "x"), attrs)
            elif name.lower().startswith(("intensity_", "amplitude_", "phase_", "sigma0_")):
                # These views are redundant with full-aperture complex I/Q here.
                if _pol(name) not in pols:
                    raise ValueError(f"Unmatched derived signal band: {name}")
            else:
                raise ValueError(f"Unclassified acquisition band {name!r}; refusing silent data loss")
    fields["pol"] = (np.asarray(pols, dtype="U2"), ("pol",), {})
    return fields, elevation, shape


def publish_tile(source, dataset, *, kind, identity, swath, grid_id,
                 acquisition_ids, metadata_refs, processing):
    source, dataset = Path(source), Path(dataset)
    for component in (identity, swath, grid_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", component) or component in {".", ".."}:
            raise ValueError(f"Unsafe dataset path component: {component!r}")
    native = zarr.open_group(str(source), mode="r")
    attrs = dict(native.attrs)
    cell = attrs.get("tile")
    if not cell or not re.fullmatch(r"[A-Za-z0-9_.-]+", cell) or cell in {".", ".."}:
        raise ValueError(f"Unsafe cell ID: {cell!r}")
    fields, elevation, shape = normalized_fields(native, kind)
    time_attrs = {"units": "nanoseconds since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian", "time_scale": "UTC"}
    if kind == "acquisition":
        fields["time"] = (np.array([sensing_time_ns(acquisition_ids[0], attrs.get("acquisition_time"))], dtype=np.int64), ("time",), time_attrs)
    else:
        reference, secondary = map(sensing_time_ns, acquisition_ids)
        fields["pair"] = (np.array([identity]), ("pair",), {})
        fields["reference_time"] = (np.array([reference], dtype=np.int64), ("pair",), time_attrs)
        fields["secondary_time"] = (np.array([secondary], dtype=np.int64), ("pair",), time_attrs)
        fields["temporal_baseline"] = (np.array([(secondary - reference) / 1e9]), ("pair",), {"unit": "s"})
    transform = attrs.get("transform")
    if not transform or len(transform) != 6:
        raise ValueError("Missing source affine transform")
    spatial = {key: attrs[key] for key in ("epsg", "crs_wkt", "transform", "pixel_spacing") if key in attrs}
    spatial["shape"] = list(shape)
    a, b, c, d, e, f = map(float, transform)
    # Coordinates describe original sampling; no snapping or resampling.
    if b == 0 and d == 0:
        fields["x"] = (c + (np.arange(shape[1]) + .5) * a, ("x",), {"unit": "m"})
        fields["y"] = (f + (np.arange(shape[0]) + .5) * e, ("y",), {"unit": "m"})
    attrs_out = {
        "schema": SCHEMA, "kind": kind, "cell_id": cell, "grid_id": grid_id,
        "observation_id": identity, "acquisition_ids": acquisition_ids, "swath": swath,
        "metadata_refs": metadata_refs, "processing": processing,
        "pass_direction": attrs.get("pass_direction"), "mission": attrs.get("mission"),
        "acquisition_time": attrs.get("acquisition_time"),
        **spatial,
    }
    data_root = dataset / "data"
    if elevation is not None:
        data, dem_attrs = elevation
        static_attrs = {"schema": SCHEMA, "kind": "static", "cell_id": cell,
                        "grid_id": grid_id, **spatial,
                        "dem_provenance": processing.get("dem", {}),
                        "source_band_attributes": dem_attrs}
        terrain_id = fingerprint(static_attrs, {"elevation": data})
        static_attrs["asset_id"] = terrain_id
        terrain = data_root / "static" / cell / f"{terrain_id}.zarr"
        static_fields = {
            "elevation": (data, ("y", "x"), {"unit": "m", **dem_attrs}),
            "valid": (_finite(data, dem_attrs), ("y", "x"), {}),
            **{k: v for k, v in fields.items() if k in {"x", "y"}},
        }
        write_store(terrain, static_attrs, static_fields)
        attrs_out["terrain_ref"] = relative_asset(terrain, dataset)
    # The actual samples/metadata are part of the key, so reruns cannot
    # overwrite different outputs or silently reuse an incompatible DEM grid.
    field_metadata = {
        name: {"dimensions": dims, "attributes": field_attrs}
        for name, (_, dims, field_attrs) in fields.items()
    }
    asset_id = fingerprint(
        {"attributes": attrs_out, "fields": field_metadata},
        {k: v[0] for k, v in fields.items()},
    )
    attrs_out["asset_id"] = asset_id
    target = data_root / ("pairs" if kind == "pair" else "acquisitions") / identity / swath / cell / f"{asset_id}.zarr"
    write_store(target, attrs_out, fields)
    return {"cell_id": cell, "swath": swath, "kind": kind, "observation_id": identity,
            "asset_id": asset_id, "uri": relative_asset(target, dataset),
            "height": shape[0], "width": shape[1]}


def load_observation(dataset, uri):
    """Load one observation plus referenced metadata (arrays remain lazy Zarr arrays)."""
    import pyarrow.parquet as pq

    dataset = Path(dataset)
    group = zarr.open_group(str(dataset / uri), mode="r")
    metadata = {}
    for role, reference in group.attrs["metadata_refs"].items():
        directory = dataset / reference
        metadata[role] = {path.stem: pq.read_table(path) for path in directory.glob("*.parquet")}
    terrain = group.attrs.get("terrain_ref")
    return {"data": group, "metadata": metadata,
            "terrain": zarr.open_group(str(dataset / terrain), mode="r") if terrain else None}


def find_assets(dataset, cell_id, *, kind=None, swath=None, series_id=None, modality="S1"):
    """Query a cell's catalog without copying arrays or selecting a 'latest' version."""
    import pyarrow.dataset as ds

    if series_id is None:
        catalog = Path(dataset) / "catalog"
        paths = sorted(str(path) for path in catalog.glob("**/assets.parquet"))
    else:
        for component in (str(series_id), modality):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", component) or component in {".", ".."}:
                raise ValueError(f"Unsafe source grouping ID: {component!r}")
        catalog = Path(dataset) / "series" / str(series_id) / modality
        paths = sorted(str(path) for path in catalog.glob("**/*.parquet"))
    if not paths:
        return []
    predicate = ds.field("cell_id") == cell_id
    if kind is not None:
        predicate &= ds.field("kind") == kind
    if swath is not None:
        predicate &= ds.field("swath") == swath
    table = ds.dataset(paths, format="parquet").to_table(filter=predicate)
    # Two observation batches can reference the same immutable asset.
    unique = {row["uri"]: row for row in table.to_pylist()}
    return sorted(unique.values(), key=lambda row: (row["observation_id"], row["swath"], row["asset_id"]))
