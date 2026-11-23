"""Terrain correction for mixed complex and bounded polarimetric bands."""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from pyproj import CRS

from sarpyx.snapflow.dimap import (
    clone_source_raster_metadata,
    copy_materialized_src_files,
    materialized_band_names,
    spectral_band_names,
    validate_same_dimensions,
)
from sarpyx.snapflow.gpt import create_gpt_operator
from sarpyx.snapflow.raster import _read_crs_wkt, _read_geotransform

POLARIMETRIC_BANDS = frozenset({"alpha", "entropy", "anisotropy"})


def polarimetric_tc_params(tc_params: dict, bands: list[str]) -> dict:
    """Retain production geometry but use a range-preserving scalar kernel."""
    params = dict(tc_params)
    for key in ("output_name", "ordered_tops_bands", "split_polarimetric_resampling"):
        params.pop(key, None)
    params.update(
        source_bands=list(bands), output_complex=False, img_resampling_method="BILINEAR_INTERPOLATION",
        save_dem=False, save_lat_lon=False, save_incidence_angle_from_ellipsoid=False,
        save_local_incidence_angle=False, save_projected_local_incidence_angle=False,
        save_layover_shadow_mask=False,
    )
    return params


def _split_gpt_limits(gpt_kwargs: dict) -> tuple[dict, dict] | None:
    """Return conservative parallel budgets only when the caller set all limits."""
    q = gpt_kwargs.get("gpt_parallelism")
    if not isinstance(q, int) or q < 2:
        return None
    unit_to_mb = {"m": 1, "g": 1024}
    budget = {}
    for key in ("gpt_memory", "gpt_cache_size"):
        match = re.fullmatch(r"([1-9][0-9]*)([mMgG])", str(gpt_kwargs.get(key)))
        if not match:
            return None
        budget[key] = int(match[1]) * unit_to_mb[match[2].lower()]
    if budget["gpt_memory"] < 4096 or budget["gpt_cache_size"] < 256:
        return None
    split = budget["gpt_memory"] // 2
    cache = min(budget["gpt_cache_size"] // 2, split // 2)
    scalar_q = max(1, q // 3)
    return tuple(
        dict(gpt_kwargs, gpt_parallelism=threads, gpt_memory=f"{heap}M", gpt_cache_size=f"{cache}M")
        for threads, heap in ((q - scalar_q, budget["gpt_memory"] - split), (scalar_q, split))
    )


def _require_payload(product: Path, bands: list[str]) -> None:
    names = spectral_band_names(product)
    materialized = set(materialized_band_names(product))
    for band in bands:
        if names.count(band) != 1 or band not in materialized or any(
            not (product.with_suffix(".data") / f"{band}.{ext}").is_file()
            for ext in ("hdr", "img")
        ):
            raise RuntimeError(f"Missing or duplicate DIMAP raster {band} in {product}")


def _append_aligned_polarimetry(iq: Path, polar: Path, bands: list[str]) -> None:
    validate_same_dimensions(polar, iq)
    iq_crs = CRS.from_wkt(_read_crs_wkt(iq))
    polar_crs = CRS.from_wkt(_read_crs_wkt(polar))
    if not iq_crs.equals(polar_crs, ignore_axis_order=True) or not all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-7)
        for a, b in zip(_read_geotransform(iq), _read_geotransform(polar), strict=True)
    ):
        raise RuntimeError(f"Complex and polarimetric TC grids differ: {iq}, {polar}")
    _require_payload(polar, bands)
    if set(spectral_band_names(iq)) & set(bands):
        raise RuntimeError(f"Refusing to replace existing polarimetric bands in {iq}")
    for band in bands:
        for ext in ("hdr", "img"):
            if (iq.with_suffix(".data") / f"{band}.{ext}").exists():
                raise RuntimeError(f"Polarimetric payload already exists in {iq}: {band}.{ext}")
    copy_materialized_src_files(polar.with_suffix(".data"), iq.with_suffix(".data"), bands, overwrite=False)
    clone_source_raster_metadata(polar, iq, bands)
    _require_payload(iq, bands)


def run_split_terrain_correction(
    source: Path, outdir: Path, output_name: str, tc_params: dict, gpt_kwargs: dict,
    *, create_operator=create_gpt_operator,
) -> Path:
    """Run I/Q and scalar TC from the same radar DIM and join without resampling."""
    outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / f"{output_name}.dim"
    if final.exists() or final.with_suffix(".data").exists():
        raise FileExistsError(f"Refusing to overwrite an existing TC product: {final}")

    iq_params = dict(tc_params)
    polar_bands = [b for b in iq_params["source_bands"] if b.lower() in POLARIMETRIC_BANDS]
    iq_params["source_bands"] = [b for b in iq_params["source_bands"] if b not in polar_bands]
    if not iq_params["source_bands"] or not polar_bands:
        raise ValueError("Split TC requires both complex and polarimetric source bands")
    polar_params = polarimetric_tc_params(tc_params, polar_bands)

    with TemporaryDirectory(prefix=".split-tc-", dir=outdir) as temporary:
        stage = Path(temporary)

        def run(branch, params, budget):
            op = create_operator(source, stage / branch, "BEAM-DIMAP", **budget)
            result = op.TerrainCorrection(output_name=output_name, **params)
            if not result or not Path(result).is_file():
                raise RuntimeError(f"{branch} Terrain Correction failed: {op.last_error_summary()}")
            return Path(result)

        limits = _split_gpt_limits(gpt_kwargs)
        if limits is None:
            print("Terrain Correction: running I/Q and polarimetric passes sequentially")
            iq = run("complex", iq_params, gpt_kwargs)
            polar = run("polarimetry", polar_params, gpt_kwargs)
        else:
            print(f"Terrain Correction: parallel I/Q and polarimetric passes "
                  f"(threads {limits[0]['gpt_parallelism']}+{limits[1]['gpt_parallelism']})")
            with ThreadPoolExecutor(max_workers=2) as executor:
                iq_task = executor.submit(run, "complex", iq_params, limits[0])
                pol_task = executor.submit(run, "polarimetry", polar_params, limits[1])
                iq, polar = iq_task.result(), pol_task.result()

        _require_payload(iq, iq_params["source_bands"])
        _append_aligned_polarimetry(iq, polar, polar_bands)
        # Publish DIM last so a reader never sees a product with missing rasters.
        iq.with_suffix(".data").rename(final.with_suffix(".data"))
        iq.rename(final)
    return final
