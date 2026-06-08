"""WorldSAR tile cutting and TOPS swath tiling orchestration."""

from __future__ import annotations

from pathlib import Path

from sarpyx.hooks.worldsar import product_output_name
from sarpyx.utils.geos import grid_cell_utm_bbox
from sarpyx.utils.wkt_utils import sentinel1_swath_wkt_extractor_safe
from sarpyx.utils.worldsar_h5 import normalize_expected_tile_geometries
from sarpyx.snapflow import config
from sarpyx.snapflow.gpt import run_gpt_op
from sarpyx.snapflow.h5_quality import summarize_h5_raster_quality, summarize_zarr_raster_quality
from sarpyx.snapflow.nisar_tiles import write_nisar_bbox_tile
from sarpyx.snapflow.product import extract_product_id
from sarpyx.snapflow.raster import _dim_footprint_wkt, _pixel_region_is_within_bounds, _read_geotransform, _read_raster_size, _update_h5_corners, _utm_bbox_to_pixel_region, _write_tile_subsets_from_dim, _write_tile_subsets_from_dim_rectangles
from sarpyx.snapflow.report_manifest import write_tiling_manifest
from sarpyx.snapflow.reports import _write_cut_report
from sarpyx.snapflow.tile_writers import normalize_tile_writer, tile_glob_pattern, tile_output_path
from sarpyx.snapflow.tile_crs import group_rectangles_by_epsg, prepare_products_by_epsg
from sarpyx.snapflow.tile_selection import select_intersecting_grid_rectangles

tiling = config.tiling


def _validate_tile_result(tile_name, output_path, label, tile_writer="zarr"):
    tile_writer = "h5" if tile_writer == "zarr" and output_path.suffix.lower() == ".h5" else tile_writer
    if not output_path.exists():
        return {"tile": tile_name, "status": "failed", "reason": f"output missing after {label}", "output_path": str(output_path)}
    size = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file()) if output_path.is_dir() else output_path.stat().st_size
    if size == 0:
        return {"tile": tile_name, "status": "failed", "reason": f"output artifact is empty after {label}", "output_path": str(output_path)}
    if tile_writer == "zarr":
        try:
            quality = summarize_zarr_raster_quality(output_path)
        except Exception as exc:
            return {"tile": tile_name, "status": "failed", "reason": f"Zarr quality audit failed: {type(exc).__name__}: {exc}", "output_path": str(output_path)}
        if not quality["raster_data_ok"]:
            import shutil

            shutil.rmtree(output_path, ignore_errors=True)
            reason = quality.get("raster_quality_reason") or f"no usable raster pixels; nodata_fraction={quality['nodata_fraction']:.3f}"
            return {"tile": tile_name, "status": "partial", "reason": reason, "output_path": str(output_path), **quality}
        return {"tile": tile_name, "status": "success", "output_path": str(output_path), "size_bytes": size, "tile_writer": tile_writer, **quality}
    if tile_writer != "h5":
        return {"tile": tile_name, "status": "success", "output_path": str(output_path), "size_bytes": size, "tile_writer": tile_writer}
    try:
        quality = summarize_h5_raster_quality(output_path)
    except Exception as exc:
        return {"tile": tile_name, "status": "failed", "reason": f"H5 quality audit failed: {type(exc).__name__}: {exc}", "output_path": str(output_path)}
    if not quality["raster_data_ok"]:
        output_path.unlink(missing_ok=True)
        reason = quality.get("raster_quality_reason") or f"no usable raster pixels; nodata_fraction={quality['nodata_fraction']:.3f}"
        return {"tile": tile_name, "status": "partial", "reason": reason, "output_path": str(output_path), **quality}
    return {"tile": tile_name, "status": "success", "output_path": str(output_path), "size_bytes": size, **quality}


def to_geotiff(product_path, output_dir, geo_region=None, output_name=None, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, gpt_cache_size=None):
    if geo_region is None:
        raise ValueError("Geo region WKT string must be provided.")
    return run_gpt_op(product_path, output_dir, "GDAL-GTiff-WRITER", "Write", gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout, gpt_cache_size=gpt_cache_size)


def subset(product_path, output_dir, geo_region=None, region=None, output_name=None, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, gpt_cache_size=None):
    if geo_region is None and region is None:
        raise AssertionError('Either geo_region (WKT) or region (pixel coords "x,y,width,height") must be provided.')
    kwargs = {"copy_metadata": True, "output_name": output_name}
    if geo_region is not None:
        kwargs["geo_region"] = geo_region
    if region is not None:
        kwargs["region"] = region
    return run_gpt_op(product_path, output_dir, "HDF5", "Subset", gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout, gpt_cache_size=gpt_cache_size, **kwargs)


def swath_splitter(swath, product_path, output_dir, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, gpt_cache_size=None, **extra):
    return run_gpt_op(
        product_path,
        output_dir,
        "BEAM-DIMAP",
        "topsar_split",
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
        gpt_cache_size=gpt_cache_size,
        subswath=f"IW{swath}",
        **extra,
    )


def _cut_single_tile(rect, product_path, cuts_dir, product_mode, gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size=None, dimap_precut=False, tile_writer="zarr", pre_write_hook=None):
    tile_name = rect["BL"]["properties"]["name"]
    tile_writer = normalize_tile_writer(tile_writer)
    tile_path = tile_output_path(cuts_dir, tile_name, tile_writer)
    try:
        if product_mode == "NISAR":
            epsg = int(rect["BL"]["properties"]["epsg"].split(":")[1])
            tile_path = write_nisar_bbox_tile(product_path, tile_path, tile_name, grid_cell_utm_bbox(rect, epsg), tile_writer, pre_write_hook)
        else:
            epsg = int(rect["BL"]["properties"]["epsg"].split(":")[1])
            utm_bbox = grid_cell_utm_bbox(rect, epsg)
            region = _utm_bbox_to_pixel_region(utm_bbox, _read_geotransform(product_path))
            if not _pixel_region_is_within_bounds(region, _read_raster_size(product_path)):
                raise ValueError(f"Pixel region {region} is outside raster bounds {_read_raster_size(product_path)[0]}x{_read_raster_size(product_path)[1]}.")
            direct_error = None
            if Path(product_path).suffix.lower() == ".dim":
                try:
                    if not dimap_precut:
                        _write_tile_subsets_from_dim(product_path, [(region, tile_path, tile_name)], tile_writer=tile_writer, pre_write_hook=pre_write_hook)
                except Exception as exc:
                    direct_error = exc
            if Path(product_path).suffix.lower() != ".dim" or direct_error is not None:
                if tile_writer != "h5":
                    raise RuntimeError(f"Direct DIMAP {tile_writer} tile cut failed for {tile_name}: {direct_error}")
                if direct_error is not None:
                    print(f"Direct DIMAP H5 tile cut failed for {tile_name}: {direct_error}. Falling back to SNAP HDF5 subset.")
                tile_path = Path(
                    subset(
                        product_path,
                        cuts_dir,
                        output_name=tile_name,
                        region=region,
                        gpt_memory=gpt_memory,
                        gpt_parallelism=gpt_parallelism,
                        gpt_timeout=gpt_timeout,
                        gpt_cache_size=gpt_cache_size,
                    )
                )
            if tile_writer == "h5":
                _update_h5_corners(tile_path, utm_bbox, epsg)
        return _validate_tile_result(tile_name, tile_path, "tile cut", tile_writer=tile_writer)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        normalized = reason.lower()
        if (
            "does not intersect with product bounds" in normalized
            or ("pixel region" in normalized and "invalid" in normalized)
            or "outside raster bounds" in normalized
        ):
            return {"tile": tile_name, "status": "skipped", "reason": reason, "output_path": str(tile_path)}
        return {"tile": tile_name, "status": "failed", "reason": reason, "output_path": str(tile_path)}


def _actual_tiles_from_run_results(results, cuts_dir, tile_writer):
    actual_tiles = set()
    for result in results:
        if result.get("status") in {"skipped", "partial"}:
            continue
        output_path = Path(result.get("output_path") or tile_output_path(cuts_dir, result["tile"], tile_writer))
        if output_path.exists():
            actual_tiles.add(str(result["tile"]))
    return sorted(actual_tiles)


def _run_tiling(
    product_wkt,
    grid_geoj_path,
    source_product,
    intermediate_product,
    cuts_outdir,
    product_mode,
    gpt_memory,
    gpt_parallelism,
    gpt_timeout,
    gpt_cache_size=None,
    terrain_correction=None,
    tile_writer="zarr",
    pre_write_hook=None,
    output_name=None,
    direct_cuts_dir=False,
    report_outdir=None,
):
    if grid_geoj_path is None or not Path(grid_geoj_path).exists():
        raise FileNotFoundError(f"grid_10km.geojson does not exist: {grid_geoj_path}")
    rectangles = select_intersecting_grid_rectangles(product_wkt, grid_geoj_path)
    if not rectangles:
        raise ValueError("No rectangles formed; check WKT coverage and grid alignment.")
    name = output_name or (extract_product_id(intermediate_product.as_posix()) if product_mode != "NISAR" else intermediate_product.stem)
    if name is None:
        raise ValueError(f"Could not extract product id from: {intermediate_product}")
    cuts_dir = Path(cuts_outdir) if direct_cuts_dir else cuts_outdir / name
    cuts_dir.mkdir(parents=True, exist_ok=True)
    tile_writer = normalize_tile_writer(tile_writer)
    gpt_kwargs = {"gpt_memory": gpt_memory, "gpt_parallelism": gpt_parallelism, "gpt_timeout": gpt_timeout, "gpt_cache_size": gpt_cache_size}
    if product_mode == "NISAR" or intermediate_product.suffix.lower() != ".dim":
        rectangle_groups = {0: rectangles}
        products_by_epsg = {0: intermediate_product}
    else:
        rectangle_groups = group_rectangles_by_epsg(rectangles)
        products_by_epsg = prepare_products_by_epsg(intermediate_product, list(rectangle_groups), gpt_kwargs, terrain_correction=terrain_correction)
    results = []
    for epsg, group_rectangles in rectangle_groups.items():
        group_product = products_by_epsg[epsg]
        dimap_precut = False
        if product_mode != "NISAR" and Path(group_product).suffix.lower() == ".dim":
            try:
                _write_tile_subsets_from_dim_rectangles(group_product, group_rectangles, cuts_dir, tile_writer=tile_writer, pre_write_hook=pre_write_hook)
                dimap_precut = True
            except Exception as exc:
                print(f"Direct DIMAP {tile_writer} batch cut failed for EPSG:{epsg}: {exc}. Falling back to per-tile cuts.")
        for rect in group_rectangles:
            result = _cut_single_tile(rect, group_product, cuts_dir, product_mode, gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size, dimap_precut, tile_writer, pre_write_hook)
            result["epsg"] = epsg
            result["intermediate_product"] = str(group_product)
            results.append(result)
    candidate_tiles = sorted({rect["BL"]["properties"]["name"] for rect in rectangles})
    skipped_tiles = sorted({r["tile"] for r in results if r.get("status") == "skipped"})
    partial_tiles = sorted({r["tile"] for r in results if r.get("status") == "partial"})
    failed_tiles = sorted({r["tile"] for r in results if r.get("status") == "failed"})
    actual_tiles = _actual_tiles_from_run_results(results, cuts_dir, tile_writer)
    expected_tiles = sorted(set(candidate_tiles) - set(skipped_tiles) - set(partial_tiles))
    missing_tiles = sorted(set(expected_tiles) - set(skipped_tiles) - set(actual_tiles))
    extra_tiles = sorted(set(actual_tiles) - set(expected_tiles))
    candidate_tile_geometries = normalize_expected_tile_geometries(rectangles)
    expected_tile_geometries = {tile: candidate_tile_geometries[tile] for tile in expected_tiles if tile in candidate_tile_geometries}
    crs_groups = {f"EPSG:{epsg}": len(group_rectangles) for epsg, group_rectangles in rectangle_groups.items() if epsg}
    report_path = _write_cut_report(
        Path(report_outdir) if report_outdir is not None else cuts_dir,
        name,
        source_product,
        intermediate_product,
        product_wkt,
        expected_tiles,
        actual_tiles,
        results,
        missing_tiles,
        extra_tiles,
        candidate_tiles=candidate_tiles,
        partial_tiles=partial_tiles,
        crs_groups=crs_groups,
        intermediate_products={f"EPSG:{epsg}": path for epsg, path in products_by_epsg.items() if epsg},
    )
    for result in results:
        status = result.get("status")
        if status == "success":
            print(f"Tile saved: {result.get('output_path', '')}")
        elif status == "skipped":
            print(f"Skipped tile {result.get('tile', 'UNKNOWN')}: {result.get('reason', '?')}")
        elif status == "partial":
            print(f"Partial tile {result.get('tile', 'UNKNOWN')}: {result.get('reason', '?')}")
        else:
            print(f"Failed tile {result.get('tile', 'UNKNOWN')}: {result.get('reason', '?')}")
    tiling_result = {
        "name": name,
        "cuts_dir": cuts_dir,
        "report_path": report_path,
        "grid_path": grid_geoj_path,
        "cut_failed": bool(missing_tiles or failed_tiles),
        "tile_writer": tile_writer,
        "source_wkt": product_wkt,
        "candidate_tile_geometries": candidate_tile_geometries,
        "expected_tile_geometries": expected_tile_geometries,
        "candidate_tiles": candidate_tiles,
        "expected_tiles": expected_tiles,
        "actual_tiles": actual_tiles,
        "skipped_tiles": skipped_tiles,
        "partial_tiles": partial_tiles,
        "failed_tiles": failed_tiles,
        "missing_tiles": missing_tiles,
        "extra_tiles": extra_tiles,
        "crs_groups": crs_groups,
        "skipped_tile_reasons": {r["tile"]: r.get("reason", "") for r in results if r.get("status") == "skipped"},
        "partial_tile_reasons": {r["tile"]: r.get("reason", "") for r in results if r.get("status") == "partial"},
    }
    write_tiling_manifest(report_path, tiling_result)
    return tiling_result


def _resolve_tiling_wkt(product_wkt, source_product, intermediate_product, product_mode, swath=None):
    intermediate_path = Path(intermediate_product)
    if intermediate_path.suffix.lower() == ".dim":
        try:
            return _dim_footprint_wkt(intermediate_path)
        except Exception as exc:
            print(f"[WARN] Failed to derive raster footprint from {intermediate_path}: {type(exc).__name__}: {exc}")
    if product_mode == "S1TOPS" and swath:
        derived_wkt = sentinel1_swath_wkt_extractor_safe(source_product, swath, display_results=False, verbose=False)
        if derived_wkt:
            return derived_wkt
    return product_wkt


def _run_tops_swath_tiling(product_wkt, grid_geoj_path, product_path, intermediate, cuts_outdir, product_mode, gpt_kwargs, tile_writer="zarr", pre_write_hook=None):
    from sarpyx.snapflow.runtime import PipelineContext, PipelineStep, run_step
    from sarpyx.snapflow.tiling_runtime import finalize_tops_tiling

    product_name = product_output_name(product_path)
    metadata = {
        "product_wkt": product_wkt,
        "grid_path": grid_geoj_path,
        "cuts_outdir": cuts_outdir,
        "product_mode": product_mode,
        "gpt_kwargs": dict(gpt_kwargs),
        "tile_writer": tile_writer,
        "pre_write_hook": pre_write_hook,
        "product_name": product_name,
    }
    swath_results = []
    for swath, swath_product in intermediate.items():
        ctx = PipelineContext(
            product_path,
            swath_product,
            Path(swath_product).parent,
            None,
            {"tc": swath_product},
            {**metadata, "swath": swath},
            dict(gpt_kwargs),
        )
        swath_results.append(run_step(ctx, PipelineStep("WorldSARTiling", {"intermediate_ref": "tc", "collect": True}, "tiling")))
    return finalize_tops_tiling(product_wkt, grid_geoj_path, cuts_outdir, intermediate, swath_results, product_name=product_name)


def _verify_tops_tile_coverage(product_wkt, grid_geoj_path, cuts_outdir, swath_products, swath_wkts=None, tile_writer="zarr"):
    rectangles = select_intersecting_grid_rectangles(product_wkt, grid_geoj_path)
    if not rectangles:
        return
    expected_tiles = {rect["BL"]["properties"]["name"] for rect in rectangles}
    produced_tiles = {tile_file.stem for tile_file in Path(cuts_outdir).rglob(tile_glob_pattern(tile_writer))}
    missing = sorted(expected_tiles - produced_tiles)
    print("\n[TOPS Aggregate Coverage]")
    print(f"  Expected tiles (from full product WKT): {len(expected_tiles)}")
    if swath_wkts:
        swath_expected_tiles = set()
        for swath_wkt in swath_wkts.values():
            swath_rectangles = select_intersecting_grid_rectangles(swath_wkt, grid_geoj_path)
            swath_expected_tiles.update(rect["BL"]["properties"]["name"] for rect in swath_rectangles)
        print(f"  Expected tiles (union of swath WKTs):  {len(swath_expected_tiles)}")
    print(f"  Produced tiles (across all swaths):     {len(expected_tiles - set(missing))}")
    print(f"  Missing tiles:                          {len(missing)}")
    if missing:
        print(f"  Missing tile names: {missing}")
    if not produced_tiles:
        raise RuntimeError("TOPS tiling produced zero tiles across all swaths.")
