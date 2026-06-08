"""Runtime support for declared WorldSAR tiling steps."""

from __future__ import annotations

from pathlib import Path

from sarpyx.hooks.worldsar import validate_worldsar_zarr_tile_group
from sarpyx.snapflow import config
from sarpyx.snapflow.report_manifest import write_tiling_manifest as _write_tiling_manifest
from sarpyx.snapflow.reports import (
    _run_db_indexing,
    _validate_tile_group,
    _write_h5_validation_report_pdf,
    create_merged_tile_database_from_groups,
    delete_swath_tile_databases,
)
from sarpyx.snapflow.tiling import _resolve_tiling_wkt, _run_tiling, _verify_tops_tile_coverage
from sarpyx.snapflow.tile_writers import normalize_tile_writer


def _tiling_metadata(ctx):
    metadata = ctx.metadata
    required = ("product_wkt", "grid_path", "cuts_outdir", "product_mode")
    missing = [name for name in required if name not in metadata]
    if missing:
        return None
    return {
        "product_wkt": metadata["product_wkt"],
        "grid_path": Path(metadata["grid_path"]),
        "cuts_outdir": Path(metadata["cuts_outdir"]),
        "product_mode": metadata["product_mode"],
        "gpt_kwargs": dict(metadata.get("gpt_kwargs", ctx.gpt_kwargs)),
        "terrain_correction_runs": dict(metadata.get("terrain_correction_runs", {})),
        "tile_writer": normalize_tile_writer(metadata.get("tile_writer", "zarr")),
        "pre_write_hook": metadata.get("pre_write_hook"),
        "report_outdir": Path(metadata["report_outdir"]) if metadata.get("report_outdir") else Path(metadata["cuts_outdir"]),
        "product_name": metadata.get("product_name"),
    }


def _generic_tile_group(tiling_result, intermediate_product, swath=None):
    results = [{"tile": tile, "status": "success"} for tile in tiling_result.get("actual_tiles", [])]
    return {
        "name": tiling_result["name"],
        "swath": swath,
        "cuts_dir": str(tiling_result["cuts_dir"]),
        "intermediate_product": str(intermediate_product),
        "expected_bands": [],
        "results": results,
        "rows": [],
        "tile_writer": tiling_result.get("tile_writer", "zarr"),
    }


def run_tiling_step(ctx, intermediate_product, swath: str | None = None, collect: bool = False):
    data = _tiling_metadata(ctx)
    if data is None:
        return {"skipped": True, "reason": "tiling metadata not available", "intermediate": intermediate_product}
    product_wkt = data["product_wkt"]
    product_mode = data["product_mode"]
    grid_path = data["grid_path"]
    source_product = Path(ctx.original_product)
    intermediate_product = Path(intermediate_product)
    product_name = data.get("product_name")
    collapse_swath_tiles = product_mode == "S1TOPS" and bool(swath)
    if product_name:
        product_cuts_dir = data["cuts_outdir"] / product_name
        cuts_outdir = product_cuts_dir if collapse_swath_tiles or not swath else product_cuts_dir / swath
        cut_report_outdir = data["report_outdir"] / product_name / swath if swath else data["report_outdir"]
        output_name = product_name
        direct_cuts_dir = True
    else:
        cuts_outdir = data["cuts_outdir"] if collapse_swath_tiles or not swath else data["cuts_outdir"] / swath
        cut_report_outdir = data["report_outdir"]
        output_name = None
        direct_cuts_dir = collapse_swath_tiles
    name = intermediate_product.stem
    tiling_wkt = _resolve_tiling_wkt(product_wkt, source_product, intermediate_product, product_mode, swath=swath)
    if config.tiling:
        terrain_correction = data.get("terrain_correction_runs", {}).get(str(intermediate_product))
        tiling_result = _run_tiling(
            tiling_wkt,
            grid_path,
            source_product,
            intermediate_product,
            cuts_outdir,
            product_mode,
            terrain_correction=terrain_correction,
            tile_writer=data["tile_writer"],
            pre_write_hook=data["pre_write_hook"],
            output_name=output_name,
            direct_cuts_dir=direct_cuts_dir,
            report_outdir=cut_report_outdir,
            **data["gpt_kwargs"],
        )
        tiling_result["pre_tc_wkt"] = product_wkt
        tiling_result["post_tc_wkt"] = tiling_wkt
        _write_tiling_manifest(tiling_result["report_path"], tiling_result)
        name = tiling_result["name"]
        if data["tile_writer"] == "h5":
            validation_group = _validate_tile_group(tiling_result["cuts_dir"], intermediate_product, swath=swath, tiling_result=tiling_result)
        elif data["tile_writer"] == "zarr":
            validation_group = validate_worldsar_zarr_tile_group(tiling_result["cuts_dir"], tiling_result, intermediate_product, swath=swath)
        else:
            validation_group = _generic_tile_group(tiling_result, intermediate_product, swath=swath)
    else:
        validation_cuts_dir = cuts_outdir if direct_cuts_dir else cuts_outdir / name
        validation_group = _validate_tile_group(
            validation_cuts_dir,
            intermediate_product,
            swath=swath,
            tiling_result={"pre_tc_wkt": product_wkt, "post_tc_wkt": tiling_wkt, "source_wkt": tiling_wkt, "report_source_wkt": tiling_wkt},
        )
        tiling_result = {"cut_failed": False, "name": name, "cuts_dir": validation_cuts_dir}
    if validation_group["rows"]:
        _run_db_indexing(validation_group["rows"], name, swath=swath, cuts_outdir=tiling_result["cuts_dir"])
    result = {"name": name, "swath": swath, "swath_wkt": tiling_wkt, "tiling_result": tiling_result, "validation_group": validation_group}
    if collect:
        if tiling_result.get("cut_failed"):
            result["error"] = RuntimeError(f"Tile cutting failed; report: {tiling_result['report_path']}")
        return result
    pdf_path = data["report_outdir"] / f"{product_name or name}_{data['tile_writer']}_validation_report.pdf"
    _write_h5_validation_report_pdf(pdf_path, product_name or name, [validation_group])
    if tiling_result.get("cut_failed"):
        raise RuntimeError(f"Tile cutting failed; report: {tiling_result['report_path']}")
    if any(item["status"] != "success" for item in validation_group["results"]):
        raise RuntimeError(f"{data['tile_writer'].upper()} validation failed; report: {pdf_path}")
    result["pdf_path"] = pdf_path
    return result


def finalize_tops_tiling(product_wkt, grid_path, cuts_outdir, swath_products, swath_results, report_outdir=None, product_name=None):
    validation_groups = [item["validation_group"] for item in swath_results]
    tile_writer = next((group.get("tile_writer") for group in validation_groups if group.get("tile_writer")), "zarr")
    if validation_groups:
        report_name = product_name or next((item["name"] for item in swath_results if item.get("name")), validation_groups[0]["name"])
        pdf_path = Path(report_outdir or cuts_outdir) / f"{report_name}_{tile_writer}_validation_report.pdf"
        _write_h5_validation_report_pdf(pdf_path, report_name, validation_groups)
        if config.db_indexing and any(group.get("rows") for group in validation_groups):
            db_dir = config.resolve_db_dir(cuts_outdir)
            create_merged_tile_database_from_groups(validation_groups, db_dir, report_name)
            swaths = [item.get("swath") for item in swath_results if item.get("swath")]
            delete_swath_tile_databases(db_dir, swaths, report_name)
    errors = {item["swath"]: item["error"] for item in swath_results if item.get("error")}
    if errors:
        swath_wkts = {item["swath"]: item["swath_wkt"] for item in swath_results if item.get("swath")}
        verify_kwargs = {"swath_wkts": swath_wkts}
        if tile_writer != "h5":
            verify_kwargs["tile_writer"] = tile_writer
        _verify_tops_tile_coverage(product_wkt, grid_path, cuts_outdir, swath_products, **verify_kwargs)
    if validation_groups and any(result["status"] != "success" for group in validation_groups for result in group["results"]):
        raise RuntimeError(f"{tile_writer.upper()} validation failed; report: {pdf_path}")
    return validation_groups
