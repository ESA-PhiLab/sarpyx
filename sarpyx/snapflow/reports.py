"""WorldSAR tile validation reports and metadata indexing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sarpyx.utils.io import read_h5
from sarpyx.utils.meta import normalize_sar_timestamp
from sarpyx.utils.worldsar_h5 import (
    enrich_validation_results_with_h5_structure,
    resolve_expected_band_names_from_dim_product,
    validate_h5_tile,
    write_h5_validation_report_pdf,
)
from sarpyx.snapflow import config
from sarpyx.snapflow.h5_quality import summarize_h5_raster_quality
from sarpyx.snapflow.report_manifest import hydrate_tiling_result, write_tiling_manifest

_hydrate_tiling_result = hydrate_tiling_result
_write_tiling_manifest = write_tiling_manifest


def _write_cut_report(
    report_dir,
    product_name,
    product_path,
    intermediate_product,
    product_wkt,
    expected_tiles,
    actual_tiles,
    results,
    missing_tiles,
    extra_tiles,
    candidate_tiles=None,
    partial_tiles=None,
    crs_groups=None,
    intermediate_products=None,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    failed = [result for result in results if result.get("status") == "failed"]
    skipped = [result for result in results if result.get("status") == "skipped"]
    partial = [result for result in results if result.get("status") == "partial"]
    outside = [result for result in skipped if "outside raster bounds" in str(result.get("reason", ""))]
    other_skipped = [result for result in skipped if result not in outside]
    candidate_tiles = sorted(candidate_tiles or expected_tiles)
    partial_tiles = sorted(partial_tiles or [result.get("tile") for result in partial])
    status = "SUCCESS" if not failed and not missing_tiles else "FAILURE"
    lines = [
        "WorldSAR tile cutting report",
        f"Timestamp (UTC): {timestamp}",
        f"Product name: {product_name}",
        f"Product path: {product_path}",
        f"Intermediate product: {intermediate_product}",
        f"Cuts output dir: {report_dir}",
        f"Product WKT: {product_wkt}",
        "Tile selection policy: grid-cell rectangles intersecting the processed raster footprint",
        "Expected tile policy: expected tiles are full-data deliverables after excluding edge/intersection partials",
        "Partial tile policy: partial edge tiles are reported separately and are not counted as missing deliverables",
        "Full-data tile policy: successful H5 tiles must contain 100% usable raster pixels",
        "",
        f"Candidate intersecting tiles: {len(candidate_tiles)}",
        f"Expected full-data tiles: {len(expected_tiles)}",
        f"Actual tiles on disk: {len(actual_tiles)}",
        f"Successful tiles (this run): {len([r for r in results if r.get('status') == 'success'])}",
        f"Partial edge tiles (not deliverables): {len(partial_tiles)}",
        f"Skipped tiles (outside raster bounds): {len(outside)}",
        f"Skipped tiles (other): {len(other_skipped)}",
        f"Failed tiles (this run): {len(failed)}",
        f"Missing tiles: {len(missing_tiles)}",
        f"Unexpected tiles: {len(extra_tiles)}",
    ]
    if crs_groups:
        lines.extend(["", "CRS groups:"])
        lines.extend(f"- {epsg}: {count} candidate intersecting tiles" for epsg, count in sorted(crs_groups.items()))
    if intermediate_products:
        lines.extend(["", "Intermediate products by CRS:"])
        lines.extend(f"- {epsg}: {path}" for epsg, path in sorted(intermediate_products.items()))
    for label, rows in (("Partial edge tiles", partial), ("Skipped tiles", skipped), ("Failed tiles", failed)):
        if rows:
            lines.extend(["", f"{label}:"])
            lines.extend(f"- {r.get('tile', 'UNKNOWN')}: {r.get('reason', '?')} | {r.get('output_path', '')}" for r in rows)
    if missing_tiles:
        lines.extend(["", "Missing tiles (expected but not found on disk):", *[f"- {tile}" for tile in missing_tiles]])
    if extra_tiles:
        lines.extend(["", "Unexpected tiles (found on disk but not expected):", *[f"- {tile}" for tile in extra_tiles]])
    report_path = report_dir / f"{product_name}_cuts_report_{status}.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _resolve_expected_tile_bands(intermediate_product) -> list[str]:
    intermediate_product = Path(intermediate_product)
    if intermediate_product.suffix.lower() == ".h5":
        from sarpyx.snapflow.nisar_tiles import select_nisar_tile_polarizations

        return select_nisar_tile_polarizations(intermediate_product)
    return resolve_expected_band_names_from_dim_product(intermediate_product)


def _validate_tile_group(cuts_dir, intermediate_product, swath=None, tiling_result=None):
    tiling_result = hydrate_tiling_result(tiling_result)
    cuts_dir = Path(cuts_dir)
    expected_bands = _resolve_expected_tile_bands(intermediate_product)
    tile_files = sorted(cuts_dir.glob("*.h5"))
    if not tile_files:
        raise FileNotFoundError(f"No H5 tiles found in {cuts_dir}{f' for swath {swath}' if swath else ''}.")
    results = [validate_h5_tile(tile_file, expected_bands, swath=swath) for tile_file in tile_files]
    group = {
        "name": cuts_dir.name,
        "swath": swath,
        "cuts_dir": str(cuts_dir),
        "intermediate_product": str(intermediate_product),
        "expected_bands": expected_bands,
        "results": results,
        "rows": [result["quickinfo_row"] for result in results],
    }
    group.update(enrich_validation_results_with_h5_structure(results))
    for tile_file, result in zip(tile_files, results, strict=True):
        quality = summarize_h5_raster_quality(tile_file)
        result.update(quality)
        if not quality["raster_data_ok"]:
            result["status"] = "failed"
    if tiling_result is not None:
        expected_tiles = sorted(tiling_result.get("expected_tiles", tiling_result.get("expected_tile_geometries", {}).keys()))
        group.update(
            {
                "candidate_tiles": sorted(tiling_result.get("candidate_tiles", tiling_result.get("candidate_tile_geometries", {}).keys())),
                "expected_tiles": expected_tiles,
                "actual_tiles": sorted(tiling_result.get("actual_tiles", [result["tile"] for result in results])),
                "candidate_tile_count": tiling_result.get("candidate_tile_count"),
                "expected_tile_count": tiling_result.get("expected_tile_count"),
                "actual_tile_count": tiling_result.get("actual_tile_count"),
                "partial_tile_count": tiling_result.get("partial_tile_count"),
                "skipped_tile_count": tiling_result.get("skipped_tile_count"),
                "failed_tile_count": tiling_result.get("failed_tile_count"),
                "missing_tile_count": tiling_result.get("missing_tile_count"),
                "extra_tile_count": tiling_result.get("extra_tile_count"),
                "missing_tiles": sorted(tiling_result.get("missing_tiles", [])),
                "extra_tiles": sorted(tiling_result.get("extra_tiles", [])),
                "skipped_tiles": sorted(tiling_result.get("skipped_tiles", [])),
                "partial_tiles": sorted(tiling_result.get("partial_tiles", [])),
                "skipped_tile_reasons": dict(tiling_result.get("skipped_tile_reasons", {})),
                "partial_tile_reasons": dict(tiling_result.get("partial_tile_reasons", {})),
                "failed_tiles": sorted(set(tiling_result.get("failed_tiles", [])) | {r["tile"] for r in results if r["status"] != "success"}),
                "pre_tc_wkt": tiling_result.get("pre_tc_wkt"),
                "post_tc_wkt": tiling_result.get("post_tc_wkt") or tiling_result.get("report_source_wkt") or tiling_result.get("source_wkt"),
                "source_wkt": tiling_result.get("source_wkt"),
                "report_source_wkt": tiling_result.get("report_source_wkt") or tiling_result.get("source_wkt"),
                "candidate_tile_geometries": tiling_result.get("candidate_tile_geometries", {}),
                "expected_tile_geometries": tiling_result.get("expected_tile_geometries", {}),
                "cut_failed": tiling_result.get("cut_failed", False),
                "cut_report_path": str(tiling_result["report_path"]) if tiling_result.get("report_path") else None,
            }
        )
    return group


def _write_h5_validation_report_pdf(report_path, product_name, validation_groups):
    return write_h5_validation_report_pdf(report_path, product_name, validation_groups)


def create_tile_database_from_rows(rows, output_db_folder, output_name):
    if not rows:
        raise ValueError("No validated tile metadata rows available.")
    db = pd.DataFrame(rows)
    out = Path(output_db_folder)
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / f"{output_name}_core_metadata.parquet"
    db.to_parquet(output_file, index=False)
    print(f"Core metadata saved to {output_file}")
    return db


def create_merged_tile_database_from_groups(validation_groups, output_db_folder, output_name):
    rows = []
    for group in validation_groups:
        rows.extend(group.get("rows") or [])
    if not rows:
        raise ValueError("No validated tile metadata rows available.")
    db = pd.DataFrame(rows)
    out = Path(output_db_folder)
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / f"{output_name}.parquet"
    db.to_parquet(output_file, index=False)
    print(f"Merged core metadata saved to {output_file}")
    return db


def delete_swath_tile_databases(output_db_folder, swaths, output_name):
    out = Path(output_db_folder)
    for swath in swaths:
        if not swath:
            continue
        (out / f"{swath}_{output_name}_core_metadata.parquet").unlink(missing_ok=True)


def create_tile_database(input_folder, output_db_folder):
    tile_path = Path(input_folder)
    h5_tiles = list(tile_path.rglob("*.h5"))
    if not h5_tiles:
        raise FileNotFoundError(f"No .h5 tiles found in {tile_path}")
    db = pd.DataFrame()
    for tile_file in h5_tiles:
        _data, metadata = read_h5(tile_file)
        row = pd.Series(metadata["quickinfo"])
        row["first_line_time"] = normalize_sar_timestamp(row.get("first_line_time"))
        row["ID"] = tile_file.stem
        db = pd.concat([db, pd.DataFrame([row])], ignore_index=True)
    out = Path(output_db_folder)
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / f"{tile_path.name}_core_metadata.parquet"
    db.to_parquet(output_file, index=False)
    print(f"Core metadata saved to {output_file}")
    return db


def _run_db_indexing(validation_rows, name, swath=None, cuts_outdir=None):
    if not config.db_indexing:
        return
    db_dir = config.resolve_db_dir(cuts_outdir)
    if isinstance(validation_rows, (str, Path)):
        db = create_tile_database((Path(validation_rows) / name).as_posix(), db_dir)
    else:
        db = create_tile_database_from_rows(validation_rows, db_dir, f"{swath}_{name}" if swath else name)
    if db.empty:
        raise RuntimeError("Database creation failed, resulting DataFrame is empty.")
    print("Database created successfully.")
