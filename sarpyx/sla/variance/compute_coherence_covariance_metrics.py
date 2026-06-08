#!/usr/bin/env python3
"""
Compute redundancy and spatial-variability metrics for sub-aperture feature TIFs.

The script discovers coherence*.tif and covariance*.tif files under
--input-root, or processes explicit paths passed with --tif-paths. It extracts
product, swath, lookset, window, and polarization metadata from folder and file
names, then writes an Excel workbook with summary, patch-level metrics,
band-level variability, selected files, skipped files, and NMI matrix sheets.

Leave --filename-contains unset to process all matching feature TIFs under the
root. Use it only when a filename-level filter is needed, for example _IW2_.

Usage:
    python compute_coherence_covariance_metrics.py --input-root /path/to/worldsar_subap_config_features --filename-contains _IW2_ --output-xlsx coherence_covariance_metrics.xlsx
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used when tqdm is unavailable.
    tqdm = None


DEFAULT_PROCESS_KINDS = ("coherence", "covariance")


def classify_tif(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("coherence"):
        return "coherence"
    if name.startswith("covariance"):
        return "covariance"
    if name.startswith("phase_variance"):
        return "phase_variance"
    return "other"


def discover_tifs(
    input_root: Path | None,
    glob_pattern: str,
    tif_paths: Sequence[Path] | None,
    filename_contains: Sequence[str],
    filename_excludes: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []

    if input_root is not None:
        paths.extend(sorted(p for p in input_root.glob(glob_pattern) if p.is_file()))

    if tif_paths:
        paths.extend(tif_paths)

    unique_paths = sorted({p.expanduser().resolve() for p in paths})

    def keep(path: Path) -> bool:
        name = path.name
        if filename_contains and not all(text in name for text in filename_contains):
            return False
        if filename_excludes and any(text in name for text in filename_excludes):
            return False
        return path.suffix.lower() in {".tif", ".tiff"}

    return [path for path in unique_paths if keep(path)]


def build_paths_dataframe(
    tif_paths: Sequence[Path],
    process_kinds: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    skipped_records = []

    for path in tif_paths:
        kind = classify_tif(path)
        if kind not in process_kinds:
            skipped_records.append(
                {"path": str(path), "filename": path.name, "reason": f"kind_not_processed:{kind}"}
            )
            continue

        records.append({"path": str(path), "kind": kind, "filename": path.name})

    return pd.DataFrame(records), pd.DataFrame(skipped_records)


def iter_patch_windows(
    height: int,
    width: int,
    patch_size: int = 1024,
    drop_incomplete: bool = True,
) -> Iterable[tuple[Window, tuple[int, int, int, int]]]:
    for r0 in range(0, height, patch_size):
        for c0 in range(0, width, patch_size):
            r1 = min(r0 + patch_size, height)
            c1 = min(c0 + patch_size, width)
            if drop_incomplete and ((r1 - r0) < patch_size or (c1 - c0) < patch_size):
                continue
            yield Window(c0, r0, c1 - c0, r1 - r0), (r0, r1, c0, c1)


def get_band_names(src: rasterio.io.DatasetReader) -> list[str]:
    names = []
    for i, desc in enumerate(src.descriptions, start=1):
        names.append(f"band_{i}" if desc is None or desc == "" else desc)
    return names


def select_feature_bands(src: rasterio.io.DatasetReader, kind: str) -> tuple[list[int], list[str]]:
    band_names = get_band_names(src)

    if kind == "coherence":
        keep = [i + 1 for i, name in enumerate(band_names) if name.lower() != "gamma_mean"]
        if len(keep) == len(band_names) and len(band_names) > 1:
            if band_names[-1].lower().startswith("band_"):
                keep = list(range(1, src.count))
    elif kind == "covariance":
        keep = list(range(1, src.count + 1))
    else:
        raise ValueError("kind must be 'coherence' or 'covariance'")

    keep_names = [band_names[i - 1] for i in keep]
    return keep, keep_names


def parse_config_from_path(path: Path | str) -> dict[str, str | None]:
    path = Path(path)
    swath = lookset = win = product = None
    for part in path.parts:
        if re.fullmatch(r"IW\d+", part):
            swath = part
        elif re.fullmatch(r"L\d+", part):
            lookset = part
        elif re.fullmatch(r"W\d+", part):
            win = part
        elif part.endswith("_TC.data"):
            product = part

    pol = None
    match = re.search(r"_(VV|VH)\.tiff?$", path.name, flags=re.IGNORECASE)
    if match:
        pol = match.group(1).upper()

    return {"swath": swath, "lookset": lookset, "window": win, "product": product, "pol": pol}


def covariance_band_group(name: str) -> str:
    if re.fullmatch(r"C\d+\d+", name):
        return "cov_diag"
    if name.startswith("ReC") or name.startswith("ImC"):
        return "cov_cross"
    return "other"


def safe_nanmean(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def safe_nanmedian(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmedian(x))


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def fast_nmi_discrete(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    x = x.astype(np.int64, copy=False)
    y = y.astype(np.int64, copy=False)
    joint = np.bincount(x * bins + y, minlength=bins * bins).reshape(bins, bins)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    hx = entropy_from_counts(px)
    hy = entropy_from_counts(py)
    hxy = entropy_from_counts(joint.ravel())
    if hx == 0 or hy == 0 or np.isnan(hx) or np.isnan(hy) or np.isnan(hxy):
        return np.nan
    mi = hx + hy - hxy
    return float(2.0 * mi / (hx + hy))


def discretize_patch_stack(
    patch_stack: np.ndarray,
    bins: int = 64,
    pmin: float = 1,
    pmax: float = 99,
    mask_all_zero_as_nodata: bool = True,
    max_pixels: int | None = 200_000,
    seed: int = 42,
) -> np.ndarray | None:
    valid = np.all(np.isfinite(patch_stack), axis=0)
    if mask_all_zero_as_nodata:
        valid &= ~np.all(patch_stack == 0, axis=0)
    if valid.sum() == 0:
        return None

    vals = patch_stack[:, valid]
    if max_pixels is not None and vals.shape[1] > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(vals.shape[1], size=max_pixels, replace=False)
        vals = vals[:, idx]

    lo, hi = np.nanpercentile(vals, [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None

    edges = np.linspace(lo, hi, bins + 1)
    return np.digitize(np.clip(vals, lo, hi), edges[1:-1]).astype(np.uint8)


def nmi_matrix_from_discretized(disc: np.ndarray, bins: int = 64) -> np.ndarray:
    n_bands = disc.shape[0]
    mat = np.full((n_bands, n_bands), np.nan, dtype=np.float32)
    np.fill_diagonal(mat, 1.0)

    valid_band = np.array([np.unique(disc[i]).size >= 2 for i in range(n_bands)])

    for i in range(n_bands):
        if not valid_band[i]:
            continue
        xi = disc[i]
        for j in range(i + 1, n_bands):
            if not valid_band[j]:
                continue
            val = fast_nmi_discrete(xi, disc[j], bins=bins)
            mat[i, j] = val
            mat[j, i] = val
    return mat


def spatial_variability_patch(
    patch_stack: np.ndarray,
    band_names: Sequence[str],
    kind: str,
    mask_all_zero_as_nodata: bool = True,
    eps: float = 1e-8,
    min_abs_mean: float = 1e-8,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Coherence gamma_ij: std / abs(mean)
    Covariance Cii: std / abs(mean)

    Covariance cross terms ReCij/ImCij are intentionally not evaluated here:
    they were not used for ranking and tend to produce uninformative values.
    """
    n_bands = patch_stack.shape[0]
    values = np.full(n_bands, np.nan, dtype=np.float32)
    methods = []
    groups = []

    common_valid = np.all(np.isfinite(patch_stack), axis=0)
    if mask_all_zero_as_nodata:
        common_valid &= ~np.all(patch_stack == 0, axis=0)
    if common_valid.sum() == 0:
        return values, ["none"] * n_bands, ["none"] * n_bands

    for b in range(n_bands):
        name = band_names[b]
        x = patch_stack[b][common_valid].astype(np.float64)

        if kind == "coherence":
            group = "coherence"
        elif kind == "covariance":
            group = covariance_band_group(name)
        else:
            group = "other"
        groups.append(group)

        if group == "cov_cross":
            methods.append("not_computed")
            continue

        if x.size == 0:
            methods.append("none")
            continue

        sigma = np.nanstd(x)
        if not np.isfinite(sigma):
            methods.append("none")
            continue

        if group in {"coherence", "cov_diag"}:
            methods.append("cv_mean")
            mu = np.nanmean(x)
            if not np.isfinite(mu) or abs(mu) < min_abs_mean:
                continue
            values[b] = sigma / (abs(mu) + eps)
        else:
            methods.append("cv_mean")
            mu = np.nanmean(x)
            if not np.isfinite(mu) or abs(mu) < min_abs_mean:
                continue
            values[b] = sigma / (abs(mu) + eps)

    return values, methods, groups


def summarize_patch_variability(
    var_patch: np.ndarray,
    band_names: Sequence[str],
    kind: str,
) -> dict[str, float]:
    out = {
        "patch_mean_spatial_variability": safe_nanmean(var_patch),
        "patch_median_spatial_variability": safe_nanmedian(var_patch),
        "patch_coherence_mean_cv": np.nan,
        "patch_coherence_median_cv": np.nan,
        "patch_cov_diag_mean_cv": np.nan,
        "patch_cov_diag_median_cv": np.nan,
    }

    if kind == "coherence":
        out["patch_coherence_mean_cv"] = safe_nanmean(var_patch)
        out["patch_coherence_median_cv"] = safe_nanmedian(var_patch)
    elif kind == "covariance":
        groups = np.array([covariance_band_group(name) for name in band_names])
        diag_mask = groups == "cov_diag"
        if np.any(diag_mask):
            out["patch_cov_diag_mean_cv"] = safe_nanmean(var_patch[diag_mask])
            out["patch_cov_diag_median_cv"] = safe_nanmedian(var_patch[diag_mask])
    return out


def analyze_feature_tif(
    tif_path: Path | str,
    kind: str,
    patch_size: int = 1024,
    bins: int = 64,
    pmin: float = 1,
    pmax: float = 99,
    max_pixels: int | None = 200_000,
    mask_all_zero_as_nodata: bool = True,
    drop_incomplete: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    tif_path = Path(tif_path)
    patch_rows = []
    band_var_rows = []
    patch_nmi_mats = []
    patch_var_values = []
    patch_ids = []

    with rasterio.open(tif_path) as src:
        band_indices, band_names = select_feature_bands(src, kind=kind)
        height, width = src.height, src.width
        n_bands = len(band_indices)

        meta = parse_config_from_path(tif_path)
        meta.update(
            {
                "path": str(tif_path),
                "filename": tif_path.name,
                "kind": kind,
                "height": height,
                "width": width,
                "selected_bands": n_bands,
                "total_bands_in_tif": src.count,
            }
        )

        if verbose:
            print(f"\nAnalyzing {kind}: {tif_path}")
            print(f"selected bands: {n_bands}")
            print(f"first bands: {band_names[:10]}")

        for patch_index, (window, patch_id) in enumerate(
            iter_patch_windows(height, width, patch_size=patch_size, drop_incomplete=drop_incomplete)
        ):
            patch_stack = src.read(band_indices, window=window).astype(np.float32)

            disc = discretize_patch_stack(
                patch_stack,
                bins=bins,
                pmin=pmin,
                pmax=pmax,
                mask_all_zero_as_nodata=mask_all_zero_as_nodata,
                max_pixels=max_pixels,
                seed=patch_index,
            )
            if disc is None:
                continue

            nmi_mat = nmi_matrix_from_discretized(disc, bins=bins)
            var_patch, var_methods, var_groups = spatial_variability_patch(
                patch_stack,
                band_names=band_names,
                kind=kind,
                mask_all_zero_as_nodata=mask_all_zero_as_nodata,
            )

            upper = nmi_mat[np.triu_indices(n_bands, 1)]
            mean_nmi = float(np.nanmean(upper)) if upper.size > 0 else np.nan
            complementarity = float(1.0 - mean_nmi) if np.isfinite(mean_nmi) else np.nan
            var_summary = summarize_patch_variability(var_patch, band_names, kind)

            r0, r1, c0, c1 = patch_id
            patch_rows.append(
                {
                    **meta,
                    "patch_index": patch_index,
                    "row_start": r0,
                    "row_end": r1,
                    "col_start": c0,
                    "col_end": c1,
                    "valid_used_for_nmi": int(disc.shape[1]),
                    "patch_mean_nmi": mean_nmi,
                    "patch_1_minus_mean_nmi": complementarity,
                    **var_summary,
                }
            )

            for band_pos, band_name in enumerate(band_names):
                if var_methods[band_pos] == "not_computed":
                    continue

                band_var_rows.append(
                    {
                        **meta,
                        "patch_index": patch_index,
                        "band_position_selected": band_pos + 1,
                        "band_name": band_name,
                        "variability_group": var_groups[band_pos],
                        "variability_method": var_methods[band_pos],
                        "spatial_variability": (
                            float(var_patch[band_pos]) if np.isfinite(var_patch[band_pos]) else np.nan
                        ),
                    }
                )

            patch_nmi_mats.append(nmi_mat)
            patch_var_values.append(var_patch)
            patch_ids.append(patch_id)

            if verbose:
                if kind == "coherence":
                    var_msg = f"coh CV={var_summary['patch_coherence_median_cv']:.4f}"
                else:
                    var_msg = f"cov diag CV={var_summary['patch_cov_diag_median_cv']:.4f}"

                print(
                    f"patch {patch_index:04d} | "
                    f"valid_used={disc.shape[1]} | "
                    f"mean NMI={mean_nmi:.4f} | "
                    f"mean 1-NMI={complementarity:.4f} | "
                    f"{var_msg}"
                )

    if len(patch_rows) == 0:
        raise ValueError(f"No valid patches found for {tif_path}")

    patch_df = pd.DataFrame(patch_rows)
    band_var_df = pd.DataFrame(band_var_rows)
    patch_nmi_mats_arr = np.stack(patch_nmi_mats, axis=0)
    patch_var_values_arr = np.stack(patch_var_values, axis=0)

    mean_nmi_matrix = np.nanmean(patch_nmi_mats_arr, axis=0)
    median_nmi_matrix = np.nanmedian(patch_nmi_mats_arr, axis=0)
    upper = mean_nmi_matrix[np.triu_indices(mean_nmi_matrix.shape[0], 1)]

    if upper.size > 0:
        mean_redundancy_nmi = float(np.nanmean(upper))
        mean_complementarity_1_minus_nmi = float(1.0 - mean_redundancy_nmi)
    else:
        mean_redundancy_nmi = np.nan
        mean_complementarity_1_minus_nmi = np.nan

    summary = {
        **parse_config_from_path(tif_path),
        "path": str(tif_path),
        "filename": tif_path.name,
        "kind": kind,
        "height": int(patch_df["height"].iloc[0]),
        "width": int(patch_df["width"].iloc[0]),
        "selected_bands": int(patch_df["selected_bands"].iloc[0]),
        "total_bands_in_tif": int(patch_df["total_bands_in_tif"].iloc[0]),
        "n_valid_patches": int(len(patch_df)),
        "mean_nmi": mean_redundancy_nmi,
        "complementarity_1_minus_mean_nmi": mean_complementarity_1_minus_nmi,
        "mean_spatial_variability": safe_nanmean(patch_df["patch_mean_spatial_variability"]),
        "median_spatial_variability": safe_nanmedian(patch_df["patch_median_spatial_variability"]),
        "coherence_mean_cv": safe_nanmean(patch_df["patch_coherence_mean_cv"]),
        "coherence_median_cv": safe_nanmedian(patch_df["patch_coherence_median_cv"]),
        "covariance_diag_mean_cv": safe_nanmean(patch_df["patch_cov_diag_mean_cv"]),
        "covariance_diag_median_cv": safe_nanmedian(patch_df["patch_cov_diag_median_cv"]),
    }

    return {
        "summary": summary,
        "patch_df": patch_df,
        "band_var_df": band_var_df,
        "patch_nmi_mats": patch_nmi_mats_arr,
        "mean_nmi_matrix": mean_nmi_matrix,
        "median_nmi_matrix": median_nmi_matrix,
        "patch_var_values": patch_var_values_arr,
        "band_names": band_names,
        "patch_ids": patch_ids,
    }


def autosize_excel_columns(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame, max_width: int = 45) -> None:
    worksheet = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns):
        if df.empty:
            width = min(max(len(str(col)) + 2, 10), max_width)
        else:
            values = df[col].astype(str).replace("nan", "")
            width = min(max(values.map(len).max(), len(str(col))) + 2, max_width)
        worksheet.set_column(i, i, width)


def write_metrics_excel(
    output_xlsx: Path,
    summary_df: pd.DataFrame,
    patch_metrics_df: pd.DataFrame,
    band_variability_df: pd.DataFrame,
    nmi_matrix_long_df: pd.DataFrame,
    paths_df: pd.DataFrame,
    skipped_final_df: pd.DataFrame,
) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        sheets = {
            "summary": summary_df,
            "patch_metrics": patch_metrics_df,
            "band_variability": band_variability_df,
            "mean_nmi_matrix_long": nmi_matrix_long_df,
            "selected_files": paths_df,
            "skipped": skipped_final_df,
        }

        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            autosize_excel_columns(writer, sheet_name, df)

        for sheet_name, df in sheets.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            if not df.empty:
                ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))


def iter_rows_with_progress(df: pd.DataFrame, quiet: bool):
    rows = df.iterrows()
    if tqdm is None or quiet:
        return rows
    return tqdm(rows, total=len(df), desc="Processing TIFs")


def run(args: argparse.Namespace) -> None:
    process_kinds = set(args.process_kinds)
    tif_paths = discover_tifs(
        input_root=args.input_root,
        glob_pattern=args.glob_pattern,
        tif_paths=args.tif_paths,
        filename_contains=args.filename_contains,
        filename_excludes=args.filename_excludes,
    )
    paths_df, skipped_df = build_paths_dataframe(tif_paths, process_kinds)

    if args.limit is not None:
        paths_df = paths_df.head(args.limit).copy()

    print("Total discovered TIFs:", len(tif_paths))
    print("Selected files:", len(paths_df))
    print("Skipped files:", len(skipped_df))

    if paths_df.empty:
        raise SystemExit("No selected coherence/covariance TIFs found. Check input paths and filters.")

    summary_rows = []
    all_patch_metrics = []
    all_band_variability = []
    all_skipped = skipped_df.to_dict("records") if not skipped_df.empty else []
    matrix_records = []

    for _, row in iter_rows_with_progress(paths_df, quiet=args.quiet):
        tif_path = row["path"]
        kind = row["kind"]

        try:
            res = analyze_feature_tif(
                tif_path=tif_path,
                kind=kind,
                patch_size=args.patch_size,
                bins=args.bins,
                pmin=args.pmin,
                pmax=args.pmax,
                max_pixels=args.max_pixels_for_nmi,
                mask_all_zero_as_nodata=not args.keep_all_zero_pixels,
                drop_incomplete=not args.keep_incomplete_patches,
                verbose=not args.quiet,
            )

            summary_rows.append(res["summary"])
            all_patch_metrics.append(res["patch_df"])
            all_band_variability.append(res["band_var_df"])

            band_names = res["band_names"]
            mean_nmi_matrix = res["mean_nmi_matrix"]

            for i, bi in enumerate(band_names):
                for j, bj in enumerate(band_names):
                    matrix_records.append(
                        {
                            "path": tif_path,
                            "filename": Path(tif_path).name,
                            "kind": kind,
                            "band_i": bi,
                            "band_j": bj,
                            "mean_nmi": (
                                float(mean_nmi_matrix[i, j])
                                if np.isfinite(mean_nmi_matrix[i, j])
                                else np.nan
                            ),
                        }
                    )

        except Exception as exc:  # Keep batch processing resilient.
            print(f"ERROR processing {tif_path}: {exc}")
            all_skipped.append(
                {
                    "path": tif_path,
                    "filename": Path(tif_path).name,
                    "reason": f"processing_error:{type(exc).__name__}: {exc}",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    patch_metrics_df = pd.concat(all_patch_metrics, ignore_index=True) if all_patch_metrics else pd.DataFrame()
    band_variability_df = (
        pd.concat(all_band_variability, ignore_index=True) if all_band_variability else pd.DataFrame()
    )
    nmi_matrix_long_df = pd.DataFrame(matrix_records)
    skipped_final_df = pd.DataFrame(all_skipped)

    print("Processed files:", len(summary_df))
    print("Patch rows:", len(patch_metrics_df))
    print("Band variability rows:", len(band_variability_df))
    print("Skipped rows:", len(skipped_final_df))

    write_metrics_excel(
        output_xlsx=args.output_xlsx,
        summary_df=summary_df,
        patch_metrics_df=patch_metrics_df,
        band_variability_df=band_variability_df,
        nmi_matrix_long_df=nmi_matrix_long_df,
        paths_df=paths_df,
        skipped_final_df=skipped_final_df,
    )

    print(f"Excel written to: {args.output_xlsx.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NMI redundancy and spatial variability metrics from coherence/covariance TIFs."
    )
    input_group = parser.add_argument_group("inputs")
    input_group.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root directory used with --glob-pattern to discover TIF files.",
    )
    input_group.add_argument(
        "--glob-pattern",
        default="**/*.tif",
        help="Pattern relative to --input-root. Default: **/*.tif",
    )
    input_group.add_argument(
        "--tif-paths",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit TIF paths. Can be used with or without --input-root.",
    )
    input_group.add_argument(
        "--filename-contains",
        action="append",
        default=[],
        help="Keep only TIFs whose filename contains this text. Repeatable. Example: --filename-contains _IW2_",
    )
    input_group.add_argument(
        "--filename-excludes",
        action="append",
        default=[],
        help="Skip TIFs whose filename contains this text. Repeatable.",
    )
    input_group.add_argument(
        "--process-kinds",
        nargs="+",
        choices=("coherence", "covariance"),
        default=list(DEFAULT_PROCESS_KINDS),
        help="Feature families to process. Default: coherence covariance",
    )

    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path("coherence_covariance_metrics.xlsx"),
        help="Output Excel workbook path.",
    )
    parser.add_argument("--patch-size", type=int, default=1024, help="Patch size in pixels.")
    parser.add_argument("--bins", type=int, default=64, help="Histogram bins for NMI discretization.")
    parser.add_argument("--pmin", type=float, default=1, help="Lower percentile for discretization.")
    parser.add_argument("--pmax", type=float, default=99, help="Upper percentile for discretization.")
    parser.add_argument(
        "--max-pixels-for-nmi",
        type=int,
        default=200_000,
        help="Maximum pixels sampled per patch for NMI. Use 0 to disable subsampling.",
    )
    parser.add_argument(
        "--keep-incomplete-patches",
        action="store_true",
        help="Process edge patches smaller than --patch-size.",
    )
    parser.add_argument(
        "--keep-all-zero-pixels",
        action="store_true",
        help="Keep pixels where all selected bands are zero instead of treating them as nodata.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of selected TIFs to process.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")

    args = parser.parse_args()

    if args.input_root is None and not args.tif_paths:
        parser.error("Provide --input-root and/or --tif-paths.")
    if args.patch_size < 1:
        parser.error("--patch-size must be positive.")
    if args.bins < 2:
        parser.error("--bins must be at least 2.")
    if args.max_pixels_for_nmi == 0:
        args.max_pixels_for_nmi = None

    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
