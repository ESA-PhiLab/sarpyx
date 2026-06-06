#!/usr/bin/env python3
"""
Compute azimuth-subaperture features from WorldSAR TC products.

Input can be one *_TC.data product directory or a root directory containing
multiple products. For each product, the script discovers valid I/Q
sub-aperture pairs, polarizations, optional IW swath prefixes, and optional
look-set prefixes.

Outputs are GeoTIFF feature stacks written under --output-root while preserving
the relative product folder structure:
  - coherence_<prefix>_<pol>.tif
  - covariance_<prefix>_<pol>.tif
  - phase_variance_<prefix>_<pol>.tif

Usage:
    python compute_subap_features.py --input-root /path/to/tc_products --output-root /path/to/feature_tifs --win-size 5
"""

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter


IQ_RE = re.compile(
    r"^(?:(?P<lookset>L\d+)_)?(?P<part>[iq])_(?:(?P<prefix>IW\d+)_)?(?P<pol>VV|VH)(?:_SA(?P<sa>\d+))?\.img$",
    re.IGNORECASE,
)


class LookSet(NamedTuple):
    prefix: str
    pol: str
    subaps: Tuple[int, ...]
    paths: Dict[int, Dict[str, Path]]


def local_mean(arr: np.ndarray, size: int) -> np.ndarray:
    if np.iscomplexobj(arr):
        real = uniform_filter(arr.real, size=size, mode="nearest")
        imag = uniform_filter(arr.imag, size=size, mode="nearest")
        return real + 1j * imag
    return uniform_filter(arr, size=size, mode="nearest")


def coherence(si: np.ndarray, sj: np.ndarray, win: int, eps: float = 1e-8) -> np.ndarray:
    num = np.abs(local_mean(si * np.conj(sj), size=win))
    den = np.sqrt(
        local_mean(np.abs(si) ** 2, size=win) *
        local_mean(np.abs(sj) ** 2, size=win)
    )
    return num / (den + eps)


def phase_variance(stack: np.ndarray) -> np.ndarray:
    phases = np.angle(stack)
    coh_phase = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return 1.0 - coh_phase


def covariance_terms(stack: Sequence[np.ndarray], win: int) -> Dict[str, np.ndarray]:
    cov: Dict[str, np.ndarray] = {}

    for idx, arr in enumerate(stack, start=1):
        cov[f"C{idx}{idx}"] = local_mean(arr * np.conj(arr), size=win).real.astype(np.float32)

    for i, j in itertools.combinations(range(len(stack)), 2):
        cij = local_mean(stack[i] * np.conj(stack[j]), size=win)
        cov[f"ReC{i + 1}{j + 1}"] = cij.real.astype(np.float32)
        cov[f"ImC{i + 1}{j + 1}"] = cij.imag.astype(np.float32)

    return cov


def build_output_profile(src_profile: dict, count: int) -> dict:
    profile = src_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=count,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    return profile


def clamp_window(
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    max_width: int,
    max_height: int,
) -> Window:
    col_off = max(0, col_off)
    row_off = max(0, row_off)
    width = min(width, max_width - col_off)
    height = min(height, max_height - row_off)
    return Window(col_off, row_off, width, height)


def read_complex_window(
    src_i: rasterio.io.DatasetReader,
    src_q: rasterio.io.DatasetReader,
    window: Window,
) -> np.ndarray:
    i = src_i.read(1, window=window).astype(np.float32)
    q = src_q.read(1, window=window).astype(np.float32)
    return i + 1j * q


def crop_center(arr: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    return arr[top:top + height, left:left + width]


def make_suffix(prefix: str, pol: str) -> str:
    return f"{prefix}_{pol}" if prefix else pol


def discover_look_sets(product_dir: Path) -> List[LookSet]:
    grouped: Dict[Tuple[str, str], Dict[int, Dict[str, Path]]] = {}

    for img_path in sorted(product_dir.glob("*.img")):
        match = IQ_RE.match(img_path.name)
        if not match:
            continue

        sa = match.group("sa")
        if sa is None:
            continue

        lookset = (match.group("lookset") or "").upper()
        prefix = (match.group("prefix") or "").upper()
        prefix = "_".join(part for part in (lookset, prefix) if part)
        pol = match.group("pol").upper()
        part = match.group("part").lower()
        sa_idx = int(sa)

        grouped.setdefault((prefix, pol), {}).setdefault(sa_idx, {})[part] = img_path

    look_sets: List[LookSet] = []

    for (prefix, pol), sa_map in sorted(grouped.items()):
        valid_subaps = tuple(sorted(sa for sa, iq in sa_map.items() if "i" in iq and "q" in iq))
        if len(valid_subaps) < 2:
            continue
        look_sets.append(
            LookSet(
                prefix=prefix,
                pol=pol,
                subaps=valid_subaps,
                paths={sa: sa_map[sa] for sa in valid_subaps},
            )
        )

    return look_sets


def process_product(
    product_dir: Path,
    out_root: Path,
    win_size: int,
    verbose: bool = True,
    rel_parent: Optional[Path] = None,
) -> None:
    product_out_dir = out_root / (rel_parent or Path(".")) / product_dir.name
    product_out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nProcessing product: {product_dir}")

    look_sets = discover_look_sets(product_dir)
    if not look_sets:
        if verbose:
            print("  No valid sub-aperture I/Q groups found. Skipping.")
        return

    for look_set in look_sets:
        suffix = make_suffix(look_set.prefix, look_set.pol)

        if verbose:
            print(f"  [{suffix}] Found subaps: {', '.join(f'SA{sa}' for sa in look_set.subaps)}")

        first_sa = look_set.subaps[0]
        with rasterio.open(look_set.paths[first_sa]["i"]) as src_ref:
            height = src_ref.height
            width = src_ref.width

            coh_pairs = list(itertools.combinations(look_set.subaps, 2))
            coh_names = [f"gamma{a}{b}" for a, b in coh_pairs] + ["gamma_mean"]
            cov_names = [f"C{sa}{sa}" for sa in look_set.subaps]
            for a, b in coh_pairs:
                cov_names.extend([f"ReC{a}{b}", f"ImC{a}{b}"])

            coh_out = product_out_dir / f"coherence_{suffix}.tif"
            cov_out = product_out_dir / f"covariance_{suffix}.tif"
            phs_out = product_out_dir / f"phase_variance_{suffix}.tif"

            coh_profile = build_output_profile(src_ref.profile, count=len(coh_names))
            cov_profile = build_output_profile(src_ref.profile, count=len(cov_names))
            phs_profile = build_output_profile(src_ref.profile, count=1)

            halo = win_size // 2

            readers = []
            try:
                for sa in look_set.subaps:
                    readers.append(
                        (
                            sa,
                            rasterio.open(look_set.paths[sa]["i"]),
                            rasterio.open(look_set.paths[sa]["q"]),
                        )
                    )

                with rasterio.open(coh_out, "w", **coh_profile) as dst_coh, \
                     rasterio.open(cov_out, "w", **cov_profile) as dst_cov, \
                     rasterio.open(phs_out, "w", **phs_profile) as dst_phs:

                    for band_idx, name in enumerate(coh_names, start=1):
                        dst_coh.set_band_description(band_idx, name)

                    for band_idx, name in enumerate(cov_names, start=1):
                        dst_cov.set_band_description(band_idx, name)

                    dst_phs.set_band_description(1, "phase_variance")

                    for _, core_window in src_ref.block_windows(1):
                        ext_window = clamp_window(
                            col_off=int(core_window.col_off) - halo,
                            row_off=int(core_window.row_off) - halo,
                            width=int(core_window.width) + 2 * halo,
                            height=int(core_window.height) + 2 * halo,
                            max_width=width,
                            max_height=height,
                        )

                        stack_ext = []
                        for _, src_i, src_q in readers:
                            stack_ext.append(read_complex_window(src_i, src_q, ext_window))

                        coh_ext: Dict[str, np.ndarray] = {}
                        coh_sum = None
                        for a, b in coh_pairs:
                            arr = coherence(
                                stack_ext[look_set.subaps.index(a)],
                                stack_ext[look_set.subaps.index(b)],
                                win=win_size,
                            )
                            coh_ext[f"gamma{a}{b}"] = arr
                            coh_sum = arr if coh_sum is None else coh_sum + arr
                        coh_ext["gamma_mean"] = coh_sum / len(coh_pairs)

                        cov_ext = covariance_terms(stack_ext, win=win_size)
                        phv_ext = phase_variance(np.stack(stack_ext, axis=0)).astype(np.float32)

                        top = int(core_window.row_off - ext_window.row_off)
                        left = int(core_window.col_off - ext_window.col_off)
                        h = int(core_window.height)
                        w = int(core_window.width)

                        for band_idx, name in enumerate(coh_names, start=1):
                            dst_coh.write(
                                crop_center(coh_ext[name], top, left, h, w).astype(np.float32),
                                band_idx,
                                window=core_window,
                            )

                        for band_idx, name in enumerate(cov_names, start=1):
                            dst_cov.write(
                                crop_center(cov_ext[name], top, left, h, w).astype(np.float32),
                                band_idx,
                                window=core_window,
                            )

                        dst_phs.write(
                            crop_center(phv_ext, top, left, h, w).astype(np.float32),
                            1,
                            window=core_window,
                        )
            finally:
                for _, src_i, src_q in readers:
                    src_i.close()
                    src_q.close()

        if verbose:
            print(f"       {coh_out}")
            print(f"       {cov_out}")
            print(f"       {phs_out}")


def find_tc_products(root: Path) -> List[Path]:
    if root.is_dir() and root.name.endswith("_TC.data"):
        return [root]
    return sorted(p for p in root.rglob("*_TC.data") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute coherence, covariance, and phase variance from WorldSAR TC products."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="TC product root or single *_TC.data directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory where output feature folders are created.")
    parser.add_argument("--win-size", type=int, default=5, help="Local averaging window size. Must be odd.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    args = parser.parse_args()

    if args.win_size < 1 or args.win_size % 2 == 0:
        raise ValueError("--win-size must be a positive odd integer.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    products = find_tc_products(args.input_root)
    if not products:
        print(f"No *_TC.data directories found under: {args.input_root}")
        return

    print(f"Found {len(products)} *_TC.data products.")
    for product_dir in products:
        rel_parent = None
        if args.input_root in product_dir.parents:
            rel_parent = product_dir.parent.relative_to(args.input_root)
        process_product(
            product_dir=product_dir,
            out_root=args.output_root,
            win_size=args.win_size,
            verbose=not args.quiet,
            rel_parent=rel_parent,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
