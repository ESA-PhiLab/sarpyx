"""Sub-aperture feature derivation for WorldSAR Zarr tile payloads."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, replace
import warnings
from typing import Any

import numpy as np

from sarpyx.sla.variance.compute_subap_features import adjacent_phase_variance, coherence, local_mean

SUBAP_BAND_RE = re.compile(
    r"^(?:(?P<lookset>L\d+)_)?(?P<part>[iq])_(?:(?P<prefix>IW\d+)_)?(?P<pol>HH|HV|VH|VV)_SA(?P<sa>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SubapFeatureConfig:
    enabled: bool = False
    window_size: int = 5
    window_sizes: tuple[int, ...] = ()

    def validate(self) -> None:
        for window in self.window_sizes:
            if not isinstance(window, int) or window < 1 or window % 2 == 0:
                raise ValueError("window_sizes must contain positive odd integers")
        if len(set(self.window_sizes)) != len(self.window_sizes):
            raise ValueError("window_sizes must not contain duplicates")
        if self.window_size < 1 or self.window_size % 2 == 0:
            raise ValueError(f"sub-aperture feature window must be a positive odd integer, got {self.window_size}")


def add_subap_features(
    arrays: dict[str, np.ndarray],
    band_attrs: dict[str, dict[str, Any]],
    config: SubapFeatureConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[str]]:
    if not config.enabled:
        return arrays, band_attrs, []
    config.validate()
    if config.window_sizes:
        result = {n: a for n, a in arrays.items() if not _is_raw_subap_name(n)}
        attrs = {n: a for n, a in band_attrs.items() if n in result}
        names = []
        for window in config.window_sizes:
            derived, derived_attrs, derived_names = add_subap_features(
                arrays, band_attrs, replace(config, window_size=window, window_sizes=()))
            for name in derived_names:
                target = f"{name}_W{window}"
                result[target] = derived[name]
                attrs[target] = {**derived_attrs[name], "window_size": window}
                names.append(target)
        return result, attrs, names
    groups = _subap_groups(arrays)
    if not groups:
        return arrays, band_attrs, []

    out_arrays = {name: data for name, data in arrays.items() if not _is_raw_subap_name(name)}
    out_attrs = {name: attrs for name, attrs in band_attrs.items() if name in out_arrays}
    feature_names: list[str] = []
    for suffix, subaps in groups.items():
        if len(subaps) < 2:
            continue
        ordered = sorted(subaps)
        stack = [subaps[sa]["i"].astype(np.float32) + 1j * subaps[sa]["q"].astype(np.float32) for sa in ordered]
        for a, b in itertools.combinations(ordered, 2):
            name = f"subap_coherence_{suffix}_gamma{a}{b}"
            out_arrays[name] = coherence(stack[ordered.index(a)], stack[ordered.index(b)], win=config.window_size).astype(np.float32)
            out_attrs[name] = _feature_attrs(suffix, "coherence")
            feature_names.append(name)
        coh_arrays = [out_arrays[name] for name in feature_names if name.startswith(f"subap_coherence_{suffix}_gamma")]
        if coh_arrays:
            name = f"subap_coherence_{suffix}_gamma_mean"
            out_arrays[name] = np.mean(np.stack(coh_arrays, axis=0), axis=0).astype(np.float32)
            out_attrs[name] = _feature_attrs(suffix, "coherence")
            feature_names.append(name)
        for a, b, left, right in zip(ordered[:-1], ordered[1:], stack[:-1], stack[1:], strict=True):
            if b != a + 1:
                continue
            name = f"subap_phase_diff_{suffix}_pd{a}{b}"
            out_arrays[name] = _pair_phase_difference(left, right, config.window_size)
            out_attrs[name] = {**_feature_attrs(suffix, "phase_diff"), "unit": "rad"}
            feature_names.append(name)
        if len(stack) >= 3:
            if any(b != a + 1 for a, b in zip(ordered, ordered[1:])):
                raise ValueError(f"Nonconsecutive subapertures in {suffix}: {ordered}")
            name = f"subap_adjacent_phase_variance_{suffix}"
            out_arrays[name] = adjacent_phase_variance(
                np.stack(stack), win=config.window_size, method="covariance")
            out_attrs[name] = {**_feature_attrs(suffix, "adjacent_phase_variance"),
                               "window_size": config.window_size,
                               "estimator": "adjacent_covariance_circular_variance",
                               "frequency_spacing": "assumed_equal_not_corrected"}
            feature_names.append(name)
        else:
            warnings.warn(f"{suffix}: adjacent phase variance omitted; at least 3 looks required", UserWarning)
    return out_arrays, out_attrs, feature_names


def _pair_phase_difference(left: np.ndarray, right: np.ndarray, win: int) -> np.ndarray:
    """Validation-framework phase difference; equal weight for valid phasors."""
    valid = np.isfinite(left) & np.isfinite(right) & (left != 0) & (right != 0)
    phase = np.angle(np.where(valid, left, 0) * np.conj(np.where(valid, right, 0)))
    if win == 1:
        return np.where(valid, phase, np.nan).astype(np.float32)
    phasor = np.where(valid, np.exp(1j * phase), 0)
    count = local_mean(valid.astype(np.float32), win)
    resultant = local_mean(phasor, win)
    with np.errstate(invalid="ignore"):
        return np.angle(resultant / np.where(count > 0, count, np.nan)).astype(np.float32)


def _subap_groups(arrays: dict[str, np.ndarray]) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    groups: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for name, data in arrays.items():
        match = SUBAP_BAND_RE.match(name)
        if not match:
            continue
        prefix = "_".join(part for part in ((match.group("lookset") or "").upper(), (match.group("prefix") or "").upper(), match.group("pol").upper()) if part)
        groups.setdefault(prefix, {}).setdefault(int(match.group("sa")), {})[match.group("part").lower()] = data
    return {
        suffix: {sa: iq for sa, iq in subaps.items() if {"i", "q"} <= set(iq)}
        for suffix, subaps in groups.items()
    }


def _is_raw_subap_name(name: str) -> bool:
    return SUBAP_BAND_RE.match(name) is not None


def _feature_attrs(suffix: str, family: str) -> dict[str, str]:
    return {"unit": "linear", "feature_family": family, "subap_group": suffix}
