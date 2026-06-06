"""Sub-aperture feature derivation for WorldSAR Zarr tile payloads."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from sarpyx.sla.variance.compute_subap_features import coherence, covariance_terms, phase_variance

SUBAP_BAND_RE = re.compile(
    r"^(?:(?P<lookset>L\d+)_)?(?P<part>[iq])_(?:(?P<prefix>IW\d+)_)?(?P<pol>HH|HV|VH|VV)_SA(?P<sa>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SubapFeatureConfig:
    enabled: bool = False
    window_size: int = 5

    def validate(self) -> None:
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
        for term_name, data in covariance_terms(stack, win=config.window_size).items():
            name = f"subap_covariance_{suffix}_{term_name}"
            out_arrays[name] = data.astype(np.float32)
            out_attrs[name] = _feature_attrs(suffix, "covariance")
            feature_names.append(name)
        name = f"subap_phase_variance_{suffix}"
        out_arrays[name] = phase_variance(np.stack(stack, axis=0)).astype(np.float32)
        out_attrs[name] = _feature_attrs(suffix, "phase_variance")
        feature_names.append(name)
    return out_arrays, out_attrs, feature_names


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
