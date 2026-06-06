"""Sentinel-1 stripmap WorldSAR recipe."""

from __future__ import annotations

from sarpyx.snapflow.runtime import PipelineStep

DEFAULT_ORBIT_TYPE = "Sentinel Precise (Auto Download)"
INPUT_KIND = "single"
PRODUCT_MODE = "S1STRIP"
DEFAULT_SUBAP_DECOMPOSITIONS = [2]
DEFAULT_MAP_PROJECTION = "AUTO:42001"
DEFAULT_PIXEL_SPACING_M = 10.0
PDEC_PARAMS = {"decomposition": "H-Alpha Dual Pol Decomposition", "window_size": 5}
FALLBACK_BANDMERGE_STEPS = (
    PipelineStep("update_dim_add_bands_from_data_dir", {"dim_ref": "cal", "save_ref": "cal", "verbose": False}),
    PipelineStep("BandMerge", {"source_refs": ("pdec", "cal"), "output_name_ref": "pdec"}),
)


def steps(
    orbit_type: str = DEFAULT_ORBIT_TYPE,
    orbit_continue_on_fail: bool = False,
    sentinel_tc_source_band: str | None = None,
    sentinel_subap_decompositions: list[int] | None = None,
):
    subap_params = {
        "n_decompositions": sentinel_subap_decompositions or DEFAULT_SUBAP_DECOMPOSITIONS,
        "byte_order": 1,
        "VERBOSE": False,
        "update_dim": False,
    }
    tc_params = {
        "map_projection": DEFAULT_MAP_PROJECTION,
        "pixel_spacing_in_meter": DEFAULT_PIXEL_SPACING_M,
        "source_bands": [sentinel_tc_source_band] if sentinel_tc_source_band else None,
        "save_selected_source_band": True,
    }
    return [
        PipelineStep("ApplyOrbitFile", {"orbit_type": orbit_type, "orbit_continue_on_fail": orbit_continue_on_fail}, "orbit"),
        PipelineStep("Calibration", {"output_complex": True}, "cal"),
        PipelineStep("do_subaps", subap_params, "subaps"),
        PipelineStep("polarimetric_decomposition", dict(PDEC_PARAMS), "pdec"),
        PipelineStep(
            "merge_iq_into_pdec",
            {"src_ref": "cal", "pdec_ref": "pdec", "is_tops": False, "overwrite_copied_files": False, "backup": False, "fallback_steps": FALLBACK_BANDMERGE_STEPS},
            "merged",
        ),
        PipelineStep("TerrainCorrection", tc_params, "tc"),
        PipelineStep("WorldSARTiling", {"intermediate_ref": "tc"}, "tiling"),
    ]
