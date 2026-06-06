"""Sentinel-1 TOPS WorldSAR recipe."""

from __future__ import annotations

from sarpyx.snapflow.runtime import PipelineStep

####################################### DEBUG MODE                      #################################
DEBUG=False
#########################################################################################################
DEFAULT_ORBIT_TYPE = "Sentinel Precise (Auto Download)"
INPUT_KIND = "single"
PRODUCT_MODE = "S1TOPS"
DEFAULT_SWATH = "IW1" if DEBUG else None
DEFAULT_SWATHS = ("IW1", "IW2", "IW3")
DEFAULT_FIRST_BURST = 1
DEFAULT_LAST_BURST = 3 if DEBUG else 9999
DEFAULT_SUBAP_DECOMPOSITIONS = [2]
DEFAULT_MAP_PROJECTION = "AUTO:42001"
DEFAULT_PIXEL_SPACING_M = 10.0

TOPS_SPLIT_PARAMS = {
    "subswath": DEFAULT_SWATH or "IW*",
    "first_burst_index": DEFAULT_FIRST_BURST,
    "last_burst_index": DEFAULT_LAST_BURST,
}

TOPS_SUBAP_PARAMS = {
    "byte_order": 1,
    "VERBOSE": False,
    "update_dim": False,
    "tops_iw_mode": True,
    "iw_apply_spectrum_normalization": False,
    "iw_energy_compensation": True,
    "iw_flip_output": True,
    "iw_row_equalization": False,
    "iw_doppler_centroid_correction": True,
    "iw_dc_smooth_win": 129,
    "iw_equal_energy_split": True,
    "iw_crosslook_row_balance": True,
    "iw_crosslook_row_balance_smooth_win": 257,
    "iw_crosslook_row_balance_clip": 1.5,
}
PDEC_PARAMS = {"decomposition": "H-Alpha Dual Pol Decomposition", "window_size": 5}





def steps(
    orbit_type: str = DEFAULT_ORBIT_TYPE,
    orbit_continue_on_fail: bool = False,
    sentinel_tc_source_band: str | None = None,
    sentinel_subap_decompositions: list[int] | None = None,
):
    subap_params = dict(TOPS_SUBAP_PARAMS)
    subap_params["n_decompositions"] = sentinel_subap_decompositions or DEFAULT_SUBAP_DECOMPOSITIONS
    tc_params = {
        "map_projection": DEFAULT_MAP_PROJECTION,
        "pixel_spacing_in_meter": DEFAULT_PIXEL_SPACING_M,
        "source_bands": [sentinel_tc_source_band] if sentinel_tc_source_band else None,
        "save_selected_source_band": True,
    }
    return [
        PipelineStep("TopsarSplit", dict(TOPS_SPLIT_PARAMS), "split"),
        PipelineStep("ApplyOrbitFile", {"orbit_type": orbit_type, "orbit_continue_on_fail": orbit_continue_on_fail}, "orbit"),
        PipelineStep("Calibration", {"output_complex": True}, "cal"),
        PipelineStep("TopsarDerampDemod", {}, "deramp"),
        PipelineStep("Deburst", {}, "deb"),
        PipelineStep("do_subaps", subap_params, "subaps"),
        PipelineStep("polarimetric_decomposition", dict(PDEC_PARAMS), "pdec"),
        PipelineStep(
            "merge_iq_into_pdec",
            {"src_ref": "deb", "pdec_ref": "pdec", "is_tops": True, "overwrite_copied_files": False, "backup": False},
            "merged",
        ),
        PipelineStep("TerrainCorrection", tc_params, "tc"),
        PipelineStep("WorldSARTiling", {"intermediate_ref": "tc", "collect": True}, "tiling"),
    ]
