"""Sentinel-1 TOPS InSAR recipe."""

from __future__ import annotations

from sarpyx.snapflow.runtime import PipelineStep

DEFAULT_ORBIT_TYPE = "Sentinel Precise (Auto Download)"
INPUT_KIND = "double"
PRODUCT_MODE = "S1INSAR"
DEFAULT_DEM_NAME = "Copernicus 30m Global DEM"
DEFAULT_MAP_PROJECTION = "AUTO:42001"
DEFAULT_PIXEL_SPACING_M = 10.0
DEFAULT_FIRST_BURST = 1
DEFAULT_LAST_BURST = 9999


def steps(
    orbit_type: str = DEFAULT_ORBIT_TYPE,
    orbit_continue_on_fail: bool = False,
    subswath: str | None = None,
    selected_polarisations: list[str] | None = None,
    use_esd: bool = True,
    dem_name: str = DEFAULT_DEM_NAME,
    pixel_spacing_in_meter: float = DEFAULT_PIXEL_SPACING_M,
    map_projection: str = DEFAULT_MAP_PROJECTION,
):
    split_params = {
        "subswath": subswath,
        "selected_polarisations": selected_polarisations,
        "first_burst_index": DEFAULT_FIRST_BURST,
        "last_burst_index": DEFAULT_LAST_BURST,
    }
    orbit_params = {"orbit_type": orbit_type, "continue_on_fail": orbit_continue_on_fail}
    return [
        PipelineStep("TopsarSplit", {"source_ref": "master", "outdir": "master", **split_params}, "master_split"),
        PipelineStep("ApplyOrbitFile", {"source_ref": "master_split", "outdir": "master", **orbit_params}, "master_orbit"),
        PipelineStep("TopsarSplit", {"source_ref": "slave", "outdir": "slave", **split_params}, "slave_split"),
        PipelineStep("ApplyOrbitFile", {"source_ref": "slave_split", "outdir": "slave", **orbit_params}, "slave_orbit"),
        PipelineStep("TopsarCoregistration", {"master_ref": "master_orbit", "slave_ref": "slave_orbit", "outdir": "pair", "use_esd": use_esd, "dem_name": dem_name}, "coreg"),
        PipelineStep("Deburst", {"source_ref": "coreg", "outdir": "pair"}, "coreg_deb"),
        PipelineStep("Interferogram", {"source_ref": "coreg_deb", "outdir": "pair", "subtract_flat_earth_phase": True}, "ifg"),
        PipelineStep("TopoPhaseRemoval", {"source_ref": "ifg", "outdir": "pair", "dem_name": dem_name}, "topo"),
        PipelineStep("TerrainCorrection", {"source_ref": "subset", "outdir": "pair", "dem_name": dem_name, "pixel_spacing_in_meter": pixel_spacing_in_meter, "map_projection": map_projection}, "terrain_corrected"),
        PipelineStep("WorldSARTiling", {"intermediate_ref": "terrain_corrected"}, "tiling"),
    ]
