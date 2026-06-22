"""Sentinel-1 TOPS SNAP2StaMPS export recipe."""

from __future__ import annotations

from sarpyx.pipelines.double_product.s1_insar import DEFAULT_DEM_NAME as DEFAULT_INSAR_DEM_NAME
from sarpyx.snapflow.runtime import PipelineStep

DEFAULT_ORBIT_TYPE = "Sentinel Precise (Auto Download)"
INPUT_KIND = "double"
PRODUCT_MODE = "S1STAMPS"
DEFAULT_DEM_NAME = DEFAULT_INSAR_DEM_NAME
DEFAULT_FIRST_BURST = 1
DEFAULT_LAST_BURST = 9999
DEFAULT_TARGET_FOLDER = "stamps"
DEFAULT_STAMPS_PREP_METHOD = "snap"
DEFAULT_STAMPS_DA_THRESHOLD = 0.4
DEFAULT_STAMPS_RG_PATCHES = 1
DEFAULT_STAMPS_AZ_PATCHES = 1
DEFAULT_STAMPS_RG_OVERLAP = 50
DEFAULT_STAMPS_AZ_OVERLAP = 50


def steps(
    orbit_type: str = DEFAULT_ORBIT_TYPE,
    orbit_continue_on_fail: bool = False,
    subswath: str | None = None,
    selected_polarisations: list[str] | None = None,
    first_burst_index: int = DEFAULT_FIRST_BURST,
    last_burst_index: int = DEFAULT_LAST_BURST,
    use_esd: bool = True,
    dem_name: str = DEFAULT_DEM_NAME,
    subset: bool = False,
    subset_region: str | None = None,
    polygon_wkt: str | None = None,
    validate_inputs: bool = True,
    target_folder: str | None = DEFAULT_TARGET_FOLDER,
    psi_format: bool = True,
    run_stamps_prep: bool = True,
    stamps_prep_method: str = DEFAULT_STAMPS_PREP_METHOD,
    stamps_prep_master_date: str | None = None,
    stamps_prep_da_threshold: float = DEFAULT_STAMPS_DA_THRESHOLD,
    stamps_prep_rg_patches: int | None = DEFAULT_STAMPS_RG_PATCHES,
    stamps_prep_az_patches: int | None = DEFAULT_STAMPS_AZ_PATCHES,
    stamps_prep_rg_overlap: int | None = DEFAULT_STAMPS_RG_OVERLAP,
    stamps_prep_az_overlap: int | None = DEFAULT_STAMPS_AZ_OVERLAP,
    stamps_prep_maskfile: str | None = None,
    stamps_prep_command: str | None = None,
    stamps_prep_config_file: str | None = None,
    stamps_prep_workdir: str | None = None,
):
    if subset and not (subset_region or polygon_wkt):
        raise ValueError("subset=True requires subset_region or polygon_wkt.")

    split_params = {
        "subswath": subswath,
        "selected_polarisations": selected_polarisations,
        "first_burst_index": first_burst_index,
        "last_burst_index": last_burst_index,
    }
    orbit_params = {"orbit_type": orbit_type, "continue_on_fail": orbit_continue_on_fail}
    recipe = [
        PipelineStep("TopsarSplit", {"source_ref": "master", "outdir": "master", **split_params}, "master_split"),
        PipelineStep("ApplyOrbitFile", {"source_ref": "master_split", "outdir": "master", **orbit_params}, "master_orbit"),
        PipelineStep("TopsarSplit", {"source_ref": "slave", "outdir": "slave", **split_params}, "slave_split"),
        PipelineStep("ApplyOrbitFile", {"source_ref": "slave_split", "outdir": "slave", **orbit_params}, "slave_orbit"),
        PipelineStep("TopsarCoregistration", {"master_ref": "master_orbit", "slave_ref": "slave_orbit", "outdir": "coreg", "use_esd": use_esd, "dem_name": dem_name}, "coreg"),
        PipelineStep(
            "Interferogram",
            {
                "source_ref": "coreg",
                "outdir": "ifg",
                "subtract_flat_earth_phase": True,
                "dem_name": dem_name,
                "output_elevation": True,
                "output_lat_lon": True,
            },
            "ifg_raw",
        ),
        PipelineStep("Deburst", {"source_ref": "ifg_raw", "outdir": "ifg"}, "ifg"),
        PipelineStep(
            "AddElevation",
            {"source_ref": "ifg", "outdir": "ifg", "dem_name": dem_name, "elevation_band_name": "elevation"},
            "ifg_elev",
        ),
        PipelineStep(
            "AddStampsLatLonBands",
            {"source_ref": "ifg_elev", "outdir": "ifg", "output_name": "ifg_stamps_ready"},
            "ifg_stamps_ready",
        ),
        PipelineStep("Deburst", {"source_ref": "coreg", "outdir": "coreg"}, "coreg_deb"),
    ]

    coreg_ref = "coreg_deb"
    ifg_ref = "ifg_stamps_ready"
    if subset:
        subset_params = {"region": subset_region, "geo_region": polygon_wkt, "copy_metadata": True}
        recipe.extend(
            [
                PipelineStep("Subset", {"source_ref": ifg_ref, "outdir": "ifg", **subset_params}, "ifg_subset"),
                PipelineStep("Subset", {"source_ref": coreg_ref, "outdir": "coreg", **subset_params}, "coreg_subset"),
            ]
        )
        coreg_ref = "coreg_subset"
        ifg_ref = "ifg_subset"

    if validate_inputs:
        recipe.append(
            PipelineStep("ValidateStampsInputs", {"coreg_ref": coreg_ref, "ifg_ref": ifg_ref}, "validated_stamps_inputs")
        )

    recipe.append(
        PipelineStep(
            "StampsExport",
            {
                "coreg_ref": coreg_ref,
                "ifg_ref": ifg_ref,
                "outdir": "export",
                "target_folder": target_folder,
                "psi_format": psi_format,
                "output_name": "stamps_export",
            },
            "stamps_export",
        )
    )
    if run_stamps_prep:
        recipe.append(
            PipelineStep(
                "StampsPrep",
                {
                    "target_folder": target_folder,
                    "method": stamps_prep_method,
                    "master_date": stamps_prep_master_date,
                    "da_threshold": stamps_prep_da_threshold,
                    "rg_patches": stamps_prep_rg_patches,
                    "az_patches": stamps_prep_az_patches,
                    "rg_overlap": stamps_prep_rg_overlap,
                    "az_overlap": stamps_prep_az_overlap,
                    "maskfile": stamps_prep_maskfile,
                    "command": stamps_prep_command,
                    "config_file": stamps_prep_config_file,
                    "workdir": stamps_prep_workdir,
                },
                "stamps_prep",
            )
        )
    return recipe
