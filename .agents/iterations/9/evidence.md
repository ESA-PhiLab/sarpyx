Existing `validation_feature_out` had 28 Zarr tiles; strict raster audit found 22 failures. `456U_93R` had valid_fraction=0.566101 and nodata_fraction=0.433899.

`validate_worldsar_zarr_tile` checked structure, metadata, and chunks only; it did not inspect raster pixels.

Fresh bounded `s1_tops` launch reached Terrain-Correction but failed because SNAP could not download EGM96 DEM auxiliary data from `step.esa.int` in the restricted network environment.

Tiling-only rerun against existing local TC produced candidate=526, expected=6, actual=6, partial=22, failed=0, missing=0, extra=0, cut_failed=False. Partial Zarr directories were removed during validation.

Local InSAR master/slave pair exists under `data/bursts/extracted/{master,slave}`. Both are S1A IW2 VV/VH, relative orbit 117, slice 8, with start times 2025-02-17T17:06:12 and 2025-02-05T17:06:13.

Initial `s1_insar` launch failed after master TOPSAR-Split because `Apply-Orbit-File` used `continueOnFail=false` and orbit downloads are unavailable offline.

Retry with `orbit_continue_on_fail=true` completed master/slave TOPSAR-Split and Apply-Orbit-File, then failed in TOPSAR coregistration/Back-Geocoding because SNAP could not download EGM96 DEM auxiliary data from `step.esa.int`.

Network-enabled `s1_insar` completed the local master/slave burst pair through TOPSAR Coregistration, Deburst, Interferogram, TopoPhaseRemoval, Subset, and Terrain-Correction. Final TC product: `data/insar/validation_out_network/pair/S1A_SLC_20250217T170613_249406_IW2_VV_468546_SPLIT_ORB_COREG_DEB_IFG_TOPO_SUB_TC.dim`.

Manual InSAR Zarr tiling using the completed TC product and `S1INSAR` hook produced candidate=56, expected=9, actual=9, partial=20, failed=0, missing=0, extra=0, cut_failed=False. All 9 remaining Zarr tiles validated successfully.

Fresh bounded `s1_tops` launch using the populated SNAP cache completed SNAP processing, WorldSAR feature intermediates, and Zarr cuts. It saved 6 full-coverage Zarr tiles, rejected 22 partial/no-data tiles, including `456U_93R` with nodata_fraction=0.433899, and direct validation showed all 6 kept tiles had valid_fraction=1.0 and status=success.

The same `s1_tops` command exited nonzero after successful tiling because `_run_db_indexing()` called `config.resolve_db_dir()` without a cuts directory, so `--cuts-outdir` could not supply the documented `<cuts>/_db` default.
