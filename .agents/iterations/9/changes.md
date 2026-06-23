Added Zarr raster-quality summarization mirroring H5 no-data policy.

Zarr tile validation now fails all-zero/all-NaN/partial raster tiles and reports valid/nodata fractions.

Zarr cut result validation now removes partial/no-data tile directories and reports them as partial.

Generic `pipeline s1_tops`/`s1_strip` now installs the WorldSAR Zarr pre-write hook so documented S1 pipeline runs get minimal training metadata.

`s1_insar.steps()` now exposes `orbit_type` and `orbit_continue_on_fail`, matching the S1 TOPS recipe's offline orbit fallback.

`pipeline s1_insar` now declares product mode `S1INSAR`; runtime tiling derives product WKT from the master product when no explicit product WKT is provided and installs a WorldSAR Zarr hook with subaperture features disabled.

WorldSAR Zarr metadata validation now requires `subap_feature_bands` only for `S1TOPS` and `S1STRIP`, not for `S1INSAR`.

Runtime DB indexing now receives the active cuts directory so `config.resolve_db_dir(cuts_outdir)` can default to `<cuts>/_db` after tiling.
