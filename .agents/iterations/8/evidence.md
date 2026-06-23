# Evidence

- StaMPS post-export preparation is a separate external-tool step after SNAP `StampsExport`.
- `mt_prep_snap` usage is `mt_prep_snap yyyymmdd datadir da_thresh [rg_patches az_patches rg_overlap az_overlap maskfile]`.
- The existing export under `data/2stamps_out_pipeline_fixed/stamps` has `rslc`, `diff0`, `geo`, and `dem`.
- The master date for the current master product resolves to `20250217`.
- Local environment has `tcsh`/`csh`, but no `mt_prep_snap`, `mt_prep_gamma`, MATLAB, SNAPHU, or Triangle on `PATH`.
