# Changes

- Added DIMAP payload validation before running SNAP on redirect products.
- Added validation and cleanup for partial `worldsar_subap_tc` outputs before reuse.
- Added one retry for subap Terrain-Correction after rebuilding the redirect product with copied raster payloads.
- Hardened raster copy/link helper to fail on missing source payloads and rebuild mismatched destinations.
- Added regressions for missing subap payloads and partial subap TC reuse.
