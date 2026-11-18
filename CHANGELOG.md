# Changelog

## 2.0.0 (unreleased)

### RFI acquisition timing (generator 2.0.0)

Related issue: [#36](https://github.com/ESA-PhiLab/sarpyx/issues/36).

- Replace the flattened-stream reshape with waveform evaluation at actual raw
  sample acquisition times. The waveform advances through receive gaps and
  unselected range samples. Masks do not alter pulse timing.
- Require explicit acquisition `fs`, `fo`, `bw` and relative per-line sample
  start times. Add a constant-PRI Sentinel-1 metadata helper with dropped-line,
  SWST, counter and timestamp validation. Acquisition-mode calls must be updated.
  Explicit `mode="legacy"` keeps
  the original algorithm and global NumPy seed behavior, with a warning.
- Separate pulse-width duty from pulse-slot activation probability (default 1).
  Use an analytic Hann envelope and preserve fractional-sample pulse intervals.
- Normalize SIR over the whole stored burst after masking and tapering; record
  achieved SIR. Zero-overlap realizations remain zero with undefined SIR.
- Record a reproducible local seed, active pulse slots, timing inputs, algorithm
  version, source hash and optional generation commit in JSON-safe metadata.
- Add regression tests, focused pull-request CI for Python 3.11/3.12, and an
  [existing-data notice and migration guide](docs/user_guide/rfi_generation.md).

This change does not modify existing RFInject releases. Affected legacy arrays
are not acquisition-time ground truth for physical pulse duration or PRF.
Corrected datasets require regeneration, separate versioning and provenance.
The package version is `2.0.0` because the default generation API has changed.
Existing callers must supply acquisition timing or select `mode="legacy"`.
The RFI algorithm and metadata contract also use `2.0.0`. Dataset versions remain
separate. This package version has not yet been released.
