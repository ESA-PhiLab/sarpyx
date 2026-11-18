# RFI generation: acquisition timing and existing dataset notice

## Notice for existing RFInject data

The legacy `GenerateRFISignal` implementation mapped consecutive samples of a
one-dimensional interference stream into a selected azimuth-range rectangle by
reshaping. It did not advance the waveform through the elapsed time between
selected segments of successive receive windows. Its stored `PRF` and `T`
parameters therefore describe the synthetic stream, not necessarily physical
pulse timing on the SAR acquisition timeline. Selecting a narrower range region
also changed the apparent timing across rows.

This behaviour is present in the [code reviewed in issue #36](https://github.com/ESA-PhiLab/sarpyx/blob/ba2bf57c48fc5c2460df2d0b64f393fa253da20b/sarpyx/utils/rfigen.py)
and the [released example identified in that issue](https://github.com/ESA-PhiLab/sarpyx/issues/36).
The reviewed code revision is not, by itself, proof of the exact revision used
to generate every released product. The example's inspected metadata did not
record that generator revision. Credit to @Rich-Hall for identifying the timing
problem and providing a reproducible example.

Existing affected arrays should not be used as acquisition-time ground truth
for physical interferer pulse duration or PRF. They remain available for
controlled experiments against the known injected arrays, with the synthetic
model's limitations stated. Timing artefacts can influence detection, mitigation
and focusing results; transfer to real interference needs separate validation.

**Updating this code does not repair previously generated arrays.** A corrected
dataset requires regeneration and a separate version with recorded provenance.
Preserve previous releases for reproducibility. Any future dataset-card update
should carry this notice. Do not silently replace old arrays or relabel their
pulse parameters as physical acquisition-time measurements. Code and dataset
versions are separate.

## Version 2 timing model

For a raw echo array indexed by azimuth row `a` and range column `r`, the
interference is evaluated at

```text
t[a, r] = line_start_times[a] + r / fs
RFI[a, r] = amplitude_scale * mask[a, r] * x(t[a, r])
```

`line_start_times[a]` is the time of the first *stored* complex sample in row
`a`, in seconds relative to the first stored sample in row 0. It starts at zero.
`fs` is the actual complex-sample rate after range decimation. The waveform's
pulse index, envelope and chirp phase advance through all unobserved time.
A pulse that ends during a gap does not resume in the next row. A pulse long
enough to overlap several windows is evaluated at its correct age in each one.

The mask only selects samples from this timeline. Changing it does not restart
or retime the underlying pulse train. Whole-burst SIR normalization may change
amplitude when mask size changes. There is no continuous high-rate array spanning
the acquisition gaps; the implementation evaluates one selected row at a time.

## Required caller update

The default is `mode="acquisition"`. The old call `GenerateRFISignal(echo)`
is no longer sufficient. Version 2
requires explicit acquisition radar parameters and `line_start_times`; it fails
with an explanatory error when they are missing. No timing is inferred from
interferer PRF, array dimensions or hard-coded Sentinel-1 defaults. The return
value remains `(interference, parameters)`.

For one unmodified, decoded Sentinel-1 echo burst, use ordered metadata with
one record per echo row. Both records and pandas DataFrames are accepted.
The radar example below uses `decode.py` metadata in physical units:

```python
from sarpyx.processor.core.code2physical import range_dec_to_sample_rate
from sarpyx.utils.rfigen import GenerateRFISignal, sentinel1_receive_times

# echo: raw complex array; metadata: decoded records aligned with its rows.
first = metadata.iloc[0] if hasattr(metadata, "iloc") else metadata[0]
radar = {
    "fs": range_dec_to_sample_rate(int(first["range_decimation"])),
    "fo": 5.405e9,  # Nominal Sentinel-1 carrier; use the measured value if available.
    "bw": abs(first["tx_ramp_rate"]) * first["tx_pulse_length"],
}
receive_times = sentinel1_receive_times(metadata)
interference, parameters = GenerateRFISignal(
    echo, radar, line_start_times=receive_times, seed=20261107,
    parameters={"pulse_probability": 1.0},
    # generator_revision="<full commit SHA of the clean generation checkout>",
)
contaminated = echo + interference
# Write a new versioned output. Save the complete parameters and source headers.
```

For `code2physical.py` records, replace the radar dictionary with:

```python
radar = {
    "fs": first["sample_rate_hz"] if "sample_rate_hz" in first
          else range_dec_to_sample_rate(int(first["range_decimation"])),
    "fo": 5.405e9,
    "bw": abs(first["tx_pulse_ramp_rate_hz_per_s"]) * first["tx_pulse_length_s"],
}
```

These are two metadata schemas. Do not pass packet codes as physical values.
Update each production caller before regeneration. Keep source arrays and headers.

### Explicit legacy mode

Use legacy mode only to reproduce the original flattened-stream algorithm:

```python
import numpy as np

np.random.seed(1234)
interference, parameters = GenerateRFISignal(echo, radar, mode="legacy")
```

This call emits a `RuntimeWarning`. It keeps the original algorithm, global
NumPy random state and original metadata, including the NumPy array `t`.
It adds `generator_version="legacy"` and `timing_model="flattened_stream"`.
Legacy metadata is not JSON-safe without conversion. Legacy mode rejects all
acquisition-only keywords with non-`None` values: `line_start_times`, `seed`,
`parameters`, `affected_az`, `affected_rg` and `generator_revision`.
Unknown modes fail. There is no automatic fallback to legacy mode.

## Sentinel-1 metadata assumptions

`sentinel1_receive_times` handles an ordered, constant-PRI, single-channel echo
burst. It uses PRI-counter differences and decoded PRI, including counter gaps
for missing lines, plus changes in the decoded sampling-window start time
(SWST). The filter transient suppression delay is `320 / (8 * f_ref)`. This common
delay cancels in relative times independently of decimation. Constant range
decimation is required because one stored array uses one sample rate.
Do not add the echo's `rank * PRI` to the receive-window schedule.

All records must include `pri_count`, `coarse_time`, `fine_time`,
`range_decimation`, `rx_channel_id`, `swath_number`, `data_take_id` and
`signal_type=0`. Timing fields are either decoded `pri` and
`swst`, or `pri_s` and `sampling_window_start_time_s`.
Conflicting aliases fail. PRI and configuration fields must remain constant;
SWST may change but must stay below PRI. Unsigned counter differences preserve
wraps and missing-line gaps.

Required coarse/fine packet times cross-check the counter reconstruction within
`2 * 2**-16` seconds.
The fine-time field is quantized; directly differencing those timestamps can
introduce apparent line-to-line jitter. The helper retains the constant PRI
instead. See the [Sentinel-1 packet specification](https://sentinels.copernicus.eu/documents/247904/2142675/Sentinel-1-SAR-Space-Packet-Protocol-Data-Unit.pdf),
sections 3.2.1, 3.2.5.9-12, for timestamp, PRI, SWST and stored-window definitions.

The helper rejects variable PRI, inconsistent counters, mixed channels/swaths,
calibration packets and changing decimation. For other acquisition modes,
construct relative `line_start_times` from a validated acquisition schedule.
Do not substitute a single average PRI across mode changes or missing intervals.
If range data have been cropped or resampled, reconstruct both the first stored
sample times and the sample rate for those processed arrays. This utility is for
raw acquisition samples; it is not a forward model for focused SLC pixels.

For a deliberately uniform simulated schedule with no missing lines:

```python
import numpy as np
receive_times = np.arange(echo.shape[0], dtype=np.float64) * sar_pri_seconds
interference, parameters = GenerateRFISignal(
    echo, radar, line_start_times=receive_times, seed=1234
)
```

## Waveform and SIR definitions

- `PRF` is the nominal interfering pulse-slot rate in Hz.
- `duty` is the ratio `T * PRF`, with `0 < duty <= 1`; `T` is physical pulse
  support duration in seconds. `K = bw / T` is the chirp rate.
- `pulse_probability` independently controls Bernoulli activation of each slot.
  It defaults to 1 (regular train). For a long sparse train, expected temporal
  occupancy before receive gating is approximately `duty * pulse_probability`.
  Finite observations, missed pulses and masking can change detected rates.
  The legacy generator reused `duty` for both width and active-slot selection;
  that coupling is not retained in version 2.
- `pulse_start` is the first nominal pulse onset relative to the first sample.
  The default random onset lies in the preceding PRI, allowing an initial
  partial pulse. Explicit overrides may place the first onset elsewhere.
- The envelope is a continuous Hann function over `[0, T)`, sampled analytically;
  this replaces the legacy sample-count-defined Hann template. Each pulse uses
  the same pulse-local phase law. No sample rounding is applied to the PRI.
- The nominal frequency sweep must lie inside the specified radar bandwidth,
  which must not exceed `fs`. Hann sidelobes are not strictly bandlimited. This
  model does not simulate the receiver's analogue filters, nonlinear saturation,
  propagation, Doppler, or interferer geometry.
- `SIR_dB` means `10 log10(mean(abs(clean)**2) / mean(abs(RFI)**2))`, with both
  means over the **entire stored array**, including zero interference samples.
  Normalization is applied after tapering, activation and masking. This corrects
  the legacy peak-envelope scaling and explicitly differs from an active-region
  SIR or a PSD ratio. `achieved_SIR_dB` records the measured result.
- If no active pulse intersects the selected windows, the output is zero, a
  warning is raised, and `achieved_SIR_dB` is `None`. The generator does not move
  or redraw pulses to manufacture an intersection.

Optional `parameters` overrides use the keys above plus `bw`, `freqShift` and
`SIR_dB`. `affected_az=(start, stop)` and `affected_rg=(start, stop)` select
half-open extents. All sizes, times and radar parameters are validated.

## Reproduction and provenance

Save the complete returned parameter dictionary, the source clean data and
its acquisition headers. The dictionary includes:

- generator version, module source SHA-256, optional generation commit,
  sarpyx version and NumPy version;
- local random seed, RNG type, selected active pulse indices and pulse epoch;
- physical waveform parameters, masks, receive times and sample rate;
- normalization convention, amplitude scale and achieved SIR.

To repeat a call, use the same seed, physical overrides, extents, radar parameters,
receive times and clean input with the same generator and numerical environment.
The stored active slots also describe the realized pulse train independently of
future RNG changes. The old redundant sampled-template field `t` is replaced by
explicit analytic waveform parameters and the `continuous_hann` envelope label.
Cross-version or cross-platform bitwise equality is not promised.

Record the full commit SHA from a clean checkout used to generate a release;
the module source hash alone does not identify the external pipeline. Release
manifests should additionally identify source products, decoder version, complete
configuration, dataset version and software environment. `generator_revision`
remains `None` if the caller does not supply it; it is never fabricated.

## Validation and release follow-up

Run `python -m pytest tests/test_rfigen.py -q`. The tests compare generated samples
against an independent sum of continuous-time pulses, exercise acquisition gaps,
long pulses crossing windows, noninteger sample PRIs, mask invariance, missing
lines, timestamp quantization, SIR and seeded reproduction.

Before publishing regenerated data, review this model against representative
real interference and compare old/new detection and mitigation experiments.
Synchronize the dataset card and publication's timing equations, parameter
ranges, SIR convention and reproducibility statements with the released code.
A software regression suite establishes implementation behaviour; it does not
establish that all real interferer phenomenology is represented.
