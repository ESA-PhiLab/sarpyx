from __future__ import annotations

import hashlib
from pathlib import Path
import secrets
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from scipy.constants import c

from sarpyx import __version__ as sarpyx_version


GENERATOR_VERSION = "2.0.0"
SENTINEL1_TIMESTAMP_TOLERANCE_S = 2 * 2**-16


def freq2wavelen(f_hz):
    return c / f_hz


def fspl(r_m, wavelen_m):
    return (4 * np.pi * r_m / wavelen_m) ** 2


def sentinel1_receive_times(metadata: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return relative times of the first stored complex sample in each line.

    Accept one ordered, constant-PRI Sentinel-1 burst as decoded records or a
    pandas DataFrame. ``pri`` and ``swst`` must already be in seconds, and
    ``pri_count`` must be the instrument's integer counter. The physical-unit
    aliases ``pri_s`` and ``sampling_window_start_time_s`` are also accepted.
    Full decoded timing and identity headers are required for every line.
    Acquisition gaps advance time by their counter gaps. SWST changes remain.

    Times are relative to the first line's first stored sample. The common
    filter transient suppression delay, 320 / (8 * f_ref), cancels in this
    difference. Constant range decimation is required for one sample rate per
    stored array, independently of this common delay.
    Rank is an echo propagation-delay parameter, not an additional delay between
    these receive windows. Required coarse/fine timestamps cross-check the
    counters; their quantization is not used to introduce artificial PRI jitter.

    Variable-PRI, mixed-channel or reordered acquisitions are rejected. For a
    different acquisition scheme, construct ``line_start_times`` explicitly.

    Examples:
    from sarpyx.utils.rfigen import sentinel1_receive_times

    headers = [
        dict(pri=600e-6, swst=100e-6, pri_count=count,
             coarse_time=1000, fine_time=fine, range_decimation=8,
             rx_channel_id=1, swath_number=10, data_take_id=1, signal_type=0)
        for count, fine in ((100, 0.1), (101, 0.1006), (103, 0.1018))
    ]
    receive_times = sentinel1_receive_times(headers)
    """
    if hasattr(metadata, "to_dict"):
        metadata = metadata.to_dict("records")
    try:
        records = [dict(record) for record in metadata]
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain decoded header records") from exc
    if not records:
        raise ValueError("metadata must contain at least one acquisition line")
    for record in records:
        for field, alias in (("pri", "pri_s"), ("swst", "sampling_window_start_time_s")):
            if alias in record:
                if field in record and record[field] != record[alias]:
                    raise ValueError(f"conflicting metadata fields: {field} and {alias}")
                record[field] = record[alias]
    required = (
        "pri", "swst", "pri_count", "coarse_time", "fine_time",
        "range_decimation", "rx_channel_id", "swath_number", "data_take_id", "signal_type",
    )
    if any(field not in record for record in records for field in required):
        raise ValueError("metadata needs decoded pri, swst, pri_count, coarse_time, fine_time, "
                         "range_decimation, rx_channel_id, swath_number, data_take_id and signal_type")
    try:
        pri = np.asarray([r["pri"] for r in records], dtype=np.float64)
        swst = np.asarray([r["swst"] for r in records], dtype=np.float64)
        counts = np.asarray([r["pri_count"] for r in records], dtype=np.float64)
        coarse = np.asarray([r["coarse_time"] for r in records], dtype=np.float64)
        fine = np.asarray([r["fine_time"] for r in records], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("metadata needs numeric decoded pri, swst, pri_count, "
                         "coarse_time and fine_time fields") from exc
    if (not np.all(np.isfinite(pri)) or np.any(pri <= 0) or np.any(pri >= 1)
            or not np.all(np.isfinite(swst)) or np.any(swst < 0)
            or np.any(swst >= pri)):
        raise ValueError("pri and swst must be decoded seconds, with 0 <= swst < pri < 1")
    if not np.allclose(pri, pri[0], rtol=0, atol=1e-12):
        raise ValueError("variable PRI requires explicitly reconstructed line_start_times")
    if (not np.all(np.isfinite(counts)) or np.any(counts != np.floor(counts))
            or np.any(counts < 0) or np.any(counts >= 2**32)
            or any(isinstance(r["pri_count"], (bool, np.bool_)) for r in records)):
        raise ValueError("pri_count must contain unsigned 32-bit integer counters")
    for field in ("range_decimation", "swath_number", "rx_channel_id", "data_take_id"):
        try:
            values = np.asarray([r[field] for r in records], dtype=np.float64)
            if (not np.all(np.isfinite(values)) or np.any(values < 0)
                    or np.any(values != np.floor(values))
                    or any(isinstance(r[field], (bool, np.bool_)) for r in records)):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"metadata {field} must contain finite nonnegative integers") from exc
        if np.any(values != values[0]):
            raise ValueError(f"metadata must have a constant {field} within one burst")
    if any(r["signal_type"] != 0 for r in records):
        raise ValueError("sentinel1_receive_times supports echo packets only")
    steps = np.diff(counts.astype(np.int64)) % (2**32)
    if np.any(steps <= 0) or np.any(steps >= 2**31):
        raise ValueError("pri_count must advance in acquisition order without duplicates")
    elapsed = np.concatenate(([0.0], np.cumsum(steps * pri[0])))
    if (not np.all(np.isfinite(coarse)) or np.any(coarse != np.floor(coarse))
            or np.any(coarse < 0) or np.any(coarse >= 2**32)
            or any(isinstance(r["coarse_time"], (bool, np.bool_)) for r in records)):
        raise ValueError("coarse_time must contain unsigned 32-bit seconds")
    if not np.all(np.isfinite(fine)) or np.any(fine < 0) or np.any(fine >= 1):
        raise ValueError("fine_time must contain decoded fractional seconds in [0, 1)")
    # Subtract before adding to avoid precision loss at the GPS epoch.
    packet_elapsed = (coarse - coarse[0]) + (fine - fine[0])
    if np.any(np.abs(packet_elapsed - elapsed) > SENTINEL1_TIMESTAMP_TOLERANCE_S):
        raise ValueError("packet timestamps disagree with constant-PRI counter timing")
    times = elapsed + (swst - swst[0])
    if np.any(np.diff(times) <= 0):
        raise ValueError("receive-window start times must increase")
    return times


def _extent(value, size: int, name: str) -> tuple[int, int]:
    try:
        if len(value) != 2 or any(isinstance(v, (bool, np.bool_)) or int(v) != v for v in value):
            raise ValueError
        start, stop = map(int, value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer (start, stop) pair") from exc
    if not 0 <= start < stop <= size:
        raise ValueError(f"{name} must satisfy 0 <= start < stop <= {size}")
    return start, stop


def _mean_power(array: np.ndarray) -> float:
    """Accumulate in float64 without a full complex128 copy of a raw burst."""
    total = 0.0
    for row in array:
        values = np.asarray(row, dtype=np.complex128)
        total += float(np.vdot(values, values).real)
    return total / array.size


def _sample_chirp_train(times: np.ndarray, ir: dict, active: np.ndarray) -> np.ndarray:
    """Evaluate the continuous-time, pulse-local Hann chirp at supplied times."""
    result = np.zeros(times.shape, dtype=np.complex64)
    if not len(active):
        return result
    period = 1.0 / ir["PRF"]
    slots = np.floor((times - ir["pulse_start"]) / period).astype(np.int64)
    tau = times - (ir["pulse_start"] + slots * period)
    valid = (slots >= 0) & (slots < len(active)) & (tau >= 0) & (tau < ir["T"])
    valid &= active[np.clip(slots, 0, len(active) - 1)]
    local = tau[valid]
    centered = local - ir["T"] / 2
    envelope = np.sin(np.pi * local / ir["T"]) ** 2
    result[valid] = envelope * np.exp(
        1j * np.pi * ir["K"] * centered**2
        + 1j * 2 * np.pi * ir["freqShift"] * centered
    )
    return result


def GenerateRFISignal(
    sqd: np.ndarray,
    RadPar: Mapping[str, float] | None = None,
    verbose: bool = False,
    *,
    mode: str = "acquisition",
    line_start_times: Sequence[float] | None = None,
    seed: int | None = None,
    parameters: Mapping[str, float] | None = None,
    affected_az: tuple[int, int] | None = None,
    affected_rg: tuple[int, int] | None = None,
    generator_revision: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate RFI at acquisition sample times, or explicitly replay legacy RFI.

    ``sqd`` is a finite, nonempty (azimuth, range) raw echo array. ``RadPar``
    must explicitly supply ``fs``, ``fo`` and ``bw`` in Hz for this acquisition.
    ``line_start_times`` gives each row's first stored sample time in seconds,
    relative to the first row (which must be zero). Sample [a, r] is evaluated
    at ``line_start_times[a] + r / fs``. Use ``sentinel1_receive_times`` for
    decoded, constant-PRI Sentinel-1 headers. No timing is inferred from RFI PRF.

    ``parameters`` optionally overrides ``PRF``, ``duty`` (pulse width / PRI),
    ``bw``, ``freqShift``, ``SIR_dB``, ``pulse_probability`` or ``pulse_start``.
    T = duty / PRF; overlapping pulses are not supported. ``pulse_probability``
    independently controls Bernoulli activation of nominal pulse slots (default
    1: a regular train). ``pulse_start`` is the first nominal pulse onset relative
    to the first stored sample; by default it is drawn within the preceding PRI.
    The Hann envelope is evaluated analytically over the physical pulse duration.

    Half-open azimuth/range extents gate samples after timing is established.
    Changing the mask does not change pulse arrivals, chirp phase or activation.
    SIR is normalized using mean clean and interference powers over the *whole
    stored array*, including zeros. Mask changes can therefore change amplitude.
    If no active pulse intersects the selected samples, return zero interference,
    warn, and record ``achieved_SIR_dB=None``; do not silently redraw the train.

    A local seeded NumPy generator leaves global RNG state untouched. Returned
    metadata is JSON serializable and records the seed, active slots, receive
    times, physical parameters, source hash and optional generation commit.
    Version 2 requires explicit acquisition timing and radar parameters; old
    calls without them fail. ``mode="legacy"`` explicitly selects the unchanged
    flattened-stream algorithm, emits a RuntimeWarning and adds legacy labels.
    It keeps global ``np.random.seed`` behavior and the original metadata types,
    including the sampled ``t`` array. Acquisition-only keywords are rejected.

    Examples:
    import numpy as np
    from sarpyx.utils.rfigen import GenerateRFISignal

    echo = np.ones((8, 200), dtype=np.complex64)
    radar = {"fs": 4e6, "fo": 5.405e9, "bw": 1e6}
    interference, parameters = GenerateRFISignal(
        echo, radar, line_start_times=np.arange(8) * 600e-6, seed=12,
        parameters={"pulse_start": 0, "duty": 0.8},
        affected_az=(0, 8), affected_rg=(0, 200),
    )

    # Replay the historical model; its timing limits still apply.
    np.random.seed(12)
    legacy_interference, legacy_parameters = GenerateRFISignal(
        echo, radar, mode="legacy",
    )
    """
    if mode not in ("acquisition", "legacy"):
        raise ValueError("mode must be 'acquisition' or 'legacy'")
    if mode == "legacy":
        options = (line_start_times, seed, parameters, affected_az, affected_rg, generator_revision)
        if any(value is not None for value in options):
            raise ValueError("acquisition-only options are not supported in legacy mode; use np.random.seed")
        warnings.warn(
            "legacy RFI uses flattened-stream timing and nominal SIR scaling; "
            "its pulse timing is not acquisition-time ground truth",
            RuntimeWarning, stacklevel=2,
        )
        result, ir = _generate_legacy_rfi(sqd, RadPar, verbose)
        ir.update(generator_version="legacy", timing_model="flattened_stream")
        return result, ir
    if generator_revision is not None and not isinstance(generator_revision, str):
        raise ValueError("generator_revision must be a string or None")
    sqd = np.asarray(sqd)
    if sqd.ndim != 2 or not all(sqd.shape) or not np.issubdtype(sqd.dtype, np.number):
        raise ValueError("sqd must be a nonempty numeric 2D raw echo array")
    if not np.all(np.isfinite(sqd)):
        raise ValueError("sqd must contain only finite samples")
    if not isinstance(RadPar, Mapping) or any(k not in RadPar for k in ("fs", "fo", "bw")):
        raise ValueError("RadPar must explicitly provide acquisition fs, fo and bw in Hz")
    try:
        fs, fo, radar_bw = (float(RadPar[k]) for k in ("fs", "fo", "bw"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("RadPar fs, fo and bw must be finite numeric values in Hz") from exc
    if not all(np.isfinite(v) and v > 0 for v in (fs, fo, radar_bw)) or radar_bw > fs:
        raise ValueError("fs, fo and bw must be finite and positive, with bw <= fs")
    rows, cols = sqd.shape
    if line_start_times is None:
        raise ValueError("line_start_times is required; use sentinel1_receive_times(metadata)")
    times = np.asarray(line_start_times, dtype=np.float64)
    if times.shape != (rows,) or not np.all(np.isfinite(times)) or times[0] != 0:
        raise ValueError("line_start_times must have one finite time per row, starting at zero")
    if np.any(np.diff(times) <= 0) or np.any(np.diff(times) < cols / fs - 1e-12):
        raise ValueError("line_start_times must describe ordered, non-overlapping receive windows")
    signal_power = _mean_power(sqd)
    if not np.isfinite(signal_power) or signal_power <= 0:
        raise ValueError("sqd must have finite, positive mean power to set SIR")
    if seed is None:
        seed = secrets.randbits(64)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if parameters is not None and not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping of RFI parameter names to numbers")
    supplied = dict(parameters or {})
    allowed = {"PRF", "duty", "bw", "freqShift", "SIR_dB", "pulse_probability", "pulse_start"}
    if set(supplied) - allowed:
        raise ValueError(f"unsupported parameters: {sorted(map(str, set(supplied) - allowed))}")
    # Always draw these values, so overrides and masks do not shift the RNG.
    randoms = rng.random(8)
    ir = {
        "PRF": float(1000 + int(1000 * randoms[0])),
        "duty": float(0.1 + 0.4 * randoms[1]),
        "bw": float((0.005 + 0.045 * randoms[2]) * radar_bw),
        "SIR_dB": float(-25 + 20 * randoms[3]),
        "pulse_probability": 1.0,
    }
    try:
        ir.update({k: float(v) for k, v in supplied.items()})
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("RFI parameters must contain finite numeric values") from exc
    ir.setdefault("freqShift", float((randoms[4] - 0.5) * (radar_bw - ir["bw"])))
    if (not all(np.isfinite(v) for v in ir.values()) or ir["PRF"] <= 0
            or not 0 < ir["duty"] <= 1 or not 0 < ir["bw"] <= radar_bw
            or not 0 <= ir["pulse_probability"] <= 1):
        raise ValueError("invalid RFI parameters: require PRF > 0, 0 < duty <= 1, valid bandwidth/probability")
    if abs(ir["freqShift"]) + ir["bw"] / 2 > radar_bw / 2:
        raise ValueError("the nominal chirp sweep must lie inside the radar bandwidth")
    ir.update({"fc": fo + ir["freqShift"], "T": ir["duty"] / ir["PRF"], "Gain": 1.0})
    if (not np.isfinite(1.0 / ir["PRF"])
            or not np.isfinite(ir["fc"]) or ir["fc"] <= 0
            or not np.isfinite(ir["T"]) or ir["T"] * fs < 2):
        raise ValueError("require a finite pulse period, positive carrier and at least two samples per pulse duration")
    ir.setdefault("pulse_start", float(-randoms[5] / ir["PRF"]))
    ir["Lambda"] = float(freq2wavelen(ir["fc"]))
    ir["K"] = ir["bw"] / ir["T"]
    if not all(np.isfinite(ir[k]) for k in ("Lambda", "K", "pulse_start")):
        raise ValueError("derived RFI parameters must be finite")
    last_time = float(times[-1]) + (cols - 1) / fs
    last_slot = (last_time - ir["pulse_start"]) * ir["PRF"] if last_time >= ir["pulse_start"] else -1.0
    if not np.isfinite(last_slot) or last_slot >= 10_000_000:
        raise ValueError("too many pulse slots; check time units, pulse_start and PRF")
    slot_count = max(0, int(np.floor(last_slot)) + 1)
    active = rng.random(slot_count) < ir["pulse_probability"]
    az_size = max(1, int(rows * (0.05 + 0.9 * randoms[6])))
    rg_size = max(1, int(cols * (0.05 + 0.9 * randoms[7])))
    az_start = int(rng.integers(0, rows - az_size + 1))
    rg_start = int(rng.integers(0, cols - rg_size + 1))
    az = _extent(affected_az if affected_az is not None else (az_start, az_start + az_size), rows, "affected_az")
    rg = _extent(affected_rg if affected_rg is not None else (rg_start, rg_start + rg_size), cols, "affected_rg")
    result = np.zeros((rows, cols), dtype=np.complex64)
    range_times = np.arange(rg[0], rg[1], dtype=np.float64) / fs
    # Evaluate one window at a time; never allocate a waveform at fs across gaps.
    with np.errstate(over="ignore", invalid="ignore"):
        for row in range(*az):
            result[row, rg[0]:rg[1]] = _sample_chirp_train(times[row] + range_times, ir, active)
    unscaled_power = _mean_power(result)
    if not np.isfinite(unscaled_power):
        raise ValueError("RFI waveform cannot be represented with the supplied parameters")
    scale = 0.0
    achieved_sir = None
    if unscaled_power > 0:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale = float(np.power(10.0, (np.log10(signal_power) - ir["SIR_dB"] / 10
                                         - np.log10(unscaled_power)) / 2))
        if not np.isfinite(scale) or scale <= 0 or scale > np.finfo(np.float32).max:
            raise ValueError("requested SIR cannot be represented in complex64")
        result *= scale
        measured_power = _mean_power(result)
        if not np.isfinite(measured_power) or measured_power <= 0:
            raise ValueError("requested SIR cannot be represented in complex64")
        achieved_sir = float(10 * (np.log10(signal_power) - np.log10(measured_power)))
    else:
        warnings.warn("No active RFI samples intersect the selected receive windows; SIR is undefined", RuntimeWarning, stacklevel=2)
    ir.update({
        "AffectedAz": list(az), "AffectedRg": list(rg),
        "generator_version": GENERATOR_VERSION,
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_revision": generator_revision,
        "sarpyx_version": sarpyx_version, "numpy_version": np.__version__,
        "timing_model": "acquisition_sample_times",
        "seed": seed, "rng": type(rng.bit_generator).__name__,
        "fs": fs, "radar_fo": fo, "radar_bw": radar_bw,
        "line_start_times_s": times.tolist(), "array_shape": list(sqd.shape),
        "active_pulse_indices": np.flatnonzero(active).tolist(),
        "pulse_slot_count": slot_count, "envelope": "continuous_hann",
        "pulse_phase_model": "identical_pulse_local_phase",
        "sir_reference": "whole_burst_mean_power",
        "achieved_SIR_dB": achieved_sir, "amplitude_scale": scale,
    })
    if verbose:
        print(f"[INFO] RFI generator {GENERATOR_VERSION}: acquisition sample times, "
              f"PRF={ir['PRF']:g} Hz, T={ir['T'] * 1e6:.3f} us, "
              f"achieved whole-burst SIR={achieved_sir}")
    return result, ir


# Preserve legacy arithmetic and global RNG draws for exact replay.
def _generate_legacy_rfi(sqd, RadPar=None, verbose=False):
    """Generate Radio Frequency Interference (RFI) signal for SAR data.

    Args:
        sqd (np.ndarray): Input SAR single look complex data with shape (rows, cols).
        RadPar (dict): Radar parameters containing:
            - 'bw': Bandwidth in Hz
            - 'fo': Center frequency in Hz
            - 'fs': Sampling frequency in Hz

    Returns:
        tuple: A tuple containing:
            - sInterf (np.ndarray): Generated interference signal with same shape as sqd
            - IR (dict): Dictionary containing RFI parameters used in generation
    """


    if sqd is None:
        raise ValueError('The parameter "sqd" must be a valid numpy ndarray.')


    RadPar = {
            'fo': 5405000454.33435,
            'fs': 64345238.12571428,
            'bw': 56504455.48389234,
            'ts': 1.554116558005821e-08
    } if RadPar is None else RadPar


    IR = {}

    # === RFI Physical Parameters ===
    IR['freqShift'] = (np.random.rand() - 0.5) * RadPar['bw']  # shift within radar BW
    IR['fc'] = RadPar['fo'] + IR['freqShift']
    IR['SIR_dB'] = np.random.uniform(-25, -5)  # control strength
    IR['bw'] = np.random.uniform(0.005, 0.05) * RadPar['bw']  # narrower than radar BW
    IR['PRF'] = np.random.randint(1000, 2000)
    IR['duty'] = np.random.uniform(0.1, 0.5)  # sparse duty
    IR['Gain'] = 1.0
    IR['Lambda'] = freq2wavelen(IR['fc'])
    IR['T'] = IR['duty'] / IR['PRF']
    IR['K'] = IR['bw'] / IR['T']
    fs = RadPar['fs']
    IR['t'] = np.arange(-IR['T']/2, IR['T']/2, 1/fs)

    # === Chirp Pulse Generation ===
    chirp = np.exp(1j * np.pi * IR['K'] * IR['t']**2)
    chirp *= np.exp(1j * 2 * np.pi * IR['freqShift'] * IR['t'])  # center-shift
    chirp *= np.hanning(len(chirp))  # envelope
    chirp /= np.max(np.abs(chirp))

    # === Pulse Train (1D) ===
    rows, cols = sqd.shape
    burst_duration = rows / IR['PRF']
    IRstream_len = int(burst_duration * fs)
    pulse_spacing = int(fs / IR['PRF'])
    pulse_len = len(chirp)
    IRstream = np.zeros(IRstream_len, dtype=np.complex64)

    pulse_indices = np.arange(0, IRstream_len - pulse_len, pulse_spacing)
    N_total = len(pulse_indices)
    N_active = int(IR['duty'] * N_total)
    active_idx = np.random.choice(pulse_indices, N_active, replace=False)
    for idx in active_idx:
        IRstream[idx:idx+pulse_len] += chirp

    # === Map to Burst Grid (Spatial Submask) ===
    IRmap = np.zeros((rows, cols), dtype=np.complex64)
    # Random RFI spatial extent (5-95% of image dimensions)
    az_fraction = np.random.uniform(0.05, 0.95)
    rg_fraction = np.random.uniform(0.05, 0.95)
    sub_rows = int(rows * az_fraction)
    sub_cols = int(cols * rg_fraction)

    # Random position ensuring no border overflow
    az_start = np.random.randint(0, max(1, rows - sub_rows + 1))
    az_stop = az_start + sub_rows
    rg_start = np.random.randint(0, max(1, cols - sub_cols + 1))
    rg_stop = rg_start + sub_cols

    # reshape with tiling
    IRstream_crop = IRstream[:sub_rows * sub_cols]
    IRmap[az_start:az_stop, rg_start:rg_stop] = IRstream_crop.reshape((sub_rows, sub_cols))

    # === Power Calibration ===
    signal_power_lin = np.mean(np.abs(sqd)**2)
    signal_power_dB = 10 * np.log10(signal_power_lin)
    PrRFI_dB = signal_power_dB - IR['SIR_dB']
    PrRFI = 10 ** (PrRFI_dB / 10)
    sInterf = IRmap * np.sqrt(PrRFI)

    # === Metadata & Logging ===
    IR['AffectedAz'] = (az_start, az_stop)
    IR['AffectedRg'] = (rg_start, rg_stop)
    if verbose:
        print(f"[INFO] Injected RFI: freq offset = {IR['freqShift']/1e6:.2f} MHz, bw = {IR['bw']/1e6:.2f} MHz")
        print(f"[INFO] SIR = {IR['SIR_dB']} dB, Affected area: Az[{az_start}:{az_stop}], Rg[{rg_start}:{rg_stop}]")

    return sInterf, IR
