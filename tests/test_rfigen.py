"""Acquisition-time RFI regression tests; no network or radar binaries required."""

import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from sarpyx.utils.rfigen import GenerateRFISignal, sentinel1_receive_times


RADAR = {"fs": 4e6, "fo": 5.405e9, "bw": 1e6}
PARAMS = {"PRF": 1000, "duty": 0.16, "bw": 20e3,
          "freqShift": 70e3, "SIR_dB": -8, "pulse_start": 0.0}
LEGACY_GENERATOR = run_path(Path(__file__).parent / "fixtures" / "rfigen_legacy.py")["GenerateRFISignal"]


@pytest.fixture(autouse=True)
def preserve_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def generate(rows=8, cols=200, *, times=None, parameters=None, **kwargs):
    physical = dict(PARAMS, **(parameters or {}))
    return GenerateRFISignal(
        np.ones((rows, cols), dtype=np.complex64), RADAR,
        line_start_times=np.arange(rows) * 600e-6 if times is None else times,
        seed=kwargs.pop("seed", 12), parameters=physical,
        affected_az=kwargs.pop("affected_az", (0, rows)),
        affected_rg=kwargs.pop("affected_rg", (0, cols)), **kwargs,
    )


def direct_reference(times, ir):
    """Independent sum over physical pulse onsets, without slot/floor mapping."""
    reference = np.zeros(times.shape, dtype=np.complex128)
    for index in ir["active_pulse_indices"]:
        onset = ir["pulse_start"] + index / ir["PRF"]
        selected = (times >= onset) & (times < onset + ir["T"])
        tau = times[selected] - onset
        u = tau - ir["T"] / 2
        window = (1 - np.cos(2 * np.pi * tau / ir["T"])) / 2
        reference[selected] += window * np.exp(
            2j * np.pi * ir["freqShift"] * u + 1j * np.pi * ir["K"] * u**2
        )
    return reference


def test_pulse_does_not_resume_after_unobserved_gap():
    signal, ir = generate(rows=4)
    assert np.any(signal[0])
    assert not np.any(signal[1:])
    # A flattened 50-us-per-row stream would continue this 160-us chirp here.
    false_times = np.arange(800).reshape(4, 200) / RADAR["fs"]
    assert np.any(direct_reference(false_times, ir)[1:])


def test_long_pulse_advances_envelope_and_phase_across_gap():
    signal, ir = generate(rows=3, parameters={"duty": 0.8})
    times = np.arange(3)[:, None] * 600e-6 + np.arange(200) / RADAR["fs"]
    expected = direct_reference(times, ir)
    np.testing.assert_allclose(signal / ir["amplitude_scale"], expected, atol=1e-7)
    assert np.any(signal[1])  # This pulse overlaps two receive windows.
    assert not np.allclose(expected[0], expected[1])


def test_pulse_train_matches_continuous_reference_with_irregular_windows():
    starts = np.array([0, 550e-6, 1.31e-3, 2.44e-3, 3.02e-3, 4.7e-3])
    signal, ir = generate(rows=6, times=starts, parameters={"duty": 0.7, "pulse_start": -90e-6})
    times = starts[:, None] + np.arange(200) / RADAR["fs"]
    np.testing.assert_allclose(signal / ir["amplitude_scale"], direct_reference(times, ir), atol=1e-7)


def test_multiple_pulses_per_window_and_fractional_sample_pri():
    starts = np.array([0, .0031, .0076])
    signal, ir = generate(rows=3, cols=8000, times=starts,
                          parameters={"PRF": 1731.2345, "duty": 0.37})
    times = starts[:, None] + np.arange(8000) / RADAR["fs"]
    np.testing.assert_allclose(signal / ir["amplitude_scale"], direct_reference(times, ir), atol=1e-7)


def test_pulse_onsets_and_width_are_on_physical_timeline():
    starts = np.array([0, 1e-3, 2e-3])
    signal, ir = generate(rows=3, cols=1200, times=starts,
                          parameters={"duty": 0.15, "pulse_start": 50e-6})
    for row in range(3):
        support = np.flatnonzero(np.abs(signal[row]) > np.max(np.abs(signal[row])) * 1e-5)
        assert abs(support[0] / RADAR["fs"] - 50e-6) < 1e-6
        assert abs(support[-1] / RADAR["fs"] - 200e-6) < 1e-6
    assert ir["T"] == pytest.approx(150e-6)
    np.testing.assert_allclose(signal[0], signal[1], atol=1e-5)


def test_spatial_mask_does_not_retime_or_reselect_pulses():
    parameters = {"duty": 0.7, "pulse_probability": 0.6}
    whole, a = generate(parameters=parameters)
    crop, b = generate(parameters=parameters, affected_az=(1, 7), affected_rg=(70, 145))
    assert a["active_pulse_indices"] == b["active_pulse_indices"]
    assert a["pulse_start"] == b["pulse_start"]
    np.testing.assert_allclose(crop[1:7, 70:145] / b["amplitude_scale"],
                               whole[1:7, 70:145] / a["amplitude_scale"], atol=1e-7)
    outside = crop.copy()
    outside[1:7, 70:145] = 0
    assert not np.any(outside)


def test_sir_is_measured_over_whole_burst_after_mask_and_taper():
    result, ir = generate(parameters={"duty": 0.8, "SIR_dB": -11.0},
                          affected_az=(1, 7), affected_rg=(25, 90))
    measured = -10 * np.log10(np.mean(np.abs(result.astype(np.complex128))**2))
    assert measured == pytest.approx(-11, abs=1e-5)
    assert ir["achieved_SIR_dB"] == pytest.approx(measured)
    assert ir["sir_reference"] == "whole_burst_mean_power"


def test_pulses_entirely_in_gaps_are_not_relocated_to_force_sir():
    with pytest.warns(RuntimeWarning, match="No active RFI samples"):
        signal, ir = generate(rows=2, parameters={"pulse_start": 300e-6, "duty": 0.1})
    assert not np.any(signal)
    assert ir["achieved_SIR_dB"] is None
    assert ir["amplitude_scale"] == 0


def test_inactive_train_returns_zero_and_json_safe_metadata():
    with pytest.warns(RuntimeWarning, match="SIR is undefined"):
        signal, ir = generate(parameters={"pulse_probability": 0})
    assert not np.any(signal)
    assert ir["active_pulse_indices"] == []
    json.dumps(ir, allow_nan=False)


def test_regular_train_does_not_randomly_drop_pulses_by_default():
    _, ir = generate()
    assert ir["active_pulse_indices"] == list(range(ir["pulse_slot_count"]))
    assert ir["pulse_probability"] == 1


def test_seeded_generation_replays_and_leaves_global_rng_untouched():
    np.random.seed(9823)
    expected = np.random.random(5)
    np.random.seed(9823)
    first, a = generate(parameters={"pulse_probability": 0.7})
    np.testing.assert_array_equal(np.random.random(5), expected)
    second, b = generate(parameters={"pulse_probability": 0.7})
    np.testing.assert_array_equal(first, second)
    assert a == b
    json.dumps(a, allow_nan=False)
    source = Path(__file__).parents[1] / "sarpyx/utils/rfigen.py"
    assert a["generator_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert a["timing_model"] == "acquisition_sample_times"


def test_automatically_generated_seed_is_recorded_and_replayable():
    first, a = generate(seed=None, generator_revision="reviewed-commit")
    second, b = generate(seed=a["seed"], generator_revision="reviewed-commit")
    np.testing.assert_array_equal(first, second)
    assert a == b
    assert a["generator_revision"] == "reviewed-commit"


def headers(counts=(100, 101, 103), swst=(100e-6, 102e-6, 104e-6)):
    return [{"pri_count": n, "pri": 600e-6, "swst": w, "range_decimation": 8,
             "swath_number": 10, "rx_channel_id": 1, "data_take_id": 1,
             "signal_type": 0, "coarse_time": 1000, "fine_time": fine}
            for n, w, fine in zip(counts, swst, (0.1, 0.1006, 0.1018))]


def test_metadata_keeps_missing_lines_and_swst_changes():
    np.testing.assert_allclose(sentinel1_receive_times(headers()), [0, 602e-6, 1804e-6], atol=1e-15)


def test_metadata_supports_counter_wrap_and_dataframe():
    import pandas as pd
    h = headers(counts=(2**32 - 1, 0, 2))
    np.testing.assert_allclose(sentinel1_receive_times(pd.DataFrame(h)), [0, 602e-6, 1804e-6], atol=1e-15)


def test_metadata_uses_pri_counter_not_quantized_packet_intervals():
    h = headers(counts=(203688, 203689, 203690), swst=(126.5228491537653e-6,) * 3)
    for r, fine in zip(h, [0.9008102416992188, 0.9013900756835938, 0.9019851684570312]):
        r.update(pri=582.3674372819869e-6, coarse_time=1389473212, fine_time=fine)
    actual = sentinel1_receive_times(h)
    np.testing.assert_allclose(np.diff(actual), 582.3674372819869e-6, atol=1e-15)


@pytest.mark.parametrize("field,value,match", [
    ("pri", 700e-6, "variable PRI"), ("range_decimation", 7, "range_decimation"),
    ("rx_channel_id", 2, "rx_channel_id"), ("signal_type", 8, "echo packets"),
    ("pri_count", 100, "acquisition order"), ("swst", 100, "decoded seconds"),
])
def test_unsupported_metadata_fails_explicitly(field, value, match):
    h = headers()
    h[1][field] = value
    with pytest.raises(ValueError, match=match):
        sentinel1_receive_times(h)


def test_metadata_rejects_counter_timestamp_disagreement():
    h = headers()
    for row in h:
        row.update(coarse_time=100, fine_time=0.1)
    with pytest.raises(ValueError, match="disagree"):
        sentinel1_receive_times(h)


def test_metadata_rejects_raw_units_and_missing_fields():
    raw = headers()
    raw[0].update(pri=21859, swst=4700)
    with pytest.raises(ValueError, match="decoded seconds"):
        sentinel1_receive_times(raw)
    with pytest.raises(ValueError, match="decoded pri"):
        sentinel1_receive_times([{}])


@pytest.mark.parametrize("starts", [None, [0], [0, np.nan], [0, -0.001], [0, 1e-6], [1, 1.001]])
def test_missing_or_invalid_timing_cannot_silently_use_flattening(starts):
    with pytest.raises(ValueError, match="line_start_times"):
        GenerateRFISignal(np.ones((2, 200)), RADAR, line_start_times=starts)


@pytest.mark.parametrize("radar", [None, {}, dict(RADAR, fs=0), dict(RADAR, bw=8e6)])
def test_missing_or_invalid_radar_parameters_fail(radar):
    with pytest.raises(ValueError):
        GenerateRFISignal(np.ones((2, 200)), radar, line_start_times=[0, .001])


@pytest.mark.parametrize("parameters", [
    {"PRF": 0}, {"duty": 0}, {"duty": 1.2}, {"bw": -1},
    {"pulse_probability": 1.1}, {"freqShift": 0.6e6}, {"T": 100e-6},
    {"SIR_dB": np.nan}, {"pulse_start": np.nan},
])
def test_invalid_waveform_parameters_fail(parameters):
    with pytest.raises(ValueError):
        generate(parameters=parameters)


def test_small_arrays_have_nonempty_valid_default_extents():
    result, ir = GenerateRFISignal(np.ones((1, 10)), RADAR, line_start_times=[0], seed=1,
                                   parameters=dict(PARAMS, pulse_start=-10e-6))
    assert result.shape == (1, 10)
    assert ir["AffectedAz"] == [0, 1]
    assert 0 <= ir["AffectedRg"][0] < ir["AffectedRg"][1] <= 10


@pytest.mark.parametrize("mode", [None, "", "LEGACY", "unknown", 1])
def test_unknown_modes_fail(mode):
    with pytest.raises(ValueError, match="mode"):
        generate(mode=mode)


@pytest.mark.parametrize("seed", [0, 12, 9823])
@pytest.mark.parametrize("radar", [None, RADAR])
@pytest.mark.parametrize("shape", [(4, 64), (32, 256)])
def test_legacy_matches_original_outputs_metadata_and_rng(seed, radar, shape):
    clean = np.ones(shape, dtype=np.complex64)
    np.random.seed(seed)
    expected, expected_metadata = LEGACY_GENERATOR(clean, radar)
    expected_state = np.random.get_state()
    np.random.seed(seed)
    with pytest.warns(RuntimeWarning, match="legacy"):
        actual, metadata = GenerateRFISignal(clean, radar, mode="legacy")
    actual_state = np.random.get_state()
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert metadata.pop("generator_version") == "legacy"
    assert metadata.pop("timing_model") == "flattened_stream"
    assert metadata.keys() == expected_metadata.keys()
    for key, value in expected_metadata.items():
        assert type(metadata[key]) is type(value), key
        if isinstance(value, np.ndarray):
            assert metadata[key].dtype == value.dtype, key
        np.testing.assert_array_equal(metadata[key], value)
    assert actual_state[0] == expected_state[0]
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[2:] == expected_state[2:]
    np.testing.assert_array_equal(clean, np.ones_like(clean))


@pytest.mark.parametrize("options", [
    {"seed": 1}, {"line_start_times": [0, .001]}, {"parameters": {}},
    {"affected_az": (0, 2)}, {"affected_rg": (0, 200)},
    {"generator_revision": "commit"},
])
def test_legacy_rejects_acquisition_options_before_using_rng(options):
    before = np.random.get_state()
    with pytest.raises(ValueError, match="legacy"):
        GenerateRFISignal(np.ones((2, 200)), mode="legacy", **options)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_corrected_generation_leaves_input_unchanged():
    clean = np.arange(1600, dtype=np.float32).reshape(8, 200).astype(np.complex64) + 1j
    before = clean.copy()
    result, _ = GenerateRFISignal(
        clean, RADAR, line_start_times=np.arange(8) * 600e-6,
        seed=12, parameters=PARAMS, affected_az=(0, 8), affected_rg=(0, 200),
    )
    np.testing.assert_array_equal(clean, before)
    assert result.dtype == np.complex64
    assert not np.shares_memory(result, clean)


@pytest.mark.parametrize("clean", [
    None, [], np.ones(4), np.ones((0, 2)), np.ones((2, 0)),
    np.array([["text"]]), np.array([[np.nan]]), np.array([[np.inf]]),
    np.zeros((1, 2)),
])
def test_invalid_clean_input_fails(clean):
    with pytest.raises(ValueError, match="sqd"):
        GenerateRFISignal(clean, RADAR, line_start_times=[0])


@pytest.mark.parametrize("seed", [-1, True, np.bool_(False), 1.5, "12"])
def test_invalid_seeds_fail(seed):
    with pytest.raises(ValueError, match="seed"):
        generate(seed=seed)


@pytest.mark.parametrize("name", ["affected_az", "affected_rg"])
@pytest.mark.parametrize("extent", [
    (), (0,), (0, 1, 2), (-1, 1), (0, 0), (2, 1), (0, 201),
    (False, 1), (0.5, 1), (0, np.inf), (0, None), 1,
])
def test_invalid_extents_fail(name, extent):
    with pytest.raises(ValueError, match=name):
        generate(**{name: extent})


@pytest.mark.parametrize("parameters", [
    {"PRF": 1e-320}, {"PRF": 1e308}, {"SIR_dB": -1e6}, {"SIR_dB": 1e6},
    {"PRF": 1e-320, "duty": 1e-320}, {"PRF": 1e-308, "duty": 0.1},
])
def test_unrepresentable_waveform_parameters_fail(parameters):
    with pytest.raises(ValueError):
        generate(parameters=parameters)


def test_non_string_revision_cannot_make_metadata_non_serializable():
    with pytest.raises(ValueError, match="generator_revision"):
        generate(generator_revision=object())


def test_metadata_supports_physical_unit_field_names():
    h = headers()
    for row in h:
        row["pri_s"] = row.pop("pri")
        row["sampling_window_start_time_s"] = row.pop("swst")
    np.testing.assert_allclose(sentinel1_receive_times(h), [0, 602e-6, 1804e-6], atol=1e-15)


@pytest.mark.parametrize("field,alias", [
    ("pri", "pri_s"), ("swst", "sampling_window_start_time_s"),
])
def test_metadata_rejects_conflicting_aliases(field, alias):
    h = headers()
    for row in h:
        row[alias] = row[field]
    np.testing.assert_allclose(sentinel1_receive_times(h), [0, 602e-6, 1804e-6], atol=1e-15)
    h[1][alias] += 1e-6
    with pytest.raises(ValueError, match="conflict"):
        sentinel1_receive_times(h)


@pytest.mark.parametrize("field", [
    "pri", "swst", "pri_count", "coarse_time", "fine_time", "range_decimation",
    "rx_channel_id", "swath_number", "data_take_id", "signal_type",
])
@pytest.mark.parametrize("partial", [False, True])
def test_metadata_requires_full_headers(field, partial):
    h = headers()
    for row in h[1:2] if partial else h:
        del row[field]
    with pytest.raises(ValueError, match="metadata"):
        sentinel1_receive_times(h)


@pytest.mark.parametrize("field,value", [
    ("coarse_time", -1), ("coarse_time", 2**32), ("coarse_time", 0.5),
    ("coarse_time", np.nan), ("coarse_time", np.inf),
    ("fine_time", -0.1), ("fine_time", 1), ("fine_time", np.nan),
    ("pri_count", -1), ("pri_count", 2**32), ("pri_count", 1.5),
    ("pri_count", np.nan), ("pri_count", True),
])
def test_metadata_rejects_invalid_counter_and_timestamp_values(field, value):
    h = headers()
    h[1][field] = value
    with pytest.raises(ValueError, match=field):
        sentinel1_receive_times(h)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("factor", [0.99, 1.01])
def test_metadata_timestamp_tolerance_boundary(sign, factor):
    h = headers()
    h[1]["fine_time"] += sign * factor * 2 * 2**-16
    if factor < 1:
        np.testing.assert_allclose(sentinel1_receive_times(h), [0, 602e-6, 1804e-6], atol=1e-15)
    else:
        with pytest.raises(ValueError, match="disagree"):
            sentinel1_receive_times(h)


def test_metadata_rejects_empty_headers_and_swst_at_pri():
    with pytest.raises(ValueError, match="metadata"):
        sentinel1_receive_times([])
    h = headers()
    h[1]["swst"] = h[1]["pri"]
    with pytest.raises(ValueError, match="decoded seconds"):
        sentinel1_receive_times(h)


@pytest.mark.parametrize("field,value", [
    ("range_decimation", None), ("range_decimation", np.inf),
    ("rx_channel_id", -1), ("swath_number", 1.5),
    ("data_take_id", True),
])
def test_invalid_constant_header_identity_fails(field, value):
    h = headers()
    for row in h:
        row[field] = value
    with pytest.raises(ValueError, match=field):
        sentinel1_receive_times(h)
