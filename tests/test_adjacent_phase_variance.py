import numpy as np
import pytest
from scipy.ndimage import uniform_filter
from sarpyx.sla.variance.compute_subap_features import adjacent_phase_variance, phase_variance
from sarpyx.hooks.subap_features import SubapFeatureConfig, add_subap_features


def test_ramp_invariance_and_original_counterexample():
    for slope in (0, np.pi/2, 2*np.pi/3):
        stack = np.exp(1j * np.arange(3)[:, None, None] * slope) * np.ones((3, 17, 17))
        np.testing.assert_allclose(adjacent_phase_variance(stack, 5), 0, atol=1e-7)
    np.testing.assert_allclose(phase_variance(stack), 1, atol=1e-7)


@pytest.mark.parametrize("window", [1, 3, 5, 7, 9, 15])
def test_against_independent_fmnisar_formula(window):
    rng = np.random.default_rng(12)
    stack = rng.normal(size=(3, 31, 29)) + 1j*rng.normal(size=(3, 31, 29))
    products = stack[:-1]*stack[1:].conj()
    def boxcar(x):
        return np.lib.stride_tricks.sliding_window_view(x, (window, window)).mean((-1,-2))
    cv = np.stack([boxcar(p) for p in products])
    reference = 1 - np.abs(np.mean(cv / np.abs(cv), axis=0))
    actual = adjacent_phase_variance(stack, window, method="covariance")
    h = window//2
    np.testing.assert_allclose(actual[h:31-h or None, h:29-h or None], reference, atol=1e-7)
    raw = 1-np.abs(np.mean(products/np.abs(products), axis=0))
    np.testing.assert_allclose(adjacent_phase_variance(stack, window), uniform_filter(raw, window, mode="nearest"), atol=1e-7)


def test_invalid_and_degenerate_samples():
    with pytest.raises(ValueError, match="three"):
        adjacent_phase_variance(np.ones((2, 4, 4), complex))
    stack = np.ones((3, 9, 9), complex)
    stack[:, 4, 4] = 0
    assert np.isnan(adjacent_phase_variance(stack)[4,4])
    assert adjacent_phase_variance(stack, 3)[4,4] == 0
    stack[:] = np.nan
    assert np.isnan(adjacent_phase_variance(stack, 3)).all()


def test_multiwindow_hook_and_no_old_features():
    arrays = {f"{part}_VV_SA{k}": np.full((19, 19), value, np.float32)
              for k in (1,2,3) for part,value in (("i",1),("q",0))}
    out, attrs, names = add_subap_features(arrays, {}, SubapFeatureConfig(enabled=True, window_sizes=(1,5)))
    assert "subap_adjacent_phase_variance_VV_W5" in names
    assert {"subap_phase_diff_VV_pd12_W5", "subap_phase_diff_VV_pd23_W5"} <= set(names)
    assert not any(n.startswith("subap_covariance_") for n in names)
    assert not any(n.startswith("subap_phase_variance_") or "pd_mean" in n for n in names)
    assert attrs["subap_adjacent_phase_variance_VV_W5"]["window_size"] == 5
    assert attrs["subap_adjacent_phase_variance_VV_W5"]["estimator"] == "adjacent_covariance_circular_variance"
    assert not any(n.startswith("i_") for n in out)
