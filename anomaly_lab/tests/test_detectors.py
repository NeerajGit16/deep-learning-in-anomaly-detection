import numpy as np
import pytest

from adlab import detectors as det
from adlab.metrics import dr_at_far

FAST = ["rolling_z", "ewma", "cusum", "pca", "iforest"]


def _step_case(seed=0):
    rng = np.random.default_rng(seed)
    fit = rng.normal(size=(400, 4))
    X = rng.normal(size=(600, 4))
    X[300:, 2] += 4.0  # one channel's level jumps by 4 standard deviations
    y = np.zeros(600, int)
    y[300:] = 1
    return fit, X, y


@pytest.mark.parametrize("key", FAST)
def test_single_channel_step_is_detected(key):
    fit, X, y = _step_case()
    s = det.make(key).fit(fit).score(X)
    assert dr_at_far([s], [y], 0.01)["dr"] > 0.5


@pytest.mark.parametrize("key", FAST)
def test_scores_are_causal(key):
    """Changing the future must not change the score at earlier times."""
    fit, X, _ = _step_case()
    model = det.make(key).fit(fit)
    a = model.score(X)
    X2 = X.copy()
    X2[400:] += 10.0
    b = model.score(X2)
    np.testing.assert_allclose(a[:400], b[:400])


def test_pca_sees_a_correlation_break_that_per_channel_stats_miss():
    """Two sensors that normally move together start moving in opposite directions,
    each staying inside its usual range (H2 in miniature)."""
    rng = np.random.default_rng(1)
    t = rng.normal(size=1000)
    fit = np.c_[t[:500], t[:500] + 0.1 * rng.normal(size=500)]
    X = np.c_[t[500:], t[500:] + 0.1 * rng.normal(size=500)]
    X[250:, 1] = -X[250:, 0] + 0.1 * rng.normal(size=250)
    y = np.zeros(500, int)
    y[250:] = 1
    pca = dr_at_far([det.make("pca", var=0.5).fit(fit).score(X)], [y], 0.01)["dr"]
    z = dr_at_far([det.make("rolling_z", window=1).fit(fit).score(X)], [y], 0.01)["dr"]
    assert pca > 0.8
    assert z < 0.2


def test_constant_channel_does_not_break_standardising():
    rng = np.random.default_rng(2)
    fit = np.c_[rng.normal(size=200), np.ones(200)]
    X = np.c_[rng.normal(size=50), np.ones(50)]
    for key in FAST:
        assert np.all(np.isfinite(det.make(key).fit(fit).score(X)))


def test_matrix_profile_flags_a_level_shift():
    pytest.importorskip("stumpy")
    fit, X, y = _step_case()
    s = det.make("matrix_profile", m=8).fit(fit).score(X)
    assert dr_at_far([s], [y], 0.01)["dr"] > 0.5
