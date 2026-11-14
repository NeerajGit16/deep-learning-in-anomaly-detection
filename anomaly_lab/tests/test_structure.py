import numpy as np

from adlab import structure as S
from adlab.data import Dataset, Run
from adlab.metrics import dr_at_far
from adlab.simulate import linked_sensors


def test_curvature_measure_follows_the_dial():
    names = [f"s{i}" for i in range(6)]
    straight = S.curvature_summary(linked_sensors(800, 6, curve=0.0, seed=0), names, n_boot=50)
    curved = S.curvature_summary(linked_sensors(800, 6, curve=1.0, seed=0), names, n_boot=50)
    assert abs(straight["curved"]) < 0.03
    assert curved["curved"] > 0.10 and curved["curved_low"] > 0.03


def _break_case(seed=1):
    rng = np.random.default_rng(seed)
    t = rng.normal(size=900)
    X = np.c_[t, t + 0.1 * rng.normal(size=900), rng.normal(size=900)]
    y = np.zeros(900, int)
    y[750:] = 1
    X[750:, 1] = -X[750:, 0] + 0.1 * rng.normal(size=150)   # link flips, each sensor stays in range
    return X[:500], X[500:], y[500:]


def test_link_residual_catches_a_broken_link_and_is_causal():
    fit, X, y = _break_case()
    model = S.LinkResidual(curved=False).fit(fit)
    s = model.score(X)
    assert dr_at_far([s], [y], 0.01)["dr"] > 0.8
    X2 = X.copy()
    X2[300:] += 5
    np.testing.assert_allclose(s[:300], model.score(X2)[:300])


def test_fault_types_separates_a_level_fault_from_a_link_fault():
    fit, X_link, y = _break_case()
    X_level = _break_case(seed=2)[1].copy()
    y2 = np.zeros(len(X_level), int)
    y2[250:] = 1
    X_level[:250, 1] = X_level[:250, 0] + 0.1 * np.random.default_rng(3).normal(size=250)
    X_level[250:, 1] = X_level[250:, 0]        # restore the link, then shift an unlinked sensor
    X_level[250:, 2] += 6
    ds = Dataset("toy", [Run("link", X_link, y, fit, "link break"), Run("level", X_level, y2, fit, "level shift")],
                 ["a", "b", "c"], 1.0)
    table = S.fault_types(ds, with_curved=False)
    assert table.loc["level shift", "kind"] == "single sensor"
    assert table.loc["link break", "kind"] == "hidden from level checks"
