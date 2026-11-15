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


def test_fault_types_from_saved_scores_match_fresh_scoring(tmp_path, monkeypatch):
    """saved=True must give the same table without fitting anything again, under either calibration."""
    import pandas as pd

    from adlab import run
    monkeypatch.setattr(run, "CACHE", tmp_path)
    fit, X_link, y = _break_case()
    X_level = _break_case(seed=2)[1].copy()
    X_level[250:, 1] = X_level[250:, 0]
    X_level[250:, 2] += 6
    quiet = [Run(f"normal{i}", _break_case(seed=10 + i)[0], np.zeros(500, int), fit, "normal") for i in range(3)]
    ds = Dataset("toy", [Run("link", X_link, y, fit, "link break"), Run("level", X_level, y, fit, "level shift")],
                 ["a", "b", "c"], 1.0, val_normal=quiet, calibration="matched", matched_start=250)
    for key in ("rolling_z", "ewma", "cusum", "pca", "link_straight", "matrix_profile"):
        run.score_dataset(ds, key, seed=0)
    for calibration in ("matched", "pooled"):
        fresh = S.fault_types(ds, with_curved=False, calibration=calibration)
        saved = S.fault_types(ds, with_curved=False, saved=True, calibration=calibration)
        pd.testing.assert_frame_equal(fresh, saved)
    assert saved.loc["level shift", "kind"] == "single sensor"
    assert saved.loc["link break", "kind"] == "hidden from level checks"


def test_sensors_moved_follows_the_calibration():
    """A sensor that wanders in the later part of every run, fault or not, should not count as moved by the fault."""
    rng = np.random.default_rng(0)
    fit = rng.normal(size=(500, 2))
    y = (np.arange(300) >= 100).astype(int)

    def run_with_late_wander():
        x = rng.normal(size=(300, 2))
        x[100:, 0] += 3            # sensor a sits higher late in every run: that is normal here
        return x

    quiet = [Run(f"n{i}", run_with_late_wander(), np.zeros(300, int), fit, "normal") for i in range(5)]
    ds = Dataset("toy", [Run("r", run_with_late_wander(), y, fit, "nothing wrong")], ["a", "b"], 1.0,
                 val_normal=quiet, calibration="matched", matched_start=100)
    assert S.sensors_moved(ds) == [0]
    assert S.sensors_moved(ds, calibration="pooled") == [1]
