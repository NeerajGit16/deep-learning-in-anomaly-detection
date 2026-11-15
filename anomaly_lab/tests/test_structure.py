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


# ---------------------------------------------------------------- delayed links (lags) and runs

def test_a_delayed_straight_link_shows_only_when_past_readings_are_offered():
    from adlab.simulate import delayed_sensors
    X = delayed_sensors(900, delay=3, curve=0.0, seed=0)
    now = S.curvature_summary(X, list("abc"), k=3, n_boot=20)["table"]
    with_past = S.curvature_summary(X, list("abc"), k=3, n_boot=20, lags=(1, 2, 3))["table"]
    assert abs(now.loc["b", "straight"]) < 0.05                       # looks unlinked at the same moment
    assert with_past.loc["b", "straight"] > 0.85                      # 1 / (1 + 0.3^2) = 0.92 at best
    assert abs(with_past.loc["b", "curved"]) < 0.03                   # and the delayed link is straight
    assert abs(with_past.loc["a", "straight"]) < 0.05                 # the future of b does not explain a


def test_a_delayed_curved_link_shows_as_curved_share_only_with_the_right_lag():
    from adlab.simulate import delayed_sensors
    X = delayed_sensors(900, delay=2, curve=1.0, seed=1)
    wrong = S.curvature_summary(X, list("abc"), k=3, n_boot=20, lags=(1,))["table"]
    right = S.curvature_summary(X, list("abc"), k=3, n_boot=20, lags=(1, 2))["table"]
    assert abs(wrong.loc["b", "curved"]) < 0.05 and abs(wrong.loc["b", "straight"]) < 0.05
    assert right.loc["b", "curved"] > 0.6 and abs(right.loc["b", "straight"]) < 0.1


def test_lagged_features_use_only_the_other_sensors_and_stay_inside_a_run():
    Z = np.arange(12, dtype=float).reshape(6, 2)                      # readings 0..5 of sensors (even, odd)
    y, F = S._lagged(Z, (1, 3))
    assert y.shape == (3, 2) and F.shape == (3, 6)
    assert np.array_equal(y[0], Z[3]) and np.array_equal(F[0], np.r_[Z[3], Z[2], Z[0]])
    # two runs: sensor b copies sensor a one reading later, but only inside each run
    rng = np.random.default_rng(0)
    runs = []
    for _ in range(6):
        a = rng.normal(size=200)
        runs.append(np.c_[a, np.r_[rng.normal(), a[:-1]] + 0.1 * rng.normal(size=200)])
    h = S.held_out_by_run(runs, k=3, lags=(1,))
    assert len(h["Z"]) == 6 * 199 and set(h["run"]) == set(range(6))  # the first reading of every run is dropped
    assert S._r2(h["Z"], h["straight"])[1] > 0.95


def test_no_lags_gives_the_same_numbers_as_before_the_lag_option():
    """Single recording, no lags: the by-run code path must reproduce the plain one exactly."""
    X = linked_sensors(400, 5, curve=0.5, seed=3)
    Z, lin, both = S.held_out_predictions(X)
    mu, sd = X.mean(0), X.std(0)
    z = (X - mu) / sd
    test = np.arange(80)                                              # first of five blocks
    train = np.arange(80, 400)
    m = S._straight().fit(z[train][:, 1:], z[train, 0])
    np.testing.assert_allclose(lin[test, 0], m.predict(z[test][:, 1:]))
    assert np.array_equal(Z, z)


def test_curvature_by_run_resamples_runs_and_reports_every_sensor():
    runs = np.split(linked_sensors(1800, 5, curve=1.0, seed=0), 6)       # six runs with the same links
    names = [f"s{i}" for i in range(5)]
    out = S.curvature_by_run(runs, names, k=3, n_boot=100)
    assert out["runs"] == 6 and out["readings"] == 1800
    assert out["curved"] > 0.08 and out["curved_low"] > 0.03 and out["curved_low"] <= out["curved"] <= out["curved_high"]
    table = out["table"]
    assert list(table.columns) == ["straight", "curved", "curved_low", "curved_high"] and list(table.index) == names
    assert (table["curved_low"] <= table["curved"] + 1e-12).all() and (table["curved"] <= table["curved_high"] + 1e-12).all()
    assert out["sensors_with_curved_over_10pts"] == int((table["curved"] > 0.10).sum()) >= 2
    straight = S.curvature_by_run(np.split(linked_sensors(1800, 5, curve=0.0, seed=0), 6), names, k=3, n_boot=100)
    assert abs(straight["curved"]) < 0.03 and straight["sensors_with_curved_over_10pts"] == 0
    by_lag = S.curvature_by_lag(runs[:2], names[:3] + names[3:], {"now": (), "one back": (1,)}, k=2, n_boot=20)
    t = S.lag_table(by_lag)
    assert list(t.index) == ["now", "one back"] and t["sensors gaining over 0.10 from curves"].dtype.kind == "i"


def _two_setting_runs(seed=0, runs_per_setting=4, n=300):
    """Runs recorded at two settings. Sensor b follows sensor a at one setting and mirrors it at
    the other; sensor c shows which setting the machine is at. Within a setting the link is straight."""
    rng = np.random.default_rng(seed)
    runs, setting = [], []
    for sign in (+1, -1):
        for _ in range(runs_per_setting):
            a = rng.normal(size=n)
            runs.append(np.c_[a, sign * a + 0.2 * rng.normal(size=n), 2.0 * sign + 0.3 * rng.normal(size=n)])
            setting.append(sign)
    return runs, setting


def test_links_that_depend_on_the_setting_show_as_curved_share_when_runs_are_pooled():
    """Pooled over both settings, no one straight line links a and b (they cancel), but a flexible
    model that also sees the setting sensor predicts b well: a mode-dependent link counts as
    curved. Each setting alone is straight."""
    runs, setting = _two_setting_runs()
    pooled = S.curvature_by_run(runs, list("abc"), k=3, n_boot=200)
    assert pooled["table"].loc["b", "straight"] < 0.1
    assert pooled["table"].loc["b", "curved"] > 0.6 and pooled["table"].loc["b", "curved_low"] > 0.5
    assert pooled["curved_low"] > 0.03                                   # the average over the three sensors
    one = S.curvature_by_run([r for r, s in zip(runs, setting) if s > 0], list("abc"), k=3, n_boot=200)
    assert one["table"].loc["b", "straight"] > 0.9 and abs(one["table"].loc["b", "curved"]) < 0.03


def test_movement_by_run_names_the_run_that_holds_most_of_a_sensors_movement():
    """Five runs alike, except sensor b swings 30 times wider in run "d": that run holds nearly
    all of b's movement, while sensor a's movement is spread evenly."""
    rng = np.random.default_rng(0)
    runs = [rng.normal(size=(200, 2)) for _ in range(5)]
    runs[3][:, 1] *= 30
    out = S.movement_by_run(runs, ["a", "b"], names=list("abcde"))
    assert out.loc["b", "largest share"] > 0.99 and out.loc["b", "run"] == "d"
    assert out.loc["a", "largest share"] < 0.3 and out.loc["a", "even share"] == 0.2


def test_lag_check_refuses_a_delay_that_is_not_among_the_lags_offered():
    """With a delay of 3 and lags 1, 2 and 5 on offer the check would show nothing at every row."""
    import pytest
    with pytest.raises(ValueError):
        S.lag_check(delay=3)
