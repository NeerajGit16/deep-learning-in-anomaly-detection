import numpy as np

from adlab import metrics as M
from adlab.stats import bootstrap_dr, paired_differences, point_dr


def test_perfect_scores_catch_everything():
    y = np.r_[np.zeros(900), np.ones(100)].astype(int)
    out = M.dr_at_far([y.astype(float)], [y], 0.01)
    assert out["dr"] == 1.0 and out["far"] <= 0.01


def test_random_scores_catch_about_the_false_alarm_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.3).astype(int)
    out = M.dr_at_far([rng.random(len(y))], [y], 0.05)
    assert abs(out["dr"] - 0.05) < 0.01


def test_detection_rate_ignores_the_anomaly_share():
    """Same detector quality, different share of anomalies: DR at fixed FAR stays put."""
    rng = np.random.default_rng(1)
    rates = []
    for share in (0.02, 0.35):
        y = (rng.random(200_000) < share).astype(int)
        s = rng.normal(size=len(y)) + 2.0 * y
        rates.append(M.dr_at_far([s], [y], 0.01)["dr"])
    assert abs(rates[0] - rates[1]) < 0.02


def test_segments_and_delays():
    y = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    assert M.segments(y) == [(1, 4), (6, 8)]
    s = np.array([0, 0, 0, 5, 0, 0, 0, 0, 0], float)
    d = M.detection_delays(s, y, 1.0)
    assert d[0] == 2.0 and np.isnan(d[1])


def test_deployable_leave_one_run_out_uses_other_runs():
    rng = np.random.default_rng(2)
    runs, labels = [], []
    for _ in range(5):
        y = np.r_[np.zeros(500), np.ones(100)].astype(int)
        runs.append(rng.normal(size=600) + 4.0 * y)
        labels.append(y)
    out = M.deployable(runs, labels, 0.01)
    assert out["far_realised"] < 0.03 and out["dr"] > 0.9 and out["events_flagged"] == 1.0


def test_normalise_uses_fit_quantile():
    fit = np.arange(101, dtype=float)
    assert np.isclose(M.normalise(np.array([99.0]), fit, 0.99)[0], 1.0)


def test_bootstrap_pairs_are_paired_and_sized():
    rng = np.random.default_rng(3)
    labels = [np.r_[np.zeros(200), np.ones(50)].astype(int) for _ in range(10)]
    a = [rng.normal(size=250) + 3 * y for y in labels]
    b = [rng.normal(size=250) + 1 * y for y in labels]
    draws = bootstrap_dr({"a": a, "b": b}, labels, 0.01, n_boot=200)
    rows = paired_differences(draws, point_dr({"a": a, "b": b}, labels, 0.01))
    assert len(rows) == 1 and rows[0]["gap"] > 0.3 and rows[0]["ci90_low"] > 0
