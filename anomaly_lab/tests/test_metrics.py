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


def _faults_that_differ(seed=4, kinds=4, runs_each=10):
    """Runs in clusters: how easy a fault is to catch depends on its kind, not on the run."""
    rng = np.random.default_rng(seed)
    scores, labels, kind = [], [], []
    for k in range(kinds):
        for _ in range(runs_each):
            y = np.r_[np.zeros(200), np.ones(100)].astype(int)
            scores.append(rng.normal(size=300) + (1.0 + 1.5 * k) * y)
            labels.append(y)
            kind.append(f"kind {k}")
    return scores, labels, kind


def test_cluster_resampling_draws_whole_clusters():
    from adlab.stats import resample_indices
    clusters = ["a"] * 3 + ["b"] * 5 + ["c"] * 2
    for idx in resample_indices(10, 50, np.random.default_rng(0), clusters):
        drawn = np.bincount(idx, minlength=10)
        # every run of a cluster is drawn the same number of times
        assert len(set(drawn[:3])) == 1 and len(set(drawn[3:8])) == 1 and len(set(drawn[8:])) == 1
    plain = resample_indices(10, 50, np.random.default_rng(0))
    assert all(len(idx) == 10 for idx in plain)


def test_resampling_kinds_of_fault_gives_a_wider_range_than_resampling_runs():
    scores, labels, kind = _faults_that_differ()
    by_run = bootstrap_dr({"m": scores}, labels, 0.01, n_boot=400)["m"]
    by_kind = bootstrap_dr({"m": scores}, labels, 0.01, n_boot=400, clusters=kind)["m"]
    assert np.std(by_kind) > 2 * np.std(by_run)


def _drifting_runs(rng, n, length=400):
    """Runs with no fault in which three sensors wander slowly away from where they started."""
    return [np.cumsum(0.05 * rng.normal(size=(length, 3)), axis=0) + rng.normal(size=(length, 3)) for _ in range(n)]


def test_matched_window_calibration_keeps_cusum_honest_on_drifting_runs():
    """A CUSUM chart adds small deviations up, so on runs that wander its score climbs through
    every run, fault or no fault. Nothing is wrong in any of these runs, so every alarm is false.

    First-window calibration puts the alarm line where 1 in 100 of each run's first quarter
    crosses it, then counts alarms in the rest: far more than 1 in 100.
    Matched-window calibration puts the line where 1 in 100 readings cross it in the same later
    stretch of separate fault-free runs: about 1 in 100 again.
    """
    from adlab import detectors as det
    rng = np.random.default_rng(0)
    model = det.make("cusum").fit(rng.normal(size=(2000, 3)))
    later = (np.arange(400) >= 100).astype(int)                    # where a fault would be; nothing happens
    scored = [model.score(x) for x in _drifting_runs(rng, 200)]
    separate = [model.score(x)[100:] for x in _drifting_runs(rng, 200)]
    labels = [later] * len(scored)
    first_window = M.dr_at_far(scored, labels, 0.01, "pooled")
    matched = M.dr_at_far(scored, labels, 0.01, "matched", separate)
    assert first_window["far"] <= 0.011 and first_window["dr"] > 0.25      # "1 in 100" on paper, many in practice
    assert 0.004 < matched["dr"] < 0.025                                    # about 1 in 100
    assert matched["far_first_window"] < 0.001                              # the honest line is far above the early scores


def test_matched_calibration_needs_fault_free_scores_and_resamples_them():
    rng = np.random.default_rng(1)
    labels = [np.r_[np.zeros(100), np.ones(100)].astype(int) for _ in range(12)]
    scores = [rng.normal(size=200) + 3 * y for y in labels]
    calib = [rng.normal(size=100) for _ in range(30)]
    try:
        M.dr_at_far(scores, labels, 0.01, "matched")
        raise AssertionError("expected an error without calibration scores")
    except ValueError:
        pass
    point = point_dr({"m": scores}, labels, 0.01, "matched", {"m": calib})["m"]
    assert point == M.dr_at_far(scores, labels, 0.01, "matched", calib)["dr"]
    many = bootstrap_dr({"m": scores}, labels, 0.01, n_boot=300, mode="matched", calib={"m": calib})["m"]
    one = bootstrap_dr({"m": scores}, labels, 0.01, n_boot=300, mode="matched", calib={"m": [np.concatenate(calib)]})["m"]
    assert many.min() <= point <= many.max()
    assert np.std(many) > np.std(one)          # one calibration run cannot be resampled: its line never moves


def test_alarm_counts_match_a_plain_count():
    from adlab.stats import AlarmCounts
    rng = np.random.default_rng(2)
    samples = [rng.normal(size=n) for n in (5, 50, 0, 17)]
    counts = AlarmCounts(samples)
    for thr in (-1.0, 0.0, 0.7):
        assert list(counts.above(thr)) == [float((s > thr).sum()) for s in samples]
    assert counts.rates(0.0)[2] == 0.0          # an empty run gives 0, not an error
