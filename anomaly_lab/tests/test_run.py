"""The pilot runner, on a toy dataset with its cache in a temporary folder."""
import numpy as np
import pytest

from adlab import detectors as det
from adlab import metrics as M
from adlab import run
from adlab.data import Dataset, Run


def _toy(seed=0, runs=4):
    """Two linked sensors and one unlinked. In half the runs the link flips; in the rest one sensor jumps."""
    rng = np.random.default_rng(seed)

    def series(n):
        t = rng.normal(size=n)
        return np.c_[t, t + 0.1 * rng.normal(size=n), rng.normal(size=n)]

    fit = series(400)
    out = []
    for i in range(runs):
        X, y = series(300), np.zeros(300, int)
        y[200:] = 1
        if i % 2:
            X[200:, 1] = -X[200:, 0] + 0.1 * rng.normal(size=100)
            group = "link break"
        else:
            X[200:, 2] += 6
            group = "level shift"
        out.append(Run(f"run{i}", X, y, fit, group))
    return Dataset("toy", out, ["a", "b", "c"], 1.0)


def test_every_method_in_the_pilot_is_a_registered_detector():
    assert set(run.METHODS) <= set(det.REGISTRY)
    assert {"link_straight", "link_curved"} <= set(run.METHODS)
    for key, settings in run.METHODS.items():
        det.make(key, **settings)          # the settings are ones the detector accepts


def test_link_residuals_go_through_scoring_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CACHE", tmp_path)
    ds = _toy()
    methods = ["rolling_z", "link_straight", "link_curved"]
    for m in methods:
        path = run.score_dataset(ds, m, seed=0)
        assert path.parent == tmp_path and path.exists()
        scores, val, _ = run.load_scores(ds.name, m, 0)
        assert [len(s) for s in scores] == [len(r.y) for r in ds.runs] and val is None
    res = run.report_dataset(ds, methods, n_seeds=5, n_boot=50)
    table = {row["method"]: row for row in res["summary"]}
    assert set(table) == {"Rolling z-score", "Straight-link residual", "Curved-link residual"}
    assert all(row["seeds"] == 1 for row in table.values())
    assert all(row["dr_ci90_low"] <= row["dr_at_1pct"] <= row["dr_ci90_high"] for row in table.values())
    by_group = {(r["method"], r["group"]): r["dr_at_1pct"] for r in res["by_group"]}
    # the flipped link is invisible to a per-sensor check and obvious to a link check
    assert by_group[("Straight-link residual", "link break")] > 0.8
    assert by_group[("Rolling z-score", "link break")] < 0.3
    assert len(res["pairs"]) == 3
    # the margin is a fixed constant; the measured wobble is reported next to it but does not set it
    assert res["noise"]["margin"] == run.MARGIN == 0.12
    assert res["noise"]["median_pair_sd"] > 0
    # VUS-PR appears only when the optional TSB-AD package is installed, never as "nan"
    report = run._markdown({"toy": res})
    assert ("VUS-PR" in report) == M.vus_available() == ("vus_pr_mean" in res["summary"][0])
    assert "nan" not in report and "0.12" in report


def test_report_on_part_of_a_dataset_and_with_whole_kinds_resampled(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CACHE", tmp_path)
    ds = _toy(runs=8)
    methods = ["rolling_z", "link_straight"]
    for m in methods:
        run.score_dataset(ds, m, seed=0)
    full = run.report_dataset(ds, methods, 1, 100)
    by_kind = run.report_dataset(ds, methods, 1, 100, resample="groups")
    part = run.report_dataset(ds, methods, 1, 100, keep=[r.group == "level shift" for r in ds.runs])

    def rate(res, name):
        return next(r["dr_at_1pct"] for r in res["summary"] if r["method"] == name)

    assert full["runs"] == by_kind["runs"] == 8 and part["runs"] == 4
    assert rate(full, "Rolling z-score") == rate(by_kind, "Rolling z-score")       # same number, different range
    assert {r["group"] for r in part["by_group"]} == {"level shift"}
    assert rate(part, "Rolling z-score") > rate(full, "Rolling z-score")            # the level shifts are its easy half
    assert run.by_fault(part, "Rolling z-score").index.tolist() == ["level shift"]
    assert run.by_fault(full, "Rolling z-score")["link break"] < 0.3
    text = run.rates(full)
    assert list(text.index)[0] == "Straight-link residual" and "(" in text.iloc[0]
    per_run = run.per_run_rates(ds, "rolling_z", group="link break")
    assert len(per_run) == 4 and per_run.max() < 0.5


def _drifting_toy(seed=0, numbers=12):
    """Every run wanders slowly, fault or not. Faults: one that does nothing, one clear jump.
    Returns the scored runs, their no-fault twins, both judged on separate fault-free runs."""
    rng = np.random.default_rng(seed)
    fit = rng.normal(size=(500, 3))
    y = (np.arange(400) >= 100).astype(int)

    def base():
        return rng.normal(size=(400, 3)) + 0.002 * np.arange(400)[:, None]

    separate = Dataset("toy_cal", [Run(f"cal/{i}", base(), y, fit, "no fault", {"run": 100 + i}) for i in range(30)],
                       list("abc"), 1.0, matched_start=100)
    faulty, twins = [], []
    for r in range(numbers):
        x = base()
        twins.append(Run(f"nofault/{r}", x, y, fit, "no fault", {"run": r}))
        faulty.append(Run(f"nothing/{r}", x.copy(), y, fit, "does nothing", {"run": r}))
        jump = x.copy()
        jump[100:, 0] += 8
        faulty.append(Run(f"jump/{r}", jump, y, fit, "jump", {"run": r}))
    kw = dict(calibration="matched", matched_start=100, matched_from=separate)
    return Dataset("toy", faulty, list("abc"), 1.0, **kw), Dataset("toy_twins", twins, list("abc"), 1.0, **kw), separate


def test_excess_alarms_over_the_twin(tmp_path, monkeypatch):
    """A CUSUM chart's score rises with time in these runs. On the first-window line it 'catches'
    a fault that does nothing. The matched line and the twins both show there is nothing there."""
    monkeypatch.setattr(run, "CACHE", tmp_path)
    ds, twins, separate = _drifting_toy()
    for d in (ds, twins, separate):
        run.score_dataset(d, "cusum", seed=0)

    old = run.report_dataset(ds, ["cusum"], 1, 50, calibration="pooled")
    new = run.report_dataset(ds, ["cusum"], 1, 50)
    nothing = lambda res: next(r["dr_at_1pct"] for r in res["by_group"] if r["group"] == "does nothing")   # noqa: E731
    assert old["calibration"] == "pooled" and new["calibration"] == "matched"
    # the report under the new rule carries the old-rule number alongside, and the written report shows it
    assert np.isclose(new["summary"][0]["dr_old_rule"], old["summary"][0]["dr_at_1pct"])
    assert "dr_old_rule" not in old["summary"][0]
    assert "Old rule (first 8 hours)" in run._markdown({"toy": new})
    assert new["calibration_runs"] == 30 and "set on 30 separate fault-free run(s)" in run._markdown({"toy": new})
    assert "Old rule" not in run._markdown({"toy": old})
    assert nothing(old) > 0.1 and nothing(new) < 0.05
    assert old["summary"][0]["far_late_fault_free"] > 0.1                   # what the old line lets through later
    assert abs(new["summary"][0]["far_late_fault_free"] - 0.01) < 0.002     # the matched line: 1 in 100 by construction
    late = run.late_false_alarms(ds, ["cusum"], n_boot=100)
    assert np.isclose(late.loc["CUSUM chart", "false alarms later, old line"], old["summary"][0]["far_late_fault_free"])
    assert late.loc["CUSUM chart", "low"] <= late.iloc[0, 0] <= late.loc["CUSUM chart", "high"]

    table = run.excess_over_twin(ds, twins, "cusum", n_boot=200)
    assert abs(table.loc["does nothing", "excess alarms over the twin"]) < 1e-12
    assert table.loc["does nothing", "low"] == table.loc["does nothing", "high"] == 0
    assert table.loc["jump", "excess alarms over the twin"] > 0.9 and table.loc["jump", "low"] > 0.8
    assert table.loc["jump", "alarms in twins"] < 0.1
    overall = run.excess_summary(ds, twins, ["cusum"], n_boot=200)
    row = overall.loc["CUSUM chart"]
    assert np.isclose(row["excess alarms over the twin"], table["excess alarms over the twin"].mean())
    assert row["low"] <= row["excess alarms over the twin"] <= row["high"]
    assert run.per_run_rates(ds, "cusum", "jump").min() > 0.9


def test_quick_score_uses_the_datasets_own_calibration():
    ds, _, _ = _drifting_toy(numbers=6)
    matched = run.quick_score(ds, lambda: det.make("cusum"))
    ds.calibration = "pooled"
    first_window = run.quick_score(ds, lambda: det.make("cusum"))
    assert first_window["caught"] > matched["caught"]            # the jump is caught either way; the drift only by the old line
    assert 0.45 < matched["caught"] < 0.56                       # half the runs have the jump, half have nothing


def test_changing_a_setting_gives_a_new_cache_file(monkeypatch):
    before = run._cache_path("toy", "link_straight", 0)
    monkeypatch.setitem(run.METHODS, "link_straight", {"window": 9})
    assert run._cache_path("toy", "link_straight", 0) != before


def test_gap_table_flips_signs_and_applies_the_fixed_margin():
    res = {"pairs": [
        {"method_a": "A", "method_b": "B", "gap": 0.20, "sd": 0.03, "ci90_low": 0.15, "ci90_high": 0.25},
        {"method_a": "A", "method_b": "C", "gap": 0.05, "sd": 0.01, "ci90_low": 0.03, "ci90_high": 0.07},
        {"method_a": "D", "method_b": "A", "gap": 0.30, "sd": 0.20, "ci90_low": -0.02, "ci90_high": 0.60},
        {"method_a": "B", "method_b": "C", "gap": -0.15, "sd": 0.01, "ci90_low": -0.17, "ci90_high": -0.13},
    ]}
    t = run.gap_table(res, "A")
    assert list(t.index) == ["B", "C", "D"]
    assert t.loc["B", "verdict"] == "ahead"                     # 20 points, range clear of zero
    assert t.loc["C", "verdict"] == "tie"                       # real but under the 12-point margin
    assert t.loc["D", "verdict"] == "tie"                       # big gap, but the range includes zero
    assert t.loc["D", "gap"] == -0.30 and (t.loc["D", "low"], t.loc["D", "high"]) == (-0.60, 0.02)
    b = run.gap_table(res, "B", ["A"])
    assert b.loc["A", "verdict"] == "behind" and b.loc["A", "gap"] == -0.20 and len(b) == 1
    res["pairs"] += [{"method_a": "B", "method_b": "D", "gap": 0.0, "sd": 0.1, "ci90_low": -0.1, "ci90_high": 0.1},
                     {"method_a": "C", "method_b": "D", "gap": 0.0, "sd": 0.1, "ci90_low": -0.1, "ci90_high": 0.1}]
    res["summary"] = [{"method": m, "dr_at_1pct": v} for m, v in (("B", 0.5), ("A", 0.7), ("D", 0.9), ("C", 0.65))]
    v = run.verdict_table(res)
    assert list(v.index) == list(v.columns) == ["D", "A", "C", "B"]             # best first
    assert v.loc["A", "B"] == "ahead" and v.loc["B", "A"] == "behind" and v.loc["A", "C"] == "tie"
    assert v.loc["B", "C"] == "behind" and v.loc["C", "B"] == "ahead" and v.loc["A", "A"] == ""
