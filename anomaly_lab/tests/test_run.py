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
