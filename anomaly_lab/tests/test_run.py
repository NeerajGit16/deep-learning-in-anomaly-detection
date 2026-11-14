"""The pilot runner, on a toy dataset with its cache in a temporary folder."""
import numpy as np
import pytest

from adlab import detectors as det
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


def test_changing_a_setting_gives_a_new_cache_file(monkeypatch):
    before = run._cache_path("toy", "link_straight", 0)
    monkeypatch.setitem(run.METHODS, "link_straight", {"window": 9})
    assert run._cache_path("toy", "link_straight", 0) != before
