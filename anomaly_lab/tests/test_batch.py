"""Batch processes (adlab/batch.py) and the stage-told link detector. No data needed."""
import numpy as np
import pandas as pd

from adlab import batch, progress, structure
from adlab import detectors as det


def test_rounds_hold_out_every_group_and_never_put_a_batch_on_both_sides():
    groups = ["a"] * 34 + ["b"] * 36 + ["c"] * 37
    rounds = batch.staggered_rounds(groups)
    assert len(rounds) == len(groups) and set(rounds) == set(range(5))
    sizes = np.bincount(rounds)
    assert sizes.max() - sizes.min() <= 1                       # the count carries on, so rounds are even
    g = np.array(groups)
    for k in range(5):
        held, fitted = set(np.flatnonzero(rounds == k)), set(np.flatnonzero(rounds != k))
        assert not held & fitted and len(held | fitted) == len(groups)
        for name in "abc":                                       # every group on both sides in every round
            assert (g[rounds == k] == name).any() and (g[rounds != k] == name).any()
    t = batch.round_table(rounds, groups, "experiment")
    assert (t["experiment a: held out"] + t["experiment a: fitted on"] == 34).all()


def test_stage_parts_and_folds_follow_their_batches():
    runs = [np.arange(10.0)[:, None] + i for i in range(3)]
    stages = [np.r_[np.full(4, 1), np.full(6, 2)]] * 3
    parts, kept = batch.stage_parts(runs, stages, np.array([0, 1, 2]), 2)
    assert [len(p) for p in parts] == [6, 6, 6] and list(kept) == [0, 1, 2]
    folds = batch.as_folds(parts, kept)
    assert [set(f) for f in folds] == [{0}, {1}, {2}]


def test_between_share_is_one_for_pure_level_shifts_and_zero_for_none():
    rng = np.random.default_rng(0)
    stage = np.repeat([0, 1, 2], 200)
    X = np.column_stack([stage * 10.0 + 0.001 * rng.normal(size=600), rng.normal(size=600), np.ones(600)])
    share = structure.between_share(X, stage)
    assert share[0] > 0.99 and share[1] < 0.02 and np.isnan(share[2])


def test_stage_told_link_detector_takes_the_stage_out_of_the_errors():
    """A sensor whose level jumps between stages, while the others do not: a straight model that is
    not told the stage misses the jump; one told the stage does not. The stage column itself is
    never predicted or scored."""
    rng = np.random.default_rng(1)
    n = 1200
    stage = np.repeat([4, 5], n // 2)
    a, b = rng.normal(size=n), rng.normal(size=n)
    c = a + b + 3.0 * (stage == 5) + 0.1 * rng.normal(size=n)
    X = np.column_stack([a, b, c])
    plain = det.make("link_straight").fit(X)
    told = det.make("link_straight_told").fit(np.column_stack([X, stage]))
    err_plain = np.abs(plain.errors(X)[:, 2]).mean()
    err_told = told.errors(np.column_stack([X, stage]))
    assert err_told.shape == (n, 3)
    assert np.abs(err_told[:, 2]).mean() < 0.5 * err_plain
    assert told.score(np.column_stack([X, stage])).shape == (n,)
    assert det.REGISTRY["link_straight_told"].name == "Straight-link residual, stage told"


def test_report_with_lines_per_group():
    """Two groups whose normal scores sit at different levels. A line per group catches the faults
    of the low group that one shared line would miss, and false alarms on check batches stay near 1%."""
    rng = np.random.default_rng(2)
    normal_groups = ["x"] * 20 + ["y"] * 20
    roles = ["line", "check"] * 20
    level = {"x": 0.0, "y": 5.0}
    normal = [rng.normal(level[g], 1.0, 200) for g in normal_groups]
    fault_groups = ["x"] * 4 + ["y"] * 4
    faults = [rng.normal(level[g] + 4.0, 1.0, 100) for g in fault_groups]
    kinds = ["k1", "k2"] * 4
    per_method = {"A": [(faults, normal)], "B": [([f - 1.0 for f in faults], normal)]}
    each = batch.report(per_method, kinds, fault_groups, normal_groups, roles, per_group=True, n_boot=200, verbose=False)
    one = batch.report(per_method, kinds, fault_groups, normal_groups, roles, per_group=False, n_boot=200, verbose=False)
    a_each = next(r for r in each["summary"] if r["method"] == "A")
    a_one = next(r for r in one["summary"] if r["method"] == "A")
    assert a_each["dr_at_1pct"] > a_one["dr_at_1pct"] + 0.2
    assert abs(a_each["far_check"] - 0.01) < 0.01 and a_each["dr_ci90_low"] <= a_each["dr_at_1pct"] <= a_each["dr_ci90_high"]
    assert {p["method_a"] for p in each["pairs"]} == {"A"} and each["pairs"][0]["gap"] > 0
    assert set(pd.DataFrame(each["by_group"])["group"]) == {"k1", "k2"}
    table = batch.check_table(each)
    assert list(table.columns) == ["experiment x", "experiment y"]


def test_fault_kind_labels_follow_the_old_rule():
    t = pd.DataFrame({"level checks": [0.9, 0.9, 0.1, 0.1], "straight links": [0.0, 0.0, 0.8, 0.1],
                      "sensors moved": [1, 6, 0, 0]}, index=list("abcd"))
    assert structure.kind_of_fault(t) == ["single sensor", "many sensors", "hidden from level checks", "hard for all"]


def test_stay_awake_counts_nested_loops():
    before = progress._awake
    progress.stay_awake(True)
    progress.stay_awake(True)
    progress.stay_awake(False)
    assert progress._awake == before + 1
    progress.stay_awake(False)
    progress.stay_awake(False)                                   # an extra release never goes below zero
    assert progress._awake == max(0, before - 1) or progress._awake == 0
