"""Round 4, parts 3 and 5: false alarms before the fault, and the repeat share (adlab/round4.py)."""
import numpy as np

from adlab import posctl, round4


def test_false_alarms_before_the_fault_and_each_runs_own_line():
    rng = np.random.default_rng(0)
    scores, labels = [], []
    for _ in range(10):
        s = rng.normal(size=300)
        s[200:] += 3.0                                         # the fault
        scores.append(s)
        labels.append((np.arange(300) >= 200).astype(int))
    out = round4._carry_one(scores, labels, thr=1.0)           # a line far too low for these runs
    assert out.shape == (10, 5) and (out[:, 1] == 200).all()
    share = out[:, 0].sum() / out[:, 1].sum()
    assert 0.12 < share < 0.20                                 # P(N(0,1) > 1) is about 0.16
    assert (out[:, 2] > 0.7).all()                             # each run's own line still catches most of the fault


def test_bootstrap_share_and_mean():
    p, lo, hi = round4._boot([1, 1, 1, 1], [10, 10, 10, 10])
    assert p == 0.1 and lo == hi == 0.1
    p, lo, hi = round4._boot([0.2, 0.4], [1, 1], mean_of_runs=True)
    assert np.isclose(p, 0.3) and lo <= p <= hi


def test_repeat_share_is_high_when_runs_repeat_and_low_when_they_never_do():
    rng = np.random.default_rng(1)
    pattern = np.cumsum(rng.normal(size=(200, 3)), axis=0)
    repeating = [pattern + 0.01 * rng.normal(size=pattern.shape) for _ in range(6)]
    never = [posctl.normal_run(400, 1.0, 10, 50 + i) for i in range(6)]
    high, low = round4.repeat_share(repeating, n_boot=200), round4.repeat_share(never, n_boot=200)
    assert high["repeat share"] > 0.9 and low["repeat share"] < 0.2
    assert high["low"] <= high["repeat share"] <= high["high"]
