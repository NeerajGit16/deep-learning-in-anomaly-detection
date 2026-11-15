"""The port of the Cranfield authors' monitoring example (adlab/cva.py)."""
import numpy as np

from adlab import cva


def _process(n, seed, shift=0.0):
    """Three linked sensors with some memory from one reading to the next."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3))
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.normal(size=3) @ np.array([[1, 0.5, 0], [0, 1, 0.5], [0, 0, 1]])
    x[n // 2:, 0] += shift
    return x


def test_past_and_future_are_stacked_as_in_the_authors_code():
    y = np.arange(20).reshape(10, 2)                     # reading t is [2t, 2t+1]
    past, future = cva._past(y, 3, 2), cva._future(y, 3, 2)
    assert past.shape == (6, 6) and future.shape == (4, 6)
    assert past[:, 0].tolist() == [4, 5, 2, 3, 0, 1]     # newest of the three past readings first
    assert future[:, 0].tolist() == [6, 7, 8, 9]         # then the two readings that follow
    assert cva._past(y, 3, 0).shape == (6, 8)            # monitoring: every reading from the third on


def test_kde_limit_is_close_to_the_plain_percentile():
    x = np.random.default_rng(0).chisquare(5, 5000)
    assert abs(cva.kde_limit(x, 0.99) - np.quantile(x, 0.99)) < 1.5


def test_cva_flags_a_shift_and_stays_quiet_before_it():
    train = [_process(1500, 0), _process(1500, 1)]
    out = cva.cva_monitor(train, _process(1000, 2, shift=8.0), n=4, p=3, f=3)
    half = 500 - out["first"]
    assert len(out["T2"]) == 1000 - 2 and out["first"] == 2
    assert out["train_above"]["T2"] < 0.03 and out["train_above"]["SPE"] < 0.03
    before = max(np.mean(out["T2"][:half] > out["T2_limit"]), np.mean(out["SPE"][:half] > out["SPE_limit"]))
    after = max(np.mean(out["T2"][half:] > out["T2_limit"]), np.mean(out["SPE"][half:] > out["SPE_limit"]))
    assert before < 0.05 and after > 0.9


def test_first_above_counts_readings_from_one_and_stays_inside_the_span():
    above = np.array([False, True, False, False, True, True, False])      # values for readings 3..9 (offset 2)
    assert cva.first_above(above, 3, 9, 2) == 4
    assert cva.first_above(above, 5, 9, 2) == 7
    assert cva.first_above(above, 5, 6, 2) is None

