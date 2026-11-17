"""Round 4, part 1: balanced folds and the fold audit's rule (adlab/folds.py)."""
import numpy as np

from adlab import detectors as det
from adlab import folds


def test_whole_runs_are_dealt_in_turn_and_match_the_round_3_folds():
    groups = [m for m in range(6) for _ in range(5)]           # 6 modes x 5 runs, mode by mode, as on the six-mode plant
    lengths = [40] * 30
    labels = np.repeat(groups, lengths)
    new = folds.balanced_folds(labels, lengths)
    old = det.StraightLinkBalanced()
    old.context_ = {"lengths": lengths, "fit_groups": groups}
    assert [list(a) for a in new] == [list(b) for b in old._folds(len(labels))]


def test_every_group_stays_in_the_fitted_part():
    rng = np.random.default_rng(0)
    lengths = [300, 250]
    labels = np.concatenate([np.repeat(rng.integers(0, 7, 12), 25), np.repeat(rng.integers(0, 7, 10), 25)])
    for held in folds.balanced_folds(labels, lengths):
        rest = np.setdiff1d(np.arange(len(labels)), held)
        for g in np.unique(labels):
            assert np.mean(labels[rest] == g) > 0
            assert (labels[rest] == g).sum() >= 0.6 * (labels == g).sum()


def test_a_group_with_few_visits_is_cut_into_pieces():
    labels = np.array([0] * 100 + [1] * 50)                      # one visit each
    fold = folds.fold_of(labels, [150])
    assert set(fold[:100]) == set(range(5)) and set(fold[100:]) == set(range(5))
    assert np.all(np.diff(fold[:100]) >= 0)                       # contiguous pieces, in time order


def test_the_blocks_of_time_can_leave_a_group_unseen():
    labels = np.repeat([0, 1, 2, 3, 4, 5], 50)                   # each group one fifth or less of the data
    table = folds.left(labels, det.blocked_folds(len(labels), 5))
    assert (table["smallest share left"].astype(float) < folds.UNSEEN).any()
    table = folds.left(labels, folds.balanced_folds(labels, None))
    assert (table["smallest share left"].astype(float) >= 0.75).all()


def test_balanced_aside_covers_every_reading_once():
    labels = np.repeat([0, 1, 0, 2, 1], 40)
    train, aside = folds.balanced_aside(labels, [100, 100])
    covered = np.zeros(len(labels), int)
    for a, b in train + aside:
        covered[a:b] += 1
    assert (covered == 1).all()
    assert all(b <= 100 or a >= 100 for a, b in train + aside)  # no segment crosses a run boundary


def test_a_balanced_copy_shares_the_scores_that_do_not_use_folds():
    assert folds.shared_name("cranfield_balanced", "pca") == "cranfield"
    assert folds.shared_name("cranfield_balanced", "link_curved") == "cranfield_balanced"
    assert folds.shared_name("cranfield_balanced", "deep_ae") == "cranfield_balanced"
    assert folds.shared_name("tep6_balanced_blocks", "deep_ae") == "tep6"
    assert folds.shared_name("tep6_told_balanced_blocks", "link_straight_told") == "tep6_told_balanced_blocks"
    assert folds.shared_name("skab", "link_curved") == "skab"


def test_link_detectors_use_balanced_folds_when_told_the_groups():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    labels = np.repeat([0, 1, 2], 100)
    m = det.StraightLinkResidual()
    m.context_ = {"lengths": [300], "fold_groups": {"blocks": labels}}
    assert [len(f) for f in m._folds(300)] == [60] * 5
    assert all(set(labels[f]) == {0, 1, 2} for f in m._folds(300))
