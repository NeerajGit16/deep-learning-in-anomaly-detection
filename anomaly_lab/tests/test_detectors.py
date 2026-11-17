import numpy as np
import pytest

from adlab import detectors as det
from adlab.metrics import dr_at_far

FAST = ["rolling_z", "ewma", "cusum", "pca", "iforest", "link_straight", "link_curved"]


def _step_case(seed=0):
    rng = np.random.default_rng(seed)
    fit = rng.normal(size=(400, 4))
    X = rng.normal(size=(600, 4))
    X[300:, 2] += 4.0  # one channel's level jumps by 4 standard deviations
    y = np.zeros(600, int)
    y[300:] = 1
    return fit, X, y


@pytest.mark.parametrize("key", FAST)
def test_single_channel_step_is_detected(key):
    fit, X, y = _step_case()
    s = det.make(key).fit(fit).score(X)
    assert dr_at_far([s], [y], 0.01)["dr"] > 0.5


@pytest.mark.parametrize("key", FAST)
def test_scores_are_causal(key):
    """Changing the future must not change the score at earlier times."""
    fit, X, _ = _step_case()
    model = det.make(key).fit(fit)
    a = model.score(X)
    X2 = X.copy()
    X2[400:] += 10.0
    b = model.score(X2)
    np.testing.assert_allclose(a[:400], b[:400])


def test_pca_sees_a_correlation_break_that_per_channel_stats_miss():
    """Two sensors that normally move together start moving in opposite directions,
    each staying inside its usual range (H2 in miniature)."""
    rng = np.random.default_rng(1)
    t = rng.normal(size=1000)
    fit = np.c_[t[:500], t[:500] + 0.1 * rng.normal(size=500)]
    X = np.c_[t[500:], t[500:] + 0.1 * rng.normal(size=500)]
    X[250:, 1] = -X[250:, 0] + 0.1 * rng.normal(size=250)
    y = np.zeros(500, int)
    y[250:] = 1
    pca = dr_at_far([det.make("pca", var=0.5).fit(fit).score(X)], [y], 0.01)["dr"]
    z = dr_at_far([det.make("rolling_z", window=1).fit(fit).score(X)], [y], 0.01)["dr"]
    assert pca > 0.8
    assert z < 0.2


def test_constant_channel_does_not_break_standardising():
    rng = np.random.default_rng(2)
    fit = np.c_[rng.normal(size=200), np.ones(200)]
    X = np.c_[rng.normal(size=50), np.ones(50)]
    for key in FAST:
        assert np.all(np.isfinite(det.make(key).fit(fit).score(X)))


def test_matrix_profile_flags_a_level_shift():
    pytest.importorskip("stumpy")
    fit, X, y = _step_case()
    s = det.make("matrix_profile", m=8).fit(fit).score(X)
    assert dr_at_far([s], [y], 0.01)["dr"] > 0.5


def test_fast_window_distances_match_stumpy():
    """Matrix Profile scores runs with its own nearest-stretch search; it must give stumpy's distances."""
    stumpy = pytest.importorskip("stumpy")
    rng = np.random.default_rng(5)
    smooth = np.cumsum(rng.normal(size=700))                     # wanders, like a slow sensor
    stepped = np.round(rng.normal(size=700))                     # few distinct values: many stretches repeat exactly
    for series in (rng.normal(size=700), smooth, stepped, 1e3 + stepped):
        a, b = series[400:], series[:400]
        for m in (4, 16):
            ref = stumpy.aamp(a, m, T_B=b, ignore_trivial=False)[:, 0].astype(float)
            got = det.nearest_window_distance(a, b, m, rows=64)
            np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)
    assert (det.nearest_window_distance(stepped[400:], stepped[:400], 4) == 0).any()   # exact repeats give exactly 0


# ---------------------------------------------------------------- link-residual detectors in the registry

def test_link_residuals_are_registered_under_their_own_names():
    straight, curved = det.make("link_straight"), det.make("link_curved")
    assert isinstance(straight, det.LinkResidual) and not straight.curved
    assert isinstance(curved, det.LinkResidual) and curved.curved
    # run.py reads the label and the stochastic flag from the class, before any detector is made
    assert det.REGISTRY["link_straight"].name == straight.name == "Straight-link residual"
    assert det.REGISTRY["link_curved"].name == curved.name == "Curved-link residual"
    assert not det.REGISTRY["link_straight"].stochastic and not det.REGISTRY["link_curved"].stochastic
    assert {"link_straight", "link_curved"} <= set(det.available())


@pytest.mark.parametrize("first", ["adlab.detectors", "adlab.structure", "adlab.run"])
def test_no_circular_import_whichever_module_comes_first(first):
    """Each module is imported first in a fresh Python, then the registry is used."""
    import subprocess
    import sys
    from pathlib import Path
    code = (f"import {first}; import adlab.structure as S; import adlab.detectors as D; "
            "assert S.LinkResidual is D.LinkResidual; D.make('link_straight'); D.make('link_curved')")
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_moving_the_detector_did_not_change_its_reference_errors():
    """The curved detector's held-out errors on the fit data are the curvature measure's own
    held-out predictions (structure.held_out_predictions), as before the class moved."""
    from adlab import structure as S
    from adlab.simulate import linked_sensors
    fit = linked_sensors(300, 4, curve=0.5, seed=0)
    Z, straight, both = S.held_out_predictions(fit)
    for key, pred in (("link_straight", straight), ("link_curved", both)):
        model = det.make(key).fit(fit)
        np.testing.assert_allclose(model.score_fit(fit), model._combine(Z - pred))


@pytest.mark.parametrize("rows", [400, 10_500])
def test_curved_link_residual_gives_the_same_scores_for_every_seed(rows):
    """Why it is not marked stochastic. 10,500 rows is above the size where scikit-learn would
    otherwise switch on early stopping with a random hold-out."""
    rng = np.random.default_rng(0)
    t = rng.normal(size=rows + 200)
    data = np.c_[t, np.tanh(t) + 0.2 * rng.normal(size=len(t)), rng.normal(size=len(t))]
    fit, X = data[:rows], data[rows:]
    a = det.make("link_curved", seed=0).fit(fit).score(X)
    b = det.make("link_curved", seed=7).fit(fit).score(X)
    np.testing.assert_array_equal(a, b)


def test_balanced_folds_hold_out_one_run_of_every_group_and_fall_back_without_groups():
    from adlab import detectors as D
    m = D.StraightLinkStageToldBalanced()
    m.context_ = {"lengths": [10] * 6, "fit_groups": [1, 1, 1, 2, 2, 2]}
    groups = np.repeat([1, 2], 30)
    folds = m._folds(60)
    assert len(folds) == 3 and all(sorted(set(groups[f])) == [1, 2] and len(f) == 20 for f in folds)
    m.context_ = {}
    assert len(m._folds(60)) == m.k                                   # the old blocks of time
