"""Deep models (adlab/deep.py) on tiny simulated data, on the CPU. Needs PyTorch (the "deep" or
"deep-cpu" extra); skipped without it."""
import numpy as np
import pytest

from adlab import deep
from adlab import detectors as det
from adlab.simulate import linked_sensors

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def small_and_private(tmp_path, monkeypatch):
    """Train for a few passes only, on the CPU, and keep trained networks out of results/cache."""
    monkeypatch.setattr(deep, "STORE", tmp_path / "deep")
    monkeypatch.setattr(deep, "LOG", tmp_path / "deep" / "log.jsonl")
    monkeypatch.setattr(deep, "MAX_EPOCHS", 3)
    monkeypatch.setattr(deep, "device", lambda: torch.device("cpu"))


def fitted(key, X, lengths=None):
    model = det.make(key, **({"seed": 0} if det.REGISTRY[key].stochastic else {}))
    if hasattr(model, "context_"):
        model.context_ = {"dataset": "etch-test", "lengths": lengths}
    return model.fit(X)


def test_held_aside_data_is_whole_runs():
    segs = deep.segments(1000, [100] * 10)
    train, aside = deep.split_aside(segs)
    assert len(aside) == 2 and len(train) == 8 and not set(aside) & set(train)
    train, aside = deep.split_aside(deep.segments(1000, None))          # one stretch: its last fifth
    assert aside == [(800, 1000)] and train == [(0, 800)]
    assert deep.windows_for("tep6") == (10, 20) and deep.windows_for("etch") == (5, 10)


@pytest.mark.parametrize("key", ["deep_ae", "deep_lstm", "deep_gdn", "deep_gdn_paper", "deep_usad", "straight_forecaster"])
def test_one_score_per_reading_and_no_peeking_ahead(key):
    X = linked_sensors(600, 4, curve=0.5, seed=0)
    model = fitted(key, X[:400], [100] * 4)
    test = X[400:].copy()
    s = model.score(test)
    assert s.shape == (200,) and np.isfinite(s).all()
    later = test.copy()
    later[150:] += 5.0                                                  # change the future only
    np.testing.assert_allclose(model.score(later)[:150], s[:150], rtol=1e-5, atol=1e-6)
    assert np.isfinite(model.score_fit(X[:400])).all()


def test_networks_stay_small_and_settings_are_chosen_on_held_aside_data():
    X = linked_sensors(600, 6, curve=0.5, seed=1)
    for key in ("deep_lstm", "deep_gdn", "deep_usad"):
        model = fitted(key, X, [100] * 6)
        assert deep.weights(model.net_) < 1_000_000
        assert model.opt_["W"] in deep.windows_for("etch")
        assert model.aside_error_ >= 0
    big = deep._nets()["USAD"](20 * 52, 64)                             # the largest USAD used (W 20, 52 sensors)
    assert deep.weights(big) < 1_000_000
    gdn = deep._nets()["GDN"](52, 20, 64, 10)
    assert deep.weights(gdn) < 1_000_000


def test_a_second_score_reuses_the_trained_network():
    X = linked_sensors(500, 4, curve=0.5, seed=2)
    a = fitted("deep_gdn", X, [100] * 5)
    b = fitted("deep_gdn_paper", X, [100] * 5)                          # same network, the paper's score
    np.testing.assert_allclose(a.errors(X[:50]), b.errors(X[:50]), rtol=1e-6)
    assert not np.allclose(a.score(X[:50]), b.score(X[:50]))


def test_the_deep_models_are_registered_but_not_run_by_default():
    from adlab import run
    assert set(deep.MODELS) <= set(det.REGISTRY) and "straight_forecaster" in det.REGISTRY
    assert not set(deep.MODELS) & set(run.METHODS)
    assert all(run.settings(k) == {} for k in deep.MODELS)


def test_a_window_longer_than_the_held_aside_stretch_is_not_used():
    """SKAB's shortest session has 104 fit readings: its held-aside fifth (21) is shorter than the
    30-reading window, so only the 10-reading window can be judged."""
    X = linked_sensors(140, 4, curve=0.5, seed=3)
    for key in ("straight_forecaster", "deep_lstm"):
        model = det.make(key, **({"seed": 0} if det.REGISTRY[key].stochastic else {}))
        model.context_ = {"dataset": "skab-test", "lengths": None}
        model.fit(X[:104])
        w = model.w_ if key == "straight_forecaster" else model.opt_["W"]
        assert w == 10 and np.isfinite(model.score(X[104:])).all()
