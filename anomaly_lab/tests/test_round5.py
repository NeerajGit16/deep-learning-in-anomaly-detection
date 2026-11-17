"""Round 5: the etcher's setting columns, the long training budget and the PCA-plus-correction autoencoder."""
import numpy as np
import pytest

from adlab import deep, etch, round5


def test_every_etch_fault_names_one_of_the_six_settings_and_their_columns_exist():
    assert set(round5.SETTING_COLUMNS) == set(k for k in etch.FAULT_KINDS)
    for cols in round5.SETTING_COLUMNS.values():
        assert all(c in etch.CHANNELS for c in cols)
    assert len(round5.ALL_SETTINGS) == 7 and round5.setting_of("BCl3 +10") == "BCl3"


def test_the_budgets_are_as_fixed():
    assert deep.BUDGETS["v1"] == {"max": 100, "patience": 10, "lr_patience": None}
    assert deep.BUDGETS["v2"] == {"max": 1000, "patience": 30, "lr_patience": 10}
    assert deep.AutoencoderLong.budget == "v2" and deep.Autoencoder.budget == "v1"
    assert deep.LSTMLong.name == "LSTM forecaster, long training"


def test_the_learning_rate_is_halved_after_ten_passes_without_improvement():
    torch = pytest.importorskip("torch")
    assert deep._scheduler(None, deep.BUDGETS["v1"]) is None
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w], lr=1e-3)
    sched = deep._scheduler(opt, deep.BUDGETS["v2"])
    sched.step(1.0)                                   # the first value sets the best
    for _ in range(11):
        sched.step(1.0)                               # no improvement
    assert abs(opt.param_groups[0]["lr"] - 5e-4) < 1e-12


def test_pca_plus_correction_starts_as_pca():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6))
    centre = Z.mean(0)
    _, _, vt = np.linalg.svd(Z - centre, full_matrices=False)
    P = vt[:2].T
    net = deep._nets()["PCAC"](centre, P)
    with torch.no_grad():
        out = net(torch.as_tensor(Z, dtype=torch.float32)).numpy()
    pca = centre + (Z - centre) @ P @ P.T
    assert np.allclose(out, pca, atol=1e-4)
    assert sum(p.numel() for p in net.parameters()) > 0                  # only the correction is trained
    assert all(not b.requires_grad for b in net.buffers())


def test_a_long_trained_autoencoder_is_stored_apart():
    a, b = deep.Autoencoder(), deep.AutoencoderLong()
    X = np.zeros((10, 3))
    assert a._store(X, "x").name != b._store(X, "x").name
    assert "v2" in b._store(X, "x").name
