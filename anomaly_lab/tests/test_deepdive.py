"""The ingredient models of notebook 11 (adlab/deepdive.py). Made-up data; no download needed."""
import numpy as np

from adlab import deepdive as DD


def _data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 2))
    X = np.column_stack([f[:, 0], f[:, 1], f[:, 0] + f[:, 1], np.maximum(0, f[:, 0]) * 2, f[:, 1] - f[:, 0]])
    return X + 0.1 * rng.normal(size=X.shape)


def _fit(model, X, runs=6):
    model.context_ = {"dataset": "made-up", "lengths": [len(X) // runs] * runs}
    return model.fit(X)


def test_feature_pca_scores_are_causal_and_one_per_reading():
    X = _data()
    for cls in (DD.PCAHinges, DD.FeaturePCA):
        m = _fit(cls(), X)
        s = m.score(X)
        assert s.shape == (len(X),) and np.all(np.isfinite(s))
        Y = X.copy()
        Y[600:] += 5.0                                   # changing later readings leaves earlier scores alone
        assert np.allclose(m.score(Y)[:600], s[:600])


def test_corners_help_where_a_sensor_has_a_corner():
    """Sensor 3 is a hinge of sensor 0: PCA allowed corners reconstructs it better on held-aside data."""
    X, test = _data(seed=0), _data(seed=1)
    plain, hinges = _fit(DD.FeaturePCA(), X), _fit(DD.PCAHinges(), X)
    err_plain = np.abs(plain.errors(test)[:, 3]).mean()
    err_hinges = np.abs(hinges.errors(test)[:, 3]).mean()
    assert err_hinges < err_plain


def test_group_pca_uses_the_last_column_as_the_group_and_never_scores_it():
    rng = np.random.default_rng(2)
    X = _data()
    g = (np.arange(len(X)) // 100) % 2
    X[g == 1] = X[g == 1] @ np.diag([1, -1, 1, 1, -1])  # the links differ between the two groups
    m = _fit(DD.GroupPCA(), np.column_stack([X, g]))
    assert set(m.models_) == {0, 1}
    assert m.errors(np.column_stack([X, g])).shape == X.shape
    s1 = m.score(np.column_stack([X, g]))
    s2 = m.score(np.column_stack([X, 1 - g]))         # told the wrong group, the readings look odd
    assert np.median(s2) > np.median(s1)
    del rng


def test_switch_columns_are_found_only_for_outputs_that_sit_at_an_edge():
    from adlab import indpensim
    d = len(indpensim.CHANNELS)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1200, d))
    j = DD._columns("outputs")[0]
    X[:, j] = np.where(rng.random(1200) < 0.3, 0.0, rng.uniform(1, 5, 1200))   # off 30% of the time
    m = _fit(DD.PCASwitches(), X)
    assert [(k, side) for k, side, _ in m.switches_] == [(j, "low")]
