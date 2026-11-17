"""The schedule-aware checks (detectors.UnseenValue, NearestNormal, RecipeAtHour). Made-up data."""
import numpy as np

from adlab import detectors as D


def _batches(n=6, length=50, seed=0):
    """Batches with one stepped output (col 0), one continuous output (col 1) and one measurement (col 2)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        step = np.where(np.arange(length) < length // 2, 10.0, 30.0)
        cont = np.linspace(0, 5, length) + 0.1 * rng.standard_normal(length)
        meas = step / 10 + 0.1 * rng.standard_normal(length)
        out.append(np.column_stack([step, cont, meas]))
    return out


def _fit(model, batches, outputs=(0, 1), groups=None):
    X = np.vstack(batches)
    model.context_ = {"lengths": [len(b) for b in batches], "outputs": list(outputs), "fit_groups": groups}
    return model.fit(X)


def test_unseen_value_looks_only_at_outputs_with_few_values_and_scores_unseen_ones():
    m = _fit(D.UnseenValue(), _batches())
    assert m.columns_ == [0]                                   # the continuous output has too many values
    test = np.array([[10.0, 99.0, 0.0], [20.0, 0.0, 0.0], [30.0 + 1e-9, 0.0, 0.0]])
    s = m.score(test)
    assert s[0] == 0 and s[2] == 0 and s[1] > 0               # seen, unseen, seen within the tolerance
    assert np.all(m.score_fit(None) == 0)                      # every fitting batch's values are in the others


def test_nearest_normal_uses_only_the_outputs_and_k5_is_at_least_k1():
    bs = _batches()
    m1, m5 = _fit(D.NearestNormal(), bs), _fit(D.NearestNormal5(), bs)
    x = bs[0][:5].copy()
    assert np.allclose(m1.score(x), 0)
    y = x.copy()
    y[:, 2] += 100                                             # a measurement, not an output
    assert np.allclose(m1.score(y), m1.score(x))
    z = x.copy()
    z[:, 0] = 20.0
    assert np.all(m1.score(z) > 0.5) and np.all(m5.score(z) >= m1.score(z) - 1e-12)


def test_recipe_at_this_hour_compares_with_recipe_batches_at_the_same_position():
    bs = _batches()
    groups = ["recipe"] * 5 + ["operator"]
    bs[5][:, 0] = 99.0                                         # the operator batch leaves the recipe
    m = _fit(D.RecipeAtHour(), bs, groups=groups)
    ok = bs[0].copy()
    late = bs[0].copy()
    late[:10, 0] = 30.0                                        # the second step's value, early: wrong for the hour
    assert np.median(m.score(ok)) < 5
    assert m.score(late)[:10].min() > 50
    longer = np.vstack([bs[0], bs[0][-10:]])                   # past the end: the last hour is used
    assert np.all(np.isfinite(m.score(longer)))
    assert np.median(m.score(bs[5])) > 50                      # a non-recipe batch scores high by design


def test_schedule_checks_score_each_reading_from_it_and_its_position_only():
    bs = _batches()
    for cls in (D.UnseenValue, D.NearestNormal, D.NearestNormal5, D.RecipeAtHour):
        m = _fit(cls(), bs, groups=["recipe"] * len(bs))
        x = bs[1].copy()
        s = m.score(x)
        x[30:] += 7.0
        assert np.allclose(m.score(x)[:30], s[:30]), cls.__name__
