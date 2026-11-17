"""Round 4, part 2: the classic simple detectors (adlab/detectors.py)."""
import numpy as np

from adlab import detectors as det


def _ring(n, seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.cos(a), np.sin(a)]) + noise * rng.normal(size=(n, 2))


def test_nearest_normal_on_all_sensors():
    X = _ring(2000)
    m = det.NearestAll().fit(X)
    assert m.score(X[:5]).max() < 1e-6                             # a fitting reading is its own nearest
    centre = np.zeros((1, 2))                                      # inside every sensor's range, off the ring
    assert m.score(centre)[0] > 10 * np.quantile(m.score(_ring(500, 1)), 0.99)
    m5 = det.NearestAll5().fit(X)
    assert m5.score(_ring(200, 2)).mean() > m.score(_ring(200, 2)).mean()


def test_nearest_normal_subsamples_and_holds_out_whole_runs():
    X = _ring(3000)
    m = det.NearestAll(max_fit=1000)
    m.context_ = {"lengths": [1500, 1500]}
    m.fit(X)
    assert len(m.ref_) == 1000 and (m.fit_scores_ > 0).all()


def test_unseen_value_on_all_sensors_uses_only_stepped_sensors():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=600), rng.integers(0, 4, 600) * 2.0])
    m = det.UnseenAll().fit(X)
    assert m.columns_ == [1]
    assert m.score(np.array([[0.3, 2.0], [0.3, 3.0]])).tolist()[0] == 0 and m.score(np.array([[0.3, 3.0]]))[0] > 0
    assert det.applicable_unseen(X) == [1]


def _ar(n, phi, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(size=2) * np.sqrt(1 - phi ** 2)
    return x


def test_dynamic_pca_sees_a_change_in_how_readings_follow_each_other():
    fit = _ar(4000, 0.95, 0)
    test = np.vstack([_ar(500, 0.95, 1), _ar(500, -0.95, 2)])    # same spread, the memory flips
    d = det.DynamicPCA().fit(fit)
    p = det.PCA().fit(fit)
    line_d, line_p = np.quantile(d.score(fit), 0.99), np.quantile(p.score(fit), 0.99)
    assert np.mean(d.score(test[500:]) > line_d) > 0.5
    assert np.mean(p.score(test[500:]) > line_p) < 0.1


def test_dynamic_pca_never_mixes_two_runs():
    X = np.vstack([np.zeros((50, 2)) + [0, 1], np.zeros((50, 2)) + [5, 1]])
    X += np.random.default_rng(0).normal(size=X.shape) * 0.01
    d = det.DynamicPCA()
    d.context_ = {"lengths": [50, 50]}
    d.fit(X)
    assert d.mu_.shape == (6,)


def test_cva_detector_flags_a_shift_and_leaves_out_still_sensors():
    rng = np.random.default_rng(0)
    x = np.zeros((3000, 4))
    for t in range(1, 3000):
        x[t, :3] = 0.8 * x[t - 1, :3] + rng.normal(size=3) @ np.array([[1, 0.5, 0], [0, 1, 0.5], [0, 0, 1]])
    x[:, 3] = 7.0                                                  # a sensor that never moves
    c = det.CVA().fit(x[:2000])
    assert c.live_.tolist() == [True, True, True, False] and c.n_ == 14
    test = x[2000:].copy()
    test[500:, 0] += 8.0
    s = c.score(test)
    line = np.quantile(c.score(x[:2000]), 0.99)
    assert np.mean(s[:500] > line) < 0.05 and np.mean(s[505:] > line) > 0.9


def test_kernel_pca_flags_a_reading_off_a_bent_shape():
    X = _ring(5000)
    k = det.KernelPCA().fit(X)
    assert len(k.S_) == 3000 and k.q_ >= 2
    normal = k.score(_ring(1000, 3))
    assert k.score(np.zeros((1, 2)))[0] > np.quantile(normal, 0.99)
    assert np.mean(normal > 1.0) < 0.05                            # its own line (99th percentile) carries over


def test_recipe_hour_stepped_covers_only_stepped_outputs():
    rng = np.random.default_rng(0)
    batches = [np.column_stack([np.repeat([1.0, 2.0, 3.0], 20), rng.normal(size=60)]) for _ in range(6)]
    X = np.vstack(batches)
    m = det.RecipeHourStepped()
    m.context_ = {"lengths": [60] * 6, "outputs": [0, 1]}
    m.fit(X)
    assert m.cols_ == [0]
