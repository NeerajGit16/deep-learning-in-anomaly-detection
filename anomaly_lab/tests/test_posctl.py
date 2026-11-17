"""Round 4, part 4: the positive control's generator (adlab/posctl.py)."""
import numpy as np

from adlab import posctl


def test_drivers_are_smooth_with_unit_variance():
    h = posctl.drivers(20000, 1)
    assert h.shape == (20000, 3)
    assert np.allclose(h.std(0), 1, atol=0.1)
    step = np.diff(h, axis=0).std(0)
    assert (step < 0.1).all()                                    # smooth: small changes from one reading to the next


def test_bend_zero_is_flat_and_bend_one_is_not():
    for bend, flat in ((0.0, True), (1.0, False)):
        X = posctl.normal_run(4000, bend, 10, 5)
        Z = (X - X.mean(0)) / X.std(0)
        ev = np.linalg.svd(Z, compute_uv=False) ** 2
        share = ev[:3].sum() / ev.sum()
        assert (share > 0.99) == flat


def test_ten_sensors_are_the_first_ten_of_forty():
    h = posctl.drivers(500, 3)
    s10, b10 = posctl._parts(h, 10)
    s40, b40 = posctl._parts(h, 40)
    assert np.allclose(s10, s40[:, :10]) and np.allclose(b10, b40[:, :10])


def test_noise_is_two_to_five_percent():
    u = posctl._coefficients()["u"]
    assert u.min() >= 0.02 and u.max() <= 0.05


def test_a_fault_moves_one_sensor_by_half_its_spread_and_stays_in_range():
    fit = np.vstack([posctl.normal_run(500, 1.0, 10, 10_000 + i) for i in range(4)])
    lo, hi, centre = fit.min(0), fit.max(0), np.median(fit, 0)
    X, y, j = posctl.faulty_run(1.0, 10, 30_000, lo, hi, centre)
    clean = posctl.normal_run(posctl.TEST_RUN, 1.0, 10, 30_000)
    moved = X - clean
    assert y.sum() == posctl.TEST_RUN - posctl.FAULT_FROM
    assert np.allclose(moved[:, [k for k in range(10) if k != j]], 0) and np.allclose(moved[:150, j], 0)
    sd = posctl.scales(1.0, 10)["sd"][j]
    assert np.median(np.abs(moved[150:, j])) == np.float64(np.median(np.abs(moved[150:, j])))
    assert np.isclose(np.abs(moved[150:, j]).max(), 0.5 * sd)
    assert (X[150:, j] >= lo[j]).all() and (X[150:, j] <= hi[j]).all()


def test_settings_order_and_dataset_shape():
    s = posctl.settings()
    assert len(s) == 24 and s[0] == (0.0, 10, 2000) and s[-1] == (2.0, 40, 50000)
    ds = posctl.dataset(0.5, 10, 2000)
    assert len(ds.runs) == 40 and len(ds.val_normal) == 20 and ds.fit_lengths == [500] * 4
    assert ds.runs[0].X.shape == (300, 10) and ds.calibration == "matched"
