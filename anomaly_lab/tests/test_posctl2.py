"""Round 5, part 3: the positive control's second version, as designed and after the check."""
import numpy as np

from adlab import posctl2 as p


def test_ten_sensors_without_a_bend_are_the_same_in_both_designs():
    a = p.normal_run(600, 0.0, 10, 123, p.GAIN, design="first")
    b = p.normal_run(600, 0.0, 10, 123, p.GAIN, design="final")
    assert np.allclose(a, b)


def test_the_final_design_makes_the_bent_part_one_and_a_half_times_as_large():
    first = p.scales(1.0, 10, p.GAIN, "first")["bent"]
    final = p.scales(1.0, 10, p.GAIN, "final")["bent"]
    assert np.allclose(final, 1.5 * first)


def test_groups_of_sensors_have_their_own_drivers():
    def cross(design):
        X = p.normal_run(20_000, 1.0, 20, 7, p.GAIN, design=design)
        C = np.corrcoef(X.T)
        return np.abs(C[:10, 10:]).max()
    assert cross("final") < 0.15            # group 0 and group 1 move independently
    assert cross("first") > 0.5             # one set of drivers for all: strongly linked


def test_every_sensor_moves_and_the_fault_stays_in_range():
    lo, hi = -np.ones(10) * 50, np.ones(10) * 50
    X, y, j = p.faulty_run(1.0, 10, 70_000, lo, hi, np.zeros(10), 2.0, p.GAIN)
    assert X.shape == (p.TEST_RUN, 10) and y.sum() == p.TEST_RUN - p.FAULT_FROM
    assert np.all(X.std(0) > 0) and 0 <= j < 10
