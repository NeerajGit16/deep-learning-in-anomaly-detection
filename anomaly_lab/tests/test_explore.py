import numpy as np
import pytest

from adlab import data as D
from adlab import explore

needs_data = pytest.mark.skipif(not (D.RAW / "tep" / "d00.dat").exists() or not (D.RAW / "skab" / "data").exists(),
                                reason="run `python -m adlab.download` first")


def test_variable_tables_cover_every_channel():
    assert list(D.TEP_INFO["variable"]) == D.TEP_CHANNELS
    assert len(D.SKAB_INFO) == 8


@needs_data
def test_frames_have_time_sensors_and_fault():
    s = D.skab_frame("valve1/1")
    assert list(s.columns) == ["second"] + list(D.SKAB_INFO["sensor"]) + ["fault"]
    assert D.skab_frame("anomaly-free/anomaly-free")["fault"].sum() == 0
    t = D.tep_frame(1)
    assert t.shape == (960, 54) and t["fault"].iloc[160] == 1 and t["fault"].iloc[159] == 0
    assert len(D.skab_experiments()) == 34


@needs_data
def test_overview_and_fingerprint_shapes():
    tep = D.load_tep()
    assert explore.overview(tep)["runs"].sum() == 21
    fp = explore.fingerprint(tep)
    assert fp.shape == (21, 52) and np.isfinite(fp.values).all()
    # faults 3, 9 and 15 barely move anything; fault 6 moves a lot
    assert fp.iloc[[2, 8, 14]].mean(axis=1).max() < 1.2 < fp.iloc[5].mean()


def test_movement_by_stretch_sees_runs_that_start_calm():
    """Runs that all start from the same state and then wander: calm at first, normal later."""
    from adlab.data import Run
    rng = np.random.default_rng(0)
    fit = rng.normal(size=(4000, 3))
    runs = []
    for _ in range(40):
        grow = np.minimum(np.arange(200) / 50, 1.0)[:, None]          # the wander builds up over 50 readings
        x = rng.normal(size=(200, 3)) * grow
        runs.append(Run("r", x, np.zeros(200, int), fit, "normal"))
    table = explore.movement_by_stretch(runs, edges=(0, 50, 200), sample_hours=1.0)
    assert list(table.index) == ["hours 0-50", "hours 50-200"]
    assert table.loc["hours 0-50", "movement"] < 0.5 < 0.9 < table.loc["hours 50-200", "movement"] < 1.1
    assert table.loc["hours 0-50", "sensors identical at the start"] == 3
    assert table.loc["hours 50-200", "sensors identical at the start"] == 0
    assert table.loc["hours 0-50", "sensors under 0.8"] == 3 and table.loc["hours 50-200", "sensors under 0.8"] == 0


def test_tidy_numbers():
    assert [explore.tidy(v) for v in (0.83, 9.0, 0.0061, -0.0001, 35.125, 10)] == ["0.83", "9", "0.006", "0", "35.125", "10"]
