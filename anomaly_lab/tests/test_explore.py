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
