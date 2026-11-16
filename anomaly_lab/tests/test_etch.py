"""Metal etch loader (adlab/etch.py). Needs data/raw/metal_etch/MACHINE_Data.mat (etch.download)."""
import numpy as np
import pytest

from adlab import data as D
from adlab import etch

pytestmark = pytest.mark.skipif(not (D.RAW / etch.FOLDER / etch.FILE).exists(),
                                reason="needs MACHINE_Data.mat (etch.download)")


def test_row_counts_and_columns():
    normal, faulty = etch.normal_wafers(), etch.faulty_wafers()
    assert len(normal) == 107 and len(faulty) == 21                     # 108 normal in the file, one 3-reading stub left out
    assert "l3125" not in {w["wafer"] for w in normal}
    assert sum(w["readings"] for w in normal) == 10767
    assert [sum(w["experiment"] == e for w in normal) for e in etch.EXPERIMENTS] == [34, 36, 37]
    assert [sum(w["experiment"] == e for w in faulty) for e in etch.EXPERIMENTS] == [9, 6, 6]
    assert len(etch.CHANNELS) == 19 and not {"Time", "Step Number"} & set(etch.CHANNELS)
    for w in normal + faulty:
        assert w["X"].shape == (w["readings"], 19) and np.isfinite(w["X"]).all()
        assert np.all(np.diff(w["time"]) >= 0)                           # put in time order (5 wafers repeat one time stamp)
        assert set(w["step"]) <= {4, 5} and np.all(np.diff(w["step"]) >= 0)


def test_no_wafer_is_both_fitted_on_and_held_out():
    s = etch.split()
    names, rounds, exps = np.array(s["names"]), s["rounds"], np.array(s["experiments"])
    assert len(set(names)) == len(names) == 107
    for k in range(5):
        held, fitted = set(names[rounds == k]), set(names[rounds != k])
        assert held and not held & fitted
        assert set(exps[rounds != k]) == set(etch.EXPERIMENTS)          # all three experiments fitted on, every round
        assert set(exps[rounds == k]) == set(etch.EXPERIMENTS)


def test_detector_wafers_are_never_scored_after_being_fitted_on():
    ds, roles, normal = etch.dataset(), etch.roles(), etch.normal_wafers()
    fitted = {w["wafer"] for w, r in zip(normal, roles) if r == "fit"}
    scored = {r.name for r in ds.runs} | {r.name for r in ds.val_normal}
    assert not fitted & scored
    assert len(ds.runs) == 21 and all(r.y.all() for r in ds.runs)       # every reading of a faulty wafer is faulty
    assert all(not r.y.any() for r in ds.val_normal)
    assert ds.runs[0].fit.shape == (sum(w["readings"] for w, r in zip(normal, roles) if r == "fit"), 19)
    for e in etch.EXPERIMENTS:                                           # every experiment has line and check wafers
        assert {r.meta["role"] for r in ds.val_normal if r.meta["experiment"] == e} == {"line", "check"}
    told = etch.dataset(told=True)
    assert told.runs[0].X.shape[1] == 20 and set(told.runs[0].X[:, -1]) <= {4, 5}
