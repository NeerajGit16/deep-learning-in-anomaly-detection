"""IndPenSim loader (adlab/indpensim.py). Needs data/raw/indpensim (indpensim.download)."""
import numpy as np
import pytest

from adlab import data as D
from adlab import indpensim as I

needs_data = pytest.mark.skipif(not ((D.RAW / I.FOLDER / I.CACHE).exists() or (D.RAW / I.FOLDER / I.CSV).exists()),
                                reason="needs the IndPenSim data (indpensim.download, about 0.5 GB)")


def test_every_column_has_one_role():
    assert len(I.COLUMNS) == I.PROCESS_COLUMNS
    roles = {role for _, role, _ in I.COLUMNS}
    assert roles == {I.TIME, I.OUTPUT, I.ONLINE, I.OFFLINE, I.STATE, I.LABEL, I.RAMAN}
    assert len({name for name, _, _ in I.COLUMNS}) == I.PROCESS_COLUMNS


@needs_data
def test_batches_strategies_and_faults():
    d = I.readings()
    assert d.shape[0] == 113935 and d["batch"].nunique() == 100
    c = I.check_labels()
    assert c.attrs["time restarts"] == 100 and c["batch again agrees"].all()
    assert (c.loc[1:30, "strategy"] == "recipe").all() and (c.loc[31:60, "strategy"] == "operator").all()
    assert (c.loc[61:90, "strategy"] == "APC").all() and (c.loc[91:100, "strategy"] == "recipe").all()
    assert list(c.index[c["faulty batch"] == 1]) == list(range(91, 101))
    assert c.loc[1:90, "readings with fault = 1"].sum() == 0 and (c.loc[91:100, "readings with fault = 1"] > 0).all()
    spans = I.fault_spans()
    assert len(spans) == 10 and (spans["fault starts (h)"] >= 20).all()


@needs_data
def test_columns_that_never_move_are_marked():
    t = I.column_table()
    still = set(t.index[~t["moves in normal batches"].astype(bool)])
    assert {"agitator", "ammonia shots"} <= still
    assert np.isclose(t.loc["penicillin offline", "share missing (normal batches)"], 0.98, atol=0.01)


@needs_data
def test_no_batch_is_both_fitted_on_and_held_out():
    s = I.split()
    names, rounds, strategies = np.array(s["names"]), s["rounds"], np.array(s["strategies"])
    assert len(names) == len(set(names)) == 90 and sum(len(r) for r in s["runs"]) == 102410
    assert all(r.shape[1] == len(I.CHANNELS) == 20 for r in s["runs"])
    for k in range(5):
        held, fitted = set(names[rounds == k]), set(names[rounds != k])
        assert held and not held & fitted
        assert set(strategies[rounds != k]) == {"recipe", "operator", "APC"}     # every strategy fitted on, every round
    assert all(set(st) == set(range(10)) for st in s["stages"])


@needs_data
def test_detector_batches_are_never_scored_after_being_fitted_on():
    ds = I.detector_dataset()
    roles, normal = I.roles(), I.normal_batches()
    fitted = {f"batch{b['batch']}" for b, r in zip(normal, roles) if r == "fit"}
    assert len(fitted) == 60 and len(ds.val_normal) == 30 and len(ds.runs) == 10
    assert not fitted & ({r.name for r in ds.runs} | {r.name for r in ds.val_normal})
    assert {r.meta["strategy"] for r in ds.runs} == {"recipe"}
    for s in ("recipe", "operator", "APC"):
        assert sum(r.meta["strategy"] == s for r in ds.val_normal) == 10
    assert sum(ds.fit_lengths) == len(ds.runs[0].fit)
