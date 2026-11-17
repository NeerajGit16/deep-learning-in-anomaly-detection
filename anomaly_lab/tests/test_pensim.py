"""The PenSimPy pilot helpers (adlab/pensim.py) and the part-D pilot set-up (adlab/mechanism.py). PenSimPy itself
runs in its own environment, so the generation test skips when it is not installed."""
import numpy as np
import pytest

from adlab import indpensim, mechanism, pensim, simulate


def test_held_until_rewrites_until_profiles_for_a_recipe_that_holds_from_each_time():
    profile = [{"time": 100, "value": 0}, {"time": 102, "value": 4000}, {"time": 130, "value": 0}]
    out = pensim.held_until(profile)
    assert out == [{"time": 0, "value": 0}, {"time": 100, "value": 4000}, {"time": 102, "value": 0}]

    def value_at(sps, t):                         # "from" semantics, as PenSimPy's Recipe class reads them
        return [sp["value"] for sp in sps if sp["time"] <= t][-1]
    assert value_at(out, 101) == 4000 and value_at(out, 110) == 0 and value_at(out, 50) == 0


def test_every_indpensim_channel_has_a_pensimpy_state():
    assert set(pensim.CHANNEL_MAP) == set(indpensim.CHANNELS)


def test_generate_one_batch_when_pensimpy_is_installed():
    pytest.importorskip("pensimpy")
    b = pensim.generate(seed=1, fault=pensim.SUBSTRATE_FAULT)
    assert list(b.columns) == ["time"] + indpensim.CHANNELS + ["fault"]
    on = b["fault"] == 1
    assert set(np.round(b.loc[on, "sugar feed"], 6)) <= {2.0, 20.0}


def test_the_pilot_dataset_fits_on_normal_batches_and_sets_its_line_on_others():
    ds = mechanism.dataset(1.0)
    assert len(ds.runs) == mechanism.N_FAULTY and len(ds.val_normal) == mechanism.N_LINE
    assert sum(ds.fit_lengths) == len(ds.runs[0].fit)
    assert all(r.y.sum() > 0 for r in ds.runs) and all(r.y.sum() == 0 for r in ds.val_normal)
    fit_first = simulate.recipe_batches(mechanism.N_FIT, dial=1.0, seed=mechanism.SEEDS["fit"])[0]["X"]
    assert np.array_equal(ds.runs[0].fit[:len(fit_first)], fit_first)
    assert not any(np.array_equal(v.X[:20], fit_first[:20]) for v in ds.val_normal)
