"""Checks against the real files. Skipped until the data has been downloaded."""
import numpy as np
import pandas as pd
import pytest

from adlab import data as D


def _have(name):
    probe = {"skab": D.RAW / "skab" / "data", "tep": D.RAW / "tep" / "d00.dat",
             "tep_rieth": D.RAW / "tep_rieth" / D.RIETH_FILES["faulty_testing"][1]}[name]
    return probe.exists()


@pytest.mark.skipif(not _have("skab"), reason="run `python -m adlab.download` first")
def test_skab_shape():
    ds = D.load_skab()
    assert len(ds.runs) == 34 and len(ds.channels) == 8
    for r in ds.runs:
        assert r.X.shape[1] == 8 and len(r.X) == len(r.y)
        assert r.fit.shape[0] in (104, 400)
        assert r.y.sum() > 0 and (r.y[0] == 0 or r.name == "other/2")


@pytest.mark.skipif(not _have("tep"), reason="run `python -m adlab.download` first")
def test_tep_shape_and_onset():
    ds = D.load_tep()
    assert len(ds.runs) == 21 and ds.runs[0].fit.shape == (500, 52)
    for r in ds.runs:
        assert r.X.shape == (960, 52)
        assert r.y[:160].sum() == 0 and r.y[160:].all()
    assert ds.val_normal[0].X.shape == (960, 52)
    # alarm lines come from the one fault-free test run, over the same samples as the faults
    assert ds.calibration == "matched" and ds.matched_start == 160 and ds.matched_from is None
    assert [r.name for r in ds.calibration_runs()] == ["normal_te"]


# ---------------------------------------------------------------- Rieth files

def _rieth_frame(n_faults=2, n_runs=3, length=4, seed=0):
    """A small table laid out like the Rieth files, with its rows shuffled."""
    rng = np.random.default_rng(seed)
    cols = (["faultNumber", "simulationRun", "sample"] + [f"xmeas_{i}" for i in range(1, 42)]
            + [f"xmv_{i}" for i in range(1, 12)])
    rows = [[f, r, s] + [1000 * f + 100 * r + s + v / 100 for v in range(52)]
            for f in range(1, n_faults + 1) for r in range(1, n_runs + 1) for s in range(1, length + 1)]
    df = pd.DataFrame(rows, columns=cols)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def test_rieth_table_becomes_runs_in_order():
    out = D._frame_to_runs(_rieth_frame(), n_runs=2)
    assert out.shape == (2, 2, 4, 52)
    assert out[1, 0, 2, 0] == 2000 + 100 + 3 and out[0, 1, 3, 51] == 1000 + 200 + 4 + 0.51
    with pytest.raises(ValueError):
        D._frame_to_runs(_rieth_frame().rename(columns={"xmv_11": "other"}), 2)
    with pytest.raises(ValueError):
        D._frame_to_runs(_rieth_frame().iloc[:-1], 3)          # a run with a sample missing


def test_first_difference():
    a = np.zeros((3, 5, 2))
    b = a.copy()
    b[0, 3, 1] = 1
    b[2, 0, 0] = 1
    assert list(D.first_difference(a, b)) == [3, -1, 0]


needs_rieth = pytest.mark.skipif(not _have("tep_rieth"), reason="needs the Rieth files (data.download_rieth)")


@needs_rieth
def test_rieth_shapes_and_run_numbers():
    ds = D.load_tep_rieth()
    assert ds.name == "tep_rieth" and len(ds.runs) == 1000 and len(ds.val_normal) == 20
    assert ds.runs[0].fit.shape == (10000, 52) and all(r.fit is ds.runs[0].fit for r in ds.runs + ds.val_normal)
    for r in ds.runs:
        assert r.X.shape == (960, 52) and r.y[:160].sum() == 0 and r.y[160:].all()
    assert sorted({r.meta["fault"] for r in ds.runs}) == list(range(1, 21))
    # held-out normal runs use run numbers that no scored run uses
    assert {r.meta["run"] for r in ds.runs} == set(range(1, 51))
    assert [v.name for v in ds.val_normal] == [f"normal/run{n:03d}" for n in range(51, 71)]
    control = D.load_tep_rieth_control()
    assert len(control.runs) == 50 and control.val_normal is None and control.runs[0].y[160:].all()
    # alarm lines: fault-free runs 51-100, which are neither fit runs nor twins of the scored runs
    assert ds.calibration == control.calibration == "matched" and ds.matched_start == 160
    cal = ds.calibration_runs()
    assert ds.matched_from.name == control.matched_from.name == "tep_rieth_calibration"
    assert [r.meta["run"] for r in cal] == list(range(51, 101)) and all(r.fit is ds.runs[0].fit for r in cal)
    assert not {r.meta["run"] for r in cal} & {r.meta["run"] for r in control.runs}
    assert D.load_tep_rieth(fit_runs=1).matched_from.name != ds.matched_from.name
    assert D.load_tep_rieth(fit_runs=1).name != ds.name          # other choices of runs get their own cache name


@needs_rieth
def test_rieth_fault_starts_at_index_160_read_off_the_twins():
    faulty = D.rieth_runs("faulty_testing", 50)
    twins = D.rieth_runs("fault_free_testing", 50)[0]
    first = np.array([D.first_difference(f, twins) for f in faulty])       # (20 faults, 50 runs)
    assert first.min() == D.RIETH_ONSET == D.TEP_ONSET                      # nothing differs before the fault
    steps = [f - 1 for f in (1, 2, 3, 4, 5, 6, 7, 14, 15, 19)]              # these change the readings at once
    assert (first[steps] == D.RIETH_ONSET).all()
    table = D.rieth_onset_check()
    assert table["earliest first difference"].min() == 160 and len(table) == 20
    # the scored runs really are the twins of the control runs before the fault
    ds, control = D.load_tep_rieth(), D.load_tep_rieth_control()
    for r in ds.runs[::37]:
        assert np.array_equal(r.X[:160], control.runs[r.meta["run"] - 1].X[:160])


@needs_rieth
@pytest.mark.skipif(not _have("tep"), reason="needs the Braatz files too")
def test_rieth_runs_are_fresh():
    """No Rieth run used here repeats a Braatz file, and fit, scored and held-out runs do not repeat each other."""
    ds, braatz = D.load_tep_rieth(), D.load_tep()
    b_fit, b_normal = braatz.runs[0].fit, braatz.val_normal[0].X
    fit_runs = ds.runs[0].fit.reshape(20, 500, 52)
    assert not any(np.allclose(x, b_fit) for x in fit_runs)
    assert not any(np.allclose(v.X, b_normal) for v in ds.val_normal)
    for r in ds.runs:
        assert not np.allclose(r.X[:160], braatz.runs[r.meta["fault"] - 1].X[:160])
    # samples 3-160 (after the shared starting state) differ between any two different run numbers
    first_hours = {r.name: r.X[2:160].tobytes() for r in ds.runs[:50] + ds.calibration_runs()}
    assert len(set(first_hours.values())) == 100
    assert not any(x[2:160].tobytes() in set(first_hours.values()) for x in fit_runs)


# ---------------------------------------------------------------- Cranfield flow rig

def _flows(spec, noise=0.0, seed=0):
    """Made-up readings with only the air (column 8) and water (column 9) flows filled in."""
    rng = np.random.default_rng(seed)
    air = np.concatenate([np.full(n, a) for a, w, n in spec])
    water = np.concatenate([np.full(n, w) for a, w, n in spec])
    X = np.zeros((len(air), 23))
    X[:, 7] = air + noise * 0.0003 * rng.normal(size=len(air))
    X[:, 8] = water + noise * 0.01 * rng.normal(size=len(air))
    return X


def test_flow_settings_names_each_stretch_and_marks_changes_as_between():
    X = _flows([(0.0208, 0.5, 300), (0.0300, 0.5, 100), (0.0417, 2.0, 300), (0.0417, 6.0, 20), (0.0417, 3.5, 200)], noise=1.0)
    label = D.flow_settings(X)
    names = [D.setting_name(label[a]) for a, b in D.stretches(label)]
    assert names == ["air 0.0208, water 0.5", "between settings", "air 0.0417, water 2", "between settings",
                     "air 0.0417, water 3.5"]            # 0.0300 is no setting; 20 readings at one is too short
    assert (label[:290] == 0).all() and (label[420:690] == 3 * 5 + 2).all()


def test_cranfield_labels_cover_the_published_spans_counted_from_one():
    y = D.cranfield_labels(10, [(3, 5), (9, 10)])
    assert y.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]


needs_cranfield = pytest.mark.skipif(not (D.RAW.joinpath(*D.CRANFIELD_DIR) / "Training.mat").exists(),
                                     reason="needs the Cranfield files under data/raw/cranfield")


@needs_cranfield
def test_cranfield_shapes():
    ds = D.load("cranfield")
    assert len(ds.runs) == 16 and len(ds.channels) == 23 and ds.sample_seconds == 1.0
    assert {r.X.shape[1] for r in ds.runs} == {23} and ds.runs[0].fit.shape == (9825 + 13200, 23)
    assert [r.name for r in ds.val_normal] == ["T1"] and ds.calibration == "matched" and ds.matched_start == 0
    assert all(0 < r.y.mean() < 1 for r in ds.runs)                      # every run has normal and faulty readings
    assert len({r.group for r in ds.runs}) == 6
    normal = D.cranfield_normal(columns=24)
    assert all(np.ptp(X[:, 23]) < 0.002 for X in normal.values())        # column 24 does not move in normal running


@needs_cranfield
def test_cranfield_published_fault_times_agree_with_the_recorded_fault_traces():
    t = D.cranfield_fault_times_check()
    gap = (t["published start"] - t["trace first moves"]).abs()
    assert gap.drop("2_1").max() <= 1          # the fault trace starts moving at the published start
    assert gap["2_1"] == 156                   # the one exception: published 2244, trace moves at 2400
    back = t["trace back to start value"].dropna()
    assert (t.loc[back.index, "published end"] - back).abs().max() <= 1


@needs_cranfield
def test_cranfield_normal_runs_each_pass_through_the_flow_settings():
    for X in D.cranfield_normal().values():
        label = D.flow_settings(X)
        assert len(set(label) - {D.BETWEEN}) >= 19 and 0.05 < (label == D.BETWEEN).mean() < 0.2
