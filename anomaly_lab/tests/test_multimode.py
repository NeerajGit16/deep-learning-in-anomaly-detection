"""Multi-mode Tennessee Eastman (adlab/multimode.py). The first tests need no data and no network."""
import numpy as np
import pytest

from adlab import data as D
from adlab import multimode as MM


def test_run_paths_follow_the_files_layout():
    paths = MM.fault_free_paths(3)
    assert len(paths) == 60 and len(set(paths)) == 60
    assert paths[0] == "Mode3/SpVariation/SimulationCompleted/SP1/tRamp_0/SpMagnitude100"
    assert paths[-1] == "Mode3/SpVariation/SimulationCompleted/SP12/tRamp_40/SpMagnitude100"
    moves = MM.transition_paths(3)
    assert len(moves) == 20 and "Mode3ToMode3" not in " ".join(moves)
    assert moves[0] == "Mode3/ModeTransition/SimulationCompleted/Mode3ToMode1/TransitionTime10"


def test_runs_are_located_in_a_file_and_read_straight_from_their_place(tmp_path):
    """A small file laid out like the real ones: locate() finds the runs that exist, and
    read_block() fetches a run's readings from its offset without going through h5py."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    path = tmp_path / "TEP_Mode3.h5"
    wanted = MM.fault_free_paths(3)[:3]
    runs = {p: rng.normal(size=(20, 54)) for p in wanted[:2]}            # the third run is missing from the file
    with h5py.File(path, "w") as h:
        for p, values in runs.items():
            g = h.create_group(p)
            g.create_dataset("processdata", data=values)
            g.create_dataset("setpoint_init", data=np.arange(36.0).reshape(3, 12))
    blob = path.read_bytes()
    with h5py.File(path, "r") as h:
        found = MM.locate(h, wanted, verbose=False)
    assert [e["path"] for e in found] == wanted[:2] and found[0]["shape"] == [20, 54]
    assert found[0]["setpoints"][2][0] == 24.0
    for e in found:
        got = MM.read_block(lambda start, end: blob[start: end + 1], e)
        assert np.array_equal(got, runs[e["path"]])


def test_only_plainly_stored_readings_are_accepted(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "packed.h5"
    with h5py.File(path, "w") as h:
        g = h.create_group("Mode1/x")
        g.create_dataset("processdata", data=np.zeros((20, 54)), compression="gzip")
        g.create_dataset("setpoint_init", data=np.zeros((3, 12)))
    with h5py.File(path, "r") as h, pytest.raises(OSError):
        MM.locate(h, ["Mode1/x"], verbose=False)


needs_multimode = pytest.mark.skipif(not all((D.RAW / MM.FOLDER / f"mode{m}_fault_free.npy").exists() for m in MM.MODES),
                                     reason="needs the multi-mode runs (multimode.download)")


@needs_multimode
def test_multimode_shapes_and_columns():
    for m in MM.MODES:
        block = MM.fault_free(m)
        assert block.shape == (60, 2000, 52) and np.isfinite(block).all()
        moves, table = MM.transitions(m)
        assert moves.shape[1:] == (2000, 52) and len(table) == len(moves) and (table["starts at hour"] == 30).all()
        assert m not in set(table["to mode"])
    labels = MM.index(1)["labels"]
    assert len(labels) == 54 and labels[0] == "Time" and labels[-1] == "Agitator" and labels[9] == "Reactor Temperature"


@needs_multimode
def test_only_five_fault_free_runs_per_mode_are_different_simulations():
    """The 60 stored fault-free runs of a mode are 5 simulations stored 12 times each. The measure
    must use each simulation once, or a model would be scored on a copy of a run it was fitted on."""
    t = MM.copies_table()
    assert (t["different simulations"] == 5).all() and (t["the different ones are stored runs"] == "1, 2, 3, 4, 5").all()
    s = MM.split()
    assert len(s["runs"]) == 30 and len({r.tobytes() for r in s["runs"]}) == 30
    assert all(len(r) == 2000 - MM.START_UP for r in s["runs"])
    fold, mode = np.array([f[0] for f in s["folds"]]), np.array([m[0] for m in s["modes"]])
    for k in range(5):                                   # every round: 6 runs held out, one of every mode
        assert sorted(mode[fold == k]) == list(MM.MODES) and (fold != k).sum() == 24
    moves = MM.transition_split()
    assert len({r.tobytes() for r in moves["runs"]}) == len(moves["runs"]) == sum(len(MM.index(m)["transitions"]) for m in MM.MODES)


@needs_multimode
def test_most_pooled_movement_is_the_gap_between_modes():
    """Levels held at one set-point in every mode have no between-mode movement; most sensors are
    nearly all between-mode movement. Shares are between 0 and 1."""
    t = MM.between_modes_share()
    share = t["share of movement that is between modes"].dropna()
    assert len(t) == 52 and ((share > -1e-9) & (share < 1 + 1e-9)).all()
    assert share["XMEAS(8)"] < 0.01 and share.median() > 0.9


@needs_multimode
def test_sensors_curved_inside_a_mode_are_listed_with_their_best_partner():
    tiny = {"mode 3": {"table": __import__("pandas").DataFrame(
        {"straight": [0.4, 0.9], "curved": [0.6, 0.0], "curved_low": [0.5, 0.0], "curved_high": [0.7, 0.0]},
        index=["XMV(5)", "XMEAS(1)"])}}
    t = MM.within_curved(tiny)
    assert list(t.index) == [("mode 3", "XMV(5)")]
    row = t.iloc[0]
    assert row["best single predictor"] in D.TEP_CHANNELS and row["best single predictor"] != "XMV(5)"
    assert 0.5 < row["share of time at its highest value"] < 1          # the valve sits at its stop most of the time



needs_faults = pytest.mark.skipif(not all((D.RAW / MM.FOLDER / f"mode{m}_faults.npz").exists() for m in MM.MODES),
                                  reason="needs the multi-mode faulty runs (multimode.download_faults)")


@needs_faults
def test_one_full_strength_run_of_every_fault_in_every_mode():
    for m in MM.MODES:
        runs, idx = MM.faulty(m), MM.faults_index(m)
        assert sorted(runs) == list(MM.FAULTS)
        assert all(e["strength"] == 1.0 for e in idx["faults"])
        assert all(r.shape[1] == 52 and np.isfinite(r).all() for r in runs.values())
        assert all(len(r) <= 2000 for r in runs.values())
    for m in (1, 4):                       # mode 4's runs are stored without the extra reading at time 0
        times = MM.alignment_check(m)
        assert times["fault-free first, last"] == times["faulty first, last"] == (0.05, 100.0)


@needs_faults
def test_no_faulty_run_is_a_twin_of_a_normal_run():
    """A faulty run that shares random numbers with a normal run would, before its fault, be a copy
    of a run the detectors may be fitted on."""
    t = MM.twin_check()
    assert (t["faulty runs identical to a normal run before the fault"] == 0).all()
