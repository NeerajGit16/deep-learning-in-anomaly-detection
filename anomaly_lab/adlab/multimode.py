"""Multi-mode Tennessee Eastman: the same chemical plant run in six operating modes.

Reinartz, Kulahci and Ravn (2021), "An extended Tennessee Eastman simulation dataset for
fault-detection and decision support systems", Computers and Chemical Engineering 149, 107281.
Data: data.dtu.dk, doi:10.11583/DTU.13385936 (public domain, CC0).

The data is one file of about 24 GB per mode. Only a small part is needed here, so it is read
out of the remote files in pieces (see remote.py) and never downloaded whole:

  fault-free runs    60 per mode. The files have no group called "normal". They are the
                     setpoint-variation runs in which the setpoint is "changed" to 100% of its
                     own value, that is, not changed: SpVariation / SimulationCompleted /
                     SP1..SP12 / tRamp_0..tRamp_40 / SpMagnitude100. Each is its own simulation.
  transition runs    up to 20 per mode: the plant is moved from this mode to another one over
                     10, 20, 30 or 40 hours, starting 30 hours in.

Every run is 100 hours, one reading every 3 minutes (2,000 readings). Columns: time, the 41
measurements and 12 valves of the original plant. Dropping the time and the twelfth valve (the
agitator, which never moves) leaves the same 52 columns, in the same order, as the Braatz and
Rieth files (data.TEP_CHANNELS).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, progress

FILES = {1: 26003087, 2: 26003162, 3: 26003492, 4: 26024015, 5: 26028068, 6: 26034659}   # figshare file ids
MODES = tuple(FILES)
FOLDER = "tep_multimode"
READINGS, COLUMNS = 2000, 54
RAMPS = (0, 10, 20, 30, 40)
TRANSITION_HOURS = (10, 20, 30, 40)
CHANGE_AT_HOUR = 30                    # setpoint changes, transitions and faults all start 30 hours in


def url(mode: int) -> str:
    return f"https://ndownloader.figshare.com/files/{FILES[mode]}"


# ---------------------------------------------------------------- finding the runs inside a file

def fault_free_paths(mode: int) -> list[str]:
    """Where the 60 fault-free runs of a mode sit in its file, in a fixed order (run 1 to run 60)."""
    return [f"Mode{mode}/SpVariation/SimulationCompleted/SP{sp}/tRamp_{ramp}/SpMagnitude100"
            for sp in range(1, 13) for ramp in RAMPS]


def transition_paths(mode: int) -> list[str]:
    """Where the transition runs of a mode would sit (those that shut the plant down are stored elsewhere)."""
    return [f"Mode{mode}/ModeTransition/SimulationCompleted/Mode{mode}ToMode{to}/TransitionTime{hours}"
            for to in MODES if to != mode for hours in TRANSITION_HOURS]


def locate(h5, paths: list[str], verbose: bool = True, what: str = "finding runs") -> list[dict]:
    """For each path that exists in the open file: where its readings are stored, and its setpoints.

    Returns dicts with "path", "offset" and "shape" of the run's `processdata`, and "setpoints"
    (3 rows: before, after, hour of the change). The readings must be stored plainly (one
    unbroken, uncompressed block), which is how these files are written; anything else stops here.
    """
    found = []
    for path in progress.track(paths, what, verbose=verbose):
        if path not in h5:
            continue
        d = h5[path]["processdata"]
        if d.chunks is not None or d.compression is not None or d.dtype != np.dtype("<f8"):
            raise OSError(f"{path}/processdata is not stored as one plain block of numbers")
        found.append({"path": path, "offset": int(d.id.get_offset()), "shape": [int(n) for n in d.shape],
                      "setpoints": h5[path]["setpoint_init"][()].tolist()})
    return found


def read_block(fetch, entry: dict) -> np.ndarray:
    """One run's readings, fetched straight from its place in the file."""
    rows, cols = entry["shape"]
    raw = fetch(entry["offset"], entry["offset"] + rows * cols * 8 - 1)
    return np.frombuffer(raw, dtype="<f8").reshape(rows, cols)


# ---------------------------------------------------------------- downloading

def _paths(raw: Path, mode: int) -> dict:
    base = raw / FOLDER
    return {"normal": base / f"mode{mode}_fault_free.npy", "transitions": base / f"mode{mode}_transitions.npy",
            "index": base / f"mode{mode}_index.json"}


def download_mode(mode: int, raw: Path = data.RAW, threads: int = 2, verbose: bool = False) -> dict:
    """Fetch one mode's fault-free and transition runs out of its remote file and save them.

    Writes mode<N>_fault_free.npy (runs, readings, 54 columns), mode<N>_transitions.npy and
    mode<N>_index.json (which run is which, the setpoints, and how many bytes were transferred).
    Does nothing if they are already there. Returns the index.
    """
    import h5py
    from .remote import RangeFile, http_fetcher

    out = _paths(raw, mode)
    if all(p.exists() for p in out.values()):
        return json.loads(out["index"].read_text())
    out["index"].parent.mkdir(parents=True, exist_ok=True)
    remote = RangeFile(url(mode), block=1 << 16, limit=2 << 30)
    with h5py.File(remote, "r") as h5:
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in np.ravel(h5["Processdata_Labels"][()])]
        normal = locate(h5, fault_free_paths(mode), verbose, f"mode {mode}: finding fault-free runs")
        moves = locate(h5, transition_paths(mode), verbose, f"mode {mode}: finding transition runs")
    fetch, _ = http_fetcher(url(mode))
    entries = normal + moves
    with ThreadPoolExecutor(threads) as pool:
        blocks = list(progress.track(pool.map(lambda e: read_block(fetch, e), entries), f"mode {mode}: downloading runs",
                                     total=len(entries), verbose=verbose))
    for kind, part in (("normal", blocks[:len(normal)]), ("transitions", blocks[len(normal):])):
        short = min(len(b) for b in part)                      # a few runs carry one extra reading at time 0
        np.save(out[kind], np.stack([b[len(b) - short:] for b in part]))
    index = {"mode": mode, "labels": labels, "fault_free": normal, "transitions": moves,
             "megabytes_transferred": round((remote.fetched_bytes + sum(b.nbytes for b in blocks)) / 1e6, 1)}
    out["index"].write_text(json.dumps(index, indent=1))
    progress.note(f"mode {mode}: saved {len(normal)} fault-free and {len(moves)} transition runs, "
                  f"{index['megabytes_transferred']} MB transferred")
    return index


def download(raw: Path = data.RAW, modes: tuple = MODES, together: int = 3, verbose: bool = True) -> Path:
    """Fetch the fault-free and transition runs of every mode (about 0.5 GB out of 143 GB).

    `together` modes are fetched at once, each in its own process. The server is slow (about
    0.25 MB a second per connection), so this takes about a quarter of an hour.
    """
    todo = [m for m in modes if not all(p.exists() for p in _paths(raw, m).values())]
    if todo:
        with ProcessPoolExecutor(min(together, len(todo))) as pool:
            jobs = [pool.submit(download_mode, m, raw) for m in todo]
            for job in progress.track(as_completed(jobs), "downloading multi-mode runs (modes done)", total=len(jobs),
                                      verbose=verbose, leave=True):
                job.result()
    return raw / FOLDER


# ---------------------------------------------------------------- loading

def index(mode: int, raw: Path = data.RAW) -> dict:
    return json.loads(_paths(raw, mode)["index"].read_text())


def _sensors(block: np.ndarray) -> np.ndarray:
    """Drop the time (first column) and the agitator valve (last): the 52 Braatz/Rieth columns."""
    return block[..., 1:53]


def fault_free(mode: int, raw: Path = data.RAW) -> np.ndarray:
    """A mode's fault-free runs: (60 runs, 2,000 readings, 52 sensors), in the order of fault_free_paths."""
    if not _paths(raw, mode)["normal"].exists():
        download(raw, (mode,))
    return _sensors(np.load(_paths(raw, mode)["normal"]))


def transitions(mode: int, raw: Path = data.RAW) -> tuple[np.ndarray, pd.DataFrame]:
    """A mode's transition runs (runs, 2,000 readings, 52 sensors) and a table saying what each one is."""
    if not _paths(raw, mode)["transitions"].exists():
        download(raw, (mode,))
    rows = []
    for e in index(mode, raw)["transitions"]:
        name = e["path"].split("/")
        rows.append({"from mode": mode, "to mode": int(name[3].split("ToMode")[1]),
                     "hours taken": int(name[4].replace("TransitionTime", "")), "starts at hour": e["setpoints"][2][0]})
    return _sensors(np.load(_paths(raw, mode)["transitions"])), pd.DataFrame(rows)


# ---------------------------------------------------------------- which runs are fitted, which are held out
#
# Fixed before anything was measured.
#
# Distinct runs. The files hold 60 fault-free runs per mode, but only 5 are different
# simulations: the 12 "setpoints" SP1..SP12 with the same ramp setting are byte-for-byte the same
# run (see copies_table). The 5 distinct runs of a mode are the SP1 runs, one per ramp setting.
#
# Start-up. Every run starts from the same stored state and takes a few hours to settle into
# its normal wander, so the first 5 hours (100 readings) of every run are left out.
#
# Hold-out. Five rounds. In round k, run k of every mode (6 whole runs) is held out, and the
# models are fitted on the other 4 runs of every mode (24 runs). So every run is scored exactly
# once, by models that never saw it, and every mode is in both the fitted and the held-out set
# in every round. Ranges come from resampling the 30 runs.

DISTINCT = range(5)                 # within a mode's 60 stored runs, the first five (SP1, ramps 0-40) are the distinct ones
START_UP = 100                      # readings left out at the start of every run (5 hours)
SAME_MOMENT = "same moment (the go/no-go measure)"
TOLD = "same moment, mode told to the straight model"
WITH_MOVES = "same moment, with runs that move between modes added"
LAG_SETS = {"+ 1 back": (1,), "+ 1, 2 back": (1, 2), "+ 1, 2, 5 back": (1, 2, 5), "+ 1, 2, 5, 10 back": (1, 2, 5, 10)}


def copies_table() -> pd.DataFrame:
    """Per mode: how many of the 60 stored fault-free runs are different simulations, and how many copies each has."""
    rows = {}
    for m in MODES:
        block = fault_free(m)
        groups: dict = {}
        for i in range(len(block)):
            groups.setdefault(block[i].tobytes(), []).append(i)
        firsts = sorted(g[0] for g in groups.values())
        rows[f"mode {m}"] = {"fault-free runs stored": len(block), "different simulations": len(groups),
                             "copies of each": ", ".join(str(len(g)) for g in groups.values()),
                             "the different ones are stored runs": ", ".join(str(i + 1) for i in firsts),
                             "transition runs": len(index(m)["transitions"])}
    return pd.DataFrame(rows).T


def start_up_table(hours=(1, 2, 5, 10, 20)) -> pd.DataFrame:
    """How far the sensors sit from their later level during the first hours of a run.

    For each mode and each hour: the number of sensors whose average over that hour (across the
    mode's 5 runs) is more than half a spread away from their average over hours 50-100. With 5
    runs a handful of sensors are that far off at any hour by chance; the first hours stand out.
    """
    rows = {}
    for m in MODES:
        d = fault_free(m)[list(DISTINCT)]
        late = d[:, 1000:].reshape(-1, d.shape[2])
        sd = np.where(late.std(0) > 0, late.std(0), 1.0)
        z = np.abs(d.mean(0) - late.mean(0)) / sd
        rows[f"mode {m}"] = {f"hour {h}": int((z[h * 20 - 20: h * 20].mean(0) > 0.5).sum()) for h in hours}
    out = pd.DataFrame(rows).T
    out.columns.name = "sensors more than half a spread from their later level"
    return out


def split(modes: tuple = MODES) -> dict:
    """The fault-free runs used for the measure: {"runs", "folds", "modes", "names"}, one entry per run.

    A run's fold is the round in which it is held out (0 to 4): its number within its mode.
    """
    runs, folds, labels, names = [], [], [], []
    for m in modes:
        block = fault_free(m)
        for k in DISTINCT:
            run = block[k][START_UP:]
            runs.append(run)
            folds.append(np.full(len(run), k))
            labels.append(np.full(len(run), m))
            names.append(f"mode {m}, run {k + 1}")
    return {"runs": runs, "folds": folds, "modes": labels, "names": names}


def split_table() -> pd.DataFrame:
    """Round by round: which runs are held out and which are fitted on."""
    rows = {}
    for k in DISTINCT:
        others = ", ".join(str(j + 1) for j in DISTINCT if j != k)
        rows[f"round {k + 1}"] = {"held out and scored": f"run {k + 1} of every mode (6 runs)",
                                  "fitted on": f"runs {others} of every mode (24 runs)",
                                  "readings held out": 6 * (READINGS - START_UP),
                                  "readings fitted on": 24 * (READINGS - START_UP)}
    return pd.DataFrame(rows).T


def transition_split() -> dict:
    """The transition runs with a round each: a run that takes 10, 20, 30 or 40 hours is held out in
    round 1, 2, 3 or 4. So when one transition between two modes is held out, the others between
    the same two modes (if the files have them) are fitted on."""
    runs, folds = [], []
    for m in MODES:
        block, table = transitions(m)
        for run, hours in zip(block, table["hours taken"]):
            runs.append(run[START_UP:])
            folds.append(np.full(len(run) - START_UP, TRANSITION_HOURS.index(int(hours))))
    return {"runs": runs, "folds": folds}


def measures(n_boot: int = 500, verbose: bool = True) -> dict:
    """Every pooled curvature measure: {name: summary}. Each is saved in results/cache as soon as it is done.

    All use the five rounds described above; the ranges come from resampling the scored runs.
    """
    from . import structure
    s = split()
    common = dict(n_boot=n_boot, verbose=verbose)

    def by_run(what, runs=s["runs"], folds=s["folds"], **kw):
        return lambda: structure.curvature_by_run(runs, data.TEP_CHANNELS, what=f"six modes, {what}", folds=folds,
                                                  **common, **kw)

    out = {SAME_MOMENT: structure.cached("multimode__same_moment", by_run("same moment")),
           TOLD: structure.cached("multimode__mode_told", by_run("mode told", modes=s["modes"]))}
    for name, lags in LAG_SETS.items():
        out[name] = structure.cached("multimode__lags_" + "_".join(map(str, lags)), by_run(name, lags=lags))
    moves = transition_split()
    out[WITH_MOVES] = structure.cached("multimode__with_transitions", by_run(
        "with transition runs", runs=s["runs"] + moves["runs"], folds=s["folds"] + moves["folds"]))
    progress.note("six modes: pooled measures done")
    return out


def within_modes(n_boot: int = 500, verbose: bool = True) -> dict:
    """The same-moment measure inside each mode on its own: {"mode 1": summary, ...}.

    Five rounds as before, within the mode: each of its 5 runs is held out in turn and predicted
    by models fitted on the other 4. With 5 runs the ranges are rough.
    """
    from . import structure
    out = {}
    for m in MODES:
        s = split((m,))
        out[f"mode {m}"] = structure.cached(f"multimode__within_mode{m}", lambda: structure.curvature_by_run(
            s["runs"], data.TEP_CHANNELS, what=f"inside mode {m}", folds=s["folds"], n_boot=n_boot, verbose=verbose))
    return out


def measure_table(by_name: dict) -> pd.DataFrame:
    """One row per measure: the straight share (0 to 1) and the curved share in points (100 x share), with ranges.

    The curved shares here are small fractions of a point, so they are shown in points to keep
    their digits: +0.07 points is a curved share of 0.0007.
    """
    rows = {name: {"straight share": s["straight"], "straight, low": s["straight_low"], "straight, high": s["straight_high"],
                   "curved share, points": 100 * s["curved"], "curved low, points": 100 * s["curved_low"],
                   "curved high, points": 100 * s["curved_high"],
                   "sensors gaining over 5 points": s["sensors_with_curved_over_5pts"],
                   "sensors gaining over 10 points": s["sensors_with_curved_over_10pts"],
                   "held-out runs": s["runs"], "held-out readings": s["readings"]}
            for name, s in by_name.items()}
    out = pd.DataFrame(rows).T.astype({"sensors gaining over 5 points": int, "sensors gaining over 10 points": int,
                                       "held-out runs": int, "held-out readings": int})
    out.index.name = "measure"
    return out


def sensor_table(summary: dict) -> pd.DataFrame:
    """Every sensor's straight and curved share (with the range of its curved share), most curved first."""
    info = data.TEP_INFO.set_index("variable")["what it measures"]
    return pd.concat([info.reindex(summary["table"].index), summary["table"]], axis=1).sort_values("curved", ascending=False)


def most_curved(summary: dict, over: float = 0.10) -> list[str]:
    t = summary["table"]["curved"].sort_values(ascending=False)
    return list(t.index[t > over])


def readings(modes: tuple = MODES) -> tuple[np.ndarray, np.ndarray]:
    """All the readings used for the measure, stacked, and the mode of each."""
    s = split(modes)
    return np.vstack(s["runs"]), np.concatenate(s["modes"])


def between_modes_share() -> pd.DataFrame:
    """Per sensor: how much of its movement in the pooled data is the gap between the modes' averages.

    1 means the sensor only differs from mode to mode and barely moves inside a mode; 0 means its
    average is the same in every mode (a level held at one set-point, say) and all its movement
    is wander inside the modes.
    """
    from . import structure
    share = structure.between_share(*readings())
    out = pd.DataFrame({"what it measures": list(data.TEP_INFO["what it measures"]),
                        "share of movement that is between modes": share}, index=data.TEP_CHANNELS)
    out.index.name = "sensor"
    return out.sort_values("share of movement that is between modes")


def within_curved(inside: dict, over: float = 0.10) -> pd.DataFrame:
    """Inside single modes: every sensor that gains more than `over` from curves, one row per (mode, sensor).

    Beside the shares: the other sensor that on its own says most about it inside that mode, and
    how much of the time the sensor sits exactly at its highest value. A valve that is fully open
    (or held against a limit) most of the time shows up here.
    """
    from . import structure
    info = data.TEP_INFO.set_index("variable")["what it measures"]
    rows = {}
    for name, summary in inside.items():
        m = int(name.split()[-1])
        X, _ = readings((m,))
        t = summary["table"]
        for s in t.index[t["curved"] > over]:
            j = data.TEP_CHANNELS.index(s)
            i, share = structure.best_single_predictor(X, j)
            rows[(name, s)] = {"what it measures": info[s], "straight": t.loc[s, "straight"], "curved": t.loc[s, "curved"],
                               "curved_low": t.loc[s, "curved_low"], "curved_high": t.loc[s, "curved_high"],
                               "best single predictor": data.TEP_CHANNELS[i], "which measures": info[data.TEP_CHANNELS[i]],
                               "share of time at its highest value": float((X[:, j] == X[:, j].max()).mean())}
    out = pd.DataFrame(rows).T
    out.index.names = ["mode", "sensor"]
    return out


def within_scatter(table: pd.DataFrame):
    """Each row of within_curved as a panel: the sensor against its best single predictor, inside its mode."""
    from . import plots
    panels = []
    for (name, s), row in table.iterrows():
        X, _ = readings((int(name.split()[-1]),))
        partner = row["best single predictor"]
        panels.append({"x": X[:, data.TEP_CHANNELS.index(partner)], "y": X[:, data.TEP_CHANNELS.index(s)],
                       "title": f"{name}: {row['what it measures']}", "across": f"{row['which measures']} ({partner})",
                       "up": s})
    return plots.scatter_panels(panels)


def best_partner(sensors: list) -> pd.DataFrame:
    """For each sensor: the one other sensor that on its own says most about it."""
    from . import structure
    X, _ = readings()
    info = data.TEP_INFO.set_index("variable")["what it measures"]
    rows = {}
    for s in sensors:
        i, share = structure.best_single_predictor(X, data.TEP_CHANNELS.index(s))
        rows[s] = {"what it measures": info[s], "best single predictor": data.TEP_CHANNELS[i],
                   "which measures": info[data.TEP_CHANNELS[i]], "share explained": share}
    out = pd.DataFrame(rows).T
    out.index.name = "sensor"
    return out


def curved_scatter(sensors: list):
    """Each sensor against its best single predictor, one small panel per mode (plots.scatter_by_mode)."""
    from . import plots
    X, mode = readings()
    partners = best_partner(sensors)["best single predictor"]
    info = data.TEP_INFO.set_index("variable")["what it measures"]
    describe = {c: f"{info[c]} ({c})" for c in data.TEP_CHANNELS}
    return plots.scatter_by_mode(X, mode, list(zip(sensors, partners)), data.TEP_CHANNELS, describe)


def standard_plant_check(inside: dict) -> pd.DataFrame:
    """Mode 1 beside the standard plant of notebook 04 (20 fault-free Rieth runs, same-moment measure).

    Both should show a curved share of about zero. The straight shares need not agree: mode 1 here
    is held at different conditions and wanders less (see column_check).
    """
    from . import structure
    plant = structure.cached("lags_plant__lags_none", lambda: structure.curvature_by_run(
        list(data.rieth_runs("fault_free_training", 20)[0]), data.TEP_CHANNELS, what="standard plant, same moment"))
    return measure_table({"standard plant (notebook 04, 20 runs)": plant, "mode 1 here (5 runs)": inside["mode 1"]})


def column_check() -> pd.DataFrame:
    """Mode 1 here against the fault-free Rieth runs, column by column: are these the same 52 sensors?

    For every column: its average in mode 1's fault-free runs and in the Rieth fault-free training
    runs, and the gap between them measured in units of the Rieth runs' own spread.
    """
    here = fault_free(1)[list(DISTINCT)].reshape(-1, 52)
    rieth = data.rieth_runs("fault_free_training", 20)[0].reshape(-1, 52)
    gap = (here.mean(0) - rieth.mean(0)) / np.where(rieth.std(0) > 0, rieth.std(0), 1.0)
    out = pd.DataFrame({"what it measures": list(data.TEP_INFO["what it measures"]), "average, mode 1 here": here.mean(0),
                        "average, Rieth runs": rieth.mean(0), "gap, in Rieth spreads": gap,
                        "spread here / spread in Rieth": here.std(0) / np.where(rieth.std(0) > 0, rieth.std(0), 1.0)},
                       index=data.TEP_CHANNELS)
    out.index.name = "sensor"
    return out


def mode_table() -> pd.DataFrame:
    """What sets the six modes apart: the average of a few key readings in each mode's fault-free runs."""
    keys = {"XMEAS(17)": "product flow", "XMEAS(7)": "reactor pressure", "XMEAS(8)": "reactor level",
            "XMEAS(9)": "reactor temperature", "XMEAS(40)": "% G in product", "XMEAS(41)": "% H in product"}
    rows = {}
    for m in MODES:
        block = fault_free(m)[list(DISTINCT)].reshape(-1, 52)
        rows[f"mode {m}"] = {f"{label} ({c})": block[:, data.TEP_CHANNELS.index(c)].mean() for c, label in keys.items()}
    return pd.DataFrame(rows).T

