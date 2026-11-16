"""Metal etch: a LAM 9600 plasma etcher, 129 wafers (Wise et al. 1999, Eigenvector Research). Notebook 07.

The file MACHINE_Data.mat (about 2 MB) holds 108 normal wafers and 21 wafers with a fault
induced on purpose, from three experiments (29, 31 and 33) run weeks apart. Each wafer is a
batch: about 100 readings, one a second, through two steps of the etch recipe (numbered 4 and 5
in the file). The step plays the role of the stage.
"""
from __future__ import annotations

import functools
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import batch, progress, structure
from .data import RAW, Dataset, Run

URL = "https://eigenvector.com/data/Etch/MACHINE_Data.mat"
FOLDER, FILE = "metal_etch", "MACHINE_Data.mat"
AGENT = "adlab-data-loader/0.1"
EXPERIMENTS = ("29", "31", "33")
SHORTEST = 50                    # a normal wafer with fewer readings is an incomplete recording and is left out
STEPS = (4, 5)
LEFT_OUT = {"Time": "the time of the reading, not a measurement",
            "Step Number": "the recipe step: used as the stage of the batch, not as a sensor"}
INFO = pd.Series({
    "BCl3 Flow": "Flow of boron trichloride gas into the chamber",
    "Cl2 Flow": "Flow of chlorine gas into the chamber",
    "RF Btm Pwr": "Radio-frequency power to the bottom electrode, under the wafer",
    "RF Btm Rfl Pwr": "Bottom radio-frequency power reflected back",
    "Endpt A": "Endpoint signal: plasma light used to tell when the metal is etched through",
    "He Press": "Helium pressure behind the wafer, which cools it",
    "Pressure": "Chamber pressure",
    "RF Tuner": "Bottom power's matching unit: tuning setting",
    "RF Load": "Bottom power's matching unit: load setting",
    "RF Phase Err": "Bottom power's matching unit: phase error",
    "RF Pwr": "A second radio-frequency power reading (the file does not say which)",
    "RF Impedance": "Bottom power's matching unit: impedance",
    "TCP Tuner": "Top coil's matching unit: tuning setting",
    "TCP Phase Err": "Top coil's matching unit: phase error",
    "TCP Impedance": "Top coil's matching unit: impedance",
    "TCP Top Pwr": "Power to the top coil, which makes the plasma",
    "TCP Rfl Pwr": "Top coil power reflected back",
    "TCP Load": "Top coil's matching unit: load setting",
    "Vat Valve": "Position of the valve that holds the chamber pressure",
}, name="what it measures")
CHANNELS = list(INFO.index)
FAULT_KINDS = {"TCP": "top coil power", "RF": "radio-frequency power", "Pr": "chamber pressure",
               "Cl2": "chlorine flow", "BCl3": "BCl3 flow", "He": "helium chuck"}
# Detectors: within each experiment, normal wafers in order are dealt these roles in turn.
ROLES = ("fit", "fit", "fit", "line", "check")
CACHE = "etch"
TOLD_NAME = "etch_told"


def download(raw: Path = RAW) -> Path:
    """Fetch MACHINE_Data.mat (about 2 MB) into data/raw/metal_etch/. Nothing if it is already there."""
    path = Path(raw) / FOLDER / FILE
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        blob = response.read()
    part = path.with_suffix(".part")
    part.write_bytes(blob)
    part.replace(path)
    return path


def in_time_order(w: np.ndarray) -> np.ndarray:
    """A wafer's rows sorted by time (column 0). In the file, the last row of almost every wafer is
    its first reading, filed at the end."""
    return w[np.argsort(w[:, 0], kind="stable")]


def _wafer(name: str, w, columns: list, fault: str | None) -> dict:
    raw_rows = np.asarray(w, float)
    rows = in_time_order(raw_rows)
    return {"wafer": name[:5], "experiment": name[1:3], "fault": fault,
            "kind": None if fault is None else FAULT_KINDS[fault.split()[0]],
            "X": rows[:, [columns.index(c) for c in CHANNELS]], "step": rows[:, columns.index("Step Number")].astype(int),
            "time": rows[:, columns.index("Time")], "readings": len(rows),
            "first row out of place": bool(len(raw_rows) > 1 and np.any(np.diff(raw_rows[:, 0]) < 0))}


@functools.lru_cache(maxsize=2)
def _read(raw: str) -> dict:
    from scipy.io import loadmat
    d = loadmat(download(Path(raw)), squeeze_me=True, struct_as_record=False)["LAMDATA"]
    columns = [str(v).strip() for v in d.variables]
    normal = [_wafer(str(n).strip(), w, columns, None) for n, w in zip(d.calib_names, d.calibration)]
    faulty = [_wafer(str(n).strip(), w, columns, str(f).strip())
              for n, f, w in zip(d.test_names, d.fault_names, d.test)]
    return {"columns": columns, "normal": normal, "faulty": faulty,
            "information": [str(line).rstrip() for line in np.atleast_1d(d.INFORMATION)]}


def information(raw: Path = RAW) -> str:
    """The file's own INFORMATION field."""
    return "\n".join(_read(str(raw))["information"])


def normal_wafers(raw: Path = RAW) -> list:
    """The normal wafers used: every one with at least SHORTEST readings, in the file's order
    (experiment 29, 31, 33; wafer numbers rising)."""
    return [w for w in _read(str(raw))["normal"] if w["readings"] >= SHORTEST]


def faulty_wafers(raw: Path = RAW) -> list:
    return list(_read(str(raw))["faulty"])


def wafer_table(raw: Path = RAW) -> pd.DataFrame:
    """Per experiment: normal wafers in the file and used, faulty wafers, readings per wafer."""
    r = _read(str(raw))
    rows = {}
    for e in EXPERIMENTS:
        normal = [w for w in r["normal"] if w["experiment"] == e]
        faulty = [w for w in r["faulty"] if w["experiment"] == e]
        rows[f"experiment {e}"] = {
            "normal wafers in the file": len(normal),
            "normal wafers used": sum(w["readings"] >= SHORTEST for w in normal),
            "faulty wafers": len(faulty),
            "readings per wafer, fewest": min(w["readings"] for w in normal + faulty),
            "readings per wafer, middle": int(np.median([w["readings"] for w in normal + faulty])),
            "readings per wafer, most": max(w["readings"] for w in normal + faulty)}
    return pd.DataFrame(rows).T


def oddities(raw: Path = RAW) -> pd.DataFrame:
    """What was found in the file before measuring anything: rows out of time order, and short wafers."""
    r = _read(str(raw))
    rows = {}
    for label, ws in (("normal", r["normal"]), ("faulty", r["faulty"])):
        steps_in_order = all(np.all(np.diff(w["step"]) >= 0) for w in ws)
        rows[f"{label} wafers"] = {
            "wafers": len(ws),
            "with the first reading filed last": sum(w["first row out of place"] for w in ws),
            "with two readings stamped with the same time": ", ".join(
                w["wafer"] for w in ws if np.any(np.diff(w["time"]) == 0)) or "none",
            "step 4 before step 5 once in time order": "every wafer" if steps_in_order else "not every wafer",
            "wafers under 50 readings": ", ".join(f"{w['wafer']} ({w['readings']})" for w in ws if w["readings"] < SHORTEST) or "none",
            "wafers with under 20 readings in a step": ", ".join(
                f"{w['wafer']} (step {s}: {int((w['step'] == s).sum())})" for w in ws for s in STEPS
                if (w["step"] == s).sum() < 20 and w["readings"] >= 20) or "none"}
    return pd.DataFrame(rows).T


def columns_table(raw: Path = RAW) -> pd.DataFrame:
    """Every column of the file: kept or left out (and why), and how it moves in the normal wafers used."""
    r = _read(str(raw))
    ws = normal_wafers(raw)
    rows = {}
    for c in r["columns"]:
        if c in LEFT_OUT:
            rows[c] = {"kept": "no", "what it is": LEFT_OUT[c]}
            continue
        x = np.concatenate([w["X"][:, CHANNELS.index(c)] for w in ws])
        values, counts = np.unique(x, return_counts=True)
        rows[c] = {"kept": "yes" if len(values) > 1 else "no: never moves", "what it is": INFO[c],
                   "distinct values": len(values), "share at commonest value": counts.max() / len(x),
                   "lowest": x.min(), "highest": x.max()}
    out = pd.DataFrame(rows).T
    out.index.name = "column"
    return out


# ---------------------------------------------------------------- curvature: which wafers are fitted, which held out

def split(raw: Path = RAW, without: tuple = ()) -> dict:
    """The normal wafers for the curvature measure: {"runs", "stages", "rounds", "names", "experiments"}.

    Rounds from batch.staggered_rounds by experiment: each round holds out about a fifth of the
    wafers of every experiment and fits on the rest of every experiment. `without` leaves wafers
    out (for the outlier-run rule); the others keep their rounds.
    """
    ws = normal_wafers(raw)
    rounds = batch.staggered_rounds([w["experiment"] for w in ws])
    keep = [i for i, w in enumerate(ws) if w["wafer"] not in without]
    return {"runs": [ws[i]["X"] for i in keep], "stages": [ws[i]["step"] for i in keep], "rounds": rounds[keep],
            "names": [ws[i]["wafer"] for i in keep], "experiments": [ws[i]["experiment"] for i in keep]}


def split_table(raw: Path = RAW) -> pd.DataFrame:
    s = split(raw)
    return batch.round_table(s["rounds"], s["experiments"], "experiment")


def outlier_check(raw: Path = RAW) -> tuple[pd.DataFrame, list]:
    s = split(raw)
    return batch.outlier_check(s["runs"], CHANNELS, s["names"])


def _tag(without: tuple) -> str:
    return CACHE + ("_without_" + "_".join(without) if without else "")


def measures(without: tuple = (), n_boot: int = 500, verbose: bool = True) -> dict:
    """The pooled measures (headline, step told, lags), each cached as soon as it is done."""
    s = split(without=without)
    out = batch.measures(s["runs"], s["stages"], s["rounds"], CHANNELS, _tag(without), n_boot=n_boot, verbose=verbose)
    progress.note(f"{_tag(without)}: pooled measures done")
    return out


def within_steps(without: tuple = (), n_boot: int = 500, verbose: bool = True) -> dict:
    s = split(without=without)
    return batch.within_stages(s["runs"], s["stages"], s["rounds"], CHANNELS, _tag(without),
                               label=lambda k: f"step {k}", n_boot=n_boot, verbose=verbose)


def between_table(without: tuple = ()) -> pd.DataFrame:
    s = split(without=without)
    return batch.between_table(s["runs"], s["stages"], CHANNELS, s["experiments"], "experiments", INFO)


def sensor_table(summary: dict) -> pd.DataFrame:
    return batch.sensor_table(summary, INFO)


def readings(without: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    s = split(without=without)
    return np.vstack(s["runs"]), np.concatenate(s["stages"])


def best_partner(sensors: list, without: tuple = ()) -> pd.DataFrame:
    return batch.best_partners(readings(without)[0], sensors, CHANNELS, INFO)


def shape_clues(sensors: list, without: tuple = ()) -> pd.DataFrame:
    return batch.shape_clues(*readings(without), sensors, CHANNELS)


def curved_scatter(sensors: list, without: tuple = ()):
    """Each sensor against its best single predictor, coloured by recipe step."""
    from . import plots
    X, step = readings(without)
    partners = best_partner(sensors, without)["best single predictor"]
    describe = {c: f"{INFO[c]} ({c})" for c in CHANNELS}
    return plots.scatter_by_stage(X, step, list(zip(sensors, partners)), CHANNELS, describe,
                                  {k: f"step {k}" for k in STEPS})


EXPERIMENT_TOLD = "same moment, experiment told to the straight model"
BOTH_TOLD = "same moment, step and experiment told to the straight model"
TENTHS_TOLD = "same moment, step and tenth of the step told to the straight model"


def tenths(step: np.ndarray, slices: int = 10) -> np.ndarray:
    """For each reading of one wafer: 10 x its step number + which tenth of that step it falls in (0 to 9)."""
    step = np.asarray(step)
    out = np.empty(len(step), int)
    for s in np.unique(step):
        idx = np.flatnonzero(step == s)
        out[idx] = 10 * int(s) + np.minimum(slices * np.arange(len(idx)) // len(idx), slices - 1)
    return out


def experiment_checks(n_boot: int = 500, verbose: bool = True) -> dict:
    """Added after the headline was seen (not part of the rules fixed in advance).

    The three experiments sit at different levels, and some curved sensors have most of their
    movement between experiments. These ask whether the curved share is just those level shifts:
    the straight model told the experiment (its own level per experiment), told both step and
    experiment (its own level for each step of each experiment), and the measure inside each
    experiment on its own (its wafers only, the same rounds). And one more: the straight model
    told where in the step each reading is (its own level for each tenth of each step), the
    stage used for Pensim, to see how much of the gain is the phase of the wafer.
    """
    s = split()
    folds = batch.as_folds(s["runs"], s["rounds"])
    exp = [np.full(len(r), int(e)) for r, e in zip(s["runs"], s["experiments"])]
    both = [100 * st + e for st, e in zip(s["stages"], exp)]

    def by(what, runs=s["runs"], f=folds, **kw):
        return lambda: structure.curvature_by_run(runs, CHANNELS, folds=f, n_boot=n_boot, verbose=verbose,
                                                  what=f"{CACHE}, {what}", **kw)

    out = {EXPERIMENT_TOLD: structure.cached(f"{CACHE}__experiment_told", by("experiment told", modes=exp)),
           BOTH_TOLD: structure.cached(f"{CACHE}__step_experiment_told", by("step and experiment told", modes=both)),
           TENTHS_TOLD: structure.cached(f"{CACHE}__tenths_told", by("step and tenth told",
                                                                    modes=[tenths(st) for st in s["stages"]]))}
    for e in EXPERIMENTS:
        keep = [i for i, x in enumerate(s["experiments"]) if x == e]
        runs = [s["runs"][i] for i in keep]
        out[f"inside experiment {e}"] = structure.cached(
            f"{CACHE}__inside_experiment_{e}", by(f"inside experiment {e}", runs=runs, f=[folds[i] for i in keep]))
    return out


def scatter_by_experiment(sensors: list):
    """The same pairs as curved_scatter, each reading coloured by experiment instead of step."""
    from . import plots
    s = split()
    X = np.vstack(s["runs"])
    exp = np.concatenate([np.full(len(r), int(e)) for r, e in zip(s["runs"], s["experiments"])])
    partners = best_partner(sensors)["best single predictor"]
    describe = {c: f"{INFO[c]} ({c})" for c in CHANNELS}
    return plots.scatter_by_stage(X, exp, list(zip(sensors, partners)), CHANNELS, describe,
                                  {int(e): f"experiment {e}" for e in EXPERIMENTS})


def gain_by_position(sensors: list, edges: tuple = (0, 5, 15, 45, 55, 65, 200)) -> pd.DataFrame:
    """Where in a wafer each sensor's curved gain comes from (headline measure, same rounds).

    For each stretch of readings (counted from the start of the wafer, one a second): the share
    of the sensor's whole curved gain that comes from that stretch. The last row is the share of
    all readings in each stretch, for comparison: a stretch holding far more of the gain than of
    the readings is where the flexible model does its work.
    """
    s = split()
    h = structure.cached(f"{CACHE}__held_out", lambda: structure.held_out_by_run(
        s["runs"], folds=batch.as_folds(s["runs"], s["rounds"]), verbose=False))
    pos = h["position"]
    labels = [f"seconds {a}-{b}" if b < 200 else f"seconds {a} on" for a, b in zip(edges[:-1], edges[1:])]
    rows = {}
    for name in sensors:
        j = CHANNELS.index(name)
        gain = (h["Z"][:, j] - h["straight"][:, j]) ** 2 - (h["Z"][:, j] - h["both"][:, j]) ** 2
        rows[name] = {lab: gain[(pos >= a) & (pos < b)].sum() / gain.sum() for lab, a, b in zip(labels, edges[:-1], edges[1:])}
    rows["share of all readings"] = {lab: float(((pos >= a) & (pos < b)).mean()) for lab, a, b in zip(labels, edges[:-1], edges[1:])}
    out = pd.DataFrame(rows).T
    out.index.name = "sensor"
    return out


def over_a_wafer(sensors: list, experiment: str = "29", wafers: int = 6):
    """A few normal wafers of one experiment, each sensor over the ~100 seconds of a wafer."""
    from . import plots
    ws = [w for w in normal_wafers() if w["experiment"] == experiment][:wafers]
    return plots.over_a_batch([w["X"] for w in ws], [w["step"] for w in ws], CHANNELS, sensors,
                              {c: f"{INFO[c]} ({c})" for c in CHANNELS},
                              f"{len(ws)} normal wafers of experiment {experiment}")


# ---------------------------------------------------------------- detectors

def fault_span_table(raw: Path = RAW, beyond: float = 3.0) -> pd.DataFrame:
    """Is each induced fault there from the start of the wafer to its end?

    Each step of each wafer is cut into thirds. For every third: the number of sensors whose
    average sits more than `beyond` spreads from the same third of the same experiment's normal
    wafers (the spread is across those normal wafers). A fault present all through the wafer
    shows sensors out in every third. A labelling check only: no detector is involved.
    """
    def thirds(w):
        out = np.full((len(STEPS), 3, len(CHANNELS)), np.nan)
        for a, s in enumerate(STEPS):
            part = w["X"][w["step"] == s]
            if len(part) >= 6:
                for b, chunk in enumerate(np.array_split(part, 3)):
                    out[a, b] = chunk.mean(0)
        return out

    rows = {}
    for w in faulty_wafers(raw):
        same = np.array([thirds(n) for n in normal_wafers(raw) if n["experiment"] == w["experiment"]])
        mu, sd = np.nanmean(same, 0), np.nanstd(same, 0)
        z = np.abs(thirds(w) - mu) / np.where(sd > 0, sd, 1.0)
        row = {"fault": w["fault"], "experiment": w["experiment"], "readings": w["readings"]}
        for a, s in enumerate(STEPS):
            for b, third in enumerate(("first", "middle", "last")):
                row[f"step {s}, {third} third"] = "no data" if np.isnan(z[a, b]).all() else int((z[a, b] > beyond).sum())
        row["largest distance, in spreads"] = float(np.nanmax(z))
        rows[w["wafer"]] = row
    out = pd.DataFrame(rows).T
    out.index.name = "wafer"
    return out


def roles(raw: Path = RAW) -> list[str]:
    """For the detectors, each normal wafer's role: within each experiment, wafers in order are
    dealt ROLES in turn (fit, fit, fit, line, check, fit, ...)."""
    ws = normal_wafers(raw)
    out, seen = [], {}
    for w in ws:
        k = seen.get(w["experiment"], 0)
        out.append(ROLES[k % len(ROLES)])
        seen[w["experiment"]] = k + 1
    return out


def role_table(raw: Path = RAW) -> pd.DataFrame:
    ws, r = normal_wafers(raw), roles(raw)
    rows = {f"experiment {e}": {role: sum(w["experiment"] == e and x == role for w, x in zip(ws, r))
                                for role in ("fit", "line", "check")} for e in EXPERIMENTS}
    out = pd.DataFrame(rows).T
    out.columns = ["fitted on", "set the alarm line", "only scored (false-alarm check)"]
    return out


def dataset(told: bool = False, raw: Path = RAW) -> Dataset:
    """Faulty wafers as runs (every reading labelled faulty), the line and check wafers as
    val_normal, and the fit wafers of all three experiments joined as the fit data.

    told=True appends the recipe step as a last column, for the stage-told straight-link detector.
    """
    def x(w):
        return np.column_stack([w["X"], w["step"]]) if told else w["X"]

    ws, r = normal_wafers(raw), roles(raw)
    fit = np.vstack([x(w) for w, role in zip(ws, r) if role == "fit"])
    runs = [Run(w["wafer"], x(w), np.ones(w["readings"], int), fit, group=w["kind"],
                meta={"experiment": w["experiment"], "fault": w["fault"]}) for w in faulty_wafers(raw)]
    val = [Run(w["wafer"], x(w), np.zeros(w["readings"], int), fit, group="normal",
               meta={"experiment": w["experiment"], "role": role}) for w, role in zip(ws, r) if role != "fit"]
    return Dataset(TOLD_NAME if told else CACHE, runs, CHANNELS + (["Step Number"] if told else []), 1.0, val,
                   fit_lengths=[w["readings"] for w, role in zip(ws, r) if role == "fit"])


def methods() -> list[str]:
    from . import run
    from . import detectors as det
    return [m for m in run.METHODS if m in det.available()] + ["link_straight_told"]


def score(verbose: bool = True):
    """Score every faulty, line and check wafer with every detector (cached in results/cache)."""
    from . import run
    run.score_all([dataset()], methods=[m for m in methods() if m in run.METHODS], n_seeds=5, verbose=verbose)
    run.score_all([dataset(told=True)], methods=["link_straight_told"], verbose=verbose)


def report(per_experiment: bool = True, n_boot: int = 1000, verbose: bool = True) -> dict:
    """batch.report on the metal etch wafers: line per experiment (the primary) or one line for all."""
    from . import run
    from . import detectors as det
    score(verbose)

    def compute():
        ds = dataset()
        per_method = {det.REGISTRY[m].name: [run.load_scores(TOLD_NAME if m == "link_straight_told" else ds.name, m, s)[:2]
                                             for s in run._seeds(m, 5)] for m in methods()}
        return batch.report(per_method, [r.group for r in ds.runs], [r.meta["experiment"] for r in ds.runs],
                            [r.meta["experiment"] for r in ds.val_normal], [r.meta["role"] for r in ds.val_normal],
                            per_group=per_experiment, n_boot=n_boot, verbose=verbose,
                            what="resampling faulty wafers")

    return structure.cached(f"etch_report_{'per_experiment' if per_experiment else 'one_line'}", compute)


def moved(per_experiment: bool = True) -> list[int]:
    ds = dataset()
    v = ds.val_normal
    return batch.sensors_moved(ds.runs[0].fit, [r.X for r in ds.runs], [r.meta["experiment"] for r in ds.runs],
                               [r.X for r in v], [r.meta["experiment"] for r in v], [r.meta["role"] for r in v],
                               per_group=per_experiment)


def kinds(res: dict, per_experiment: bool = True) -> pd.DataFrame:
    return batch.fault_kinds(res, moved(per_experiment), [r.group for r in dataset().runs])


def fault_list(raw: Path = RAW) -> pd.DataFrame:
    """The 21 faulty wafers: the fault as the file names it, its kind, and the experiment."""
    ws = faulty_wafers(raw)
    return pd.DataFrame({"fault": [w["fault"] for w in ws], "kind of fault": [w["kind"] for w in ws],
                         "experiment": [w["experiment"] for w in ws], "readings": [w["readings"] for w in ws]},
                        index=pd.Index([w["wafer"] for w in ws], name="wafer"))
