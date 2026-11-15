"""Datasets as lists of runs with one common shape.

Every run carries
    X      (T, d)  the series to score
    y      (T,)    1 where the sample is labelled anomalous
    fit    (n, d)  normal data the detector is fitted on for this run
    group          anomaly type or fault, used for per-type reporting

Datasets come from public archives (GitHub, Harvard Dataverse) and are cached under data/raw/.
"""
from __future__ import annotations

import functools
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = {
    "skab": "https://github.com/waico/SKAB/archive/refs/heads/master.zip",
    "tep": "https://github.com/camaramm/tennessee-eastman-profBraatz/archive/refs/heads/master.zip",
}


@dataclass
class Run:
    name: str
    X: np.ndarray
    y: np.ndarray
    fit: np.ndarray
    group: str
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    runs: list[Run]
    channels: list[str]
    sample_seconds: float
    # Separate normal series for deployable thresholds, scored with the same fit as the runs.
    # None means thresholds are calibrated leave-one-run-out on the other runs' normal samples.
    val_normal: list[Run] | None = None
    # How the primary metric sets its threshold (alarm line):
    #   "pooled"   one line across runs, from the normal samples of the scored runs themselves.
    #   "per_run"  each run's own line, right when each run has its own fit and recording session.
    #   "matched"  one line, from separate fault-free runs over the same stretch of a run as the
    #              fault covers (samples from `matched_start` on). Right when the normal part of
    #              a scored run comes before the fault and looks calmer than the hours after it,
    #              as in Tennessee Eastman, where every run starts from the same state.
    calibration: str = "pooled"
    matched_start: int | None = None
    # The dataset whose runs are those fault-free runs (scored and cached under its own name).
    # None means the fault-free runs are val_normal.
    matched_from: Dataset | None = None

    def calibration_runs(self) -> list[Run]:
        """The fault-free runs that set the alarm line under "matched" calibration."""
        return self.matched_from.runs if self.matched_from is not None else (self.val_normal or [])


def download(name: str, raw: Path = RAW) -> Path:
    """Fetch a dataset's GitHub archive and unpack it into data/raw/<name>. Skips if already there."""
    target = raw / name
    if target.exists() and any(target.iterdir()):
        return target
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCES[name], timeout=180) as response:
        blob = response.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            parts = PurePosixPath(member).parts[1:]  # drop the "<repo>-master/" folder
            if not parts or member.endswith("/") or ".." in parts:
                continue
            if name == "skab" and parts[0] != "data":
                continue
            out = target.joinpath(*parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(member))
    return target


# ---------------------------------------------------------------- SKAB

_SKAB_OTHER = {
    1: "fluid leak or addition", 2: "fluid leak or addition", 3: "fluid leak or addition",
    4: "fluid leak or addition", 5: "rotor imbalance", 6: "rotor imbalance", 7: "rotor imbalance",
    8: "rotor imbalance", 9: "rotor imbalance", 10: "water level change", 11: "water level change",
    12: "cavitation", 13: "cavitation", 14: "hot water inflow",
}


def _skab_group(folder: str, stem: str) -> str:
    if folder == "valve1":
        return "inlet valve closed"
    if folder == "valve2":
        return "outlet valve closed"
    return _SKAB_OTHER[int(stem)]


def load_skab(raw: Path = RAW, fit_rows: int = 400) -> Dataset:
    """SKAB water-pump rig: 34 experiments, 8 sensors at 1 Hz.

    Follows the SKAB benchmark protocol: each experiment's first `fit_rows` rows are the
    normal data its detector is fitted on, the rest is scored. One file (other/2) turns
    anomalous at row 104, so its fit stops there. Each experiment is its own recording
    session (temperatures drift between and within them), so thresholds are set per run.
    """
    base = raw / "skab" / "data"
    if not base.exists():
        download("skab", raw)
    runs, channels = [], None
    for path in sorted(base.glob("*/*.csv")):
        if path.parent.name == "anomaly-free":
            continue
        df = pd.read_csv(path, sep=";")
        cols = [c for c in df.columns if c not in ("datetime", "anomaly", "changepoint")]
        channels = channels or cols
        X = df[cols].to_numpy(float)
        y = df["anomaly"].to_numpy(int)
        onset = int(np.flatnonzero(y)[0])
        n_fit = min(fit_rows, onset)
        runs.append(Run(
            name=f"{path.parent.name}/{path.stem}",
            X=X[n_fit:], y=y[n_fit:], fit=X[:n_fit],
            group=_skab_group(path.parent.name, path.stem),
            meta={"fit_rows": n_fit},
        ))
    return Dataset("skab", runs, channels, 1.0, None, calibration="per_run")


# ---------------------------------------------------------------- Tennessee Eastman (Braatz set)

TEP_FAULTS = {
    1: "step: A/C feed ratio", 2: "step: B composition", 3: "step: D feed temperature",
    4: "step: reactor cooling water temp", 5: "step: condenser cooling water temp",
    6: "step: A feed loss", 7: "step: C header pressure loss",
    8: "random: A, B, C feed composition", 9: "random: D feed temperature",
    10: "random: C feed temperature", 11: "random: reactor cooling water temp",
    12: "random: condenser cooling water temp", 13: "drift: reaction kinetics",
    14: "sticking: reactor cooling water valve", 15: "sticking: condenser cooling water valve",
    16: "unknown", 17: "unknown", 18: "unknown", 19: "unknown", 20: "unknown",
    21: "valve stuck: stream 4",
}
TEP_ONSET = 160  # in every test run the fault starts at sample 161 (index 160)
TEP_CHANNELS = [f"XMEAS({i})" for i in range(1, 42)] + [f"XMV({i})" for i in range(1, 12)]


def load_tep(raw: Path = RAW) -> Dataset:
    """Tennessee Eastman, Braatz lab files: 52 variables every 3 minutes.

    Detectors are fitted on the normal training run d00.dat (500 samples). Each of the 21
    fault test runs has 960 samples, normal for the first 160. The normal test run
    d00_te.dat is held out: its samples 161-960 set the alarm lines (matched-window
    calibration; see Dataset). It is the only such run in these files, so the lines rest on
    one run of 800 samples.
    """
    base = raw / "tep"
    if not (base / "d00.dat").exists():
        download("tep", raw)
    fit = np.loadtxt(base / "d00.dat")
    if fit.shape[0] == 52:  # the normal training file is stored transposed
        fit = fit.T
    runs = []
    for f in range(1, 22):
        X = np.loadtxt(base / f"d{f:02d}_te.dat")
        y = np.zeros(len(X), int)
        y[TEP_ONSET:] = 1
        runs.append(Run(f"fault{f:02d}", X, y, fit, group=f"{f:02d} {TEP_FAULTS[f]}",
                        meta={"fault": f, "onset": TEP_ONSET}))
    normal = np.loadtxt(base / "d00_te.dat")
    val = [Run("normal_te", normal, np.zeros(len(normal), int), fit, group="normal")]
    return Dataset("tep", runs, TEP_CHANNELS, 180.0, val, calibration="matched", matched_start=TEP_ONSET)


# ---------------------------------------------------------------- Tennessee Eastman (Rieth set)
#
# Rieth, Amsel, Tran, Cook (2017), "Additional Tennessee Eastman Process Simulation Data for
# Anomaly Detection Evaluation", Harvard Dataverse, doi:10.7910/DVN/6C3JR1 (public domain).
# The same plant model as the Braatz files, run 500 times per case with different random
# numbers. None of its runs repeats a Braatz file (checked in tests/test_data.py), so they are
# fresh runs for any detector whose settings were chosen on the Braatz test runs.
#
# Twins. Testing run number r uses the same random numbers in every file. So faulty run r of
# any fault is identical to fault-free testing run r until the fault starts, and the fault-free
# run shows what that same run would have done with no fault. Three consequences:
#   - the fault start can be read off the data exactly (rieth_onset_check);
#   - the first 8 hours of run r are the same readings for all 20 faults, so there are only as
#     many different stretches of normal data as there are run numbers;
#   - held-out normal runs must use run numbers that no scored run uses.

RIETH_FILES = {   # kind: (Dataverse file id, file name, md5)
    "fault_free_training": (3031241, "TEP_FaultFree_Training.RData", "ec126484534331f85001d8c4ebce6d17"),
    "fault_free_testing": (3031240, "TEP_FaultFree_Testing.RData", "38ad9810fc871026157086ae2c2f0ee9"),
    "faulty_testing": (3031243, "TEP_Faulty_Testing.RData", "556bdb64c83021bc0c5f92e427753565"),
}
RIETH_FAULTS = range(1, 21)   # this set has faults 1-20; fault 21 of the Braatz files is not in it
# In the 960-sample testing runs the fault starts at sample 161 (index 160), as in the Braatz
# test runs. Read off the data by rieth_onset_check() and checked in tests/test_data.py.
RIETH_ONSET = 160


def download_rieth(raw: Path = RAW, kinds=tuple(RIETH_FILES)) -> Path:
    """Fetch the Rieth .RData files (about 910 MB in all) into data/raw/tep_rieth. Skips files already there."""
    target = raw / "tep_rieth"
    target.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        file_id, name, md5 = RIETH_FILES[kind]
        out = target / name
        if out.exists():
            continue
        # The server turns away Python's default client name, so the request names this code.
        request = urllib.request.Request(f"https://dataverse.harvard.edu/api/access/datafile/{file_id}",
                                         headers={"User-Agent": "adlab-data-loader/0.1"})
        part, digest = out.with_suffix(".part"), hashlib.md5()
        with urllib.request.urlopen(request, timeout=180) as response, open(part, "wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != md5:
            part.unlink()
            raise OSError(f"{name}: the download does not match the published checksum; try again")
        part.replace(out)
    return target


def _frame_to_runs(df: pd.DataFrame, n_runs: int) -> np.ndarray:
    """Rieth table -> array (faults, n_runs, samples, 52), keeping simulation runs 1..n_runs of each fault.

    The table has one row per sample: faultNumber, simulationRun, sample, then the 52 variables.
    """
    expected = ["faultNumber", "simulationRun", "sample"] + [f"xmeas_{i}" for i in range(1, 42)]         + [f"xmv_{i}" for i in range(1, 12)]
    if list(df.columns) != expected:
        raise ValueError("unexpected columns in the Rieth table")
    df = df[df["simulationRun"] <= n_runs]
    faults, runs, samples = (df[c].to_numpy() for c in ("faultNumber", "simulationRun", "sample"))
    n_faults, length = len(np.unique(faults)), int(samples.max())
    if len(df) != n_faults * n_runs * length:
        raise ValueError(f"expected {n_runs} runs of {length} samples for each of {n_faults} faults")
    order = np.lexsort((samples, runs, faults))          # by fault, then run, then sample
    out = df.iloc[:, 3:].to_numpy(float)[order].reshape(n_faults, n_runs, length, 52)
    if not (samples[order].reshape(n_faults, n_runs, length) == np.arange(1, length + 1)).all():
        raise ValueError("samples are not numbered 1..n within every run")
    return out


def rieth_runs(kind: str, n_runs: int, raw: Path = RAW) -> np.ndarray:
    """Simulation runs 1..n_runs of one Rieth file, as an array (faults, n_runs, samples, 52).

    The fault-free files have one "fault" (normal running). Reading the big faulty file needs
    about 10 GB of memory and a minute, so the chosen runs are saved as a .npy file next to it
    and read from there afterwards.
    """
    base = download_rieth(raw, (kind,))
    saved = base / f"{kind}_first{n_runs}.npy"
    if saved.exists():
        return np.load(saved)
    for other in base.glob(f"{kind}_first*.npy"):          # a saved file with more runs will do
        if int(other.stem.rsplit("first", 1)[1]) > n_runs:
            return np.load(other, mmap_mode="r")[:, :n_runs].copy()
    import pyreadr
    out = _frame_to_runs(pyreadr.read_r(str(base / RIETH_FILES[kind][1]))[kind], n_runs)
    np.save(saved, out)
    return out


def first_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For arrays (runs, samples, sensors): per run, the first sample index where a and b differ (-1 if never)."""
    differs = (a != b).any(-1)
    return np.where(differs.any(-1), differs.argmax(-1), -1)


def rieth_onset_check(raw: Path = RAW, n_runs: int = 50) -> pd.DataFrame:
    """Where does each fault start? Read off the data by comparing every faulty run with its twin.

    One row per fault: the earliest and latest first-difference index over the runs, and how
    many runs already differ at index RIETH_ONSET. Faults that switch on as a step differ from
    the twin at once; faults that are a slowly growing random disturbance can take a few
    samples before any reading changes.
    """
    faulty = rieth_runs("faulty_testing", n_runs, raw)
    twins = rieth_runs("fault_free_testing", n_runs, raw)[0]
    rows = []
    for i, f in enumerate(RIETH_FAULTS):
        first = first_difference(faulty[i], twins)
        rows.append({"fault": f"{f:02d} {TEP_FAULTS[f]}", "earliest first difference": int(first.min()),
                     "latest first difference": int(first.max()),
                     f"runs that differ at index {RIETH_ONSET}": int((first == RIETH_ONSET).sum()), "runs": n_runs})
    return pd.DataFrame(rows).set_index("fault")


def _rieth_name(base: str, chosen: tuple, default: tuple) -> str:
    """The name goes into cache file names, so a different choice of runs gets its own cached scores."""
    return base if chosen == default else base + "_" + "_".join(str(c) for c in chosen)


@functools.lru_cache(maxsize=None)
def _rieth_fit(raw: Path, fit_runs: int) -> np.ndarray:
    """Fit data shared by every Rieth dataset with the same fit_runs (one array, so one fitted model)."""
    return rieth_runs("fault_free_training", fit_runs, raw)[0].reshape(-1, 52)


def _fault_free_set(base: str, group: str, raw: Path, first_run: int, runs: int, fit_runs: int,
                    default: tuple, matched_from: Dataset | None) -> Dataset:
    """Fault-free testing runs first_run..first_run+runs-1, labelled over the hours a fault would cover."""
    fit = _rieth_fit(raw, fit_runs)
    x = rieth_runs("fault_free_testing", first_run + runs - 1, raw)[0][first_run - 1:]
    y = (np.arange(x.shape[1]) >= RIETH_ONSET).astype(int)
    out = [Run(f"nofault/run{first_run + r:03d}", x[r], y.copy(), fit, group=group,
               meta={"fault": 0, "run": first_run + r, "onset": RIETH_ONSET}) for r in range(runs)]
    return Dataset(_rieth_name(base, (first_run, runs, fit_runs), default), out, TEP_CHANNELS, 180.0, None,
                   calibration="matched" if matched_from is not None else "pooled",
                   matched_start=RIETH_ONSET, matched_from=matched_from)


def load_tep_rieth_calibration(raw: Path = RAW, first_run: int = 51, runs: int = 50, fit_runs: int = 20) -> Dataset:
    """Fault-free runs that set the alarm lines for the Rieth runs (matched-window calibration).

    Fault-free testing runs 51-100 by default. They are neither the fit runs nor the twins of
    the scored runs (run numbers 1-50), so they share no random numbers with either.
    """
    return _fault_free_set("tep_rieth_calibration", "00 no fault (calibration)", raw, first_run, runs, fit_runs,
                           (51, 50, 20), None)


def load_tep_rieth(raw: Path = RAW, runs_per_fault: int = 50, fit_runs: int = 20, val_runs: int = 20,
                   calibration_runs: int = 50) -> Dataset:
    """Tennessee Eastman, Rieth files: fresh runs of the same plant, many per fault.

    Scored runs       testing runs 1..runs_per_fault of each of faults 1-20: 960 samples each
                      (48 hours), normal for the first 160. meta["run"] is the run number.
    Fit data          fault-free training runs 1..fit_runs joined end to end (500 samples each).
    Alarm lines       set on the next calibration_runs fault-free testing runs after those run
                      numbers, over samples 161-960 (matched-window calibration; see Dataset).
    Held-out normal   the first val_runs of those same fault-free runs, whole, kept for the
                      older "deployable threshold" numbers.
    None of the fault-free runs shares random numbers with the scored runs or the fit data.
    Which runs are used is fixed by their numbers, not chosen by looking at results.
    """
    fit = _rieth_fit(raw, fit_runs)
    normal = rieth_runs("fault_free_testing", runs_per_fault + val_runs, raw)[0][runs_per_fault:]
    faulty = rieth_runs("faulty_testing", runs_per_fault, raw)
    y = (np.arange(faulty.shape[2]) >= RIETH_ONSET).astype(int)
    runs = [Run(f"fault{f:02d}/run{r + 1:03d}", faulty[i, r], y.copy(), fit, group=f"{f:02d} {TEP_FAULTS[f]}",
                meta={"fault": f, "run": r + 1, "onset": RIETH_ONSET})
            for i, f in enumerate(RIETH_FAULTS) for r in range(runs_per_fault)]
    val = [Run(f"normal/run{runs_per_fault + r + 1:03d}", x, np.zeros(len(x), int), fit, group="normal")
           for r, x in enumerate(normal)]
    name = _rieth_name("tep_rieth", (runs_per_fault, fit_runs, val_runs), (50, 20, 20))
    return Dataset(name, runs, TEP_CHANNELS, 180.0, val, calibration="matched", matched_start=RIETH_ONSET,
                   matched_from=load_tep_rieth_calibration(raw, runs_per_fault + 1, calibration_runs, fit_runs))


def load_tep_rieth_control(raw: Path = RAW, runs: int = 50, fit_runs: int = 20, calibration_runs: int = 50) -> Dataset:
    """The twins of the scored Rieth runs: the same runs with no fault at all.

    Fault-free testing runs 1..runs, fitted like load_tep_rieth and judged against the same
    alarm lines. The label marks the hours where the fault would have been (from index 160),
    although nothing is wrong there. Whatever a detector flags in those hours is not detection.
    """
    out = _fault_free_set("tep_rieth_control", "00 no fault (twin)", raw, 1, runs, fit_runs, (1, 50, 20),
                          load_tep_rieth_calibration(raw, runs + 1, calibration_runs, fit_runs))
    # keep the cache names of the first scoring: they did not include the first run number
    out.name = _rieth_name("tep_rieth_control", (runs, fit_runs), (50, 20))
    return out


# ---------------------------------------------------------------- plain-English descriptions

SKAB_INFO = pd.DataFrame([
    ("Accelerometer1RMS", "Vibration of the pump, sensor 1", "g"),
    ("Accelerometer2RMS", "Vibration of the pump, sensor 2", "g"),
    ("Current", "Electric current drawn by the motor", "amps"),
    ("Pressure", "Water pressure just after the pump", "bar"),
    ("Temperature", "Temperature of the motor body", "deg C"),
    ("Thermocouple", "Temperature of the water in the loop", "deg C"),
    ("Voltage", "Voltage across the motor", "volts"),
    ("Volume Flow RateRMS", "How fast water flows round the loop", "litres/min"),
], columns=["sensor", "what it measures", "unit"])

_TEP_CONT = [
    ("A feed (stream 1)", "kscmh"), ("D feed (stream 2)", "kg/h"), ("E feed (stream 3)", "kg/h"),
    ("A and C feed (stream 4)", "kscmh"), ("Recycle flow", "kscmh"), ("Reactor feed rate", "kscmh"),
    ("Reactor pressure", "kPa"), ("Reactor level", "%"), ("Reactor temperature", "deg C"),
    ("Purge rate", "kscmh"), ("Separator temperature", "deg C"), ("Separator level", "%"),
    ("Separator pressure", "kPa"), ("Separator underflow", "m3/h"), ("Stripper level", "%"),
    ("Stripper pressure", "kPa"), ("Stripper underflow", "m3/h"), ("Stripper temperature", "deg C"),
    ("Stripper steam flow", "kg/h"), ("Compressor work", "kW"),
    ("Reactor cooling water outlet temp", "deg C"), ("Separator cooling water outlet temp", "deg C"),
]
_TEP_LAB = ([f"Reactor feed: % of component {c}" for c in "ABCDEF"]
            + [f"Purge gas: % of component {c}" for c in "ABCDEFGH"]
            + [f"Product: % of component {c}" for c in "DEFGH"])
_TEP_VALVES = ["D feed flow", "E feed flow", "A feed flow", "A and C feed flow", "Compressor recycle valve",
               "Purge valve", "Separator liquid flow", "Stripper liquid product flow", "Stripper steam valve",
               "Reactor cooling water flow", "Condenser cooling water flow"]

TEP_INFO = pd.DataFrame(
    [(TEP_CHANNELS[i], d, u, "sensor reading, every 3 min") for i, (d, u) in enumerate(_TEP_CONT)]
    + [(TEP_CHANNELS[22 + i], d, "mol %", "lab analysis, updates every 6-15 min")
       for i, d in enumerate(_TEP_LAB)]
    + [(TEP_CHANNELS[41 + i], d, "% open", "valve / controller setting")
       for i, d in enumerate(_TEP_VALVES)],
    columns=["variable", "what it measures", "unit", "kind"])


def skab_frame(name: str = "valve1/1", raw: Path = RAW) -> pd.DataFrame:
    """One SKAB experiment as a DataFrame: time in seconds, 8 sensors, and the fault label.

    name is "<folder>/<file>", e.g. "valve1/1", "other/9" or "anomaly-free/anomaly-free".
    """
    df = pd.read_csv(raw / "skab" / "data" / f"{name}.csv", sep=";", parse_dates=["datetime"])
    df.insert(0, "second", (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().astype(int))
    df = df.drop(columns=["datetime", "changepoint"], errors="ignore").rename(columns={"anomaly": "fault"})
    if "fault" not in df:
        df["fault"] = 0
    df["fault"] = df["fault"].astype(int)
    return df


def skab_experiments(raw: Path = RAW) -> list[str]:
    """Names of the 34 SKAB experiments, for skab_frame()."""
    base = raw / "skab" / "data"
    return [f"{p.parent.name}/{p.stem}" for p in sorted(base.glob("*/*.csv")) if p.parent.name != "anomaly-free"]


def tep_frame(fault: int = 1, raw: Path = RAW) -> pd.DataFrame:
    """One Tennessee Eastman test run as a DataFrame: hour, 52 variables, and the fault label.

    fault=0 gives the normal test run.
    """
    X = np.loadtxt(raw / "tep" / f"d{fault:02d}_te.dat")
    df = pd.DataFrame(X, columns=TEP_CHANNELS)
    df.insert(0, "hour", np.arange(len(df)) * 3 / 60)
    df["fault"] = 0 if fault == 0 else (np.arange(len(df)) >= TEP_ONSET).astype(int)
    return df


LOADERS = {"skab": load_skab, "tep": load_tep, "tep_rieth": load_tep_rieth,
           "tep_rieth_control": load_tep_rieth_control, "tep_rieth_calibration": load_tep_rieth_calibration}


def load(name: str, raw: Path = RAW) -> Dataset:
    return LOADERS[name](raw)
