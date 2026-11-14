"""Datasets as lists of runs with one common shape.

Every run carries
    X      (T, d)  the series to score
    y      (T,)    1 where the sample is labelled anomalous
    fit    (n, d)  normal data the detector is fitted on for this run
    group          anomaly type or fault, used for per-type reporting

Datasets come from public GitHub archives and are cached under data/raw/.
"""
from __future__ import annotations

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
    # How the primary metric sets its threshold: "pooled" (one threshold across runs, right when
    # all runs share one fit) or "per_run" (each run's own threshold, right when each run has
    # its own fit and recording session).
    calibration: str = "pooled"


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
    d00_te.dat is held out for deployable thresholds.
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
    return Dataset("tep", runs, TEP_CHANNELS, 180.0, val)


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


LOADERS = {"skab": load_skab, "tep": load_tep}


def load(name: str, raw: Path = RAW) -> Dataset:
    return LOADERS[name](raw)
