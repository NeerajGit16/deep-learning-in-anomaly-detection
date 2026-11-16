"""IndPenSim: 100 simulated batches of an industrial-scale (100,000 litre) penicillin fermentation.

Goldrick, Duran-Villalobos, Jankauskas, Lovett, Farid and Lennox (2019), "Modern day monitoring
and control challenges outlined on an industrial-scale benchmark fermentation process",
Computers and Chemical Engineering 130, 106471. Data: Mendeley Data, doi:10.17632/pdnjz7zz5x
(version 2), CC BY 4.0. One zip file of about 0.5 GB (about 2.5 GB unpacked, mostly Raman spectra).

Batches 1-30 follow a fixed recipe, 31-60 are run by operators, 61-90 by an advanced process
controller using the Raman readings, and 91-100 contain faults.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from . import progress
from .data import RAW

FOLDER = "indpensim"
FILE = "100_Batches_IndPenSim.zip"
SHA256 = "bc434c6cf96444fee3c915d434ac84e2a9c6492be4daebcfa4aed9cdff181f00"
SIZE = 503792426
FILES_API = "https://data.mendeley.com/public-api/datasets/pdnjz7zz5x/files?folder_id=root&version=2"
AGENT = "adlab-data-loader/0.1"


def _download_url() -> str:
    """Mendeley's current download link for the zip (it is looked up each time)."""
    request = urllib.request.Request(FILES_API, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        files = json.loads(response.read())
    return next(f["content_details"]["download_url"] for f in files if f["filename"] == FILE)


def download(raw: Path = RAW, verbose: bool = True) -> Path:
    """Fetch the zip into data/raw/indpensim/, check its SHA-256, and unpack it there.

    Nothing is fetched if the zip is already there and correct. About 0.5 GB.
    """
    folder = Path(raw) / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / FILE
    if not (target.exists() and target.stat().st_size == SIZE):
        part = target.with_suffix(".part")
        request = urllib.request.Request(_download_url(), headers={"User-Agent": AGENT})
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=300) as response, open(part, "wb") as f:
            def chunks():
                while block := response.read(1 << 20):
                    yield block
            for block in progress.track(chunks(), "downloading IndPenSim (MB)", total=-(-SIZE // (1 << 20)),
                                        verbose=verbose, leave=True):
                f.write(block)
                digest.update(block)
        if digest.hexdigest() != SHA256:
            raise OSError(f"{FILE}: SHA-256 does not match the one Mendeley publishes")
        part.replace(target)
        progress.note("IndPenSim zip downloaded and checked")
    _unpack(target, folder, verbose)
    return folder


def _unpack(zip_path: Path, folder: Path, verbose: bool = True) -> None:
    """Unpack every member inside `folder` (members whose names would land outside it are refused)."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for m in members:
            name = PurePosixPath(m.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts:
                raise OSError(f"refusing zip member outside the folder: {m.filename}")
        todo = [m for m in members if not ((folder / m.filename).exists() and (folder / m.filename).stat().st_size == m.file_size)]
        for m in progress.track(todo, "unpacking IndPenSim", verbose=verbose):
            z.extract(m, folder)


CSV = "Mendeley_data/100_Batches_IndPenSim_V3.csv"
STATISTICS = "Mendeley_data/100_Batches_IndPenSim_Statistics.csv"
PROCESS_COLUMNS = 39                 # the first 39 columns; the other 2,200 are the Raman spectrum
CACHE = "indpensim_process.npz"


def process(raw: Path = RAW) -> pd.DataFrame:
    """The 39 process and label columns of every reading of every batch (no Raman spectra).

    The first call reads the 2.6 GB CSV (a few minutes) and saves the columns as
    data/raw/indpensim/indpensim_process.npz; later calls read that file in a second.
    Column names are as the file gives them (see column_table for what they hold).
    """
    folder = Path(raw) / FOLDER
    cached = folder / CACHE
    if not cached.exists():
        if not (folder / CSV).exists():
            download(raw)
        frame = pd.read_csv(folder / CSV, usecols=range(PROCESS_COLUMNS), na_values=["NaN"], low_memory=False)
        np.savez_compressed(cached, values=frame.to_numpy(float), columns=np.array(frame.columns, dtype=str))
        progress.note("IndPenSim process columns cached")
    with np.load(cached) as z:
        return pd.DataFrame(z["values"], columns=[str(c) for c in z["columns"]])


def statistics(raw: Path = RAW) -> pd.DataFrame:
    """The per-batch table that comes with the data: penicillin harvested, yield and a fault flag."""
    return pd.read_csv(Path(raw) / FOLDER / STATISTICS)


# ---------------------------------------------------------------- what each column is

TIME, OUTPUT, ONLINE, OFFLINE, STATE, LABEL, RAMAN = ("time", "controller output", "online measurement",
                                                      "offline lab assay", "simulator state", "id or flag",
                                                      "Raman spectrum")
# By position: (short name, role, what it is). The last seven headers in the file are shifted:
# the file's own note says the batch number sits under "2-PAT control", and the values show
# the rest (see check_labels). The roles here follow the values, not the headers.
COLUMNS = [
    ("time", TIME, "hours since the batch started"),
    ("aeration", OUTPUT, "air flow into the vessel (L/h)"),
    ("agitator", OUTPUT, "agitator speed (RPM)"),
    ("sugar feed", OUTPUT, "sugar feed rate (L/h)"),
    ("acid flow", OUTPUT, "acid added to hold the pH (L/h)"),
    ("base flow", OUTPUT, "base added to hold the pH (L/h)"),
    ("cooling water", OUTPUT, "heating/cooling water flow to hold the temperature (L/h)"),
    ("heating water", OUTPUT, "heating water flow (L/h)"),
    ("dilution water", OUTPUT, "water added for dilution (L/h)"),
    ("head pressure", OUTPUT, "air pressure above the broth (bar)"),
    ("broth dumped", OUTPUT, "broth taken out of the vessel (L/h)"),
    ("substrate", STATE, "sugar concentration in the broth (g/L): the simulator's own state"),
    ("dissolved oxygen", ONLINE, "oxygen dissolved in the broth (mg/L)"),
    ("penicillin", STATE, "penicillin concentration (g/L): the simulator's own state"),
    ("volume", ONLINE, "broth volume (L)"),
    ("weight", ONLINE, "vessel weight (kg)"),
    ("pH", ONLINE, "pH of the broth"),
    ("temperature", ONLINE, "broth temperature (K)"),
    ("heat generated", STATE, "heat given off by the culture (kJ): the simulator's own state"),
    ("CO2 off-gas", ONLINE, "carbon dioxide in the exhaust gas (%)"),
    ("PAA flow", OUTPUT, "flow of the precursor phenylacetic acid (L/h)"),
    ("PAA offline", OFFLINE, "phenylacetic acid concentration, lab sample (g/L)"),
    ("oil flow", OUTPUT, "soybean oil feed (L/h)"),
    ("NH3 offline", OFFLINE, "ammonia concentration, lab sample (g/L)"),
    ("oxygen uptake", ONLINE, "oxygen uptake rate, from the exhaust gas (g/min)"),
    ("O2 off-gas", ONLINE, "oxygen in the exhaust gas (%)"),
    ("penicillin offline", OFFLINE, "penicillin concentration, lab sample (g/L)"),
    ("biomass offline", OFFLINE, "biomass concentration, lab sample (g/L)"),
    ("CO2 evolution", ONLINE, "carbon dioxide evolution rate, from the exhaust gas (g/h)"),
    ("ammonia shots", OUTPUT, "ammonia added in shots (kg)"),
    ("viscosity offline", OFFLINE, "broth viscosity, lab sample (centipoise)"),
    ("fault", LABEL, "1 while a fault is acting, 0 otherwise (header: Fault reference)"),
    ("operator", LABEL, "1 if operators run the batch, 0 otherwise (header: Control ref)"),
    ("PAT", LABEL, "2 if the Raman-based controller runs the batch, 1 otherwise (header: 1- No Raman spec)"),
    ("batch", LABEL, "batch number (header: 1-Raman spec recorded)"),
    ("batch again", LABEL, "batch number again (header: 2-PAT control)"),
    ("faulty batch", LABEL, "1 for a batch with a fault, 0 otherwise (header: Batch reference)"),
    ("Raman 1", RAMAN, "first Raman intensity (header: Batch ID)"),
    ("Raman 2", RAMAN, "second Raman intensity (header: Fault flag)"),
]
STRATEGIES = {"recipe": "fixed recipe", "operator": "operators", "APC": "advanced process control (Raman)"}


def readings(raw: Path = RAW) -> pd.DataFrame:
    """Every reading of every batch with short column names, plus "strategy" (recipe, operator or APC)."""
    d = process(raw)
    d.columns = [c[0] for c in COLUMNS]
    d["batch"] = d["batch"].astype(int)
    d["strategy"] = np.where(d["PAT"] == 2, "APC", np.where(d["operator"] == 1, "operator", "recipe"))
    return d


def check_labels(raw: Path = RAW) -> pd.DataFrame:
    """Per batch: the values found in each label column, to show what the shifted headers hold."""
    d = readings(raw)
    time_restarts = int(np.sum(np.diff(d["time"].to_numpy()) < 0)) + 1
    g = d.groupby("batch")
    out = pd.DataFrame({"readings": g.size(), "hours": g["time"].max(), "strategy": g["strategy"].first(),
                        "batch again agrees": g.apply(lambda x: bool((x["batch again"] == x.name).all())),
                        "faulty batch": g["faulty batch"].max().astype(int),
                        "readings with fault = 1": g["fault"].sum().astype(int)})
    out.attrs["time restarts"] = time_restarts
    return out


def column_table(raw: Path = RAW) -> pd.DataFrame:
    """Every column: its role, what it is, and whether it moves within the normal batches (1-90)."""
    d = readings(raw)
    normal = d[d["batch"] <= 90]
    rows = {}
    for name, role, what in COLUMNS:
        x = normal[name]
        rows[name] = {"role": role, "what it is": what, "share missing (normal batches)": float(x.isna().mean()),
                      "moves in normal batches": bool(x.dropna().nunique() > 1)}
    out = pd.DataFrame(rows).T
    out.index.name = "column"
    return out


def fault_spans(raw: Path = RAW) -> pd.DataFrame:
    """Per faulty batch: where the fault flag is 1, as hours and readings, and whether it is one unbroken stretch."""
    d = readings(raw)
    rows = {}
    for b, x in d[d["faulty batch"] == 1].groupby("batch"):
        on = np.flatnonzero(x["fault"].to_numpy() == 1)
        t = x["time"].to_numpy()
        pieces = int(np.sum(np.diff(on) > 1)) + 1 if len(on) else 0
        rows[int(b)] = {"strategy": x["strategy"].iloc[0], "readings": len(x), "hours": float(t[-1]),
                        "fault starts (h)": float(t[on[0]]) if len(on) else np.nan,
                        "fault ends (h)": float(t[on[-1]]) if len(on) else np.nan,
                        "readings with fault": len(on), "stretches": pieces}
    out = pd.DataFrame(rows).T
    out.index.name = "batch"
    return out
