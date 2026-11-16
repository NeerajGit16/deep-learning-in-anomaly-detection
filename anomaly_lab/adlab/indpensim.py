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


SPLIT_HEADER = 33                    # "1- No Raman spec, 1-Raman spec recorded, 2-PAT control(PAT_ref)" was cut in three


def raw_field_counts(raw: Path = RAW) -> dict:
    """Fields in the CSV's header line and in its first data line (None if the CSV is not unpacked)."""
    import csv
    path = Path(raw) / FOLDER / CSV
    if not path.exists():
        return {"header fields": None, "data fields": None, "headers": None}
    with open(path, newline="") as f:
        r = csv.reader(f)
        header, first = next(r), next(r)
    return {"header fields": len(header), "data fields": len(first), "headers": header}


def column_evidence(raw: Path = RAW) -> pd.DataFrame:
    """Every process column: the header the file puts over it, the header it belongs to, the name used here,
    and the evidence from its values (normal batches 1-90 unless said).

    The file's header line has two more fields than its data lines: the header of column 33 contains two
    commas and was cut into three, so from column 33 on each header sits two places to the right of its data.
    The header a column belongs to is therefore the one two places further right.
    """
    d, counts = readings(raw), raw_field_counts(raw)
    headers = counts["headers"] or list(process(raw).columns)
    n = d[d["batch"] <= 90]
    t = d["time"].to_numpy()
    restarts = np.r_[0, np.flatnonzero(np.diff(t) < 0) + 1]
    count = np.repeat(np.arange(1, len(restarts) + 1), np.diff(np.r_[restarts, len(t)]))
    st = statistics(raw)
    per_batch = d.groupby("batch")

    def corr(a, b, frame=n):
        x = frame[[a, b]].dropna()
        return float(np.corrcoef(x[a], x[b])[0, 1])

    def batches_where(mask):
        u = np.unique(d.loc[mask, "batch"])
        return f"{u.min()}-{u.max()} ({len(u)} batches)" if len(u) else "none"

    dumped = n["broth dumped"] < 0
    dv = n["volume"].diff()
    special = {
        "time": f"rises in steps of {np.unique(np.round(np.diff(t)[np.diff(t) > 0], 3))[0]} h and starts again at "
                f"{np.unique(t[restarts])[0]} h {len(restarts)} times, once per batch",
        "agitator": "always 100 RPM: the agitator is never moved",
        "broth dumped": f"never above 0; when it is below 0 the volume falls at the next reading "
                        f"{100 * float((dv.shift(-1)[dumped] < 0).mean()):.0f}% of the time",
        "penicillin": f"rises through each batch (correlation with time {corr('penicillin', 'time'):.2f})",
        "weight": f"follows the volume (correlation {corr('volume', 'weight'):.3f}; about "
                  f"{float(np.median(n['weight'] / n['volume'])):.2f} kg per litre)",
        "temperature": "297-301 K, about 25 °C",
        "CO2 off-gas": f"moves with the CO2 evolution rate (correlation {corr('CO2 off-gas', 'CO2 evolution'):.2f})",
        "oxygen uptake": f"moves against the O2 in the off-gas (correlation {corr('oxygen uptake', 'O2 off-gas'):.2f})",
        "O2 off-gas": "0.15-0.23: a fraction, about 19%, though the header says %",
        "penicillin offline": f"agrees with the penicillin state where both exist (correlation "
                              f"{corr('penicillin offline', 'penicillin'):.3f}): a lab sample of it",
        "CO2 evolution": f"moves with the CO2 in the off-gas (correlation {corr('CO2 off-gas', 'CO2 evolution'):.2f})",
        "ammonia shots": "always 0",
        "acid flow": f"acid flows when the pH is high: median pH {n.loc[n['acid flow'] > 0, 'pH'].median():.2f} while it "
                     f"flows, {n.loc[n['acid flow'] == 0, 'pH'].median():.2f} otherwise",
        "base flow": f"more base when the pH is low (correlation with pH {corr('base flow', 'pH'):.2f})",
        "substrate": f"high in the first 5 hours (median {n.loc[n['time'] <= 5, 'substrate'].median():.2f} g/L), "
                     f"then near 0 (median {n.loc[n['time'] > 20, 'substrate'].median():.4f}): the sugar is eaten as fed",
        "volume": "under the vessel's 100,000 L; rises while dilution water flows "
                  f"(median {n.groupby('batch')['volume'].diff()[n['dilution water'] > 0].median():.0f} L per reading)",
    }
    for lab in ("PAA offline", "NH3 offline", "biomass offline", "viscosity offline"):
        special[lab] = "a value only every 10-12 hours, missing otherwise: a lab sample"
    special = {**special,
        "fault": f"0 in every reading of batches 1-90; 1 in some readings of batches {batches_where(d['fault'] == 1)}",
        "operator": f"1 exactly in batches {batches_where(d['operator'] == 1)}: the operator-run batches",
        "PAT": f"2 exactly in batches {batches_where(d['PAT'] == 2)}: the Raman-controlled batches; 1 otherwise",
        "batch": f"equals the batch count (how many times the time has restarted): "
                 f"{bool((d['batch'] == count).all())}",
        "batch again": f"equals the batch count as well: {bool((d['batch again'] == count).all())}",
        "faulty batch": f"constant within each batch; 1 exactly in batches {batches_where(d['faulty batch'] == 1)}; "
                        f"agrees with the fault column of the statistics file for all 100 batches: "
                        f"{bool((per_batch['faulty batch'].max().to_numpy() == st.iloc[:, -1].to_numpy()).all())}",
        "Raman 1": f"0 to {d['Raman 1'].max():,.0f}, the scale of the Raman intensities",
        "Raman 2": f"moves with Raman 1 (correlation {corr('Raman 1', 'Raman 2', d):.5f}): the next wavenumber",
    }
    rows = {}
    for i, (name, role, what) in enumerate(COLUMNS):
        over = headers[i]
        belongs = headers[i + 2] if i >= SPLIT_HEADER else headers[i]
        if i >= SPLIT_HEADER:
            belongs = {33: "PAT ref (1- No Raman spec, 1-Raman spec recorded, 2-PAT control)"}.get(i, belongs)
            belongs = {37: "2400 (first Raman wavenumber)", 38: "2399"}.get(i, belongs)
        x = (n if i < 31 else d)[name].dropna()
        values = f"{x.min():.4g} to {x.max():.4g}" + (f", {100 * n[name].isna().mean():.0f}% missing"
                                                      if n[name].isna().any() else "") + (" (all batches)" if i >= 31 else "")
        rows[i] = {"header over it in the file": over.strip(), "header it belongs to": belongs.strip(),
                   "name here": name, "role here": role, "values": values,
                   "evidence": special.get(name, "header not shifted (it is before column 33); values fit its quantity and unit")}
    out = pd.DataFrame(rows).T
    out.index.name = "column"
    out.attrs.update({k: v for k, v in counts.items() if k != "headers"})
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


# ---------------------------------------------------------------- task B3: measures and detectors
#
# Fixed before anything was computed (notebook 08, section 4):
#   columns   online measurements and controller outputs that move in normal batches (CHANNELS)
#   unit      the batch; five staggered rounds by strategy, every round fitting on all three
#   stage     ten equal slices of each batch's progress (0-10%, 10-20%, ...)
#   detectors fit on 60 normal batches, alarm line on the other 30 (10 per strategy, which fit
#             nothing), one line for all strategies; also a line from the recipe batches alone,
#             because the faulty batches all follow the recipe

CHANNELS = ["dissolved oxygen", "volume", "weight", "pH", "temperature", "CO2 off-gas", "oxygen uptake",
            "O2 off-gas", "CO2 evolution",
            "aeration", "sugar feed", "acid flow", "base flow", "cooling water", "heating water", "dilution water",
            "head pressure", "broth dumped", "PAA flow", "oil flow"]
LEFT_OUT_STILL = ["agitator", "ammonia shots"]           # controller outputs that never move in normal batches
SLICES = 10
STRATEGY_CODE = {"recipe": 0, "operator": 1, "APC": 2}
LINE_POSITION = 2                                        # within each strategy, every third batch sets the line
CACHE_NAME = "indpensim"
STRATEGY_TOLD = "same moment, control strategy told to the straight model"
INFO = pd.Series({name: what for name, _, what in COLUMNS})


def stage_of(n: int, slices: int = SLICES) -> np.ndarray:
    """Which tenth of a batch of n readings each reading is in (0 to 9)."""
    return np.minimum(slices * np.arange(n) // n, slices - 1)


def batches(raw: Path = RAW) -> list[dict]:
    """Every batch: {"batch", "strategy", "X" (readings x CHANNELS), "time", "fault" (0/1 per reading),
    "stage", "faulty"}, in batch order."""
    d = readings(raw)
    out = []
    for b, x in d.groupby("batch", sort=True):
        out.append({"batch": int(b), "strategy": x["strategy"].iloc[0], "X": x[CHANNELS].to_numpy(float),
                    "time": x["time"].to_numpy(), "fault": x["fault"].to_numpy().astype(int),
                    "stage": stage_of(len(x)), "faulty": bool(x["faulty batch"].max() == 1)})
    return out


def normal_batches(raw: Path = RAW) -> list[dict]:
    return [b for b in batches(raw) if not b["faulty"]]


def split(raw: Path = RAW, without: tuple = ()) -> dict:
    """The 90 normal batches for the curvature measure, with rounds (batch.staggered_rounds by strategy)."""
    from . import batch as B
    bs = normal_batches(raw)
    rounds = B.staggered_rounds([b["strategy"] for b in bs])
    keep = [i for i, b in enumerate(bs) if b["batch"] not in without]
    return {"runs": [bs[i]["X"] for i in keep], "stages": [bs[i]["stage"] for i in keep], "rounds": rounds[keep],
            "names": [bs[i]["batch"] for i in keep], "strategies": [bs[i]["strategy"] for i in keep]}


def split_table(raw: Path = RAW) -> pd.DataFrame:
    from . import batch as B
    s = split(raw)
    return B.round_table(s["rounds"], s["strategies"], "strategy")


def outlier_check(raw: Path = RAW):
    from . import batch as B
    s = split(raw)
    return B.outlier_check(s["runs"], CHANNELS, s["names"])


def _tag(without: tuple) -> str:
    return CACHE_NAME + ("_without_" + "_".join(map(str, without)) if without else "")


def measures(without: tuple = (), n_boot: int = 500, verbose: bool = True) -> dict:
    """Headline, stage told, control strategy told and the four lag sets (each cached when done)."""
    from . import batch as B, structure
    s = split(without=without)
    tag = _tag(without)
    out = B.measures(s["runs"], s["stages"], s["rounds"], CHANNELS, tag, n_boot=n_boot, verbose=verbose)
    codes = [np.full(len(r), STRATEGY_CODE[g]) for r, g in zip(s["runs"], s["strategies"])]
    out[STRATEGY_TOLD] = structure.cached(f"{tag}__strategy_told", lambda: structure.curvature_by_run(
        s["runs"], CHANNELS, folds=B.as_folds(s["runs"], s["rounds"]), modes=codes, n_boot=n_boot, verbose=verbose,
        what=f"{tag}, strategy told"))
    first = [B.SAME_MOMENT, B.TOLD, STRATEGY_TOLD]
    progress.note(f"{tag}: pooled measures done")
    return {k: out[k] for k in first + [k for k in out if k not in first]}


def within_stages(without: tuple = (), n_boot: int = 500, verbose: bool = True) -> dict:
    from . import batch as B
    s = split(without=without)
    return B.within_stages(s["runs"], s["stages"], s["rounds"], CHANNELS, _tag(without),
                           label=lambda k: f"{10 * k}-{10 * k + 10}%", n_boot=n_boot, verbose=verbose)


def between_table(without: tuple = ()) -> pd.DataFrame:
    from . import batch as B
    s = split(without=without)
    return B.between_table(s["runs"], s["stages"], CHANNELS, s["strategies"], "strategies", INFO)


def sensor_table(summary: dict) -> pd.DataFrame:
    from . import batch as B
    return B.sensor_table(summary, INFO)


def pooled_readings(without: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    s = split(without=without)
    return np.vstack(s["runs"]), np.concatenate(s["stages"])


def best_partner(sensors: list, without: tuple = ()) -> pd.DataFrame:
    from . import batch as B
    return B.best_partners(pooled_readings(without)[0], sensors, CHANNELS, INFO)


def shape_clues(sensors: list, without: tuple = ()) -> pd.DataFrame:
    from . import batch as B
    return B.shape_clues(*pooled_readings(without), sensors, CHANNELS)


def curved_scatter(sensors: list, without: tuple = (), every: int = 3):
    from . import plots
    X, stage = pooled_readings(without)
    partners = best_partner(sensors, without)["best single predictor"]
    return plots.scatter_by_stage(X, stage, list(zip(sensors, partners)), CHANNELS,
                                  {c: f"{INFO[c]} ({c})" for c in CHANNELS},
                                  {k: f"{10 * k}-{10 * k + 10}%" for k in range(SLICES)}, every=every)


# ---------------------------------------------------------------- detectors

def roles(raw: Path = RAW) -> list[str]:
    """Each normal batch's role for the detectors: within each strategy, in batch order, every third
    batch (the 3rd, 6th, ...) sets the alarm line and the others are fitted on (20 + 10 per strategy)."""
    seen, out = {}, []
    for b in normal_batches(raw):
        k = seen.get(b["strategy"], 0)
        out.append("line" if k % 3 == LINE_POSITION else "fit")
        seen[b["strategy"]] = k + 1
    return out


def role_table(raw: Path = RAW) -> pd.DataFrame:
    bs, r = normal_batches(raw), roles(raw)
    rows = {s: {"fitted on": sum(b["strategy"] == s and x == "fit" for b, x in zip(bs, r)),
                "set the alarm line": sum(b["strategy"] == s and x == "line" for b, x in zip(bs, r)), "scored": 0}
            for s in STRATEGY_CODE}
    rows["faulty (recipe)"] = {"fitted on": 0, "set the alarm line": 0, "scored": sum(b["faulty"] for b in batches(raw))}
    return pd.DataFrame(rows).T


def fault_kind_labels(raw: Path = RAW) -> dict:
    """Faulty batches with the same fault timing (start, end and number of fault readings) get the same
    letter: {batch: "A", ...}. The data does not say which fault each batch carries."""
    spans = fault_spans(raw)
    keys = {b: (row["fault starts (h)"], row["fault ends (h)"], row["readings with fault"]) for b, row in spans.iterrows()}
    letters, out = {}, {}
    for b in sorted(keys):
        letters.setdefault(keys[b], "ABCDEFGHIJ"[len(letters)])
        out[int(b)] = letters[keys[b]]
    return out


# The simulator's process faults, as its Python port (PenSimPy, peni_env_setup.py, MIT licence) codes
# them: hours in which a controller output is forced to a fixed value (one step = 0.2 h). Faults 7 and
# 8 (temperature and pH sensor drifts) and 6 (all at once) do not appear in batches 91-100.
FAULT_CATALOGUE = {
    "aeration": {"column": "aeration", "stretches": [(20, 24), (100, 110)], "forced to": [20]},
    "vessel back pressure": {"column": "head pressure", "stretches": [(100, 104), (200, 240)], "forced to": [2]},
    "substrate feed": {"column": "sugar feed", "stretches": [(20, 30), (76, 92), (200, 214)], "forced to": [2, 20]},
    "base flow": {"column": "base flow", "stretches": [(80, 84), (140, 160)], "forced to": [5, 10]},
    "coolant flow": {"column": "cooling water", "stretches": [(70, 90), (240, 270)], "forced to": [2, 10]},
}


def fault_types(raw: Path = RAW) -> pd.DataFrame:
    """Per faulty batch: which fault of FAULT_CATALOGUE it carries, with the evidence. A batch matches a fault
    when every stretch of its fault flag starts at one of the fault's stretches (a stretch can be cut short
    by the end of the batch, and stretches after the batch's end are missing), and the controller output
    named holds only the fault's forced values while the flag is on."""
    d = readings(raw)
    timing = fault_kind_labels(raw)
    rows = {}
    for b, x in d[d["faulty batch"] == 1].groupby("batch"):
        on = x["fault"].to_numpy() == 1
        t = x["time"].to_numpy()
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        spans = [(float(t[a]), float(t[e - 1])) for a, e in zip(edges[::2], edges[1::2])]
        match = []
        for name, f in FAULT_CATALOGUE.items():
            starts = [s for s, _ in f["stretches"]]
            values = set(np.round(x.loc[on, f["column"]].to_numpy(), 6))
            if all(any(abs(s0 - s) < 0.25 for s in starts) for s0, _ in spans) and values <= set(f["forced to"]):
                match.append(name)
        rows[int(b)] = {"timing": timing[int(b)], "fault stretches (h)": ", ".join(f"{a:g}-{e:g}" for a, e in spans),
                        "batch ends (h)": float(t[-1]), "fault type": " or ".join(match) if match else "no match",
                        "controller output held at": ", ".join(f"{v:g}" for v in sorted(set(np.round(
                            x.loc[on, FAULT_CATALOGUE[match[0]]["column"]], 6)))) if len(match) == 1 else ""}
    out = pd.DataFrame(rows).T
    out.index.name = "batch"
    return out


def detector_dataset(told: bool | str = False, raw: Path = RAW):
    """Faulty batches as runs (labels: the per-reading fault flag), the 30 line batches as
    val_normal (whole batches set the line), the 60 fit batches joined as the fit data.
    told=True (or "stage") appends the stage (tenth of the batch) as a last column; told="strategy"
    appends the control strategy's code instead (STRATEGY_CODE; notebook 11)."""
    from .data import Dataset, Run
    extra = {True: "stage", "stage": "stage", "strategy": "strategy"}.get(told)

    def x(b):
        if extra == "stage":
            return np.column_stack([b["X"], b["stage"]])
        if extra == "strategy":
            return np.column_stack([b["X"], np.full(len(b["X"]), STRATEGY_CODE[b["strategy"]])])
        return b["X"]

    bs, r = normal_batches(raw), roles(raw)
    fit = np.vstack([x(b) for b, role in zip(bs, r) if role == "fit"])
    kinds = fault_kind_labels(raw)
    runs = [Run(f"batch{b['batch']}", x(b), b["fault"], fit, group=f"timing {kinds[b['batch']]}",
                meta={"batch": b["batch"], "strategy": b["strategy"]}) for b in batches(raw) if b["faulty"]]
    val = [Run(f"batch{b['batch']}", x(b), np.zeros(len(b["X"]), int), fit, group="normal",
               meta={"batch": b["batch"], "strategy": b["strategy"]}) for b, role in zip(bs, r) if role == "line"]
    name = {"stage": "indpensim_told", "strategy": "indpensim_strategy"}.get(extra, "indpensim")
    return Dataset(name, runs, CHANNELS + ([extra] if extra else []), 720.0, val,
                   calibration="matched", matched_start=0,
                   fit_lengths=[len(b["X"]) for b, role in zip(bs, r) if role == "fit"])


TOLD_KEY = "link_straight_told"


def detector_methods() -> list[str]:
    from . import detectors as det, run
    return [m for m in run.METHODS if m in det.available()] + [TOLD_KEY]


def score(verbose: bool = True):
    from . import run
    run.score_all([detector_dataset()], methods=[m for m in detector_methods() if m != TOLD_KEY], n_seeds=5,
                  verbose=verbose)
    run.score_all([detector_dataset(told=True)], methods=[TOLD_KEY], verbose=verbose)


def detector_report(line: str = "all", n_boot: int = 1000, verbose: bool = True) -> dict:
    """line="all": one line from all 30 line batches (the primary). line="recipe": the line from the
    10 recipe line batches alone (the faulty batches follow the recipe)."""
    from . import run, structure
    score(verbose)
    ds = detector_dataset()
    keep = None if line == "all" else [r.meta["strategy"] == line for r in ds.val_normal]
    return structure.cached(f"indpensim_report_{line}", lambda: run.report_dataset(
        ds, detector_methods(), 5, n_boot, verbose=verbose, sources={TOLD_KEY: detector_dataset(told=True)},
        line_keep=keep))


def before_fault(line: str = "all") -> pd.DataFrame:
    """False alarms on the faulty batches' own normal readings, before the fault and after it first starts."""
    from . import run
    ds = detector_dataset()
    keep = None if line == "all" else [r.meta["strategy"] == line for r in ds.val_normal]
    return run.false_alarms_around_fault(ds, detector_methods(), sources={TOLD_KEY: detector_dataset(told=True)},
                                         line_keep=keep)


def fault_kinds() -> pd.DataFrame:
    from . import structure
    return structure.cached("indpensim_kinds", lambda: structure.fault_types(detector_dataset(), saved=True))
