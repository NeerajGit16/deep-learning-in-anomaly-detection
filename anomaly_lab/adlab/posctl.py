"""Round 4, part 4: a positive control (notebook 14). Data built, before any results, so that a deep model SHOULD win.

Everything below was fixed before any detector was run on it.

Normal behaviour
    Three hidden drivers vary smoothly and continuously over time: each is white noise smoothed with a Gaussian
    window (standard deviation 20 readings), scaled to unit variance. Every run is a fresh stretch, so normal
    states never repeat exactly.
    Each sensor j = straight part + bend x bent part + noise:
        straight part   a_j . h                                    (h = the three drivers)
        bent part       sum_i b_ji h_i^2 + sum_{i<l} c_jil h_i h_l + sum_i e_ji sin(2 h_i),
                        scaled to the straight part's variance (on a long reference stretch), so at bend 1 the two
                        parts are the same size
        noise           u_j x the sensor's spread without noise, u_j drawn between 2% and 5%
    The coefficients a, b, c, e (standard normal) and u come from one fixed seed (4242), drawn for 40 sensors; the
    10-sensor data use the first 10.

Faults (a broken bent link)
    In each faulty run (300 readings) the second half (readings 150-299) is faulty: one sensor (drawn per run)
    is moved off the surface by half its normal standard deviation, held for the whole stretch, towards the
    middle of its normal range (so every sensor stays inside its normal range; a reading that would leave the
    range of the fitting data is put back at its edge). Its level and spread stay normal; only its link to the
    other sensors breaks.

Three dials (24 settings)
    bend strength 0, 0.5, 1, 2; sensors 10 and 40; normal fitting readings 2,000, 10,000 and 50,000 (runs of 500
    readings; the smaller sets are the first runs of the larger, and every setting uses the same driver paths).
    The alarm line: one line from 20 separate normal runs of 300 readings (matched, whole runs), at 1 false alarm
    in 100. Scored: 40 faulty runs. Ranges resample runs (1,000 draws).

Detectors
    The full lineup: the eight standard detectors, the kinked-link, the straight forecaster, PCA scored like the
    autoencoder, part 2's classic detectors (nearest normal reading on all sensors k = 1 and 5, dynamic PCA, CVA,
    kernel PCA; the unseen value check does not apply, no sensor takes 30 or fewer values), the autoencoder and
    the LSTM with notebook 09's design (bottleneck from PCA's component count; LSTM settings chosen on held-aside
    data with seed 0), three seeds where a detector has seeds. Nothing is tuned on the faults.

Added during the run, for compute (before any result at 10,000 or 50,000 readings was seen, and using no faulty
data): at 10,000 and 50,000 readings the LSTM does not choose its setting again; it uses the setting it chose on
held-aside normal data at 2,000 readings for the same bend and number of sensors (as the generated campaign reused
IndPenSim's choice). On this laptop's small GPU, choosing again (four settings, then two more seeds) would have taken
over 20 hours. The autoencoder is unchanged (its one setting comes from PCA's rule).

Fixed predictions
    P1  at bend 0, PCA and the straight-link detector match the autoencoder (no gap of 12 points with a range clear
        of zero, either way);
    P2  at bend 1 or more, with 40 sensors and limited data (2,000 and 10,000 readings), the autoencoder beats both
        nearest-neighbour checks and kernel PCA by the margin.
    Reported: the first setting (in the order bend, then sensors, then readings, each from small to large) where
    any deep model beats the best simple detector by 12 points with a range clear of zero; if none does, that is
    said plainly, as a limit of this project's deep setup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRIVERS = 3
SMOOTH = 20
COEF_SEED = 4242
MAX_SENSORS = 40
BENDS = (0.0, 0.5, 1.0, 2.0)
SENSORS = (10, 40)
READINGS = (2000, 10000, 50000)
RUN = 500                    # readings per fitting run
TEST_RUN, FAULT_FROM = 300, 150
N_FAULTY, N_LINE = 40, 20
SHIFT = 0.5                  # the fault, in the sensor's normal standard deviations
SEEDS = {"fit": 10_000, "line": 20_000, "faulty": 30_000}
REFERENCE = (999, 200_000)   # seed and length of the reference stretch that sets the scales


def drivers(n: int, seed: int) -> np.ndarray:
    """(n, 3) smooth drivers: white noise through a Gaussian window (s.d. SMOOTH readings), unit variance."""
    rng = np.random.default_rng(seed)
    half = 4 * SMOOTH
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / SMOOTH) ** 2)
    k /= np.sqrt((k ** 2).sum())                 # unit variance after smoothing
    white = rng.normal(size=(n + 2 * half, DRIVERS))
    return np.column_stack([np.convolve(white[:, i], k, mode="valid") for i in range(DRIVERS)])


def _coefficients() -> dict:
    rng = np.random.default_rng(COEF_SEED)
    pairs = [(i, l) for i in range(DRIVERS) for l in range(i + 1, DRIVERS)]
    return {"a": rng.normal(size=(MAX_SENSORS, DRIVERS)), "b": rng.normal(size=(MAX_SENSORS, DRIVERS)),
            "c": rng.normal(size=(MAX_SENSORS, len(pairs))), "e": rng.normal(size=(MAX_SENSORS, DRIVERS)),
            "u": rng.uniform(0.02, 0.05, MAX_SENSORS), "pairs": pairs}


def _parts(h: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """(straight part, bent part before scaling), each (n, d)."""
    c = _coefficients()
    straight = h @ c["a"][:d].T
    prods = np.column_stack([h[:, i] * h[:, l] for i, l in c["pairs"]])
    bent = (h ** 2) @ c["b"][:d].T + prods @ c["c"][:d].T + np.sin(2 * h) @ c["e"][:d].T
    return straight, bent


_SCALES: dict = {}


def scales(bend: float, d: int) -> dict:
    """{"bent": factor making the bent part as large as the straight part, "noise": noise s.d., "sd": each sensor's
    normal s.d. with noise} per sensor, from the reference stretch (cached in memory)."""
    key = (bend, d)
    if key not in _SCALES:
        s, b = _parts(drivers(REFERENCE[1], REFERENCE[0]), d)
        factor = s.std(0) / b.std(0)
        signal = s + bend * factor * b
        noise = _coefficients()["u"][:d] * signal.std(0)
        _SCALES[key] = {"bent": factor, "noise": noise, "sd": np.sqrt(signal.var(0) + noise ** 2)}
    return _SCALES[key]


def sensors(h: np.ndarray, bend: float, d: int, seed: int) -> np.ndarray:
    """Sensor readings (n, d) for driver paths h, with noise drawn from `seed`."""
    sc = scales(bend, d)
    s, b = _parts(h, d)
    rng = np.random.default_rng(seed + 7)
    return s + bend * sc["bent"] * b + rng.normal(size=s.shape) * sc["noise"]


def normal_run(n: int, bend: float, d: int, seed: int) -> np.ndarray:
    return sensors(drivers(n, seed), bend, d, seed)


def faulty_run(bend: float, d: int, seed: int, lo: np.ndarray, hi: np.ndarray, centre: np.ndarray) -> tuple:
    """(readings, labels, faulty sensor): a normal run with one sensor's link broken from FAULT_FROM on."""
    X = normal_run(TEST_RUN, bend, d, seed)
    j = int(np.random.default_rng(seed + 11).integers(d))
    part = X[FAULT_FROM:, j]
    sign = -np.sign(part.mean() - centre[j]) or 1.0
    X[FAULT_FROM:, j] = np.clip(part + sign * SHIFT * scales(bend, d)["sd"][j], lo[j], hi[j])
    y = (np.arange(TEST_RUN) >= FAULT_FROM).astype(int)
    return X, y, j


def name(bend: float, d: int, n: int) -> str:
    return f"posctl_b{bend:g}_d{d}_n{n}"


def settings() -> list[tuple]:
    """Every setting, in the reporting order: bend, then sensors, then readings."""
    return [(b, d, n) for b in BENDS for d in SENSORS for n in READINGS]


def lstm_choice(bend: float, d: int) -> dict | None:
    """The LSTM setting chosen (on held-aside normal data) at 2,000 readings for this bend and number of sensors."""
    import hashlib
    import json
    from .deep import STORE
    fit = np.vstack([normal_run(RUN, bend, d, SEEDS["fit"] + i) for i in range(READINGS[0] // RUN)])
    path = STORE / f"lstm__{hashlib.md5(np.ascontiguousarray(fit).tobytes()).hexdigest()[:12]}__choice.json"
    return json.loads(path.read_text())["setting"] if path.exists() else None


def dataset(bend: float, d: int, n: int):
    """The Dataset for one setting."""
    from .data import Dataset, Run
    fit = np.vstack([normal_run(RUN, bend, d, SEEDS["fit"] + i) for i in range(n // RUN)])
    lo, hi, centre = fit.min(0), fit.max(0), np.median(fit, 0)
    runs = []
    for i in range(N_FAULTY):
        X, y, j = faulty_run(bend, d, SEEDS["faulty"] + i, lo, hi, centre)
        runs.append(Run(f"faulty{i:02d}", X, y, fit, group=f"sensor {j}", meta={"sensor": j}))
    val = [Run(f"normal{i:02d}", normal_run(TEST_RUN, bend, d, SEEDS["line"] + i), np.zeros(TEST_RUN, int), fit,
               group="normal") for i in range(N_LINE)]
    fixed = lstm_choice(bend, d) if n > READINGS[0] else None
    return Dataset(name(bend, d, n), runs, [f"s{j}" for j in range(d)], 1.0, val, calibration="matched",
                   matched_start=0, fit_lengths=[RUN] * (n // RUN), deep_settings={"lstm": fixed} if fixed else None)


SIMPLE = ["rolling_z", "ewma", "cusum", "pca", "iforest", "matrix_profile", "link_straight", "link_curved",
          "link_kinked", "straight_forecaster", "pca_twin", "nearest_all_1", "nearest_all_5", "dpca", "cva", "kpca"]
DEEP = ["deep_ae", "deep_lstm"]
N_SEEDS = 3


def methods() -> list:
    from . import deep
    return SIMPLE + [m for m in DEEP if m in deep.self_checks()["keep"]]


def score(bend: float, d: int, n: int, which: str = "all", verbose: bool = True):
    """Score one setting (cached). which: "simple", "deep" or "all"."""
    from . import run
    ds = dataset(bend, d, n)
    keys = {"simple": SIMPLE, "deep": DEEP, "all": SIMPLE + DEEP}[which]
    run.score_all([ds], methods=keys, n_seeds=N_SEEDS, verbose=verbose)


def report(bend: float, d: int, n: int, n_boot: int = 1000) -> dict:
    from . import run, structure
    return structure.cached(f"{name(bend, d, n)}__report", lambda: run.report_dataset(
        dataset(bend, d, n), methods(), N_SEEDS, n_boot, calibration="matched", verbose=False))


# ---------------------------------------------------------------- notebook 14: pictures and tables

def look(sensors_shown=(0, 1, 2, 3), bends=BENDS, n: int = 3000):
    """Normal readings of two sensor pairs at each bend strength (one panel per bend)."""
    import matplotlib.pyplot as plt
    from . import plots
    fig, axes = plt.subplots(2, len(bends), figsize=(3.2 * len(bends), 6.0))
    for k, bend in enumerate(bends):
        X = normal_run(n, bend, 10, 77)
        for row, (a, b) in enumerate(((sensors_shown[0], sensors_shown[1]), (sensors_shown[2], sensors_shown[3]))):
            ax = axes[row, k]
            ax.plot(X[:, a], X[:, b], ".", ms=1.5, color=plots.BLUE, alpha=0.5)
            ax.set_xlabel(f"sensor {a}")
            ax.set_ylabel(f"sensor {b}")
            ax.grid(color=plots.GRID, lw=0.5)
            if row == 0:
                ax.set_title(f"bend {bend:g}")
    fig.suptitle("Normal readings, two pairs of sensors, at each bend strength (10 sensors)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def components_table() -> pd.DataFrame:
    """How many components PCA keeps (the autoencoder's bottleneck), per bend and number of sensors (2,000 readings)."""
    from .detectors import PCA
    rows = {}
    for bend in BENDS:
        for d in SENSORS:
            X = np.vstack([normal_run(RUN, bend, d, SEEDS["fit"] + i) for i in range(4)])
            Z = (X - X.mean(0)) / X.std(0)
            ev = np.linalg.svd(Z, compute_uv=False) ** 2
            rows[(f"bend {bend:g}", f"{d} sensors")] = {"components PCA keeps (90%)": PCA().fit(X).k_,
                                                        "variance in the first 3 components": ev[:3].sum() / ev.sum()}
    out = pd.DataFrame(rows).T
    out.index.names = ["bend", "sensors"]
    return out


def fault_picture(bend: float = 1.0, d: int = 10, run: int = 0):
    """One faulty run: the moved sensor over time, with what it would have read, and against its closest partner."""
    import matplotlib.pyplot as plt
    from . import plots
    fit = np.vstack([normal_run(RUN, bend, d, SEEDS["fit"] + i) for i in range(4)])
    lo, hi, centre = fit.min(0), fit.max(0), np.median(fit, 0)
    seed = SEEDS["faulty"] + run
    X, y, j = faulty_run(bend, d, seed, lo, hi, centre)
    clean = normal_run(TEST_RUN, bend, d, seed)
    partner = int(np.argsort(-np.abs(np.corrcoef(fit.T)[j]))[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    plots._shade(ax, y)
    ax.plot(clean[:, j], color=plots.INK2, lw=1, ls=(0, (3, 2)), label="what it would have read")
    ax.plot(X[:, j], color=plots.ORANGE, lw=1.4, label=f"sensor {j}, as read")
    ax.axhline(lo[j], color=plots.GRID, lw=1)
    ax.axhline(hi[j], color=plots.GRID, lw=1)
    ax.set_xlabel("reading")
    ax.set_title("The faulty sensor over time (shaded: the fault)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.plot(fit[:, partner], fit[:, j], ".", ms=1.5, color=plots.BLUE, alpha=0.3, label="normal fitting readings")
    ax.plot(X[y == 1, partner], X[y == 1, j], ".", ms=4, color=plots.ORANGE, label="faulty readings")
    ax.set_xlabel(f"sensor {partner}")
    ax.set_ylabel(f"sensor {j}")
    ax.set_title("Against its most closely linked sensor")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"One faulty run at bend {bend:g}, {d} sensors (grey lines: the sensor's normal range)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def _names(keys):
    from . import detectors as det
    return [det.REGISTRY[k].name for k in keys]


def grid() -> pd.DataFrame:
    """Every setting (in the reporting order): the deep models, the best simple detector and the gaps."""
    from . import deep
    rows = {}
    for s in settings():
        res = report(*s)
        summ = deep._summary(res)["dr_at_1pct"]
        simple = [m for m in _names(SIMPLE) if m in summ.index]
        deepn = [m for m in _names(DEEP) if m in summ.index]
        best = summ[simple].idxmax()
        best_deep = summ[deepn].idxmax()
        g = deep.gap(res, best_deep, best)
        row = {"autoencoder": summ.get("Autoencoder"), "LSTM": summ.get("LSTM forecaster"),
               "best simple detector": best, "best simple caught": summ[best],
               "best deep minus best simple, points": 100 * g["gap"],
               "range": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}", "verdict": g["verdict"]}
        for other in ("PCA (T2 + SPE)", "Straight-link residual", "Nearest normal reading, all sensors (k = 1)",
                      "Nearest normal readings, all sensors (k = 5)", "Kernel PCA (RBF)"):
            if other in summ.index and "Autoencoder" in summ.index:
                row[f"AE minus {other}"] = 100 * deep.gap(res, "Autoencoder", other)["gap"]
        rows[(s[0], s[1], s[2])] = row
    out = pd.DataFrame(rows).T
    out.index.names = ["bend", "sensors", "fitting readings"]
    return out


def predictions() -> pd.DataFrame:
    """The two predictions fixed in advance, setting by setting: did they hold?"""
    from . import deep
    rows = []
    for s in settings():
        res = report(*s)
        if s[0] == 0:
            gs = {o: deep.gap(res, "Autoencoder", o) for o in ("PCA (T2 + SPE)", "Straight-link residual")}
            rows.append({"prediction": "P1: at bend 0, PCA and straight-link match the autoencoder", "bend": s[0],
                         "sensors": s[1], "fitting readings": s[2],
                         "gaps (autoencoder minus each)": "; ".join(f"{o}: {100 * g['gap']:+.1f} ({100 * g['low']:+.1f} to {100 * g['high']:+.1f}) {g['verdict']}" for o, g in gs.items()),
                         "held": all(g["verdict"] == "tie" for g in gs.values())})
        if s[0] >= 1 and s[1] == 40 and s[2] in (2000, 10000):
            others = ("Nearest normal reading, all sensors (k = 1)", "Nearest normal readings, all sensors (k = 5)", "Kernel PCA (RBF)")
            gs = {o: deep.gap(res, "Autoencoder", o) for o in others}
            rows.append({"prediction": "P2: at bend 1 or more, 40 sensors, limited data, the autoencoder beats nearest-neighbour and kernel PCA",
                         "bend": s[0], "sensors": s[1], "fitting readings": s[2],
                         "gaps (autoencoder minus each)": "; ".join(f"{o}: {100 * g['gap']:+.1f} ({100 * g['low']:+.1f} to {100 * g['high']:+.1f}) {g['verdict']}" for o, g in gs.items()),
                         "held": all(g["verdict"] == "ahead" for g in gs.values())})
    return pd.DataFrame(rows).set_index(["prediction", "bend", "sensors", "fitting readings"])


def first_win() -> dict | None:
    """The first setting (bend, then sensors, then readings) where a deep model beats the best simple detector by the
    margin, with the gap; None if there is none."""
    from . import deep
    for s in settings():
        res = report(*s)
        summ = deep._summary(res)["dr_at_1pct"]
        best = summ[[m for m in _names(SIMPLE) if m in summ.index]].idxmax()
        for m in [m for m in _names(DEEP) if m in summ.index]:
            g = deep.gap(res, m, best)
            if g["verdict"] == "ahead":
                return {"setting": s, "deep model": m, "best simple": best, **g}
    return None


def gap_picture(table: pd.DataFrame | None = None):
    """The best deep model minus the best simple detector at every setting, as a grid (points)."""
    from . import plots
    t = grid() if table is None else table
    v = t["best deep minus best simple, points"].astype(float).unstack([1, 2])
    v.index = [f"bend {b:g}" for b in v.index]
    v.columns = [f"{d} sensors, {n:,} readings" for d, n in v.columns]
    return plots.heatmap(v, "Best deep model minus best simple detector, points (a win needs +12 and a range clear of 0)",
                         vmax=max(30.0, float(np.nanmax(np.abs(v.values)))), diverging=True, fmt="{:+.0f}",
                         colorbar_label="points")


def all_detectors(bend: float, d: int, n: int) -> pd.DataFrame:
    """Every detector's catch rate (mean of seeds) and range at one setting, best first."""
    from . import run
    return run.rates(report(bend, d, n)).to_frame()
