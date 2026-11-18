"""Round 5, part 3: the positive control, version 2 (notebook 14, sections 9 to 13).

Written before any result of version 2. It fixes the two weaknesses that notebook 14's section 5 found in version 1:

Truly bent for every model
    Three hidden drivers as in version 1 (smooth, never repeating). Each sensor j = straight part + bend x its own bent
    part + noise, where its own bent part is a sum of 6 tanh units of the three drivers that no other sensor shares:
        bent_j(h) = sum_u w_ju tanh(G v_ju . h + c_ju),   v_ju standard normal (3 values), c_ju uniform on (-1, 1),
                                                           w_ju standard normal, gain G = 1.5,
    scaled to the straight part's variance on a long reference stretch (so bend 1 makes the two the same size). With
    6 functions of its own per sensor, no sensor is a straight-line mix of the others. Noise 2 to 5% of the sensor's
    spread. Coefficients from one fixed seed (5151), drawn for 40 sensors; 10 sensors use the first 10.
    The check, on normal data only, before any fault is scored: at bend 1, for 10 and for 40 sensors, the curved share
    (notebook 04's measure: 20 normal runs of 500 readings, whole runs held out in five rounds) must be at least +10
    points with its whole 90% range above +5 (so the straight-link model's held-out fit is clearly below the
    curved-link model's). If it is not, the gain G is multiplied by 1.5 and the check repeated (at most 5 times), on
    normal data only, and the change is reported.

Faults at three sizes
    One sensor (drawn per run) moved by 0.5, 1 or 2 of its normal spread from reading 150 to 299 of a 300-reading run,
    towards the middle of its range, and put back at the edge of the fitting data's range if it would leave it. Each
    setting has 40 faulty runs of each size (the same driver paths and sensors for every size) and 20 normal line
    runs. One dataset per setting holds all 120 faulty runs; each size is reported on its own.

Settings
    bend 0, 1 and 2; 10 and 40 sensors; 2,000, 10,000 and 50,000 normal fitting readings (runs of 500).

Detectors
    The full simple lineup of version 1 (posctl.SIMPLE: the per-sensor checks, PCA, isolation forest, matrix profile,
    the straight-, curved- and kinked-link residuals, the straight forecaster, PCA scored like the autoencoder, the
    nearest normal reading on all sensors with k = 1 and 5, dynamic PCA, CVA and kernel PCA). The deep models with
    part 2's long training budget: the autoencoder, the LSTM and the PCA-plus-correction autoencoder, three seeds.
    For compute, as in version 1: the LSTM chooses its setting on held-aside normal data at 2,000 readings and reuses
    it at 10,000 and 50,000 for the same bend and sensors.

Fixed prediction
    At bend 1 or 2 with 50,000 readings, at least one deep model beats every simple detector (including the curved-link
    residual, the nearest-neighbour checks and kernel PCA) by 12 points with a range clear of zero, at some fault size.
    Reported: the first setting (bend, then sensors, then readings, then fault size, each from small to large) where a
    deep model beats the best simple detector by the margin, or plainly that there is none.

Changed after the check (on normal data only, before any fault was scored)
    The check failed at 40 sensors whatever the gain: from 1.5 to 11.4 the curved share stayed at +2 to +3 points
    (at 10 sensors it passed at every gain, +12 to +13). With all 40 sensors driven by the same three drivers, the 39
    other sensors carry so many different bends of those drivers that a straight-line mix of them copies any one
    sensor's bend (the straight-link model explains 90 to 95%). Sharper tanh units do not change that.
    Two changes, tried on normal data only (check_steps) and kept because together they pass:
        groups   40 sensors are four groups of ten, each built like the 10-sensor setting from its OWN three drivers
                 (independent of the other groups'). The 10-sensor setting is unchanged (one group).
        size     the bent part is 1.5 times as large at every bend (the check's own step): at bend 1 it is 1.5 times
                 the straight part's size, at bend 2 three times.
    Alone, the groups give +9.1 at 40 sensors (just short) and the larger bend +2.8; together +10.6 (range +10.2 to
    +11.0) at 40 sensors and +14.3 at 10. The gain stays at 1.5. Everything else is as written above.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import posctl
from .posctl import DRIVERS, MAX_SENSORS, REFERENCE, RUN, TEST_RUN, FAULT_FROM, N_FAULTY, N_LINE, drivers

COEF_SEED = 5151
UNITS = 6
GAIN = 1.5
BENDS = (0.0, 1.0, 2.0)
SENSORS = (10, 40)
READINGS = (2000, 10000, 50000)
SIZES = (0.5, 1.0, 2.0)
SEEDS = {"fit": 50_000, "line": 60_000, "faulty": 70_000}
CHECK = {"curved share": 0.10, "range low": 0.05, "steps": 5, "factor": 1.5}
SIMPLE = posctl.SIMPLE
DEEP = ["deep_ae_v2", "deep_lstm_v2", "deep_pcac_v2"]
N_SEEDS = 3
GROUP = 10                      # sensors per group of drivers (the "groups" change)
GROUP_SEED_STEP = 7919          # group g's drivers use seed + g x this (group 0: the seed itself)
DESIGNS = {"first": {"groups": False, "size": 1.0},     # as written before the check
           "final": {"groups": True, "size": 1.5}}      # after the check (normal data only)
DESIGN = "final"


def _coefficients() -> dict:
    rng = np.random.default_rng(COEF_SEED)
    return {"a": rng.normal(size=(MAX_SENSORS, DRIVERS)), "v": rng.normal(size=(MAX_SENSORS, UNITS, DRIVERS)),
            "c": rng.uniform(-1, 1, size=(MAX_SENSORS, UNITS)), "w": rng.normal(size=(MAX_SENSORS, UNITS)),
            "u": rng.uniform(0.02, 0.05, MAX_SENSORS)}


def _drivers(n: int, seed: int, d: int, design: str) -> np.ndarray:
    """(n, groups, 3): one set of three drivers per group of sensors (a single group unless the design has groups)."""
    groups = -(-d // GROUP) if DESIGNS[design]["groups"] else 1
    return np.stack([drivers(n, seed + GROUP_SEED_STEP * g) for g in range(groups)], 1)


def _parts(h: np.ndarray, d: int, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """(straight part, bent part before scaling), each (n, d); every sensor's bent part uses its own tanh units of its
    group's drivers (h: (n, groups, 3))."""
    c = _coefficients()
    H = h[:, np.minimum(np.arange(d) // GROUP, h.shape[1] - 1), :]                  # (n, d, 3): each sensor's drivers
    straight = np.einsum("njk,jk->nj", H, c["a"][:d])
    pre = np.einsum("njk,juk->nju", H, gain * c["v"][:d]) + c["c"][:d][None]       # (n, d, units)
    bent = (np.tanh(pre) * c["w"][:d][None]).sum(2)
    return straight, bent


_SCALES: dict = {}


def scales(bend: float, d: int, gain: float, design: str = DESIGN) -> dict:
    key = (bend, d, gain, design)
    if key not in _SCALES:
        s, b = _parts(_drivers(REFERENCE[1], REFERENCE[0], d, design), d, gain)
        factor = DESIGNS[design]["size"] * s.std(0) / b.std(0)
        signal = s + bend * factor * b
        noise = _coefficients()["u"][:d] * signal.std(0)
        _SCALES[key] = {"bent": factor, "noise": noise, "sd": np.sqrt(signal.var(0) + noise ** 2)}
    return _SCALES[key]


def normal_run(n: int, bend: float, d: int, seed: int, gain: float, design: str = DESIGN) -> np.ndarray:
    sc = scales(bend, d, gain, design)
    s, b = _parts(_drivers(n, seed, d, design), d, gain)
    rng = np.random.default_rng(seed + 7)
    return s + bend * sc["bent"] * b + rng.normal(size=s.shape) * sc["noise"]


# ---------------------------------------------------------------- the check on normal data, and the gain it fixes

def check_once(gain: float, d: int, n_boot: int = 300, design: str = DESIGN) -> dict:
    """The curved share at bend 1 for one gain, number of sensors and design (normal data only)."""
    from . import structure
    runs = [normal_run(RUN, 1.0, d, SEEDS["fit"] + i, gain, design) for i in range(20)]
    folds = [np.full(RUN, i % 5) for i in range(20)]
    res = structure.curvature_by_run(runs, [f"s{j}" for j in range(d)], folds=folds, n_boot=n_boot, verbose=False,
                                     what=f"positive control 2 check, {d} sensors, gain {gain:g}, {design} design")
    ok = res["curved"] >= CHECK["curved share"] and res["curved_low"] > CHECK["range low"]
    return {"gain": gain, "sensors": d, "curved share": res["curved"], "low": res["curved_low"],
            "high": res["curved_high"], "straight share": res.get("straight", np.nan), "passed": bool(ok)}


def check() -> pd.DataFrame:
    """The check as designed: the first design, strengthening the bend (gain x 1.5, at most 5 times) until both
    sensor counts pass (cached). It did not pass at 40 sensors."""
    from . import structure

    def compute():
        rows, gain = [], GAIN
        for _ in range(CHECK["steps"] + 1):
            here = [check_once(gain, d, design="first") for d in SENSORS]
            rows += here
            if all(r["passed"] for r in here):
                break
            gain *= CHECK["factor"]
        return pd.DataFrame(rows)
    return structure.cached("posctl2_check", compute)


def check_steps() -> pd.DataFrame:
    """After the check failed (normal data only): the two changes alone and together at gain 1.5, then the final
    design at both sensor counts (cached)."""
    from . import structure

    def compute():
        rows = []
        for label, groups, size, d in (("as designed", False, 1.0, 40), ("groups only", True, 1.0, 40),
                                       ("larger bend only", False, 1.5, 40), ("both (final)", True, 1.5, 40),
                                       ("both (final)", True, 1.5, 10)):
            DESIGNS["_try"] = {"groups": groups, "size": size}
            try:
                rows.append({"design": label, **check_once(GAIN, d, design="_try")})
            finally:
                DESIGNS.pop("_try", None)
                for k in [k for k in _SCALES if k[3] == "_try"]:
                    del _SCALES[k]
        return pd.DataFrame(rows)
    return structure.cached("posctl2_check_steps", compute)


def gain() -> float:
    """The gain of the final design: 1.5, which passes the check at both sensor counts (raises otherwise)."""
    t = check_steps()
    final = t[t["design"] == "both (final)"]
    if not (len(final) == len(SENSORS) and final["passed"].all()):
        raise RuntimeError("the final design does not pass the check on normal data")
    return GAIN


# ---------------------------------------------------------------- datasets

def faulty_run(bend, d, seed, lo, hi, centre, size, g):
    X = normal_run(TEST_RUN, bend, d, seed, g)
    j = int(np.random.default_rng(seed + 11).integers(d))
    part = X[FAULT_FROM:, j]
    sign = -np.sign(part.mean() - centre[j]) or 1.0
    X[FAULT_FROM:, j] = np.clip(part + sign * size * scales(bend, d, g)["sd"][j], lo[j], hi[j])
    return X, (np.arange(TEST_RUN) >= FAULT_FROM).astype(int), j


def name(bend: float, d: int, n: int) -> str:
    return f"posctl2_b{bend:g}_d{d}_n{n}"


def settings() -> list[tuple]:
    return [(b, d, n) for b in BENDS for d in SENSORS for n in READINGS]


def lstm_choice(bend: float, d: int) -> dict | None:
    """The long-trained LSTM's setting chosen at 2,000 readings for this bend and number of sensors."""
    import hashlib
    import json
    from .deep import STORE
    g = gain()
    fit = np.vstack([normal_run(RUN, bend, d, SEEDS["fit"] + i, g) for i in range(READINGS[0] // RUN)])
    path = STORE / f"lstm__{hashlib.md5(np.ascontiguousarray(fit).tobytes()).hexdigest()[:12]}__v2__choice.json"
    return json.loads(path.read_text())["setting"] if path.exists() else None


def dataset(bend: float, d: int, n: int):
    """One setting: 40 faulty runs of each fault size (group "size 0.5" etc.) and 20 normal line runs."""
    from .data import Dataset, Run
    g = gain()
    fit = np.vstack([normal_run(RUN, bend, d, SEEDS["fit"] + i, g) for i in range(n // RUN)])
    lo, hi, centre = fit.min(0), fit.max(0), np.median(fit, 0)
    runs = []
    for size in SIZES:
        for i in range(N_FAULTY):
            X, y, j = faulty_run(bend, d, SEEDS["faulty"] + i, lo, hi, centre, size, g)
            runs.append(Run(f"size{size:g}_faulty{i:02d}", X, y, fit, group=f"size {size:g}", meta={"sensor": j, "size": size}))
    val = [Run(f"normal{i:02d}", normal_run(TEST_RUN, bend, d, SEEDS["line"] + i, g), np.zeros(TEST_RUN, int), fit,
               group="normal") for i in range(N_LINE)]
    fixed = lstm_choice(bend, d) if n > READINGS[0] else None
    return Dataset(name(bend, d, n), runs, [f"s{j}" for j in range(d)], 1.0, val, calibration="matched",
                   matched_start=0, fit_lengths=[RUN] * (n // RUN), deep_settings={"lstm": fixed} if fixed else None)


def score(bend: float, d: int, n: int, which: str = "all", verbose: bool = True):
    from . import run
    keys = {"simple": SIMPLE, "deep": DEEP, "all": SIMPLE + DEEP}[which]
    run.score_all([dataset(bend, d, n)], methods=keys, n_seeds=N_SEEDS, verbose=verbose)


def report(bend: float, d: int, n: int, size: float, n_boot: int = 1000) -> dict:
    from . import run, structure
    ds = dataset(bend, d, n)
    keep = [r.meta["size"] == size for r in ds.runs]
    return structure.cached(f"{name(bend, d, n)}__size{size:g}__report", lambda: run.report_dataset(
        ds, SIMPLE + DEEP, N_SEEDS, n_boot, keep=keep, calibration="matched", verbose=False))


# ---------------------------------------------------------------- notebook 14, version 2: tables and pictures

def _names(keys):
    from . import detectors as det
    return [det.REGISTRY[k].name for k in keys]


def grid() -> pd.DataFrame:
    """Every setting and fault size: the deep models, the best simple detector, and the best deep model's gap to it."""
    from . import deep
    rows = {}
    for s in settings():
        for size in SIZES:
            res = report(*s, size)
            summ = deep._summary(res)["dr_at_1pct"]
            simple = [m for m in _names(SIMPLE) if m in summ.index]
            deepn = [m for m in _names(DEEP) if m in summ.index]
            best, bd = summ[simple].idxmax(), summ[deepn].idxmax()
            g = deep.gap(res, bd, best)
            row = {m: summ[m] for m in deepn}
            row.update({"best simple detector": best, "best simple caught": summ[best], "best deep model": bd,
                        "best deep minus best simple, points": 100 * g["gap"],
                        "range": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}", "verdict": g["verdict"]})
            for other in ("PCA (T2 + SPE)", "Straight-link residual", "Curved-link residual",
                          "Nearest normal reading, all sensors (k = 1)", "Kernel PCA (RBF)"):
                row[other] = summ.get(other)
            rows[(s[0], s[1], s[2], size)] = row
    out = pd.DataFrame(rows).T
    out.index.names = ["bend", "sensors", "fitting readings", "fault size"]
    return out


def prediction() -> pd.DataFrame:
    """The fixed prediction, setting by setting (bend 1 or 2, 50,000 readings): did a deep model beat every simple
    detector by the margin at some fault size?"""
    t = grid()
    part = t[(t.index.get_level_values(0) >= 1) & (t.index.get_level_values(2) == 50000)]
    rows = {}
    for (b, d, n), sub in part.groupby(level=[0, 1, 2]):
        wins = sub[sub["verdict"] == "ahead"]
        rows[(b, d, n)] = {"sizes where a deep model wins": ", ".join(f"{i[3]:g}" for i in wins.index) or "none",
                           "best gap, points": float(sub["best deep minus best simple, points"].astype(float).max()),
                           "held": len(wins) > 0}
    out = pd.DataFrame(rows).T
    out.index.names = ["bend", "sensors", "fitting readings"]
    return out


def first_win() -> dict | None:
    t = grid()
    for idx, r in t.iterrows():
        if r["verdict"] == "ahead":
            return {"setting": idx, "deep model": r["best deep model"], "best simple": r["best simple detector"],
                    "gap": r["best deep minus best simple, points"], "range": r["range"]}
    return None


def against_every_simple(bend: float, d: int, n: int, size: float, deep_name: str) -> pd.DataFrame:
    """One deep model against EVERY simple detector at one setting and fault size (the prediction says "beats every
    simple detector"; the grid compares with the best one): gap in points, range and verdict, closest first."""
    from . import deep
    res = report(bend, d, n, size)
    s = deep._summary(res)["dr_at_1pct"]
    rows = {}
    for m in [x for x in _names(SIMPLE) if x in s.index]:
        g = deep.gap(res, deep_name, m)
        rows[m] = {"simple detector caught": s[m], "gap, points": 100 * g["gap"],
                   "range": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}", "verdict": g["verdict"]}
    out = pd.DataFrame(rows).T.sort_values("gap, points")
    out.index.name = f"{deep_name} ({s[deep_name]:.2f}) against"
    return out


def gap_picture(table: pd.DataFrame | None = None):
    from . import plots
    t = grid() if table is None else table
    v = t["best deep minus best simple, points"].astype(float).unstack([1, 2])
    v.index = [f"bend {b:g}, fault {s:g}" for b, s in v.index]
    v.columns = [f"{d} sensors, {n:,} readings" for d, n in v.columns]
    return plots.heatmap(v, "Version 2: best deep model minus best simple detector, points (a win needs +12 and a range clear of 0)",
                         vmax=max(30.0, float(np.nanmax(np.abs(v.values)))), diverging=True, fmt="{:+.0f}",
                         colorbar_label="points")


def look(n: int = 3000):
    """Normal readings of two sensor pairs at each bend (10 sensors), as in version 1's section 2."""
    import matplotlib.pyplot as plt
    from . import plots
    g = gain()
    fig, axes = plt.subplots(2, len(BENDS), figsize=(3.4 * len(BENDS), 6.0))
    for k, bend in enumerate(BENDS):
        X = normal_run(n, bend, 10, 77, g)
        for row, (a, b) in enumerate(((0, 1), (2, 3))):
            ax = axes[row, k]
            ax.plot(X[:, a], X[:, b], ".", ms=1.5, color=plots.BLUE, alpha=0.5)
            ax.set_xlabel(f"sensor {a}")
            ax.set_ylabel(f"sensor {b}")
            ax.grid(color=plots.GRID, lw=0.5)
            if row == 0:
                ax.set_title(f"bend {bend:g}")
    fig.suptitle("Version 2: normal readings, two pairs of sensors, at each bend (10 sensors)", x=0.01, ha="left")
    fig.tight_layout()
    return fig
