"""Notebook 12: artificial data with the ingredient that makes the autoencoder win on IndPenSim (follow-up
round, part D). The generator is simulate.recipe_batches; this module measures it and runs the pilot.

Fixed before computing:

D2  The curved share and the kinked share (as notebook 08: same moment, pooled, whole batches held out in five
    rounds) on 60 normal batches at dial 0, 0.25, 0.5, 0.75, 1, 1.5 and 2. Expected: about zero at dial 0. The
    "IndPenSim-like" dial is the one on that grid whose curved share is nearest IndPenSim's headline (+14.8
    points); "the middle" is half of it. Chosen on normal data only.
D3  A pilot at dial 0, the middle and the IndPenSim-like dial. At each: detectors fitted on 60 normal batches,
    one alarm line from 30 other normal batches (matched, as on IndPenSim), scored on 30 faulty batches (one fault
    each). Detectors: straight-link, kinked-link, PCA scored like the autoencoder, PCA + corners (the best
    straight-line ingredient model of notebook 11 on the three batches that carry the win) and the autoencoder
    (three seeds). The question: does the autoencoder's gain over the best of the four simple detectors grow
    with the dial? Margin rule and ranges from resampling faulty batches as everywhere else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import simulate as SIM

CHANNELS = SIM.REC_OUTPUTS + SIM.REC_MEASURES
DIALS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
INDPENSIM_CURVED = 14.8                     # notebook 08's headline curved share, points
N_FIT, N_LINE, N_FAULTY = 60, 30, 30
SEEDS = {"fit": 11, "line": 12, "faulty": 13}
SIMPLE = ["link_straight", "link_kinked", "pca_twin", "pca_hinges"]
SCHEDULE = ["unseen_value", "nearest_1", "nearest_5", "recipe_hour"]      # round 3, part 3
SIZES = (1.0, 0.5, 0.25)                                                  # round 3: fault size (1 = as before)
DEEP = ["deep_ae"]


def _tag(dial: float) -> str:
    return f"{dial:g}".replace(".", "p")


def normal_batches(dial: float, n: int = N_FIT, seed: int = SEEDS["fit"]) -> list[dict]:
    return SIM.recipe_batches(n, dial=dial, seed=seed)


def curvature(dial: float, second: str = "flexible", n_boot: int = 500) -> dict:
    """The curved share (second="flexible") or the kinked gain (second="kinked") on 60 normal batches."""
    from . import structure
    runs = [b["X"] for b in normal_batches(dial)]
    folds = [np.full(len(r), i % 5) for i, r in enumerate(runs)]       # whole batches held out, five rounds
    return structure.cached(f"mechanism_curvature__d{_tag(dial)}__{second}", lambda: structure.curvature_by_run(
        runs, CHANNELS, folds=folds, n_boot=n_boot, verbose=False, what=f"dial {dial:g}, {second}", second=second))


def curvature_table(dials=DIALS) -> pd.DataFrame:
    rows = {}
    for d in dials:
        c, k = curvature(d), curvature(d, "kinked")
        rows[d] = {"curved share, points": 100 * c["curved"], "curved range": f"{100 * c['curved_low']:+.1f} to {100 * c['curved_high']:+.1f}",
                   "straight share": c["straight"], "kinked gain, points": 100 * k["curved"],
                   "kinked share of the curve": k["curved"] / c["curved"] if c["curved"] > 0.002 else np.nan}
    out = pd.DataFrame(rows).T
    out.index.name = "dial"
    return out


def indpensim_like(table: pd.DataFrame | None = None) -> float:
    t = curvature_table() if table is None else table
    return float((t["curved share, points"].astype(float) - INDPENSIM_CURVED).abs().idxmin())


def pilot_dials() -> list[float]:
    d = indpensim_like()
    return [0.0, d / 2, d]


def dataset(dial: float, size: float = 1.0):
    """The pilot's dataset at one dial: 30 faulty batches to score, 30 line batches, 60 fit batches joined.
    size (round 3): the fault size; 1 is the pilot of notebook 12's first version, with the same name."""
    from .data import Dataset, Run
    fit = normal_batches(dial)
    line = SIM.recipe_batches(N_LINE, dial=dial, seed=SEEDS["line"])
    faulty = SIM.recipe_batches(N_FAULTY, dial=dial, seed=SEEDS["faulty"], faulty=True, fault_size=size)
    F = np.vstack([b["X"] for b in fit])
    runs = [Run(f"faulty{b['batch']}", b["X"], b["fault"], F, group=b["fault kind"],
                meta={"output held": b["output held"]}) for b in faulty]
    val = [Run(f"line{b['batch']}", b["X"], np.zeros(len(b["X"]), int), F, group="normal") for b in line]
    name = f"mechanism_d{_tag(dial)}" + ("" if size == 1.0 else f"_size{_tag(size)}")
    return Dataset(name, runs, CHANNELS, 60.0, val, calibration="matched", matched_start=0,
                   fit_lengths=[len(b["X"]) for b in fit], output_columns=list(range(len(SIM.REC_OUTPUTS))),
                   fit_groups=["recipe"] * len(fit))


def score(dial: float, verbose: bool = True):
    from . import deep, deepdive, run  # noqa: F401  (deepdive registers PCA + corners)
    ds = dataset(dial)
    run.score_all([ds], methods=SIMPLE, n_seeds=1, verbose=verbose)
    run.score_all([ds], methods=DEEP, n_seeds=deep.SEEDS, verbose=verbose)


def report(dial: float, n_boot: int = 1000, verbose: bool = True) -> dict:
    from . import deep, run, structure
    score(dial, verbose)
    return structure.cached(f"mechanism_report__d{_tag(dial)}", lambda: run.report_dataset(
        dataset(dial), SIMPLE + DEEP, deep.SEEDS, n_boot, calibration="matched", verbose=verbose))


def pilot_table(dials=None) -> pd.DataFrame:
    """Per dial: each detector's catch rate, and the autoencoder's gap to the best simple detector."""
    from . import deep
    from . import detectors as det
    rows = {}
    for d in dials or pilot_dials():
        res = report(d, verbose=False)
        s = deep._summary(res)["dr_at_1pct"]
        simple = [det.REGISTRY[k].name for k in SIMPLE]
        best = s[simple].idxmax()
        g = deep.gap(res, "Autoencoder", best)
        rows[d] = {**{m: s[m] for m in simple + ["Autoencoder"]}, "best simple": best,
                   "autoencoder minus best simple, points": 100 * g["gap"],
                   "range": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}", "verdict": g["verdict"],
                   "spread across seeds": s.get("Autoencoder") and deep._summary(res).loc["Autoencoder", "dr_at_1pct_seed_sd"]}
    out = pd.DataFrame(rows).T
    out.index.name = "dial"
    return out


# ---------------------------------------------------------------- round 3, part 3: a finer dial, three fault sizes
#
# Fixed before computing: the pilot rerun at every dial of DIALS and every fault size of SIZES (1, a half, a
# quarter), with the four schedule-aware checks added to the simple detectors. For each size: the first dial where
# the autoencoder's gap over the best simple detector (schedule-aware checks included) exceeds 12 points.

def score_r3(dial: float, size: float, verbose: bool = False):
    from . import deep, deepdive, run  # noqa: F401
    ds = dataset(dial, size)
    run.score_all([ds], methods=SIMPLE + SCHEDULE, n_seeds=1, verbose=verbose)
    run.score_all([ds], methods=DEEP, n_seeds=deep.SEEDS, verbose=verbose)


def report_r3(dial: float, size: float, n_boot: int = 1000) -> dict:
    from . import deep, run, structure
    score_r3(dial, size)
    return structure.cached(f"mechanism_report_r3__d{_tag(dial)}__size{_tag(size)}", lambda: run.report_dataset(
        dataset(dial, size), SIMPLE + SCHEDULE + DEEP, deep.SEEDS, n_boot, calibration="matched", verbose=False))


def sweep_table() -> pd.DataFrame:
    """Every dial and size: each detector's catch rate, the best simple detector (schedule-aware checks included)
    and the autoencoder's gap over it, with its range and verdict."""
    from . import deep
    from . import detectors as det
    rows = {}
    for size in SIZES:
        for d in DIALS:
            res = report_r3(d, size)
            s = deep._summary(res)["dr_at_1pct"]
            simple = [det.REGISTRY[k].name for k in SIMPLE + SCHEDULE]
            best = s[simple].idxmax()
            g = deep.gap(res, "Autoencoder", best)
            rows[(size, d)] = {**{m: s[m] for m in simple + ["Autoencoder"]}, "best simple": best,
                               "gap, points": 100 * g["gap"], "range": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}",
                               "verdict": g["verdict"]}
    out = pd.DataFrame(rows).T
    out.index.names = ["fault size", "dial"]
    return out


def crossover(table: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each fault size: the first dial where the autoencoder's gap over the best simple detector exceeds 12
    points, and whether its range is clear of zero there (a win by the margin rule)."""
    t = sweep_table() if table is None else table
    rows = {}
    for size in SIZES:
        part = t.loc[size]
        over = part[part["gap, points"].astype(float) > 12]
        if len(over):
            d = over.index[0]
            rows[size] = {"first dial with a gap over 12 points": d, "gap there": float(over.loc[d, "gap, points"]),
                          "range there": over.loc[d, "range"], "a win by the margin rule": over.loc[d, "verdict"] == "ahead"}
        else:
            rows[size] = {"first dial with a gap over 12 points": "none", "gap there": np.nan, "range there": "",
                          "a win by the margin rule": False}
    out = pd.DataFrame(rows).T
    out.index.name = "fault size"
    return out
