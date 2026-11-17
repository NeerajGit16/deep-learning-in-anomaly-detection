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


def dataset(dial: float):
    """The pilot's dataset at one dial: 30 faulty batches to score, 30 line batches, 60 fit batches joined."""
    from .data import Dataset, Run
    fit = normal_batches(dial)
    line = SIM.recipe_batches(N_LINE, dial=dial, seed=SEEDS["line"])
    faulty = SIM.recipe_batches(N_FAULTY, dial=dial, seed=SEEDS["faulty"], faulty=True)
    F = np.vstack([b["X"] for b in fit])
    runs = [Run(f"faulty{b['batch']}", b["X"], b["fault"], F, group=b["fault kind"],
                meta={"output held": b["output held"]}) for b in faulty]
    val = [Run(f"line{b['batch']}", b["X"], np.zeros(len(b["X"]), int), F, group="normal") for b in line]
    return Dataset(f"mechanism_d{_tag(dial)}", runs, CHANNELS, 60.0, val, calibration="matched", matched_start=0,
                   fit_lengths=[len(b["X"]) for b in fit])


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
