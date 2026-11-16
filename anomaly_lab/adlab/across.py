"""Across datasets (notebook 10): the curved share, the kinked share and three detection gaps side by side.

Six datasets: the water pump (SKAB), the chemical plant at one setting (Rieth runs), the Cranfield
flow rig, the six-mode plant, the metal etcher and IndPenSim. Each number comes from that dataset's
own notebook and design; the kinked share is new and uses the same hold-out rounds as the
dataset's headline curved share.

The kinked share: a straight-line model allowed corners (detectors.KinkedRidge: hinge terms at each
input's 10th, 50th and 90th percentiles, then the same ridge regression) takes the flexible model's
place in the measure. Its gain over the plain straight model, in points, is the "kinked gain";
the "kinked share" is that gain over the curved share: how much of the curve a few corners recover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import structure

DATASETS = ["pump", "plant", "rig", "tep6", "etch", "indpensim"]
TITLES = {"pump": "Water pump (SKAB)", "plant": "Chemical plant, one setting", "rig": "Flow rig (Cranfield)",
          "tep6": "Chemical plant, six modes", "etch": "Metal etcher", "indpensim": "Penicillin batches (IndPenSim)"}
REPORT = {"pump": "skab", "plant": "tep_rieth", "rig": "cranfield", "tep6": "tep6", "etch": "etch", "indpensim": "indpensim"}
SOURCE = {"pump": "notebook 04: the pump's 32 sessions (other/13 left out by the outlier rule), five blocks of time",
          "plant": "notebook 04: 20 fault-free Rieth runs, five blocks of time",
          "rig": "notebook 05: runs T1 + T2, whole visits to a setting held out, staggered",
          "tep6": "notebook 06: 30 runs, five rounds of whole runs",
          "etch": "notebook 07: 107 wafers, five rounds of whole wafers",
          "indpensim": "notebook 08: 90 batches, five rounds of whole batches"}
STRAIGHT = ["Straight-link residual", "Kinked-link residual", "Straight-link residual, stage told"]
FLEXIBLE = ["Curved-link residual"]


def design(name: str) -> dict:
    """The runs, sensors and hold-out of a dataset's headline curved share, and its cache name."""
    from . import batch, cranfield, data, etch, indpensim, multimode
    if name == "pump":
        sessions = [r for r in data.load("skab").runs if len(r.fit) == 400]
        runs = [r.fit - r.fit.mean(0) for r in sessions if r.name != "other/13"]
        return {"runs": runs, "channels": list(data.SKAB_INFO["sensor"]), "kw": {}, "cache": "lags_pump_without_13__lags_none"}
    if name == "plant":
        return {"runs": list(data.rieth_runs("fault_free_training", 20)[0]), "channels": data.TEP_CHANNELS, "kw": {},
                "cache": "lags_plant__lags_none"}
    if name == "rig":
        runs, labels = cranfield.normal(cranfield.HEADLINE)
        held, units = structure.staggered_folds(labels)
        return {"runs": runs, "channels": data.CRANFIELD_CHANNELS, "kw": {"folds": held, "units": units},
                "cache": "cranfield_T1T2__same_moment"}
    if name == "tep6":
        s = multimode.split()
        return {"runs": s["runs"], "channels": data.TEP_CHANNELS, "kw": {"folds": s["folds"]}, "cache": "multimode__same_moment"}
    if name == "etch":
        s = etch.split()
        return {"runs": s["runs"], "channels": etch.CHANNELS, "kw": {"folds": batch.as_folds(s["runs"], s["rounds"])},
                "cache": "etch__same_moment"}
    if name == "indpensim":
        s = indpensim.split()
        return {"runs": s["runs"], "channels": indpensim.CHANNELS, "kw": {"folds": batch.as_folds(s["runs"], s["rounds"])},
                "cache": "indpensim__same_moment"}
    raise KeyError(name)


def curved(name: str) -> dict:
    """The dataset's headline curved share, as its notebook computed it (read from results/cache)."""
    d = design(name)
    return structure.cached(d["cache"], lambda: structure.curvature_by_run(d["runs"], d["channels"], n_boot=500,
                                                                           verbose=False, **d["kw"]))


def kinked(name: str) -> dict:
    """The same measure with the kinked straight model in the flexible model's place (cached)."""
    d = design(name)
    return structure.cached(f"across__{name}__kinked", lambda: structure.curvature_by_run(
        d["runs"], d["channels"], n_boot=500, verbose=False, what=f"{name}, kinked", second="kinked", **d["kw"]))


def _gap(res: dict, a: str, b: str) -> dict:
    from .deep import gap
    return gap(res, a, b)


def gaps(name: str) -> dict:
    """The three detection gaps on one dataset, from its report with every detector (deep.report)."""
    from . import deep
    from . import detectors as det
    res = deep.report(REPORT[name])
    s = pd.DataFrame(res["summary"]).set_index("method")["dr_at_1pct"]
    deep_names = [det.REGISTRY[k].name for k in deep.DEEP if det.REGISTRY[k].name in s.index]
    straight = [m for m in STRAIGHT if m in s.index]
    flexible = [m for m in FLEXIBLE + deep_names if m in s.index]
    best_deep = s[deep_names].idxmax() if deep_names else None
    best_straight, best_flexible = s[straight].idxmax(), s[flexible].idxmax()
    out = {"curved-link minus straight-link": _gap(res, "Curved-link residual", "Straight-link residual")}
    out["best deep minus straight-link"] = (_gap(res, best_deep, "Straight-link residual") | {"which": best_deep}) if best_deep \
        else {"gap": np.nan, "low": np.nan, "high": np.nan, "verdict": "no deep model", "which": None}
    out["best flexible minus best straight-line"] = _gap(res, best_flexible, best_straight) | {"which": f"{best_flexible} vs {best_straight}"}
    return out


def table() -> pd.DataFrame:
    """One row per dataset: the curved share (points, with range), the kinked gain and share, and the three gaps."""
    rows = {}
    for name in DATASETS:
        c, k, g = curved(name), kinked(name), gaps(name)
        row = {"curved share": 100 * c["curved"], "curved low": 100 * c["curved_low"], "curved high": 100 * c["curved_high"],
               "kinked gain": 100 * k["curved"], "kinked low": 100 * k["curved_low"], "kinked high": 100 * k["curved_high"],
               "kinked share of the curve": k["curved"] / c["curved"] if c["curved"] > 0.002 else np.nan}
        for label, v in g.items():
            row[f"{label}, points"] = 100 * v["gap"]
            row[f"{label}, range"] = f"{100 * v['low']:+.1f} to {100 * v['high']:+.1f}"
            row[f"{label}, verdict"] = v["verdict"]
            if "which" in v:
                row[f"{label}, which"] = v["which"]
        row["curvature from"] = SOURCE[name]
        rows[TITLES[name]] = row
    out = pd.DataFrame(rows).T
    out.index.name = "dataset"
    return out


GAPS = ["curved-link minus straight-link", "best deep minus straight-link", "best flexible minus best straight-line"]


def same_order(frame: pd.DataFrame) -> pd.DataFrame:
    """For each gap: of the pairs of datasets, how many put the curved share and the gap in the same order
    (both higher on the same dataset), how many in opposite orders, and how many are tied on one of them."""
    rows = {}
    for g in GAPS:
        x = frame["curved share"].astype(float).to_numpy()
        y = frame[f"{g}, points"].astype(float).to_numpy()
        same = opposite = tied = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if np.isnan(x[i] + x[j] + y[i] + y[j]) or x[i] == x[j] or y[i] == y[j]:
                    tied += 1
                elif (x[i] - x[j]) * (y[i] - y[j]) > 0:
                    same += 1
                else:
                    opposite += 1
        rows[g] = {"pairs in the same order": same, "pairs in opposite orders": opposite, "pairs tied or missing": tied,
                   "pairs": same + opposite + tied}
    out = pd.DataFrame(rows).T
    out.index.name = "gap"
    return out


def chart(frame: pd.DataFrame):
    """The curved share against each gap, one panel per gap, every dataset a point with its ranges."""
    import matplotlib.pyplot as plt
    from . import plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, g in zip(axes, GAPS):
        for i, (name, r) in enumerate(frame.iterrows()):
            x, y = float(r["curved share"]), float(r[f"{g}, points"])
            if np.isnan(y):
                continue
            lo, hi = (float(v) for v in r[f"{g}, range"].split(" to "))
            ax.hlines(y, float(r["curved low"]), float(r["curved high"]), color=plots.BLUE, lw=1.5)
            ax.vlines(x, lo, hi, color=plots.BLUE, lw=1.5)
            ax.plot(x, y, "o", color=plots.BLUE, ms=7, mec=plots.BG, mew=1.5)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 5 if i % 2 == 0 else -11), fontsize=7,
                        color=plots.INK)                     # labels staggered up and down, so neighbours do not overlap
        for v in (-12, 12):
            ax.axhline(v, color=plots.ORANGE, lw=1, ls=(0, (3, 3)))
        ax.axhline(0, color=plots.INK2, lw=0.8)
        ax.axvline(3, color=plots.INK2, lw=0.8, ls=(0, (2, 3)))
        ax.set_xlabel("curved share on normal data, points (dotted: +3)")
        ax.set_ylabel(f"{g}, points (dashed: 12-point margin)")
        ax.set_title(g.capitalize(), fontsize=9)
        ax.grid(color=plots.GRID, lw=0.6)
    fig.tight_layout()
    return fig
