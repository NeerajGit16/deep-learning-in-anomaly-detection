"""Round 3, part 2: the generated PenSimPy campaign (notebook 13). The batches are made by adlab.pensim in
PenSimPy's own environment (python -m adlab.pensim campaign); this module reads them in the project's environment.

Fixed before computing (as asked, written down first):

Batches (adlab.pensim.campaign_plan, every seed recorded in data/raw/pensim_campaign/plan.json; lengths vary as
published, round(230 + 25 x a normal draw) hours): 100 recipe-driven and 60 operator-run normal batches; the
substrate feed fault, recipe-driven, as published (A, 30), held at recipe values used at other times (B, 30: 8 L/h
at 20-30 h, 15 L/h at 76-92 and 200-214 h) and scaled to 80%, 50% and 20% of its scheduled value (C, 15 each);
fault A on operator-run batches (15); the aeration, back pressure, base flow and coolant flow faults, recipe-driven
(10 each).

Splits. Recipe normal: the first 60 (by seed) to fit, of which every fifth (12) is held aside for the deep models'
early stopping and settings; the other 40 set alarm lines. Operator normal: the first 40 to fit, the other 20 for
lines. Two setups:
  recipe  fit and lines on recipe batches only; scores every recipe-driven fault
  mixed   fit and lines on recipe and operator batches together; scores fault A on both kinds of batch
One alarm line per setup from its line batches (whole batches, as on IndPenSim).

Detectors: the eight standard ones, the kinked-link, the straight forecaster, PCA and the curved-link scored like
the autoencoder, the four schedule-aware checks, the autoencoder (all channels, and the 11 controller outputs only)
and the LSTM forecaster, with notebook 09's IndPenSim settings and no retuning: the LSTM at IndPenSim's choice
(window 5, 2 layers of 64), the straight forecaster at its window (10), the autoencoder's bottleneck by its rule
(PCA's 90% of variance); three seeds where a detector has seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import indpensim
from .pensim import CAMPAIGN

N_FIT_RECIPE, N_FIT_OPERATOR = 60, 40
DEEP_SETTINGS = {"lstm": {"W": 5, "hidden": 64, "layers": 2}, "straight_forecaster": {"W": 10}}
SCORED = {"recipe": ["A_recipe", "B_recipe", "C80_recipe", "C50_recipe", "C20_recipe", "aeration_recipe",
                     "pressure_recipe", "base_flow_recipe", "coolant_recipe"],
          "mixed": ["A_recipe", "A_operator"]}
VARIANT_TITLES = {"A_recipe": "A: as published", "B_recipe": "B: held at recipe values", "C80_recipe": "C: 80% of schedule",
                  "C50_recipe": "C: 50% of schedule", "C20_recipe": "C: 20% of schedule", "A_operator": "A on operator batches",
                  "aeration_recipe": "aeration", "pressure_recipe": "back pressure", "base_flow_recipe": "base flow",
                  "coolant_recipe": "coolant flow"}
METHODS_SIMPLE = ["rolling_z", "ewma", "cusum", "pca", "iforest", "matrix_profile", "link_straight", "link_curved",
                  "link_kinked", "straight_forecaster", "pca_twin", "link_curved_alike"]
METHODS_SCHEDULE = ["unseen_value", "nearest_1", "nearest_5", "recipe_hour"]
METHODS_DEEP = ["deep_ae", "deep_ae_outputs", "deep_lstm"]
SEEDS = 3


def plan(folder: Path = CAMPAIGN) -> pd.DataFrame:
    return pd.DataFrame(json.loads((Path(folder) / "plan.json").read_text()))


def load(name: str, folder: Path = CAMPAIGN) -> pd.DataFrame:
    return pd.read_csv(Path(folder) / f"{name}.csv.gz")


def batches(group: str, folder: Path = CAMPAIGN) -> list[dict]:
    """Every batch of a group, in seed order: {"name", "seed", "strategy", "X" (readings x CHANNELS), "fault", "time"}."""
    p = plan(folder)
    out = []
    for _, j in p[p["group"] == group].sort_values("seed").iterrows():
        b = load(j["name"], folder)
        out.append({"name": j["name"], "seed": int(j["seed"]), "strategy": j["strategy"], "group": group,
                    "X": b[indpensim.CHANNELS].to_numpy(float), "fault": b["fault"].to_numpy().astype(int),
                    "time": b["time"].to_numpy(float)})
    return out


def split() -> dict:
    """{"recipe fit", "recipe line", "operator fit", "operator line"}: lists of batches (fixed, by seed order)."""
    rec, op = batches("normal_recipe"), batches("normal_operator")
    return {"recipe fit": rec[:N_FIT_RECIPE], "recipe line": rec[N_FIT_RECIPE:],
            "operator fit": op[:N_FIT_OPERATOR], "operator line": op[N_FIT_OPERATOR:]}


def dataset(setup: str):
    """The Dataset for a setup ("recipe" or "mixed")."""
    from .data import Dataset, Run
    s = split()
    fit_b = s["recipe fit"] + (s["operator fit"] if setup == "mixed" else [])
    line_b = s["recipe line"] + (s["operator line"] if setup == "mixed" else [])
    F = np.vstack([b["X"] for b in fit_b])
    runs = [Run(b["name"], b["X"], b["fault"], F, group=b["group"], meta={"strategy": b["strategy"], "seed": b["seed"]})
            for g in SCORED[setup] for b in batches(g)]
    val = [Run(b["name"], b["X"], np.zeros(len(b["X"]), int), F, group="normal", meta={"strategy": b["strategy"]})
           for b in line_b]
    return Dataset(f"indpensim_campaign_{setup}", runs, indpensim.CHANNELS, 720.0, val, calibration="matched",
                   matched_start=0, fit_lengths=[len(b["X"]) for b in fit_b], output_columns=indpensim.OUTPUT_COLUMNS,
                   fit_groups=[b["strategy"] for b in fit_b], deep_settings=DEEP_SETTINGS)


def methods() -> list[str]:
    from . import detectors as det
    return [m for m in METHODS_SIMPLE + METHODS_SCHEDULE + METHODS_DEEP if m in det.available()]


def score(setup: str, verbose: bool = True):
    from . import deepdive, run  # noqa: F401  (deepdive registers the outputs-only autoencoder and curved-link alike)
    ds = dataset(setup)
    run.score_all([ds], methods=methods(), n_seeds=SEEDS, verbose=verbose)


def report(setup: str, variant: str, n_boot: int = 1000, verbose: bool = False) -> dict:
    """Every detector on one fault variant of a setup, ranges from resampling batches (cached)."""
    from . import deepdive, run, structure  # noqa: F401
    ds = dataset(setup)
    keep = [r.group == variant for r in ds.runs]
    return structure.cached(f"campaign__{setup}__{variant}", lambda: run.report_dataset(
        ds, methods(), SEEDS, n_boot, keep=keep, calibration="matched", verbose=verbose))


# ---------------------------------------------------------------- the generated batches against the published ones

def _published(strategy: str) -> list[pd.DataFrame]:
    d = indpensim.readings()
    pick = d[(d["strategy"] == strategy) & (d["faulty batch"] == 0)]
    return [x[["time"] + indpensim.CHANNELS].reset_index(drop=True) for _, x in pick.groupby("batch")]


def compare_ranges(strategy: str = "recipe") -> pd.DataFrame:
    """Each channel's 1st percentile, median and 99th percentile: generated normal batches of a strategy against
    the published normal batches of the same strategy; batch lengths in attrs."""
    gen = batches(f"normal_{strategy}")
    G = pd.DataFrame(np.vstack([b["X"] for b in gen]), columns=indpensim.CHANNELS)
    pub = _published(strategy)
    P = pd.concat(pub)
    rows = {}
    for c in indpensim.CHANNELS:
        a, b = (np.quantile(x[c].dropna(), [0.01, 0.5, 0.99]) for x in (G, P))
        rows[c] = {"generated: 1%": a[0], "generated: median": a[1], "generated: 99%": a[2],
                   "published: 1%": b[0], "published: median": b[1], "published: 99%": b[2]}
    out = pd.DataFrame(rows).T
    out.index.name = "channel"
    out.attrs["batch hours"] = {"generated": (min(b["time"][-1] for b in gen), max(b["time"][-1] for b in gen)),
                                "published": (min(float(x["time"].iloc[-1]) for x in pub), max(float(x["time"].iloc[-1]) for x in pub))}
    out.attrs["batches"] = {"generated": len(gen), "published": len(pub)}
    return out


def curved_share(n_boot: int = 500) -> pd.DataFrame:
    """Notebook 08's measure (same moment, pooled, whole batches held out in five rounds) on the 60 generated recipe
    fitting batches and on the 30 published recipe batches."""
    from . import structure
    gen = [b["X"] for b in split()["recipe fit"]]
    pub = [x[indpensim.CHANNELS].to_numpy(float) for x in _published("recipe")]
    rows = {}
    for label, runs in (("generated recipe batches (60)", gen), ("published recipe batches (30)", pub)):
        folds = [np.full(len(r), i % 5) for i, r in enumerate(runs)]
        res = structure.cached(f"campaign_curved__{label.split()[0]}", lambda runs=runs, folds=folds: structure.curvature_by_run(
            runs, indpensim.CHANNELS, folds=folds, n_boot=n_boot, verbose=False))
        rows[label] = {"straight share": res["straight"], "curved share, points": 100 * res["curved"],
                       "range": f"{100 * res['curved_low']:+.1f} to {100 * res['curved_high']:+.1f}"}
    out = pd.DataFrame(rows).T
    out.index.name = "batches"
    return out


# ---------------------------------------------------------------- the fixed questions

FAMILIES = {"per-sensor checks": ["rolling_z", "ewma", "cusum"], "PCA": ["pca", "pca_twin"],
            "links": ["link_straight", "link_kinked", "link_curved", "link_curved_alike"],
            "shape in time": ["matrix_profile", "straight_forecaster", "iforest"],
            "schedule-aware checks": METHODS_SCHEDULE, "deep": METHODS_DEEP}


def _names(keys):
    from . import deepdive  # noqa: F401  (registers the outputs-only autoencoder and the curved-link alike)
    from . import detectors as det
    return [det.REGISTRY[k].name for k in keys]


def _row(res, a, b):
    from . import deep
    g = deep.gap(res, a, b)
    return 100 * g["gap"], f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}", g["verdict"]


def variant_table(setup: str, variant: str) -> pd.DataFrame:
    """Every detector on one variant: caught (mean over seeds), its range and false alarms on the faulty batches'
    own normal readings, with its family."""
    from . import deep
    s = deep._summary(report(setup, variant))
    fam = {n: f for f, keys in FAMILIES.items() for n in _names(keys)}
    out = s[["dr_at_1pct", "dr_ci90_low", "dr_ci90_high", "far_own_normal"]].copy()
    out.columns = ["caught (mean of seeds)", "range low (seed 0)", "range high (seed 0)", "false alarms on own normal readings"]
    out.insert(0, "family", [fam.get(m, "") for m in out.index])
    return out.sort_values("caught (mean of seeds)", ascending=False)


def questions(setup: str = "recipe") -> pd.DataFrame:
    """Per variant: the autoencoder against the best previous simple detector (Q1), the best schedule-aware check
    (Q2) and the best simple detector of all; and the best deep model against the best simple detector (Q6)."""
    from . import deep
    rows = {}
    previous = _names(METHODS_SIMPLE)
    schedule = _names(METHODS_SCHEDULE)
    deepn = _names(METHODS_DEEP)
    for v in SCORED[setup]:
        res = report(setup, v)
        s = deep._summary(res)["dr_at_1pct"]
        prev = s[[m for m in previous if m in s.index]].idxmax()
        sch = s[[m for m in schedule if m in s.index]].idxmax()
        best = s[[m for m in previous + schedule if m in s.index]].idxmax()
        bdeep = s[[m for m in deepn if m in s.index]].idxmax()
        r = {"faulty batches": res["runs"], "autoencoder": s["Autoencoder"]}
        for label, other in (("best previous simple", prev), ("best schedule-aware check", sch), ("best simple of all", best)):
            g, rng, verdict = _row(res, "Autoencoder", other)
            r[label] = f"{other} ({s[other]:.2f})"
            r[f"autoencoder minus {label}, points"] = g
            r[f"autoencoder minus {label}, range"] = rng
            r[f"autoencoder minus {label}, verdict"] = verdict
        g, rng, verdict = _row(res, bdeep, best)
        r["best deep"] = f"{bdeep} ({s[bdeep]:.2f})"
        r["best deep minus best simple, points"], r["best deep minus best simple, range"], r["best deep minus best simple, verdict"] = g, rng, verdict
        rows[VARIANT_TITLES[v]] = r
    out = pd.DataFrame(rows).T
    out.index.name = f"fault ({setup} setup)"
    return out


def q3_table() -> pd.DataFrame:
    """Fault B: the schedule-aware checks and the deep models, with ranges."""
    t = variant_table("recipe", "B_recipe")
    return t[t["family"].isin(["schedule-aware checks", "deep"])]


def q4_table() -> pd.DataFrame:
    """Fault C, and A for reference: each family's best catch rate at each fault size."""
    rows = {}
    for v in ("C80_recipe", "C50_recipe", "C20_recipe", "A_recipe"):
        t = variant_table("recipe", v)
        rows[VARIANT_TITLES[v]] = {f: float(t.loc[t["family"] == f, "caught (mean of seeds)"].max()) for f in FAMILIES}
        rows[VARIANT_TITLES[v]]["best detector"] = t.index[0]
    out = pd.DataFrame(rows).T
    out.index.name = "fault size"
    return out


def q5_table() -> pd.DataFrame:
    """Fault A in the mixed setup (recipe and operator batches in fitting and lines) against the recipe setup:
    the schedule-aware checks and the autoencoder, and their gaps."""
    from . import deep
    rows = {}
    for setup, v in (("recipe", "A_recipe"), ("mixed", "A_recipe"), ("mixed", "A_operator")):
        res = report(setup, v)
        s = deep._summary(res)
        for m in _names(METHODS_SCHEDULE) + ["Autoencoder", "Autoencoder, outputs only"]:
            r = {"caught": s.loc[m, "dr_at_1pct"], "range": f"{s.loc[m, 'dr_ci90_low']:.2f}-{s.loc[m, 'dr_ci90_high']:.2f}",
                 "false alarms on own normal readings": s.loc[m, "far_own_normal"]}
            if m != "Autoencoder":
                g, rng, verdict = _row(res, "Autoencoder", m)
                r.update({"autoencoder minus it, points": g, "range of gap": rng, "verdict": verdict})
            rows[(f"{setup} setup, {VARIANT_TITLES[v]}", m)] = r
    out = pd.DataFrame(rows).T
    out.index.names = ["setup and fault", "detector"]
    return out
