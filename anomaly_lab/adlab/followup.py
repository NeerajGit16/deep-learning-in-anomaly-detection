"""Follow-up round, part A: checks on earlier results. Nothing here changes a setting or refits a model;
every number is read from scores already in results/cache unless said otherwise.

A4  The one deep-model win (the autoencoder on IndPenSim): which batches carry it, with each faulty batch
    left out in turn, per seed, with the line set on the recipe batches alone, and against PCA scored
    like the autoencoder and the kinked-link detector.
A5  One-sensor faults (SKAB, Tennessee Eastman): do deep models beat the best per-sensor check?
A6  The six-mode plant: why telling the straight-link detector the mode made it worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from . import run
from . import stats as S
from .run import FAR, MARGIN

AE = "Autoencoder"
CURVED, STRAIGHT, KINKED, PCA_TWIN = ("Curved-link residual", "Straight-link residual", "Kinked-link residual",
                                      "PCA, scored like the autoencoder")


# ---------------------------------------------------------------- A4: the IndPenSim win

def _ips():
    from . import deep
    spec = deep.dataset("indpensim")
    methods = deep.SIMPLE + deep.SCHEDULE + deep.self_checks()["keep"]          # round 3: with the schedule checks
    return spec, methods


def _best_simple(res: dict) -> str:
    from . import deep
    from . import detectors as det
    s = deep._summary(res)["dr_at_1pct"]
    simple = [det.REGISTRY[k].name for k in deep.SIMPLE_ALL if det.REGISTRY[k].name in s.index]
    return s[simple].idxmax()


def ips_report(leave_out: int | None = None, line: str = "all", n_boot: int = 1000) -> dict:
    """The IndPenSim report with every detector (as deep.report), optionally without one faulty batch and/or
    with the alarm line set on one strategy's line batches only (cached)."""
    from . import structure
    spec, methods = _ips()
    ds = spec["ds"]
    if leave_out is None and line == "all":
        from . import deep
        return deep.report("indpensim")
    keep = [r.meta["batch"] != leave_out for r in ds.runs]
    line_keep = None if line == "all" else [r.meta["strategy"] == line for r in ds.val_normal]
    return structure.cached(f"followup_ips__without_{leave_out}__line_{line}__with_schedule_checks", lambda: run.report_dataset(
        ds, methods, 5, n_boot, keep=keep, calibration="matched", verbose=False, sources=spec["sources"],
        line_keep=line_keep))


def ips_gaps(res: dict) -> dict:
    """The autoencoder against the best simple detector, PCA scored like it, the kinked-link and the straight-link."""
    from . import deep
    s = deep._summary(res)["dr_at_1pct"]
    best = _best_simple(res)
    out = {"autoencoder caught": float(s[AE]), "best simple": best, "best simple caught": float(s[best])}
    for label, other in [("best simple", best), ("PCA scored alike", PCA_TWIN), ("kinked-link", KINKED),
                         ("straight-link", STRAIGHT)]:
        g = deep.gap(res, AE, other)
        out[f"vs {label}"] = 100 * g["gap"]
        out[f"vs {label}, range"] = f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}"
        out[f"vs {label}, verdict"] = g["verdict"]
    return out


def _seed_scores(method: str, seed: int, source=None):
    spec, _ = _ips()
    src = source or spec["sources"].get(method, spec["ds"])
    runs, _, _ = run.load_scores(src.name, method, seed)
    return runs, run.calibration_scores(src, method, seed)


def ips_by_batch() -> pd.DataFrame:
    """Catch rate per faulty batch (each detector's own alarm line; mean over its seeds), with its fault type."""
    from . import deep, indpensim
    from . import detectors as det
    spec, _ = _ips()
    ds = spec["ds"]
    labels = [r.y for r in ds.runs]
    names = [r.meta["batch"] for r in ds.runs]
    keys = {AE: "deep_ae", CURVED: "link_curved", STRAIGHT: "link_straight", KINKED: "link_kinked", PCA_TWIN: "pca_twin",
            "Straight forecaster": "straight_forecaster", "LSTM forecaster": "deep_lstm",
            "Unseen value check": "unseen_value", "Nearest normal reading (k = 1)": "nearest_1"}
    cols = {}
    for label, key in keys.items():
        seeds = run._seeds(key, deep.SEEDS if key in deep.DEEP else 5)
        per = []
        for seed in seeds:
            runs, cal = _seed_scores(key, seed)
            thr = M.dr_at_far(runs, labels, FAR, "matched", cal)["threshold"]
            per.append(M.dr_by_group(runs, labels, names, thr))
        cols[det.REGISTRY[key].name] = pd.DataFrame(per).mean()
    every = {}
    for key in deep.SIMPLE_ALL:                   # the best of every simple detector, batch by batch
        src = spec["sources"].get(key, ds)
        if not run._cache_path(src.name, key, 0).exists():
            continue
        runs, cal = _seed_scores(key, 0)
        thr = M.dr_at_far(runs, labels, FAR, "matched", cal)["threshold"]
        every[det.REGISTRY[key].name] = pd.Series(M.dr_by_group(runs, labels, names, thr))
    every = pd.DataFrame(every)
    out = pd.DataFrame(cols)
    out["best simple detector on this batch (seed 0)"] = every.max(1)
    out["which"] = every.idxmax(1)
    types = indpensim.fault_types()
    out.insert(0, "fault type", types.loc[out.index, "fault type"])
    out.insert(1, "timing", types.loc[out.index, "timing"])
    out.insert(2, "faulty readings", [int(np.sum(r.y)) for r in ds.runs])
    out["autoencoder minus curved-link, points"] = 100 * (out[AE] - out[CURVED])
    out.index.name = "batch"
    return out


def ips_leave_one_out() -> pd.DataFrame:
    """The autoencoder's gaps with each faulty batch left out in turn (the best simple detector recomputed each time)."""
    from . import indpensim
    types = indpensim.fault_types()
    rows = {}
    for b in types.index:
        rows[int(b)] = {"fault type left out": types.loc[b, "fault type"], **ips_gaps(ips_report(leave_out=int(b)))}
    out = pd.DataFrame(rows).T
    out.index.name = "batch left out"
    return out


def ips_per_seed(n_boot: int = 1000) -> pd.DataFrame:
    """Each autoencoder seed alone against the best simple detector (seed 0 for both detectors that have seeds)."""
    from . import deep, structure
    from . import detectors as det

    def compute():
        spec, _ = _ips()
        ds = spec["ds"]
        labels = [r.y for r in ds.runs]
        res = deep.report("indpensim")
        best = _best_simple(res)
        best_key = next(k for k in deep.SIMPLE_ALL if det.REGISTRY[k].name == best)
        rows = {}
        for seed in range(deep.SEEDS):
            primary, calib = {}, {}
            primary[AE], calib[AE] = _seed_scores("deep_ae", seed)
            primary[best], calib[best] = _seed_scores(best_key, 0)
            draws = S.bootstrap_dr(primary, labels, FAR, n_boot, mode="matched", calib=calib, verbose=False)
            point = S.point_dr(primary, labels, FAR, "matched", calib)
            pair = S.paired_differences(draws, point)[0]
            lo, hi = sorted((pair["ci90_low"], pair["ci90_high"])) if pair["method_a"] == AE else \
                sorted((-pair["ci90_low"], -pair["ci90_high"]))
            gap = point[AE] - point[best]
            win = abs(gap) >= MARGIN and (lo > 0 or hi < 0)
            rows[f"seed {seed}"] = {"autoencoder caught": point[AE], "best simple": best, "best simple caught": point[best],
                                    "gap, points": 100 * gap, "range": f"{100 * lo:+.1f} to {100 * hi:+.1f}",
                                    "verdict": ("ahead" if gap > 0 else "behind") if win else "tie"}
        out = pd.DataFrame(rows).T
        out.index.name = "autoencoder seed"
        return out

    return structure.cached("followup_ips__per_seed__with_schedule_checks", compute)


def ips_summary() -> pd.DataFrame:
    """The headline gaps under the main line and under the line set on the recipe batches alone."""
    return pd.DataFrame({"one line for all strategies (the headline)": ips_gaps(ips_report()),
                         "line from the recipe batches alone": ips_gaps(ips_report(line="recipe"))}).T


# ---------------------------------------------------------------- A5: one-sensor faults
#
# Fixed before computing: a "one-sensor fault" is a fault that structure.fault_types labels "single sensor"
# on that dataset under its own alarm line (the per-sensor checks catch at least half of it and at most
# three sensors move): SKAB (per-run line), the Rieth runs of Tennessee Eastman (matched line, notebook 03's
# "fresh runs, new line") and the six-mode plant (notebook 06). The best per-sensor check is the best of
# rolling z, EWMA and CUSUM on those faults. Each deep model that passed its self-checks is compared with
# it by the margin rule, ranges from resampling runs.

LEVEL = ["rolling_z", "ewma", "cusum"]
ONE_SENSOR_DATASETS = {"skab": "Water pump (SKAB)", "tep_rieth": "Tennessee Eastman, one setting (Rieth runs)",
                       "tep6": "Tennessee Eastman, six modes"}


def _missing(name):
    def fail():
        raise FileNotFoundError(f"results/cache/{name}.pkl: run the notebook that computes it first")
    return fail


def fault_kind_table(name: str) -> pd.DataFrame:
    from . import data, multimode, structure
    if name == "skab":
        return structure.cached("skab_kinds", lambda: structure.fault_types(data.load("skab"), saved=True))
    if name == "tep_rieth":
        return structure.cached("fresh_kinds", _missing("fresh_kinds"))["fresh runs, new line"]
    if name == "tep6":
        return multimode.fault_kinds()
    raise KeyError(name)


def one_sensor_faults(name: str) -> list[str]:
    kinds = fault_kind_table(name)
    return [str(g) for g, k in kinds["kind"].items() if k == "single sensor"]


def one_sensor_report(name: str, n_boot: int = 1000) -> dict:
    from . import deep, structure
    spec = deep.dataset(name)
    ds = spec["ds"]
    faults = set(one_sensor_faults(name))
    keep = [str(r.group) in faults for r in ds.runs]
    models = deep.self_checks()["keep"]
    methods = LEVEL + models
    return structure.cached(f"followup_one_sensor__{name}__{'_'.join(models)}", lambda: run.report_dataset(
        ds, methods, 5, n_boot, keep=keep, calibration=spec["calibration"], verbose=False, sources=spec["sources"]))


def one_sensor_table(name: str) -> pd.DataFrame:
    """Per deep model: its catch rate on the one-sensor faults, against the best per-sensor check there."""
    from . import deep
    from . import detectors as det
    res = one_sensor_report(name)
    s = deep._summary(res)["dr_at_1pct"]
    level = [det.REGISTRY[k].name for k in LEVEL]
    best = s[level].idxmax()
    rows = {}
    for key in deep.self_checks()["keep"]:
        m = det.REGISTRY[key].name
        g = deep.gap(res, m, best)
        rows[m] = {"faults": ", ".join(one_sensor_faults(name)), "runs": res["runs"], "caught": s[m],
                   "best per-sensor check": best, "it caught": s[best], "gap, points": 100 * g["gap"],
                   "range": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}", "verdict": g["verdict"]}
    out = pd.DataFrame(rows).T
    out.index.name = "deep model"
    return out


def one_sensor_tables() -> pd.DataFrame:
    return pd.concat({title: one_sensor_table(name) for name, title in ONE_SENSOR_DATASETS.items()},
                     names=["dataset", "deep model"])


# ---------------------------------------------------------------- A6: the six-mode plant, mode told
#
# The idea to check (the user's): without the mode, the straight model must lean on strong cross-sensor
# relations to tell the modes apart, and faults break those; told the mode, it leans on the weak links
# inside a mode. Compared, for the straight-link detector fitted on the six-mode fit data with and without
# the mode: the size of each sensor's cross-sensor coefficients, the spread of its held-out errors, and how
# far the faults push each version past its own lines. Report only; nothing is refitted for scoring.

def _six_mode_models():
    from . import detectors as det
    from . import multimode, structure

    def compute():
        plain_ds, told_ds = multimode.detector_dataset(), multimode.detector_dataset(told=True)
        plain = det.make("link_straight").fit(plain_ds.runs[0].fit)
        told = det.make("link_straight_told").fit(told_ds.runs[0].fit)
        out = {"plain": plain, "told": told, "channels": plain_ds.channels}
        for key, model, ds in (("plain", plain, plain_ds), ("told", told, told_ds)):
            X, E = model._separate(ds.runs[0].fit)
            Z = model._z(X)
            out[f"{key} held-out errors"] = Z - model._held_out(Z, E)
            over, worst = [], []
            for r in ds.runs:
                s = np.abs(model.errors(r.X)) / model.ref_now_
                f = r.y.astype(bool)
                over.append((s[f] > 1).sum(1))
                worst.append(s[f].max(1))
            out[f"{key} sensors over their line"] = np.concatenate(over)
            out[f"{key} largest surprise"] = np.concatenate(worst)
        return out

    return structure.cached("followup_six_mode_told_models", compute)


def mode_told_by_sensor() -> pd.DataFrame:
    """Per sensor: summed size of its cross-sensor coefficients (standardised units) and the spread of its
    held-out errors, without and with the mode told; and how far apart the told model puts the six modes' levels."""
    m = _six_mode_models()
    plain, told = m["plain"], m["told"]
    rows = {}
    for j, (others, m1, _) in plain.models_.items():
        o2, t1, _ = told.models_[j]
        n = int(o2.sum())
        levels = t1.coef_[n:]
        rows[m["channels"][j]] = {
            "cross-sensor coefficients, no mode": float(np.abs(m1.coef_).sum()),
            "cross-sensor coefficients, mode told": float(np.abs(t1.coef_[:n]).sum()),
            "mode levels, spread (told)": float(levels.max() - levels.min()) if len(levels) else np.nan,
            "error spread, no mode": float(m[f"plain held-out errors"][:, j].std()),
            "error spread, mode told": float(m[f"told held-out errors"][:, j].std()),
        }
    out = pd.DataFrame(rows).T
    out["coefficients, told / none"] = out["cross-sensor coefficients, mode told"] / out["cross-sensor coefficients, no mode"]
    out["error spread, told / none"] = out["error spread, mode told"] / out["error spread, no mode"]
    out.index.name = "sensor"
    return out


def mode_told_summary() -> pd.DataFrame:
    t = mode_told_by_sensor()
    m = _six_mode_models()
    rows = {"no mode": {}, "mode told": {}}
    for key, label in (("plain", "no mode"), ("told", "mode told")):
        col = "no mode" if key == "plain" else "mode told"
        rows[label] = {"median cross-sensor coefficients (summed, per sensor)": t[f"cross-sensor coefficients, {col}"].median(),
                       "median held-out error spread (standardised units)": t[f"error spread, {col}"].median(),
                       "fault readings: median sensors over their own line": float(np.median(m[f"{key} sensors over their line"])),
                       "fault readings: mean sensors over their own line": float(np.mean(m[f"{key} sensors over their line"])),
                       "fault readings: median largest surprise (1 = the line)": float(np.median(m[f"{key} largest surprise"]))}
    return pd.DataFrame(rows).T


def mode_told_folds() -> pd.DataFrame:
    """Why the told model's held-out errors are wide. The fit data hold the six modes one after another (four
    runs each), and the detector's five blocks of time each hold out most of one mode, so the told model has
    to predict a mode whose level it never saw. For comparison only (no detector is changed): the median
    held-out error spread of both models when each fold instead holds out one run of every mode."""
    from . import multimode, structure

    def compute():
        m = _six_mode_models()
        told_ds = multimode.detector_dataset(told=True)
        fit = told_ds.runs[0].fit
        mode = fit[:, -1]
        n_run = told_ds.fit_lengths[0]
        run_in_mode = (np.arange(len(fit)) // n_run) % len(multimode.FIT_RUNS)
        blocks = np.array_split(np.arange(len(fit)), 5)
        rows = {}
        for i, b in enumerate(blocks):
            share = pd.Series(mode[b]).value_counts(normalize=True)
            rows[f"block {i + 1}"] = {"mode mostly held out": int(share.idxmax()), "its share of the block": float(share.max()),
                                      "rows of that mode left to fit on": int(np.sum(np.delete(mode, b) == share.idxmax()))}
        blocks_table = pd.DataFrame(rows).T
        spreads = {}
        for key in ("plain", "told"):
            model = m[key]
            X = fit if key == "told" else fit[:, :-1]
            Xs, E = model._separate(X)
            Z = model._z(Xs)
            pred = np.zeros_like(Z)
            for k in range(len(multimode.FIT_RUNS)):
                test = np.flatnonzero(run_in_mode == k)
                train = np.flatnonzero(run_in_mode != k)
                for j, (others, _, _) in model.models_.items():
                    A = np.hstack([Z[:, others], E])
                    m1, m2 = model._fit_pair(A[train], Z[train, j])
                    pred[test, j] = model._predict(m1, m2, A[test])
            spreads[key] = np.median([(Z - pred)[:, j].std() for j in model.models_])
        t = mode_told_by_sensor()
        compare = pd.DataFrame({
            "median held-out error spread, the detector's blocks of time": {"no mode": t["error spread, no mode"].median(),
                                                                             "mode told": t["error spread, mode told"].median()},
            "median held-out error spread, one run of every mode held out": {"no mode": spreads["plain"], "mode told": spreads["told"]}})
        return {"blocks": blocks_table, "spreads": compare}

    return structure.cached("followup_six_mode_told_folds", compute)


# ---------------------------------------------------------------- round 3, part 1: schedule-aware checks on IndPenSim

def schedule_checks_table() -> pd.DataFrame:
    """Each schedule-aware check on IndPenSim (notebook 08's setup): catch rate with range, false alarms on the
    faulty batches' own normal readings, and the autoencoder minus it, under the line from all 30 line batches
    and under the line from the 10 recipe line batches."""
    from . import deep
    from . import detectors as det
    rows = {}
    for line in ("all", "recipe"):
        res = ips_report(line=line)
        s = deep._summary(res)
        for key in deep.SCHEDULE:
            m = det.REGISTRY[key].name
            g = deep.gap(res, AE, m)
            rows[(m, "one line for all strategies" if line == "all" else "line from the recipe batches")] = {
                "caught": s.loc[m, "dr_at_1pct"], "range": f"{s.loc[m, 'dr_ci90_low']:.2f}-{s.loc[m, 'dr_ci90_high']:.2f}",
                "false alarms on the faulty batches' normal readings": s.loc[m, "far_own_normal"],
                "autoencoder caught": s.loc[AE, "dr_at_1pct"], "autoencoder minus it, points": 100 * g["gap"],
                "gap range": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}",
                "within 12 points of the autoencoder": abs(100 * g["gap"]) < 12}
    out = pd.DataFrame(rows).T
    out.index.names = ["check", "alarm line"]
    return out


def recipe_hour_on_normal() -> pd.DataFrame:
    """Why the recipe-at-this-hour check misses: on the 30 normal line batches, the 99th percentile of its score
    and the output that most often gives the largest gap, by control strategy (normal data only)."""
    from . import detectors as det
    from . import indpensim
    ds = indpensim.detector_dataset()
    m = det.RecipeAtHour()
    m.context_ = {"lengths": ds.fit_lengths, "outputs": ds.output_columns, "fit_groups": ds.fit_groups}
    m.fit(ds.runs[0].fit)
    names = [indpensim.CHANNELS[i] for i in m.cols_]
    med, spread = m.profile_
    rows = []
    for r in ds.val_normal:
        Xo = r.X[:, m.cols_]
        idx = np.minimum(np.arange(len(Xo)), len(med) - 1)
        S = np.abs(Xo - med[idx]) / spread[idx]
        rows.append({"strategy": r.meta["strategy"], "99th percentile of the score": float(np.quantile(S.max(1), 0.99)),
                     "output giving the largest gap most often": names[int(np.bincount(S.argmax(1), minlength=len(names)).argmax())]})
    t = pd.DataFrame(rows)
    out = t.groupby("strategy").agg(**{"line batches": ("strategy", "size"),
                                       "median of the 99th percentiles": ("99th percentile of the score", "median"),
                                       "commonest driver": ("output giving the largest gap most often", lambda s: s.value_counts().index[0])})
    # the substrate fault for comparison: the score on the three batches' 76-92 h readings
    fault = []
    for r in ds.runs:
        if r.meta["batch"] in (91, 95, 97):
            Xo = r.X[:, m.cols_]
            idx = np.minimum(np.arange(len(Xo)), len(med) - 1)
            S = (np.abs(Xo - med[idx]) / spread[idx]).max(1)
            t_h = np.arange(len(Xo)) * 0.2 + 0.2
            fault.append(np.median(S[(t_h >= 76) & (t_h <= 92)]))
    out.attrs["median score, faulty readings at 76-92 h (batches 91, 95, 97)"] = float(np.median(fault))
    return out


# ---------------------------------------------------------------- round 3, part 4: balanced folds on the six-mode plant

def balanced_folds_table() -> pd.DataFrame:
    """The straight-link detector, plain and told the mode, with its error scale set on the old blocks of time and
    on balanced folds (one run of every mode held out at a time), beside the straight forecaster; six-mode plant."""
    from . import deep
    res = deep.report("tep6", verbose=False)
    s = deep._summary(res)
    names = ["Straight-link residual", "Straight-link residual, stage told", "Straight-link residual, balanced folds",
             "Straight-link residual, stage told, balanced folds", "Straight forecaster", "LSTM forecaster", "Autoencoder"]
    rows = {}
    for m in names:
        g = deep.gap(res, m, "Straight-link residual") if m != "Straight-link residual" else None
        rows[m.replace("stage told", "mode told")] = {
            "caught": s.loc[m, "dr_at_1pct"], "range": f"{s.loc[m, 'dr_ci90_low']:.2f}-{s.loc[m, 'dr_ci90_high']:.2f}",
            "false alarms before the faults": s.loc[m, "far_own_normal"],
            "minus the straight-link (blocks), points": 100 * g["gap"] if g else 0.0,
            "range of that gap": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}" if g else ""}
    out = pd.DataFrame(rows).T
    out.index.name = "detector"
    return out
