"""Round 4 (notebooks 09 and 10): the fold audit, classic simple detectors on every dataset, false alarms
before the fault, and the repeat share.

Part 1, the fold audit (folds.py has the rule, fixed before computing). Where the audit flags a dataset,
its fold-using detectors are scored again on a copy of the dataset with balanced folds (folds.balanced_copy),
and every number that moves by more than 2 points is listed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import folds, structure

MOVED = 0.02            # a number is listed as moved when it changes by more than 2 points


# ---------------------------------------------------------------- part 1: the fold audit

def audit() -> pd.DataFrame:
    return structure.cached("round4_fold_audit", folds.audit)


def switched() -> dict:
    """{dataset key: {scheme: [kinds of group]}} for the datasets the audit flags."""
    return folds.flagged(audit())


def _groups(name: str, kinds: list) -> np.ndarray:
    s = folds.fit_structure(name)
    labels = [np.asarray(s["kinds"][k]).astype(str) for k in kinds]
    out = labels[0]
    for extra in labels[1:]:
        out = np.char.add(np.char.add(out, "|"), extra)
    return out


def spec(name: str, balanced: bool = True) -> dict:
    """deep.dataset(name), with balanced-fold copies of the dataset (and of its told copy) where the audit flags it."""
    from . import deep
    base = deep.dataset(name)
    flags = switched().get(name) if balanced else None
    if not flags:
        return base
    kinds = sorted({k for ks in flags.values() for k in ks})
    groups = _groups(name, kinds)
    schemes = tuple(flags)

    def bal(ds):
        return folds.balanced_copy(ds, groups, schemes) if ds is not None else None

    out = dict(base)
    out["ds"] = bal(base["ds"])
    out["sources"] = {m: bal(d) for m, d in base["sources"].items()}
    out["also"] = [bal(d) for d in base["also"]]
    out["balanced"] = {"kinds": kinds, "schemes": list(schemes)}
    return out


def fold_methods(name: str) -> list:
    """The registry keys on this dataset whose scores change with balanced folds."""
    from . import deep
    flags = switched().get(name, {})
    out = []
    if folds.BLOCKS in flags:
        out += [m for m in ("link_straight", "link_curved", "link_kinked", "link_straight_told")
                if m != "link_straight_told" or deep.dataset(name)["sources"]]
    if folds.ASIDE in flags:
        out += ["straight_forecaster", "pca_twin"] + deep.self_checks()["keep"]
    return out


def score_balanced(name: str, verbose: bool = True):
    """Score the fold-using detectors on the balanced copy of a flagged dataset (cached)."""
    from . import deep, progress, run
    s = spec(name)
    if "balanced" not in s:
        return
    progress.note(f"round 4: balanced folds on {name}")
    keys = fold_methods(name)
    told = [m for m in keys if m in s["sources"]]
    deep_keys = [m for m in keys if m in deep.MODELS]
    run.score_all([s["ds"]], methods=[m for m in keys if m not in told and m not in deep_keys], n_seeds=1, verbose=verbose)
    if deep_keys:
        run.score_all([s["ds"]], methods=deep_keys, n_seeds=deep.SEEDS, verbose=verbose)
    for m in told:
        run.score_all([s["sources"][m]], methods=[m], n_seeds=1, verbose=verbose)
    progress.note(f"round 4: balanced folds on {name} done")


def _lineup(name: str) -> list:
    """Notebook 09's lineup on a dataset (round 3), less the round-3 balanced-fold extras."""
    from . import deep
    base = deep.dataset(name)
    keep = deep.self_checks()["keep"]
    return [m for m in deep.SIMPLE + deep._schedule(base) + keep
            if m != "link_straight_told" or base["sources"] or name == "etch"]


def _report(name: str, balanced: bool, calibration: str | None = None, n_boot: int = 1000) -> dict:
    from . import run
    s = spec(name, balanced)
    return run.report_dataset(s["ds"], _lineup(name), 5, n_boot, calibration=calibration or s["calibration"],
                              verbose=False, sources=s["sources"])


LINES = {"cranfield": {"each run's own line (notebook 09)": "per_run", "line from run T1 (notebook 05)": "matched"},
         "tep6": {"line from separate normal runs": "matched"}}


def fold_reports(name: str) -> dict:
    """{line: {"before": report, "after": report}} for a flagged dataset (cached)."""
    def compute():
        return {line: {"before": _report(name, False, cal), "after": _report(name, True, cal)}
                for line, cal in LINES[name].items()}
    return structure.cached(f"round4_fold_reports__{name}", compute)


def _rates(res: dict) -> pd.DataFrame:
    return pd.DataFrame(res["summary"]).set_index("method")


def fold_changes(name: str) -> pd.DataFrame:
    """Every detector whose catch rate moves by more than 2 points with balanced folds, per alarm line."""
    rows = []
    for line, pair in fold_reports(name).items():
        b, a = _rates(pair["before"]), _rates(pair["after"])
        for m in a.index:
            d = float(a.loc[m, "dr_at_1pct"] - b.loc[m, "dr_at_1pct"])
            if abs(d) > MOVED:
                rows.append({"alarm line": line, "detector": m,
                             "before": f"{b.loc[m, 'dr_at_1pct']:.3f} ({b.loc[m, 'dr_ci90_low']:.2f}-{b.loc[m, 'dr_ci90_high']:.2f})",
                             "after": f"{a.loc[m, 'dr_at_1pct']:.3f} ({a.loc[m, 'dr_ci90_low']:.2f}-{a.loc[m, 'dr_ci90_high']:.2f})",
                             "change, points": 100 * d})
    out = pd.DataFrame(rows)
    return out.set_index(["alarm line", "detector"]) if len(out) else out


def _points(text: str) -> str:
    """'-0.05 to +0.02' (shares) as '-5 to +2' (points)."""
    lo, hi = (float(v) for v in text.split(" to "))
    return f"{100 * lo:+.0f} to {100 * hi:+.0f}"


def gap_changes(name: str) -> pd.DataFrame:
    """Notebook 09's gaps (each deep model to its twin and to the best simple detector) before and after."""
    from . import deep
    rows = []
    for line, pair in fold_reports(name).items():
        tb, ta = deep.table(pair["before"]), deep.table(pair["after"])
        for model in ta.index:
            for col, what in (("gap to twin", "against twin"), ("gap to best simple", "against best simple")):
                d = float(ta.loc[model, col] - tb.loc[model, col])
                rows.append({"alarm line": line, "deep model": model, "gap": col,
                             "before": f"{100 * tb.loc[model, col]:+.1f} ({_points(tb.loc[model, col + ', range'])}) {tb.loc[model, what]}",
                             "after": f"{100 * ta.loc[model, col]:+.1f} ({_points(ta.loc[model, col + ', range'])}) {ta.loc[model, what]}",
                             "best simple after": ta.loc[model, "best simple detector"] if col == "gap to best simple" else ta.loc[model, "twin"],
                             "change, points": 100 * d, "moved": abs(d) > MOVED})
    return pd.DataFrame(rows).set_index(["alarm line", "deep model", "gap"])


# ---------------------------------------------------------------- part 2: classic simple detectors on every dataset

STEPPED = "recipe_hour_stepped"


def _datasets_to_score(name: str) -> list:
    """The dataset and the separate datasets its alarm line comes from (not the told copies)."""
    s = spec(name, balanced=False)
    return [s["ds"]] + [d for d in s["also"] if d is not None and not d.name.endswith("_told")
                        and d.name != "etch_told" and d is not s["ds"]]


def _fits(ds) -> list:
    seen, out = set(), []
    for r in ds.runs:
        if id(r.fit) not in seen:
            seen.add(id(r.fit))
            out.append(r.fit)
    return out


def classic(name: str) -> list:
    """Part 2's detectors on a dataset: the unseen value check on all sensors only where some sensor takes 30 or fewer
    distinct values in a fitting array; recipe at this hour (stepped outputs) only where controller outputs are named."""
    from . import detectors as det
    ds = _datasets_to_score(name)[0] if not name.startswith("campaign_") else _campaign_ds(name)
    keys = [k for k in det.CLASSIC if k != "unseen_all" or any(det.applicable_unseen(f) for f in _fits(ds))]
    return keys + ([STEPPED] if ds.output_columns else [])


def _campaign_ds(name: str):
    from . import campaign
    return campaign.dataset(name.split("_", 1)[1])


def score_classic(name: str, verbose: bool = True):
    from . import progress, run
    progress.note(f"round 4 part 2: classic detectors on {name}")
    keys = classic(name)
    targets = [_campaign_ds(name)] if name.startswith("campaign_") else _datasets_to_score(name)
    for ds in targets:
        run.score_all([ds], methods=keys, n_seeds=1, verbose=verbose)
    progress.note(f"round 4 part 2: {name} done")


def lineup(name: str) -> list:
    """Round 4's lineup on a real dataset: notebook 09's (with balanced folds where the audit flags it) and part 2's."""
    return _lineup(name) + classic(name)


def report(name: str, n_boot: int = 1000, verbose: bool = False) -> dict:
    """Every detector of round 4 on one real dataset, under its own alarm line (cached)."""
    from . import batch, etch, run
    from . import detectors as det
    s = spec(name)
    methods = lineup(name)

    def compute():
        if name == "etch":
            ds = s["ds"]
            per_method = {}
            for m in methods:
                src = etch.TOLD_NAME if m == "link_straight_told" else ds.name
                seeds = [k for k in run._seeds(m, 5) if run._cache_path(src, m, k).exists()]
                per_method[det.REGISTRY[m].name] = [run.load_scores(src, m, k)[:2] for k in seeds]
            return batch.report(per_method, [r.group for r in ds.runs], [r.meta["experiment"] for r in ds.runs],
                                [r.meta["experiment"] for r in ds.val_normal], [r.meta["role"] for r in ds.val_normal],
                                per_group=True, n_boot=n_boot, verbose=verbose)
        return run.report_dataset(s["ds"], methods, 5, n_boot, calibration=s["calibration"], verbose=verbose,
                                  sources=s["sources"])

    return structure.cached(f"round4_report__{name}", compute)


def campaign_methods(setup: str) -> list:
    from . import campaign
    return campaign.methods() + classic(f"campaign_{setup}")


def campaign_report(setup: str, variant: str, n_boot: int = 1000) -> dict:
    """Every detector of round 4 on one fault variant of the generated campaign (cached)."""
    from . import campaign, deepdive, run  # noqa: F401
    ds = campaign.dataset(setup)
    keep = [r.group == variant for r in ds.runs]
    return structure.cached(f"round4_campaign__{setup}__{variant}", lambda: run.report_dataset(
        ds, campaign_methods(setup), campaign.SEEDS, n_boot, keep=keep, calibration="matched", verbose=False))


# ---------------------------------------------------------------- part 3: false alarms before the fault
#
# Fixed before computing. For every detector on every dataset, under the dataset's own alarm line (round 4's
# report): the share of the faulty runs' own normal readings before the fault (every reading ahead of a run's
# first faulty reading) above the line, pooled over runs. Where it exceeds 3%, also the catch rate at each run's
# own line: the line that gives 1 false alarm in 100 on that run's readings before the fault (its 99th
# percentile), averaged over runs (each run counts once, as "each run's own line" on Cranfield). Point values are
# means over seeds; ranges come from resampling the faulty runs (1,000 draws) with seed 0, moved by the mean's
# difference from seed 0. On the metal etcher every reading of a faulty wafer is faulty, so the share of the
# check wafers' readings (normal wafers neither fitted on nor setting the line) above the line stands in.

CARRY = 0.03
N_BOOT = 1000


def _boot(values_by_run, weights_by_run, n_boot=N_BOOT, seed=0, mean_of_runs=False):
    """Point value and 90% range of a share over runs (resampled whole)."""
    v, w = np.asarray(values_by_run, float), np.asarray(weights_by_run, float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(v), (n_boot, len(v)))
    if mean_of_runs:
        point, draws = float(np.mean(v)), v[picks].mean(1)
    else:
        point, draws = float(v.sum() / max(w.sum(), 1)), v[picks].sum(1) / np.maximum(w[picks].sum(1), 1)
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return point, float(lo), float(hi)


def _carry_one(scores, labels, thr):
    """Per run: (count above the line before the fault, readings before the fault, own-line catch rate, count
    above the main line in the fault, faulty readings)."""
    out = []
    thr = thr if np.ndim(thr) else [thr] * len(scores)
    for s, y, t in zip(scores, labels, thr):
        y = np.asarray(y, bool)
        first = int(np.flatnonzero(y)[0]) if y.any() else len(y)
        before = s[:first]
        own = float(np.quantile(before, 0.99)) if len(before) else np.nan
        out.append((float((before > t).sum()), len(before), float(np.mean(s[y] > own)) if len(before) else np.nan,
                    float((s[y] > t).sum()), int(y.sum())))
    return np.array(out)


def carry_over(ds, methods, calibration, sources=None, keep=None, n_seeds=5) -> pd.DataFrame:
    """Part 3 for one dataset (not the etcher): per detector, false alarms before the fault and, where they exceed
    3%, the catch rate at each run's own line."""
    from . import metrics as M
    from . import run
    from . import detectors as det
    sources = sources or {}
    keep = [True] * len(ds.runs) if keep is None else list(keep)
    labels = [r.y for r, k in zip(ds.runs, keep) if k]
    rows = {}
    for m in methods:
        src = sources.get(m, ds)
        seeds = [s for s in run._seeds(m, n_seeds) if run._cache_path(src.name, m, s).exists()]
        if not seeds:
            continue
        per_seed = []
        for seed in seeds:
            scores = [s for s, k in zip(run.load_scores(src.name, m, seed)[0], keep) if k]
            thr = M.dr_at_far(scores, labels, run.FAR, calibration, run.calibration_scores(src, m, seed))["threshold"]
            per_seed.append(_carry_one(scores, labels, thr))
        mean = float(np.mean([p[:, 0].sum() / p[:, 1].sum() for p in per_seed]))
        p0 = per_seed[0]
        fa, lo, hi = _boot(p0[:, 0], p0[:, 1])
        shift = mean - fa
        row = {"false alarms before the fault": mean, "low": max(lo + shift, 0.0), "high": hi + shift,
               "runs": len(labels), "over 3%": bool(mean > CARRY)}
        if calibration == "per_run":
            row["caught, dataset's line"] = float(np.mean([np.mean(p[:, 3] / p[:, 4]) for p in per_seed]))
        else:
            row["caught, dataset's line"] = float(np.mean([p[:, 3].sum() / p[:, 4].sum() for p in per_seed]))
        if mean > CARRY:
            ok = ~np.isnan(p0[:, 2])
            own = float(np.mean([np.nanmean(p[:, 2]) for p in per_seed]))
            c, clo, chi = _boot(p0[ok, 2], np.ones(ok.sum()), mean_of_runs=True)
            row.update({"caught, each run's own line": own, "own line low": clo + own - c, "own line high": chi + own - c})
        rows[det.REGISTRY[m].name] = row
    out = pd.DataFrame(rows).T
    out.index.name = "detector"
    return out


def carry_over_etch(methods) -> pd.DataFrame:
    """The etcher: the share of check wafers' readings above each experiment's line (from its line wafers)."""
    from . import batch, etch, run
    from . import detectors as det
    ds = etch.dataset()
    groups = [r.meta["experiment"] for r in ds.val_normal]
    roles = [r.meta["role"] for r in ds.val_normal]
    line_idx = {g: [i for i, (gg, r) in enumerate(zip(groups, roles)) if gg == g and r == "line"] for g in dict.fromkeys(groups)}
    check = [i for i, r in enumerate(roles) if r == "check"]
    rows = {}
    for m in methods:
        src = etch.TOLD_NAME if m == "link_straight_told" else ds.name
        seeds = [s for s in run._seeds(m, 5) if run._cache_path(src, m, s).exists()]
        vals = []
        for seed in seeds:
            normal = run.load_scores(src, m, seed)[1]
            lines = batch._lines(normal, line_idx, run.FAR, True)
            vals.append(np.array([[float((normal[i] > lines[groups[i]]).sum()), len(normal[i])] for i in check]))
        mean = float(np.mean([v[:, 0].sum() / v[:, 1].sum() for v in vals]))
        fa, lo, hi = _boot(vals[0][:, 0], vals[0][:, 1])
        rows[det.REGISTRY[m].name] = {"false alarms on check wafers": mean, "low": max(lo + mean - fa, 0.0),
                                      "high": hi + mean - fa, "check wafers": len(check), "over 3%": bool(mean > CARRY)}
    out = pd.DataFrame(rows).T
    out.index.name = "detector"
    return out


def carry_table(name: str) -> pd.DataFrame:
    """Part 3 on one real dataset, round 4's lineup and line (cached)."""
    def compute():
        if name == "etch":
            return carry_over_etch(lineup(name))
        s = spec(name)
        return carry_over(s["ds"], lineup(name), s["calibration"], s["sources"])
    return structure.cached(f"round4_carry__{name}", compute)


def carry_campaign(setup: str, variant: str) -> pd.DataFrame:
    from . import campaign
    ds = campaign.dataset(setup)
    return structure.cached(f"round4_carry_campaign__{setup}__{variant}", lambda: carry_over(
        ds, campaign_methods(setup), "matched", keep=[r.group == variant for r in ds.runs], n_seeds=campaign.SEEDS))


# ---------------------------------------------------------------- part 5: the repeat share
#
# Fixed before computing. On normal data only. The normal fitting runs of a dataset (the runs its detectors are
# fitted on; for the water pump, every session's fitting readings), standardised on all of them together, every
# sensor. The dataset's noise level: the median distance between consecutive readings within one run. Each run is
# held out in turn: for each of its readings, the nearest reading from another run. The repeat share is the share
# of readings whose nearest reading in another run is closer than the noise level. At most 20,000 readings are
# held out (a random sample, seed 0) and at most 50,000 readings of the other runs searched (seed 0). Its range
# comes from resampling the held-out runs (1,000 draws).

QUERIES, POOL = 20_000, 50_000


def normal_runs(name: str) -> list:
    """The normal fitting runs of a dataset, as a list of arrays."""
    from . import campaign, data
    if name == "skab":
        return [r.fit for r in data.load("skab").runs]
    if name.startswith("campaign_"):
        ds = campaign.dataset(name.split("_", 1)[1])
    elif name.startswith("posctl_"):
        from . import posctl
        b, d, n = name.split("_")[1:]
        ds = posctl.dataset(float(b[1:]), int(d[1:]), int(n[1:]))
    else:
        ds = spec(name, balanced=False)["ds"]
    fit = ds.runs[0].fit
    edges = np.cumsum([0] + list(ds.fit_lengths))
    return [fit[a:b] for a, b in zip(edges[:-1], edges[1:])]


def repeat_share(runs: list, n_boot: int = N_BOOT) -> dict:
    """{"repeat share", "low", "high", "noise level", "runs", "readings"} for a list of normal runs."""
    from .detectors import NearestAll
    X = np.vstack(runs)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = [(r - mu) / sd for r in runs]
    noise = float(np.median(np.concatenate([np.sqrt((np.diff(z, axis=0) ** 2).sum(1)) for z in Z if len(z) > 1])))
    run_id = np.concatenate([np.full(len(z), i) for i, z in enumerate(Z)])
    allz = np.vstack(Z)
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(len(allz), min(QUERIES, len(allz)), replace=False))
    pool = np.sort(rng.choice(len(allz), min(POOL, len(allz)), replace=False))
    nn = NearestAll(k=1, max_fit=len(pool))
    nn.ref_, nn.ref_run_ = allz[pool], run_id[pool]
    nn.ref32_ = nn.ref_.astype(np.float32)
    nn.ref_sq_ = (nn.ref32_ ** 2).sum(1)
    d = nn._dist(allz[q], exclude_run=run_id[q])
    close = d < noise
    per_run = np.array([[close[run_id[q] == i].sum(), (run_id[q] == i).sum()] for i in range(len(Z))], float)
    point, lo, hi = _boot(per_run[:, 0], per_run[:, 1], n_boot)
    return {"repeat share": point, "low": lo, "high": hi, "noise level": noise, "runs": len(Z), "readings": len(q),
            "median distance to another run, in noise levels": float(np.median(d) / noise)}


REPEAT_DATASETS = ["skab", "tep_rieth", "cranfield", "tep6", "etch", "indpensim", "campaign_recipe", "campaign_mixed"]


def repeat_table() -> pd.DataFrame:
    def compute():
        rows = {folds.TITLES[n]: repeat_share(normal_runs(n)) for n in REPEAT_DATASETS}
        out = pd.DataFrame(rows).T
        out.index.name = "dataset"
        return out
    return structure.cached("round4_repeat_share", compute)


# ---------------------------------------------------------------- summaries for notebooks 09, 10 and 13

REAL = ["etch", "skab", "tep_rieth", "cranfield", "tep6", "indpensim"]


def _title(name: str) -> str:
    from . import deep
    return deep.TITLES[name]


def reports() -> dict:
    return {n: report(n) for n in REAL}


def overview() -> pd.DataFrame:
    """Notebook 09's overview (each deep model against its twin and the best simple detector), on round 4's reports."""
    from . import deep
    return pd.concat({_title(n): deep.table(report(n)) for n in REAL}, names=["dataset", "deep model"])


def classic_table() -> pd.DataFrame:
    """Part 2's detectors on every real dataset: catch rate (range); blank where a detector does not apply."""
    from . import detectors as det
    rows = {}
    for n in REAL:
        s = pd.DataFrame(report(n)["summary"]).set_index("method")
        row = {}
        for k in det.CLASSIC + [STEPPED]:
            m = det.REGISTRY[k].name
            if m in s.index:
                row[m] = f"{s.loc[m, 'dr_at_1pct']:.2f} ({s.loc[m, 'dr_ci90_low']:.2f}-{s.loc[m, 'dr_ci90_high']:.2f})"
        rows[_title(n)] = row
    out = pd.DataFrame(rows)
    out.index.name = "detector"
    return out.fillna("does not apply")


def _round3(name: str) -> dict:
    from . import deep
    return deep.report(name)


def best_simple_changes() -> pd.DataFrame:
    """Per real dataset: the best simple detector and the best deep model's gap to it, round 3 against round 4."""
    from . import deep
    rows = {}
    for n in REAL:
        row = {}
        for label, res in (("round 3", _round3(n)), ("round 4", report(n))):
            t = deep.table(res)
            best_deep = t["caught (mean of 3 seeds)"].astype(float).idxmax()
            row[f"best simple, {label}"] = f"{t['best simple detector'].iloc[0]} ({float(t['best simple caught'].iloc[0]):.2f})"
            row[f"best deep minus best simple, {label}"] = (f"{100 * float(t.loc[best_deep, 'gap to best simple']):+.1f} "
                                                           f"({_points(t.loc[best_deep, 'gap to best simple, range'])}) "
                                                           f"{t.loc[best_deep, 'against best simple']}")
        rows[_title(n)] = row
    out = pd.DataFrame(rows).T
    out.index.name = "dataset"
    return out


def campaign_questions() -> pd.DataFrame:
    """Every campaign variant: the best simple detector and the best deep model, round 3's lineup against round 4's."""
    from . import campaign, deep
    from . import detectors as det
    rows = {}
    deepn = campaign._names(campaign.METHODS_DEEP)
    for setup in ("recipe", "mixed"):
        for v in campaign.SCORED[setup]:
            row = {}
            for label, res in (("round 3", campaign.report(setup, v)), ("round 4", campaign_report(setup, v))):
                s = deep._summary(res)["dr_at_1pct"]
                simple = [m for m in s.index if m not in deepn]
                best, bd = s[simple].idxmax(), s[[m for m in deepn if m in s.index]].idxmax()
                g = deep.gap(res, bd, best)
                row[f"best simple, {label}"] = f"{best} ({s[best]:.2f})"
                row[f"best deep minus best simple, {label}"] = (f"{100 * g['gap']:+.1f} ({100 * g['low']:+.1f} to "
                                                               f"{100 * g['high']:+.1f}) {g['verdict']}")
            rows[(setup, campaign.VARIANT_TITLES[v])] = row
    out = pd.DataFrame(rows).T
    out.index.names = ["setup", "fault"]
    return out


def campaign_classic(setup: str = "recipe") -> pd.DataFrame:
    """Part 2's detectors on every variant of one campaign setup: catch rate (range)."""
    from . import campaign
    from . import detectors as det
    rows = {}
    for v in campaign.SCORED[setup]:
        s = deep_summary(campaign_report(setup, v))
        rows[campaign.VARIANT_TITLES[v]] = {det.REGISTRY[k].name: f"{s.loc[det.REGISTRY[k].name, 'dr_at_1pct']:.2f} "
                                            f"({s.loc[det.REGISTRY[k].name, 'dr_ci90_low']:.2f}-{s.loc[det.REGISTRY[k].name, 'dr_ci90_high']:.2f})"
                                            for k in classic(f"campaign_{setup}") if det.REGISTRY[k].name in s.index}
    out = pd.DataFrame(rows)
    out.index.name = "detector"
    return out


def deep_summary(res: dict) -> pd.DataFrame:
    return pd.DataFrame(res["summary"]).set_index("method")


def carry_summary() -> pd.DataFrame:
    """Part 3 in one table: per dataset, how many detectors raise over 3% false alarms before the fault, which, and
    for those, the catch rate at each run's own line beside the dataset's line."""
    rows = []
    for n in REAL:
        t = carry_table(n)
        col = "false alarms on check wafers" if n == "etch" else "false alarms before the fault"
        for m, r in t.iterrows():
            row = {"dataset": _title(n), "detector": m,
                   "false alarms before the fault": f"{float(r[col]):.3f} ({float(r['low']):.3f}-{float(r['high']):.3f})",
                   "over 3%": bool(r["over 3%"])}
            if "caught, dataset's line" in r and pd.notna(r.get("caught, dataset's line")):
                row["caught, dataset's line"] = float(r["caught, dataset's line"])
            if bool(r["over 3%"]) and "caught, each run's own line" in r and pd.notna(r.get("caught, each run's own line")):
                row["caught, each run's own line"] = (f"{float(r['caught, each run' + chr(39) + 's own line']):.2f} "
                                                      f"({float(r['own line low']):.2f}-{float(r['own line high']):.2f})")
            rows.append(row)
    return pd.DataFrame(rows).set_index(["dataset", "detector"])


def carry_counts() -> pd.DataFrame:
    t = carry_summary()
    out = t.groupby(level=0, sort=False)["over 3%"].agg(["sum", "size"])
    out.columns = ["detectors over 3%", "detectors"]
    out["which"] = [", ".join(t.loc[d].index[t.loc[d, "over 3%"].astype(bool)]) for d in out.index]
    return out


def campaign_carry_counts() -> pd.DataFrame:
    """Part 3 on the campaign: per setup and fault, the detectors over 3% false alarms before the fault, and for them
    the catch rate at each run's own line beside the line from the line batches."""
    from . import campaign
    rows = []
    for setup in ("recipe", "mixed"):
        for v in campaign.SCORED[setup]:
            t = carry_campaign(setup, v)
            over = t[t["over 3%"].astype(bool)]
            rows.append({"setup": setup, "fault": campaign.VARIANT_TITLES[v], "detectors over 3%": len(over),
                         "detectors": len(t),
                         "which (false alarms before; caught at the line -> at each batch's own line)": "; ".join(
                             f"{m} ({float(r['false alarms before the fault']):.2f}; {float(r['caught, dataset' + chr(39) + 's line']):.2f} -> "
                             f"{float(r['caught, each run' + chr(39) + 's own line']):.2f})" for m, r in over.iterrows())})
    return pd.DataFrame(rows).set_index(["setup", "fault"])
