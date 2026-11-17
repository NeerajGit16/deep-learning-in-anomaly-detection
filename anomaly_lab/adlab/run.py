"""Baseline pilot: score every run with every baseline, then report.

    python -m adlab.run score  --datasets skab tep     # cached; rerun to resume
    python -m adlab.run report --datasets skab tep     # tables, figures, noise and margin

Scores are cached in results/cache/, so `score` can be stopped and restarted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import detectors as det
from . import metrics as M
from . import progress
from . import stats as S

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"

METHODS = {
    "rolling_z": {"window": 5},
    "ewma": {"lam": 0.2},
    "cusum": {"k": 0.5},
    "pca": {"var": 0.90},
    "iforest": {"window": 10, "n_estimators": 100},
    "matrix_profile": {"m": 16},
    "link_straight": {"window": 5},
    "link_curved": {"window": 5},
}
# Detectors that only make sense for some data, so they are scored only when asked for by name
# and never by default: the stage-told straight-link residual needs a stage column (batch.py).
EXTRA_METHODS = {
    "link_straight_told": {"window": 5},
    "link_kinked": {"window": 5},
    # The deep models and the LSTM's straight twin (adlab/deep.py): their settings are chosen
    # inside fit, on held-aside normal data, so nothing is set here.
    "deep_ae": {}, "deep_lstm": {}, "deep_gdn": {}, "deep_gdn_paper": {}, "deep_usad": {},
    "straight_forecaster": {},
    "pca_twin": {},          # added after the first deep-model results: PCA scored like the autoencoder
    # round 3: schedule-aware simple checks (detectors.py); they read the dataset's controller outputs
    "unseen_value": {}, "nearest_1": {}, "nearest_5": {}, "recipe_hour": {},
    # round 3, part 4: the straight-link detector (and told the mode) with its error scale on balanced folds
    "link_straight_balanced": {"window": 5}, "link_straight_told_balanced": {"window": 5},
}
FAR = 0.01
# A gap between two methods smaller than this is not called a win. It is a fixed number, so it
# does not move when detectors are added. It was chosen from the first pilot with six detectors,
# where twice the typical chance wobble of a gap came to 0.116 (SKAB) and 0.122 (Tennessee
# Eastman). Reports still print the measured wobble, for information only.
MARGIN = 0.12


def settings(method: str) -> dict:
    """A detector's settings: from METHODS, or from EXTRA_METHODS for the ones scored only on request."""
    return METHODS[method] if method in METHODS else EXTRA_METHODS[method]


def _cache_path(ds: str, method: str, seed: int) -> Path:
    """Cache file per dataset, method, settings and seed: changing a setting gives a new file."""
    import hashlib
    tag = hashlib.md5(json.dumps(settings(method), sort_keys=True).encode()).hexdigest()[:6]
    return CACHE / f"{ds}__{method}__{tag}__s{seed}.npz"


def score_dataset(ds: D.Dataset, method: str, seed: int, verbose: bool = True) -> Path:
    """Fit once per distinct fit array, score every run and held-out normal run, normalise."""
    path = _cache_path(ds.name, method, seed)
    if path.exists():
        return path
    params = dict(settings(method))
    if det.REGISTRY[method].stochastic:
        params["seed"] = seed
    t0 = time.time()
    fitted: dict[int, tuple] = {}

    def scored(run: D.Run) -> np.ndarray:
        key = id(run.fit)
        if key not in fitted:
            model = det.make(method, **params)
            if hasattr(model, "context_"):        # models that hold data aside whole runs at a time, and log
                model.context_ = {"dataset": ds.name, "run": run.name, "lengths": ds.fit_lengths,
                                  "outputs": ds.output_columns, "fit_groups": ds.fit_groups,
                                  "settings": ds.deep_settings}
            model.fit(run.fit)
            fitted[key] = (model, model.score_fit(run.fit))
        model, fit_scores = fitted[key]
        return M.normalise(model.score(run.X), fit_scores)

    todo = list(ds.runs) + list(ds.val_normal or [])
    what = f"scoring {ds.name} / {method}" + (f" / seed {seed}" if det.REGISTRY[method].stochastic else "")
    done = [scored(r) for r in progress.track(todo, what, verbose=verbose)]
    runs, val = done[:len(ds.runs)], done[len(ds.runs):]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scores=np.concatenate(runs), lengths=np.array([len(s) for s in runs]),
        val=np.concatenate(val) if val else np.array([]), val_lengths=np.array([len(s) for s in val]),
        seconds=time.time() - t0,
    )
    return path


def load_scores(ds_name: str, method: str, seed: int):
    z = np.load(_cache_path(ds_name, method, seed))
    runs = np.split(z["scores"], np.cumsum(z["lengths"])[:-1])
    val = np.split(z["val"], np.cumsum(z["val_lengths"])[:-1]) if len(z["val_lengths"]) else None
    return runs, val, float(z["seconds"])


def _seeds(method: str, n_seeds: int) -> list[int]:
    return list(range(n_seeds)) if det.REGISTRY[method].stochastic else [0]


def score_all(datasets=("skab", "tep"), methods=None, n_seeds: int = 5, verbose: bool = True):
    """Score every run of every dataset with every method (cached, so reruns are instant).

    verbose shows progress bars: one over the (dataset, method, seed) jobs still to do, one over
    the runs of the job in hand. Jobs already in the cache are skipped silently.
    """
    methods = [m for m in (methods or METHODS) if m in det.available()]
    datasets = [D.load(ds) if isinstance(ds, str) else ds for ds in datasets]   # names, or Datasets already loaded
    jobs = [(ds, m, seed) for ds in datasets for m in methods for seed in _seeds(m, n_seeds)
            if not _cache_path(ds.name, m, seed).exists()]
    if not jobs:          # everything is already in the cache: nothing to show
        return
    for ds, m, seed in progress.track(jobs, "scoring jobs", verbose=verbose, leave=True):
        score_dataset(ds, m, seed, verbose=verbose)


def calibration_scores(ds: D.Dataset, method: str, seed: int = 0) -> list | None:
    """Cached scores of a dataset's fault-free calibration runs, cut to the stretch of a run the fault covers.

    These set the alarm line under matched-window calibration (see data.Dataset). None if the
    dataset has no such runs.
    """
    if ds.matched_start is None:
        return None
    if ds.matched_from is not None:
        source = load_scores(ds.matched_from.name, method, seed)[0]
    else:
        source = load_scores(ds.name, method, seed)[1]
    return [s[ds.matched_start:] for s in source] if source else None


def quick_score(ds: D.Dataset, make_detector, far: float = FAR) -> dict:
    """Try any detector on a whole dataset in one line, no caching.

    make_detector() must return a fresh, unfitted detector. Returns the catch rate at `far`
    false alarms (calibrated the same way as the pilot) and the scores per run.
    """
    fitted: dict[int, tuple] = {}

    def scored(r: D.Run) -> np.ndarray:
        if id(r.fit) not in fitted:
            model = make_detector().fit(r.fit)
            fitted[id(r.fit)] = (model, model.score_fit(r.fit))
        model, reference = fitted[id(r.fit)]
        return M.normalise(model.score(r.X), reference)

    scores = [scored(r) for r in ds.runs]
    calib = ([scored(r)[ds.matched_start:] for r in ds.calibration_runs()]
             if ds.calibration == "matched" else None)
    out = M.dr_at_far(scores, [r.y for r in ds.runs], far, ds.calibration, calib)
    return {"caught": out["dr"], "false_alarms": out["far"], "scores": scores}


def clear_cache(dataset: str | None = None, method: str | None = None):
    """Delete cached scores, e.g. after changing a method's settings in METHODS."""
    for p in CACHE.glob(f"{dataset or '*'}__{method or '*'}__*.npz"):
        p.unlink()


def cmd_score(args):
    methods = [m for m in (args.methods or METHODS) if m in det.available()]
    for name in args.datasets:
        ds = D.load(name)
        for m in methods:
            for seed in _seeds(m, args.seeds):
                p = score_dataset(ds, m, seed)
                print(f"{name:5s} {m:15s} seed {seed}: {p.name}", flush=True)     # the bar shows the runs; this names the file


def _label(method: str) -> str:
    return det.REGISTRY[method].name


def report_dataset(ds: D.Dataset, methods: list[str], n_seeds: int, n_boot: int,
                   resample: str | list = "runs", keep: list | None = None,
                   calibration: str | None = None, verbose: bool = True, sources: dict | None = None,
                   line_keep: list | None = None) -> dict:
    """Every method's catch rate on a dataset, from its cached scores, with intervals and paired gaps.

    resample     what the intervals redraw: "runs" (single runs), "groups" (whole kinds of fault,
                 every run of a drawn kind together), or a list with one label per run (runs that
                 share a label are drawn together).
    keep         optional True/False per run, to report on only part of the dataset.
    calibration  how the alarm line is set, if not the dataset's own way (see data.Dataset).
                 E.g. "pooled" on a Tennessee Eastman set gives the older first-8-hours line.
    sources      {method: Dataset} for methods scored on another copy of the same runs (for
                 example with the mode added as a last column, for a detector told the mode).
                 Their scores and calibration scores are read from that copy.
    line_keep    optional True/False per calibration run (matched line): set the line on part of them.
    """
    sources = sources or {}
    keep = [True] * len(ds.runs) if keep is None else list(keep)
    kept = [r for r, k in zip(ds.runs, keep) if k]
    labels = [r.y for r in kept]
    groups = [r.group for r in kept]
    clusters = {"runs": None, "groups": groups}[resample] if isinstance(resample, str) \
        else [c for c, k in zip(resample, keep) if k]
    share = float(np.mean([y.mean() for y in labels]))
    mode = calibration or ds.calibration
    summary, by_group, primary, calibs = [], [], {}, {}
    with_vus = M.vus_available()     # the VUS-PR column is left out when TSB-AD is not installed
    for m in progress.track(methods, f"reading scores of {ds.name}", verbose=verbose):
        src = sources.get(m, ds)
        seeds = [s for s in _seeds(m, n_seeds) if _cache_path(src.name, m, s).exists()]
        if not seeds:
            continue
        per_seed = []
        for seed in seeds:
            runs, val, secs = load_scores(src.name, m, seed)
            runs = [s for s, k in zip(runs, keep) if k]
            cal = calibration_scores(src, m, seed)
            if line_keep is not None:
                cal = [c for c, k in zip(cal, line_keep) if k] if cal else cal
                val = [v for v, k in zip(val, line_keep) if k] if val else val
            p1 = M.dr_at_far(runs, labels, FAR, mode, cal)
            p5 = M.dr_at_far(runs, labels, 0.05, mode, cal)
            dep = M.deployable(runs, labels, FAR, val)
            per_seed.append({
                "dr_at_1pct": p1["dr"], "far_check": p1["far"], "dr_at_5pct": p5["dr"],
                "deploy_far": dep["far_realised"], "deploy_dr": dep["dr"],
                "deploy_events_flagged": dep["events_flagged"],
                "deploy_median_delay_s": dep["median_delay"] * ds.sample_seconds,
                "ap_mean": np.mean([M.average_precision(s, y) for s, y in zip(runs, labels)]),
                "seconds": secs,
            })
            if with_vus:
                per_seed[-1]["vus_pr_mean"] = np.mean([M.vus_pr(s, y) for s, y in zip(runs, labels)])
            if cal and mode != "per_run":
                # false alarms this line gives on fault-free runs, over the fault's stretch of a run
                per_seed[-1]["far_late_fault_free"] = float(np.mean(np.concatenate(cal) > p1["threshold"]))
            if mode == "matched":
                # false alarms this line gives on the scored runs' own normal readings (0.01 if it carries over)
                per_seed[-1]["far_own_normal"] = p1["far_first_window"]
                # for comparison: the catch rate under the older rule (line set on the runs' own first part)
                per_seed[-1]["dr_old_rule"] = M.dr_at_far(runs, labels, FAR, "pooled")["dr"]
            if seed == seeds[0]:
                primary[_label(m)], calibs[_label(m)] = runs, cal
                for g, v in M.dr_by_group(runs, labels, groups, p1["threshold"]).items():
                    by_group.append({"method": _label(m), "group": g, "dr_at_1pct": v})
        frame = pd.DataFrame(per_seed)
        row = {"dataset": ds.name, "method": _label(m), "seeds": len(seeds)}
        row.update(frame.mean().to_dict())
        row["dr_at_1pct_seed_sd"] = float(frame["dr_at_1pct"].std(ddof=0))
        row["random_ap"] = share
        summary.append(row)

    calibs = calibs if mode == "matched" else None
    draws = S.bootstrap_dr(primary, labels, FAR, n_boot, mode=mode, clusters=clusters, calib=calibs,
                           verbose=verbose, what=f"resampling {ds.name}")
    point = S.point_dr(primary, labels, FAR, mode, calibs)
    pairs = S.paired_differences(draws, point)
    for row in summary:
        lo, hi = np.nanquantile(draws[row["method"]], [0.05, 0.95])
        row["dr_ci90_low"], row["dr_ci90_high"] = float(lo), float(hi)
    sds = np.array([p["sd"] for p in pairs]) if pairs else np.array([np.nan])     # one method: no pairs
    # The wobble is measured and reported; the margin is the fixed MARGIN, not derived from it.
    noise = {"median_pair_sd": float(np.median(sds)), "max_pair_sd": float(np.max(sds)), "margin": MARGIN}
    return {"summary": summary, "by_group": by_group, "pairs": pairs, "noise": noise,
            "runs": len(kept), "anomaly_share": share, "calibration": mode,
            "calibration_runs": len(ds.calibration_runs()) if mode == "matched" else None}


def rates(res: dict, digits: int = 2) -> pd.Series:
    """Each method's catch rate with its 90% range as text, e.g. "0.81 (0.68-0.92)", best first."""
    df = pd.DataFrame(res["summary"]).sort_values("dr_at_1pct", ascending=False)
    text = [f"{r['dr_at_1pct']:.{digits}f} ({r['dr_ci90_low']:.{digits}f}-{r['dr_ci90_high']:.{digits}f})"
            for _, r in df.iterrows()]
    return pd.Series(text, index=df["method"].rename("detector"), name="catch rate")


def report_old_and_new(ds: D.Dataset, methods: list[str] | None = None, n_seeds: int = 5, n_boot: int = 1000,
                       **kw) -> dict:
    """report_dataset twice: under the dataset's own alarm line ("new") and the older first-part line ("old").

    For datasets with matched-window calibration (Tennessee Eastman). The old line is "pooled":
    set on the scored runs' own normal part, which comes before the fault.
    """
    methods = methods or [k for k in METHODS if k in det.available()]
    progress.note(f"report on {ds.name}, new line then old line, {n_boot} resamples each")
    return {"new": report_dataset(ds, methods, n_seeds, n_boot, **kw),
            "old": report_dataset(ds, methods, n_seeds, n_boot, calibration="pooled", **kw)}


def old_and_new_table(pair: dict, digits: int = 2) -> pd.DataFrame:
    """Per detector: catch rate under the old and the new alarm line (with ranges), and what changing the line did.

    `pair` comes from report_old_and_new. "false alarms later, old line" is the share of fault-free
    readings from the fault's stretch of a run that cross the old line (about 0.01 if the line is fair).
    """
    new = pd.DataFrame(pair["new"]["summary"]).set_index("method")
    old = pd.DataFrame(pair["old"]["summary"]).set_index("method")
    out = pd.DataFrame({"caught, old line": rates(pair["old"], digits), "caught, new line": rates(pair["new"], digits)})
    out["change"] = (new["dr_at_1pct"] - old["dr_at_1pct"]).round(digits) + 0.0     # + 0.0: no "-0"
    out["false alarms later, old line"] = old["far_late_fault_free"].round(3)
    out = out.loc[new.sort_values("dr_at_1pct", ascending=False).index]
    out.index.name = "detector"
    return out


def report_two_lines(ds: D.Dataset, other: str = "per_run", methods: list[str] | None = None, n_seeds: int = 5,
                     n_boot: int = 1000, **kw) -> dict:
    """report_dataset under the dataset's own alarm line ("main") and under another way of setting it ("other")."""
    methods = methods or [k for k in METHODS if k in det.available()]
    progress.note(f"report on {ds.name}: its own alarm line, then '{other}', {n_boot} resamples each")
    return {"main": report_dataset(ds, methods, n_seeds, n_boot, **kw),
            "other": report_dataset(ds, methods, n_seeds, n_boot, calibration=other, **kw)}


def two_lines_table(pair: dict, digits: int = 2) -> pd.DataFrame:
    """Per detector: catch rate (with range) under both alarm lines, from report_two_lines.

    "false alarms on the runs' own normal readings" is the share of the scored runs' normal
    readings that cross the main line. It is about 0.01 if a line set on separate normal data
    carries over to the scored runs; far more means their normal readings look different to
    the detector, and the catch rate under that line is then not a fair "1 false alarm in 100".
    """
    main = pd.DataFrame(pair["main"]["summary"]).set_index("method")
    out = pd.DataFrame({"caught, line from separate normal data": rates(pair["main"], digits)})
    if "far_own_normal" in main:
        out["false alarms on the runs' own normal readings"] = main["far_own_normal"].round(3)
    out["caught, each run's own line"] = rates(pair["other"], digits)
    out = out.loc[main.sort_values("dr_at_1pct", ascending=False).index]
    out.index.name = "detector"
    return out


def false_alarms_around_fault(ds: D.Dataset, methods: list[str] | None = None, far: float = FAR,
                              sources: dict | None = None, keep: list | None = None,
                              line_keep: list | None = None) -> pd.DataFrame:
    """Where a matched alarm line's false alarms fall in the scored runs: before the fault, or after it has started.

    For datasets whose runs have normal readings both before a fault and after it is removed.
    "before" is every reading ahead of a run's first fault reading; "after" is the normal
    readings that come later (once the fault has been removed, or between two faults). If the
    line carries over from the separate normal data, both are about `far`. A high "before" says
    the runs' normal readings look different to the detector from the day it was fitted on; a
    high "after" with a low "before" says the system has not settled back.
    sources and keep as in report_dataset.
    """
    sources = sources or {}
    keep = [True] * len(ds.runs) if keep is None else list(keep)
    kept = [r for r, k in zip(ds.runs, keep) if k]
    rows = {}
    for m in methods or [k for k in METHODS if k in det.available()]:
        src = sources.get(m, ds)
        scores = [s for s, k in zip(load_scores(src.name, m, 0)[0], keep) if k]
        cal = calibration_scores(src, m, 0)
        cal = cal if line_keep is None else [c for c, k in zip(cal, line_keep) if k]
        thr = M.dr_at_far(scores, [r.y for r in kept], far, "matched", cal)["threshold"]
        before, after = [], []
        for s, r in zip(scores, kept):
            first = int(np.flatnonzero(r.y)[0])
            before.append(s[:first] > thr)
            after.append(s[first:][r.y[first:] == 0] > thr)
        rows[_label(m)] = {"false alarms before the fault": float(np.concatenate(before).mean()),
                           "false alarms after the fault": float(np.concatenate(after).mean()),
                           "runs with over 0.05 before the fault": int(sum(b.mean() > 0.05 for b in before)),
                           "runs": len(scores)}
    out = pd.DataFrame(rows).T.astype({"runs with over 0.05 before the fault": int, "runs": int})
    out.index.name = "detector"
    return out


def fault_table(res: dict, digits: int = 2) -> pd.DataFrame:
    """Catch rate of every detector (columns, best first) on every kind of fault (rows), from report_dataset."""
    order = pd.DataFrame(res["summary"]).sort_values("dr_at_1pct", ascending=False)["method"].tolist()
    out = pd.DataFrame(res["by_group"]).pivot(index="group", columns="method", values="dr_at_1pct")[order].round(digits)
    out.index.name, out.columns.name = "kind of fault", None
    return out


def late_false_alarms(ds: D.Dataset, methods: list[str] | None = None, far: float = FAR,
                      n_boot: int = 1000, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """What the OLD alarm line lets through later in a run, when nothing is wrong.

    The old line ("pooled" calibration) is set on the scored runs' own normal part, which comes
    before the fault. This is the share of fault-free readings from the fault's stretch of a
    run (the calibration runs) that cross that line. A detector whose score does not rise
    during a run on its own gives about `far`.

    Ranges redraw the scored runs (by run number, where runs have one) and the fault-free runs.
    """
    labels = [np.asarray(r.y, bool) for r in ds.runs]
    clusters = [r.meta["run"] for r in ds.runs] if all("run" in r.meta for r in ds.runs) else None
    rows = {}
    for m in (methods or [k for k in METHODS if k in det.available()]):
        scores = load_scores(ds.name, m, 0)[0]
        normals = [s[~y] for s, y in zip(scores, labels)]
        late = S.AlarmCounts(calibration_scores(ds, m, 0))
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(late.sizes), (n_boot, len(late.sizes)))

        def rate(idx, cal):
            thr = np.quantile(np.concatenate([normals[i] for i in idx]), 1 - far)
            return late.above(thr)[cal].sum() / late.sizes[cal].sum()

        point = rate(np.arange(len(scores)), np.arange(len(late.sizes)))
        pairs = list(zip(S.resample_indices(len(scores), n_boot, rng, clusters), picks))
        draws = np.array([rate(idx, cal) for idx, cal in
                          progress.track(pairs, f"resampling late false alarms / {m}", verbose=verbose)])
        lo, hi = S.interval(draws)
        rows[_label(m)] = {"false alarms later, old line": point, "low": lo, "high": hi}
    out = pd.DataFrame(rows).T
    out.index.name = "detector"
    return out


def _twin_draws(ds: D.Dataset, twins: D.Dataset, method: str, far: float, n_boot: int, seed: int,
                verbose: bool = True) -> dict:
    """Alarm rates of every fault run and its twin at the matched alarm line, point values and resamples."""
    faulty = S.AlarmCounts([s[np.asarray(r.y, bool)] for s, r in zip(load_scores(ds.name, method, 0)[0], ds.runs)])
    twin = S.AlarmCounts([s[np.asarray(r.y, bool)] for s, r in zip(load_scores(twins.name, method, 0)[0], twins.runs)])
    numbers = [r.meta["run"] for r in twins.runs]
    faults = list(dict.fromkeys(r.group for r in ds.runs))
    where = {(r.group, r.meta["run"]): i for i, r in enumerate(ds.runs)}
    grid = np.array([[where[(f, n)] for n in numbers] for f in faults])       # (faults, run numbers)
    rng = np.random.default_rng(seed)
    line, lines = S.line_draws(calibration_scores(ds, method, 0), far, n_boot, rng)
    number_picks = rng.integers(0, len(numbers), (n_boot, len(numbers)))
    fault_picks = rng.integers(0, len(faults), (n_boot, len(faults)))

    def at(thr, cols):
        in_fault = faulty.rates(thr)[grid][:, cols]          # (faults, drawn run numbers)
        in_twin = twin.rates(thr)[cols]
        return in_fault.mean(1), in_twin.mean(), (in_fault - in_twin).mean(1)

    point = at(line, np.arange(len(numbers)))
    draws = [at(thr, cols) for thr, cols in
             progress.track(list(zip(lines, number_picks)), f"resampling twins / {method}", verbose=verbose)]
    return {"faults": faults, "point": point, "fault_picks": fault_picks,
            "caught": np.array([d[0] for d in draws]), "twin": np.array([d[1] for d in draws]),
            "excess": np.array([d[2] for d in draws])}


def excess_over_twin(ds: D.Dataset, twins: D.Dataset, method: str, far: float = FAR,
                     n_boot: int = 1000, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """One detector, fault by fault: its alarms in the fault runs minus its alarms in their twins.

    `twins` holds the same runs with no fault (matched by meta["run"]). For each fault run, the
    share of its fault hours above the alarm line, minus the share of the same hours above the
    line in its twin, averaged over the runs of that fault. This is what the fault itself adds:
    alarms the detector would have raised anyway do not count.

    Ranges redraw the run numbers (a drawn number brings its run of every fault and its twin)
    and the fault-free runs that set the alarm line.
    """
    d = _twin_draws(ds, twins, method, far, n_boot, seed, verbose)
    caught, twin, excess = d["point"]
    rows = {}
    for j, f in enumerate(d["faults"]):
        lo, hi = S.interval(d["excess"][:, j])
        rows[f] = {"alarms in fault runs": caught[j], "alarms in twins": twin,
                   "excess alarms over the twin": excess[j], "low": lo, "high": hi}
    out = pd.DataFrame(rows).T
    out.index.name = "kind of fault"
    return out


def excess_summary(ds: D.Dataset, twins: D.Dataset, methods: list[str] | None = None, far: float = FAR,
                   n_boot: int = 1000, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Per detector, over all faults: excess alarms over the twin, and the alarms in the twins themselves.

    The twins are fault-free, so a well-set alarm line gives about `far` alarms in them.
    Ranges redraw the faults, the run numbers and the fault-free runs that set the alarm line.
    """
    rows = {}
    for m in (methods or [k for k in METHODS if k in det.available()]):
        d = _twin_draws(ds, twins, m, far, n_boot, seed, verbose)
        caught, twin, excess = d["point"]
        overall = np.array([e[p].mean() for e, p in zip(d["excess"], d["fault_picks"])])
        lo, hi = S.interval(overall)
        t_lo, t_hi = S.interval(d["twin"])
        rows[_label(m)] = {"alarms in fault runs": caught.mean(), "alarms in twins": twin, "twins, low": t_lo,
                           "twins, high": t_hi, "excess alarms over the twin": excess.mean(), "low": lo, "high": hi}
    out = pd.DataFrame(rows).T
    out.index.name = "detector"
    return out


def per_run_rates(ds: D.Dataset, method: str, group: str | None = None, far: float = FAR) -> pd.Series:
    """Catch rate of every single run (optionally of one kind of fault), at the dataset's alarm line."""
    scores = load_scores(ds.name, method, 0)[0]
    labels = [r.y for r in ds.runs]
    thr = M.dr_at_far(scores, labels, far, ds.calibration, calibration_scores(ds, method, 0))["threshold"]
    thr = thr if np.ndim(thr) else [thr] * len(scores)
    out = {r.name: float(np.mean(s[np.asarray(y, bool)] > t))
           for r, s, y, t in zip(ds.runs, scores, labels, thr) if group is None or r.group == group}
    return pd.Series(out, name="catch rate")


def by_fault(res: dict, method: str) -> pd.Series:
    """One detector's catch rate for every kind of fault, from a report_dataset() result (display name)."""
    df = pd.DataFrame(res["by_group"])
    return df[df["method"] == method].set_index("group")["dr_at_1pct"].rename(method)


def gap_table(res: dict, method: str, others: list[str] | None = None) -> pd.DataFrame:
    """Gaps in catch rate between `method` and other methods, from a report_dataset() result.

    Names are the detectors' display names, e.g. "Straight-link residual". A positive gap means
    `method` catches more. The range is the 90% interval of the gap when runs are resampled,
    using the same resampled runs for both methods. `method` is called 'ahead' or 'behind' only
    if the gap is at least MARGIN and its range does not include zero; otherwise it is a 'tie'.
    """
    rows = {}
    for p in res["pairs"]:
        if method not in (p["method_a"], p["method_b"]):
            continue
        flip = p["method_b"] == method
        other = p["method_a"] if flip else p["method_b"]
        gap, lo, hi = ((-p["gap"], -p["ci90_high"], -p["ci90_low"]) if flip
                       else (p["gap"], p["ci90_low"], p["ci90_high"]))
        clear = abs(gap) >= MARGIN and (lo > 0 or hi < 0)
        rows[other] = {"gap": gap, "low": lo, "high": hi,
                       "verdict": ("ahead" if gap > 0 else "behind") if clear else "tie"}
    out = pd.DataFrame(rows).T.loc[others if others is not None else list(rows)]
    out[["gap", "low", "high"]] = out[["gap", "low", "high"]].astype(float)
    out.index.name = f"{method} against"
    return out


def verdict_table(res: dict) -> pd.DataFrame:
    """Every pair of detectors at once: is the row detector 'ahead' of, 'behind' or in a 'tie' with the column one?

    The same rule as gap_table: ahead or behind needs a gap of at least MARGIN whose range does
    not include zero. Detectors are listed best first.
    """
    order = pd.DataFrame(res["summary"]).sort_values("dr_at_1pct", ascending=False)["method"].tolist()
    out = pd.DataFrame("", index=order, columns=order)
    for m in order:
        gaps = gap_table(res, m)
        for other in gaps.index:
            out.loc[m, other] = gaps.loc[other, "verdict"]
    out.index.name = "row against column"
    return out


def _figures(results: dict, out: Path):
    import matplotlib.pyplot as plt

    from . import plots
    for name, fig in (("pilot_overall.png", plots.pilot_overall(results, FAR)),
                      ("pilot_by_type.png", plots.pilot_by_type(results))):
        fig.savefig(out / name, dpi=160)
        plt.close(fig)


def _markdown(results: dict) -> str:
    lines = ["# Baseline pilot", "",
             "Primary metric: share of anomalous samples flagged when each method's threshold "
             "gives 1% false alarms on normal samples (random scoring gets 0.01). "
             "Intervals resample runs 1,000 times.", ""]
    for name, res in results.items():
        df = pd.DataFrame(res["summary"]).sort_values("dr_at_1pct", ascending=False)
        how = {"per_run": "one threshold per run, averaged over runs",
               "pooled": "one threshold across all runs",
               "matched": f"one threshold, set on {res.get('calibration_runs')} separate fault-free run(s) over the "
                          "same hours of a run as the fault",
               }[res["calibration"]]
        with_vus, with_old = "vus_pr_mean" in df, "dr_old_rule" in df
        head = (["Method", "DR @1% FA", "90% interval"] + (["Old rule (first 8 hours)"] if with_old else [])
                + ["DR @5% FA", "Deployable FA", "Deployable DR", "Events flagged", "Median delay (s)"]
                + (["VUS-PR"] if with_vus else []) + ["AP"])
        lines += [f"## {name.upper()} ({res['runs']} runs, {res['anomaly_share']:.0%} of scored samples "
                  f"anomalous; {how})", "",
                  "| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
        for _, r in df.iterrows():
            cells = [r["method"], f"{r['dr_at_1pct']:.3f}", f"{r['dr_ci90_low']:.2f}-{r['dr_ci90_high']:.2f}"]
            cells += [f"{r['dr_old_rule']:.3f}"] if with_old else []
            cells += [f"{r['dr_at_5pct']:.3f}", f"{r['deploy_far']:.3f}", f"{r['deploy_dr']:.3f}",
                      f"{r['deploy_events_flagged']:.2f}", f"{r['deploy_median_delay_s']:.0f}"]
            cells += ([f"{r['vus_pr_mean']:.3f}"] if with_vus else []) + [f"{r['ap_mean']:.3f}"]
            lines.append("| " + " | ".join(cells) + " |")
        n = res["noise"]
        lines += ["", f"Random-score AP on this dataset is about {res['anomaly_share']:.2f}.",
                  f"A gap between two methods under **{n['margin']:.2f}** is not called a win (fixed margin). "
                  f"For information, the measured chance wobble of a gap between two methods here: "
                  f"typical {n['median_pair_sd']:.3f}, largest {n['max_pair_sd']:.3f} (bootstrap s.d.).", ""]
    return "\n".join(lines)


def cmd_report(args):
    methods = [m for m in (args.methods or METHODS) if m in det.available()]
    RESULTS.mkdir(exist_ok=True)
    results = {}
    for name in args.datasets:
        res = report_dataset(D.load(name), methods, args.seeds, args.boot)
        results[name] = res
        pd.DataFrame(res["summary"]).to_csv(RESULTS / f"{name}_summary.csv", index=False)
        pd.DataFrame(res["by_group"]).to_csv(RESULTS / f"{name}_by_type.csv", index=False)
        pd.DataFrame(res["pairs"]).to_csv(RESULTS / f"{name}_pairs.csv", index=False)
    (RESULTS / "noise_and_margin.json").write_text(
        json.dumps({n: r["noise"] for n, r in results.items()}, indent=2))
    (RESULTS / "pilot_report.md").write_text(_markdown(results), encoding="utf-8")
    _figures(results, RESULTS)
    print((RESULTS / "pilot_report.md").read_text(encoding="utf-8"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["score", "report"])
    p.add_argument("--datasets", nargs="+", default=["skab", "tep"], choices=list(D.LOADERS))
    p.add_argument("--methods", nargs="+", choices=list(METHODS))
    p.add_argument("--seeds", type=int, default=5, help="seeds for stochastic methods")
    p.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    args = p.parse_args(argv)
    {"score": cmd_score, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
