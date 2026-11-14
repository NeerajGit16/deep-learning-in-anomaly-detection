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
FAR = 0.01


def _cache_path(ds: str, method: str, seed: int) -> Path:
    """Cache file per dataset, method, settings and seed: changing a setting gives a new file."""
    import hashlib
    tag = hashlib.md5(json.dumps(METHODS[method], sort_keys=True).encode()).hexdigest()[:6]
    return CACHE / f"{ds}__{method}__{tag}__s{seed}.npz"


def score_dataset(ds: D.Dataset, method: str, seed: int) -> Path:
    """Fit once per distinct fit array, score every run and held-out normal run, normalise."""
    path = _cache_path(ds.name, method, seed)
    if path.exists():
        return path
    params = dict(METHODS[method])
    if det.REGISTRY[method].stochastic:
        params["seed"] = seed
    t0 = time.time()
    fitted: dict[int, tuple] = {}

    def scored(run: D.Run) -> np.ndarray:
        key = id(run.fit)
        if key not in fitted:
            model = det.make(method, **params).fit(run.fit)
            fitted[key] = (model, model.score_fit(run.fit))
        model, fit_scores = fitted[key]
        return M.normalise(model.score(run.X), fit_scores)

    runs = [scored(r) for r in ds.runs]
    val = [scored(r) for r in ds.val_normal] if ds.val_normal else []
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
    """Score every run of every dataset with every method (cached, so reruns are instant)."""
    methods = [m for m in (methods or METHODS) if m in det.available()]
    for name in datasets:
        ds = D.load(name)
        for m in methods:
            for seed in _seeds(m, n_seeds):
                p = score_dataset(ds, m, seed)
                if verbose:
                    print(f"{name:5s} {m:15s} seed {seed}: {p.name}", flush=True)


def quick_score(ds: D.Dataset, make_detector, far: float = FAR) -> dict:
    """Try any detector on a whole dataset in one line, no caching.

    make_detector() must return a fresh, unfitted detector. Returns the catch rate at `far`
    false alarms (calibrated the same way as the pilot) and the scores per run.
    """
    scores = []
    for r in ds.runs:
        model = make_detector().fit(r.fit)
        scores.append(M.normalise(model.score(r.X), model.score_fit(r.fit)))
    out = M.dr_at_far(scores, [r.y for r in ds.runs], far, ds.calibration)
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
                print(f"{name:5s} {m:15s} seed {seed}: {p.name}", flush=True)


def _label(method: str) -> str:
    return det.REGISTRY[method].name


def report_dataset(ds: D.Dataset, methods: list[str], n_seeds: int, n_boot: int) -> dict:
    labels = [r.y for r in ds.runs]
    groups = [r.group for r in ds.runs]
    share = float(np.mean([y.mean() for y in labels]))
    mode = ds.calibration
    summary, by_group, primary = [], [], {}
    for m in methods:
        seeds = [s for s in _seeds(m, n_seeds) if _cache_path(ds.name, m, s).exists()]
        if not seeds:
            continue
        per_seed = []
        for seed in seeds:
            runs, val, secs = load_scores(ds.name, m, seed)
            p1 = M.dr_at_far(runs, labels, FAR, mode)
            p5 = M.dr_at_far(runs, labels, 0.05, mode)
            dep = M.deployable(runs, labels, FAR, val)
            per_seed.append({
                "dr_at_1pct": p1["dr"], "far_check": p1["far"], "dr_at_5pct": p5["dr"],
                "deploy_far": dep["far_realised"], "deploy_dr": dep["dr"],
                "deploy_events_flagged": dep["events_flagged"],
                "deploy_median_delay_s": dep["median_delay"] * ds.sample_seconds,
                "ap_mean": np.mean([M.average_precision(s, y) for s, y in zip(runs, labels)]),
                "vus_pr_mean": np.mean([M.vus_pr(s, y) for s, y in zip(runs, labels)]),
                "seconds": secs,
            })
            if seed == seeds[0]:
                primary[_label(m)] = runs
                for g, v in M.dr_by_group(runs, labels, groups, p1["threshold"]).items():
                    by_group.append({"method": _label(m), "group": g, "dr_at_1pct": v})
        frame = pd.DataFrame(per_seed)
        row = {"dataset": ds.name, "method": _label(m), "seeds": len(seeds)}
        row.update(frame.mean().to_dict())
        row["dr_at_1pct_seed_sd"] = float(frame["dr_at_1pct"].std(ddof=0))
        row["random_ap"] = share
        summary.append(row)

    draws = S.bootstrap_dr(primary, labels, FAR, n_boot, mode=mode)
    point = S.point_dr(primary, labels, FAR, mode)
    pairs = S.paired_differences(draws, point)
    for row in summary:
        lo, hi = np.nanquantile(draws[row["method"]], [0.05, 0.95])
        row["dr_ci90_low"], row["dr_ci90_high"] = float(lo), float(hi)
    sds = np.array([p["sd"] for p in pairs])
    noise = {"median_pair_sd": float(np.median(sds)), "max_pair_sd": float(np.max(sds)),
             "margin": float(max(0.05, 2 * np.median(sds)))}
    return {"summary": summary, "by_group": by_group, "pairs": pairs, "noise": noise,
            "runs": len(ds.runs), "anomaly_share": share, "calibration": mode}


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
        how = ("one threshold per run, averaged over runs" if res["calibration"] == "per_run"
               else "one threshold across all runs")
        lines += [f"## {name.upper()} ({res['runs']} runs, {res['anomaly_share']:.0%} of scored samples "
                  f"anomalous; {how})", "",
                  "| Method | DR @1% FA | 90% interval | DR @5% FA | Deployable FA | Deployable DR | Events flagged | Median delay (s) | VUS-PR | AP |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['method']} | {r['dr_at_1pct']:.3f} | {r['dr_ci90_low']:.2f}-{r['dr_ci90_high']:.2f} | "
                f"{r['dr_at_5pct']:.3f} | {r['deploy_far']:.3f} | {r['deploy_dr']:.3f} | "
                f"{r['deploy_events_flagged']:.2f} | {r['deploy_median_delay_s']:.0f} | "
                f"{r['vus_pr_mean']:.3f} | {r['ap_mean']:.3f} |")
        n = res["noise"]
        lines += ["", f"Random-score AP on this dataset is about {res['anomaly_share']:.2f}.",
                  f"Paired-gap noise: median bootstrap s.d. {n['median_pair_sd']:.3f}, "
                  f"largest {n['max_pair_sd']:.3f}. Margin = max(0.05, 2 x median) = **{n['margin']:.3f}**.", ""]
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
