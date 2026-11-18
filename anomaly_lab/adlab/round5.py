"""Round 5: two checks on the conclusions, and a corrected positive control (notebooks 09, 10 and 14).

Part 1, the metal etcher's unseen-value win (fixed before computing).
    Each etcher fault changed one setting, and the data's authors then edited that setting's column back, so that
    the fault would show only through the other sensors. An unseen-value check could be catching traces of that
    edit rather than the machine's response.
    The six settings and their columns: TCP power -> "TCP Top Pwr"; RF power -> "RF Btm Pwr" and "RF Pwr" (the
    file has two radio-frequency power readings and does not say which follows the setting, so both count);
    pressure -> "Pressure"; Cl2 -> "Cl2 Flow"; BCl3 -> "BCl3 Flow"; He chuck -> "He Press".
    For each faulty wafer: which sensors carry its unseen values, and whether its own setting's column is one.
    Then the unseen-value check and the other top detectors are fitted and scored again
        (a) with each wafer's own setting column(s) left out (each wafer is scored by detectors fitted without
            them; its alarm line comes from the line wafers scored the same way),
        (b) with all six settings' columns (seven columns) left out for every wafer,
    and reported beside the full-column result. The top detectors: round 4's five best simple detectors on the
    etcher (the unseen value check on all sensors, the straight-link residual told the recipe step, the curved-link
    residual, PCA scored like the autoencoder, the CUSUM chart) and the best deep model (the autoencoder, notebook
    09's design). Everything else as in notebook 07: one alarm line per experiment from its line wafers, ranges
    from resampling faulty wafers and line wafers together.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import structure

# ---------------------------------------------------------------- part 1: the metal etcher

SETTING_COLUMNS = {"TCP": ["TCP Top Pwr"], "RF": ["RF Btm Pwr", "RF Pwr"], "Pr": ["Pressure"], "Cl2": ["Cl2 Flow"],
                   "BCl3": ["BCl3 Flow"], "He": ["He Press"]}
ALL_SETTINGS = [c for cols in SETTING_COLUMNS.values() for c in cols]
TOP = ["unseen_all", "link_straight_told", "link_curved", "pca_twin", "cusum", "deep_ae"]
VARIANTS = {"all columns": "the 19 sensors, as in round 4",
            "own setting left out": "each wafer without its own setting's column(s)",
            "all settings left out": "every wafer without the six settings' seven columns"}


def setting_of(fault: str) -> str:
    return fault.split()[0]


def etch_sets(told: bool = False) -> dict:
    """{key: Dataset}: the etcher with one setting's columns left out (key = setting), or all of them ("settings")."""
    from . import etch
    out = {k: etch.dataset(told=told, drop=tuple(cols), tag=k) for k, cols in SETTING_COLUMNS.items()}
    out["settings"] = etch.dataset(told=told, drop=tuple(ALL_SETTINGS), tag="settings")
    return out


def score_etch(verbose: bool = True):
    """Score the top detectors on every reduced copy of the etcher (cached; small)."""
    from . import deep, progress, run
    progress.note("round 5 part 1: scoring the etcher without its setting columns")
    plain, told = etch_sets(), etch_sets(told=True)
    for key in plain:
        run.score_all([plain[key]], methods=[m for m in TOP if m not in ("link_straight_told", "deep_ae")], n_seeds=1,
                      verbose=verbose)
        run.score_all([plain[key]], methods=["deep_ae"], n_seeds=deep.SEEDS, verbose=verbose)
        run.score_all([told[key]], methods=["link_straight_told"], n_seeds=1, verbose=verbose)
    progress.note("round 5 part 1: etcher scored")


def _scores(name: str, method: str, told_name: str | None = None) -> list:
    """[(fault scores, normal scores)] per seed, from the cache."""
    from . import run
    src = told_name if method == "link_straight_told" else name
    seeds = [s for s in run._seeds(method, 5) if run._cache_path(src, method, s).exists()]
    return [run.load_scores(src, method, s)[:2] for s in seeds]


def unseen_by_wafer() -> pd.DataFrame:
    """Per faulty wafer: the columns where its readings take values never seen in the fitting wafers (share of the
    wafer's readings, each column alone), whether its own setting's column is one, and how many of its readings are
    flagged only through its own setting's column(s)."""
    from . import etch
    from .detectors import UnseenAll
    ds = etch.dataset()
    fit = ds.runs[0].fit
    m = UnseenAll().fit(fit)
    cols, seen, scale, tol = m.learned_
    rows = []
    for r in ds.runs:
        X = r.X
        unseen = {}
        for j, v, t in zip(cols, seen, tol):
            k = np.clip(np.searchsorted(v, X[:, j]), 1, len(v) - 1)
            gap = np.minimum(np.abs(X[:, j] - v[k - 1]), np.abs(X[:, j] - v[k]))
            unseen[etch.CHANNELS[j]] = gap > t
        own = SETTING_COLUMNS[setting_of(r.meta["fault"])]
        any_unseen = np.any(list(unseen.values()), axis=0)
        own_unseen = np.any([unseen[c] for c in own if c in unseen], axis=0) if any(c in unseen for c in own) \
            else np.zeros(len(X), bool)
        other_unseen = np.any([u for c, u in unseen.items() if c not in own], axis=0)
        rows.append({"wafer": r.name, "fault": r.meta["fault"], "own setting's column(s)": ", ".join(own),
                     "readings": len(X), "readings with an unseen value": float(any_unseen.mean()),
                     "own column unseen": float(own_unseen.mean()),
                     "flagged only through its own column": float((own_unseen & ~other_unseen).mean()),
                     "other columns unseen": float(other_unseen.mean()),
                     "columns with unseen values (share of readings)": ", ".join(
                         f"{c} {u.mean():.2f}" for c, u in sorted(unseen.items(), key=lambda kv: -kv[1].mean()) if u.any())})
    return pd.DataFrame(rows).set_index("wafer")


def _report_own(per_key: dict, fault_keys: list, fault_kinds: list, fault_groups: list, normal_groups: list,
                normal_roles: list, far: float = 0.01, n_boot: int = 1000, seed: int = 0) -> dict:
    """batch.report when each faulty wafer is scored by its own copy of the detectors (variant a).

    per_key       {detector name: [ {key: (fault scores, normal scores)} per seed ]}
    fault_keys    per faulty wafer: the key of the copy that scores it
    The line for a wafer comes from the line wafers scored by the same copy. Ranges resample the faulty wafers and,
    in the same draw, each experiment's line wafers (the same wafers for every copy)."""
    from .batch import _above, _lines
    from .run import MARGIN
    from .stats import paired_differences
    groups = list(dict.fromkeys(normal_groups))
    line_idx = {g: [i for i, (gg, r) in enumerate(zip(normal_groups, normal_roles)) if gg == g and r == "line"]
                for g in groups}
    check_idx = [i for i, r in enumerate(normal_roles) if r == "check"]
    rng = np.random.default_rng(seed)
    n = len(fault_keys)
    fault_picks = rng.integers(0, n, (n_boot, n))
    line_picks = [{g: rng.integers(0, len(idx), len(idx)) for g, idx in line_idx.items()} for _ in range(n_boot)]
    summary, by_group, draws, point = [], [], {}, {}

    def above_all(by_key, picks=None, which=None):
        out = np.zeros(n)
        for key in set(fault_keys):
            faults, normal = by_key[key]
            lines = _lines(normal, line_idx, far, True, picks)
            for i in range(n):
                if fault_keys[i] == key:
                    out[i] = _above([np.sort(np.asarray(faults[i], float))], [fault_groups[i]], lines)[0]
        return out

    for name, seeds in per_key.items():
        per_seed = []
        for k, by_key in enumerate(seeds):
            sizes = np.array([len(by_key[fault_keys[i]][0][i]) for i in range(n)], float)
            above = above_all(by_key)
            checks = []
            for key in set(fault_keys):
                normal = by_key[key][1]
                lines = _lines(normal, line_idx, far, True)
                weight = sum(1 for fk in fault_keys if fk == key) / n
                checks.append(weight * np.mean(np.concatenate([normal[i] > lines[normal_groups[i]] for i in check_idx])))
            per_seed.append({"dr_at_1pct": above.sum() / sizes.sum(), "far_check": float(np.sum(checks))})
            if k == 0:
                point[name] = per_seed[-1]["dr_at_1pct"]
                kinds = pd.Series(above).groupby(np.asarray(fault_kinds), sort=False).sum() / \
                    pd.Series(sizes).groupby(np.asarray(fault_kinds), sort=False).sum()
                by_group += [{"method": name, "group": g, "dr_at_1pct": float(v)} for g, v in kinds.items()]
                # resampling: precompute each wafer's count above every drawn line
                d = np.empty(n_boot)
                for b in range(n_boot):
                    a = above_all(by_key, line_picks[b])
                    idx = fault_picks[b]
                    d[b] = a[idx].sum() / sizes[idx].sum()
                draws[name] = d
        frame = pd.DataFrame(per_seed)
        row = {"method": name, "seeds": len(seeds), **frame.mean().to_dict(),
               "dr_at_1pct_seed_sd": float(frame["dr_at_1pct"].std(ddof=0))}
        row["dr_ci90_low"], row["dr_ci90_high"] = map(float, np.nanquantile(draws[name], [0.05, 0.95]))
        summary.append(row)
    pairs = paired_differences(draws, point)
    sds = np.array([p["sd"] for p in pairs]) if pairs else np.array([np.nan])
    return {"summary": summary, "by_group": by_group, "pairs": pairs, "draws": draws,
            "noise": {"median_pair_sd": float(np.median(sds)), "max_pair_sd": float(np.max(sds)), "margin": MARGIN},
            "runs": n, "calibration": "one line per group"}


def etch_reports(n_boot: int = 1000) -> dict:
    """{variant: report} for the top detectors on the etcher (cached)."""
    def compute():
        from . import batch, etch, run
        from . import detectors as det
        full = etch.dataset()
        plain, told = etch_sets(), etch_sets(told=True)
        kinds = [r.group for r in full.runs]
        groups = [r.meta["experiment"] for r in full.runs]
        ngroups = [r.meta["experiment"] for r in full.val_normal]
        roles = [r.meta["role"] for r in full.val_normal]
        names = {m: det.REGISTRY[m].name for m in TOP}
        out = {}
        out["all columns"] = batch.report({names[m]: _scores(etch.CACHE, m, etch.TOLD_NAME) for m in TOP},
                                          kinds, groups, ngroups, roles, n_boot=n_boot, verbose=False)
        out["all settings left out"] = batch.report(
            {names[m]: _scores(plain["settings"].name, m, told["settings"].name) for m in TOP},
            kinds, groups, ngroups, roles, n_boot=n_boot, verbose=False)
        keys = [setting_of(r.meta["fault"]) for r in full.runs]
        per_key = {}
        for m in TOP:
            by_seed = {k: _scores(plain[k].name, m, told[k].name) for k in SETTING_COLUMNS}
            n_seeds = min(len(v) for v in by_seed.values())
            per_key[names[m]] = [{k: by_seed[k][s] for k in SETTING_COLUMNS} for s in range(n_seeds)]
        out["own setting left out"] = _report_own(per_key, keys, kinds, groups, ngroups, roles, n_boot=n_boot)
        return out
    return structure.cached("round5_etch_reports", compute)


def etch_table() -> pd.DataFrame:
    """Per top detector: catch rate (range) and false alarms on the check wafers, under the three variants."""
    reps = etch_reports()
    cols = {}
    for v, res in reps.items():
        s = pd.DataFrame(res["summary"]).set_index("method")
        cols[f"caught, {v}"] = [f"{s.loc[m, 'dr_at_1pct']:.2f} ({s.loc[m, 'dr_ci90_low']:.2f}-{s.loc[m, 'dr_ci90_high']:.2f})"
                                for m in s.index]
        cols[f"false alarms on check wafers, {v}"] = s["far_check"].round(3).tolist()
        index = s.index
    out = pd.DataFrame(cols, index=index)
    out.index.name = "detector"
    return out


def etch_gaps() -> pd.DataFrame:
    """The unseen value check minus each other top detector, under each variant (points, range, verdict)."""
    from . import deep
    rows = []
    for v, res in etch_reports().items():
        for m in [x for x in pd.DataFrame(res["summary"])["method"] if x != "Unseen value check, all sensors"]:
            g = deep.gap(res, "Unseen value check, all sensors", m)
            rows.append({"variant": v, "against": m, "gap, points": 100 * g["gap"],
                         "range": f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}", "verdict": g["verdict"]})
    return pd.DataFrame(rows).set_index(["variant", "against"])


TOP_LONG = ["deep_ae_v2", "deep_pcac_v2"]
# Added in part 2, after seeing its results (not in part 1's design): with the long training, the PCA-plus-correction
# autoencoder is the etcher's best detector after the unseen value check (0.56 against the straight-link residual's
# 0.43). The same three variants are run for the two long-trained autoencoders, beside part 1's five simple top
# detectors, so that their lead can be checked without the edited columns too. Detectors outside part 1's top five
# are not rerun.


def score_etch_long(verbose: bool = True):
    """Score the long-trained autoencoders on every reduced copy of the etcher (cached)."""
    from . import deep, progress, run
    progress.note("round 5 part 2: long-trained autoencoders on the etcher without its setting columns")
    plain = etch_sets()
    for key in plain:
        run.score_all([plain[key]], methods=TOP_LONG, n_seeds=deep.SEEDS, verbose=verbose)
    progress.note("round 5 part 2: etcher without its setting columns scored")


def etch_reports_long(n_boot: int = 1000) -> dict:
    """{variant: report}: part 1's simple top detectors and the long-trained autoencoders on the etcher (cached)."""
    def compute():
        from . import batch, deep, deepdive, etch  # noqa: F401
        from . import detectors as det
        full = etch.dataset()
        plain, told = etch_sets(), etch_sets(told=True)
        kinds = [r.group for r in full.runs]
        groups = [r.meta["experiment"] for r in full.runs]
        ngroups = [r.meta["experiment"] for r in full.val_normal]
        roles = [r.meta["role"] for r in full.val_normal]
        methods = [m for m in TOP if m != "deep_ae"] + TOP_LONG
        names = {m: det.REGISTRY[m].name for m in methods}
        out = {}
        out["all columns"] = batch.report({names[m]: _scores(etch.CACHE, m, etch.TOLD_NAME) for m in methods},
                                          kinds, groups, ngroups, roles, n_boot=n_boot, verbose=False)
        out["all settings left out"] = batch.report(
            {names[m]: _scores(plain["settings"].name, m, told["settings"].name) for m in methods},
            kinds, groups, ngroups, roles, n_boot=n_boot, verbose=False)
        keys = [setting_of(r.meta["fault"]) for r in full.runs]
        per_key = {}
        for m in methods:
            by_seed = {k: _scores(plain[k].name, m, told[k].name) for k in SETTING_COLUMNS}
            n_seeds = min(len(v) for v in by_seed.values())
            per_key[names[m]] = [{k: by_seed[k][s] for k in SETTING_COLUMNS} for s in range(n_seeds)]
        out["own setting left out"] = _report_own(per_key, keys, kinds, groups, ngroups, roles, n_boot=n_boot)
        return out
    return structure.cached("round5_etch_reports_long", compute)


def etch_long_table() -> pd.DataFrame:
    """Per detector: catch rate (range) under the three variants, with the long-trained autoencoders; and the best
    long-trained autoencoder against the best of the simple top detectors (points, range, verdict)."""
    from . import deep
    from . import detectors as det
    reps = etch_reports_long()
    long_names = {det.REGISTRY[m].name for m in TOP_LONG}
    cols, gaps = {}, {}
    for v, res in reps.items():
        s = pd.DataFrame(res["summary"]).set_index("method")
        cols[v] = {m: f"{s.loc[m, 'dr_at_1pct']:.2f} ({s.loc[m, 'dr_ci90_low']:.2f}-{s.loc[m, 'dr_ci90_high']:.2f})"
                   for m in s.index}
        r = s["dr_at_1pct"]
        bd = r[[m for m in r.index if m in long_names]].idxmax()
        simple = r[[m for m in r.index if m not in long_names]]
        for label, pool in (("best simple", simple), ("best simple apart from the unseen value check",
                                                       simple.drop("Unseen value check, all sensors"))):
            bs = pool.idxmax()
            g = deep.gap(res, bd, bs)
            gaps[(v, label)] = {"best long-trained autoencoder": f"{bd} ({r[bd]:.2f})", "against": f"{bs} ({r[bs]:.2f})",
                                "gap, points": f"{100 * g['gap']:+.1f} ({100 * g['low']:+.1f} to {100 * g['high']:+.1f})",
                                "verdict": g["verdict"]}
    rates = pd.DataFrame(cols)
    rates.index.name = "detector"
    gaps = pd.DataFrame(gaps).T
    gaps.index.names = ["variant", "against"]
    return {"rates": rates, "gaps": gaps}


def etch_by_fault() -> pd.DataFrame:
    """The unseen value check's catch rate per kind of fault, under each variant."""
    out = {}
    for v, res in etch_reports().items():
        bg = pd.DataFrame(res["by_group"])
        out[v] = bg[bg["method"] == "Unseen value check, all sensors"].set_index("group")["dr_at_1pct"]
    df = pd.DataFrame(out)
    df.index.name = "kind of fault"
    return df


# ---------------------------------------------------------------- part 2: are the deep models under-trained?
#
# Fixed before computing.
#   The audit: every deep-model training in the log (notebook 09's budget: at most 100 passes, stopping after 10
#   passes without improvement on held-aside normal data). A training that used all 100 passes was CUT SHORT: its
#   held-aside error had improved within its last 10 passes, so it was still falling when it stopped.
#   The rerun ("long training", deep.BUDGETS["v2"]): up to 1,000 passes, stopping after 30 passes without
#   improvement, the learning rate halved after 10. Every deep model again: the autoencoder, the LSTM (choosing its
#   setting again where notebook 09 chose it; the campaign keeps IndPenSim's fixed setting), the autoencoder on the
#   controller outputs (campaign), and the new PCA-plus-correction autoencoder (deep.PCACorrection). GDN, GDN with
#   the paper's score and USAD face notebook 09's self-checks again under the long budget; any that passes is run
#   too. Both autoencoders also face the curvature check. Datasets as in round 4 (balanced folds where the fold
#   audit flagged them). Every deep result that moves by more than 2 points is listed; "best deep minus best simple"
#   and notebook 10 are recomputed with the long-training models in place of notebook 09's.

V1_V2 = {"deep_ae": "deep_ae_v2", "deep_lstm": "deep_lstm_v2", "deep_gdn": "deep_gdn_v2",
         "deep_gdn_paper": "deep_gdn_paper_v2", "deep_usad": "deep_usad_v2", "deep_ae_outputs": "deep_ae_outputs_v2"}
REAL = ["etch", "skab", "tep_rieth", "cranfield", "tep6", "indpensim"]


def training_audit(budget: str = "v1", positive_control: bool = True) -> pd.DataFrame:
    """Per dataset and deep model: trainings, passes used, and how many were cut short (used every pass the budget
    allows: notebook 09's 100, or the long budget's 1,000 with budget="v2")."""
    import json
    from .deep import BUDGETS, LOG
    cap = BUDGETS[budget]["max"]
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("budget", "v1") == budget and r.get("dataset") not in (None, "simulated")
            and "_without_" not in r["dataset"]           # part 1's etcher copies are not part of the audit
            and (positive_control or not r["dataset"].startswith("posctl"))]
    cut = f"cut short (all {cap:,} passes)"
    out = {}
    for r in rows:
        ds = r["dataset"]
        family = ("positive control" if ds.startswith("posctl") else "generator (notebook 12)" if ds.startswith("mechanism")
                  else re.sub(r"_balanced(_blocks|_aside)?$", " (balanced folds)", ds))
        key = (family, r["model"].replace(" (v2)", ""))
        o = out.setdefault(key, {"trainings": 0, cut: 0, "passes": []})
        o["trainings"] += 1
        o[cut] += int(r["passes"] >= cap)
        o["passes"].append(r["passes"])
    table = pd.DataFrame({k: {"trainings": v["trainings"], cut: v[cut], "share cut short": v[cut] / v["trainings"],
                              "passes, median": float(np.median(v["passes"])), "passes, fewest": int(min(v["passes"])),
                              "passes, most": int(max(v["passes"]))}
                          for k, v in out.items()}).T
    table.index.names = ["dataset", "model"]
    order = list(dict.fromkeys(k[0] for k in out))          # each dataset's rows together, in order of first training
    return table.loc[order]


def _curvature(keys) -> pd.DataFrame:
    """Notebook 09's curvature check for any autoencoder: its reconstruction error on held-out simulated normal data
    against PCA's with the same number of components (below 0.8 x PCA's at curve 1, above at curve 0)."""
    from . import deep
    from .detectors import PCA
    rows = {}
    for key in keys:
        for curve in (0.0, 1.0):
            fit, test = deep._simulated(curve)
            m = deep._fit_with_context(key, fit)
            err = float(np.mean(m.errors(test) ** 2))
            pca = PCA(var=0.90).fit(fit)
            Zp = pca._z(test)
            pca_err = float(np.mean((Zp - Zp @ pca.P_ @ pca.P_.T) ** 2))
            ratio = err / pca_err
            ok = ratio < deep.CHECK_RATIO if curve == 1.0 else ratio > deep.CHECK_RATIO
            rows[(key, f"curve {curve:g}")] = {"model error": err, "PCA error": pca_err, "model / PCA": ratio,
                                               "expected": "below 0.8" if curve == 1.0 else "above 0.8", "passed": bool(ok)}
    out = pd.DataFrame(rows).T
    out.index.names = ["model", "simulated links"]
    return out


def self_checks_v2() -> dict:
    """Notebook 09's two self-checks under the long budget: {"curvature", "link break", "keep", "dropped"}."""
    from . import deep
    from . import detectors as det
    curv = structure.cached("deep_check_curvature_v2", lambda: _curvature(["deep_ae_v2", "deep_pcac_v2"]))
    brk = structure.cached("deep_check_link_break_v2", lambda: deep.link_break_check(
        deep.DEEP_V2 + ["straight_forecaster", "pca", "link_straight"]))
    dropped = {}
    for key in deep.DEEP_V2:
        name = det.REGISTRY[key].name
        if name in brk.index and not brk.loc[name, "passed"]:
            dropped[key] = f"link-break check: caught {brk.loc[name, 'caught']:.2f}, below {deep.CHECK_CAUGHT:.2f}"
    for key in ("deep_ae_v2", "deep_pcac_v2"):
        part = curv.loc[key]
        if not part["passed"].all():
            dropped[key] = "curvature check: " + "; ".join(f"{i}: {r['model / PCA']:.2f}, expected {r['expected']}"
                                                           for i, r in part.iterrows() if not r["passed"])
    return {"curvature": curv, "link break": brk, "keep": [k for k in deep.DEEP_V2 if k not in dropped], "dropped": dropped}


def long_models(name: str) -> list:
    """The long-training deep models to score on a dataset (those that passed the self-checks)."""
    keep = self_checks_v2()["keep"]
    if name.startswith("campaign_"):
        return [k for k in ("deep_ae_v2", "deep_ae_outputs_v2", "deep_lstm_v2", "deep_pcac_v2") if k in keep or k == "deep_ae_outputs_v2"]
    return keep


def _target(name: str):
    from . import campaign, round4
    if name.startswith("campaign_"):
        return campaign.dataset(name.split("_", 1)[1])
    return round4.spec(name)["ds"]


def score_long(name: str, verbose: bool = True):
    """Score the long-training deep models on one dataset (cached)."""
    from . import deep, deepdive, progress, run  # noqa: F401  (deepdive registers the outputs-only autoencoder)
    progress.note(f"round 5 part 2: long training on {name}")
    ds = _target(name)
    targets = [ds] + ([ds.matched_from] if ds.matched_from is not None else [])   # the alarm line's own runs too
    run.score_all(targets, methods=long_models(name), n_seeds=deep.SEEDS, verbose=verbose)
    progress.note(f"round 5 part 2: long training on {name} done")


def _lineup_long(name: str) -> list:
    """Round 4's lineup with notebook 09's deep models replaced by the long-training ones."""
    from . import deep, round4
    simple = [m for m in round4.lineup(name) if m not in deep.MODELS]
    return simple + long_models(name)


def report_long(name: str, n_boot: int = 1000) -> dict:
    """Round 4's report on a real dataset with the long-training deep models (cached)."""
    from . import batch, etch, round4, run
    from . import detectors as det
    s = round4.spec(name)
    methods = _lineup_long(name)

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
                                per_group=True, n_boot=n_boot, verbose=False)
        return run.report_dataset(s["ds"], methods, 5, n_boot, calibration=s["calibration"], verbose=False,
                                  sources=s["sources"])

    return structure.cached(f"round5_report__{name}", compute)


def campaign_report_long(setup: str, variant: str, n_boot: int = 1000) -> dict:
    from . import campaign, deep, deepdive, round4, run  # noqa: F401
    ds = campaign.dataset(setup)
    methods = [m for m in round4.campaign_methods(setup) if m not in deep.MODELS and m != "deep_ae_outputs"] + \
        long_models(f"campaign_{setup}")
    return structure.cached(f"round5_campaign__{setup}__{variant}", lambda: run.report_dataset(
        ds, methods, campaign.SEEDS, n_boot, keep=[r.group == variant for r in ds.runs], calibration="matched",
        verbose=False))


def _summary(res):
    return pd.DataFrame(res["summary"]).set_index("method")


def deep_moves() -> pd.DataFrame:
    """Every deep model's catch rate, notebook 09's training against the long one, on every dataset and campaign fault;
    and the PCA-plus-correction autoencoder beside them."""
    from . import campaign, deepdive, round4  # noqa: F401  (deepdive registers the outputs-only autoencoders)
    from . import detectors as det
    rows = []
    pairs = [(det.REGISTRY[a].name, det.REGISTRY[b].name) for a, b in V1_V2.items()]
    cases = [(round4._title(n), round4.report(n), report_long(n)) for n in REAL]
    cases += [(f"Campaign, {s}: {campaign.VARIANT_TITLES[v]}", round4.campaign_report(s, v), campaign_report_long(s, v))
              for s in ("recipe", "mixed") for v in campaign.SCORED[s]]
    for title, old, new in cases:
        a, b = _summary(old), _summary(new)
        for v1, v2 in pairs:
            if v1 in a.index and v2 in b.index:
                d = float(b.loc[v2, "dr_at_1pct"] - a.loc[v1, "dr_at_1pct"])
                rows.append({"dataset": title, "deep model": v1,
                             "notebook 09's training": f"{a.loc[v1, 'dr_at_1pct']:.3f} ({a.loc[v1, 'dr_ci90_low']:.2f}-{a.loc[v1, 'dr_ci90_high']:.2f})",
                             "long training": f"{b.loc[v2, 'dr_at_1pct']:.3f} ({b.loc[v2, 'dr_ci90_low']:.2f}-{b.loc[v2, 'dr_ci90_high']:.2f})",
                             "change, points": 100 * d, "moved": abs(d) > 0.02})
        pc = "PCA + correction autoencoder"
        if pc in b.index:
            rows.append({"dataset": title, "deep model": pc + " (new)", "notebook 09's training": "",
                         "long training": f"{b.loc[pc, 'dr_at_1pct']:.3f} ({b.loc[pc, 'dr_ci90_low']:.2f}-{b.loc[pc, 'dr_ci90_high']:.2f})",
                         "change, points": np.nan, "moved": False})
    return pd.DataFrame(rows).set_index(["dataset", "deep model"])


def best_deep_vs_simple() -> pd.DataFrame:
    """Per dataset and campaign fault: the best deep model against the best simple detector, round 4 against round 5."""
    from . import campaign, deep, deepdive, round4  # noqa: F401
    from . import detectors as det
    deep_names = {det.REGISTRY[k].name for k in list(deep.MODELS) + list(deep.MODELS_V2) + ["deep_ae_outputs", "deep_ae_outputs_v2"]
                  if k in det.REGISTRY}
    rows = {}
    cases = [(round4._title(n), round4.report(n), report_long(n)) for n in REAL]
    cases += [(f"Campaign, {s}: {campaign.VARIANT_TITLES[v]}", round4.campaign_report(s, v), campaign_report_long(s, v))
              for s in ("recipe", "mixed") for v in campaign.SCORED[s]]
    for title, old, new in cases:
        row = {}
        for label, res in (("round 4", old), ("long training", new)):
            s = _summary(res)["dr_at_1pct"]
            dn = [m for m in s.index if m in deep_names]
            sn = [m for m in s.index if m not in deep_names]
            bd, bs = s[dn].idxmax(), s[sn].idxmax()
            g = deep.gap(res, bd, bs)
            row[f"best deep, {label}"] = f"{bd} ({s[bd]:.2f})"
            row[f"best simple, {label}"] = f"{bs} ({s[bs]:.2f})"
            row[f"gap, {label}"] = f"{100 * g['gap']:+.1f} ({100 * g['low']:+.1f} to {100 * g['high']:+.1f}) {g['verdict']}"
        rows[title] = row
    out = pd.DataFrame(rows).T
    out.index.name = "dataset"
    return out
