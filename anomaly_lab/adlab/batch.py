"""Batch processes: a batch (a wafer counts as one) is the unit, and the stage of a batch plays
the role of the operating mode. Shared by etch.py (metal etch) and, later, pensim.py.

Curvature
    Whole batches are held out in five rounds (staggered_rounds). The measures are those of
    notebooks 05 and 06: the pooled same-moment curved share (the headline), the same with the
    stage told to the straight model, the measure inside each stage, lags as support, and the
    share of each sensor's movement that lies between stages (between_table).

Detectors
    Fitted on normal batches only. Alarm lines are set on other normal batches ("line" batches),
    either one line per group (e.g. per experiment) or one line for all. A third set of normal
    batches ("check" batches) is scored too, to see whether a line carries over to normal
    batches it has not seen: about 1 false alarm in 100 if it does. No batch a detector was
    fitted on is ever scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import detectors as det
from . import progress, structure
from .stats import paired_differences

ROUNDS = 5
SAME_MOMENT = "same moment (the headline)"
TOLD = "same moment, stage told to the straight model"
LAG_SETS = {"+ 1 back": (1,), "+ 1, 2 back": (1, 2), "+ 1, 2, 5 back": (1, 2, 5), "+ 1, 2, 5, 10 back": (1, 2, 5, 10)}
FAMILIES = {"level checks": ("rolling_z", "ewma", "cusum"), "straight links": ("pca", "link_straight"),
            "curved links": ("link_curved",), "shape in time": ("matrix_profile",)}


# ---------------------------------------------------------------- which batches are fitted on, which held out

def staggered_rounds(groups, rounds: int = ROUNDS) -> np.ndarray:
    """The round (0 to rounds - 1) in which each batch is held out.

    groups   one label per batch (e.g. the experiment it was run in), batches in time order.

    Batches are dealt to the rounds like cards, group by group in the order given, the count
    carrying on from one group to the next: 0, 1, 2, 3, 4, 0, 1, ... So every round holds out
    about a fifth of every group, spread over the group's whole time, and fits on the rest of
    every group. Carrying the count on keeps the rounds the same size.
    """
    groups = np.asarray(groups)
    out = np.empty(len(groups), int)
    dealt = 0
    for g in dict.fromkeys(groups.tolist()):
        idx = np.flatnonzero(groups == g)
        out[idx] = (dealt + np.arange(len(idx))) % rounds
        dealt += len(idx)
    return out


def round_table(rounds, groups, group_label: str = "group") -> pd.DataFrame:
    """Round by round: how many batches of each group are held out and how many are fitted on."""
    rounds, groups = np.asarray(rounds), np.asarray(groups)
    rows = {}
    for k in sorted(set(rounds.tolist())):
        row = {}
        for g in dict.fromkeys(groups.tolist()):
            row[f"{group_label} {g}: held out"] = int(((rounds == k) & (groups == g)).sum())
            row[f"{group_label} {g}: fitted on"] = int(((rounds != k) & (groups == g)).sum())
        rows[f"round {k + 1}"] = row
    return pd.DataFrame(rows).T


def as_folds(runs: list, rounds) -> list:
    """One array per batch, every reading numbered with the batch's round (for curvature_by_run)."""
    return [np.full(len(r), int(k)) for r, k in zip(runs, rounds)]


# ---------------------------------------------------------------- curvature

def outlier_check(runs: list, channels: list, names: list, limit: float = 0.5) -> tuple[pd.DataFrame, list]:
    """structure.movement_by_run by batch, largest first, and the batches that hold more than
    `limit` of any sensor's movement (the outlier-run rule)."""
    table = structure.movement_by_run(runs, channels, names).sort_values("largest share", ascending=False)
    flagged = list(dict.fromkeys(table.loc[table["largest share"] > limit, "run"]))
    return table, flagged


def measures(runs: list, stages: list, rounds, channels: list, cache: str, lags: dict | None = None,
             n_boot: int = 500, verbose: bool = True) -> dict:
    """The pooled measures: {name: summary}. Each is saved in results/cache/<cache>__... when done."""
    folds = as_folds(runs, rounds)

    def by(what, **kw):
        return lambda: structure.curvature_by_run(runs, channels, folds=folds, n_boot=n_boot, verbose=verbose,
                                                  what=f"{cache}, {what}", **kw)

    out = {SAME_MOMENT: structure.cached(f"{cache}__same_moment", by("same moment")),
           TOLD: structure.cached(f"{cache}__stage_told", by("stage told", modes=stages))}
    for name, lag in (LAG_SETS if lags is None else lags).items():
        out[name] = structure.cached(f"{cache}__lags_" + "_".join(map(str, lag)), by(name, lags=lag))
    return out


def stage_parts(runs: list, stages: list, rounds, stage) -> tuple[list, np.ndarray]:
    """Each batch's readings from one stage, as separate runs, with their batches' rounds.
    Batches with fewer than two readings in the stage are left out."""
    parts, kept = [], []
    for X, s, k in zip(runs, stages, rounds):
        here = np.asarray(s) == stage
        if here.sum() >= 2:
            parts.append(np.asarray(X)[here])
            kept.append(int(k))
    return parts, np.array(kept)


def within_stages(runs: list, stages: list, rounds, channels: list, cache: str, label=None,
                  n_boot: int = 500, verbose: bool = True) -> dict:
    """The same-moment measure inside each stage on its own: {stage name: summary}."""
    label = label or (lambda s: f"stage {s}")
    out = {}
    for s in sorted(set(np.concatenate(stages).tolist())):
        parts, kept = stage_parts(runs, stages, rounds, s)
        out[label(s)] = structure.cached(f"{cache}__inside_stage_{s}", lambda: structure.curvature_by_run(
            parts, channels, folds=as_folds(parts, kept), n_boot=n_boot, verbose=verbose,
            what=f"{cache}, inside {label(s)}"))
    return out


def measure_table(by_name: dict) -> pd.DataFrame:
    """One row per measure: straight share (0 to 1) and curved share in points (100 x share), with ranges."""
    rows = {name: {"straight share": s["straight"], "straight, low": s["straight_low"], "straight, high": s["straight_high"],
                   "curved share, points": 100 * s["curved"], "curved low, points": 100 * s["curved_low"],
                   "curved high, points": 100 * s["curved_high"],
                   "sensors gaining over 5 points": s["sensors_with_curved_over_5pts"],
                   "sensors gaining over 10 points": s["sensors_with_curved_over_10pts"],
                   "held-out batches": s["runs"], "held-out readings": s["readings"]}
            for name, s in by_name.items()}
    out = pd.DataFrame(rows).T.astype({"sensors gaining over 5 points": int, "sensors gaining over 10 points": int,
                                       "held-out batches": int, "held-out readings": int})
    out.index.name = "measure"
    return out


def sensor_table(summary: dict, info: pd.Series | None = None) -> pd.DataFrame:
    """Every sensor's straight and curved share (with the range of its curved share), most curved first.
    Shares are 0 to 1 here, as in the earlier notebooks."""
    t = summary["table"]
    out = t if info is None else pd.concat([info.reindex(t.index).rename("what it measures"), t], axis=1)
    return out.sort_values("curved", ascending=False)


def most_curved(summary: dict, over: float = 0.10) -> list:
    t = summary["table"]["curved"].sort_values(ascending=False)
    return list(t.index[t > over])


def between_table(runs: list, stages: list, channels: list, groups: list | None = None, group_label: str = "groups",
                  info: pd.Series | None = None) -> pd.DataFrame:
    """Per sensor: the share of its movement that is the gap between stages (and, if `groups` gives
    one label per batch, between those groups too). See structure.between_share."""
    X, s = np.vstack(runs), np.concatenate(stages)
    out = pd.DataFrame({"share between stages": structure.between_share(X, s)}, index=channels)
    if groups is not None:
        g = np.concatenate([np.full(len(r), str(gg)) for r, gg in zip(runs, groups)])
        out[f"share between {group_label}"] = structure.between_share(X, g)
    if info is not None:
        out.insert(0, "what it measures", info.reindex(out.index))
    out.index.name = "sensor"
    return out.sort_values("share between stages")


def best_partners(X: np.ndarray, sensors: list, channels: list, info: pd.Series | None = None) -> pd.DataFrame:
    """For each sensor: the one other sensor that on its own says most about it (structure.best_single_predictor)."""
    rows = {}
    for s in sensors:
        i, share = structure.best_single_predictor(X, channels.index(s))
        rows[s] = {"best single predictor": channels[i], "share explained": share}
        if info is not None:
            rows[s] = {"what it measures": info[s], **rows[s], "which measures": info[channels[i]]}
    out = pd.DataFrame(rows).T
    out.index.name = "sensor"
    return out


def shape_clues(X: np.ndarray, stage: np.ndarray, sensors: list, channels: list) -> pd.DataFrame:
    """Numbers that help put a name to the shape of a curved link.

    distinct values      few (2 or 3) suggests a reading that flips between states
    share at commonest   high suggests a reading held at one value most of the time (a limit)
    share at its lowest / highest   high suggests a floor or a ceiling
    share between stages high suggests separate clusters, one per stage
    """
    rows = {}
    for s in sensors:
        x = X[:, channels.index(s)]
        values, counts = np.unique(x, return_counts=True)
        rows[s] = {"distinct values": len(values), "share at commonest value": counts.max() / len(x),
                   "share at its lowest": float((x == x.min()).mean()), "share at its highest": float((x == x.max()).mean()),
                   "share between stages": float(structure.between_share(x[:, None], stage)[0])}
    out = pd.DataFrame(rows).T.astype({"distinct values": int})
    out.index.name = "sensor"
    return out


# ---------------------------------------------------------------- detectors: alarm lines per group

def _quantile(chunks: list, far: float) -> float:
    return float(np.quantile(np.concatenate(chunks), 1 - far))


def _lines(normal: list, line_idx: dict, far: float, per_group: bool, picks: dict | None = None) -> dict:
    """{group: alarm line}. picks: for each group, which of its line batches to use (a resample)."""
    chosen = {g: [normal[idx[j]] for j in (picks[g] if picks is not None else range(len(idx)))]
              for g, idx in line_idx.items()}
    if per_group:
        return {g: _quantile(c, far) for g, c in chosen.items()}
    one = _quantile([s for c in chosen.values() for s in c], far)
    return {g: one for g in chosen}


def _above(sorted_scores: list, groups: list, lines: dict) -> np.ndarray:
    """Per batch: how many of its readings are above its group's line."""
    return np.array([len(s) - np.searchsorted(s, lines[g], side="right") for s, g in zip(sorted_scores, groups)], float)


def report(per_method: dict, fault_kinds: list, fault_groups: list, normal_groups: list, normal_roles: list,
           per_group: bool = True, far: float = 0.01, n_boot: int = 1000, seed: int = 0,
           verbose: bool = True, what: str = "resampling faulty batches") -> dict:
    """Catch rate of every detector on faulty batches, every reading of which is labelled faulty.

    per_method    {detector name: list of (fault scores, normal scores) pairs, one pair per seed}:
                  fault scores is one array per faulty batch, normal scores one per normal batch
                  that was not fitted on (line and check batches, in the order of normal_roles)
    fault_kinds   per faulty batch: its kind of fault (for the table by kind)
    fault_groups  per faulty batch: its group (experiment); normal_groups the same for normal batches
    normal_roles  per normal batch scored: "line" (sets the alarm line) or "check" (only scored)
    per_group     True: one line per group, from that group's line batches; False: one line for all

    The catch rate is the share of all faulty readings above their group's line. Ranges come
    from resampling the faulty batches and, in the same draw, each group's line batches (so how
    uncertain the line is counts too). The same draws serve every detector, so gaps are paired.
    "far_check" is the share of check batches' readings above the line: about `far` if the line
    carries over to normal batches it was not set on.

    Returns a dict shaped like run.report_dataset's, so run.rates, gap_table, verdict_table and
    fault_table work on it.
    """
    groups = list(dict.fromkeys(normal_groups))
    line_idx = {g: [i for i, (gg, r) in enumerate(zip(normal_groups, normal_roles)) if gg == g and r == "line"]
                for g in groups}
    check_idx = [i for i, r in enumerate(normal_roles) if r == "check"]
    rng = np.random.default_rng(seed)
    n = len(fault_kinds)
    fault_picks = rng.integers(0, n, (n_boot, n))
    line_picks = [{g: rng.integers(0, len(idx), len(idx)) for g, idx in line_idx.items()} for _ in range(n_boot)]
    sizes = None
    summary, by_group, draws, point = [], [], {}, {}
    for name, seeds in progress.track(per_method.items(), what, total=len(per_method), verbose=verbose):
        per_seed = []
        for k, (faults, normal) in enumerate(seeds):
            sorted_faults = [np.sort(np.asarray(f, float)) for f in faults]
            sizes = np.array([len(f) for f in faults], float)
            lines = _lines(normal, line_idx, far, per_group)
            above = _above(sorted_faults, fault_groups, lines)
            check = np.concatenate([normal[i] > lines[normal_groups[i]] for i in check_idx]) if check_idx else np.array([])
            line_far = np.concatenate([normal[i] > lines[g] for g, idx in line_idx.items() for i in idx])
            per_seed.append({"dr_at_1pct": above.sum() / sizes.sum(), "far_check": float(check.mean()) if len(check) else np.nan,
                             "far_line": float(line_far.mean())})
            if k == 0:
                point[name] = per_seed[-1]["dr_at_1pct"]
                kinds = pd.Series(above).groupby(np.asarray(fault_kinds), sort=False).sum() / \
                    pd.Series(sizes).groupby(np.asarray(fault_kinds), sort=False).sum()
                by_group += [{"method": name, "group": g, "dr_at_1pct": float(v)} for g, v in kinds.items()]
                for g in groups:
                    here = [normal[i] > lines[g] for i in check_idx if normal_groups[i] == g]
                    per_seed[-1][f"far_check_{g}"] = float(np.concatenate(here).mean()) if here else np.nan
                d = np.empty(n_boot)
                for b in range(n_boot):
                    lb = _lines(normal, line_idx, far, per_group, line_picks[b])
                    idx = fault_picks[b]
                    d[b] = _above([sorted_faults[i] for i in idx], [fault_groups[i] for i in idx], lb).sum() / sizes[idx].sum()
                draws[name] = d
        frame = pd.DataFrame(per_seed)
        row = {"method": name, "seeds": len(seeds), **frame.mean().to_dict(),
               "dr_at_1pct_seed_sd": float(frame["dr_at_1pct"].std(ddof=0))}
        row["dr_ci90_low"], row["dr_ci90_high"] = map(float, np.nanquantile(draws[name], [0.05, 0.95]))
        summary.append(row)
    pairs = paired_differences(draws, point)
    sds = np.array([p["sd"] for p in pairs]) if pairs else np.array([np.nan])
    from .run import MARGIN
    return {"summary": summary, "by_group": by_group, "pairs": pairs, "draws": draws,
            "noise": {"median_pair_sd": float(np.median(sds)), "max_pair_sd": float(np.max(sds)), "margin": MARGIN},
            "runs": n, "calibration": "one line per group" if per_group else "one line for all",
            "line_batches": {g: len(v) for g, v in line_idx.items()}, "check_batches": len(check_idx)}


def lines_table(per_group: dict, one_line: dict, digits: int = 2) -> pd.DataFrame:
    """Per detector: catch rate (with range) and false alarms on unseen normal batches, under both lines."""
    from .run import rates
    a = pd.DataFrame(per_group["summary"]).set_index("method")
    b = pd.DataFrame(one_line["summary"]).set_index("method")
    out = pd.DataFrame({"caught, line per experiment": rates(per_group, digits),
                        "false alarms on unseen normal wafers, line per experiment": a["far_check"].round(3),
                        "caught, one line": rates(one_line, digits),
                        "false alarms on unseen normal wafers, one line": b["far_check"].round(3)})
    out = out.loc[a.sort_values("dr_at_1pct", ascending=False).index]
    out.index.name = "detector"
    return out


def check_table(res: dict, digits: int = 3) -> pd.DataFrame:
    """Per detector and group: false alarms on the group's unseen normal (check) batches."""
    df = pd.DataFrame(res["summary"]).set_index("method")
    cols = [c for c in df.columns if c.startswith("far_check_")]
    out = df[cols].round(digits)
    out.columns = [f"experiment {c[len('far_check_'):]}" for c in cols]
    out.index.name = "detector"
    return out


def sensors_moved(fit: np.ndarray, faults: list, fault_groups: list, normal: list, normal_groups: list,
                  normal_roles: list, per_group: bool = True, far: float = 0.01, window: int = 5,
                  share: float = 0.5) -> list[int]:
    """Per faulty batch: how many sensors, each judged alone, are out of range for most of the batch.

    As structure.sensors_moved: a sensor counts if its rolling average (standardised on the fit
    data) is beyond its own `far` line in more than `share` of the batch's readings. The lines
    come from the line batches, per group or one for all, as in report().
    """
    mu, sd = structure._standardise(fit)

    def rolled(X):
        return np.abs(det._trailing_mean((np.asarray(X) - mu) / sd, window))

    line_batches = [(g, rolled(X)) for X, g, r in zip(normal, normal_groups, normal_roles) if r == "line"]
    if per_group:
        lines = {g: np.quantile(np.vstack([z for gg, z in line_batches if gg == g]), 1 - far, axis=0)
                 for g in dict.fromkeys(normal_groups)}
    else:
        one = np.quantile(np.vstack([z for _, z in line_batches]), 1 - far, axis=0)
        lines = {g: one for g in dict.fromkeys(normal_groups)}
    return [int(((rolled(X) > lines[g]).mean(0) > share).sum()) for X, g in zip(faults, fault_groups)]


def fault_kinds(res: dict, moved: list, fault_kind_of_batch: list, caught: float = 0.5, few: int = 3) -> pd.DataFrame:
    """structure.fault_types for batch data: catch rates by family of checks per kind of fault,
    sensors moved (average over the kind's batches) and the kind's label."""
    rates = pd.DataFrame(res["by_group"]).pivot(index="group", columns="method", values="dr_at_1pct")
    label = {k: det.REGISTRY[k].name for keys in FAMILIES.values() for k in keys}
    cols = {}
    for family, keys in FAMILIES.items():
        have = [label[k] for k in keys if label[k] in rates.columns]
        if have:
            cols[family] = rates[have].max(axis=1)
    out = pd.DataFrame(cols).loc[list(dict.fromkeys(fault_kind_of_batch))]
    out["sensors moved"] = pd.Series(moved, index=fault_kind_of_batch).groupby(level=0, sort=False).mean().round(1)
    out["kind"] = structure.kind_of_fault(out, caught, few)
    out.index.name = "kind of fault"
    return out
