"""Round 4, part 1: the fold audit, and balanced folds for every dataset.

Some detectors set each sensor's error scale on fitting data held out of their own fitting:
    the link detectors (detectors.LinkResidual: straight, curved, kinked, stage or mode told)
        five blocks of time, each held out in turn;
    the deep models, the straight forecaster and the PCA scored like the autoencoder (deep.py)
        one held-aside part (deep.split_aside: every fifth run, or the last fifth of each run).
On the six-mode plant a block of time held out most of one operating mode, so that mode's normal
errors looked huge and set the scale far too wide (notebook 06, section 20). No alarm line in this
project is set on held-out blocks: lines come from separate normal runs, or each run's own normal
readings, or line wafers; the schedule checks hold out whole batches only for one overall scale,
which does not change any catch rate.

The audit (fixed before computing). For every dataset, both ways of holding data out, and every
kind of group in the fitting data (flow setting, recipe step, experiment, control strategy,
operating mode, stage of a batch): for each held-out fold (for the held-aside part: its training
part), the share of each group's fitting readings left in the part the models are fitted on. A
group is LEFT UNSEEN when under 10% of its fitting readings remain. Where any fold leaves any group
unseen, the detectors using that way of holding out switch to balanced folds on that dataset.

Balanced folds (fixed before computing). The fitting data are cut into visits: unbroken stretches
of one group inside one run (for groups that last a whole run, a visit is a run). Each group's
visits, in time order, are dealt to K = 5 folds in turn (0, 1, 2, 3, 4, 0, ...). A group with
fewer than 5 visits has its readings, in time order, cut into 5 contiguous pieces instead, one per
fold. So every fold holds out about a fifth of every group, and every group stays in the fitted
part. The held-aside part of the deep models is fold 4. Balancing is on the kind of group the
audit flags (on their combination if it flags more than one). On the six-mode plant this gives
exactly the round-3 balanced folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 5
UNSEEN = 0.10
BLOCKS = "blocks of time (link detectors)"
ASIDE = "held-aside part (deep models, straight forecaster, PCA scored alike)"
# Registry keys whose error scale is set on held-out blocks of time, or on a held-aside part.
BLOCK_METHODS = {"link_straight", "link_curved", "link_kinked", "link_straight_told"}
ASIDE_METHODS = {"deep_ae", "deep_lstm", "deep_gdn", "deep_gdn_paper", "deep_usad", "straight_forecaster", "pca_twin"}
ASIDE_METHODS |= {"deep_ae_outputs_v2", "deep_ae_v2", "deep_lstm_v2", "deep_pcac_v2", "deep_gdn_v2", "deep_gdn_paper_v2", "deep_usad_v2"}  # round 5
# A dataset copy with balanced folds is named with one of these endings; the detectors that do not use
# the balanced folds read and write the original dataset's scores.
SUFFIXES = {"_balanced_blocks": BLOCK_METHODS, "_balanced_aside": ASIDE_METHODS,
            "_balanced": BLOCK_METHODS | ASIDE_METHODS}


def shared_name(ds_name: str, method: str) -> str:
    """The dataset name a method's scores are cached under (see SUFFIXES)."""
    for end, methods in SUFFIXES.items():
        if ds_name.endswith(end):
            return ds_name if method in methods else ds_name[:-len(end)]
    return ds_name


def balanced_copy(ds, groups, schemes: tuple):
    """A copy of a Dataset whose detectors use balanced folds by `groups` (one label per fitting reading) for the
    ways of holding out in `schemes` (BLOCKS and/or ASIDE)."""
    import dataclasses
    groups = np.asarray(groups)
    fg = {"blocks": groups} if BLOCKS in schemes else {}
    fg.update({"aside": groups} if ASIDE in schemes else {})
    end = "_balanced" if len(fg) == 2 else "_balanced_blocks" if "blocks" in fg else "_balanced_aside"
    return dataclasses.replace(ds, name=ds.name + end, fold_groups=fg)


def _segments(n: int, lengths) -> list:
    if not lengths or sum(lengths) != n:
        return [(0, n)]
    edges = np.cumsum([0] + list(lengths))
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def visits(groups, lengths) -> list:
    """(start, end, group) of every unbroken stretch of one group inside one run, in time order."""
    groups = np.asarray(groups)
    out = []
    for a, b in _segments(len(groups), lengths):
        g = groups[a:b]
        cuts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(g)]
        out += [(a + int(s), a + int(e), g[s].item()) for s, e in zip(cuts[:-1], cuts[1:])]
    return out


def fold_of(groups, lengths, k: int = K) -> np.ndarray:
    """The balanced fold (0 to k-1) of every fitting reading."""
    groups = np.asarray(groups)
    fold = np.empty(len(groups), int)
    by_group: dict = {}
    for a, b, g in visits(groups, lengths):
        by_group.setdefault(g, []).append((a, b))
    for g, vs in by_group.items():
        if len(vs) >= k:
            for i, (a, b) in enumerate(vs):
                fold[a:b] = i % k
        else:
            idx = np.concatenate([np.arange(a, b) for a, b in vs])
            for f, piece in enumerate(np.array_split(idx, k)):
                fold[piece] = f
    return fold


def balanced_folds(groups, lengths, k: int = K) -> list:
    """Held-out index arrays, one per fold."""
    fold = fold_of(groups, lengths, k)
    return [np.flatnonzero(fold == f) for f in range(k) if (fold == f).any()]


def _runs_of(mask: np.ndarray, lengths) -> list:
    """(start, end) of every unbroken stretch of True inside one run."""
    out = []
    for a, b in _segments(len(mask), lengths):
        m = mask[a:b].astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
        out += [(a + int(s), a + int(e)) for s, e in zip(edges[::2], edges[1::2])]
    return out


def balanced_aside(groups, lengths, k: int = K) -> tuple[list, list]:
    """(training segments, held-aside segments) for the deep models: fold k-1 held aside."""
    aside = fold_of(groups, lengths, k) == k - 1
    return _runs_of(~aside, lengths), _runs_of(aside, lengths)


# ---------------------------------------------------------------- what each dataset's fitting data hold

def fit_structure(name: str) -> dict:
    """{"lengths": runs joined into the fitting data, "kinds": {kind of group: label per reading}, "fits": how
    many separate fitting arrays, "note"} for the fitting data the detectors use on a dataset."""
    from . import campaign, data, deep, etch, indpensim, multimode
    if name == "skab":
        ds = data.load("skab")
        return {"lengths": None, "kinds": {}, "fits": len(ds.runs), "n": int(np.mean([len(r.fit) for r in ds.runs])),
                "note": "each session is fitted on its own first readings: one session per fit, no group to leave out"}
    if name == "tep_rieth":
        ds = deep.dataset("tep_rieth")["ds"]
        return {"lengths": ds.fit_lengths, "kinds": {}, "fits": 1, "n": sum(ds.fit_lengths),
                "note": "20 fault-free runs at one setting: no setting, mode or stage to leave out"}
    if name == "cranfield":
        normal = data.cranfield_normal()
        parts = [normal[k] for k in ("T2", "T3")]
        return {"lengths": [len(p) for p in parts], "fits": 1, "n": sum(len(p) for p in parts),
                "kinds": {"flow setting": np.concatenate([data.flow_settings(p) for p in parts])},
                "note": "runs T2 and T3, each passing through many flow settings"}
    if name == "tep6":
        ds = multimode.detector_dataset()
        return {"lengths": ds.fit_lengths, "fits": 1, "n": sum(ds.fit_lengths),
                "kinds": {"operating mode": np.repeat(ds.fit_groups, ds.fit_lengths)},
                "note": "5 runs of each of 6 modes, mode by mode"}
    if name == "etch":
        ws, r = etch.normal_wafers(), etch.roles()
        fit = [w for w, role in zip(ws, r) if role == "fit"]
        return {"lengths": [w["readings"] for w in fit], "fits": 1, "n": sum(w["readings"] for w in fit),
                "kinds": {"experiment": np.concatenate([np.full(w["readings"], w["experiment"]) for w in fit]),
                          "recipe step": np.concatenate([np.asarray(w["step"]) for w in fit])},
                "note": "fit wafers of experiments 29, 31 and 33, in that order"}
    if name == "indpensim":
        ds = indpensim.detector_dataset()
        return {"lengths": ds.fit_lengths, "fits": 1, "n": sum(ds.fit_lengths),
                "kinds": {"control strategy": np.repeat(ds.fit_groups, ds.fit_lengths),
                          "stage (tenth of the batch)": np.concatenate([indpensim.stage_of(n) for n in ds.fit_lengths])},
                "note": "60 fit batches in batch order: recipe, then operators, then Raman control"}
    if name.startswith("campaign_"):
        ds = campaign.dataset(name.split("_", 1)[1])
        return {"lengths": ds.fit_lengths, "fits": 1, "n": sum(ds.fit_lengths),
                "kinds": {"control strategy": np.repeat(ds.fit_groups, ds.fit_lengths),
                          "stage (tenth of the batch)": np.concatenate([indpensim.stage_of(n) for n in ds.fit_lengths])},
                "note": "generated batches: recipe fit batches" + (", then operator fit batches" if "mixed" in name else "")}
    raise KeyError(name)


AUDITED = ["skab", "tep_rieth", "cranfield", "tep6", "etch", "indpensim", "campaign_recipe", "campaign_mixed"]
TITLES = {"skab": "Water pump (SKAB)", "tep_rieth": "Chemical plant, one setting", "cranfield": "Flow rig (Cranfield)",
          "tep6": "Chemical plant, six modes", "etch": "Metal etcher", "indpensim": "Penicillin batches (IndPenSim)",
          "campaign_recipe": "Generated campaign, recipe setup", "campaign_mixed": "Generated campaign, mixed setup"}


def _held_out(scheme: str, n: int, lengths) -> list:
    from .deep import segments, split_aside
    from .detectors import blocked_folds
    if scheme == BLOCKS:
        return blocked_folds(n, K)
    _, aside = split_aside(segments(n, lengths))
    return [np.concatenate([np.arange(a, b) for a, b in aside])]


def left(groups, held: list) -> pd.DataFrame:
    """Per group: its fitting readings, and the smallest share of them left in the fitted part over the folds."""
    groups = np.asarray(groups)
    names, counts = np.unique(groups, return_counts=True)
    rows = {}
    for g, c in zip(names, counts):
        rest = [1 - np.mean(groups[h] == g) * len(h) / c for h in held]
        rows[g] = {"readings": int(c), "smallest share left": float(min(rest)), "fold": int(np.argmin(rest))}
    return pd.DataFrame(rows).T


def audit_one(name: str) -> list:
    s = fit_structure(name)
    if not s["kinds"]:
        return [{"dataset": TITLES[name], "kind of group": "none", "held out by": scheme, "groups": 1,
                 "worst group": "", "smallest share left": 1.0, "groups left unseen": 0, "switch": "no",
                 "note": s["note"]} for scheme in (BLOCKS, ASIDE)]
    out = []
    for kind, groups in s["kinds"].items():
        for scheme in (BLOCKS, ASIDE):
            t = left(groups, _held_out(scheme, len(groups), s["lengths"]))
            worst = t["smallest share left"].astype(float).idxmin()
            unseen = int((t["smallest share left"].astype(float) < UNSEEN).sum())
            out.append({"dataset": TITLES[name], "kind of group": kind, "held out by": scheme, "groups": len(t),
                        "worst group": str(worst), "smallest share left": float(t.loc[worst, "smallest share left"]),
                        "groups left unseen": unseen, "switch": "yes" if unseen else "no", "note": s["note"]})
    return out


def audit(names=AUDITED) -> pd.DataFrame:
    """The audit table: one row per dataset, kind of group and way of holding out."""
    return pd.DataFrame([row for n in names for row in audit_one(n)]).set_index(["dataset", "kind of group", "held out by"])


def flagged(table: pd.DataFrame) -> dict:
    """{dataset key: {scheme: [kinds of group flagged]}} from the audit table."""
    keys = {v: k for k, v in TITLES.items()}
    out: dict = {}
    for (ds, kind, scheme), row in table.iterrows():
        if row["switch"] == "yes":
            out.setdefault(keys[ds], {}).setdefault(scheme, []).append(kind)
    return out
