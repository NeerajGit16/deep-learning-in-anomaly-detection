"""Metrics that compare methods fairly across datasets with different anomaly shares.

Primary: detection rate at a fixed false-alarm rate (one point on the ROC curve). It needs no
chosen threshold and does not move when the share of anomalous samples changes.

All functions take per-run lists of scores and labels. Scores should already be normalised
per run (see `normalise`) so that one threshold means the same thing on every run.
"""
from __future__ import annotations

import numpy as np


def normalise(scores: np.ndarray, fit_scores: np.ndarray, q: float = 0.99) -> np.ndarray:
    """Express scores in units of the detector's own q-quantile on its fit data."""
    ref = float(np.quantile(fit_scores, q))
    return scores / ref if ref > 0 else scores.astype(float)


def segments(y: np.ndarray) -> list[tuple[int, int]]:
    """Start (inclusive) and end (exclusive) of each run of 1s."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(y, int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def dr_at_far(scores: list, labels: list, far: float = 0.01, mode: str = "pooled") -> dict:
    """Detection rate at the threshold that gives `far` false alarms on normal samples.

    mode="pooled": one threshold across all runs; DR over all anomalous samples.
    mode="per_run": each run gets its own threshold; DR is the average over runs.
    """
    if mode == "per_run":
        thr, dr, fa = [], [], []
        for s, y in zip(scores, labels):
            y = np.asarray(y, bool)
            t = float(np.quantile(s[~y], 1 - far))
            thr.append(t)
            dr.append(float(np.mean(s[y] > t)))
            fa.append(float(np.mean(s[~y] > t)))
        return {"threshold": thr, "far": float(np.mean(fa)), "dr": float(np.mean(dr)), "per_run": dr}
    s = np.concatenate(scores)
    y = np.concatenate(labels).astype(bool)
    thr = float(np.quantile(s[~y], 1 - far))
    return {"threshold": thr, "far": float(np.mean(s[~y] > thr)), "dr": float(np.mean(s[y] > thr))}


def dr_by_group(scores: list, labels: list, groups: list, thr) -> dict:
    """Share of anomalous samples above the threshold, per anomaly type.

    `thr` is one number (pooled) or one per run (per-run calibration; runs then count equally).
    """
    per_run = np.ndim(thr) > 0
    out = {}
    for g in dict.fromkeys(groups):
        idx = [i for i, gi in enumerate(groups) if gi == g]
        if per_run:
            rates = [np.mean(scores[i][np.asarray(labels[i], bool)] > thr[i]) for i in idx]
            out[g] = float(np.mean(rates))
            continue
        s = np.concatenate([scores[i] for i in idx])
        y = np.concatenate([labels[i] for i in idx]).astype(bool)
        out[g] = float(np.mean(s[y] > thr)) if y.any() else np.nan
    return out


def detection_delays(score: np.ndarray, y: np.ndarray, thr: float) -> list[float]:
    """Samples from each anomaly's start to its first alarm inside it; nan if never flagged."""
    out = []
    for a, b in segments(y):
        hits = np.flatnonzero(score[a:b] > thr)
        out.append(float(hits[0]) if len(hits) else np.nan)
    return out


def deployable(scores: list, labels: list, far: float = 0.01, val_normal: list | None = None) -> dict:
    """Threshold set without the test labels of the run being scored.

    With `val_normal` (scores on held-out normal series) the threshold is their (1 - far)
    quantile. Without it, each run's threshold comes from the other runs' normal samples.
    Reports the false-alarm rate the threshold actually gives, the detection rate, the share
    of anomalies flagged at all, and the median delay to the first alarm.
    """
    if val_normal is not None:
        thr = float(np.quantile(np.concatenate(val_normal), 1 - far))
        thresholds = [thr] * len(scores)
    else:
        normals = [s[~np.asarray(y, bool)] for s, y in zip(scores, labels)]
        thresholds = [float(np.quantile(np.concatenate(normals[:i] + normals[i + 1:]), 1 - far))
                      for i in range(len(scores))]
    fa = da = n_norm = n_anom = 0
    delays = []
    for s, y, thr in zip(scores, labels, thresholds):
        y = np.asarray(y, bool)
        fa += int(np.sum(s[~y] > thr))
        da += int(np.sum(s[y] > thr))
        n_norm += int((~y).sum())
        n_anom += int(y.sum())
        delays += detection_delays(s, y, thr)
    d = np.array(delays)
    return {
        "far_realised": fa / max(n_norm, 1),
        "dr": da / max(n_anom, 1),
        "events_flagged": float(np.mean(~np.isnan(d))) if len(d) else np.nan,
        "median_delay": float(np.nanmedian(d)) if np.any(~np.isnan(d)) else np.nan,
    }


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, score))


def vus_available() -> bool:
    """True if the optional TSB-AD package is installed. Reports leave VUS-PR out when it is not."""
    import importlib.util
    return importlib.util.find_spec("TSB_AD") is not None


def vus_pr(score: np.ndarray, y: np.ndarray, window: int = 10) -> float:
    """VUS-PR from TSB-AD (Liu & Paparrizos, NeurIPS 2024); nan if TSB-AD is not installed."""
    try:
        from TSB_AD.evaluation.basic_metrics import generate_curve
    except ImportError:
        return np.nan
    return float(generate_curve(np.asarray(y, int), np.asarray(score, float), window, "opt", 250)[-1])
