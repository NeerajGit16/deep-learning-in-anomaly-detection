"""The Cranfield authors' own monitoring method, ported from their MATLAB example (cvatutor.m).

Only used as a check that we read their data the way they do: with their settings it should
reproduce the two figures their example draws (Figures 7 and 9 of Ruiz-Carcel et al. 2015). It
is not one of this project's detectors.

Canonical variate analysis (CVA) in a sentence: take the last `p` readings of every sensor as
"the past" and the next `f` as "the future", find the few combinations of the past that best
predict the future, and watch two numbers. T2 says how far those combinations are from their
normal range; SPE says how much of the past they fail to describe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _past(y: np.ndarray, p: int, f: int) -> np.ndarray:
    """Columns of stacked past readings: column j holds readings j+p-1, j+p-2, ..., j (newest first)."""
    n = len(y) - p - f + 1
    return np.hstack([y[p - 1 - i: p - 1 - i + n] for i in range(p)]).T


def _future(y: np.ndarray, p: int, f: int) -> np.ndarray:
    """Columns of stacked future readings: column j holds readings j+p, ..., j+p+f-1."""
    n = len(y) - p - f + 1
    return np.hstack([y[p + i: p + i + n] for i in range(f)]).T


def kde_limit(x: np.ndarray, level: float = 0.99, points: int = 100) -> float:
    """The value below which a share `level` of x lies, read off a smoothed histogram (as in gkde)."""
    x = np.asarray(x, float).ravel()
    h = np.median(np.abs(x - np.median(x))) * 1.5704 / len(x) ** 0.2
    grid = np.linspace(x.min() - 3 * h, x.max() + 3 * h, points)
    density = np.array([np.exp(-0.5 * ((t - x) / h) ** 2).sum() for t in grid]) / (h * np.sqrt(2 * np.pi) * len(x))
    cumulative = np.cumsum(np.r_[0, np.diff(grid)] * density)
    return float(grid[cumulative <= level].max())


def fit(train: list, p: int, f: int, n: int, ridge: float = 0.0) -> dict:
    """The authors' CVA fitted on the normal runs in `train`: {"J", "L", "p_mean", "p_std", "Yp"}.

    ridge adds that much to the diagonal of both correlation matrices before their Cholesky factors are
    taken (0 in the authors' example; a tiny amount keeps the factors defined when sensors move together
    exactly, as in round 4's detector on every dataset)."""
    Yp = np.hstack([_past(X, p, f) for X in train])
    Yf = np.hstack([_future(X, p, f) for X in train])
    count = Yp.shape[1]
    p_mean, p_std = Yp.mean(1, keepdims=True), Yp.std(1, ddof=1, keepdims=True)
    Yp = (Yp - p_mean) / p_std
    Yf = (Yf - Yf.mean(1, keepdims=True)) / Yf.std(1, ddof=1, keepdims=True)
    eye_p, eye_f = ridge * np.eye(len(Yp)), ridge * np.eye(len(Yf))
    Rp = np.linalg.cholesky(Yp @ Yp.T / (count - 1) + eye_p).T          # upper triangles, as MATLAB's chol
    Rf = np.linalg.cholesky(Yf @ Yf.T / count + eye_f).T
    H = np.linalg.solve(Rf.T, Yf @ Yp.T / count) @ np.linalg.inv(Rp)
    V = np.linalg.svd(H)[2].T
    V1 = V[:, :n]
    inv_Rp_t = np.linalg.inv(Rp.T)
    return {"J": V1.T @ inv_Rp_t, "L": (np.eye(len(V)) - V1 @ V1.T) @ inv_Rp_t, "p_mean": p_mean, "p_std": p_std,
            "Yp": Yp}


def stats(model: dict, Y: np.ndarray) -> tuple:
    """T2 and SPE (Q) of standardised past windows Y (one per column)."""
    return ((model["J"] @ Y) ** 2).sum(0), ((model["L"] @ Y) ** 2).sum(0)


def cva_monitor(train: list, test: np.ndarray, level: float = 0.99, n: int = 25, p: int = 15, f: int = 15) -> dict:
    """Fit on the normal runs in `train`, then compute T2 and SPE for every reading of `test`.

    Returns {"T2", "SPE"} (one value per reading from the p-th on), their limits {"T2_limit",
    "SPE_limit"} at `level`, "first" (index in `test` of the first value) and the share of the
    normal training readings above each limit.
    """
    model = fit(train, p, f, n)
    T_train, Q_train = stats(model, model["Yp"])
    T_limit, Q_limit = kde_limit(T_train, level), kde_limit(Q_train, level)
    T, Q = stats(model, (_past(test, p, 0) - model["p_mean"]) / model["p_std"])
    return {"T2": T, "SPE": Q, "T2_limit": T_limit, "SPE_limit": Q_limit, "first": p - 1,
            "train_above": {"T2": float(np.mean(T_train > T_limit)), "SPE": float(np.mean(Q_train > Q_limit))}}


def authors_example(which: str = "3_1") -> dict:
    """The authors' example on our copy of their data: train on T2 and T3, columns 1-23, test on one faulty run."""
    from . import data
    normal = data.cranfield_normal()
    out = cva_monitor([normal["T2"], normal["T3"]], data.cranfield_faulty()[which][0])
    out["spans"], out["name"] = data.CRANFIELD_TIMES[which][1], which
    return out


def check_table(results: list) -> pd.DataFrame:
    """For each example: the share of fault and of normal readings above each limit, and the
    first reading above each limit inside every published fault span (to set beside the
    detection times the paper quotes)."""
    from . import data
    rows = {}
    for r in results:
        n = r["first"] + len(r["T2"])
        y = data.cranfield_labels(n, r["spans"])[r["first"]:].astype(bool)
        row = {"published fault": ", ".join(f"{a}-{b}" for a, b in r["spans"])}
        for key in ("T2", "SPE"):
            above = r[key] > r[f"{key}_limit"]
            row[f"{key} above its limit, fault readings"] = np.mean(above[y])
            row[f"{key} above its limit, normal readings"] = np.mean(above[~y])
            row[f"{key} first above its limit, in each fault span"] = ", ".join(
                str(first_above(above, a, b, r["first"])) for a, b in r["spans"])
        rows[f"run {r['name'].replace('_', '.')}"] = row
    return pd.DataFrame(rows).T


def first_above(above: np.ndarray, start: int, end: int, offset: int):
    """Reading number (from 1) of the first value above the limit between readings start and end, or None."""
    part = above[start - 1 - offset: end - offset]
    return start + int(part.argmax()) if part.any() else None
