"""Bootstrap over runs: how much a metric, or a gap between two methods, moves by chance."""
from __future__ import annotations

import numpy as np

from .metrics import dr_at_far


def _resampled_dr(normals: list, anomalies: list, idx: np.ndarray, far: float) -> float:
    s_n = np.concatenate([normals[i] for i in idx])
    s_a = np.concatenate([anomalies[i] for i in idx])
    thr = np.quantile(s_n, 1 - far)
    return float(np.mean(s_a > thr)) if len(s_a) else np.nan


def bootstrap_dr(per_method: dict, labels: list, far: float = 0.01, n_boot: int = 1000,
                 seed: int = 0, mode: str = "pooled") -> dict:
    """Resample runs with replacement and recompute DR@FAR for every method.

    The same resampled runs are used for every method, so differences between methods are
    paired. Returns {method: array of n_boot detection rates}.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx_all = rng.integers(0, n, (n_boot, n))
    if mode == "per_run":
        return {m: np.asarray(dr_at_far(sc, labels, far, "per_run")["per_run"])[idx_all].mean(1)
                for m, sc in per_method.items()}
    masks = [np.asarray(y, bool) for y in labels]
    split = {m: ([s[~k] for s, k in zip(sc, masks)], [s[k] for s, k in zip(sc, masks)])
             for m, sc in per_method.items()}
    draws = {m: np.empty(n_boot) for m in per_method}
    for b, idx in enumerate(idx_all):
        for m, (normals, anomalies) in split.items():
            draws[m][b] = _resampled_dr(normals, anomalies, idx, far)
    return draws


def paired_differences(draws: dict, point: dict) -> list[dict]:
    """For every pair of methods: observed gap in DR, bootstrap s.d. of the gap, 90% interval."""
    names = list(draws)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = draws[a] - draws[b]
            lo, hi = np.nanquantile(diff, [0.05, 0.95])
            rows.append({"method_a": a, "method_b": b, "gap": point[a] - point[b],
                         "sd": float(np.nanstd(diff)), "ci90_low": float(lo), "ci90_high": float(hi)})
    return rows


def point_dr(per_method: dict, labels: list, far: float = 0.01, mode: str = "pooled") -> dict:
    return {m: dr_at_far(sc, labels, far, mode)["dr"] for m, sc in per_method.items()}
