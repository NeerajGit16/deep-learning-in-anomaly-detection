"""Bootstrap over runs: how much a metric, or a gap between two methods, moves by chance."""
from __future__ import annotations

import numpy as np

from .metrics import dr_at_far


def _resampled_dr(normals: list, anomalies: list, idx: np.ndarray, far: float) -> float:
    s_n = np.concatenate([normals[i] for i in idx])
    s_a = np.concatenate([anomalies[i] for i in idx])
    thr = np.quantile(s_n, 1 - far)
    return float(np.mean(s_a > thr)) if len(s_a) else np.nan


def resample_indices(n: int, n_boot: int, rng, clusters: list | None = None) -> list[np.ndarray]:
    """n_boot resamples of the run indices 0..n-1, drawn with replacement.

    clusters=None draws single runs. With `clusters` (one label per run), whole clusters are
    drawn and every run of a drawn cluster comes along. Use it when runs are not independent
    of each other: for example runs of the same kind of fault, or runs that share the same
    random numbers.
    """
    if clusters is None:
        return list(rng.integers(0, n, (n_boot, n)))
    members: dict = {}
    for i, c in enumerate(clusters):
        members.setdefault(c, []).append(i)
    groups = [np.array(v) for v in members.values()]
    picks = rng.integers(0, len(groups), (n_boot, len(groups)))
    return [np.concatenate([groups[j] for j in row]) for row in picks]


class AlarmCounts:
    """Per run: how many of its samples are above an alarm line. Quick to ask again for another line."""

    def __init__(self, samples: list):
        self.sorted = [np.sort(np.asarray(s, float)) for s in samples]
        self.sizes = np.array([len(s) for s in self.sorted], float)

    def above(self, thr: float) -> np.ndarray:
        return np.array([len(s) - np.searchsorted(s, thr, side="right") for s in self.sorted], float)

    def rates(self, thr: float) -> np.ndarray:
        """Share of each run's samples above the line."""
        return self.above(thr) / np.where(self.sizes > 0, self.sizes, 1)


def line_draws(calib: list, far: float, n_boot: int, rng) -> tuple[float, np.ndarray]:
    """Alarm line from fault-free calibration runs, and the line from each resample of those runs.

    One calibration run cannot be resampled, so then every draw is the same line: the ranges
    built on it leave out how uncertain the line itself is.
    """
    point = float(np.quantile(np.concatenate(calib), 1 - far))
    picks = rng.integers(0, len(calib), (n_boot, len(calib)))
    draws = np.array([np.quantile(np.concatenate([calib[j] for j in row]), 1 - far) for row in picks])
    return point, draws


def bootstrap_dr(per_method: dict, labels: list, far: float = 0.01, n_boot: int = 1000,
                 seed: int = 0, mode: str = "pooled", clusters: list | None = None,
                 calib: dict | None = None) -> dict:
    """Resample runs with replacement and recompute DR@FAR for every method.

    The same resampled runs are used for every method, so differences between methods are
    paired. `clusters` resamples whole clusters of runs (see resample_indices). With
    mode="matched", `calib` holds each method's fault-free calibration scores; those runs are
    resampled too, so the ranges include how uncertain the alarm line is.
    Returns {method: array of n_boot detection rates}.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx_all = resample_indices(n, n_boot, rng, clusters)
    if mode == "per_run":
        per = {m: np.asarray(dr_at_far(sc, labels, far, "per_run")["per_run"]) for m, sc in per_method.items()}
        if clusters is None:
            return {m: p[np.asarray(idx_all)].mean(1) for m, p in per.items()}
        return {m: np.array([p[idx].mean() for idx in idx_all]) for m, p in per.items()}
    masks = [np.asarray(y, bool) for y in labels]
    if mode == "matched":
        n_cal = len(next(iter(calib.values())))
        picks = rng.integers(0, n_cal, (n_boot, n_cal))       # the same calibration runs for every method
        draws = {}
        for m, sc in per_method.items():
            counts = AlarmCounts([s[k] for s, k in zip(sc, masks)])
            lines = [np.quantile(np.concatenate([calib[m][j] for j in row]), 1 - far) for row in picks]
            draws[m] = np.array([counts.above(thr)[idx].sum() / counts.sizes[idx].sum()
                                 for thr, idx in zip(lines, idx_all)])
        return draws
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


def point_dr(per_method: dict, labels: list, far: float = 0.01, mode: str = "pooled",
             calib: dict | None = None) -> dict:
    return {m: dr_at_far(sc, labels, far, mode, calib[m] if calib else None)["dr"] for m, sc in per_method.items()}


def interval(draws: np.ndarray) -> tuple[float, float]:
    """The 90% interval used throughout: the 5th and 95th percentile of the resampled values."""
    lo, hi = np.nanquantile(draws, [0.05, 0.95])
    return float(lo), float(hi)
