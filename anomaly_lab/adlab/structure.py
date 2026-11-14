"""Two questions about a dataset, answered before any deep model is trained.

1. How curved are the links between sensors?      -> link_strength(), curvature_summary()
2. What kind of fault is each fault?              -> fault_types()

Idea behind (1): predict each sensor from all the others, on normal data. First with a
straight-line model. Then let a flexible model (boosted trees) explain whatever the
straight-line model left over. The extra share of the sensor's movement the flexible
model explains is the curved part of the links. Everything is scored on stretches of
time the models did not see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import detectors as det
from . import metrics as M
from .data import Dataset
# The link-residual detector and its models live in detectors.py (next to the registry).
# They are imported here so structure.LinkResidual and structure.blocked_folds keep working.
from .detectors import LinkResidual, _flexible, _straight, blocked_folds  # noqa: F401


def _standardise(X: np.ndarray):
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd > 1e-9, sd, 1.0)


def held_out_predictions(X: np.ndarray, k: int = 5, seed: int = 0):
    """For every sensor: its held-out prediction from the other sensors, straight and straight+flexible.

    Returns (Z, straight, both), all (n, d) in standardised units. Constant sensors are predicted as 0.
    """
    mu, sd = _standardise(X)
    Z = (X - mu) / sd
    n, d = Z.shape
    live = X.std(0) > 1e-9
    lin, both = np.zeros_like(Z), np.zeros_like(Z)
    for test in blocked_folds(n, k):
        train = np.setdiff1d(np.arange(n), test)
        for j in np.flatnonzero(live):
            others = live.copy()
            others[j] = False
            A, y = Z[train][:, others], Z[train, j]
            m1 = _straight().fit(A, y)
            m2 = _flexible(seed).fit(A, y - m1.predict(A))
            p = m1.predict(Z[test][:, others])
            lin[test, j] = p
            both[test, j] = p + m2.predict(Z[test][:, others])
    return Z, lin, both


def _r2(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ((y - p) ** 2).sum(0) / np.where(ss > 0, ss, 1.0)


def link_strength(X: np.ndarray, channels: list[str], k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Per sensor: share of its movement explained by the other sensors, on held-out time blocks.

    straight   explained by a straight-line model
    curved     extra explained once a flexible model is added on top (about 0 if links are straight)
    """
    Z, lin, both = held_out_predictions(X, k, seed)
    r_lin, r_both = _r2(Z, lin), _r2(Z, both)
    out = pd.DataFrame({"straight": r_lin, "curved": r_both - r_lin, "total": r_both}, index=channels)
    out.index.name = "sensor"
    return out


def curvature_summary(X: np.ndarray, channels: list[str], k: int = 5, seed: int = 0,
                      n_boot: int = 500) -> dict:
    """One line per dataset: average straight and curved shares, with a 90% interval on the curved share.

    The interval resamples blocks of time (20 blocks), so it respects that neighbouring readings look alike.
    """
    Z, lin, both = held_out_predictions(X, k, seed)
    table = pd.DataFrame({"straight": _r2(Z, lin), "curved": _r2(Z, both) - _r2(Z, lin)}, index=channels)
    blocks = blocked_folds(len(Z), 20)
    ss = np.where(((Z - Z.mean(0)) ** 2).sum(0) > 0, ((Z - Z.mean(0)) ** 2).sum(0), 1.0) / len(Z)
    gain = ((Z - lin) ** 2 - (Z - both) ** 2) / ss      # per reading, per sensor: drop in squared error
    per_block = np.array([gain[b].mean() for b in blocks])
    rng = np.random.default_rng(seed)
    draws = per_block[rng.integers(0, len(blocks), (n_boot, len(blocks)))].mean(1)
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return {"sensors": len(channels), "readings": len(Z),
            "straight": float(table["straight"].mean()), "curved": float(table["curved"].mean()),
            "curved_low": float(lo), "curved_high": float(hi),
            "sensors_with_curved_over_5pts": int((table["curved"] > 0.05).sum()), "table": table}


# ---------------------------------------------------------------- what kind of fault is each fault?

def _scores(ds: Dataset, make_detector) -> list[np.ndarray]:
    fitted, out = {}, []
    for r in ds.runs:
        key = id(r.fit)
        if key not in fitted:
            m = make_detector().fit(r.fit)
            fitted[key] = (m, m.score_fit(r.fit))
        m, ref = fitted[key]
        out.append(M.normalise(m.score(r.X), ref))
    return out


def sensors_moved(ds: Dataset, far: float = 0.01, window: int = 5, share: float = 0.5) -> list[int]:
    """Per run: how many sensors, each judged alone, are out of range for most of the fault.

    A sensor counts if its own rolling average is beyond its own `far` false-alarm line in more
    than `share` of the fault readings.
    """
    Zs = []
    for r in ds.runs:
        mu, sd = _standardise(r.fit)
        Zs.append(np.abs(det._trailing_mean((r.X - mu) / sd, window)))
    labels = [r.y.astype(bool) for r in ds.runs]
    if ds.calibration == "pooled":
        thr = np.quantile(np.concatenate([z[~y] for z, y in zip(Zs, labels)]), 1 - far, axis=0)
        thrs = [thr] * len(Zs)
    else:
        thrs = [np.quantile(z[~y], 1 - far, axis=0) for z, y in zip(Zs, labels)]
    return [int(((z[y] > t).mean(0) > share).sum()) for z, y, t in zip(Zs, labels, thrs)]


def fault_types(ds: Dataset, far: float = 0.01, caught: float = 0.5, few: int = 3,
                with_curved: bool = True, seed: int = 0) -> pd.DataFrame:
    """One row per kind of fault: which family of checks can see it, and a label.

    level checks    best catch rate of the three per-sensor checks (rolling z-score, EWMA, CUSUM)
    straight links  best of PCA and the straight-link residual detector
    curved links    catch rate of the curved-link residual detector
    shape in time   catch rate of Matrix Profile (each sensor's recent shape against normal shapes)
    sensors moved   sensors that are out of range on their own for most of the fault
    kind            'single sensor'            level checks catch it and at most `few` sensors move
                    'many sensors'             level checks catch it and more sensors move
                    'hidden from level checks' level checks miss it; links or shape in time catch it
                    'hard for all'             nothing here catches half of it
    """
    labels = [r.y for r in ds.runs]
    groups = [r.group for r in ds.runs]
    mode = ds.calibration

    def by_group(make_detector):
        sc = _scores(ds, make_detector)
        thr = M.dr_at_far(sc, labels, far, mode)["threshold"]
        return pd.Series(M.dr_by_group(sc, labels, groups, thr))

    def best(*makers):
        return pd.concat([by_group(m) for m in makers], axis=1).max(axis=1)

    cols = {"level checks": best(*[lambda k=k: det.make(k) for k in ("rolling_z", "ewma", "cusum")]),
            "straight links": best(lambda: det.make("pca"), lambda: LinkResidual(curved=False, seed=seed))}
    if with_curved:
        cols["curved links"] = by_group(lambda: LinkResidual(curved=True, seed=seed))
    if "matrix_profile" in det.available():
        cols["shape in time"] = by_group(lambda: det.make("matrix_profile"))
    out = pd.DataFrame(cols)
    moved = pd.Series(sensors_moved(ds, far), index=groups).groupby(level=0, sort=False).mean()
    out["sensors moved"] = moved.reindex(out.index).round(1)
    others = out.drop(columns=["level checks", "sensors moved"]).max(axis=1)

    def kind(row, other):
        if row["level checks"] >= caught:
            return "single sensor" if row["sensors moved"] <= few else "many sensors"
        return "hidden from level checks" if other >= caught else "hard for all"

    out["kind"] = [kind(row, others[i]) for i, row in out.iterrows()]
    out.index.name = "kind of fault"
    return out


# ---------------------------------------------------------------- saved results, so notebooks rerun fast

def cached(name: str, compute, refresh: bool = False):
    """Run `compute()` once and keep the answer in results/cache/<name>.pkl."""
    import pickle
    from .run import CACHE
    path = CACHE / f"{name}.pkl"
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    out = compute()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def dial_check(curves=(0, 0.25, 0.5, 0.75, 1.0), n: int = 1500, d: int = 8, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Curvature measure on simulated data where the true curvature is known."""
    from .simulate import linked_sensors
    rows = []
    for c in curves:
        for s in seeds:
            r = curvature_summary(linked_sensors(n, d, curve=c, seed=s), [f"s{i}" for i in range(d)], n_boot=50)
            rows.append({"dial": c, "seed": s, "straight": r["straight"], "curved": r["curved"]})
    return pd.DataFrame(rows)


def link_detectors_overall(ds: Dataset, far: float = 0.01, n_boot: int = 1000) -> dict:
    """Catch rate of straight-link vs curved-link residual detectors, with the paired gap and its interval."""
    from . import stats
    labels = [r.y for r in ds.runs]
    sc = {"Straight-link residual": _scores(ds, lambda: LinkResidual(False)),
          "Curved-link residual": _scores(ds, lambda: LinkResidual(True))}
    point = stats.point_dr(sc, labels, far, ds.calibration)
    draws = stats.bootstrap_dr(sc, labels, far, n_boot, mode=ds.calibration)
    gap = draws["Curved-link residual"] - draws["Straight-link residual"]
    return {"straight": point["Straight-link residual"], "curved": point["Curved-link residual"],
            "straight_ci": tuple(np.nanquantile(draws["Straight-link residual"], [0.05, 0.95])),
            "gap": point["Curved-link residual"] - point["Straight-link residual"],
            "gap_ci": tuple(np.nanquantile(gap, [0.05, 0.95]))}
