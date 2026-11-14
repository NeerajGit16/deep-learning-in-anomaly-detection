"""Summary tables for getting to know a dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset


def overview(ds: Dataset) -> pd.DataFrame:
    """One row per kind of fault: how many runs, how long they are, how much of each is faulty."""
    rows = []
    for r in ds.runs:
        rows.append({"kind of fault": r.group, "minutes scored": len(r.y) * ds.sample_seconds / 60,
                     "% of samples faulty": 100 * r.y.mean()})
    df = pd.DataFrame(rows)
    out = df.groupby("kind of fault", sort=False).agg(
        runs=("minutes scored", "size"),
        **{"avg minutes per run": ("minutes scored", "mean"),
           "avg % faulty": ("% of samples faulty", "mean")})
    return out.round(1)


def fingerprint(ds: Dataset, absolute: bool = True) -> pd.DataFrame:
    """How far each sensor moves during each kind of fault.

    Measured in units of that sensor's normal wiggle (its standard deviation on the fit data),
    averaged over the faulty samples and then over the runs of that kind.
    """
    rows = {}
    for r in ds.runs:
        mu, sd = r.fit.mean(0), r.fit.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        z = (r.X[r.y == 1] - mu) / sd
        shift = np.abs(z).mean(0) if absolute else z.mean(0)
        rows.setdefault(r.group, []).append(shift)
    table = pd.DataFrame({g: np.mean(v, axis=0) for g, v in rows.items()}, index=ds.channels).T
    table.index.name = "kind of fault"
    return table


def correlations(X: np.ndarray, channels: list[str]) -> pd.DataFrame:
    """Correlation between every pair of sensors: +1 move together, -1 opposite, 0 unrelated."""
    sd = X.std(0)
    keep = sd > 1e-9
    c = np.full((X.shape[1], X.shape[1]), np.nan)
    c[np.ix_(keep, keep)] = np.corrcoef(X[:, keep], rowvar=False)
    return pd.DataFrame(c, index=channels, columns=channels)
