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



def movement_by_stretch(runs: list, edges: tuple = (0, 160, 320, 640, 960), sample_hours: float = 0.05) -> pd.DataFrame:
    """How much fault-free runs move in each stretch of a run, next to the data the detectors learned from.

    `runs` are fault-free Runs of equal length that share one fit. Readings are put in units of
    each sensor's normal wiggle on the fit data, so the fit data itself has movement 1.

    movement               average squared distance from the normal level, over runs and sensors
                           (1 = as in the fit data; lower = calmer)
    sensors under 0.8      sensors whose own movement in the stretch is under 0.8
    sensors identical at   sensors that read exactly the same in every run at the first reading
    the start              of the stretch
    """
    mu, sd = runs[0].fit.mean(0), runs[0].fit.std(0)
    z = (np.stack([r.X for r in runs]) - mu) / np.where(sd > 1e-9, sd, 1.0)        # (runs, samples, sensors)
    rows = {}
    for a, b in zip(edges[:-1], edges[1:]):
        per_sensor = (z[:, a:b] ** 2).mean((0, 1))
        rows[f"hours {a * sample_hours:g}-{b * sample_hours:g}"] = {
            "movement": float(per_sensor.mean()), "sensors under 0.8": int((per_sensor < 0.8).sum()),
            "sensors identical at the start": int((np.ptp(z[:, a], axis=0) == 0).sum())}
    out = pd.DataFrame(rows).T.astype({"sensors under 0.8": int, "sensors identical at the start": int})
    out.index.name = "stretch of a run"
    return out


def tidy(value: float) -> str:
    """A number for a table: three decimals at most, no trailing zeros, and never "-0"."""
    return f"{round(float(value), 3) + 0.0:.3f}".rstrip("0").rstrip(".")
