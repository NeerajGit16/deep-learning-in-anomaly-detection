"""A first, small simulator: linked sensors with a dial for how curved the links are.

Used to check that the curvature measure in structure.py responds to curvature and nothing
else. The full simulator (modes, lags, a fault library) builds on this.
"""
from __future__ import annotations

import numpy as np


def _curved(f: np.ndarray, kind: int) -> np.ndarray:
    """A curved function of the hidden drivers, scaled to roughly unit variance."""
    a, b = f[:, 0], f[:, 1]
    out = [a ** 2 - 1, np.tanh(2 * a) * 1.3, a * b, np.abs(b) * 1.6 - 1.3][kind % 4]
    return out / out.std()


def linked_sensors(n: int = 2000, d: int = 8, curve: float = 0.0, noise: float = 0.3,
                   memory: float = 0.9, seed: int = 0) -> np.ndarray:
    """n readings of d sensors driven by two slowly moving hidden factors.

    curve=0: every sensor is a straight-line mix of the factors.
    curve=1: every sensor after the first two is a curved function of them.
    In between, each sensor is a blend. `memory` sets how slowly the factors move
    (0 = no memory, close to 1 = slow), and `noise` is each sensor's own noise.
    The first two sensors always track the factors directly, so the curved sensors
    are predictable from other sensors and not only from hidden quantities.
    """
    rng = np.random.default_rng(seed)
    f = np.zeros((n, 2))
    shocks = rng.normal(size=(n, 2)) * np.sqrt(1 - memory ** 2)
    for t in range(1, n):
        f[t] = memory * f[t - 1] + shocks[t]
    f = f / f.std(0)
    X = np.empty((n, d))
    for j in range(d):
        w = rng.normal(size=2)
        straight = f @ w / np.linalg.norm(w)
        if j < 2:
            signal = f[:, j]
        else:
            signal = (1 - curve) * straight + curve * _curved(f, j)
        X[:, j] = signal / max(signal.std(), 1e-9) + noise * rng.normal(size=n)
    return X


def delayed_sensors(n: int = 3000, delay: int = 3, curve: float = 0.0, noise: float = 0.3, seed: int = 0) -> np.ndarray:
    """Three sensors with one link that takes `delay` readings to arrive. A check for the lag option.

    Sensor 0 is a driver that changes from reading to reading, so its reading now says nothing
    about its reading `delay` steps ago. Sensor 1 follows the driver `delay` readings later:
    as a straight-line copy (curve=0), as a curved function of it (curve=1), or a blend.
    Sensor 2 is unrelated noise. Looking only at readings taken at the same moment, sensor 1
    appears unlinked to anything.
    """
    rng = np.random.default_rng(seed)
    driver = rng.normal(size=n)
    past = np.r_[rng.normal(size=delay), driver[:-delay]] if delay else driver
    signal = (1 - curve) * past + curve * (past ** 2 - 1) / np.sqrt(2)
    follower = signal / max(signal.std(), 1e-9) + noise * rng.normal(size=n)
    return np.c_[driver, follower, rng.normal(size=n)]


# ---------------------------------------------------------------- follow-up round, part D: the recipe generator
#
# Built from notebook 11's recipe (section 5), to resemble what makes the autoencoder win on IndPenSim. Its
# value is in turning that ingredient up and down, not in reproducing the win.
#
# Fixed before computing:
#   sensors      6 controller outputs that step through a fixed recipe schedule (4 to 8 values each, values
#                up and down, steps at different points of the batch), and 6 measurements
#   the dial     each output = its best straight line over the batch + dial x (its recipe steps - that line).
#                Dial 0: every output is a straight ramp, so their joint values lie on one line. Dial 1: the
#                recipe's islands. Above 1: the islands further from the line.
#   measurements straight-line mixes, each in its own proportions, of the batch's progress and of lagged
#                responses to two outputs (the feed and the air flow), so that the only bend in the data is the
#                outputs' islands; plus noise (2% of each measurement's level) and a slow, harmless drift (an
#                offset that wanders from batch to batch)
#   batches      the same schedule on each batch's own clock; lengths vary by about 10%
#   faults       one stepped output held, for about 8% of the batch, at a value inside its normal range that
#                it does not take at that time, at least a quarter of its range away from its value then:
#                half the faults "between steps" (midway between two of its own values), half "wrong step"
#                (its value at another stage). The measurements respond (through the lags) only when the
#                output held is the feed or the air flow.

REC_OUTPUTS = ["feed", "air", "oil", "water", "pressure", "precursor"]
REC_MEASURES = ["measurement 1", "measurement 2", "measurement 3", "measurement 4", "measurement 5", "measurement 6"]


def _recipe(rng, n_outputs: int = 6):
    """The fixed recipe: for each output, step points in (0, 1) and its values (up and down)."""
    sched = []
    for _ in range(n_outputs):
        k = int(rng.integers(4, 9))
        points = np.sort(rng.uniform(0.04, 0.96, k - 1))
        palette = np.sort(rng.uniform(1, 10, k))
        values = rng.permutation(palette)
        values[0] = palette[0]                                   # starts low, as a recipe does
        sched.append((points, values))
    return sched


def _steps(points, values, p):
    return values[np.searchsorted(points, p, side="right")]


def _outputs(sched, p, dial):
    """(readings x outputs) at progress p, for a given dial."""
    out = np.empty((len(p), len(sched)))
    grid = np.linspace(0, 1, 2001)
    for k, (points, values) in enumerate(sched):
        step_g = _steps(points, values, grid)
        slope, icept = np.polyfit(grid, step_g, 1)               # the straight line nearest the steps
        line = icept + slope * p
        out[:, k] = line + dial * (_steps(points, values, p) - line)
    return out


def _lag(x, tau):
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = y[t - 1] + (x[t] - y[t - 1]) / tau
    return y


def recipe_batches(n_batches: int, dial: float = 1.0, seed: int = 0, readings: int = 250, faulty: bool = False,
                   noise: float = 0.02, recipe_seed: int = 7, fault_size: float = 1.0) -> list[dict]:
    """Batches of the recipe generator: [{"batch", "X" (readings x 12: outputs then measurements), "fault"
    (0/1 per reading), "stage" (tenth of the batch), "faulty", "fault kind", "output held"}].

    Every batch follows the recipe fixed by recipe_seed; seed draws everything else. faulty=True gives every
    batch one fault (fixed design above). fault_size (round 3) moves the held value only that share of the way
    from the output's value at the start of the stretch towards the value chosen above (1 = as designed); the same
    faults are drawn whatever the size."""
    rec_rng = np.random.default_rng(recipe_seed)
    sched = _recipe(rec_rng)
    mix = rec_rng.uniform(0.3, 1.0, (len(REC_MEASURES), 3))     # each measurement's mix of progress, feed and air
    rng = np.random.default_rng(seed)
    drift = np.cumsum(rng.normal(0, 0.01, (n_batches, len(REC_MEASURES))), axis=0)
    out = []
    for b in range(n_batches):
        n = int(round(readings * (1 + 0.1 * rng.standard_normal())))
        n = max(n, readings // 2)
        p = (np.arange(n) + 0.5) / n
        U = _outputs(sched, p, dial)
        fault = np.zeros(n, int)
        kind, held = None, None
        if faulty:
            held = int(rng.integers(0, len(sched)))
            length = max(int(0.08 * n), 3)
            start = int(rng.integers(int(0.1 * n), int(0.9 * n) - length))
            stretch = slice(start, start + length)
            u = U[:, held]
            lo, hi = float(u.min()), float(u.max())
            now = float(u[start])
            values = np.unique(np.round(u, 9))
            if rng.random() < 0.5:                               # between two of its own values
                kind = "between steps"
                candidates = (values[:-1] + values[1:]) / 2 if len(values) <= 30 else np.linspace(lo, hi, 41)[1:-1]
            else:                                                # its value at another stage of the batch
                kind = "wrong step"
                candidates = values
            candidates = candidates[np.abs(candidates - now) >= 0.25 * (hi - lo)]
            value = float(rng.choice(candidates)) if len(candidates) else (lo if now - lo > hi - now else hi)
            U[stretch, held] = now + fault_size * (value - now)
            fault[stretch] = 1
        feed, air = _lag(U[:, 0], 6.0), _lag(U[:, 1], 4.0)
        M = np.column_stack([a * p + c * feed / 5.5 + f * air / 5.5 for a, c, f in mix])
        M = M + noise * M.mean(0) * rng.standard_normal(M.shape) + drift[b]
        out.append({"batch": b, "X": np.hstack([U, M]), "fault": fault, "stage": np.minimum(10 * np.arange(n) // n, 9),
                    "faulty": faulty, "fault kind": kind, "output held": REC_OUTPUTS[held] if held is not None else None})
    return out
