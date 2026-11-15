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
