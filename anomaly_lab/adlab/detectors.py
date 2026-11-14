"""Baseline detectors.

Per-sensor checks (rolling z, EWMA, CUSUM), PCA, Isolation Forest, Matrix Profile, and the
link-residual detectors that predict each sensor from the others (straight and curved).

Every detector is fitted on normal data and returns one score per time step:

    det = make("pca").fit(X_fit)    # X_fit: (n, d) normal data
    s = det.score(X)                # (T,) higher = more anomalous

Scores at time t use only samples up to t, so every detector here could run online.
"""
from __future__ import annotations

import numpy as np


def _standardiser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd > 1e-9, sd, 1.0)  # constant channels keep their raw units


def _trailing_mean(Z: np.ndarray, w: int) -> np.ndarray:
    """Mean of the last w rows (fewer at the start), per column."""
    if w <= 1:
        return Z
    c = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z, axis=0)])
    end = np.arange(1, len(Z) + 1)
    start = np.maximum(end - w, 0)
    return (c[end] - c[start]) / (end - start)[:, None]


def _windows(Z: np.ndarray, w: int) -> np.ndarray:
    """Row t = the w samples ending at t, flattened. The first rows repeat sample 0 as padding."""
    padded = np.vstack([np.repeat(Z[:1], w - 1, axis=0), Z])
    view = np.lib.stride_tricks.sliding_window_view(padded, (w, Z.shape[1]))[:, 0]
    return view.reshape(len(Z), -1)


class Detector:
    name = "base"
    stochastic = False

    def fit(self, X: np.ndarray) -> "Detector":
        self.mu_, self.sd_ = _standardiser(X)
        return self

    def _z(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / self.sd_

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_fit(self, X: np.ndarray) -> np.ndarray:
        """In-sample scores on the fit data, used as the reference scale for normalising."""
        return self.score(X)


class RollingZ(Detector):
    """Largest absolute rolling-mean z-score across channels."""
    name = "Rolling z-score"

    def __init__(self, window: int = 5):
        self.window = window

    def score(self, X):
        return np.abs(_trailing_mean(self._z(X), self.window)).max(1)


class EWMA(Detector):
    """EWMA control chart per channel; score = largest |EWMA| in units of its steady-state s.d."""
    name = "EWMA chart"

    def __init__(self, lam: float = 0.2):
        self.lam = lam

    def score(self, X):
        Z = self._z(X)
        e = np.zeros(Z.shape[1])
        out = np.empty(len(Z))
        for t, z in enumerate(Z):
            e = self.lam * z + (1 - self.lam) * e
            out[t] = np.abs(e).max()
        return out / np.sqrt(self.lam / (2 - self.lam))


class CUSUM(Detector):
    """Two-sided CUSUM per channel on z-scores with slack k; score = largest statistic."""
    name = "CUSUM chart"

    def __init__(self, k: float = 0.5):
        self.k = k

    def score(self, X):
        Z = self._z(X)
        hi = np.zeros(Z.shape[1])
        lo = np.zeros(Z.shape[1])
        out = np.empty(len(Z))
        for t, z in enumerate(Z):
            hi = np.maximum(0.0, hi + z - self.k)
            lo = np.maximum(0.0, lo - z - self.k)
            out[t] = max(hi.max(), lo.max())
        return out


class PCA(Detector):
    """PCA monitoring: Hotelling T^2 inside the model plane and SPE (squared residual) outside it.

    stat="combined" scores each sample by the larger of T^2 and SPE, each divided by its own
    99th percentile on the fit data.
    """
    name = "PCA (T2 + SPE)"

    def __init__(self, var: float = 0.90, stat: str = "combined"):
        self.var, self.stat = var, stat

    def fit(self, X):
        super().fit(X)
        Z = self._z(X)
        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        ev = s ** 2 / (len(Z) - 1)
        k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), self.var) + 1)
        self.k_ = max(1, min(k, Z.shape[1] - 1))  # leave at least one direction for SPE
        self.P_, self.lam_ = vt[: self.k_].T, np.maximum(ev[: self.k_], 1e-12)
        t2, spe = self._stats(X)
        self.ref_ = np.quantile(t2, 0.99), np.quantile(spe, 0.99)
        return self

    def _stats(self, X):
        Z = self._z(X)
        T = Z @ self.P_
        return (T ** 2 / self.lam_).sum(1), ((Z - T @ self.P_.T) ** 2).sum(1)

    def residuals(self, X):
        """Per-channel squared residuals, for asking which channel broke the model."""
        Z = self._z(X)
        return (Z - Z @ self.P_ @ self.P_.T) ** 2

    def score(self, X):
        t2, spe = self._stats(X)
        if self.stat == "t2":
            return t2
        if self.stat == "spe":
            return spe
        ratio = lambda v, ref: v / ref if ref > 0 else np.zeros_like(v)  # noqa: E731
        return np.maximum(ratio(t2, self.ref_[0]), ratio(spe, self.ref_[1]))


class IsolationForestW(Detector):
    """Isolation Forest on sliding windows of z-scored data."""
    name = "Isolation Forest"
    stochastic = True

    def __init__(self, window: int = 10, n_estimators: int = 100, seed: int = 0):
        self.window, self.n_estimators, self.seed = window, n_estimators, seed

    def fit(self, X):
        from sklearn.ensemble import IsolationForest
        super().fit(X)
        self.model_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed)
        self.model_.fit(_windows(self._z(X), self.window)[self.window - 1:])
        return self

    def score(self, X):
        return -self.model_.score_samples(_windows(self._z(X), self.window))


class MatrixProfileAB(Detector):
    """Per-channel matrix-profile AB-join against the fit data, non-normalised distances.

    Each length-m subsequence of X ending at t gets its distance to the nearest
    subsequence of the fit data; the score is the largest such distance across channels.
    Non-normalised distances keep level shifts visible. Needs the `stumpy` package.
    """
    name = "Matrix Profile"

    def __init__(self, m: int = 16):
        self.m = m

    def fit(self, X):
        super().fit(X)
        self.fit_z_ = self._z(X)
        return self

    def _profile(self, X, self_join: bool):
        import warnings

        import stumpy
        Z = self._z(X)
        out = np.zeros(len(Z))
        for c in range(Z.shape[1]):
            if np.ptp(self.fit_z_[:, c]) == 0 and np.ptp(Z[:, c]) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self_join:  # nearest neighbour outside the trivial-match zone
                    mp = stumpy.aamp(Z[:, c].astype(float), self.m)[:, 0]
                else:
                    mp = stumpy.aamp(Z[:, c].astype(float), self.m,
                                     T_B=self.fit_z_[:, c].astype(float), ignore_trivial=False)[:, 0]
            mp = mp.astype(float)
            d = np.concatenate([np.full(self.m - 1, mp[0]), mp]) / np.sqrt(self.m)
            out = np.maximum(out, d)
        return out

    def score(self, X):
        return self._profile(X, self_join=False)

    def score_fit(self, X):
        # Joining the fit data with itself would match every window to itself (distance 0).
        return self._profile(X, self_join=True)


# ---------------------------------------------------------------- links between sensors
#
# These live here, not in structure.py, so that structure.py can import this module and this
# module never has to import structure.py back.

def blocked_folds(n: int, k: int = 5) -> list[np.ndarray]:
    """k contiguous blocks of time. Shuffled folds would leak, because neighbours in time look alike."""
    return [idx for idx in np.array_split(np.arange(n), k) if len(idx)]


def _straight():
    from sklearn.linear_model import RidgeCV
    return RidgeCV(alphas=np.logspace(-3, 3, 13))


def _flexible(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=10, l2_regularization=1.0, random_state=seed)


class LinkResidual(Detector):
    """Predict each sensor from the others; score = largest surprise across sensors.

    curved=False uses straight-line models only; curved=True adds a flexible model on top.
    A surprise is either one reading far off its prediction, or the average of the last
    `window` readings being off. Both are measured against held-out errors on the fit data.
    """
    name = "Link residual"

    def __init__(self, curved: bool = False, window: int = 5, k: int = 5, seed: int = 0):
        self.curved, self.window, self.k, self.seed = curved, window, k, seed
        self.name = "Curved-link residual" if curved else "Straight-link residual"

    def _fit_pair(self, A, y):
        if A.shape[1] == 0:              # no other sensor varies: nothing to predict from
            return None, None
        m1 = _straight().fit(A, y)
        m2 = _flexible(self.seed).fit(A, y - m1.predict(A)) if self.curved else None
        return m1, m2

    @staticmethod
    def _predict(m1, m2, A):
        if m1 is None:                   # predict the sensor's normal level (0 in standardised units)
            return np.zeros(len(A))
        p = m1.predict(A)
        return p if m2 is None else p + m2.predict(A)

    def fit(self, X):
        super().fit(X)
        Z = self._z(X)
        self.live_ = X.std(0) > 1e-9
        self.models_ = {}
        for j in np.flatnonzero(self.live_):
            others = self.live_.copy()
            others[j] = False
            self.models_[j] = (others, *self._fit_pair(Z[:, others], Z[:, j]))
        err = Z - self._held_out(Z)
        now, avg = np.abs(err), np.abs(_trailing_mean(err, self.window))
        self.ref_now_ = np.maximum(np.quantile(now, 0.99, axis=0), 1e-9)
        self.ref_avg_ = np.maximum(np.quantile(avg, 0.99, axis=0), 1e-9)
        self.fit_scores_ = self._combine(err)
        return self

    def _held_out(self, Z):
        """Each sensor's prediction on blocks of time its models were not fitted on."""
        pred = np.zeros_like(Z)
        for test in blocked_folds(len(Z), self.k):
            train = np.setdiff1d(np.arange(len(Z)), test)
            for j, (others, _, _) in self.models_.items():
                m1, m2 = self._fit_pair(Z[train][:, others], Z[train, j])
                pred[test, j] = self._predict(m1, m2, Z[test][:, others])
        return pred

    def _combine(self, err):
        now = np.abs(err) / self.ref_now_
        avg = np.abs(_trailing_mean(err, self.window)) / self.ref_avg_
        return np.maximum(now, avg).max(1)

    def errors(self, X):
        Z = self._z(X)
        err = np.zeros_like(Z)
        for j, (others, m1, m2) in self.models_.items():
            err[:, j] = Z[:, j] - self._predict(m1, m2, Z[:, others])
        return err

    def score(self, X):
        return self._combine(self.errors(X))

    def score_fit(self, X):
        return self.fit_scores_          # held-out, so the reference scale is honest


class StraightLinkResidual(LinkResidual):
    """LinkResidual with straight-line models only, as one registry entry."""
    name = "Straight-link residual"

    def __init__(self, window: int = 5, k: int = 5):
        super().__init__(curved=False, window=window, k=k)


class CurvedLinkResidual(LinkResidual):
    """LinkResidual with the flexible model added, as one registry entry.

    Not marked stochastic: with fewer than 10,000 fit rows the boosted trees give the same
    answer for every seed (scikit-learn only draws random numbers above that size, for early
    stopping). Fitted on more rows than that, the seed starts to matter.
    """
    name = "Curved-link residual"

    def __init__(self, window: int = 5, k: int = 5, seed: int = 0):
        super().__init__(curved=True, window=window, k=k, seed=seed)


REGISTRY = {
    "rolling_z": RollingZ,
    "ewma": EWMA,
    "cusum": CUSUM,
    "pca": PCA,
    "iforest": IsolationForestW,
    "matrix_profile": MatrixProfileAB,
    "link_straight": StraightLinkResidual,
    "link_curved": CurvedLinkResidual,
}


def make(key: str, **params) -> Detector:
    return REGISTRY[key](**params)


def available() -> list[str]:
    """Registry keys whose optional dependencies are installed."""
    keys = list(REGISTRY)
    try:
        import stumpy  # noqa: F401
    except ImportError:
        keys.remove("matrix_profile")
    return keys
