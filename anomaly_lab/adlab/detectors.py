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


def nearest_window_distance(a: np.ndarray, b: np.ndarray, m: int, rows: int = 256) -> np.ndarray:
    """Distance from every length-m stretch of series a to the most similar length-m stretch of series b.

    Plain (non-normalised) Euclidean distance: the same numbers as stumpy.aamp(a, m, T_B=b), to
    about 13 digits, but several times faster when b is long. The nearest stretch is found with
    one matrix product per block of rows (|a - b|^2 = |a|^2 + |b|^2 - 2 a.b; |a|^2 is the same
    along a row, so it is left out). That shortcut loses digits when a distance is tiny, so each
    distance itself is then worked out directly from the two stretches.
    """
    A = np.lib.stride_tricks.sliding_window_view(np.asarray(a, float), m)
    B = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(np.asarray(b, float), m))
    minus_2bt = np.ascontiguousarray(-2.0 * B.T)
    b2 = (B ** 2).sum(1)
    where = np.empty(len(A), int)
    for i in range(0, len(A), rows):
        d2 = A[i:i + rows] @ minus_2bt
        d2 += b2
        where[i:i + rows] = d2.argmin(1)
    return np.sqrt(((A - B[where]) ** 2).sum(1))


class MatrixProfileAB(Detector):
    """Per-channel matrix-profile AB-join against the fit data, non-normalised distances.

    Each length-m subsequence of X ending at t gets its distance to the nearest
    subsequence of the fit data; the score is the largest such distance across channels.
    Non-normalised distances keep level shifts visible. Needs the `stumpy` package (used for
    the fit data's own reference scores).
    """
    name = "Matrix Profile"

    def __init__(self, m: int = 16):
        self.m = m

    def fit(self, X):
        super().fit(X)
        self.fit_z_ = self._z(X)
        return self

    def _channel(self, Z, c, self_join: bool):
        """One channel's distances, or None if the channel is constant in both series."""
        import warnings
        if np.ptp(self.fit_z_[:, c]) == 0 and np.ptp(Z[:, c]) == 0:
            return None
        if self_join:      # nearest neighbour outside the trivial-match zone
            import stumpy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mp = stumpy.aamp(Z[:, c].astype(float), self.m)[:, 0].astype(float)
        else:
            mp = nearest_window_distance(Z[:, c], self.fit_z_[:, c], self.m)
        return np.concatenate([np.full(self.m - 1, mp[0]), mp]) / np.sqrt(self.m)

    def _profile(self, X, self_join: bool):
        Z = self._z(X)
        channels = range(Z.shape[1])
        if self_join:
            parts = [self._channel(Z, c, True) for c in channels]
        else:
            # The channels are independent, and the arithmetic releases Python's lock, so they
            # run side by side on threads. Each thread keeps to one core for its matrix products.
            import os
            from concurrent.futures import ThreadPoolExecutor

            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1, user_api="blas"), \
                    ThreadPoolExecutor(max_workers=min(len(channels), os.cpu_count() or 1)) as pool:
                parts = list(pool.map(lambda c: self._channel(Z, c, False), channels))
        out = np.zeros(len(Z))
        for d in parts:
            if d is not None:
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
    # early_stopping=False: by default scikit-learn switches early stopping on above 10,000 rows,
    # holding out a random tenth of the data, which would make the result depend on the seed and
    # on the data size. Switched off, the trees use no random numbers, so the same data always
    # gives the same model. Below 10,000 rows this changes nothing.
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=10, l2_regularization=1.0, random_state=seed,
                                         early_stopping=False)


class KinkedRidge:
    """A straight-line model allowed corners: each input x also enters as max(0, x - knot) for knots
    at its 10th, 50th and 90th percentiles in the data it is fitted on, then the same ridge regression."""

    def __init__(self, quantiles: tuple = (0.1, 0.5, 0.9)):
        self.quantiles = quantiles

    def _expand(self, A):
        return np.hstack([A] + [np.maximum(0.0, A - k) for k in self.knots_])

    def fit(self, A, y):
        self.knots_ = [np.quantile(A, q, axis=0) for q in self.quantiles]
        self.model_ = _straight().fit(self._expand(A), y)
        return self

    def predict(self, A):
        return self.model_.predict(self._expand(A))


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

    straight_model = staticmethod(_straight)

    def _fit_pair(self, A, y):
        if A.shape[1] == 0:              # no other sensor varies: nothing to predict from
            return None, None
        m1 = self.straight_model().fit(A, y)
        m2 = _flexible(self.seed).fit(A, y - m1.predict(A)) if self.curved else None
        return m1, m2

    @staticmethod
    def _predict(m1, m2, A):
        if m1 is None:                   # predict the sensor's normal level (0 in standardised units)
            return np.zeros(len(A))
        p = m1.predict(A)
        return p if m2 is None else p + m2.predict(A)

    def _separate(self, X):
        """The sensor columns of X, and any extra input columns: given to every sensor's model,
        never predicted and never scored. Plain link detectors have none."""
        return X, np.zeros((len(X), 0))

    def fit(self, X):
        X, E = self._separate(X)
        super().fit(X)
        Z = self._z(X)
        self.live_ = X.std(0) > 1e-9
        self.models_ = {}
        for j in np.flatnonzero(self.live_):
            others = self.live_.copy()
            others[j] = False
            self.models_[j] = (others, *self._fit_pair(np.hstack([Z[:, others], E]), Z[:, j]))
        err = Z - self._held_out(Z, E)
        now, avg = np.abs(err), np.abs(_trailing_mean(err, self.window))
        self.ref_now_ = np.maximum(np.quantile(now, 0.99, axis=0), 1e-9)
        self.ref_avg_ = np.maximum(np.quantile(avg, 0.99, axis=0), 1e-9)
        self.fit_scores_ = self._combine(err)
        return self

    def _folds(self, n):
        """The held-out folds that set each sensor's error scale: k blocks of time."""
        return blocked_folds(n, self.k)

    def _held_out(self, Z, E):
        """Each sensor's prediction on blocks of time its models were not fitted on."""
        pred = np.zeros_like(Z)
        for test in self._folds(len(Z)):
            train = np.setdiff1d(np.arange(len(Z)), test)
            for j, (others, _, _) in self.models_.items():
                A = np.hstack([Z[:, others], E])
                m1, m2 = self._fit_pair(A[train], Z[train, j])
                pred[test, j] = self._predict(m1, m2, A[test])
        return pred

    def _combine(self, err):
        now = np.abs(err) / self.ref_now_
        avg = np.abs(_trailing_mean(err, self.window)) / self.ref_avg_
        return np.maximum(now, avg).max(1)

    def errors(self, X):
        X, E = self._separate(X)
        Z = self._z(X)
        err = np.zeros_like(Z)
        for j, (others, m1, m2) in self.models_.items():
            err[:, j] = Z[:, j] - self._predict(m1, m2, np.hstack([Z[:, others], E]))
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

    Not marked stochastic: the boosted trees are set up to use no random numbers (see
    _flexible), so every seed gives the same scores whatever the size of the fit data.
    """
    name = "Curved-link residual"

    def __init__(self, window: int = 5, k: int = 5, seed: int = 0):
        super().__init__(curved=True, window=window, k=k, seed=seed)


class KinkedLinkResidual(LinkResidual):
    """The straight-link detector with a straight model allowed corners (KinkedRidge), scored the same way."""
    name = "Kinked-link residual"
    straight_model = staticmethod(KinkedRidge)

    def __init__(self, window: int = 5, k: int = 5):
        super().__init__(curved=False, window=window, k=k)
        self.name = type(self).name


class StraightLinkStageTold(LinkResidual):
    """Straight-link residual that is told which stage of a batch each reading belongs to.

    The LAST column of X is not a sensor: it is the stage of each reading, as a whole number
    (a recipe step, or a slice of the batch's progress). It is turned into one yes/no input per
    stage and given to every sensor's straight-line model, so each sensor gets its own level in
    each stage and the same slopes throughout. The stage itself is never predicted or scored.
    A stage that was not in the fit data gets no level of its own.

    Only for data that has a stage column, so it is not in run.METHODS (see batch.py).
    """
    name = "Straight-link residual, stage told"

    def __init__(self, window: int = 5, k: int = 5):
        super().__init__(curved=False, window=window, k=k)
        self.name = type(self).name

    def fit(self, X):
        self.stages_ = np.unique(X[:, -1])
        return super().fit(X)

    def _separate(self, X):
        return X[:, :-1], (X[:, -1][:, None] == self.stages_[None, :]).astype(float)


# ---------------------------------------------------------------- schedule-aware checks (round 3, part 1)
#
# Fixed before computing. Simple, non-deep checks that know how controller outputs behave in normal
# batches. They read from the dataset (passed in context_ like the batch lengths): which columns are
# controller outputs ("outputs"; every column if none is given), the lengths of the batches joined into
# the fitting data ("lengths"), and each fitting batch's group ("fit_groups", e.g. its control strategy).
# Scores are one per reading and use only that reading (and, for the recipe check, its position in its
# batch), so they are alarmed like every other detector. score_fit gives held-out scores: each fitting
# batch scored against the others.

def _segments_of(n, lengths):
    if not lengths or sum(lengths) != n:
        return [(0, n)]
    edges = np.cumsum([0] + list(lengths))
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


class _ScheduleCheck(Detector):
    def __init__(self):
        self.context_ = {}

    def _outputs(self, d):
        cols = self.context_.get("outputs")
        return list(cols) if cols else list(range(d))

    def score_fit(self, X):
        return self.fit_scores_


class UnseenValue(_ScheduleCheck):
    """For each controller output that takes 30 or fewer distinct values in the fitting data: was the
    reading's value ever seen there? A value counts as seen if it lies within 1e-6 of the output's range
    (in the fitting data) of a seen value. Score: 0 if every such output's value was seen; otherwise the
    largest, over those outputs, of the distance to the nearest seen value in units of the output's
    standard deviation in the fitting data (so a value just off a seen one scores less than one far off)."""
    name = "Unseen value check"
    MAX_VALUES = 30
    TOL = 1e-6

    def _learn(self, X):
        cols, seen, scale, tol = [], [], [], []
        for j in self._outputs(X.shape[1]):
            v = np.unique(X[:, j])
            if 1 < len(v) <= self.MAX_VALUES:
                cols.append(j)
                seen.append(v)
                scale.append(max(float(X[:, j].std()), 1e-12))
                tol.append(self.TOL * float(v[-1] - v[0]) + 1e-12)
        return cols, seen, scale, tol

    def _score(self, X, learned):
        cols, seen, scale, tol = learned
        out = np.zeros(len(X))
        for j, v, sc, t in zip(cols, seen, scale, tol):
            x = X[:, j]
            if len(v) == 1:
                gap = np.abs(x - v[0])
            else:
                k = np.clip(np.searchsorted(v, x), 1, len(v) - 1)
                gap = np.minimum(np.abs(x - v[k - 1]), np.abs(x - v[k]))
            out = np.maximum(out, np.where(gap <= t, 0.0, gap / sc))
        return out

    def _held(self, X, a, b):
        rest = np.vstack([X[:a], X[b:]])
        cols, _, scale, tol = self.learned_
        return cols, [np.unique(rest[:, j]) for j in cols], scale, tol

    def fit(self, X):
        super().fit(X)
        self.learned_ = self._learn(X)
        self.columns_ = self.learned_[0]
        segs = _segments_of(len(X), self.context_.get("lengths"))
        if len(segs) > 1:                           # each fitting batch against the values of the others
            self.fit_scores_ = np.concatenate([self._score(X[a:b], self._held(X, a, b)) for a, b in segs])
        else:
            self.fit_scores_ = self._score(X, self.learned_)
        return self

    def score(self, X):
        return self._score(X, self.learned_)


class NearestNormal(_ScheduleCheck):
    """Distance from the reading's controller outputs (standardised on the fitting data) to the nearest
    normal fitting reading (k = 1), or the mean distance to the k nearest (k = 5)."""
    name = "Nearest normal reading (k = 1)"

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k
        self.name = "Nearest normal reading (k = 1)" if k == 1 else f"Nearest normal readings (k = {k})"

    def _tree(self, Z):
        from sklearn.neighbors import KDTree
        return KDTree(Z)

    def _dist(self, tree, Z):
        d, _ = tree.query(Z, k=self.k)
        return d.mean(1)

    def fit(self, X):
        super().fit(X)
        self.cols_ = self._outputs(X.shape[1])
        Z = self._z(X)[:, self.cols_]
        self.tree_ = self._tree(Z)
        segs = _segments_of(len(X), self.context_.get("lengths"))
        if len(segs) > 1:                           # each fitting batch against the others
            self.fit_scores_ = np.concatenate([self._dist(self._tree(np.vstack([Z[:a], Z[b:]])), Z[a:b])
                                               for a, b in segs])
        else:
            self.fit_scores_ = self._dist(self.tree_, Z)
        return self

    def score(self, X):
        return self._dist(self.tree_, self._z(X)[:, self.cols_])


class NearestNormal5(NearestNormal):
    name = "Nearest normal readings (k = 5)"

    def __init__(self):
        super().__init__(k=5)


class RecipeAtHour(_ScheduleCheck):
    """For each controller output: the gap between the reading and the median of the normal recipe batches
    in the fitting data at the same position in the batch (the same batch hour, one reading per step), over
    their spread there (1.4826 x the median absolute deviation across those batches, at least 1% of the
    output's standard deviation over the whole fitting data, so outputs that every recipe batch holds at the
    same value still get a finite scale). Score: the largest scaled gap across outputs. Recipe batches are
    the fitting batches whose group is "recipe" (all of them if no groups are given). Past the hour where
    fewer than 3 recipe batches still run, the last hour with 3 is used. On batches not run by the recipe
    (operators' random steps, the Raman-based controller) the outputs leave the recipe by design, so such
    batches score high even when nothing is wrong."""
    name = "Recipe at this hour"
    MIN_BATCHES = 3
    FLOOR = 0.01

    def _profile(self, batches, floor):
        n = max(len(b) for b in batches)
        med, spread = [], []
        for i in range(n):
            rows = [b[i] for b in batches if len(b) > i]
            if len(rows) < self.MIN_BATCHES:
                break
            R = np.vstack(rows)
            m = np.median(R, axis=0)
            med.append(m)
            spread.append(np.maximum(1.4826 * np.median(np.abs(R - m), axis=0), floor))
        return np.array(med), np.array(spread)

    def _score(self, Xo, prof):
        med, spread = prof
        idx = np.minimum(np.arange(len(Xo)), len(med) - 1)
        return (np.abs(Xo - med[idx]) / spread[idx]).max(1)

    def fit(self, X):
        super().fit(X)
        self.cols_ = self._outputs(X.shape[1])
        Xo = X[:, self.cols_]
        floor = self.FLOOR * np.maximum(Xo.std(0), 1e-12)
        segs = _segments_of(len(X), self.context_.get("lengths"))
        groups = self.context_.get("fit_groups")
        use_groups = bool(groups) and len(groups) == len(segs)
        recipe = [i for i in range(len(segs)) if not use_groups or groups[i] == "recipe"]
        batches = [Xo[a:b] for a, b in segs]
        self.profile_ = self._profile([batches[i] for i in recipe], floor)
        scores = []
        for i, b in enumerate(batches):               # held out: a recipe batch against the other recipe batches
            if i in recipe and len(recipe) > self.MIN_BATCHES:
                prof = self._profile([batches[j] for j in recipe if j != i], floor)
            else:
                prof = self.profile_
            scores.append(self._score(b, prof))
        self.fit_scores_ = np.concatenate(scores)
        return self

    def score(self, X):
        return self._score(X[:, self.cols_], self.profile_)


# ---------------------------------------------------------------- round 3, part 4: balanced folds

class _BalancedFolds:
    """The error scale set on folds that each hold out one run of every group (e.g. one run of every operating
    mode), so every group is always in the fitted part. Needs the fitting runs' lengths and groups (context_);
    without them, the k blocks of time as before."""

    def _folds(self, n):
        segs = _segments_of(n, self.context_.get("lengths"))
        groups = self.context_.get("fit_groups")
        if not groups or len(groups) != len(segs):
            return blocked_folds(n, self.k)
        fold, seen = np.empty(n, int), {}
        for (a, b), g in zip(segs, groups):
            fold[a:b] = seen.get(g, 0)
            seen[g] = seen.get(g, 0) + 1
        return [np.flatnonzero(fold == f) for f in range(max(seen.values()))]


class StraightLinkBalanced(_BalancedFolds, StraightLinkResidual):
    name = "Straight-link residual, balanced folds"

    def __init__(self, window: int = 5, k: int = 5):
        super().__init__(window=window, k=k)
        self.name = type(self).name
        self.context_ = {}


class StraightLinkStageToldBalanced(_BalancedFolds, StraightLinkStageTold):
    name = "Straight-link residual, stage told, balanced folds"

    def __init__(self, window: int = 5, k: int = 5):
        super().__init__(window=window, k=k)
        self.name = type(self).name
        self.context_ = {}


REGISTRY = {
    "rolling_z": RollingZ,
    "ewma": EWMA,
    "cusum": CUSUM,
    "pca": PCA,
    "iforest": IsolationForestW,
    "matrix_profile": MatrixProfileAB,
    "link_straight": StraightLinkResidual,
    "link_curved": CurvedLinkResidual,
    "link_straight_told": StraightLinkStageTold,
    "link_kinked": KinkedLinkResidual,
    "unseen_value": UnseenValue,
    "nearest_1": NearestNormal,
    "nearest_5": NearestNormal5,
    "recipe_hour": RecipeAtHour,
}
SCHEDULE_CHECKS = ["unseen_value", "nearest_1", "nearest_5", "recipe_hour"]
REGISTRY.update({"link_straight_balanced": StraightLinkBalanced, "link_straight_told_balanced": StraightLinkStageToldBalanced})


def make(key: str, **params) -> Detector:
    return REGISTRY[key](**params)


def available() -> list[str]:
    """Registry keys whose optional dependencies are installed."""
    keys = list(REGISTRY)
    try:
        import stumpy  # noqa: F401
    except ImportError:
        keys.remove("matrix_profile")
    from . import deep
    if not deep.torch_available():
        keys = [k for k in keys if k not in deep.MODELS]
    return keys


# The deep models and the straight forecaster live in deep.py, which adds them to REGISTRY when
# imported. PyTorch is only needed to fit a deep model, not to import either module.
from . import deep as _deep  # noqa: E402,F401
