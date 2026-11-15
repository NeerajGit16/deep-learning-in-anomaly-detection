"""Two questions about a dataset, answered before any deep model is trained.

1. How curved are the links between sensors?      -> link_strength(), curvature_summary(),
                                                     curvature_by_run(), curvature_by_lag()
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
from . import progress
from .data import Dataset, stretches
# The link-residual detector and its models live in detectors.py (next to the registry).
# They are imported here so structure.LinkResidual and structure.blocked_folds keep working.
from .detectors import LinkResidual, _flexible, _straight, blocked_folds  # noqa: F401


def _standardise(X: np.ndarray):
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd > 1e-9, sd, 1.0)


def _lagged(Z: np.ndarray, lags: tuple) -> tuple[np.ndarray, np.ndarray]:
    """One run as (targets, features), from the first reading that has all its past readings.

    Features are every sensor's reading now, then every sensor's reading at each lag back, side
    by side: columns 0..d-1 are "now", d..2d-1 the first lag, and so on.
    """
    back = max(lags, default=0)
    return Z[back:], np.hstack([Z[back - lag: len(Z) - lag] for lag in (0, *lags)])


def held_out_by_run(runs: list, k: int = 5, seed: int = 0, lags: tuple = (), verbose: bool = True,
                    what: str = "curvature measure", folds: list | None = None, modes: list | None = None,
                    told: str = "levels") -> dict:
    """For every sensor: its held-out prediction from the OTHER sensors, straight and straight+flexible.

    runs   list of (readings, sensors) arrays: separate recordings of the same sensors. They are
           put on one scale together (pooled mean and spread).
    lags   which past readings of the other sensors the models may also use, in readings back,
           e.g. (1, 2, 5). () means only the reading taken at the same moment. A sensor's own
           past is never used. Lagged readings never reach across two runs.

    Held-out data: every run is cut into k blocks of time; fold f holds out the f-th block of
    every run. So every run (every setting, if runs were made at different settings) is in the
    training data of every fold. With lags, training readings closer than the longest lag to a
    held-out block of the same run are left out, so no reading is on both sides.

    folds  instead of the k blocks of time: one array per run giving, for every reading, the
           number of the fold that holds it out (see staggered_folds). No gap is left around
           held-out stretches then, so with lags a training reading may use, as an input, another
           sensor's reading from a held-out stretch. The held-out readings of the sensor being
           predicted are still never seen in training.
    modes  one array per run giving the operating mode (setting) of every reading, as whole
           numbers. If given, the straight-line model is "told the mode", in one of two ways,
           and the flexible model is given the mode as one more input.
    told   "levels": one straight model with its own level for every mode (the mode enters as
           one yes/no input per mode) and the same slopes in all modes. Stable even when a mode
           has little training data.
           "separate": a straight model of its own for each mode, level and slopes. The stronger
           form, but it needs enough varied training data in every mode: if a mode's training
           data is one short steady stretch, its slopes are fitted to noise and can be far off
           on another day. A held-out reading whose mode is not in the training data falls
           back on one straight model for all modes.

    Returns a dict of arrays over all kept readings: "Z" (standardised targets), "straight" and
    "both" (predictions), "run" (which run each reading came from), "position" (where in its run,
    counted after the first max(lags) readings are dropped). Constant sensors are predicted as 0.
    """
    lags = tuple(lags)
    stacked = np.vstack(runs)
    mu, sd = _standardise(stacked)
    live = stacked.std(0) > 1e-9
    d, back = stacked.shape[1], max(lags, default=0)
    targets, features, run_id, fold_id = [], [], [], []
    for r, X in enumerate(runs):
        y, F = _lagged((X - mu) / sd, lags)
        if folds is None:
            fold = np.empty(len(y), int)
            for f, idx in enumerate(blocked_folds(len(y), k)):
                fold[idx] = f
        else:
            fold = np.asarray(folds[r])[back:]
        targets.append(y), features.append(F), run_id.append(np.full(len(y), r)), fold_id.append(fold)
    Z, F = np.vstack(targets), np.vstack(features)
    run_id, fold_id = np.concatenate(run_id), np.concatenate(fold_id)
    position = np.concatenate([np.arange(len(t)) for t in targets])
    mode = None if modes is None else np.concatenate([np.asarray(m)[back:] for m in modes])
    if told not in ("levels", "separate"):
        raise ValueError("told must be 'levels' or 'separate'")
    yes_no = None if mode is None else (mode[:, None] == np.unique(mode)[None, :]).astype(float)
    fold_numbers = range(k) if folds is None else sorted(set(fold_id))
    sensor_of_column = np.tile(np.arange(d), 1 + len(lags))
    lin, both = np.zeros_like(Z), np.zeros_like(Z)
    steps = progress.track(range(len(fold_numbers) * int(live.sum())), f"{what} (models fitted)", verbose=verbose)
    for f in fold_numbers:
        test = fold_id == f
        if not test.any():
            continue
        train = ~test
        if back and folds is None:      # leave a gap of `back` readings around each run's held-out block
            for r in np.unique(run_id[test]):
                held = position[test & (run_id == r)]
                near = (run_id == r) & (position >= held.min() - back) & (position <= held.max() + back)
                train &= ~near
        for j in np.flatnonzero(live):
            cols = live[sensor_of_column] & (sensor_of_column != j)
            A, y, B = F[train][:, cols], Z[train, j], F[test][:, cols]
            if mode is None:
                m1 = _straight().fit(A, y)
                fitted, p = m1.predict(A), m1.predict(B)
            elif told == "levels":
                m1 = _straight().fit(np.c_[A, yes_no[train]], y)
                fitted, p = m1.predict(np.c_[A, yes_no[train]]), m1.predict(np.c_[B, yes_no[test]])
            else:
                fitted, p = _straight_by_mode(A, y, mode[train], B, mode[test])
            if mode is not None:
                A, B = np.c_[A, mode[train]], np.c_[B, mode[test]]
            m2 = _flexible(seed).fit(A, y - fitted)
            lin[test, j] = p
            both[test, j] = p + m2.predict(B)
            next(steps, None)
    for _ in steps:      # let the bar finish and close
        pass
    return {"Z": Z, "straight": lin, "both": both, "run": run_id, "position": position}


def _straight_by_mode(A, y, mode, B, mode_b, fewest: int = 30):
    """A straight-line model per mode. Returns its predictions for the training rows A and the rows B.

    Modes with fewer than `fewest` training rows, and rows of B whose mode is not in the
    training data, use one straight model fitted on all the training rows.
    """
    overall = _straight().fit(A, y)
    fitted, p = overall.predict(A), overall.predict(B)
    for m in np.unique(mode):
        rows = mode == m
        if rows.sum() < fewest:
            continue
        own = _straight().fit(A[rows], y[rows])
        fitted[rows] = own.predict(A[rows])
        if (mode_b == m).any():
            p[mode_b == m] = own.predict(B[mode_b == m])
    return fitted, p


def staggered_folds(labels: list, between: int = -1) -> tuple[list, list]:
    """Folds that hold out whole stretches at one setting, with every setting kept in training.

    labels   one array per run: the setting of every reading (whole numbers; `between` marks
             readings taken while the setting is being changed).
    For runs that each pass through many settings. Every unbroken stretch at one setting is held
    out whole. There are as many folds as runs, and fold f holds out setting c's stretches in
    run r when (c + r) leaves remainder f on division by the number of runs. So when a setting
    is held out in one run, its stretches in the other runs are in the training data, and every
    run is in the training data of every fold. In-between stretches are dealt round the folds
    in turn.

    Returns (folds, units): per run, the fold of every reading, and a number for the stretch
    every reading belongs to (unique across runs), for resampling stretches.
    """
    n = len(labels)
    folds, units, count = [], [], 0
    for r, label in enumerate(labels):
        label = np.asarray(label)
        fold, unit = np.empty(len(label), int), np.empty(len(label), int)
        for i, (a, b) in enumerate(stretches(label)):
            fold[a:b] = ((i if label[a] == between else int(label[a])) + r) % n
            unit[a:b] = count
            count += 1
        folds.append(fold), units.append(unit)
    return folds, units


def unseen_share(labels: list, folds: list, between: int = -1) -> float:
    """Share of held-out readings at a setting that is nowhere in that fold's training data."""
    label, fold = np.concatenate(labels), np.concatenate(folds)
    at = label != between
    unseen = np.zeros(len(label), bool)
    for f in np.unique(fold):
        seen = np.unique(label[(fold != f) & at])
        unseen |= (fold == f) & at & ~np.isin(label, seen)
    return float(unseen[at].mean())


def held_out_predictions(X: np.ndarray, k: int = 5, seed: int = 0, lags: tuple = (), verbose: bool = True):
    """held_out_by_run for one recording. Returns (Z, straight, both), all in standardised units.

    With lags, the first max(lags) readings have no complete past and are dropped from all three.
    """
    out = held_out_by_run([X], k, seed, lags, verbose)
    return out["Z"], out["straight"], out["both"]


def _r2(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ((y - p) ** 2).sum(0) / np.where(ss > 0, ss, 1.0)


def link_strength(X: np.ndarray, channels: list[str], k: int = 5, seed: int = 0, lags: tuple = ()) -> pd.DataFrame:
    """Per sensor: share of its movement explained by the other sensors, on held-out time blocks.

    straight   explained by a straight-line model
    curved     extra explained once a flexible model is added on top (about 0 if links are straight)
    lags       past readings of the other sensors the models may also use (see held_out_by_run)
    """
    Z, lin, both = held_out_predictions(X, k, seed, lags)
    r_lin, r_both = _r2(Z, lin), _r2(Z, both)
    out = pd.DataFrame({"straight": r_lin, "curved": r_both - r_lin, "total": r_both}, index=channels)
    out.index.name = "sensor"
    return out


def curvature_summary(X: np.ndarray, channels: list[str], k: int = 5, seed: int = 0,
                      n_boot: int = 500, lags: tuple = ()) -> dict:
    """One line per dataset: average straight and curved shares, with a 90% interval on the curved share.

    For one long recording. The interval resamples blocks of time (20 blocks), so it respects that
    neighbouring readings look alike. For several separate runs use curvature_by_run, which
    resamples runs. `lags`: past readings of the other sensors the models may also use.
    """
    Z, lin, both = held_out_predictions(X, k, seed, lags)
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
            "sensors_with_curved_over_5pts": int((table["curved"] > 0.05).sum()),
            "sensors_with_curved_over_10pts": int((table["curved"] > 0.10).sum()), "table": table}


def curvature_by_run(runs: list, channels: list[str], k: int = 5, seed: int = 0, n_boot: int = 500,
                     lags: tuple = (), verbose: bool = True, what: str = "curvature measure",
                     folds: list | None = None, modes: list | None = None, units: list | None = None,
                     told: str = "levels") -> dict:
    """The curvature measure for several separate runs, with ranges from resampling whole runs.

    Same measure as curvature_summary (see held_out_by_run for how readings are held out), and the
    same keys in the answer, plus "runs", "straight_low"/"straight_high", and per sensor in
    "table": its straight share, its curved share and the 90% range of its curved share.

    Each resample draws runs with replacement and recomputes every share from the drawn runs'
    held-out errors (the models are not refitted). With very few runs the ranges are rough: two
    runs can only be redrawn three ways.

    folds, modes, told   passed on to held_out_by_run.
    units          one array per run numbering the stretch each reading belongs to (see
                   staggered_folds). If given, a second set of ranges comes from resampling those
                   stretches instead of whole runs: "curved_low_units"/"curved_high_units" (and
                   the same for "straight"), in the summary and per sensor, plus "units".
    """
    h = held_out_by_run(runs, k, seed, lags, verbose, what, folds, modes, told)
    Z = h["Z"] - h["Z"].mean(0)
    sq_lin, sq_both = (h["Z"] - h["straight"]) ** 2, (h["Z"] - h["both"]) ** 2

    def ranges(group):
        """Shares from all readings, and their draws when the groups of readings are resampled."""
        n = int(group.max()) + 1

        def per_group(values):                             # (groups, sensors): sum over each group's readings
            out = np.zeros((n, values.shape[1]))
            np.add.at(out, group, values)
            return out

        total, err_lin, err_both = per_group(Z ** 2), per_group(sq_lin), per_group(sq_both)
        tiny = 1e-9 * total.sum(0)

        def shares(idx):
            t = total[idx].sum(0)
            moves = t > tiny                               # a sensor that does not move has nothing to explain: 0
            t = np.where(moves, t, 1.0)
            return (np.where(moves, 1 - err_lin[idx].sum(0) / t, 0.0),
                    np.where(moves, (err_lin[idx].sum(0) - err_both[idx].sum(0)) / t, 0.0))

        rng = np.random.default_rng(seed)
        draws = [shares(idx) for idx in rng.integers(0, n, (n_boot, n))]
        return shares(np.arange(n)), np.array([d[0] for d in draws]), np.array([d[1] for d in draws])

    (straight, curved), s_draws, c_draws = ranges(h["run"])
    table = pd.DataFrame({"straight": straight, "curved": curved,
                          "curved_low": np.quantile(c_draws, 0.05, axis=0),
                          "curved_high": np.quantile(c_draws, 0.95, axis=0)}, index=channels)
    table.index.name = "sensor"
    lo, hi = np.quantile(c_draws.mean(1), [0.05, 0.95])
    s_lo, s_hi = np.quantile(s_draws.mean(1), [0.05, 0.95])
    out = {"sensors": len(channels), "readings": len(Z), "runs": len(runs),
           "straight": float(straight.mean()), "straight_low": float(s_lo), "straight_high": float(s_hi),
           "curved": float(curved.mean()), "curved_low": float(lo), "curved_high": float(hi),
           "sensors_with_curved_over_5pts": int((curved > 0.05).sum()),
           "sensors_with_curved_over_10pts": int((curved > 0.10).sum()), "table": table}
    if units is not None:
        back = max(tuple(lags), default=0)
        unit = np.unique(np.concatenate([np.asarray(u)[back:] for u in units]), return_inverse=True)[1]
        _, s_draws, c_draws = ranges(unit)
        table["curved_low_units"] = np.quantile(c_draws, 0.05, axis=0)
        table["curved_high_units"] = np.quantile(c_draws, 0.95, axis=0)
        out["units"] = int(unit.max()) + 1
        out["curved_low_units"], out["curved_high_units"] = map(float, np.quantile(c_draws.mean(1), [0.05, 0.95]))
        out["straight_low_units"], out["straight_high_units"] = map(float, np.quantile(s_draws.mean(1), [0.05, 0.95]))
    return out


def curvature_within(runs: list, labels: list, channels: list[str], names=str, between: int = -1, seed: int = 0,
                     n_boot: int = 500, fewest: int = 60, verbose: bool = True, what: str = "within settings",
                     whole_visits: bool = False, k: int = 5) -> pd.DataFrame:
    """The curvature measure inside each setting on its own: one row per setting.

    For every setting, the data is its unbroken stretches (visits) from all the runs, each with
    its own average level removed, so what is left is how the sensors move around their level
    while the setting is held. Every visit is cut into k blocks of time, and each block is
    predicted by models fitted on the rest (of that visit and the other visits), the same way
    the pump's sessions were measured. The range of the curved share comes from resampling
    visits; with two or three visits it is rough. Settings with fewer than two visits of at
    least `fewest` readings are left out.

    whole_visits=True holds out each visit whole instead. On the Cranfield rig that does not
    work: the small movements around a held setting differ from visit to visit, so models
    fitted on other visits predict a held-out visit worse than a flat line.
    """
    rows = {}
    codes = sorted({int(c) for label in labels for c in np.unique(label) if c != between})
    for code in progress.track(codes, what, verbose=verbose):
        visits = [X[a:b] - X[a:b].mean(0) for X, label in zip(runs, labels)
                  for a, b in stretches(label) if label[a] == code and b - a >= fewest]
        if len(visits) < 2:
            continue
        s = curvature_by_run(visits, channels, k=k, seed=seed, n_boot=n_boot, verbose=False,
                             folds=[np.full(len(v), i) for i, v in enumerate(visits)] if whole_visits else None)
        rows[names(code)] = {"visits": len(visits), "readings": s["readings"], "straight share": s["straight"],
                             "curved share": s["curved"], "curved, low": s["curved_low"], "curved, high": s["curved_high"],
                             "sensors gaining over 0.10 from curves": s["sensors_with_curved_over_10pts"]}
    out = pd.DataFrame(rows).T.astype({"visits": int, "readings": int, "sensors gaining over 0.10 from curves": int})
    out.index.name = "setting"
    return out


def best_single_predictor(X: np.ndarray, j: int, slices: int = 50) -> tuple[int, float]:
    """The one other sensor that on its own says most about sensor j, and the share it explains.

    For every other sensor, the readings are cut into `slices` equal-sized slices by that
    sensor's value, and sensor j is predicted by its average within each slice. The share of
    sensor j's movement this explains is high for a straight link and for a bent one alike.
    """
    y = X[:, j]
    total = ((y - y.mean()) ** 2).sum()
    best, share = -1, -np.inf
    for i in range(X.shape[1]):
        if i == j or X[:, i].std() <= 1e-9:
            continue
        order = np.argsort(X[:, i], kind="stable")
        within = sum(((part - part.mean()) ** 2).sum() for part in np.array_split(y[order], slices))
        if 1 - within / total > share:
            best, share = i, float(1 - within / total)
    return best, share


def movement_by_run(runs: list, channels: list[str], names: list | None = None) -> pd.DataFrame:
    """Per sensor: how unevenly its movement is spread over the runs.

    "largest share" is the share of the sensor's pooled movement (squared distance from the
    pooled mean, summed over readings) that comes from a single run; "run" says which one;
    "even share" is what every run would hold if they were all alike (1 / number of runs).

    Worth a look before trusting the ranges of curvature_by_run. If one run holds nearly all of
    a sensor's movement, that sensor's share is really a statement about that one run, and a
    resample that happens to leave the run out measures something else: the ranges blow up.
    """
    centre = np.vstack(runs).mean(0)
    per_run = np.array([((np.asarray(X) - centre) ** 2).sum(0) for X in runs])
    total = per_run.sum(0)
    share = per_run / np.where(total > 0, total, 1.0)
    names = list(names) if names is not None else list(range(len(runs)))
    out = pd.DataFrame({"largest share": share.max(0), "run": [names[i] for i in share.argmax(0)],
                        "even share": 1 / len(runs)}, index=channels)
    out.index.name = "sensor"
    return out


LAG_SETS = {"same moment only": (), "+ 1 back": (1,), "+ 1, 2 back": (1, 2), "+ 1, 2, 5 back": (1, 2, 5),
            "+ 1, 2, 5, 10 back": (1, 2, 5, 10)}


def curvature_by_lag(runs: list, channels: list[str], lag_sets: dict | None = None, k: int = 5, seed: int = 0,
                     n_boot: int = 500, verbose: bool = True, what: str = "curvature", cache: str | None = None) -> dict:
    """curvature_by_run once per set of lags. Returns {name of the lag set: summary}.

    cache: a name under which each set's answer is saved in results/cache as soon as it is done
    (see cached), so a long run that is interrupted picks up where it stopped.
    """
    out = {}
    for name, lags in (lag_sets or LAG_SETS).items():
        def compute(lags=lags, name=name):
            return curvature_by_run(runs, channels, k, seed, n_boot, lags, verbose, f"{what}, {name}")
        out[name] = cached(f"{cache}__lags_{'_'.join(map(str, lags)) or 'none'}", compute) if cache else compute()
    return out


def lag_table(by_lag: dict) -> pd.DataFrame:
    """One row per set of lags: the average shares, the curved share's range, and how many sensors gain from curves."""
    rows = {name: {"straight share": s["straight"], "straight, low": s.get("straight_low", np.nan),
                   "straight, high": s.get("straight_high", np.nan),
                   "curved share": s["curved"], "curved, low": s["curved_low"],
                   "curved, high": s["curved_high"],
                   "sensors gaining over 0.05 from curves": s["sensors_with_curved_over_5pts"],
                   "sensors gaining over 0.10 from curves": s["sensors_with_curved_over_10pts"]}
            for name, s in by_lag.items()}
    out = pd.DataFrame(rows).T.astype({"sensors gaining over 0.05 from curves": int,
                                       "sensors gaining over 0.10 from curves": int})
    out.index.name = "other sensors' readings used"
    return out


# ---------------------------------------------------------------- what kind of fault is each fault?

def _scores(ds: Dataset, make_detector, with_calibration: bool = False):
    """Score every run with a freshly fitted detector (one fit per distinct fit array).

    with_calibration=True also returns the scores of the dataset's fault-free calibration runs,
    cut to the stretch the fault covers (None if the dataset has none).
    """
    fitted = {}

    def scored(r):
        key = id(r.fit)
        if key not in fitted:
            m = make_detector().fit(r.fit)
            fitted[key] = (m, m.score_fit(r.fit))
        m, ref = fitted[key]
        return M.normalise(m.score(r.X), ref)

    out = [scored(r) for r in ds.runs]
    if not with_calibration:
        return out
    cal = [scored(r)[ds.matched_start:] for r in ds.calibration_runs()] if ds.matched_start is not None else []
    return out, (cal or None)


def sensors_moved(ds: Dataset, far: float = 0.01, window: int = 5, share: float = 0.5,
                  calibration: str | None = None) -> list[int]:
    """Per run: how many sensors, each judged alone, are out of range for most of the fault.

    A sensor counts if its own rolling average is beyond its own `far` false-alarm line in more
    than `share` of the fault readings. Each sensor's line is set the same way as the detectors'
    alarm lines (`calibration`, by default the dataset's own way).
    """
    mode = calibration or ds.calibration

    def rolled(r):
        mu, sd = _standardise(r.fit)
        return np.abs(det._trailing_mean((r.X - mu) / sd, window))

    Zs = [rolled(r) for r in ds.runs]
    labels = [r.y.astype(bool) for r in ds.runs]
    if mode == "matched":
        late = np.concatenate([rolled(r)[ds.matched_start:] for r in ds.calibration_runs()])
        thrs = [np.quantile(late, 1 - far, axis=0)] * len(Zs)
    elif mode == "pooled":
        thr = np.quantile(np.concatenate([z[~y] for z, y in zip(Zs, labels)]), 1 - far, axis=0)
        thrs = [thr] * len(Zs)
    else:
        thrs = [np.quantile(z[~y], 1 - far, axis=0) for z, y in zip(Zs, labels)]
    return [int(((z[y] > t).mean(0) > share).sum()) for z, y, t in zip(Zs, labels, thrs)]


def fault_types(ds: Dataset, far: float = 0.01, caught: float = 0.5, few: int = 3,
                with_curved: bool = True, seed: int = 0, saved: bool = False,
                calibration: str | None = None) -> pd.DataFrame:
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

    saved=True reads each detector's scores from results/cache (as written by run.score_all)
    instead of fitting and scoring again. calibration sets how the alarm lines are drawn, if not
    the dataset's own way.
    """
    labels = [r.y for r in ds.runs]
    groups = [r.group for r in ds.runs]
    mode = calibration or ds.calibration

    def by_group(key):
        if saved:
            from . import run
            sc, cal = run.load_scores(ds.name, key, 0)[0], run.calibration_scores(ds, key, 0)
        else:
            extra = {"seed": seed} if key == "link_curved" else {}
            sc, cal = _scores(ds, lambda: det.make(key, **extra), with_calibration=True)
        thr = M.dr_at_far(sc, labels, far, mode, cal)["threshold"]
        return pd.Series(M.dr_by_group(sc, labels, groups, thr))

    def best(*keys):
        return pd.concat([by_group(k) for k in keys], axis=1).max(axis=1)

    cols = {"level checks": best("rolling_z", "ewma", "cusum"),
            "straight links": best("pca", "link_straight")}
    if with_curved:
        cols["curved links"] = by_group("link_curved")
    if "matrix_profile" in det.available():
        cols["shape in time"] = by_group("matrix_profile")
    out = pd.DataFrame(cols)
    moved = pd.Series(sensors_moved(ds, far, calibration=mode), index=groups).groupby(level=0, sort=False).mean()
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


def lag_check(n: int = 3000, delay: int = 2, verbose: bool = True) -> pd.DataFrame:
    """The lag option on made-up data where the truth is known (simulate.delayed_sensors).

    One sensor follows another `delay` readings late, once as a straight-line copy and once as a
    curved function. Rows: which past readings were offered. Columns: the follower's straight and
    curved share in each case. The link should appear only once the right past reading is offered,
    as straight share in the first case and as curved share in the second. The driver changes at
    random from reading to reading, so a past reading other than the right one is no help:
    `delay` has to be one of the lags offered for the link to show at all.
    """
    from .simulate import delayed_sensors
    sets = {"same moment only": (), "+ 1 back": (1,), "+ 1, 2 back": (1, 2), "+ 1, 2, 5 back": (1, 2, 5)}
    if not any(delay in lags for lags in sets.values()):
        raise ValueError(f"delay={delay} is not among the lags offered, so the check could show nothing")
    cols = {}
    for label, curve in (("straight link", 0.0), ("curved link", 1.0)):
        by_lag = curvature_by_lag([delayed_sensors(n, delay, curve)], ["driver", "follower", "unrelated"], sets,
                                  n_boot=20, verbose=verbose, what=f"check, {label}")
        for share in ("straight", "curved"):
            cols[(f"{label}, {delay} readings late", f"{share} share")] = {
                name: s["table"].loc["follower", share] for name, s in by_lag.items()}
    out = pd.DataFrame(cols)
    out.index.name = "other sensors' readings used"
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
    sc, cal = {}, {}
    for name, curved in (("Straight-link residual", False), ("Curved-link residual", True)):
        sc[name], cal[name] = _scores(ds, lambda: LinkResidual(curved), with_calibration=True)
    cal = cal if ds.calibration == "matched" else None
    point = stats.point_dr(sc, labels, far, ds.calibration, cal)
    draws = stats.bootstrap_dr(sc, labels, far, n_boot, mode=ds.calibration, calib=cal)
    gap = draws["Curved-link residual"] - draws["Straight-link residual"]
    return {"straight": point["Straight-link residual"], "curved": point["Curved-link residual"],
            "straight_ci": tuple(np.nanquantile(draws["Straight-link residual"], [0.05, 0.95])),
            "gap": point["Curved-link residual"] - point["Straight-link residual"],
            "gap_ci": tuple(np.nanquantile(gap, [0.05, 0.95]))}
