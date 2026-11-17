"""Notebook 11: why the autoencoder wins on IndPenSim. Explanatory work only: nothing here changes the
headline's settings or result. Every model below is fitted on the same 60 fit batches, with the same
held-aside batches, and judged with the same alarm line (the 30 line batches) as the headline.

Fixed before computing (follow-up round, part B):

B1  The win's batches are the three substrate feed batches (91, 95, 97; followup.ips_by_batch). For each
    of their faulty readings: does the autoencoder alarm (seed 0, the seed whose network is studied here),
    does the best simple detector (the curved-link residual) alarm? Where the autoencoder alarms and the
    curved-link does not, which sensor drives the alarm: the sensor with the largest scaled error (the
    score is the largest, over sensors, of each sensor's scaled error now or over the last 5 readings).
    The same for PCA scored like the autoencoder, on the same readings.
B3  Each test is a straight-line model given one extra ingredient, scored exactly like the autoencoder
    (per-sensor errors scaled by their 99th percentile on the held-aside batches, the same combination):
      corners     PCA on the sensors plus the kinked check's hinge terms (each sensor's distance above its
                  10th, 50th and 90th percentile), errors read on the sensors
      stages      a separate PCA per stage (tenth of the batch)
      strategies  a separate PCA per control strategy
      switches    PCA on the sensors plus, for each controller output that sits at its lowest or highest
                  normal value in at least 5% (and at most 95%) of the fit readings, a yes/no column for
                  "at or below its lowest" and/or "at or above its highest"
      sensors     the autoencoder trained on the 9 measurements only, then on the 11 controller outputs only
    Every PCA keeps the directions that hold 90% of its inputs' variance (the PCA detector's rule), each
    on its own inputs. Reported: catch rate and gap to the autoencoder, on all 10 faulty batches and on
    the 3 that carry the win. The ingredient that closes most of the gap names the ingredient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from . import run
from .detectors import Detector, _trailing_mean
from .run import FAR

WIN = (91, 95, 97)
SMOOTH = 5
CURVED = "Curved-link residual"


# ---------------------------------------------------------------- fitted models, as the headline fitted them

def _context(ds):
    return {"dataset": ds.name, "lengths": ds.fit_lengths}


def fitted(key: str, seed: int = 0, ds=None):
    """A detector fitted exactly as run.score_dataset fits it on IndPenSim (deep networks load from the cache)."""
    from . import detectors as det
    from . import indpensim
    ds = ds or indpensim.detector_dataset()
    model = det.make(key, **({"seed": seed} if det.REGISTRY[key].stochastic else {}))
    if hasattr(model, "context_"):
        model.context_ = _context(ds) | {"run": ds.runs[0].name}
    return model.fit(ds.runs[0].fit)


def _line(key: str, seed: int, src=None):
    from . import indpensim
    ds = src or indpensim.detector_dataset()
    runs, _, _ = run.load_scores(ds.name, key, seed)
    cal = run.calibration_scores(ds, key, seed)
    labels = [r.y for r in indpensim.detector_dataset().runs]
    return runs, M.dr_at_far(runs, labels, FAR, "matched", cal)["threshold"]


def scaled_errors(model, X) -> np.ndarray:
    """Each sensor's error, scaled as the score scales it: the larger of now and the last-5 average."""
    if hasattr(model, "errors"):
        err = model.errors(X)
    else:                                          # PCA twin
        err = model._errors_z(model._z(X))
    return np.maximum(np.abs(err) / model.ref_now_, np.abs(_trailing_mean(err, SMOOTH)) / model.ref_avg_)


# ---------------------------------------------------------------- B1: where and what

def readings_table(seed: int = 0) -> pd.DataFrame:
    """Every faulty reading of the three win batches: time, stage, fault stretch, sugar feed, and whether the
    autoencoder, the curved-link and PCA scored alike alarm."""
    from . import indpensim
    ds = indpensim.detector_dataset()
    ae_runs, ae_thr = _line("deep_ae", seed)
    cl_runs, cl_thr = _line("link_curved", 0)
    pt_runs, pt_thr = _line("pca_twin", 0)
    uv_runs, uv_thr = _line("unseen_value", 0)               # round 3: two schedule-aware checks
    nn_runs, nn_thr = _line("nearest_1", 0)
    bs = {b["batch"]: b for b in indpensim.batches()}
    rows = []
    for i, r in enumerate(ds.runs):
        b = r.meta["batch"]
        if b not in WIN:
            continue
        t, stage = bs[b]["time"], bs[b]["stage"]
        for k in np.flatnonzero(r.y):
            rows.append({"batch": b, "hour": float(t[k]), "stage": f"{10 * stage[k]}-{10 * stage[k] + 10}%",
                         "sugar feed": float(r.X[k, indpensim.CHANNELS.index("sugar feed")]),
                         "autoencoder alarms": bool(ae_runs[i][k] > ae_thr), "curved-link alarms": bool(cl_runs[i][k] > cl_thr),
                         "PCA scored alike alarms": bool(pt_runs[i][k] > pt_thr),
                         "unseen value check alarms (round 3)": bool(uv_runs[i][k] > uv_thr),
                         "nearest normal reading alarms (round 3)": bool(nn_runs[i][k] > nn_thr), "index": int(k)})
    out = pd.DataFrame(rows)
    out["stretch"] = pd.cut(out["hour"], [0, 50, 150, 300], labels=["20-30 h (sugar feed held at 2)",
                                                                     "76-92 h (held at 20)", "200-214 h (held at 20)"])
    out["autoencoder only"] = out["autoencoder alarms"] & ~out["curved-link alarms"]
    return out


def when_table(seed: int = 0) -> pd.DataFrame:
    """Per fault stretch and batch: faulty readings, and the share each detector alarms on."""
    t = readings_table(seed)
    g = t.groupby(["stretch", "batch"], observed=True)
    return pd.DataFrame({"faulty readings": g.size(), "stage": g["stage"].agg(lambda s: ", ".join(sorted(set(s)))),
                         "autoencoder": g["autoencoder alarms"].mean(), "curved-link": g["curved-link alarms"].mean(),
                         "PCA scored alike": g["PCA scored alike alarms"].mean(),
                         "autoencoder only": g["autoencoder only"].mean(),
                         "unseen value check (round 3)": g["unseen value check alarms (round 3)"].mean(),
                         "nearest normal reading (round 3)": g["nearest normal reading alarms (round 3)"].mean()})


def drivers(seed: int = 0) -> pd.DataFrame:
    """At the readings where the autoencoder alarms and the curved-link does not: each sensor's share of
    those alarms (it has the largest scaled error) and its mean scaled error (1 = its own 99% line), for the
    autoencoder and for PCA scored alike on the same readings."""
    from . import indpensim
    ds = indpensim.detector_dataset()
    t = readings_table(seed)
    only = t[t["autoencoder only"]]
    ae, pt, cl = fitted("deep_ae", seed), fitted("pca_twin"), _curved_link()
    S_ae, S_pt, S_cl = [], [], []
    for b, part in only.groupby("batch"):
        X = next(r.X for r in ds.runs if r.meta["batch"] == b)
        idx = part["index"].to_numpy()
        S_ae.append(scaled_errors(ae, X)[idx])
        S_pt.append(scaled_errors(pt, X)[idx])
        S_cl.append(scaled_errors(cl, X)[idx])
    S_ae, S_pt, S_cl = np.vstack(S_ae), np.vstack(S_pt), np.vstack(S_cl)
    ch = indpensim.CHANNELS
    out = pd.DataFrame({
        "autoencoder: share of these alarms it drives": pd.Series(S_ae.argmax(1)).value_counts(normalize=True).reindex(range(len(ch)), fill_value=0).to_numpy(),
        "autoencoder: mean scaled error": S_ae.mean(0),
        "PCA scored alike: share it would drive": pd.Series(S_pt.argmax(1)).value_counts(normalize=True).reindex(range(len(ch)), fill_value=0).to_numpy(),
        "PCA scored alike: mean scaled error": S_pt.mean(0),
        "curved-link: mean scaled error (added)": S_cl.mean(0)}, index=ch)
    out.attrs["readings"] = len(only)
    out.index.name = "sensor"
    return out.sort_values("autoencoder: share of these alarms it drives", ascending=False)


def _curved_link():
    """The curved-link detector as the headline fitted it (cached: its boosted trees take a few minutes).
    Added after B1's first results: its own scaled errors on the same readings, to see whether its error
    scale (set on blocks of the fit batches, which run strategy by strategy) is what hides the fault."""
    from . import structure
    return structure.cached("deepdive_curved_link_model", lambda: fitted("link_curved"))


def error_scales() -> pd.DataFrame:
    """Added after B1's first results. Each sensor's 99% error line (standardised units) for the autoencoder
    (set on held-aside batches), PCA scored alike (the same) and the curved-link detector (set on five blocks
    of the joined fit batches, which hold out mostly one control strategy at a time)."""
    from . import indpensim
    ae, pt, cl = fitted("deep_ae"), fitted("pca_twin"), _curved_link()
    return pd.DataFrame({"autoencoder": ae.ref_now_, "PCA scored alike": pt.ref_now_, "curved-link": cl.ref_now_},
                        index=indpensim.CHANNELS)


def alarm_lines() -> pd.DataFrame:
    """Added after B1's first results. Each model's alarm line in the units of its scaled errors (1 = a
    sensor's own 99% error line): the 99th percentile, over the 30 line batches' readings, of the largest
    scaled error across sensors. Also, per control strategy of the line batches, the same percentile."""
    from . import indpensim, structure

    def compute():
        ds = indpensim.detector_dataset()
        models = {"autoencoder": fitted("deep_ae"), "PCA scored alike": fitted("pca_twin"), "curved-link": _curved_link()}
        rows = {}
        for label, m in models.items():
            per = {s: [] for s in indpensim.STRATEGY_CODE}
            for r in ds.val_normal:
                per[r.meta["strategy"]].append(scaled_errors(m, r.X).max(1))
            allv = np.concatenate([np.concatenate(v) for v in per.values()])
            rows[label] = {"line, all 30 line batches": float(np.quantile(allv, 0.99)),
                           **{f"99th percentile, {s} line batches": float(np.quantile(np.concatenate(v), 0.99)) for s, v in per.items()}}
        return pd.DataFrame(rows).T

    return structure.cached("deepdive_alarm_lines", compute)


# ---------------------------------------------------------------- B2: pictures

def normal_and_faulty(seed: int = 0):
    """Normal fit-batch readings (with stage) and the win batches' faulty readings where the autoencoder
    alarms and the curved-link does not, both in raw units: (normal X, normal stage, faulty X, faulty batch)."""
    from . import indpensim
    ds = indpensim.detector_dataset()
    bs, roles = indpensim.normal_batches(), indpensim.roles()
    Xn = np.vstack([b["X"] for b, r in zip(bs, roles) if r == "fit"])
    stage = np.concatenate([b["stage"] for b, r in zip(bs, roles) if r == "fit"])
    t = readings_table(seed)
    only = t[t["autoencoder only"]]
    Xf = np.vstack([next(r.X for r in ds.runs if r.meta["batch"] == b)[part["index"].to_numpy()]
                    for b, part in only.groupby("batch")])
    fb = np.concatenate([np.full(len(part), b) for b, part in only.groupby("batch")])
    return Xn, stage, Xf, fb


def bottleneck(seed: int = 0):
    """The autoencoder's bottleneck for normal fit readings and the win's faulty readings, as 2-D: the two
    main directions of the normal readings' codes. Also PCA's first two components (PCA scored alike)."""
    import torch
    from . import deep
    Xn, stage, Xf, fb = normal_and_faulty(seed)
    ae, pt = fitted("deep_ae", seed), fitted("pca_twin")

    def code(X):
        with torch.no_grad():
            return ae.net_.enc(torch.as_tensor(ae._z(X), dtype=torch.float32, device=deep.device())).cpu().numpy()

    Cn, Cf = code(Xn), code(Xf)
    mu = Cn.mean(0)
    _, _, vt = np.linalg.svd(Cn - mu, full_matrices=False)
    ae2 = ((Cn - mu) @ vt[:2].T, (Cf - mu) @ vt[:2].T)
    pc = (((pt._z(Xn) - pt.centre_) @ pt.P_[:, :2]), ((pt._z(Xf) - pt.centre_) @ pt.P_[:, :2]))
    return {"autoencoder": ae2, "PCA": pc, "stage": stage, "batch": fb, "bottleneck size": int(Cn.shape[1])}


# ---------------------------------------------------------------- B3: one ingredient at a time

def _aside_segments(n, lengths):
    from .deep import segments, split_aside
    return split_aside(segments(n, lengths))


def _pca_basis(F, var=0.90):
    centre = F.mean(0)
    _, s, vt = np.linalg.svd(F - centre, full_matrices=False)
    share = np.cumsum(s ** 2) / np.sum(s ** 2)
    k = int(np.searchsorted(share, var) + 1)
    return centre, vt[:k].T


class FeaturePCA(Detector):
    """PCA on the standardised sensors plus extra feature columns made from them; each sensor's error is its
    own column's reconstruction error, scaled and combined as the autoencoder's. Subclasses define features."""
    name = "PCA with features"
    stochastic = False

    def __init__(self):
        self.context_ = {}

    def _features(self, Z):
        return np.zeros((len(Z), 0))

    def _learn_features(self, Z):
        pass

    def _phi(self, Z):
        F = np.hstack([Z, self._features(Z)])
        return (F - self.fmu_) / self.fsd_

    def fit(self, X):
        super().fit(X)
        Z = self._z(X)
        train_segs, aside_segs = _aside_segments(len(Z), self.context_.get("lengths"))
        train = np.concatenate([np.arange(a, b) for a, b in train_segs])
        self._learn_features(Z[train])
        F = np.hstack([Z[train], self._features(Z[train])])
        self.fmu_, sd = F.mean(0), F.std(0)
        self.fsd_ = np.where(sd > 1e-9, sd, 1.0)
        self.centre_, self.P_ = _pca_basis(self._phi(Z[train]))
        self.k_ = self.P_.shape[1]
        err = np.vstack([self._errors_z(Z[a:b]) for a, b in aside_segs if b > a])
        self.ref_now_ = np.maximum(np.quantile(np.abs(err), 0.99, axis=0), 1e-9)
        self.ref_avg_ = np.maximum(np.quantile(np.abs(_trailing_mean(err, SMOOTH)), 0.99, axis=0), 1e-9)
        self.aside_scores_ = np.concatenate([self._combine(self._errors_z(Z[a:b])) for a, b in aside_segs if b > a])
        return self

    def _errors_z(self, Z):
        C = self._phi(Z) - self.centre_
        R = C - C @ self.P_ @ self.P_.T
        return R[:, :Z.shape[1]] * self.fsd_[:Z.shape[1]]          # back to standardised sensor units

    def errors(self, X):
        return self._errors_z(self._z(X))

    def _combine(self, err):
        return np.maximum(np.abs(err) / self.ref_now_, np.abs(_trailing_mean(err, SMOOTH)) / self.ref_avg_).max(1)

    def score(self, X):
        return self._combine(self.errors(X))

    def score_fit(self, X):
        return self.aside_scores_


class PCAHinges(FeaturePCA):
    """Corners: each sensor's distance above its 10th, 50th and 90th percentile (zero below), as in the kinked check."""
    name = "PCA + corners (hinge terms)"

    def _learn_features(self, Z):
        self.knots_ = [np.quantile(Z, q, axis=0) for q in (0.1, 0.5, 0.9)]

    def _features(self, Z):
        return np.hstack([np.maximum(0.0, Z - k) for k in self.knots_])


SWITCH_SHARE = (0.05, 0.95)


class PCASwitches(FeaturePCA):
    """Switches: for each controller output that sits at its lowest or highest fit value in 5-95% of the fit
    readings, a yes/no column "at or below its lowest" and/or "at or above its highest"."""
    name = "PCA + switches (on/off columns)"

    def _learn_features(self, Z):
        outputs = _columns("outputs")
        self.switches_ = []
        lo, hi = SWITCH_SHARE
        for j in outputs:
            x = Z[:, j]
            for side, edge in (("low", x.min()), ("high", x.max())):
                at = np.mean(np.isclose(x, edge, atol=1e-9))
                if lo <= at <= hi:
                    self.switches_.append((j, side, float(edge)))

    def _features(self, Z):
        cols = [(Z[:, j] <= edge + 1e-9) if side == "low" else (Z[:, j] >= edge - 1e-9) for j, side, edge in self.switches_]
        return np.column_stack(cols).astype(float) if cols else np.zeros((len(Z), 0))


class GroupPCA(Detector):
    """A separate PCA for each group; the LAST column of X is the group (a stage or a control strategy) and
    is never scored. Errors scaled and combined as the autoencoder's, with one scale for all groups."""
    name = "PCA per group"
    stochastic = False

    def __init__(self):
        self.context_ = {}

    def fit(self, X):
        G, X = X[:, -1], X[:, :-1]
        super().fit(X)
        Z = self._z(X)
        train_segs, aside_segs = _aside_segments(len(Z), self.context_.get("lengths"))
        train = np.concatenate([np.arange(a, b) for a, b in train_segs])
        self.models_ = {g: _pca_basis(Z[train][G[train] == g]) for g in np.unique(G[train])}
        err = np.vstack([self._errors_z(Z[a:b], G[a:b]) for a, b in aside_segs if b > a])
        self.ref_now_ = np.maximum(np.quantile(np.abs(err), 0.99, axis=0), 1e-9)
        self.ref_avg_ = np.maximum(np.quantile(np.abs(_trailing_mean(err, SMOOTH)), 0.99, axis=0), 1e-9)
        self.aside_scores_ = np.concatenate([self._combine(self._errors_z(Z[a:b], G[a:b])) for a, b in aside_segs if b > a])
        return self

    def _errors_z(self, Z, G):
        out = np.zeros_like(Z)
        for g in np.unique(G):
            centre, P = self.models_.get(g, next(iter(self.models_.values())))
            C = Z[G == g] - centre
            out[G == g] = C - C @ P @ P.T
        return out

    def errors(self, X):
        return self._errors_z(self._z(X[:, :-1]), X[:, -1])

    def _combine(self, err):
        return np.maximum(np.abs(err) / self.ref_now_, np.abs(_trailing_mean(err, SMOOTH)) / self.ref_avg_).max(1)

    def score(self, X):
        return self._combine(self.errors(X))

    def score_fit(self, X):
        return self.aside_scores_


class PCAByStage(GroupPCA):
    name = "PCA per stage"


class PCAByStrategy(GroupPCA):
    name = "PCA per control strategy"


class LinkScoredAlike(Detector):
    """Added after B1's first results: the straight-link or curved-link models (each sensor predicted from
    the others at the same moment, LinkResidual._fit_pair), fitted on the autoencoder's training part only,
    with each sensor's error scaled on the autoencoder's held-aside batches and combined the same way."""
    stochastic = False
    curved = False

    def __init__(self):
        self.context_ = {}

    def fit(self, X):
        from .detectors import LinkResidual
        super().fit(X)
        Z = self._z(X)
        train_segs, aside_segs = _aside_segments(len(Z), self.context_.get("lengths"))
        train = np.concatenate([np.arange(a, b) for a, b in train_segs])
        self.link_ = LinkResidual(curved=self.curved)
        self.live_ = X.std(0) > 1e-9
        self.models_ = {}
        for j in np.flatnonzero(self.live_):
            others = self.live_.copy()
            others[j] = False
            self.models_[j] = (others, *self.link_._fit_pair(Z[train][:, others], Z[train, j]))
        err = np.vstack([self._errors_z(Z[a:b]) for a, b in aside_segs if b > a])
        self.ref_now_ = np.maximum(np.quantile(np.abs(err), 0.99, axis=0), 1e-9)
        self.ref_avg_ = np.maximum(np.quantile(np.abs(_trailing_mean(err, SMOOTH)), 0.99, axis=0), 1e-9)
        self.aside_scores_ = np.concatenate([self._combine(self._errors_z(Z[a:b])) for a, b in aside_segs if b > a])
        return self

    def _errors_z(self, Z):
        err = np.zeros_like(Z)
        for j, (others, m1, m2) in self.models_.items():
            err[:, j] = Z[:, j] - self.link_._predict(m1, m2, Z[:, others])
        return err

    def errors(self, X):
        return self._errors_z(self._z(X))

    def _combine(self, err):
        return np.maximum(np.abs(err) / self.ref_now_, np.abs(_trailing_mean(err, SMOOTH)) / self.ref_avg_).max(1)

    def score(self, X):
        return self._combine(self.errors(X))

    def score_fit(self, X):
        return self.aside_scores_


class StraightLinkAlike(LinkScoredAlike):
    name = "Straight-link, scored like the autoencoder (added)"


class CurvedLinkAlike(LinkScoredAlike):
    name = "Curved-link, scored like the autoencoder (added)"
    curved = True


def _columns(kind: str) -> list[int]:
    from . import indpensim
    role = {"measurements": indpensim.ONLINE, "outputs": indpensim.OUTPUT}[kind]
    roles = {name: r for name, r, _ in indpensim.COLUMNS}
    return [i for i, c in enumerate(indpensim.CHANNELS) if roles[c] == role]


def _autoencoder_on(kind: str, long: bool = False):
    from .deep import Autoencoder, AutoencoderLong

    class AutoencoderOn(AutoencoderLong if long else Autoencoder):
        name = f"Autoencoder, {kind} only" + (", long training" if long else "")
        log_name = f"autoencoder, {kind} only (notebook 11)"   # kept apart from the headline autoencoder in the log
        cols = _columns(kind)

        def fit(self, X):
            return super().fit(X[:, self.cols])

        def score(self, X):
            return super().score(X[:, self.cols])

    AutoencoderOn.__name__ = f"Autoencoder_{kind}" + ("_long" if long else "")
    return AutoencoderOn


INGREDIENTS = {"pca_hinges": PCAHinges, "pca_switches": PCASwitches, "pca_by_stage": PCAByStage,
               "pca_by_strategy": PCAByStrategy, "deep_ae_measurements": _autoencoder_on("measurements"),
               "deep_ae_outputs": _autoencoder_on("outputs"),
               "deep_ae_outputs_v2": _autoencoder_on("outputs", long=True),      # round 5: the long training budget
               "link_straight_alike": StraightLinkAlike, "link_curved_alike": CurvedLinkAlike}
ADDED = ["link_straight_alike", "link_curved_alike"]          # added after B1's first results
SOURCES = {"pca_by_stage": "stage", "pca_by_strategy": "strategy"}


def _register():
    from . import detectors
    detectors.REGISTRY.update(INGREDIENTS)
    run.EXTRA_METHODS.update({k: {} for k in INGREDIENTS})


_register()


def ingredient_datasets():
    """{key: the IndPenSim dataset that key is scored on} (the stage or strategy added as a last column where needed)."""
    from . import indpensim
    plain = indpensim.detector_dataset()
    extra = {"stage": indpensim.detector_dataset(told=True), "strategy": indpensim.detector_dataset(told="strategy")}
    return {k: extra[SOURCES[k]] if k in SOURCES else plain for k in INGREDIENTS}


def score_ingredients(verbose: bool = True):
    from . import deep
    for key, ds in ingredient_datasets().items():
        seeds = deep.SEEDS if key.startswith("deep_") else 1
        run.score_all([ds], methods=[key], n_seeds=seeds, verbose=verbose)


def ingredient_report(only_win: bool = False, n_boot: int = 1000) -> dict:
    from . import deep, indpensim, structure
    score_ingredients(verbose=False)
    ds = indpensim.detector_dataset()
    keep = [r.meta["batch"] in WIN for r in ds.runs] if only_win else None
    sources = {k: d for k, d in ingredient_datasets().items() if k in SOURCES}
    methods = ["deep_ae", "pca_twin", "link_curved"] + list(INGREDIENTS) + deep.SCHEDULE    # round 3: with the checks
    return structure.cached(f"deepdive_ingredients__{'win' if only_win else 'all'}__with_schedule_checks", lambda: run.report_dataset(
        ds, methods, deep.SEEDS, n_boot, keep=keep, calibration="matched", verbose=False, sources=sources))


def ingredient_table() -> pd.DataFrame:
    """Each ingredient model's catch rate and its gap to the autoencoder, on all 10 faulty batches and on the 3."""
    from . import deep
    rows = {}
    for label, only in (("all 10 batches", False), ("the 3 substrate feed batches", True)):
        res = ingredient_report(only)
        s = deep._summary(res)["dr_at_1pct"]
        ae = s["Autoencoder"]
        for m in s.index:
            g = deep.gap(res, m, "Autoencoder") if m != "Autoencoder" else {"gap": 0.0, "low": np.nan, "high": np.nan}
            rows.setdefault(m, {})
            rows[m][f"{label}: caught"] = s[m]
            rows[m][f"{label}: gap to the autoencoder, points"] = 100 * g["gap"]
            rows[m][f"{label}: range"] = f"{100 * g['low']:+.1f} to {100 * g['high']:+.1f}" if m != "Autoencoder" else ""
            rows[m][f"{label}: share of the gap closed"] = (s[m] - s["PCA, scored like the autoencoder"]) / \
                (ae - s["PCA, scored like the autoencoder"]) if m != "PCA, scored like the autoencoder" else 0.0
    out = pd.DataFrame(rows).T
    out.index.name = "model"
    return out


# ---------------------------------------------------------------- B4: numbers behind the recipe

STEPPED = ["aeration", "sugar feed", "dilution water", "head pressure", "PAA flow", "oil flow"]


def output_steps() -> pd.DataFrame:
    """Each controller output in the normal fit batches: distinct values, how often it changes in a recipe
    batch, and its share of readings at its commonest value."""
    from . import indpensim
    bs, roles = indpensim.normal_batches(), indpensim.roles()
    fit = [b for b, r in zip(bs, roles) if r == "fit"]
    rows = {}
    for j in _columns("outputs"):
        c = indpensim.CHANNELS[j]
        allx = np.concatenate([b["X"][:, j] for b in fit])
        rec = [b["X"][:, j] for b in fit if b["strategy"] == "recipe"]
        values, counts = np.unique(np.round(allx, 6), return_counts=True)
        rows[c] = {"distinct values (all fit batches)": len(values),
                   "distinct values (recipe fit batches)": len(np.unique(np.round(np.concatenate(rec), 6))),
                   "changes per recipe batch (median)": float(np.median([np.sum(np.diff(x) != 0) for x in rec])),
                   "share at commonest value": counts.max() / len(allx)}
    out = pd.DataFrame(rows).T
    out.index.name = "controller output"
    return out


def joint_states() -> pd.DataFrame:
    """How many combinations the six stepped outputs take in normal data, and whether the win's faulty
    readings (where the autoencoder alarms and the curved-link does not) fall on any of them."""
    from . import indpensim
    Xn, stage, Xf, fb = normal_and_faulty()
    cols = [indpensim.CHANNELS.index(c) for c in STEPPED]
    normal = {tuple(r) for r in np.round(Xn[:, cols], 6)}
    bs, roles = indpensim.normal_batches(), indpensim.roles()
    recipe = {tuple(r) for b, ro in zip(bs, roles) if ro == "fit" and b["strategy"] == "recipe"
              for r in np.round(b["X"][:, cols], 6)}
    seen = np.array([tuple(r) in normal for r in np.round(Xf[:, cols], 6)])
    each = {c: float(np.mean(np.isin(np.round(Xf[:, k], 6), np.round(Xn[:, k], 6)))) for c, k in zip(STEPPED, cols)}
    rows = {"combinations in the 60 normal fit batches": len(normal), "combinations in the recipe fit batches": len(recipe),
            "faulty readings whose combination occurs in normal data": float(seen.mean()),
            **{f"faulty readings whose {c} value occurs in normal data": v for c, v in each.items()}}
    return pd.Series(rows).to_frame("value")


def measurement_response() -> pd.DataFrame:
    """How far each measurement moves during the 76-92 h stretch of the three batches: its median there, minus
    the median of the normal recipe batches at the same hours, in units of their spread (the s.d. across batches)."""
    from . import indpensim
    d = indpensim.readings()
    cols = [indpensim.CHANNELS[j] for j in _columns("measurements")] + [indpensim.CHANNELS[j] for j in _columns("outputs")]
    win = (d["time"] >= 76) & (d["time"] <= 92)
    normal = d[win & (d["strategy"] == "recipe") & (d["batch"] <= 30)]
    per_batch = normal.groupby("batch")[cols].median()
    faulty = d[win & d["batch"].isin(WIN)].groupby("batch")[cols].median()
    shift = (faulty - per_batch.median()) / per_batch.std().replace(0, np.nan)
    out = shift.T
    out.columns = [f"batch {b}" for b in out.columns]
    out["mean shift (spreads)"] = out.mean(1)
    out.index.name = "sensor"
    return out.sort_values("mean shift (spreads)", key=np.abs, ascending=False)


def schedule_plot(channels: list | None = None):
    """The stepped outputs and the measurements that respond, over a batch: the normal recipe fit batches and the
    three substrate feed batches, with the 76-92 h stretch shaded."""
    from . import indpensim, plots
    channels = channels or ["sugar feed", "oil flow", "aeration", "dilution water", "heating water", "oxygen uptake"]
    bs, roles = indpensim.normal_batches(), indpensim.roles()
    normal = [(b["time"], b["X"]) for b, r in zip(bs, roles) if r == "fit" and b["strategy"] == "recipe"]
    faulty = [(b["time"], b["X"]) for b in indpensim.batches() if b["batch"] in WIN]
    desc = {c: f"{indpensim.INFO[c]} ({c})" for c in indpensim.CHANNELS}
    return plots.over_time_with_faults(normal, faulty, channels, indpensim.CHANNELS, desc, shade=(76, 92),
                                       faulty_names=[f"batch {b}" for b in WIN])
