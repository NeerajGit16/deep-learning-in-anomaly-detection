"""Deep models (notebook 09), each paired with a simple twin so the model is the only difference.

    model                       twin
    Autoencoder (one reading)   PCA            the same bottleneck size as PCA keeps; it can bend
    LSTM forecaster             Straight forecaster: a straight line from the same last W readings
    GDN (Deng and Hooi 2021)    Straight-link residual
    USAD (Audibert et al. 2020) Autoencoder

Every model has the same fit/score interface as the other detectors and gives one score per
reading that uses only readings up to that moment. Plain PyTorch, imported only when a model is
fitted, so the rest of the package works without it.

Training (the same for every model and dataset)
    Inputs are standardised with the fit data. About 20% of the fit data is held aside whole
    (every fifth run if the fit data has five or more runs, otherwise the last fifth of each
    run). Adam, learning rate 1e-3, batches of 256, at most 100 passes, stopping when the error
    on the held-aside data has not improved for 10 passes; the best pass is kept. Settings (the
    window W and the size) are chosen by the error on the held-aside data, with seed 0; other
    seeds reuse that choice. Faulty data is never seen.

Scoring
    Autoencoder, LSTM and GDN: each sensor's error is put on the scale of its errors on the
    held-aside data, then combined exactly as the straight-link detector does (detectors.LinkResidual):
    the largest, over sensors, of the error now and the average error over the last 5 readings.
    GDN (paper's score) uses the paper's own: each sensor's error less its median, over its
    interquartile range (both on the held-aside data), a running average over 3 readings, and
    the largest over sensors. USAD keeps its own score: 0.5 x (error of the first decoder) +
    0.5 x (error of the second decoder fed the first one's output), over the window ending now.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .detectors import Detector, _standardiser, _trailing_mean

# Window options per dataset family (by the start of the dataset's name).
WINDOWS = {"tep": (10, 20), "skab": (10, 30), "cranfield": (10, 30), "etch": (5, 10), "indpensim": (5, 10)}
DEFAULT_WINDOWS = (10, 20)
SMOOTH = 5                       # the straight-link detector's averaging window
GDN_SMOOTH = 3
MAX_EPOCHS, PATIENCE, BATCH, LR = 100, 10, 256, 1e-3
ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "results" / "cache" / "deep"
LOG = STORE / "training_log.jsonl"


def windows_for(dataset: str | None) -> tuple:
    for prefix, w in WINDOWS.items():
        if dataset and dataset.startswith(prefix):
            return w
    return DEFAULT_WINDOWS


def _torch():
    import torch
    return torch


def device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- held-aside split and windows

def segments(n: int, lengths: list | None) -> list[tuple[int, int]]:
    """(start, end) of each run joined into the fit data; one segment if the lengths are unknown."""
    if not lengths or sum(lengths) != n:
        return [(0, n)]
    edges = np.cumsum([0] + list(lengths))
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def split_aside(segs: list, share: float = 0.2) -> tuple[list, list]:
    """(training segments, held-aside segments): every fifth run held aside whole if there are five
    or more runs; otherwise the last fifth of each run."""
    if len(segs) >= 5:
        step = int(round(1 / share))
        return [s for i, s in enumerate(segs) if i % step != step - 1], [s for i, s in enumerate(segs) if i % step == step - 1]
    train, aside = [], []
    for a, b in segs:
        cut = b - max(1, int(round(share * (b - a))))
        train.append((a, cut))
        aside.append((cut, b))
    return train, aside


def ends(segs: list, w: int, ahead: bool) -> np.ndarray:
    """Positions p usable as targets: forecasting (ahead=True) needs readings p-w..p-1 inside the
    segment; a window ending at p (ahead=False) needs p-w+1..p."""
    first = w if ahead else w - 1
    out = [np.arange(a + first, b) for a, b in segs if b - a > first]
    return np.concatenate(out) if out else np.zeros(0, int)


def padded(Z: np.ndarray, w: int) -> np.ndarray:
    """Z with its first reading repeated w times in front, so every reading has a full window behind it."""
    return np.vstack([np.repeat(Z[:1], w, axis=0), Z])


# ---------------------------------------------------------------- the networks

def _nets():
    torch = _torch()
    nn = torch.nn

    class AE(nn.Module):
        def __init__(self, d, b):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, 64), nn.ELU(), nn.Linear(64, b))
            self.dec = nn.Sequential(nn.Linear(b, 64), nn.ELU(), nn.Linear(64, d))

        def forward(self, x):
            return self.dec(self.enc(x))

    class LSTM(nn.Module):
        def __init__(self, d, hidden, layers):
            super().__init__()
            self.lstm = nn.LSTM(d, hidden, num_layers=layers, batch_first=True)
            self.out = nn.Linear(hidden, d)

        def forward(self, x):                      # x: (batch, W, d) -> the next reading (batch, d)
            h, _ = self.lstm(x)
            return self.out(h[:, -1])

    class GDN(nn.Module):
        """Graph deviation network: learned sensor embeddings, a top-k graph by cosine similarity,
        graph attention over each sensor's last W readings, and a forecast for each sensor."""

        def __init__(self, d, w, dim, k):
            super().__init__()
            self.emb = nn.Parameter(torch.randn(d, dim) * 0.1)
            self.lin = nn.Linear(w, dim, bias=False)
            self.att = nn.Parameter(torch.randn(4 * dim) * 0.1)
            self.norm = nn.BatchNorm1d(dim)
            self.out = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
            self.k = k

        def neighbours(self):
            e = nn.functional.normalize(self.emb, dim=1)
            sim = e @ e.T
            sim.fill_diagonal_(-2.0)
            nbr = sim.topk(self.k, dim=1).indices                       # (d, k)
            own = torch.arange(len(e), device=e.device)[:, None]
            return torch.cat([own, nbr], dim=1)                          # each sensor attends to itself and k others

        def forward(self, x):                      # x: (batch, W, d)
            nodes = x.transpose(1, 2)              # (batch, d, W)
            h = self.lin(nodes)                    # (batch, d, dim)
            b, d, dim = h.shape
            g = torch.cat([self.emb.expand(b, d, dim), h], dim=2)        # (batch, d, 2 dim)
            nbr = self.neighbours()                                      # (d, k + 1)
            gi = g[:, :, None, :].expand(b, d, nbr.shape[1], 2 * dim)
            gj = g[:, nbr, :]                                            # (batch, d, k + 1, 2 dim)
            score = nn.functional.leaky_relu(torch.cat([gi, gj], dim=3) @ self.att, 0.2)
            alpha = torch.softmax(score, dim=2)                          # (batch, d, k + 1)
            z = torch.relu((alpha[..., None] * h[:, nbr, :]).sum(2))     # (batch, d, dim)
            z = self.norm(z.reshape(b * d, dim)).reshape(b, d, dim)
            return self.out(torch.relu(z) * self.emb).squeeze(-1)        # (batch, d)

    class USAD(nn.Module):
        def __init__(self, n_in, latent):
            super().__init__()
            h1, h2 = min(n_in // 2, 256), min(n_in // 4, 128)
            h1, h2 = max(h1, latent), max(h2, latent)
            self.enc = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, latent), nn.ReLU())

            def decoder():
                return nn.Sequential(nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, n_in),
                                     nn.Sigmoid())

            self.dec1, self.dec2 = decoder(), decoder()

        def forward(self, x):
            z = self.enc(x)
            w1 = self.dec1(z)
            return w1, self.dec2(z), self.dec2(self.enc(w1))

    return {"AE": AE, "LSTM": LSTM, "GDN": GDN, "USAD": USAD}


def weights(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


# ---------------------------------------------------------------- the detectors

class DeepDetector(Detector):
    """Shared training, settings choice, caching and scoring. Subclasses define the network."""
    stochastic = True
    kind = "base"                    # "snapshot", "forecast" or "window"
    link_score = True                # combine per-sensor errors like the straight-link detector

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.context_ = {}

    # --- settings and network, per subclass
    def options(self, d: int, dataset: str | None) -> list[dict]:
        raise NotImplementedError

    def build(self, d: int, opt: dict):
        raise NotImplementedError

    # --- training
    def _batches(self, idx_t, gen):
        """Shuffled batches of positions, drawn on the device (idx_t is already there)."""
        torch = _torch()
        order = idx_t[torch.randperm(len(idx_t), generator=gen, device="cpu").to(idx_t.device)]
        return [order[i:i + BATCH] for i in range(0, len(order), BATCH)]

    def _setup(self, Z, train_segs, aside_segs, w, ahead):
        torch = _torch()
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        dev = device()
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
        if self.kind == "snapshot":
            tr, ho = ends(train_segs, 1, False), ends(aside_segs, 1, False)
        else:
            tr, ho = ends(train_segs, w, ahead), ends(aside_segs, w, ahead)
        return gen, dev, Zt, torch.as_tensor(tr, device=dev), ho

    def _inputs(self, Zt, p, w):
        torch = _torch()
        if self.kind == "snapshot":
            return Zt[p]
        offs = torch.arange(-w, 0, device=Zt.device) if self.kind == "forecast" else torch.arange(-w + 1, 1, device=Zt.device)
        return Zt[p[:, None] + offs[None, :]]                          # (batch, W, d)

    def _loss(self, net, Zt, p, w, epoch=None):
        x = self._inputs(Zt, p, w)
        target = Zt[p] if self.kind != "window" else x.reshape(len(p), -1)
        pred = net(x if self.kind != "window" else x.reshape(len(p), -1))
        return ((pred - target) ** 2).mean()

    def _train(self, Z, train_segs, aside_segs, opt):
        """Train one setting; returns (network, held-aside error, passes, seconds)."""
        torch = _torch()
        w = opt.get("W", 1)
        gen, dev, Zt, tr, ho = self._setup(Z, train_segs, aside_segs, w, self.kind == "forecast")
        net = self.build(Z.shape[1], opt).to(dev)
        optim = torch.optim.Adam(net.parameters(), lr=LR)
        best, best_state, wait, passes = np.inf, None, 0, 0
        t0 = time.time()
        for epoch in range(1, MAX_EPOCHS + 1):
            net.train()
            for pt in self._batches(tr, gen):
                self._step(net, optim, Zt, pt, w, epoch)
            hold = self._aside_error(net, Zt, ho, w)
            passes = epoch
            if hold < best - 1e-7:
                best, wait = hold, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net, float(best), passes, time.time() - t0

    def _step(self, net, optim, Zt, pt, w, epoch):
        optim.zero_grad()
        loss = self._loss(net, Zt, pt, w, epoch)
        loss.backward()
        optim.step()

    def _aside_error(self, net, Zt, ho, w):
        torch = _torch()
        net.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ho), 4096):
                pt = torch.as_tensor(ho[i:i + 4096], device=Zt.device)
                total += float(self._loss(net, Zt, pt, w)) * len(pt)
                count += len(pt)
        return total / max(count, 1)

    # --- caching of trained networks (so a second score from the same network needs no training)
    def _key(self, X):
        return hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest()[:12]

    def _store(self, X, tag):
        STORE.mkdir(parents=True, exist_ok=True)
        return STORE / f"{self.family}__{self._key(X)}__{tag}"

    def fit(self, X):
        torch = _torch()
        super().fit(X)
        Z = self._space_fit(self._z(X))
        self.d_ = Z.shape[1]
        dataset = self.context_.get("dataset")
        segs = segments(len(Z), self.context_.get("lengths"))
        train_segs, aside_segs = split_aside(segs)
        aside = np.concatenate([np.arange(a, b) for a, b in aside_segs])
        train = np.concatenate([np.arange(a, b) for a, b in train_segs])
        # the error of simply predicting each sensor's average (from the training part), on held-aside data
        self.baseline_error_ = float(np.mean((Z[aside] - Z[train].mean(0)) ** 2))
        choice_file = self._store(X, "choice.json")
        if choice_file.exists():
            choice = json.loads(choice_file.read_text())
        else:                                       # choose the setting with seed 0, on held-aside data only
            keep_seed, self.seed = self.seed, 0
            tried = []
            for opt in self.usable(self.options(self.d_, dataset), train_segs, aside_segs):
                net, err, passes, secs = self._train_cached(X, Z, train_segs, aside_segs, opt)
                tried.append({"setting": opt, "held-aside error": err})
            self.seed = keep_seed
            choice = min(tried, key=lambda t: t["held-aside error"])
            choice = {"setting": choice["setting"], "tried": tried}
            choice_file.write_text(json.dumps(choice, indent=1))
        self.opt_ = choice["setting"]
        self.net_, self.aside_error_, self.passes_, self.seconds_ = self._train_cached(X, Z, train_segs, aside_segs, self.opt_)
        # errors on the held-aside data set the scale of each sensor's error
        self._set_reference(Z, aside_segs)
        return self

    def usable(self, options, train_segs, aside_segs):
        """The settings whose window leaves readings both to train on and to judge with (a short fit
        stretch can be shorter than the longer window); if none does, the one with the shortest window."""
        def ok(o):
            w = o.get("W", 1)
            ahead = self.kind == "forecast"
            return len(ends(train_segs, w, ahead)) > 0 and len(ends(aside_segs, w, ahead)) > 0
        good = [o for o in options if ok(o)]
        return good or [min(options, key=lambda o: o.get("W", 1))]

    def _space_fit(self, Z):
        """The units the network works in, fixed on the fit data (standardised units by default)."""
        return Z

    def _space(self, Z):
        return Z

    def _train_cached(self, X, Z, train_segs, aside_segs, opt):
        torch = _torch()
        tag = "_".join(f"{k}{v}" for k, v in sorted(opt.items())) + f"__s{self.seed}"
        path = self._store(X, tag + ".pt")
        net = self.build(Z.shape[1], opt).to(device())
        if path.exists():
            saved = torch.load(path, map_location=device(), weights_only=False)
            net.load_state_dict(saved["state"])
            net.eval()
            return net, saved["error"], saved["passes"], saved["seconds"]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        net, err, passes, secs = self._train(Z, train_segs, aside_segs, opt)
        peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else float("nan")
        torch.save({"state": net.state_dict(), "error": err, "passes": passes, "seconds": secs}, path)
        STORE.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"model": getattr(self, "log_name", self.family), "dataset": self.context_.get("dataset"),
                                "run": self.context_.get("run"),
                                "fit": self._key(X), "seed": self.seed, "setting": opt, "weights": weights(net),
                                "held-aside error": err, "predict-the-average error": self.baseline_error_,
                                "passes": passes, "seconds": round(secs, 2), "peak GPU MB": round(peak, 1),
                                "device": str(device())}) + "\n")
        return net, err, passes, secs

    # --- scoring
    def errors(self, X):
        """Per-sensor errors (readings x sensors) in standardised units, using only past and present readings."""
        torch = _torch()
        Z = self._space(self._z(X))
        w = self.opt_.get("W", 1)
        out = np.zeros_like(Z)
        P = padded(Z, w) if self.kind != "snapshot" else Z
        Pt = torch.as_tensor(P, dtype=torch.float32, device=device())
        shift = w if self.kind != "snapshot" else 0
        with torch.no_grad():
            for i in range(0, len(Z), 4096):
                p = torch.arange(i, min(i + 4096, len(Z)), device=Pt.device) + shift
                out[i:i + len(p)] = self._errors_at(Pt, p, w)
        return out

    def _errors_at(self, Pt, p, w):
        x = self._inputs(Pt, p, w)
        if self.kind == "window":
            raise NotImplementedError
        return (Pt[p] - self.net_(x)).cpu().numpy()

    def _set_reference(self, Z, aside_segs):
        err = np.vstack([self.errors_z(Z[a:b]) for a, b in aside_segs if b > a])
        self.ref_now_ = np.maximum(np.quantile(np.abs(err), 0.99, axis=0), 1e-9)
        self.ref_avg_ = np.maximum(np.quantile(np.abs(_trailing_mean(err, SMOOTH)), 0.99, axis=0), 1e-9)
        self.med_ = np.median(np.abs(err), axis=0)
        q75, q25 = np.quantile(np.abs(err), [0.75, 0.25], axis=0)
        self.iqr_ = np.maximum(q75 - q25, 1e-9)
        self.aside_scores_ = np.concatenate([self._combine(self.errors_z(Z[a:b])) for a, b in aside_segs if b > a])

    def errors_z(self, Zpart):
        """errors() for data already in the model's units (standardised for the autoencoder, LSTM and GDN)."""
        return self.errors(Zpart * self.sd_ + self.mu_)

    def _combine(self, err):
        now = np.abs(err) / self.ref_now_
        avg = np.abs(_trailing_mean(err, SMOOTH)) / self.ref_avg_
        return np.maximum(now, avg).max(1)

    def score(self, X):
        return self._combine(self.errors(X))

    def score_fit(self, X):
        return self.aside_scores_                   # scores on held-aside data: an honest reference scale


class Autoencoder(DeepDetector):
    """One reading in, the same reading out, through a bottleneck as wide as the number of
    components the PCA detector keeps on this data: sensors -> 64 -> b -> 64 -> sensors."""
    name = "Autoencoder"
    family = "autoencoder"
    kind = "snapshot"

    def options(self, d, dataset):
        from .detectors import PCA
        return [{"b": int(PCA(var=0.90).fit(self._train_data_for_b).k_)}]

    def fit(self, X):
        self._train_data_for_b = X
        return super().fit(X)

    def build(self, d, opt):
        return _nets()["AE"](d, opt["b"])


class LSTMForecaster(DeepDetector):
    """The last W readings of every sensor in, the next reading of every sensor out."""
    name = "LSTM forecaster"
    family = "lstm"
    kind = "forecast"

    def options(self, d, dataset):
        return [{"W": w, "hidden": h, "layers": n} for w in windows_for(dataset) for h, n in ((32, 1), (64, 2))]

    def build(self, d, opt):
        return _nets()["LSTM"](d, opt["hidden"], opt["layers"])


class GDNDetector(DeepDetector):
    """Graph deviation network, scored like the straight-link detector."""
    name = "GDN"
    family = "gdn"
    kind = "forecast"

    def options(self, d, dataset):
        return [{"W": w, "dim": e} for w in windows_for(dataset) for e in (32, 64)]

    def build(self, d, opt):
        return _nets()["GDN"](d, opt["W"], opt["dim"], min(10, d - 1))


class GDNPaperScore(GDNDetector):
    """The same trained GDN, with the paper's own score (median/IQR scaling, running average, largest)."""
    name = "GDN (paper's score)"

    def score(self, X):
        a = (np.abs(self.errors(X)) - self.med_) / self.iqr_
        return _trailing_mean(a, GDN_SMOOTH).max(1)

    def _set_reference(self, Z, aside_segs):
        super()._set_reference(Z, aside_segs)
        self.aside_scores_ = np.concatenate([_trailing_mean((np.abs(self.errors_z(Z[a:b])) - self.med_) / self.iqr_,
                                                            GDN_SMOOTH).max(1) for a, b in aside_segs if b > a])


class USADDetector(DeepDetector):
    """USAD: one encoder, two decoders trained against each other on windows of W readings."""
    name = "USAD"
    family = "usad"
    kind = "window"
    link_score = False

    def options(self, d, dataset):
        return [{"W": w, "latent": z} for w in windows_for(dataset) for z in (32, 64)]

    def build(self, d, opt):
        return _nets()["USAD"](opt["W"] * d, opt["latent"])

    def _space_fit(self, Z):
        """As in the paper: each sensor scaled to 0-1 by its lowest and highest value in the fit data."""
        self.lo_, self.hi_ = Z.min(0), Z.max(0)
        return self._space(Z)

    def _space(self, Z):
        return (Z - self.lo_) / np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)

    def _train(self, Z, train_segs, aside_segs, opt):
        torch = _torch()
        w = opt["W"]
        gen, dev, Zt, tr, ho = self._setup(Z, train_segs, aside_segs, w, False)
        net = self.build(Z.shape[1], opt).to(dev)
        opt1 = torch.optim.Adam(list(net.enc.parameters()) + list(net.dec1.parameters()), lr=LR)
        opt2 = torch.optim.Adam(list(net.enc.parameters()) + list(net.dec2.parameters()), lr=LR)
        best, best_state, wait, passes = np.inf, None, 0, 0
        t0 = time.time()
        for n in range(1, MAX_EPOCHS + 1):
            net.train()
            for p in self._batches(tr, gen):
                x = self._inputs(Zt, p, w).reshape(len(p), -1)
                # the paper's two-phase losses, weighted by the pass number n
                w1, w2, w3 = net(x)
                loss1 = (1 / n) * ((x - w1) ** 2).mean() + (1 - 1 / n) * ((x - w3) ** 2).mean()
                opt1.zero_grad()
                loss1.backward()
                opt1.step()
                w1, w2, w3 = net(x)
                loss2 = (1 / n) * ((x - w2) ** 2).mean() - (1 - 1 / n) * ((x - w3) ** 2).mean()
                opt2.zero_grad()
                loss2.backward()
                opt2.step()
            hold = self._aside_error(net, Zt, ho, w)
            passes = n
            if hold < best - 1e-7:
                best, wait = hold, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net, float(best), passes, time.time() - t0

    def _loss(self, net, Zt, p, w, epoch=None):
        """The first decoder's reconstruction error: used for stopping, for choosing the setting and
        for the check against predicting the average. (The second decoder is trained to reconstruct
        the first one's output badly, so the score's second term is large on purpose.)"""
        x = self._inputs(Zt, p, w).reshape(len(p), -1)
        w1, _, _ = net(x)
        return ((x - w1) ** 2).mean()

    def window_scores(self, Z):
        torch = _torch()
        w = self.opt_["W"]
        P = padded(Z, w - 1) if w > 1 else Z
        Pt = torch.as_tensor(P, dtype=torch.float32, device=device())
        out = np.zeros(len(Z))
        with torch.no_grad():
            for i in range(0, len(Z), 4096):
                p = torch.arange(i, min(i + 4096, len(Z)), device=Pt.device) + (w - 1)
                x = self._inputs(Pt, p, w).reshape(len(p), -1)
                w1, _, w3 = self.net_(x)
                out[i:i + len(p)] = (0.5 * ((x - w1) ** 2).mean(1) + 0.5 * ((x - w3) ** 2).mean(1)).cpu().numpy()
        return out

    def _set_reference(self, Z, aside_segs):
        self.aside_scores_ = np.concatenate([self.window_scores(Z[a:b]) for a, b in aside_segs if b > a])

    def score(self, X):
        return self.window_scores(self._space(self._z(X)))


class StraightForecaster(Detector):
    """The LSTM's twin: a straight line (ridge regression) from the last W readings of every sensor
    to the next reading of every sensor, W chosen from the same two options by held-aside error,
    scored exactly like the straight-link detector."""
    name = "Straight forecaster"
    stochastic = False

    def __init__(self):
        self.context_ = {}

    def _design(self, Z, p, w):
        return Z[p[:, None] + np.arange(-w, 0)[None, :]].reshape(len(p), -1)

    def fit(self, X):
        from sklearn.linear_model import RidgeCV
        super().fit(X)
        Z = self._z(X)
        train_segs, aside_segs = split_aside(segments(len(Z), self.context_.get("lengths")))
        best = None
        options = windows_for(self.context_.get("dataset"))
        usable = [w for w in options if len(ends(train_segs, w, True)) and len(ends(aside_segs, w, True))] or [min(options)]
        for w in usable:
            tr, ho = ends(train_segs, w, True), ends(aside_segs, w, True)
            m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(self._design(Z, tr, w), Z[tr])
            err = float(np.mean((Z[ho] - m.predict(self._design(Z, ho, w))) ** 2))
            if best is None or err < best[0]:
                best = (err, w, m)
        self.aside_error_, self.w_, self.model_ = best
        err = np.vstack([self.errors(Z[a:b] * self.sd_ + self.mu_) for a, b in aside_segs if b > a])
        self.ref_now_ = np.maximum(np.quantile(np.abs(err), 0.99, axis=0), 1e-9)
        self.ref_avg_ = np.maximum(np.quantile(np.abs(_trailing_mean(err, SMOOTH)), 0.99, axis=0), 1e-9)
        self.aside_scores_ = np.concatenate([self.score(Z[a:b] * self.sd_ + self.mu_) for a, b in aside_segs if b > a])
        return self

    def errors(self, X):
        Z = self._z(X)
        P = padded(Z, self.w_)
        p = np.arange(len(Z)) + self.w_
        return Z - self.model_.predict(self._design(P, p, self.w_))

    def score(self, X):
        err = self.errors(X)
        return np.maximum(np.abs(err) / self.ref_now_, np.abs(_trailing_mean(err, SMOOTH)) / self.ref_avg_).max(1)

    def score_fit(self, X):
        return self.aside_scores_


class PCATwin(Detector):
    """Added after the first deep-model results were seen: PCA made the autoencoder's exact twin.

    The PCA detector scores with T2 and SPE; the autoencoder scores each sensor's error like the
    straight-link detector. So the gap between those two mixes the bend with the score. This twin
    keeps PCA's straight reconstruction (the same number of directions as the autoencoder's
    bottleneck) but does everything else as the autoencoder does: it is fitted on the same
    training part, and each sensor's error is put on the scale of its errors on the same
    held-aside data and combined the same way. The autoencoder minus this twin is the bend alone."""
    name = "PCA, scored like the autoencoder"
    stochastic = False

    def __init__(self):
        self.context_ = {}

    def fit(self, X):
        from .detectors import PCA
        super().fit(X)
        Z = self._z(X)
        train_segs, aside_segs = split_aside(segments(len(Z), self.context_.get("lengths")))
        train = np.concatenate([np.arange(a, b) for a, b in train_segs])
        self.b_ = int(PCA(var=0.90).fit(X).k_)                  # the autoencoder's bottleneck size
        _, _, vt = np.linalg.svd(Z[train] - Z[train].mean(0), full_matrices=False)
        self.centre_, self.P_ = Z[train].mean(0), vt[: self.b_].T
        err = np.vstack([self._errors_z(Z[a:b]) for a, b in aside_segs if b > a])
        self.ref_now_ = np.maximum(np.quantile(np.abs(err), 0.99, axis=0), 1e-9)
        self.ref_avg_ = np.maximum(np.quantile(np.abs(_trailing_mean(err, SMOOTH)), 0.99, axis=0), 1e-9)
        self.aside_scores_ = np.concatenate([self._combine(self._errors_z(Z[a:b])) for a, b in aside_segs if b > a])
        return self

    def _errors_z(self, Z):
        C = Z - self.centre_
        return C - C @ self.P_ @ self.P_.T

    def _combine(self, err):
        return np.maximum(np.abs(err) / self.ref_now_, np.abs(_trailing_mean(err, SMOOTH)) / self.ref_avg_).max(1)

    def score(self, X):
        return self._combine(self._errors_z(self._z(X)))

    def score_fit(self, X):
        return self.aside_scores_


MODELS = {"deep_ae": Autoencoder, "deep_lstm": LSTMForecaster, "deep_gdn": GDNDetector,
          "deep_gdn_paper": GDNPaperScore, "deep_usad": USADDetector}
TWINS = {"deep_ae": "pca", "deep_lstm": "straight_forecaster", "deep_gdn": "link_straight",
         "deep_gdn_paper": "link_straight", "deep_usad": "deep_ae"}


# ---------------------------------------------------------------- notebook 09: every dataset, the same way

DEEP = ["deep_ae", "deep_lstm", "deep_gdn", "deep_gdn_paper", "deep_usad"]
SEEDS = 3
ORDER = ["etch", "skab", "tep_rieth", "cranfield", "tep6", "indpensim"]
TITLES = {"etch": "Metal etch", "skab": "Water pump (SKAB)", "tep_rieth": "Chemical plant, one setting (Rieth runs)",
          "cranfield": "Flow rig (Cranfield)", "tep6": "Chemical plant, six modes", "indpensim": "Penicillin batches (IndPenSim)"}
TWIN_NAMES = {"deep_ae": "PCA (T2 + SPE)", "deep_lstm": "Straight forecaster", "deep_gdn": "Straight-link residual",
              "deep_gdn_paper": "Straight-link residual", "deep_usad": "Autoencoder"}
SIMPLE = ["rolling_z", "ewma", "cusum", "pca", "iforest", "matrix_profile", "link_straight", "link_curved",
          "link_straight_told", "link_kinked", "straight_forecaster", "pca_twin"]


def dataset(name: str):
    """The dataset as the earlier notebooks score it, plus how its alarm line is set:
    {"ds", "calibration", "sources" (detectors scored on a told copy), "also" (other datasets to score)}."""
    from . import data, etch, indpensim, multimode
    if name == "etch":
        return {"ds": etch.dataset(), "calibration": "per experiment", "sources": {}, "also": [etch.dataset(told=True)]}
    if name == "skab":
        return {"ds": data.load("skab"), "calibration": "per_run", "sources": {}, "also": []}
    if name == "tep_rieth":
        ds = data.load_tep_rieth()
        return {"ds": ds, "calibration": "matched", "sources": {}, "also": [ds.matched_from]}
    if name == "cranfield":
        return {"ds": data.load_cranfield(), "calibration": "per_run", "sources": {}, "also": []}
    if name == "tep6":
        told = multimode.detector_dataset(told=True)
        return {"ds": multimode.detector_dataset(), "calibration": "matched", "sources": {"link_straight_told": told},
                "also": [told]}
    if name == "indpensim":
        told = indpensim.detector_dataset(told=True)
        return {"ds": indpensim.detector_dataset(), "calibration": "matched", "sources": {"link_straight_told": told},
                "also": [told]}
    raise KeyError(name)


def score(name: str, verbose: bool = True):
    """Fit and score the deep models and the straight forecaster on one dataset (cached)."""
    from . import progress, run
    spec = dataset(name)
    progress.note(f"deep models: scoring {name}")
    from . import detectors as det
    keep = self_checks()["keep"]                      # models that failed a self-check are not run
    standard = [m for m in run.METHODS if m in det.available()]
    for ds in [spec["ds"]] + [d for d in spec["also"] if d is not None and not d.name.endswith("_told")]:
        run.score_all([ds], methods=standard, n_seeds=5, verbose=verbose)            # cached already, except new datasets
        run.score_all([ds], methods=["link_kinked", "straight_forecaster"] + keep, n_seeds=SEEDS, verbose=verbose)
    for ds in [d for d in spec["also"] if d is not None and d.name.endswith("_told")]:
        run.score_all([ds], methods=["link_straight_told"], verbose=verbose)
    progress.note(f"deep models: {name} done")


def report(name: str, n_boot: int = 1000, verbose: bool = True) -> dict:
    """Every detector on one dataset (the earlier ones and the new ones), under the dataset's own alarm line."""
    from . import etch, run, structure
    spec = dataset(name)
    keep = self_checks()["keep"]
    methods = [m for m in SIMPLE + keep if m != "link_straight_told" or spec["sources"] or name == "etch"]

    def compute():
        if name == "etch":
            from . import detectors as det
            ds = spec["ds"]
            per_method = {}
            for m in methods:
                src = etch.TOLD_NAME if m == "link_straight_told" else ds.name
                seeds = [s for s in run._seeds(m, 5) if run._cache_path(src, m, s).exists()]
                per_method[det.REGISTRY[m].name] = [run.load_scores(src, m, s)[:2] for s in seeds]
            from . import batch
            return batch.report(per_method, [r.group for r in ds.runs], [r.meta["experiment"] for r in ds.runs],
                                [r.meta["experiment"] for r in ds.val_normal], [r.meta["role"] for r in ds.val_normal],
                                per_group=True, n_boot=n_boot, verbose=verbose)
        return run.report_dataset(spec["ds"], methods, 5, n_boot, calibration=spec["calibration"], verbose=verbose,
                                  sources=spec["sources"])

    return structure.cached(f"deep_report__{name}", compute)


def _summary(res: dict):
    import pandas as pd
    return pd.DataFrame(res["summary"]).set_index("method")


def gap(res: dict, a: str, b: str) -> dict:
    """a minus b in catch rate, using each detector's mean over seeds for the gap, and the range of the
    seed-0 paired gap moved by the same amount. A win needs the 12-point margin and a range clear of zero."""
    from .run import MARGIN
    s = _summary(res)
    mean_gap = float(s.loc[a, "dr_at_1pct"] - s.loc[b, "dr_at_1pct"])
    pair = next((p for p in res["pairs"] if {p["method_a"], p["method_b"]} == {a, b}), None)
    if pair is None:
        return {"gap": mean_gap, "low": float("nan"), "high": float("nan"), "verdict": "not compared"}
    sign = 1 if pair["method_a"] == a else -1
    seed0 = sign * pair["gap"]
    lo, hi = sorted((sign * pair["ci90_low"], sign * pair["ci90_high"]))
    lo, hi = lo + mean_gap - seed0, hi + mean_gap - seed0
    clear = abs(mean_gap) >= MARGIN and (lo > 0 or hi < 0)
    return {"gap": mean_gap, "low": lo, "high": hi, "verdict": ("ahead" if mean_gap > 0 else "behind") if clear else "tie"}


def table(res: dict) -> "pd.DataFrame":
    """Per deep model: catch rate (mean over 3 seeds) with range, spread across seeds, its gap to its
    twin and to the best simple detector on this dataset, and training time and GPU memory."""
    import pandas as pd
    from . import detectors as det
    s = _summary(res)
    # Every detector that is not deep. Follow-up round: the PCA scored like the autoencoder (added after
    # seeing results) now counts too; it is never the best, so no number here changed.
    simple = [det.REGISTRY[k].name for k in SIMPLE if det.REGISTRY[k].name in s.index]
    best = s.loc[simple, "dr_at_1pct"].idxmax()
    rows = {}
    for key in DEEP:
        name = det.REGISTRY[key].name
        if name not in s.index:
            continue
        twin = TWIN_NAMES[key]
        g_twin, g_best = gap(res, name, twin), gap(res, name, best)
        rows[name] = {"caught (mean of 3 seeds)": s.loc[name, "dr_at_1pct"],
                      "range (seed 0)": f"{s.loc[name, 'dr_ci90_low']:.2f}-{s.loc[name, 'dr_ci90_high']:.2f}",
                      "spread across seeds": s.loc[name, "dr_at_1pct_seed_sd"] if "dr_at_1pct_seed_sd" in s else float("nan"),
                      "twin": twin, "twin caught": s.loc[twin, "dr_at_1pct"] if twin in s.index else float("nan"),
                      "gap to twin": g_twin["gap"], "gap to twin, range": f"{g_twin['low']:+.2f} to {g_twin['high']:+.2f}",
                      "against twin": g_twin["verdict"],
                      **({"gap to PCA scored alike (added)": gap(res, name, "PCA, scored like the autoencoder")["gap"],
                          "against PCA scored alike": gap(res, name, "PCA, scored like the autoencoder")["verdict"]}
                         if key == "deep_ae" and "PCA, scored like the autoencoder" in s.index else {}),
                      "best simple detector": best, "best simple caught": s.loc[best, "dr_at_1pct"],
                      "gap to best simple": g_best["gap"], "gap to best simple, range": f"{g_best['low']:+.2f} to {g_best['high']:+.2f}",
                      "against best simple": g_best["verdict"]}
    out = pd.DataFrame(rows).T
    out.index.name = "deep model"
    return out


def training_table(dataset_name: str):
    """Per model: the setting chosen, training passes, seconds and peak GPU memory (from the training log),
    and the held-aside error against simply predicting each sensor's average."""
    import pandas as pd
    if not LOG.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if (r.get("dataset") or "").startswith(dataset_name) and "notebook 11" not in r["model"]]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["setting"] = df["setting"].astype(str)
    out = df.groupby(["model", "setting"]).agg(fits=("seconds", "size"), seconds=("seconds", "sum"),
                                               passes=("passes", "mean"), peak_MB=("peak GPU MB", "max"),
                                               weights=("weights", "max"), held_aside=("held-aside error", "mean"),
                                               average=("predict-the-average error", "mean"))
    out["beats the average"] = out["held_aside"] < out["average"]
    return out


def overview(reports: dict):
    """deep.table for every dataset, one block per dataset (in ORDER)."""
    import pandas as pd
    parts = {TITLES[n]: table(reports[n]) for n in ORDER if n in reports}
    return pd.concat(parts, names=["dataset", "deep model"])


def overview_plot(frame):
    """Each deep model's gap to its twin, dataset by dataset, with its range; dashed lines at +-12 points."""
    import matplotlib.pyplot as plt
    from . import plots
    rows = frame.reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(rows) + 1.4))
    y = list(range(len(rows)))[::-1]
    for yy, (_, r) in zip(y, rows.iterrows()):
        lo, hi = (float(v) for v in r["gap to twin, range"].split(" to "))
        ax.hlines(yy, lo, hi, color=plots.BLUE, lw=2)
        ax.plot(r["gap to twin"], yy, "o", color=plots.BLUE, ms=6, mec=plots.BG, mew=1.2)
    for x in (-0.12, 0.12):
        ax.axvline(x, color=plots.ORANGE, lw=1, ls=(0, (3, 3)))
    ax.axvline(0, color=plots.INK2, lw=1)
    ax.set_yticks(y, [f"{r['dataset']}: {r['deep model']} vs {r['twin']}" for _, r in rows.iterrows()], fontsize=7)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:+.0f}")
    ax.set_xlabel("catch rate of the deep model minus its twin, in points (dashed: the 12-point margin)")
    ax.grid(axis="x", color=plots.GRID, lw=0.6)
    ax.set_title("Deep models against their simple twins. Bars = range")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- self-checks on simulated data
#
# Fixed before any deep model was run on real data. A model that fails a check is dropped from the
# real-data runs (only that model); the reason is recorded.
#   curvature   on simulate.linked_sensors, at curve=1 the autoencoder reconstructs held-out normal
#               data better than PCA with the same number of components (its error below 0.8 x
#               PCA's); at curve=0 it does not beat PCA by much (its error above 0.8 x PCA's)
#   link break  every model catches at least 10% of the readings after one sensor's link to the
#               others is broken (its readings are shifted in time; its level and spread stay the
#               same), at 1 false alarm in 100 set on the readings before the break: ten times the
#               false-alarm rate
SIM_N, SIM_D, SIM_RUNS = 12000, 8, 10
CHECK_RATIO, CHECK_CAUGHT = 0.8, 0.10


def _simulated(curve: float, seed: int = 0):
    from .simulate import linked_sensors
    X = linked_sensors(SIM_N, SIM_D, curve=curve, seed=seed)
    return X[:10000], X[10000:].copy()


def _fit_with_context(key: str, X: np.ndarray, seed: int = 0):
    from . import detectors as det
    model = det.make(key, **({"seed": seed} if det.REGISTRY[key].stochastic else {}))
    if hasattr(model, "context_"):
        model.context_ = {"dataset": "simulated", "lengths": [len(X) // SIM_RUNS] * SIM_RUNS}
    return model.fit(X)


def curvature_check() -> "pd.DataFrame":
    """Reconstruction error on held-out simulated normal data: autoencoder against PCA (same size)."""
    import pandas as pd
    from .detectors import PCA
    rows = {}
    for curve in (0.0, 1.0):
        fit, test = _simulated(curve)
        ae = _fit_with_context("deep_ae", fit)
        Zt = ae._z(test)
        torch = _torch()
        with torch.no_grad():
            rec = ae.net_(torch.as_tensor(Zt, dtype=torch.float32, device=device())).cpu().numpy()
        pca = PCA(var=0.90).fit(fit)
        Zp = pca._z(test)
        P = pca.P_
        ae_err, pca_err = float(np.mean((Zt - rec) ** 2)), float(np.mean((Zp - Zp @ P @ P.T) ** 2))
        ratio = ae_err / pca_err
        ok = ratio < CHECK_RATIO if curve == 1.0 else ratio > CHECK_RATIO
        rows[f"curve {curve:g}"] = {"components": ae.opt_["b"], "autoencoder error": ae_err, "PCA error": pca_err,
                                    "autoencoder / PCA": ratio, "expected": "below 0.8" if curve == 1.0 else "above 0.8",
                                    "passed": bool(ok)}
    out = pd.DataFrame(rows).T
    out.index.name = "simulated links"
    return out


def link_break_check(keys: list | None = None) -> "pd.DataFrame":
    """Catch rate at 1 false alarm in 100 after one sensor's link is broken (simulated, curve 0.5)."""
    import pandas as pd
    from . import detectors as det
    fit, test = _simulated(0.5, seed=1)
    half = len(test) // 2
    test[half:, 3] = np.roll(test[half:, 3], 137)
    broken = np.r_[np.zeros(half, bool), np.ones(len(test) - half, bool)]
    rows = {}
    for key in keys or (DEEP + ["straight_forecaster", "pca", "link_straight"]):
        model = _fit_with_context(key, fit)
        s = model.score(test)
        thr = np.quantile(s[~broken], 0.99)
        caught = float(np.mean(s[broken] > thr))
        rows[det.REGISTRY[key].name] = {"caught": caught, "passed": caught >= CHECK_CAUGHT}
    out = pd.DataFrame(rows).T
    out.index.name = "detector"
    return out


def self_checks() -> dict:
    """Both checks (cached), and the deep models that passed: {"curvature", "link break", "keep", "dropped"}."""
    from . import structure
    curv = structure.cached("deep_check_curvature", curvature_check)
    brk = structure.cached("deep_check_link_break", link_break_check)
    from . import detectors as det
    dropped = {}
    if not curv["passed"].all():
        dropped["deep_ae"] = "curvature check: " + "; ".join(f"{i}: autoencoder/PCA {r['autoencoder / PCA']:.2f}, expected {r['expected']}"
                                                            for i, r in curv.iterrows() if not r["passed"])
    for key in DEEP:
        name = det.REGISTRY[key].name
        if name in brk.index and not brk.loc[name, "passed"]:
            dropped[key] = f"link-break check: caught {brk.loc[name, 'caught']:.2f}, below {CHECK_CAUGHT:.2f}"
    return {"curvature": curv, "link break": brk, "keep": [k for k in DEEP if k not in dropped], "dropped": dropped}


def _register():
    from . import detectors
    detectors.REGISTRY.update(MODELS)
    detectors.REGISTRY["straight_forecaster"] = StraightForecaster
    detectors.REGISTRY["pca_twin"] = PCATwin


_register()
