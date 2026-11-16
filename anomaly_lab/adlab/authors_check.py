"""Follow-up round A3: our GDN and USAD against the authors' code, on the simulated link-break check only.

Each variant moves one thing (or a set of things) towards the authors' code, is trained on the check's
simulated normal data with three seeds, and is scored on its test stretch (sensor 3's second half rolled
by 137 readings). Caught = the share of broken readings above the 99th percentile of the scores on the
unbroken half, as deep.link_break_check; the bar stays at 0.10. Real faults are never used.

Authors' code read: github.com/d-ailin/GDN (models/GDN.py, models/graph_layer.py, datasets/TimeDataset.py,
train.py, evaluate.py, run.sh, scripts/process_swat.py) and github.com/manigalati/usad (usad.py, USAD.ipynb).
GDN's network is re-implemented here without torch_geometric. Results are saved to
results/cache/authors_check_{gdn,usad}.json; computing them again takes about 1.5 hours on the GPU.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from . import deep
from .detectors import _trailing_mean

BAR = deep.CHECK_CAUGHT
SEEDS = (0, 1, 2)


def path(which: str):
    return deep.ROOT / "results" / "cache" / f"authors_check_{which}.json"


def check_data():
    """The link-break check's data: (fit, test, broken, training segments, held-aside segments)."""
    fit, test = deep._simulated(0.5, seed=1)
    half = len(test) // 2
    test[half:, 3] = np.roll(test[half:, 3], 137)
    broken = np.r_[np.zeros(half, bool), np.ones(len(test) - half, bool)]
    train_segs, aside_segs = deep.split_aside(deep.segments(len(fit), [len(fit) // deep.SIM_RUNS] * deep.SIM_RUNS))
    return fit, test, broken, train_segs, aside_segs


def caught(s, broken):
    return float(np.mean(s[broken] > np.quantile(s[~broken], 0.99)))


def scaled(fit, test, kind):
    """Inputs as our models see them ("z": standardised) or as the authors' code does ("minmax": 0 to 1)."""
    if kind == "z":
        mu, sd = fit.mean(0), fit.std(0)
        return (fit - mu) / sd, (test - mu) / sd
    lo, hi = fit.min(0), fit.max(0)
    return (fit - lo) / (hi - lo), (test - lo) / (hi - lo)


# ---------------------------------------------------------------- networks

def nets():
    torch = deep._torch()
    nn, F = torch.nn, torch.nn.functional

    class GDNAuthors(nn.Module):
        """The authors' GDN: graph from the top-k cosine similarities of detached embeddings (self removed,
        then a self loop added), attention on [W x || v], a bias, BatchNorm, ReLU, times the embedding,
        BatchNorm, ReLU, dropout 0.2, Linear(dim, 1)."""

        def __init__(self, d, w, dim, topk):
            super().__init__()
            self.embedding = nn.Embedding(d, dim)
            nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
            self.lin = nn.Linear(w, dim, bias=False)
            nn.init.xavier_uniform_(self.lin.weight)
            self.att_i, self.att_j = nn.Parameter(torch.empty(1, dim)), nn.Parameter(torch.empty(1, dim))
            nn.init.xavier_uniform_(self.att_i)
            nn.init.xavier_uniform_(self.att_j)
            self.att_em_i, self.att_em_j = nn.Parameter(torch.zeros(1, dim)), nn.Parameter(torch.zeros(1, dim))
            self.bias = nn.Parameter(torch.zeros(dim))
            self.bn, self.bn_out = nn.BatchNorm1d(dim), nn.BatchNorm1d(dim)
            self.dp = nn.Dropout(0.2)
            self.out = nn.Linear(dim, 1)
            self.topk = topk

        def neighbours(self):
            v = self.embedding.weight.detach()
            cos = (v @ v.T) / (v.norm(dim=1)[:, None] * v.norm(dim=1)[None, :])
            top = cos.topk(self.topk, dim=1).indices.tolist()
            rows = [[i] + [j for j in top[i] if j != i] for i in range(len(v))]
            width = max(len(r) for r in rows)
            rows = [r + [r[-1]] * (width - len(r)) for r in rows]      # pad (rare: self not in the top k)
            return torch.tensor(rows, device=v.device)

        def forward(self, x):                                           # (b, W, d) -> (b, d)
            h = self.lin(x.transpose(1, 2))
            b, d, dim = h.shape
            v = self.embedding.weight
            nbr = self.neighbours()
            si = (h * self.att_i).sum(-1) + (v * self.att_em_i).sum(-1)
            sj = (h * self.att_j).sum(-1) + (v * self.att_em_j).sum(-1)
            alpha = torch.softmax(F.leaky_relu(si[:, :, None] + sj[:, nbr], 0.2), dim=2)
            agg = (alpha[..., None] * h[:, nbr, :]).sum(2) + self.bias
            z = F.relu(self.bn(agg.reshape(b * d, dim))).reshape(b, d, dim)
            out = F.relu(self.bn_out((z * v).permute(0, 2, 1))).permute(0, 2, 1)
            return self.out(self.dp(out)).squeeze(-1)

    class USADNet(nn.Module):
        """USAD's encoder and two decoders. caps=True: our sizes (hidden layers capped at 256 and 128 and
        never narrower than the latent space); caps=False: the authors' (input/2, input/4, then the latent)."""

        def __init__(self, n_in, latent, caps=True):
            super().__init__()
            if caps:
                h1, h2 = max(min(n_in // 2, 256), latent), max(min(n_in // 4, 128), latent)
            else:
                h1, h2 = n_in // 2, n_in // 4
            self.encoder = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(),
                                         nn.Linear(h2, latent), nn.ReLU())

            def decoder():
                return nn.Sequential(nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(),
                                     nn.Linear(h1, n_in), nn.Sigmoid())

            self.decoder1, self.decoder2 = decoder(), decoder()

    return {"GDNAuthors": GDNAuthors, "USADNet": USADNet}


# ---------------------------------------------------------------- GDN: training and scores

def train_forecaster(net, Z, w, batch, max_epochs, patience, train_segs, aside_segs, seed=0):
    torch = deep._torch()
    dev = deep.device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    tr, ho = deep.ends(train_segs, w, True), deep.ends(aside_segs, w, True)
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    offs = torch.arange(-w, 0, device=dev)

    def loss_at(p):
        return ((net(Zt[p[:, None] + offs[None, :]]) - Zt[p]) ** 2).mean()

    best, state, wait, ep = np.inf, None, 0, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        order = torch.as_tensor(tr)[torch.randperm(len(tr), generator=gen)].to(dev)
        for i in range(0, len(order), batch):
            pt = order[i:i + batch]
            if len(pt) < 2:
                continue
            opt.zero_grad()
            loss_at(pt).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            hold = float(np.mean([float(loss_at(torch.as_tensor(ho[i:i + 4096], device=dev))) for i in range(0, len(ho), 4096)]))
        if hold < best - 1e-7:
            best, wait, state = hold, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    return net, best, ep


def forecast_errors(net, Z, w):
    torch = deep._torch()
    dev = deep.device()
    Pt = torch.as_tensor(deep.padded(Z, w), dtype=torch.float32, device=dev)
    offs = torch.arange(-w, 0, device=dev)
    out = np.zeros_like(Z)
    with torch.no_grad():
        for i in range(0, len(Z), 4096):
            p = torch.arange(i, min(i + 4096, len(Z)), device=dev) + w
            out[i:i + len(p)] = (Pt[p] - net(Pt[p[:, None] + offs[None, :]])).cpu().numpy()
    return out


def gdn_scores(net, Zfit, Ztest, w, aside_segs, smooth=3, eps=1e-9, test_norm=False):
    """(link-style score, paper-style score). smooth="authors": the mean of the current and 3 previous readings,
    the first 3 set to 0 (their evaluate.py). test_norm=True: each sensor's median and IQR from the scored data
    itself, as their code does; not allowed under this project's rules, shown only to see whether it matters."""
    aside = np.vstack([forecast_errors(net, Zfit[a:b], w) for a, b in aside_segs])
    e = forecast_errors(net, Ztest, w)
    ref_now = np.quantile(np.abs(aside), 0.99, axis=0)
    ref_avg = np.quantile(np.abs(_trailing_mean(aside, 5)), 0.99, axis=0)
    link = np.maximum(np.abs(e) / ref_now, np.abs(_trailing_mean(e, 5)) / ref_avg).max(1)
    src = np.abs(e) if test_norm else np.abs(aside)
    med = np.median(src, axis=0)
    q75, q25 = np.quantile(src, [0.75, 0.25], axis=0)
    a = (np.abs(e) - med) / (np.abs(q75 - q25) + eps)
    if smooth == "authors":
        sm = np.zeros_like(a)
        for i in range(3, len(a)):
            sm[i] = a[i - 3:i + 1].mean(0)
        return link, sm.max(1)
    return link, _trailing_mean(a, smooth).max(1)


def gdn_variant(label, net_fn, scaling="z", w=10, dim=64, batch=256, max_epochs=100, patience=10, seeds=SEEDS, **score_kw):
    fit, test, broken, train_segs, aside_segs = check_data()
    Zf, Zt = scaled(fit, test, scaling)
    res, t0 = [], time.time()
    for s in seeds:
        net, held, passes = train_forecaster(net_fn(fit.shape[1], w, dim), Zf, w, batch, max_epochs, patience,
                                             train_segs, aside_segs, seed=s)
        link, paper = gdn_scores(net, Zf, Zt, w, aside_segs, **score_kw)
        res.append((caught(link, broken), caught(paper, broken), held, passes))
    r = np.array(res)
    return {"variant": label, "link score": r[:, 0].tolist(), "paper score": r[:, 1].tolist(),
            "held-aside error": r[:, 2].round(4).tolist(), "passes": r[:, 3].astype(int).tolist(), "seconds": round(time.time() - t0)}


GDN_VARIANTS = [
    ("ours (W 10, dim 64)", "ours", {}),
    ("authors' network, our graph size, our training", "authors_k", {}),
    ("authors' network, top-5 graph (self + 4)", "authors_5", {}),
    ("authors' network + min-max scaling", "authors_5", {"scaling": "minmax"}),
    ("authors' network + their run.sh training (W 5, batch 32, 30 passes, patience 15)", "authors_5",
     {"scaling": "minmax", "w": 5, "batch": 32, "max_epochs": 30, "patience": 15}),
    ("as above + their smoothing (4 readings) and IQR + 0.01", "authors_5",
     {"scaling": "minmax", "w": 5, "batch": 32, "max_epochs": 30, "patience": 15, "smooth": "authors", "eps": 1e-2}),
    ("as above + median/IQR from the scored data itself (their code; not allowed here)", "authors_5",
     {"scaling": "minmax", "w": 5, "batch": 32, "max_epochs": 30, "patience": 15, "smooth": "authors", "eps": 1e-2,
      "test_norm": True}),
]


def run_gdn():
    k_ours = min(10, deep.SIM_D - 1)
    N = nets()
    makers = {"ours": lambda d, w, dim: deep._nets()["GDN"](d, w, dim, k_ours),
              "authors_k": lambda d, w, dim: N["GDNAuthors"](d, w, dim, k_ours + 1),
              "authors_5": lambda d, w, dim: N["GDNAuthors"](d, w, dim, 5)}
    return [gdn_variant(label, makers[net], **kw) for label, net, kw in GDN_VARIANTS]


# ---------------------------------------------------------------- USAD: training and scores

def usad_train(net, Z, w, batch, max_epochs, patience, authors_grads, train_segs, aside_segs, seed=0, early=True):
    """authors_grads=True copies usad.py's order: backward, step, zero_grad, so gradients one decoder gets from
    the other phase's loss are carried into the next update. False: each optimiser's gradients are cleared
    before its own update, as the paper's two losses imply (ours)."""
    torch = deep._torch()
    dev = deep.device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    tr, ho = deep.ends(train_segs, w, False), deep.ends(aside_segs, w, False)
    offs = torch.arange(-w + 1, 1, device=dev)
    net = net.to(dev)
    opt1 = torch.optim.Adam(list(net.encoder.parameters()) + list(net.decoder1.parameters()), lr=1e-3)
    opt2 = torch.optim.Adam(list(net.encoder.parameters()) + list(net.decoder2.parameters()), lr=1e-3)

    def win(p):
        return Zt[p[:, None] + offs[None, :]].reshape(len(p), -1)

    def losses(x, n):
        z = net.encoder(x)
        w1, w2 = net.decoder1(z), net.decoder2(z)
        w3 = net.decoder2(net.encoder(w1))
        return (1 / n * ((x - w1) ** 2).mean() + (1 - 1 / n) * ((x - w3) ** 2).mean(),
                1 / n * ((x - w2) ** 2).mean() - (1 - 1 / n) * ((x - w3) ** 2).mean())

    best, state, wait, n = np.inf, None, 0, 0
    for n in range(1, max_epochs + 1):
        net.train()
        order = torch.as_tensor(tr)[torch.randperm(len(tr), generator=gen)].to(dev)
        for i in range(0, len(order), batch):
            x = win(order[i:i + batch])
            if authors_grads:
                l1, _ = losses(x, n)
                l1.backward(); opt1.step(); opt1.zero_grad()        # noqa: E702
                _, l2 = losses(x, n)
                l2.backward(); opt2.step(); opt2.zero_grad()        # noqa: E702
            else:
                l1, _ = losses(x, n)
                opt1.zero_grad(); l1.backward(); opt1.step()        # noqa: E702
                _, l2 = losses(x, n)
                opt2.zero_grad(); l2.backward(); opt2.step()        # noqa: E702
        net.eval()
        with torch.no_grad():
            xs = win(torch.as_tensor(ho, device=dev))
            hold = float(((xs - net.decoder1(net.encoder(xs))) ** 2).mean())
        if not early:
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            continue
        if hold < best - 1e-7:
            best, wait, state = hold, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    return net, n


def usad_scores(net, Z, w):
    """(USAD's score: 0.5 x first decoder's error + 0.5 x second decoder fed the first one's output; the first
    decoder's error alone), over the window ending at each reading."""
    torch = deep._torch()
    dev = deep.device()
    Pt = torch.as_tensor(deep.padded(Z, w - 1), dtype=torch.float32, device=dev)
    offs = torch.arange(-w + 1, 1, device=dev)
    with torch.no_grad():
        p = torch.arange(len(Z), device=dev) + (w - 1)
        x = Pt[p[:, None] + offs[None, :]].reshape(len(p), -1)
        w1 = net.decoder1(net.encoder(x))
        w3 = net.decoder2(net.encoder(w1))
        both = 0.5 * ((x - w1) ** 2).mean(1) + 0.5 * ((x - w3) ** 2).mean(1)
        return both.cpu().numpy(), ((x - w1) ** 2).mean(1).cpu().numpy()


def usad_variant(label, w=10, latent=32, caps=True, authors_grads=False, batch=256, max_epochs=100, patience=10,
                 early=True, seeds=SEEDS):
    fit, test, broken, train_segs, aside_segs = check_data()
    Zf, Zt = scaled(fit, test, "minmax")
    res, t0 = [], time.time()
    for s in seeds:
        net, n = usad_train(nets()["USADNet"](w * fit.shape[1], latent, caps), Zf, w, batch, max_epochs, patience,
                            authors_grads, train_segs, aside_segs, seed=s, early=early)
        both, first = usad_scores(net, Zt, w)
        res.append((caught(both, broken), caught(first, broken), n))
    r = np.array(res)
    return {"variant": label, "USAD score": r[:, 0].tolist(), "first decoder only": r[:, 1].tolist(),
            "passes": r[:, 2].astype(int).tolist(), "seconds": round(time.time() - t0)}


USAD_VARIANTS = [
    ("ours (W 10, latent 32)", {}),
    ("their gradient handling", {"authors_grads": True}),
    ("their layer sizes (latent W x 100)", {"latent": 1000, "caps": False}),
    ("their gradient handling + layer sizes", {"latent": 1000, "caps": False, "authors_grads": True}),
    ("their code as released (W 12, latent 1200, batch 7919, 100 passes, last pass kept)",
     {"w": 12, "latent": 1200, "caps": False, "authors_grads": True, "batch": 7919, "max_epochs": 100, "early": False}),
    ("their training only (batch 7919, 100 passes, last kept), our gradients and sizes",
     {"batch": 7919, "max_epochs": 100, "early": False}),
]


def run_usad():
    return [usad_variant(label, **kw) for label, kw in USAD_VARIANTS]


# ---------------------------------------------------------------- results

def results(which: str) -> list:
    """The saved results (computed by run_gdn / run_usad, about 1.5 hours together); computed if missing."""
    p = path(which)
    if not p.exists():
        from . import progress
        progress.stay_awake(True)
        try:
            out = run_gdn() if which == "gdn" else run_usad()
        finally:
            progress.stay_awake(False)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
    return json.loads(p.read_text())


def table() -> pd.DataFrame:
    """One row per variant and score: the catch rate per seed, their mean, and whether any seed reaches the bar."""
    rows = []
    for model, which, scores in (("GDN", "gdn", ["link score", "paper score"]), ("USAD", "usad", ["USAD score", "first decoder only"])):
        for r in results(which):
            for sc in scores:
                rows.append({"model": model, "variant": r["variant"], "score": sc,
                             **{f"seed {i}": v for i, v in enumerate(r[sc])}, "mean": float(np.mean(r[sc])),
                             "any seed reaches 0.10": bool(max(r[sc]) >= BAR)})
    return pd.DataFrame(rows).set_index(["model", "variant", "score"])
