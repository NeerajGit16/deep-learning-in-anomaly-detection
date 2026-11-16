"""Figures for the notebook. Every function returns a matplotlib Figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FAULT_TINT = "#fbe3d6"  # shading for "fault is happening"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#efeeeb", "#f2a07f", "#c9461a"])


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlelocation": "left",
        "axes.titlesize": 10, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    })


style()


def _shade(ax, y, x=None):
    """Shade the stretches where y == 1."""
    x = np.arange(len(y)) if x is None else np.asarray(x)
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(y, int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(x[a], x[min(b, len(x) - 1)], color=FAULT_TINT, lw=0, zorder=0)


# ---------------------------------------------------------------- the idea in pictures

def three_kinds_of_fault(seed: int = 0):
    """Toy example of the three kinds of fault the project is about, with what catches each."""
    from .detectors import make

    rng = np.random.default_rng(seed)
    n, t = 400, np.arange(400)
    y = ((t >= 300) & (t < 360)).astype(int)
    f = y == 1
    wave = np.sin(t / 12)

    a1, b1 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    b1[f] += 4                                                   # one sensor jumps
    a2 = 1.5 * wave + rng.normal(0, 0.15, n)
    b2 = a2 + rng.normal(0, 0.15, n)
    b2[f] = -a2[f] + rng.normal(0, 0.15, f.sum())                # straight-line link breaks
    a3 = 1.5 * wave + rng.normal(0, 0.05, n)
    b3 = a3 ** 3 + rng.normal(0, 0.15, n)
    mu, sd, d = _pca_direction(np.c_[a3[:300], b3[:300]])
    b3[f] = mu[1] + sd[1] * d[1] / d[0] * (a3[f] - mu[0]) / sd[0] + rng.normal(0, 0.15, f.sum())
    # curved link breaks: during the fault B follows PCA's straight line instead of the curve

    cases = [
        ("1. One sensor goes off on its own", a1, b1,
         "A simple check on each sensor catches this."),
        ("2. A straight-line link breaks", a2, b2,
         "Each sensor stays in its normal range.\nPCA's straight line catches it."),
        ("3. A curved link breaks", a3, b3,
         "The fault sits right on PCA's straight line,\nso PCA misses it. A model that learns\nthe curve could catch it."),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), gridspec_kw={"height_ratios": [1, 1.35]})
    for col, (title, a, b, note) in enumerate(cases):
        X = np.c_[a, b]
        fit = X[:300]
        caught = {}
        for key, params in (("rolling_z", {"window": 1}), ("pca", {"var": 0.5})):
            caught[key] = _catch(make(key, **params).fit(fit).score(X), y, fit_len=300)
        ax = axes[0, col]
        _shade(ax, y)
        ax.plot(t, a, color=BLUE, lw=1, label="sensor A")
        ax.plot(t, b, color=ORANGE, lw=1, label="sensor B")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("time")
        if col == 0:
            ax.legend(frameon=False, loc="upper left", ncol=2)
        ax = axes[1, col]
        ax.scatter(a[~f], b[~f], s=7, color="#9ec5f4", label="normal")
        ax.scatter(a[f], b[f], s=12, color=ORANGE, label="during fault")
        if col > 0:
            mu, sd, d = _pca_direction(fit)
            span = np.linspace(-4, 4, 2)
            ax.plot(mu[0] + sd[0] * span * d[0], mu[1] + sd[1] * span * d[1], color=INK2, ls="--", lw=1.2,
                    label="PCA's straight line")
            ax.set_xlim(a.min() - 0.3, a.max() + 0.3)
            ax.set_ylim(b.min() - 0.5, b.max() + 0.5)
        ax.set_xlabel("sensor A")
        ax.set_ylabel("sensor B")
        ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        ax.text(0.02, 0.02, f"Caught at 1 false alarm in 100:\nper-sensor check {caught['rolling_z']:.0%}"
                f"   PCA {caught['pca']:.0%}", transform=ax.transAxes, va="bottom", fontsize=8.5, color=INK2)
        if col == 2:
            ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Three kinds of fault. Top: the two sensors over time (fault shaded). "
                 "Bottom: sensor A against sensor B.", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def _pca_direction(fit):
    """Mean, s.d. and main direction of the (standardised) data, as the PCA detector sees it."""
    mu, sd = fit.mean(0), fit.std(0)
    _, _, vt = np.linalg.svd((fit - mu) / sd, full_matrices=False)
    d = vt[0] if vt[0][0] > 0 else -vt[0]
    return mu, sd, d


def _catch(score, y, fit_len):
    """Share of fault samples above the line that 1% of normal (non-fit) samples cross."""
    s, yy = score[fit_len:], np.asarray(y[fit_len:], bool)
    s_normal = np.r_[score[:fit_len], s[~yy]]
    thr = np.quantile(s_normal, 0.99)
    return float(np.mean(s[yy] > thr))


# ---------------------------------------------------------------- looking at the datasets

def stacked(df: pd.DataFrame, columns: list[str], x: str, title: str, labels: dict | None = None):
    """Each column on its own strip, shared time axis, fault stretches shaded."""
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 0.62 * len(columns) + 1.0), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, columns):
        _shade(ax, df["fault"].values, df[x].values)
        ax.plot(df[x], df[c], color=BLUE, lw=0.9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(-0.005, 0.5, (labels or {}).get(c, c), transform=ax.transAxes, ha="right", va="center",
                fontsize=8, color=INK2)
    axes[0].set_title(title)
    axes[-1].set_xlabel(x)
    fig.tight_layout()
    return fig


def heatmap(table: pd.DataFrame, title: str, vmax: float | None = None, diverging: bool = False,
            annotate: bool = True, fmt: str = "{:.1f}", figsize=None, colorbar_label: str = ""):
    """Coloured grid of a table, optionally with the numbers written in."""
    vals = table.values.astype(float)
    vmax = vmax if vmax is not None else np.nanmax(np.abs(vals))
    fig, ax = plt.subplots(figsize=figsize or (min(1.0 * table.shape[1] + 3, 16), 0.42 * table.shape[0] + 1.6))
    im = ax.imshow(vals, cmap=DIV if diverging else SEQ, vmin=-vmax if diverging else 0, vmax=vmax,
                   aspect="auto")
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right", fontsize=8)
    if annotate:
        for (i, j), v in np.ndenumerate(vals):
            if np.isfinite(v):
                dark = abs(v) > 0.6 * vmax
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="white" if dark else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label(colorbar_label, color=INK2)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def all_sensors_over_time(df: pd.DataFrame, fit: np.ndarray, channels: list[str], x: str, title: str,
                          onset: float | None = None, clip: float = 5.0):
    """Every sensor as a row, coloured by how far it is from its normal level (blue low, orange high)."""
    mu, sd = fit.mean(0), fit.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    z = np.clip((df[channels].values - mu) / sd, -clip, clip).T
    fig, ax = plt.subplots(figsize=(12, 0.16 * len(channels) + 1.6))
    xs = df[x].values
    im = ax.imshow(z, cmap=DIV, vmin=-clip, vmax=clip, aspect="auto",
                   extent=[xs[0], xs[-1], len(channels) - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(len(channels)), channels, fontsize=6.5)
    if onset is not None:
        ax.axvline(onset, color=INK, lw=1.2, ls="--")
        ax.text(onset, len(channels) - 1, " fault starts ", fontsize=8, color=INK, ha="right", va="bottom",
                bbox={"facecolor": BG, "edgecolor": "none", "pad": 1})
    ax.set_xlabel(x)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label("distance from normal level\n(in normal wiggles)", color=INK2)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- how a detector is scored

def how_scoring_works(df: pd.DataFrame, sensor: str, score: np.ndarray, x: str, far: float = 0.01,
                      title: str = ""):
    """Top: one sensor. Bottom: the detector's weirdness score and the alarm line.

    The alarm line is set so that `far` of the normal samples cross it (false alarms).
    The share of fault samples above the line is the catch rate.
    """
    y = df["fault"].values.astype(bool)
    thr = np.quantile(score[~y], 1 - far)
    caught = np.mean(score[y] > thr)
    xs = df[x].values
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    _shade(a0, y, xs)
    a0.plot(xs, df[sensor], color=BLUE, lw=0.9)
    a0.set_ylabel(sensor)
    a0.set_title(title or f"One sensor ({sensor}); the fault is shaded")
    _shade(a1, y, xs)
    a1.plot(xs, score, color=INK2, lw=0.9)
    above = score > thr
    a1.scatter(xs[above & y], score[above & y], s=8, color=AQUA, zorder=3, label="fault caught")
    a1.scatter(xs[above & ~y], score[above & ~y], s=10, color=ORANGE, zorder=3, label="false alarm")
    a1.axhline(thr, color=INK, lw=1, ls="--")
    a1.text(xs[0], thr, f" alarm line: {far:.0%} of normal samples cross it", va="bottom", fontsize=8)
    a1.set_ylabel("weirdness score")
    a1.set_xlabel(x)
    a1.legend(frameon=False, loc="upper left")
    a1.set_title(f"The detector's score. Share of fault samples above the line: {caught:.0%}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- pilot results

def pilot_overall(results: dict, far: float = 0.01, titles: dict | None = None):
    """titles: optional display name per dataset key (default: the key in capitals)."""
    names = list(results)
    titles = {n: n.upper() for n in names} | (titles or {})
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 3.3), squeeze=False)
    for ax, name in zip(axes[0], names):
        df = pd.DataFrame(results[name]["summary"]).sort_values("dr_at_1pct")
        yy = np.arange(len(df))
        ax.hlines(yy, df["dr_ci90_low"], df["dr_ci90_high"], color=BLUE, lw=2)
        ax.plot(df["dr_at_1pct"], yy, "o", color=BLUE, ms=7, mec=BG, mew=1.5)
        for y0, v in zip(yy, df["dr_at_1pct"]):
            ax.text(v, y0 + 0.28, f"{v:.0%}", ha="center", fontsize=8, color=INK)
        ax.axvline(far, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_yticks(yy, df["method"])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("share of fault samples caught at 1 false alarm in 100\n(dashed line = random guessing)")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(f"{titles[name]}: bars = range it could plausibly be")
    fig.tight_layout()
    return fig


def pilot_by_type(results: dict, titles: dict | None = None):
    """titles: optional display name per dataset key (default: the key in capitals)."""
    names = list(results)
    titles = {n: n.upper() for n in names} | (titles or {})
    sizes = [len({r["group"] for r in results[n]["by_group"]}) for n in names]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.4 * len(names) + 0.12 * max(sizes)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        df = pd.DataFrame(results[name]["by_group"])
        order = pd.DataFrame(results[name]["summary"]).sort_values("dr_at_1pct")["method"].tolist()
        piv = df.pivot(index="method", columns="group", values="dr_at_1pct").loc[order[::-1]]
        ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.6 else INK)
        ax.set_yticks(range(len(piv)), piv.index)
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=40, ha="right", fontsize=7.5)
        ax.set_title(f"{titles[name]}: % of fault samples caught at 1 false alarm in 100, by kind of fault")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- how curved are the links?

def dial_check(df: pd.DataFrame):
    """Simulated data: the measured curved share against the dial that set the true curvature."""
    g = df.groupby("dial")[["straight", "curved"]].mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(g.index, g["straight"], "o-", color=BLUE, lw=2, label="explained by straight lines")
    ax.plot(g.index, g["curved"], "o-", color=ORANGE, lw=2, label="extra explained by curves")
    ax.axhline(0, color=INK2, lw=0.8)
    for x, v in g["curved"].items():
        ax.text(x, v + 0.035, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xlabel("dial: how curved the simulated links are (0 = straight, 1 = fully curved)")
    ax.set_ylabel("share of sensor movement")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="center right")
    ax.set_title("Check on simulated data: the measure follows the dial")
    fig.tight_layout()
    return fig


def curved_share(summaries: dict, title: str = "How curved are the links between sensors? Bars = plausible range",
                 low: str = "curved_low", high: str = "curved_high", mark: float | None = None, points: bool = False):
    """One row per dataset: average curved share with its 90% interval.

    low, high   which range to draw (e.g. "curved_low_units" for the range from resampling visits)
    mark        draw a second dashed line at this value (e.g. 0.03, the go/no-go mark)
    points      label the axis in points with one decimal place (+2.2) instead of whole percent;
                values under one point are labelled with two decimals (+0.07)
    """
    def show(v):
        if not points:
            return f"{v:+.1%}"
        return f"{100 * v:+.2f}" if abs(100 * v) < 0.95 else f"{100 * v:+.1f}"

    names = list(summaries)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(names) + 1.5))
    if mark is not None:
        ax.axvline(mark, color=ORANGE, lw=1, ls=(0, (3, 3)))
        ax.text(mark, len(names) - 0.35, f" {100 * mark:+.0f} point mark" if points else f" {mark:+.0%} mark",
                color=ORANGE, fontsize=8, va="top")
    for i, n in enumerate(names):
        s = summaries[n]
        ax.hlines(i, s[low], s[high], color=BLUE, lw=2)
        ax.plot(s["curved"], i, "o", color=BLUE, ms=7, mec=BG, mew=1.5)
        ax.text(s["curved"], i + 0.25, show(s["curved"]), ha="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.xaxis.set_major_formatter((lambda v, _: f"{100 * v:+.1f}") if points else (lambda v, _: f"{v:+.0%}"))
    ax.set_xlabel("extra share explained by curves, in points (0 = links are straight)" if points
                  else "extra share of sensor movement explained by curves (0 = links are straight)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Cranfield flow rig

def flows_over_time(normal: dict, labels: dict | None = None):
    """One panel per normal run: the water flow (blue, left) and the air flow (orange, right) over the run."""
    fig, axes = plt.subplots(len(normal), 1, figsize=(9, 1.7 * len(normal) + 0.6), sharex=True)
    for ax, (name, X) in zip(np.atleast_1d(axes), normal.items()):
        minutes = np.arange(len(X)) / 60
        ax.plot(minutes, X[:, 8], color=BLUE, lw=0.8)
        ax.set_ylabel(f"{name}: water, kg/s", color=BLUE)
        ax.set_ylim(0, 7)
        right = ax.twinx()
        right.plot(minutes, X[:, 7], color=ORANGE, lw=0.8)
        right.set_ylabel("air, Sm3/s", color=ORANGE)
        right.set_ylim(0.015, 0.047)
        right.spines["top"].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
    np.atleast_1d(axes)[-1].set_xlabel("minutes into the run")
    np.atleast_1d(axes)[0].set_title("Normal running: each run steps through the rig's flow settings")
    fig.tight_layout()
    return fig


def curved_by_sensor(table: pd.DataFrame, title: str, low: str = "curved, low (visits)",
                     high: str = "curved, high (visits)", mark: float | None = 0.10):
    """One row per sensor: its curved share with a range, most curved at the top."""
    t = table.sort_values("curved")
    fig, ax = plt.subplots(figsize=(7.6, 0.27 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.hlines(y, t[low], t[high], color=BLUE, lw=2)
    ax.plot(t["curved"], y, "o", color=BLUE, ms=5, mec=BG, mew=1)
    ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
    if mark is not None:
        ax.axvline(mark, color=ORANGE, lw=1, ls=(0, (3, 3)))
        ax.text(mark, len(t) - 0.4, f" {100 * mark:+.0f} points", color=ORANGE, fontsize=8, va="top")
    labels = [f"{w} ({s})" for s, w in zip(t.index, t["what it measures"])] if "what it measures" in t else list(t.index)
    ax.set_yticks(y, labels, fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:+.0f}")
    ax.set_xlabel("extra share of the sensor's movement explained by curves, in points")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def cva_check(result: dict, title: str):
    """The authors' two monitoring numbers over one faulty run, with their limits and the published fault shaded."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    x = np.arange(len(result["T2"])) + result["first"] + 1
    for ax, key in zip(axes, ("T2", "SPE")):
        for a, b in result["spans"]:
            ax.axvspan(a, b, color=FAULT_TINT, lw=0, zorder=0)
        ax.semilogy(x, result[key], color=BLUE, lw=0.8)
        ax.axhline(result[f"{key}_limit"], color=ORANGE, lw=1.5, ls=(0, (4, 2)))
        ax.set_ylabel(key)
    axes[-1].set_xlabel("reading number (shaded: the published fault)")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def scatter_by_setting(X: np.ndarray, label: np.ndarray, pairs: list, names: list, air: tuple, water: tuple,
                       every: int = 5, between: int = -1, describe: dict | None = None):
    """One row per (sensor, its best single predictor): the same scatter twice, coloured by the
    water flow setting (left) and by the air flow setting (right). Readings taken between
    settings are grey.

    If the points of every setting fall along one bent line, the link is a real curve. If each
    setting sits in its own cluster, off any common line, the "curve" is level shifts between settings.
    The left panel's title names the sensor on the upright axis; it is coloured by water flow.
    """
    describe = describe or {}
    keep = np.arange(0, len(X), every)
    lab = np.asarray(label)[keep]
    at = lab != between
    fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3.3 * len(pairs)), squeeze=False)
    water_colours = [SEQ(v) for v in np.linspace(0.25, 1.0, len(water))]
    air_colours = ["#f6c9a0", "#eb9a5a", "#d9601f", "#8f3a0c"][:len(air)]
    for row, (target, partner) in zip(axes, pairs):
        x, y = X[keep, names.index(partner)], X[keep, names.index(target)]
        for ax, colours, levels, index, what in ((row[0], water_colours, water, lab % len(water), "water flow, kg/s"),
                                                 (row[1], air_colours, air, lab // len(water), "air flow, Sm3/s")):
            ax.scatter(x[~at], y[~at], s=3, color="#c9c8c3", lw=0, label="between settings")
            for k, level in enumerate(levels):
                pick = at & (index == k)
                ax.scatter(x[pick], y[pick], s=3, color=colours[k], lw=0, label=f"{level:g}")
            ax.set_xlabel(describe.get(partner, partner))
            ax.set_ylabel(target)
            ax.grid(color=GRID, lw=0.6)
            ax.legend(title=what, fontsize=7, title_fontsize=7, markerscale=3, frameon=False, loc="best")
        row[0].set_title(describe.get(target, target))
        row[1].set_title("the same points, coloured by air flow")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- multi-mode Tennessee Eastman

def scatter_by_mode(X: np.ndarray, mode: np.ndarray, pairs: list, names: list, describe: dict | None = None,
                    every: int = 4):
    """One row per (sensor, its best single predictor), one small panel per mode.

    Each panel shows all the readings in grey and that mode's readings in blue. With six modes,
    six colours in one panel would not be told apart reliably, so each mode gets its own panel.
    If every mode's blue dots lie along one shared bent line, the link is a real curve. If each
    mode is its own patch, off any shared line, the "curve" is the modes sitting in different places.
    """
    describe = describe or {}
    keep = np.arange(0, len(X), every)
    modes = sorted(np.unique(mode))
    fig, axes = plt.subplots(len(pairs), len(modes), figsize=(2.05 * len(modes), 2.5 * len(pairs) + 0.4),
                             squeeze=False, sharex="row", sharey="row")
    for row, (target, partner) in zip(axes, pairs):
        x, y, m = X[keep, names.index(partner)], X[keep, names.index(target)], np.asarray(mode)[keep]
        for ax, k in zip(row, modes):
            ax.scatter(x, y, s=2, color="#d5d4cf", lw=0, rasterized=True)
            ax.scatter(x[m == k], y[m == k], s=2, color=BLUE, lw=0, rasterized=True)
            ax.set_title(f"mode {k}", fontsize=9)
            ax.grid(color=GRID, lw=0.6)
            ax.tick_params(labelsize=7)
        row[0].set_ylabel(target, fontsize=8)
        row[len(modes) // 2].set_xlabel(f"up: {describe.get(target, target)}    across: {describe.get(partner, partner)}",
                                        fontsize=8)
    fig.tight_layout()
    return fig


def scatter_panels(panels: list, across: int = 3):
    """Small scatter panels, one per entry: {"x", "y", "title", "across" (x label), "up" (y label)}."""
    rows = -(-len(panels) // across)
    fig, axes = plt.subplots(rows, across, figsize=(3.6 * across, 3.1 * rows + 0.2), squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, p in zip(axes.ravel(), panels):
        ax.scatter(p["x"], p["y"], s=3, color=BLUE, lw=0, alpha=0.6, rasterized=True)
        ax.set_title(p["title"], fontsize=9)
        ax.set_xlabel(p["across"], fontsize=8)
        ax.set_ylabel(p["up"], fontsize=8)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- batch processes

def _wrap(text: str, width: int = 58) -> str:
    """Long panel titles on two lines, so neighbouring panels' titles do not run into each other."""
    import textwrap
    return textwrap.fill(text, width)


def stage_colours(n: int) -> list:
    """Colours for n stages of a batch: three fixed hues for up to three stages; for more, one
    blue from light (early) to dark (late), because the stages are in order."""
    if n <= 3:
        return [BLUE, ORANGE, AQUA][:n]
    return [SEQ(v) for v in np.linspace(0.3, 1.0, n)]


def _spread(x: np.ndarray, rng, few: int = 20) -> np.ndarray:
    """A reading with few distinct values (whole numbers, a valve with 4 positions) spread a little
    sideways, by up to a third of the gap between its values, so dots on the same value can be seen."""
    values = np.unique(x)
    if len(values) > few or len(values) < 2:
        return x
    return x + rng.uniform(-1, 1, len(x)) * np.diff(values).min() / 3


def scatter_by_stage(X: np.ndarray, stage: np.ndarray, pairs: list, names: list, describe: dict | None = None,
                     stage_names: dict | None = None, every: int = 1, across: int = 2):
    """One panel per (sensor, its best single predictor), each reading coloured by the stage of the batch.

    If the points of every stage fall along one bent line, the link is a real curve. If each stage is
    its own patch, off any shared line, the "curve" is the stages sitting in different places.
    Readings that take 20 or fewer distinct values are spread a little sideways (_spread).
    """
    describe, stage_names = describe or {}, stage_names or {}
    rng = np.random.default_rng(0)
    keep = np.arange(0, len(X), every)
    st = np.asarray(stage)[keep]
    levels = sorted(np.unique(st))
    colours = stage_colours(len(levels))
    rows = -(-len(pairs) // across)
    fig, axes = plt.subplots(rows, across, figsize=(5.0 * across, 3.9 * rows + 0.3), squeeze=False)
    for ax in axes.ravel()[len(pairs):]:
        ax.set_visible(False)
    for ax, (target, partner) in zip(axes.ravel(), pairs):
        x, y = _spread(X[keep, names.index(partner)], rng), _spread(X[keep, names.index(target)], rng)
        for level, colour in zip(levels, colours):
            pick = st == level
            ax.scatter(x[pick], y[pick], s=4, color=colour, lw=0, alpha=0.6, rasterized=True,
                       label=stage_names.get(level, str(level)))
        ax.set_title(_wrap(describe.get(target, target)), fontsize=9)
        ax.set_xlabel(_wrap(describe.get(partner, partner)), fontsize=8)
        ax.set_ylabel(target, fontsize=8)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, markerscale=3, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def over_a_batch(runs: list, stages: list, channels: list, sensors: list, describe: dict | None = None,
                 what: str = "", across: int = 2):
    """One panel per sensor: each batch as a thin line over time, with the stage changes marked."""
    describe = describe or {}
    rows = -(-len(sensors) // across)
    fig, axes = plt.subplots(rows, across, figsize=(5.4 * across, 2.6 * rows + 0.3), squeeze=False)
    for ax in axes.ravel()[len(sensors):]:
        ax.set_visible(False)
    for ax, name in zip(axes.ravel(), sensors):
        j = channels.index(name)
        for X, st in zip(runs, stages):
            ax.plot(np.arange(len(X)), X[:, j], color=BLUE, lw=0.9, alpha=0.6)
            for k in np.flatnonzero(np.diff(np.asarray(st))) + 1:
                ax.axvline(k, color=INK2, lw=0.6, ls=(0, (2, 3)), alpha=0.5)
        ax.set_title(_wrap(describe.get(name, name)), fontsize=9)
        ax.set_xlabel("seconds into the batch (dotted: stage changes)", fontsize=8)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(labelsize=7)
    if what:
        fig.suptitle(what, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def place_labels(ax, xs, ys, texts, fontsize: float = 7.5, pad_px: float = 3.0):
    """Label points without overlaps: each label starts just right of its point, then labels that would
    overlap are pushed apart vertically (in screen space), and a thin line joins a moved label to its point."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    anns = [ax.annotate(t, (x, y), xytext=(7, -3), textcoords="offset points", fontsize=fontsize, color=INK,
                        va="center", ha="left") for x, y, t in zip(xs, ys, texts)]
    start = [a.xyann for a in anns]
    for _ in range(60):                                       # sweep until no two labels overlap
        boxes = [a.get_window_extent(renderer) for a in anns]
        points = ax.transData.transform(np.column_stack([xs, ys]))
        moved = False
        order = np.argsort([b.y0 for b in boxes])
        for i_pos, i in enumerate(order):
            for j in order[i_pos + 1:]:
                bi, bj = boxes[i], boxes[j]
                if bi.x1 + pad_px < bj.x0 or bj.x1 + pad_px < bi.x0:
                    continue                                  # no horizontal overlap
                need = bi.y1 + pad_px - bj.y0                 # how far j must go up to clear i
                if need > 0:
                    a_i, a_j = anns[i], anns[j]
                    a_i.xyann = (a_i.xyann[0], a_i.xyann[1] - need / 2 * to_pt)
                    a_j.xyann = (a_j.xyann[0], a_j.xyann[1] + need / 2 * to_pt)
                    moved = True
                    boxes = [a.get_window_extent(renderer) for a in anns]
        for k, a in enumerate(anns):                          # keep labels off other points' markers
            b = a.get_window_extent(renderer)
            for m, (px, py) in enumerate(points):
                if m != k and b.x0 - 4 < px < b.x1 + 4 and b.y0 - 4 < py < b.y1 + 4:
                    a.xyann = (a.xyann[0], a.xyann[1] + (b.y1 - py + 6 if py < (b.y0 + b.y1) / 2 else -(py - b.y0 + 6)) * to_pt)
                    moved = True
        if not moved:
            break
    for a, s in zip(anns, start):
        if abs(a.xyann[1] - s[1]) > 4:
            ax.annotate("", a.xy, xytext=(a.xyann[0] - 1, a.xyann[1]), textcoords="offset points",
                        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6, "shrinkA": 0, "shrinkB": 3})
    return anns
