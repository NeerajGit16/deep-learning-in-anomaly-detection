"""The Cranfield flow rig: how curved are its links, measured the ways the plan asks for.

The rig's normal runs each pass through about 20 flow settings, one after another. That changes
how readings have to be held out (see `folds`), and it is why this has its own module. The
measures themselves are in structure.py; the data is loaded in data.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, progress, structure

HEADLINE = ("T1", "T2", "T3")      # the go/no-go measure uses all three normal runs
WITHOUT_T3 = ("T1", "T2")          # shown beside it: T3 holds over half of some sensors' movement

# Past readings of the other sensors offered to the models, in readings (= seconds) back.
# The first four cover the same span of time as 1, 2, 5, 10 readings on the chemical plant
# (3, 6, 15, 30 minutes). The last covers the same number of readings (1 to 10 seconds), in
# case the rig's delays are shorter than three minutes. Chosen before anything was measured.
LAG_SETS = {"+ 3 min back": (180,), "+ 3, 6 min back": (180, 360), "+ 3, 6, 15 min back": (180, 360, 900),
            "+ 3, 6, 15, 30 min back": (180, 360, 900, 1800), "+ 1, 2, 5, 10 s back": (1, 2, 5, 10)}

SAME_MOMENT = "same moment (the go/no-go measure)"
TOLD = "same moment, setting told to the straight model"
SEPARATE = "same moment, a separate straight model per setting"
LITERAL = "same moment, blocks of time within each run"


def normal(names: tuple = HEADLINE) -> tuple[list, list]:
    """The chosen normal runs and, for each, the flow setting of every reading."""
    runs = [data.cranfield_normal()[k] for k in names]
    return runs, [data.flow_settings(X) for X in runs]


def folds(names: tuple = HEADLINE) -> pd.DataFrame:
    """Two ways of holding readings out, and what share of held-out readings sits at a flow
    setting the models never saw in training (which would test guessing at an unseen setting,
    not curvature)."""
    runs, labels = normal(names)
    blocks = []
    for label in labels:
        fold = np.empty(len(label), int)
        for f, idx in enumerate(structure.blocked_folds(len(label), 5)):
            fold[idx] = f
        blocks.append(fold)
    staggered = structure.staggered_folds(labels)[0]
    used = " + ".join(names)
    out = pd.DataFrame({"held-out readings at a setting never seen in training": {
        f"{used}: five blocks of time within each run": structure.unseen_share(labels, blocks),
        f"{used}: whole visits to a setting, staggered across runs": structure.unseen_share(labels, staggered)}})
    out.index.name = "runs used, and how readings are held out"
    return out


def measures(names: tuple = HEADLINE, n_boot: int = 500, verbose: bool = True) -> dict:
    """Every pooled curvature measure for the chosen normal runs: {name: summary}.

    Readings are held out as whole visits to a setting, staggered across runs (see
    structure.staggered_folds), except in the LITERAL row. Ranges come from resampling runs and,
    because there are only two or three runs, also from resampling visits ("..._units").
    Each measure is saved in results/cache as soon as it is done.
    """
    runs, labels = normal(names)
    held, units = structure.staggered_folds(labels)
    tag = f"cranfield_{''.join(names)}__"
    common = dict(n_boot=n_boot, verbose=verbose)

    def by_run(what, **kw):
        return lambda: structure.curvature_by_run(runs, data.CRANFIELD_CHANNELS, what=f"rig {'+'.join(names)}, {what}",
                                                  units=units, **common, **kw)

    out = {SAME_MOMENT: structure.cached(tag + "same_moment", by_run("same moment", folds=held)),
           TOLD: structure.cached(tag + "setting_levels", by_run("setting told", folds=held, modes=labels)),
           LITERAL: structure.cached(tag + "time_blocks", by_run("blocks of time"))}
    for name, lags in LAG_SETS.items():
        out[name] = structure.cached(tag + "lags_" + "_".join(map(str, lags)), by_run(name, folds=held, lags=lags))
    progress.note(f"rig {'+'.join(names)}: pooled measures done")
    return out


def separate_lines(names: tuple = HEADLINE, n_boot: int = 500, verbose: bool = True) -> pd.DataFrame:
    """The stronger way of telling the straight model the setting: a straight model of its own per setting.

    Kept apart from `measures` because it does not work on this data: every run visits each
    setting once, so a setting's model is fitted on a single steady stretch and its slopes are
    fitted to noise. The numbers show how far off it lands.
    """
    runs, labels = normal(names)
    held, units = structure.staggered_folds(labels)
    s = structure.cached(f"cranfield_{''.join(names)}__setting_told", lambda: structure.curvature_by_run(
        runs, data.CRANFIELD_CHANNELS, what=f"rig {'+'.join(names)}, separate lines", units=units, n_boot=n_boot,
        verbose=verbose, folds=held, modes=labels, told="separate"))
    t = s["table"]["straight"]
    return pd.DataFrame({SEPARATE: {"straight share, average": s["straight"], "worst sensor": t.min(),
                                    "sensors with a straight share below 0": int((t < 0).sum()),
                                    "curved share, average": s["curved"]}})


def most_curved(by_name: dict, name: str = SAME_MOMENT, over: float = 0.10) -> list[str]:
    """The sensors that gain more than `over` from curves in one measure, most curved first."""
    t = by_name[name]["table"]["curved"].sort_values(ascending=False)
    return list(t.index[t > over])


def best_partner(names: tuple = HEADLINE, sensors: list | None = None) -> pd.DataFrame:
    """For each sensor: the one other sensor that on its own says most about it.

    "share explained" is how much of the sensor's movement is explained by the average of the
    sensor within 50 equal-sized slices of the other sensor's readings, so a bent link counts
    as much as a straight one (see structure.best_single_predictor).
    """
    X = np.vstack(normal(names)[0])
    rows = {}
    for s in sensors or data.CRANFIELD_CHANNELS:
        i, share = structure.best_single_predictor(X, data.CRANFIELD_CHANNELS.index(s))
        rows[s] = {"best single predictor": data.CRANFIELD_CHANNELS[i], "share explained": share}
    out = pd.DataFrame(rows).T
    out.index.name = "sensor"
    return out


def curved_scatter(by_name: dict, names: tuple = HEADLINE, name: str = SAME_MOMENT):
    """Each sensor that gains over 10 points from curves, plotted against its best single predictor
    and coloured by flow setting (plots.scatter_by_setting), to see a real curve from separate clusters."""
    from . import plots
    sensors = most_curved(by_name, name)
    partners = best_partner(names, sensors)["best single predictor"]
    runs, labels = normal(names)
    info = data.CRANFIELD_INFO.set_index("sensor")
    describe = {s: f"{info.loc[s, 'what it measures']} ({s}), {info.loc[s, 'unit']}" for s in data.CRANFIELD_CHANNELS}
    return plots.scatter_by_setting(np.vstack(runs), np.concatenate(labels), list(zip(sensors, partners)),
                                    data.CRANFIELD_CHANNELS, data.CRANFIELD_AIR, data.CRANFIELD_WATER, describe=describe)


def measure_table(by_name: dict) -> pd.DataFrame:
    """One row per measure: shares, the curved share's two ranges, and how many sensors gain from curves."""
    rows = {name: {"straight share": s["straight"], "curved share": s["curved"],
                   "curved, low (runs)": s["curved_low"], "curved, high (runs)": s["curved_high"],
                   "curved, low (visits)": s["curved_low_units"], "curved, high (visits)": s["curved_high_units"],
                   "sensors gaining over 0.10 from curves": s["sensors_with_curved_over_10pts"],
                   "readings": s["readings"]}
            for name, s in by_name.items()}
    out = pd.DataFrame(rows).T.astype({"sensors gaining over 0.10 from curves": int, "readings": int})
    out.index.name = "measure"
    return out


def sensor_table(by_name: dict, name: str = SAME_MOMENT) -> pd.DataFrame:
    """Every sensor's straight and curved share for one measure, with both ranges, most curved first."""
    t = by_name[name]["table"]
    out = pd.concat([data.CRANFIELD_INFO.set_index("sensor")["what it measures"].reindex(t.index), t], axis=1)
    return out.rename(columns={"curved_low": "curved, low (runs)", "curved_high": "curved, high (runs)",
                               "curved_low_units": "curved, low (visits)", "curved_high_units": "curved, high (visits)"}
                      ).sort_values("curved", ascending=False)


def within_settings(names: tuple = HEADLINE, n_boot: int = 500, verbose: bool = True, whole_visits: bool = False) -> pd.DataFrame:
    """The curvature measure inside each flow setting on its own (structure.curvature_within), saved.

    whole_visits=True is the first design tried (hold out each visit whole), kept to show why it was dropped.
    """
    def compute():
        runs, labels = normal(names)
        return structure.curvature_within(runs, labels, data.CRANFIELD_CHANNELS, data.setting_name, n_boot=n_boot,
                                          verbose=verbose, what=f"rig {'+'.join(names)}, within settings",
                                          whole_visits=whole_visits)
    key = "within_settings" if whole_visits else "within_settings_blocks"
    return structure.cached(f"cranfield_{''.join(names)}__{key}", compute)


def within_summary(table: pd.DataFrame) -> pd.Series:
    """A few numbers from a within_settings table: the typical setting, and the most curved one."""
    return pd.Series({"settings": len(table), "straight share, middle setting": table["straight share"].median(),
                      "curved share, middle setting": table["curved share"].median(),
                      "curved share, most curved setting": table["curved share"].max(),
                      "settings with a curved share above 0.03": int((table["curved share"] > 0.03).sum())})
