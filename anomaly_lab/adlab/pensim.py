"""Generating IndPenSim-like batches with PenSimPy (follow-up round, part C: a feasibility pilot only).

PenSimPy (github.com/Mohan-Zhang-u/PenSimPy, MIT licence, version 1.2.5 on PyPI) is a Python port of the
IndPenSim simulator. It expects a C++ extension, fastodeint, that is not on PyPI and needs C++ build tools.
Its pure-Python path (PenSimEnv(fast=False)) does not run as published; three changes make it run here
(and a fourth, found in round 3, gives its operator-run batches their random steps; see _LatestList):

1. fastodeint is replaced by a stand-in module that is never called (fast=False is used).
2. The ODE solver: SciPy's Dormand-Prince (RK45) with tolerances 1e-6, the solver fastodeint uses
   (boost::numeric::odeint::integrate), instead of odeint. The state is copied before every call to the
   model, because PenSimPy's Python model writes to the state when the pH is 7 or more (its C++ version
   works on a copy). A non-positive hydrogen-ion concentration at a trial point is evaluated at 1e-12 and
   counted (in C++ it gives NaN without stopping); in the pilot it never happened.
3. The default recipe profiles give each value with the time it holds *until* (PenSimPy's own example
   batches show 2-hour broth discharges), but the Recipe class holds each value *from* its time, which
   drains the vessel for 28 hours. held_until rewrites the profiles.

PenSimPy needs old library versions, so it runs in its own environment (see results/
indpensim_generation_feasibility.md): this module imports nothing but NumPy and pandas at the top, and
PenSimPy only inside generate().
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# our IndPenSim channel names (indpensim.CHANNELS) -> PenSimPy's state names
CHANNEL_MAP = {"dissolved oxygen": "DO2", "volume": "V", "weight": "Wt", "pH": "pH", "temperature": "T",
               "CO2 off-gas": "CO2outgas", "oxygen uptake": "OUR", "O2 off-gas": "O2", "CO2 evolution": "CER",
               "aeration": "Fg", "sugar feed": "Fs", "acid flow": "Fa", "base flow": "Fb", "cooling water": "Fc",
               "heating water": "Fh", "dilution water": "Fw", "head pressure": "pressure", "broth dumped": "discharge",
               "PAA flow": "Fpaa", "oil flow": "Foil"}
STRATEGIES = {"recipe": {"PRBS": 0, "Raman_spec": 0}, "operator": {"PRBS": 1, "Raman_spec": 0},
              "APC": {"PRBS": 0, "Raman_spec": 2}}
SUBSTRATE_FAULT = 3                       # the simulator's fault 3: the substrate feed fault of batches 91, 95, 97
PILOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "pensim_pilot"


def held_until(profile: list) -> list:
    """Rewrite a profile of {"time", "value"} set-points, each holding until its time, as set-points each
    holding from its time (what PenSimPy's Recipe class does)."""
    out = [{"time": 0, "value": profile[0]["value"]}]
    out += [{"time": a["time"], "value": b["value"]} for a, b in zip(profile[:-1], profile[1:])]
    return [sp for i, sp in enumerate(out) if i == 0 or sp["time"] != out[i - 1]["time"]]


class _Solver:
    """Stands in for scipy.integrate.odeint inside PenSimPy (see the module's notes)."""

    def __init__(self):
        self.calls = 0
        self.clamped = 0

    def __call__(self, func, y0, t, tfirst=True, args=()):
        from scipy.integrate import solve_ivp

        def f(tt, yy):
            self.calls += 1
            yy = np.array(yy)
            if yy[6] <= 0:
                self.clamped += 1
                yy[6] = 1e-12
            return func(tt, yy, *args)

        sol = solve_ivp(f, (t[0], t[-1]), np.asarray(y0, float), method="RK45", t_eval=t, rtol=1e-6, atol=1e-6,
                        first_step=0.005)
        if not sol.success:
            raise RuntimeError(f"{sol.message} at t = {t[0]:.1f} h")
        return sol.y.T


# The substrate feed fault's hours, as the simulator codes them (steps of 0.2 h, inclusive): 20-30, 76-92, 200-214 h.
SUGAR_STRETCHES = [(100, 150), (380, 460), (1000, 1070)]
FAULT_A = {"kind": "held", "values": [2.0, 20.0, 20.0]}          # as published (the simulator's own fault 3)
# Round 3, fault B, fixed before generating: held at values the recipe uses at other times, the recipe value
# nearest fault A's (8 L/h is its value at 0-3 h, 15 L/h at 3-12 h).
FAULT_B = {"kind": "held", "values": [8.0, 15.0, 15.0]}
LENGTH_MEAN, LENGTH_SD = 230, 25                                 # batch length (h), as the MATLAB original draws it


def batch_hours(seed: int) -> int:
    """A batch's length in hours: round(230 + 25 x a normal draw), from its own generator (seed + 7777)."""
    return int(round(LENGTH_MEAN + LENGTH_SD * np.random.RandomState(seed + 7777).randn()))


def _set_length(hours: float):
    """PenSimPy reads its batch length from module constants at run time: set them for this batch."""
    import pensimpy.constants as K
    import pensimpy.data.batch_data as BD
    import pensimpy.peni_env_setup as setup
    import pensimpy.utils as U
    steps = int(round(hours * K.MINUTES_PER_HOUR / K.STEP_IN_MINUTES))
    for mod in (K, BD, setup, U):
        if hasattr(mod, "NUM_STEPS"):
            mod.NUM_STEPS = steps
        if hasattr(mod, "BATCH_LENGTH_IN_HOURS"):
            mod.BATCH_LENGTH_IN_HOURS = hours
        if hasattr(mod, "BATCH_LENGTH_IN_MINUTES"):
            mod.BATCH_LENGTH_IN_MINUTES = hours * K.MINUTES_PER_HOUR
    return steps


class _LatestList(list):
    """Round 3, fix 4. PenSimPy's operator mode stores each random step at PRBS_noise_addition[k - 1] of a list
    made at full length, then adds PRBS_noise_addition[-1], the list's last slot, which is always 0: so its
    operator batches never step. Here [-1] returns the most recently stored step, as the MATLAB original intends
    (there the list grows, so its last element is the latest)."""

    def __init__(self, n):
        super().__init__([0] * n)
        self.latest = 0

    def __setitem__(self, i, v):
        super().__setitem__(i, v)
        self.latest = v

    def __getitem__(self, i):
        return self.latest if i == -1 else super().__getitem__(i)


class _SugarOverride:
    """Wraps a RecipeCombo: during the substrate feed fault's stretches, the sugar feed the recipe hands the
    controller is held at a value ("held") or scaled ("scale"). Equivalent to the simulator's own fault 3 for
    recipe-driven batches, which also replaces the recipe's sugar feed at those steps."""

    def __init__(self, combo, variant):
        self.combo, self.variant = combo, variant

    def get_values_dict_at(self, time):
        values = self.combo.get_values_dict_at(time=time)
        k = int(round(time / 0.2))
        for i, (a, b) in enumerate(SUGAR_STRETCHES):
            if a <= k <= b:
                if self.variant["kind"] == "held":
                    values["Fs"] = self.variant["values"][i]
                else:
                    values["Fs"] = values["Fs"] * self.variant["factor"]
        return values


def generate(seed: int, fault: int = 0, strategy: str = "recipe", hours: float | str | None = None,
             sugar: dict | None = None) -> pd.DataFrame:
    """One batch: time (h), the 20 channels of notebook 08 and the fault flag. Needs PenSimPy (own environment).

    fault: the simulator's own fault number (0 none, 1 aeration, 2 back pressure, 3 substrate feed, 4 base flow,
    5 coolant flow). sugar: a substrate feed variant applied through the recipe instead ({"kind": "held",
    "values": [three values]} or {"kind": "scale", "factor": f}); its fault flag is set over SUGAR_STRETCHES.
    hours: None for PenSimPy's 230 h, a number, or "vary" for batch_hours(seed)."""
    import sys
    stub = Path(__file__).resolve().parent / "_fastodeint_stub"
    if str(stub) not in sys.path:
        sys.path.insert(0, str(stub))
    import pensimpy.peni_env_setup as setup
    from pensimpy.data import constants as C
    from pensimpy.examples.recipe import Recipe, RecipeCombo

    length = batch_hours(seed) if hours == "vary" else (230 if hours is None else hours)
    steps = _set_length(length)
    captured = {}
    original = getattr(setup, "_adlab_original_get_dataframe", setup.get_dataframe)
    setup._adlab_original_get_dataframe = original

    def keep(batch_data, include_raman):
        captured["x"] = batch_data
        return original(batch_data, include_raman)

    solver = _Solver()
    setup.odeint, setup.get_dataframe = solver, keep
    profiles = {C.FS: C.FS_DEFAULT_PROFILE, C.FOIL: C.FOIL_DEFAULT_PROFILE, C.FG: C.FG_DEFAULT_PROFILE,
                C.PRES: C.PRESS_DEFAULT_PROFILE, C.DISCHARGE: C.DISCHARGE_DEFAULT_PROFILE, C.WATER: C.WATER_DEFAULT_PROFILE,
                C.PAA: C.PAA_DEFAULT_PROFILE}
    recipe = RecipeCombo(recipe_dict={k: Recipe(held_until(prof), k) for k, prof in profiles.items()})
    if sugar is not None:
        recipe = _SugarOverride(recipe, sugar)
    env = setup.PenSimEnv(recipe_combo=recipe, fast=False)
    reset = env.reset

    def reset_with_steps():                       # fix 4: the operators' random steps
        observation, x = reset()
        x.PRBS_noise_addition = _LatestList(steps)
        return observation, x

    env.reset = reset_with_steps
    env.ctrl_flags.Faults = fault
    for flag, value in STRATEGIES[strategy].items():
        setattr(env.ctrl_flags, flag, value)
    np.random.seed(seed)                          # the operators' random steps use NumPy's global generator
    t0 = time.time()
    env.get_batches(random_seed=seed, include_raman=False)
    x = captured["x"]
    out = pd.DataFrame({name: np.asarray(getattr(x, attr).y, float).ravel()[:steps] for name, attr in CHANNEL_MAP.items()})
    out.insert(0, "time", np.asarray(x.Fs.t, float).ravel()[:steps])
    flag = (np.asarray(x.Fault_ref.y, float).ravel()[:steps] > 0).astype(int)
    if sugar is not None:
        k = np.arange(1, steps + 1)
        flag = np.zeros(steps, int)
        for a, b in SUGAR_STRETCHES:
            flag[(k >= a) & (k <= b)] = 1
    out["fault"] = flag
    out.attrs.update({"seconds": time.time() - t0, "ode calls": solver.calls, "clamped": solver.clamped,
                      "seed": seed, "fault type": fault, "strategy": strategy, "hours": length})
    return out


# ---------------------------------------------------------------- round 3: the generated campaign

CAMPAIGN = Path(__file__).resolve().parents[1] / "data" / "raw" / "pensim_campaign"


def campaign_plan() -> list[dict]:
    """Every batch of the campaign, with its seed (fixed here, before generating). Lengths vary as published."""
    groups = [("normal_recipe", 100, 1000, "recipe", 0, None), ("normal_operator", 60, 2000, "operator", 0, None),
              ("A_recipe", 30, 3000, "recipe", 3, None), ("B_recipe", 30, 3100, "recipe", 0, FAULT_B),
              ("C80_recipe", 15, 3200, "recipe", 0, {"kind": "scale", "factor": 0.8}),
              ("C50_recipe", 15, 3300, "recipe", 0, {"kind": "scale", "factor": 0.5}),
              ("C20_recipe", 15, 3400, "recipe", 0, {"kind": "scale", "factor": 0.2}),
              ("A_operator", 15, 3500, "operator", 3, None),
              ("aeration_recipe", 10, 4100, "recipe", 1, None), ("pressure_recipe", 10, 4200, "recipe", 2, None),
              ("base_flow_recipe", 10, 4400, "recipe", 4, None), ("coolant_recipe", 10, 4500, "recipe", 5, None)]
    plan = []
    for group, n, first, strategy, fault, sugar in groups:
        for i in range(n):
            plan.append({"name": f"{group}_{i:03d}", "group": group, "seed": first + i, "strategy": strategy,
                         "fault": fault, "sugar": sugar, "hours": batch_hours(first + i)})
    return plan


def _one(job):
    path = CAMPAIGN / f"{job['name']}.csv.gz"
    if path.exists():
        return {**job, "skipped": True}
    b = generate(job["seed"], job["fault"], job["strategy"], hours=job["hours"], sugar=job["sugar"])
    b.to_csv(path, index=False)
    return {**job, "seconds": round(b.attrs["seconds"], 1), "readings": len(b), "clamped": b.attrs["clamped"]}


def _stay_awake(on: bool):
    import os
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def run_campaign(workers: int = 8):
    """Generate every batch of campaign_plan() not yet on disk, in parallel processes, with a running estimate."""
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    plan = campaign_plan()
    (CAMPAIGN / "plan.json").write_text(json.dumps(plan, indent=1))
    todo = [j for j in plan if not (CAMPAIGN / f"{j['name']}.csv.gz").exists()]
    log, t0 = [], time.time()
    _stay_awake(True)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, j) for j in todo]
            for i, f in enumerate(as_completed(futures), 1):
                log.append(f.result())
                el = time.time() - t0
                print(f"{i}/{len(todo)} batches | {el / 60:.1f} min so far | about {el / i * (len(todo) - i) / 60:.1f} min left",
                      flush=True)
    finally:
        _stay_awake(False)
    old = json.loads((CAMPAIGN / "log.json").read_text()) if (CAMPAIGN / "log.json").exists() else []
    (CAMPAIGN / "log.json").write_text(json.dumps(old + log, indent=1))


def main(argv=None):
    """python -m adlab.pensim <normal batches> <faulty batches> [first seed]: the part C pilot (230 h batches),
    written to data/raw/pensim_pilot/. python -m adlab.pensim campaign [workers]: the round 3 campaign."""
    import json
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    if argv and argv[0] == "campaign":
        run_campaign(int(argv[1]) if len(argv) > 1 else 8)
        return
    args = argv + ["5", "5", "1"][len(argv):]
    n_normal, n_faulty, first = int(args[0]), int(args[1]), int(args[2])
    PILOT.mkdir(parents=True, exist_ok=True)
    log = []
    jobs = [(first + i, 0) for i in range(n_normal)] + [(first + 100 + i, SUBSTRATE_FAULT) for i in range(n_faulty)]
    for seed, fault in jobs:
        b = generate(seed, fault)
        name = f"{'faulty' if fault else 'normal'}_seed{seed}"
        b.to_csv(PILOT / f"{name}.csv", index=False)
        log.append({"batch": name, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in b.attrs.items()},
                    "readings": len(b)})
        print(json.dumps(log[-1]), flush=True)
    (PILOT / "pilot_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- comparing the pilot with the published batches
# (in the project's own environment; reads the CSV files written by main)

def load_pilot(folder: Path = PILOT) -> dict:
    """{"normal": [DataFrame, ...], "faulty": [...]} from data/raw/pensim_pilot/."""
    out = {"normal": [], "faulty": []}
    for p in sorted(Path(folder).glob("*.csv")):
        out["faulty" if p.name.startswith("faulty") else "normal"].append(pd.read_csv(p))
    return out


def _published(kind: str = "recipe") -> list[pd.DataFrame]:
    from . import indpensim
    d = indpensim.readings()
    pick = d[(d["strategy"] == kind) & (d["faulty batch"] == 0)] if kind != "faulty" else d[d["batch"].isin([91, 95, 97])]
    return [x[["time"] + indpensim.CHANNELS + ["fault"]].reset_index(drop=True) for _, x in pick.groupby("batch")]


def ranges() -> pd.DataFrame:
    """Each channel's 1st percentile, median and 99th percentile: pilot normal batches against the 30 published
    normal recipe batches; and batch lengths."""
    from . import indpensim
    pilot = pd.concat(load_pilot()["normal"])
    pub = pd.concat(_published("recipe"))
    rows = {}
    for c in indpensim.CHANNELS:
        q = lambda x: np.quantile(x.dropna(), [0.01, 0.5, 0.99])
        a, b = q(pilot[c]), q(pub[c])
        rows[c] = {"pilot: 1%": a[0], "pilot: median": a[1], "pilot: 99%": a[2],
                   "published recipe: 1%": b[0], "published recipe: median": b[1], "published recipe: 99%": b[2]}
    out = pd.DataFrame(rows).T
    out.index.name = "channel"
    lengths = {"pilot": [float(b["time"].iloc[-1]) for b in load_pilot()["normal"]],
               "published recipe": [float(b["time"].iloc[-1]) for b in _published("recipe")]}
    out.attrs["batch hours"] = {k: (min(v), max(v)) for k, v in lengths.items()}
    return out


def curved_share(n_boot: int = 500) -> pd.DataFrame:
    """The curved share (same measure as notebook 08: same moment, pooled, whole batches held out) on the 5 pilot
    normal batches and on 5 published recipe fit batches, each batch its own round."""
    from . import indpensim, structure
    pilot = [b[indpensim.CHANNELS].to_numpy(float) for b in load_pilot()["normal"]]
    bs, roles = indpensim.normal_batches(), indpensim.roles()
    pub = [b["X"] for b, r in zip(bs, roles) if r == "fit" and b["strategy"] == "recipe"][:len(pilot)]
    rows = {}
    for label, runs in (("pilot (PenSimPy)", pilot), ("published recipe batches", pub)):
        folds = [np.full(len(r), i) for i, r in enumerate(runs)]          # each batch its own round
        res = structure.cached(f"pensim_pilot_curved__{label.split()[0]}", lambda runs=runs, folds=folds: structure.curvature_by_run(
            runs, indpensim.CHANNELS, folds=folds, n_boot=n_boot, verbose=False))
        rows[label] = {"batches": len(runs), "straight share": res["straight"], "curved share, points": 100 * res["curved"],
                       "range": f"{100 * res['curved_low']:+.1f} to {100 * res['curved_high']:+.1f}"}
    out = pd.DataFrame(rows).T
    out.index.name = "data"
    return out


def fault_spans() -> pd.DataFrame:
    """Per faulty batch (pilot and published 91, 95, 97): the hours its fault flag is on, and the sugar feed then."""
    rows = {}
    for label, frames in (("pilot", load_pilot()["faulty"]), ("published", _published("faulty"))):
        for i, b in enumerate(frames):
            on = b["fault"].to_numpy() == 1
            t = b["time"].to_numpy()
            edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            spans = [(float(t[a]), float(t[e - 1])) for a, e in zip(edges[::2], edges[1::2])]
            rows[f"{label} {i + 1}"] = {"fault on (h)": ", ".join(f"{a:g}-{e:g}" for a, e in spans),
                                        "sugar feed while on": ", ".join(f"{v:g}" for v in sorted(set(np.round(b.loc[on, 'sugar feed'], 3))))}
    return pd.DataFrame(rows).T


def fault_response() -> pd.DataFrame:
    """During 76-92 h: each channel's median in the faulty batches minus the normal batches' median at the same
    hours, in units of the normal batches' spread; pilot and published side by side."""
    from . import indpensim

    def shift(normal, faulty):
        win = lambda b: b[(b["time"] >= 76) & (b["time"] <= 92)][indpensim.CHANNELS].median()
        n = pd.DataFrame([win(b) for b in normal])
        f = pd.DataFrame([win(b) for b in faulty])
        return ((f.median() - n.median()) / n.std().replace(0, np.nan))

    pilot = load_pilot()
    out = pd.DataFrame({"pilot": shift(pilot["normal"], pilot["faulty"]),
                        "published": shift(_published("recipe"), _published("faulty"))})
    out.index.name = "channel"
    return out.reindex(out["published"].abs().sort_values(ascending=False).index)
