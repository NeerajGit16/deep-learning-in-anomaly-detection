# Can we generate more IndPenSim batches?

*Follow-up round, part C, 17 November 2026. A feasibility check and a small pilot only; no large set was generated.*

The one deep-model win (notebook 09) rests on 10 faulty batches, and three of them carry it (notebook 11). About 80 to 100 faulty batches would narrow its range to roughly ±8 points. This note checks the ways to make more batches like the published ones.

## Verdict

**Yes, with PenSimPy, the Python port of the simulator.** After three small fixes it runs on this PC in its own Python environment, with no C++ tools. It takes about **11 seconds per batch** and makes the same five process faults as the published set, with the same hours and values. A pilot of 5 normal and 5 faulty batches has the published recipe batches' ranges on 15 of the 20 sensors used (the volume, weight, base flow, cooling water and acid flow differ; see C4), and its curved share (+12 points) is in IndPenSim's ballpark (+15). The faulty batches move the measurements the same way as batches 91, 95 and 97. Two gaps remain: every PenSimPy batch runs exactly 230 hours (the published ones run 179 to 290), and its operator and Raman-controlled strategies were not tried in the pilot.

**The original MATLAB simulator also runs, in GNU Octave** (free), with two small compatibility edits and the explicit `ode45` solver. It reproduces the published campaign most closely (uneven batch lengths, all three strategies built in) but takes about **5 minutes per batch**, and its code says "All rights reserved": using it to make data for this project should be cleared with its authors first. MATLAB itself is installed here, but its licence has expired.

**Recommended route: PenSimPy, as fixed here** (MIT licence, about 11 seconds per batch). About 100 normal and 100 faulty batches would take about 40 minutes on one core (or under 10 minutes on 8), and about 65 MB as CSV files (about 25 MB gzipped; measured on the pilot), plus about 0.2 GB for its Python environment. If uneven batch lengths and the operator and Raman-controlled strategies matter more than speed, the Octave route is the closer copy (about 17 hours on one core for 200 batches, 2 to 3 hours on 8), once its authors agree.

## C1. The authors' simulator, IndPenSim V2 (MATLAB)

| Question | Answer |
| --- | --- |
| Where | www.industrialpenicillinsimulation.com offers `IndPenSim_V2.01.zip` (217 MB; it unpacks to version 2.02, MATLAB code plus two example exports) as a direct download, no login. `IndPenSim_V2.zip`, also linked there, no longer exists. MATLAB File Exchange 49041 refuses automated access, so its page could not be read. |
| Licence | No licence file in the zip. Every code file says "All rights reserved. Copyright (c) The University of Manchester, University College London, Newcastle University and Perceptive Engineering". So using it to generate data for this project is a question for the authors; the code cannot be redistributed. |
| Same fault types | Yes: the code's fault flag has the same eight options as the published data (aeration, vessel back pressure, substrate feed, base flow, coolant flow, all of them, temperature sensor drift, pH sensor drift). The hours and sizes are written into `fctrl_indpensim.m`; a chosen size and start time need an edit of that file. |
| Control strategies | Yes: recipe-driven (`Control_strategy = 0`), operator-run (`1`, which adds random steps to the sugar and precursor feeds), and Raman-based control of the precursor (`Raman_spec = 2`, using `PAA_PLS_model.mat`). Batch lengths vary (230 hours ± 25) when `Batch_length = 1`, as in the published set. |
| MATLAB on this PC | MATLAB R2022a is installed, but its licence has expired ("License Manager Error -10"). Renewing it is out of scope (paid, and needs a login). |
| GNU Octave | **Runs, with two small edits to a working copy.** The portable GNU Octave 11.3 (816 MB zip, unpacked in a scratch folder; the winget installer needs administrator rights) runs IndPenSim V2.02 after (1) switching off a MATLAB-only figure setting (`DefaultFigureWindowStyle`) and (2) calling the solver with a function handle instead of a name with extra arguments after the options, which Octave does not accept. Octave's stiff solver `ode15s` (the one the published campaign used) fails at the first step, so the solver is `ode45`, the code's own option `solv = 1`. One normal recipe batch (233 hours, 1,165 readings) took **316 seconds** and one substrate feed batch (217 hours) **304 seconds**. The normal batch matches the published recipe batches closely (volume median 81,000 L against 76,000; base flow median 79 against 55 L/h; cooling-water peaks 373 against 437 L/h; the recipe outputs the same), and the faulty batch carries the fault in exactly the published stretches and values. |

## C2. PenSimPy (Python)

| Question | Answer |
| --- | --- |
| Where and licence | github.com/Mohan-Zhang-u/PenSimPy, MIT licence; version 1.2.5 on PyPI. |
| Faults | The same catalogue (faults 1 to 8), with the same hours and values as the MATLAB code. Chosen sizes and start times need an edit of `peni_env_setup.py` (allowed by the MIT licence). |
| Strategies | Recipe-driven (default), operator-run (`PRBS = 1`) and Raman-based control (`Raman_spec = 2`) are in the code. Only recipe-driven batches were tried in the pilot. Batch length is fixed at 230 hours (a constant). |
| Does it build here | Its fast path needs `fastodeint`, a C++ extension that is not on PyPI and needs C++ build tools (not installed, as asked). Its pure-Python path (`fast=False`) **does not run as published**: every batch collapses at about 100 to 136 hours. |
| What fixed it | Three changes, all in `adlab/pensim.py` (PenSimPy itself is not modified): (1) a stand-in module for the `fastodeint` import, never called; (2) the solver: SciPy's Dormand-Prince (RK45) with tolerances 1e-6, the solver `fastodeint` uses, with the state copied before each call, because the Python model writes into the solver's state when the pH reaches 7 (the C++ model works on a copy); (3) **the recipe**: PenSimPy's default profiles give each set-point with the time it holds *until* (its own example batches show 2-hour broth discharges), but its `Recipe` class holds each value *from* its time. Read that way, the vessel drains for 28 hours from hour 102 and the batch collapses. Rewriting the profiles fixes it. With these, no step needed any further guard. |
| Environment | Its own (`uv venv`, then `pensimpy==1.2.5` without its pinned dependencies, plus NumPy 1.26, SciPy 1.13, pandas 2.2): about 0.2 GB. The project's own environment is not touched. |

## C3. Other routes

| Route | Status |
| --- | --- |
| `cjpulmano/indpensim-py` on GitHub | A Python port of IndPenSim V2.02, reported as validated against MATLAB runs, with operator steps, faults, Raman and the PLS model. **No licence file**, so there is no permission to use or copy it; it also ports code marked "All rights reserved". Not tried. Worth asking its author. |
| `Mohan-Zhang-u/smpl` (MIT) and QuarticGym | Reinforcement-learning wrappers around PenSimPy: the same simulator, so the same fixes would apply. QuarticGym's repository could not be found under the name in its documentation. |
| The published campaign itself | 100 batches, of which 10 faulty; no more exist. |

## C4. The pilot

Five normal recipe batches (seeds 1 to 5) and five with the substrate feed fault (seeds 101 to 105), generated with `python -m adlab.pensim 5 5 1` in PenSimPy's environment; files under `anomaly_lab/data/raw/pensim_pilot/` (not committed). Compared with the 30 published normal recipe batches and the three published substrate feed batches (91, 95, 97) in the project's environment (`adlab.pensim.ranges`, `fault_spans`, `fault_response`, `curved_share`).

**Time.** 6 to 16 seconds per batch on one core (normal batches about 13 s, faulty about 9 s), 1,150 readings each.

**Ranges** (1st percentile, median, 99th percentile over all readings):

| channel | pilot: 1% | pilot: median | pilot: 99% | published recipe: 1% | published recipe: median | published recipe: 99% |
| --- | --- | --- | --- | --- | --- | --- |
| dissolved oxygen | 9.03 | 12.31 | 15.16 | 8.825 | 12.5 | 15.28 |
| volume | 57,924 | 84,342 | 95,446 | 57,440 | 75,568 | 85,979 |
| weight | 62,101 | 93,229 | 104,460 | 61,667 | 84,084 | 95,072 |
| pH | 6.391 | 6.5 | 6.557 | 6.394 | 6.501 | 6.6 |
| temperature | 297.8 | 298 | 298.8 | 297.9 | 298 | 298.6 |
| CO2 off-gas | 0.1206 | 1.704 | 2.135 | 0.1243 | 1.588 | 2.098 |
| oxygen uptake | 0.01263 | 1.324 | 2.008 | 0.02178 | 1.277 | 2.149 |
| O2 off-gas | 0.1824 | 0.1906 | 0.2047 | 0.1818 | 0.1911 | 0.2045 |
| CO2 evolution | 0.04441 | 1.488 | 1.891 | 0.04727 | 1.403 | 1.881 |
| aeration | 30 | 65 | 75 | 30 | 65 | 75 |
| sugar feed | 8 | 80 | 150 | 8 | 80 | 150 |
| acid flow | 0 | 0 | 0.5197 | 0 | 0 | 2.594 |
| base flow | 2.721 | 113.1 | 225 | 0 | 54.99 | 225 |
| cooling water | 0.0001 | 26.52 | 788.7 | 0.0001 | 33.78 | 436.6 |
| heating water | 0.0001 | 0.1573 | 160.2 | 0.0001 | 0.1225 | 187.8 |
| dilution water | 0 | 100 | 500 | 0 | 100 | 500 |
| head pressure | 0.6 | 0.9 | 1.1 | 0.6 | 0.9 | 1.1 |
| broth dumped | -4,000 | 0 | 0 | -4,000 | 0 | 0 |
| PAA flow | 0 | 10 | 10 | 0 | 10 | 10 |
| oil flow | 22 | 23 | 35 | 22 | 23 | 35 |

**Where they differ.** The volume and weight run higher (pilot median 84,000 L against 76,000), because every pilot batch runs 230 hours and the published recipe batches 179 to 290. The base flow runs about twice as high (median 113 against 55 L/h), the cooling water reaches higher peaks (99th percentile 789 against 437 L/h), and the acid flow lower ones (0.5 against 2.6 L/h). The controller outputs of the recipe (air flow, feeds, pressure, water, precursor, discharge) are the same.

**Faults.** Every pilot faulty batch carries the fault in exactly the published stretches (20-30, 76-92 and 200-214 hours) with the sugar feed held at 2 and then 20 L/h, as batches 91, 95 and 97.

**What the fault does** (76-92 hours; faulty batches' median minus the normal batches' median, in units of the normal batches' spread across batches):

| channel | pilot | published |
| --- | --- | --- |
| heating water | 896 | 58.2 |
| O2 off-gas | 19.4 | 4.28 |
| oxygen uptake | -23.4 | -4.21 |
| CO2 evolution | -3.72 | -2.65 |
| dissolved oxygen | 4.07 | 2.56 |
| CO2 off-gas | -3.22 | -2.47 |
| weight | -3.48 | -2.09 |
| volume | -2.61 | -1.6 |
| base flow | -13.9 | -1.49 |
| temperature | -0.59 | -0.87 |
| cooling water | -0.49 | -0.82 |
| pH | 0.12 | 0.31 |
| head pressure |  | 0 |

Every measurement moves the same way in the pilot as in the published batches. The pilot's shifts are larger in these units because its five normal batches, all the same length, differ less from one another.

**Curved share** (notebook 08's measure, each batch held out in turn):

| data | batches | straight share | curved share, points | range |
| --- | --- | --- | --- | --- |
| pilot (PenSimPy) | 5 | 0.65 | 11.9 | +11.0 to +13.1 |
| published recipe batches | 5 | -0.15 | 19.8 | +11.6 to +237.4 |

The pilot's +12 points is in the ballpark of IndPenSim's headline (+15, measured on 90 batches of all three strategies). The same measure on 5 published recipe batches is unstable (its straight share is negative and its range runs to +237), so five recipe batches are too few for a like-for-like comparison; a larger generated set would settle it.

## C5. Recommendation

Use **PenSimPy with the three fixes** (`adlab/pensim.py`) to generate about 100 normal and 100 faulty batches:

- **Time:** about 11 seconds per batch, so about 40 minutes on one core, or under 10 minutes running 8 batches at once.
- **Disk:** about 0.32 MB of CSV per batch, so about 65 MB for 200 batches (about 25 MB gzipped); 0.2 GB for the environment. Raman spectra, if wanted, would add about 20 MB per batch.
- **Before generating:** (1) decide whether batch lengths should vary as in the published set (PenSimPy's length is a constant; varying it needs a small change); (2) try the operator and Raman-controlled strategies, which the pilot did not; (3) fix the fault plan in advance: for the deep-model question, most faulty batches should carry the substrate feed fault (about 30 batches would give the three-batch win ten times the evidence), the rest spread over the other four process faults.
- **The alternative:** the original simulator in GNU Octave, about 5 minutes per batch (17 hours on one core for 200, 2 to 3 hours on 8 Octave processes), the same disk per batch plus about 0.8 GB for the portable Octave, after the authors agree to its use ("All rights reserved").
- **Keep the published set as the test of record**: train and set lines on generated batches, but report results on both.

## How to reproduce the pilot

PenSimPy pins old library versions, so it gets an environment of its own (outside the project's):

```bash
uv venv pensim-env --python 3.12
uv pip install --python pensim-env pensimpy==1.2.5 --no-deps
uv pip install --python pensim-env numpy==1.26.4 scipy==1.13.1 pandas==2.2.3
cd anomaly_lab
PYTHONPATH=. ../pensim-env/Scripts/python -m adlab.pensim 5 5 1     # 5 normal, 5 faulty, first seed 1
```

The batches land in `anomaly_lab/data/raw/pensim_pilot/` (not committed). The comparisons above run in the project's own environment: `adlab.pensim.ranges()`, `fault_spans()`, `fault_response()` and `curved_share()`.
