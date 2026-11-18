# anomaly_lab

When does deep learning beat simple statistics at spotting faults in factory sensor data?

The project works baselines-first: simple methods are scored on public industrial data before any deep model is added. The plan, in plain words, is in the project doc *Anomaly Detection — Problem Statement*.

## Start here: `main.ipynb`

Open `main.ipynb` in Jupyter or VS Code and run it top to bottom. It walks through:

1. The big idea: three kinds of fault, and which tool catches which.
2. SKAB (a real water pump): tables and pictures.
3. Tennessee Eastman (a simulated chemical plant): tables and pictures.
4. How a detector is judged: the "1 false alarm in 100" rule.
5. Eight simple detectors on both datasets: the bar deep learning has to beat.
6. Playground: change settings, look at other faults, write your own detector.
7. What's next, and why we will also generate data.

The first run downloads the data (about 30 MB) and scores everything, which takes a few minutes. Later runs read the saved scores in `results/cache/`.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/) from the repository root (Python 3.12; packages are listed in `pyproject.toml` and pinned in `uv.lock`).

```bash
uv sync                                  # creates .venv with everything needed
uv pip install --no-deps TSB-AD==1.5     # optional: one extra score (VUS-PR) used in the reports
uv run pytest                            # checks everything works
```

## What's in the folder

| Path | What it holds |
| --- | --- |
| `main.ipynb` | The notebook: run this |
| `02_how_curved.ipynb` | First research results: how curved each dataset is, and what kind of fault each fault is |
| `03_fresh_runs.ipynb` | The same detectors on 1,000 Tennessee Eastman runs they have never seen (Rieth files): which results hold |
| `04_delayed_links.ipynb` | Whether links that arrive late (one sensor following another after a delay) change the picture |
| `05_cranfield.ipynb` | A real flow rig run at 20 flow settings (Cranfield): how curved its links are, and the detectors on its 16 faulty runs |
| `06_multimode_tep.ipynb` | The chemical plant run in six operating modes (Reinartz files): how curved its links are when the modes are pooled |
| `07_metal_etch.ipynb` | A plasma etcher, wafer by wafer (a batch process): how curved its links are, and the detectors on its 21 faulty wafers |
| `08_indpensim.ipynb` | A simulated penicillin fermentation, batch by batch (IndPenSim): how curved its links are, and the detectors on its 10 faulty batches |
| `09_deep_models.ipynb` | Deep models (autoencoder, LSTM, GDN, USAD), each beside a simple twin, on all six datasets |
| `10_across_datasets.ipynb` | All six datasets side by side: the curved share, the kinked share, the repeat share and three gaps in catching faults |
| `11_indpensim_deep_dive.ipynb` | Why the autoencoder did well on the penicillin batches, ingredient by ingredient |
| `12_mechanism_generator.ipynb` | Artificial batches with that ingredient on a dial |
| `13_pensim_campaign.ipynb` | A generated campaign of 320 penicillin batches (PenSimPy) |
| `14_positive_control.ipynb` | Artificial data built so that a deep model should win |
| `adlab/data.py` | Loads SKAB, Tennessee Eastman (Braatz and Rieth files) and the Cranfield flow rig; plain-English sensor tables (`SKAB_INFO`, `TEP_INFO`, `CRANFIELD_INFO`) |
| `adlab/explore.py` | Summary tables: faults per kind, how far each sensor moves, which sensors move together |
| `adlab/plots.py` | All the figures |
| `adlab/detectors.py` | Every simple detector: the per-sensor checks, PCA, isolation forest, matrix profile, the link residual detectors (straight, curved, kinked, told the stage or mode), the schedule-aware checks and round 4's classic ones (nearest normal reading on all sensors, unseen value check, dynamic PCA, CVA, kernel PCA) |
| `adlab/folds.py` | The fold audit and balanced folds (round 4) |
| `adlab/round4.py` | Round 4's reports: the fold audit's changes, the classic detectors, false alarms before the fault, the repeat share |
| `adlab/posctl.py` | The positive control's generator and reports (notebook 14) |
| `adlab/round5.py` | Round 5's checks: the metal etcher without its edited setting columns, and the deep models trained to the end (notebook 09, sections 16 and 17) |
| `adlab/posctl2.py` | The positive control's second version: every sensor truly bent, faults of three sizes (notebook 14, sections 9 to 14) |
| `adlab/deepdive.py`, `adlab/followup.py` | Notebook 11's ingredient detectors, and the follow-up checks of notebooks 09 and 11 |
| `adlab/pensim.py`, `adlab/campaign.py` | Generating penicillin batches with PenSimPy (in its own environment), and the generated campaign's setups and reports |
| `adlab/mechanism.py`, `adlab/authors_check.py` | Notebook 12's generator study; GDN and USAD against the authors' code |
| `adlab/structure.py` | Curvature measure and fault labelling |
| `adlab/cranfield.py` | The curvature measures for the Cranfield rig, whose runs each pass through many flow settings |
| `adlab/multimode.py` | Fetches and loads the six-mode Tennessee Eastman runs, and puts its curvature measures together |
| `adlab/batch.py` | Batch processes: whole batches held out, the stage of a batch in place of the mode, and alarm lines set per group of batches |
| `adlab/deep.py` | The deep models and their twins (PyTorch, optional), their training (notebook 09's budget and round 5's longer one), the PCA-plus-correction autoencoder, self-checks and reports |
| `adlab/across.py` | The table and chart across datasets, and the kinked share |
| `adlab/download.py` | `python -m adlab.download all` fetches every dataset that needs no login |
| `adlab/indpensim.py` | Downloads and loads the IndPenSim batches, says what each column is, and puts their measures and detectors together |
| `adlab/etch.py` | Loads the metal etch wafers and puts their curvature measures and detectors together |
| `adlab/remote.py` | Reads parts of a very large remote file without downloading all of it |
| `adlab/cva.py` | The Cranfield authors' own monitoring method, ported from their example; first used to check we read their data as they do, and from round 4 also a detector on every dataset |
| `adlab/simulate.py` | A first small simulator with a curvature dial |
| `adlab/metrics.py` | Catch rate at a fixed false-alarm rate, and other scores. For Tennessee Eastman the alarm line is set on fault-free readings from the same hours of a run as the faults |
| `adlab/stats.py` | How much of a result could be luck (resampling) |
| `adlab/run.py` | Scores every run with every detector and builds the results tables |
| `adlab/progress.py` | Progress bars for the slow steps, and a progress log for jobs running in the background |
| `tests/` | Checks that the code and data behave as expected |
| `results/` | Saved scores, tables and figures |
| `results/decision_point.md` | The curvature numbers from notebooks 05 and 06 set against the go/no-go rule |

## Slow steps

Anything slow (scoring runs, the curvature measure, resampling, downloads) shows a progress bar with the time remaining. Pass `verbose=False` to switch a bar off.

While a slow step runs on Windows, the computer is asked not to go to sleep by itself, and the screen is kept on too: on laptops with Modern Standby, an idle screen turning off can pause a running job for hours even when sleep is blocked (closing the lid still sends it to sleep).

To follow a job running in the background, set `ADLAB_PROGRESS_LOG=1` before starting it. It then writes one line per step to `results/progress.log`: the time, what finished, how many are done out of how many, and the estimated time left.

## Command-line version

This does the same scoring and report as section 5 of the notebook, without Jupyter.

```bash
cd anomaly_lab
uv run python -m adlab.run score
uv run python -m adlab.run report    # writes results/pilot_report.md and figures
```

## Data sources

- **SKAB:** Katser and Kozitsin, Skoltech, 2020. https://github.com/waico/SKAB (GPL-3.0).
- **Cranfield multiphase flow rig:** Ruiz-Carcel, Cao, Mba, Lao and Samuel, 2015, *Statistical process monitoring of a multiphase flow facility*, Control Engineering Practice 42, 74-88. Data: the `cvatutor` package, version 1.1.0 (MATLAB File Exchange 50938). It is not downloaded automatically: unpack it by hand so that the `.mat` files sit in `data/raw/cranfield/cvatutor-1.1.0/CVACaseStudy/`.
- **Metal etch:** Wise, Gallagher, Butler, White and Barna, 1999, *A comparison of principal component analysis, multiway principal component analysis, trilinear decomposition and parallel factor analysis for fault detection in a semiconductor etch process*, Journal of Chemometrics 13, 379-396. Data: Eigenvector Research, `MACHINE_Data.mat` from https://eigenvector.com/resources/data-sets/metal-etch-data-for-fault-detection-evaluation/ (about 2 MB, downloaded by `etch.download()`). How the faults were made: http://www.eigenvector.com/Docs/Sensitivity.pdf.
- **IndPenSim:** Goldrick, Duran-Villalobos, Jankauskas, Lovett, Farid and Lennox, 2019, *Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process*, Computers and Chemical Engineering 130, 106471. Data: Mendeley Data, [doi:10.17632/pdnjz7zz5x](https://doi.org/10.17632/pdnjz7zz5x), version 2 (CC BY 4.0), about 0.5 GB, downloaded by `indpensim.download()`.
- **Tennessee Eastman:**
  - Six operating modes: Reinartz, Kulahci and Ravn, 2021, *An extended Tennessee Eastman simulation dataset for fault-detection and decision support systems*, Computers and Chemical Engineering 149, 107281. Data: data.dtu.dk, [doi:10.11583/DTU.13385936](https://doi.org/10.11583/DTU.13385936) (public domain). The full set is 143 GB; `06_multimode_tep.ipynb` fetches only the fault-free and mode-transition runs (about 0.5 GB, about 17 minutes) by reading them out of the remote files.
  - Plant model: Downs and Vogel, 1993.
  - Data files: Braatz lab, University of Illinois, via https://github.com/camaramm/tennessee-eastman-profBraatz.
  - Fresh runs: Rieth, Amsel, Tran and Cook, 2017, *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation*, Harvard Dataverse, [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1) (public domain). About 910 MB, downloaded only when `03_fresh_runs.ipynb` is run or with `uv run python -m adlab.download tep_rieth`.
