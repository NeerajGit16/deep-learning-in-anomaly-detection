# anomaly_lab

When does deep learning beat simple statistics at spotting faults in factory sensor data?

The project works baselines-first: simple methods are scored on public industrial data before any deep model is added. The plan, in plain words, is in the project doc *Anomaly Detection — Problem Statement*.

## Start here: `main.ipynb`

Open `main.ipynb` in Jupyter or VS Code and run it top to bottom. It walks through:

1. The big idea: three kinds of fault, and which tool catches which.
2. SKAB (a real water pump): tables and pictures.
3. Tennessee Eastman (a simulated chemical plant): tables and pictures.
4. How a detector is judged: the "1 false alarm in 100" rule.
5. Six simple detectors on both datasets: the bar deep learning has to beat.
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
| `adlab/data.py` | Loads SKAB and Tennessee Eastman; plain-English sensor tables (`SKAB_INFO`, `TEP_INFO`) |
| `adlab/explore.py` | Summary tables: faults per kind, how far each sensor moves, which sensors move together |
| `adlab/plots.py` | All the figures |
| `adlab/detectors.py` | The six simple detectors, plus the straight-link and curved-link residual detectors |
| `adlab/structure.py` | Curvature measure and fault labelling |
| `adlab/simulate.py` | A first small simulator with a curvature dial |
| `adlab/metrics.py` | Catch rate at a fixed false-alarm rate, and other scores |
| `adlab/stats.py` | How much of a result could be luck (resampling) |
| `adlab/run.py` | Scores every run with every detector and builds the results tables |
| `tests/` | Checks that the code and data behave as expected |
| `results/` | Saved scores, tables and figures |

## Command-line version

This does the same scoring and report as section 5 of the notebook, without Jupyter.

```bash
cd anomaly_lab
uv run python -m adlab.run score
uv run python -m adlab.run report    # writes results/pilot_report.md and figures
```

## Data sources

- **SKAB:** Katser and Kozitsin, Skoltech, 2020. https://github.com/waico/SKAB (GPL-3.0).
- **Tennessee Eastman:**
  - Plant model: Downs and Vogel, 1993.
  - Data files: Braatz lab, University of Illinois, via https://github.com/camaramm/tennessee-eastman-profBraatz.
