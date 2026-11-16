"""Fetch the public datasets into data/raw/. Nothing is fetched twice.

    uv run python -m adlab.download                  # SKAB and Tennessee Eastman (Braatz files), about 30 MB
    uv run python -m adlab.download skab tep         # the same, named
    uv run python -m adlab.download tep_rieth        # Tennessee Eastman, Rieth files: about 910 MB
    uv run python -m adlab.download tep_multimode    # six-mode Tennessee Eastman: about 0.7 GB read out of 143 GB, about 40 minutes
    uv run python -m adlab.download metal_etch       # the LAM 9600 etcher: about 2 MB
    uv run python -m adlab.download indpensim        # IndPenSim: about 0.5 GB (2.6 GB unpacked)
    uv run python -m adlab.download all              # every dataset that can be fetched this way

Run from the anomaly_lab folder. One dataset cannot be fetched automatically; see CRANFIELD.
"""
import sys

CRANFIELD = """The Cranfield multiphase flow rig data is not fetched automatically.
  1. Download the "cvatutor" package, version 1.1.0, from MATLAB File Exchange, entry 50938
     (https://www.mathworks.com/matlabcentral/fileexchange/50938). A MathWorks account may be needed.
  2. Unpack it so that the .mat files sit in
         anomaly_lab/data/raw/cranfield/cvatutor-1.1.0/CVACaseStudy/
     (Training.mat and FaultyCase1.mat to FaultyCase6.mat).
  The notebooks and tests that need it skip until it is there."""


def fetch(name: str):
    from . import data
    if name in data.SOURCES:
        return data.download(name)
    if name == "tep_rieth":
        return data.download_rieth()
    if name == "tep_multimode":
        from . import multimode
        multimode.download()
        return multimode.download_faults()
    if name == "metal_etch":
        from . import etch
        return etch.download()
    if name == "indpensim":
        from . import indpensim
        folder = indpensim.download()
        indpensim.process()                       # builds the small cache of the process columns
        return folder
    if name == "cranfield":
        print(CRANFIELD)
        return None
    raise SystemExit(f"unknown dataset {name!r}; choose from skab, tep, tep_rieth, tep_multimode, metal_etch, indpensim, cranfield, all")


ALL = ["skab", "tep", "tep_rieth", "tep_multimode", "metal_etch", "indpensim", "cranfield"]

if __name__ == "__main__":
    names = sys.argv[1:] or ["skab", "tep"]
    if names == ["all"]:
        names = ALL
    for name in names:
        print(name, "->", fetch(name))
