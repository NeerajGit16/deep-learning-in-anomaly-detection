"""Fetch the public datasets into data/raw/.

    python -m adlab.download              # SKAB and Tennessee Eastman (Braatz files), about 30 MB
    python -m adlab.download skab
    python -m adlab.download tep_rieth    # Tennessee Eastman, Rieth files: about 910 MB, only on request
"""
import sys

from .data import SOURCES, download, download_rieth

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SOURCES):
        print(name, "->", download_rieth() if name == "tep_rieth" else download(name))
