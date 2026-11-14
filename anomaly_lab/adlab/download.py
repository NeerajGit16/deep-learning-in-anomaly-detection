"""Fetch the public datasets into data/raw/.

    python -m adlab.download            # both
    python -m adlab.download skab
"""
import sys

from .data import SOURCES, download

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SOURCES):
        print(name, "->", download(name))
