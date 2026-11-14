"""Checks against the real files. Skipped until the data has been downloaded."""
import pytest

from adlab import data as D


def _have(name):
    probe = {"skab": D.RAW / "skab" / "data", "tep": D.RAW / "tep" / "d00.dat"}[name]
    return probe.exists()


@pytest.mark.skipif(not _have("skab"), reason="run `python -m adlab.download` first")
def test_skab_shape():
    ds = D.load_skab()
    assert len(ds.runs) == 34 and len(ds.channels) == 8
    for r in ds.runs:
        assert r.X.shape[1] == 8 and len(r.X) == len(r.y)
        assert r.fit.shape[0] in (104, 400)
        assert r.y.sum() > 0 and (r.y[0] == 0 or r.name == "other/2")


@pytest.mark.skipif(not _have("tep"), reason="run `python -m adlab.download` first")
def test_tep_shape_and_onset():
    ds = D.load_tep()
    assert len(ds.runs) == 21 and ds.runs[0].fit.shape == (500, 52)
    for r in ds.runs:
        assert r.X.shape == (960, 52)
        assert r.y[:160].sum() == 0 and r.y[160:].all()
    assert ds.val_normal[0].X.shape == (960, 52)
