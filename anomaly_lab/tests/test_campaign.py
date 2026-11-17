"""The generated campaign's splits and setups (adlab/campaign.py). Needs data/raw/pensim_campaign (generated with
PenSimPy in its own environment: python -m adlab.pensim campaign); skipped otherwise."""
import pytest

from adlab import campaign as C
from adlab import indpensim

needs_data = pytest.mark.skipif(not (C.CAMPAIGN / "plan.json").exists(), reason="needs the generated campaign")


@needs_data
def test_the_plan_has_every_batch_with_its_seed():
    p = C.plan()
    assert len(p) == 320 and p["seed"].is_unique
    counts = p.groupby("group").size()
    assert counts["normal_recipe"] == 100 and counts["normal_operator"] == 60 and counts["A_recipe"] == 30
    assert counts["B_recipe"] == 30 and counts["A_operator"] == 15 and counts["coolant_recipe"] == 10


@needs_data
def test_splits_are_fixed_and_no_batch_is_both_fitted_on_and_used_for_a_line():
    s = C.split()
    assert [len(s[k]) for k in ("recipe fit", "recipe line", "operator fit", "operator line")] == [60, 40, 40, 20]
    fit = {b["name"] for k in ("recipe fit", "operator fit") for b in s[k]}
    line = {b["name"] for k in ("recipe line", "operator line") for b in s[k]}
    assert not fit & line


@needs_data
def test_the_two_setups():
    rec, mix = C.dataset("recipe"), C.dataset("mixed")
    assert len(rec.fit_lengths) == 60 and set(rec.fit_groups) == {"recipe"} and len(rec.val_normal) == 40
    assert len(mix.fit_lengths) == 100 and mix.fit_groups.count("operator") == 40 and len(mix.val_normal) == 60
    assert {r.group for r in rec.runs} == set(C.SCORED["recipe"]) and {r.group for r in mix.runs} == {"A_recipe", "A_operator"}
    assert rec.output_columns == indpensim.OUTPUT_COLUMNS and rec.deep_settings["lstm"]["W"] == 5
    assert all(r.y.sum() > 0 for r in rec.runs)
