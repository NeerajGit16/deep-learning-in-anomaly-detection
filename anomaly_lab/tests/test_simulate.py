"""The recipe generator of part D (adlab/simulate.recipe_batches)."""
import numpy as np

from adlab import simulate as S

N_OUT = len(S.REC_OUTPUTS)


def test_batches_have_the_outputs_then_the_measurements_and_vary_in_length():
    bs = S.recipe_batches(8, dial=1.0, seed=0)
    assert all(b["X"].shape[1] == N_OUT + len(S.REC_MEASURES) for b in bs)
    lengths = [len(b["X"]) for b in bs]
    assert len(set(lengths)) > 1 and 150 < np.mean(lengths) < 350
    assert all(b["fault"].sum() == 0 and not b["faulty"] for b in bs)
    assert all(set(np.unique(b["stage"])) == set(range(10)) for b in bs)


def test_dial_zero_makes_every_output_a_straight_ramp_and_dial_one_gives_recipe_steps():
    ramp = np.vstack([b["X"][:, :N_OUT] for b in S.recipe_batches(3, dial=0.0, seed=0)])
    centred = ramp - ramp.mean(0)
    s = np.linalg.svd(centred, compute_uv=False)
    assert s[1] / s[0] < 1e-6                         # all outputs on one line
    steps = np.vstack([b["X"][:, :N_OUT] for b in S.recipe_batches(3, dial=1.0, seed=0)])
    assert all(4 <= len(np.unique(np.round(steps[:, k], 9))) <= 8 for k in range(N_OUT))


def test_the_same_seed_gives_the_same_batches_and_every_seed_follows_the_same_recipe():
    a, b = S.recipe_batches(2, seed=3), S.recipe_batches(2, seed=3)
    assert all(np.array_equal(x["X"], y["X"]) for x, y in zip(a, b))
    values = [set(np.round(np.unique(S.recipe_batches(3, seed=s)[0]["X"][:, 2]), 9)) for s in (3, 4, 5)]
    assert values[0] == values[1] == values[2]


def test_a_fault_holds_one_output_inside_its_range_at_a_value_it_does_not_take_then():
    normal = np.vstack([b["X"][:, :N_OUT] for b in S.recipe_batches(20, dial=1.0, seed=0)])
    for b in S.recipe_batches(12, dial=1.0, seed=1, faulty=True):
        on = b["fault"] == 1
        assert 0.05 < on.mean() < 0.12
        k = S.REC_OUTPUTS.index(b["output held"])
        held = np.unique(b["X"][on, k])
        assert len(held) == 1
        lo, hi = normal[:, k].min(), normal[:, k].max()
        assert lo - 1e-9 <= held[0] <= hi + 1e-9
        clean = S.recipe_batches(12, dial=1.0, seed=1)[b["batch"]]
        assert b["fault kind"] in ("between steps", "wrong step")
        if len(clean["X"]) == len(b["X"]):
            gap = np.abs(held[0] - clean["X"][on, k])
            assert gap.min() >= 0.25 * (hi - lo) - 1e-6 or b["fault kind"] == "between steps"


def test_holding_the_feed_moves_the_measurements_and_holding_the_oil_does_not():
    """Without noise, batch 0 of a faulty run and of a clean run with the same seed share length and drift."""
    seen = {}
    for seed in range(60):
        f = S.recipe_batches(1, dial=1.0, seed=seed, faulty=True, noise=0.0)[0]
        if f["output held"] in ("feed", "oil") and f["output held"] not in seen:
            c = S.recipe_batches(1, dial=1.0, seed=seed, noise=0.0)[0]
            moved = np.abs(f["X"][:, N_OUT:] - c["X"][:, N_OUT:]).max()
            seen[f["output held"]] = moved
        if len(seen) == 2:
            break
    assert seen["feed"] > 1e-3 and seen["oil"] < 1e-12


def test_fault_size_moves_the_held_value_part_of_the_way_and_keeps_the_same_faults():
    full = S.recipe_batches(6, dial=1.0, seed=1, faulty=True, noise=0.0)
    half = S.recipe_batches(6, dial=1.0, seed=1, faulty=True, noise=0.0, fault_size=0.5)
    clean = S.recipe_batches(6, dial=1.0, seed=1, noise=0.0)
    for f, h, c in zip(full, half, clean):
        assert f["output held"] == h["output held"] and np.array_equal(f["fault"], h["fault"])
        if len(c["X"]) != len(f["X"]):
            continue
        k = S.REC_OUTPUTS.index(f["output held"])
        on = np.flatnonzero(f["fault"])
        start = c["X"][on[0], k]
        assert np.allclose(h["X"][on, k], start + 0.5 * (f["X"][on, k] - start))
