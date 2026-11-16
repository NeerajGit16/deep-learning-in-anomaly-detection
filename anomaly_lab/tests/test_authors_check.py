"""The authors'-code variants of GDN and USAD (adlab/authors_check.py), on tiny data, on the CPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adlab import authors_check as A  # noqa: E402
from adlab import deep  # noqa: E402


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(deep, "device", lambda: torch.device("cpu"))


def test_gdn_authors_network_forecasts_every_sensor_and_attends_to_itself_first():
    net = A.nets()["GDNAuthors"](d=6, w=5, dim=8, topk=3)
    net.eval()
    out = net(torch.randn(4, 5, 6))
    assert out.shape == (4, 6)
    nbr = net.neighbours()
    assert nbr.shape[0] == 6 and (nbr[:, 0] == torch.arange(6)).all()
    assert all(len(set(r.tolist())) == len(r) or r[-1] == r[-2] for r in nbr)    # no repeats except padding


def test_usad_layer_sizes_ours_and_the_authors():
    N = A.nets()["USADNet"]
    ours, theirs = N(80, 32, caps=True), N(80, 1000, caps=False)
    assert [m.out_features for m in ours.encoder if hasattr(m, "out_features")] == [40, 32, 32]
    assert [m.out_features for m in theirs.encoder if hasattr(m, "out_features")] == [40, 20, 1000]


def test_the_authors_gradient_order_changes_training():
    rng = np.random.default_rng(0)
    Z = rng.random((200, 3))
    segs = [(0, 160)], [(160, 200)]
    nets = []
    for authors in (False, True):
        torch.manual_seed(0)
        net = A.nets()["USADNet"](15, 16)
        assert (net.encoder(torch.as_tensor(Z[:5].ravel()[None, :15], dtype=torch.float32)) > 0).any()   # latent alive
        nets.append(A.usad_train(net, Z, 5, 32, 3, 10, authors, *segs, seed=0, early=False)[0])
    w0, w1 = (torch.cat([p.detach().ravel() for p in n.decoder1.parameters()]) for n in nets)
    assert not torch.allclose(w0, w1)


def test_the_saved_results_list_every_variant():
    for which, variants in (("gdn", A.GDN_VARIANTS), ("usad", A.USAD_VARIANTS)):
        if not A.path(which).exists():
            pytest.skip("results not computed on this machine")
        assert [r["variant"] for r in A.results(which)] == [v[0] for v in variants]
