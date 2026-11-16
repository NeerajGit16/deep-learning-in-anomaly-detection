"""Across datasets (adlab/across.py) and the label placer it uses (plots.place_labels). No data needed."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from adlab import across, plots  # noqa: E402


def test_straight_side_counts_every_detector_that_is_not_deep_except_curved_link():
    side = across.straight_side()
    assert "Curved-link residual" not in side
    for name in ["Straight forecaster", "PCA, scored like the autoencoder", "Straight-link residual",
                 "Kinked-link residual", "Rolling z-score", "PCA (T2 + SPE)"]:
        assert name in side, name
    assert not any(n in side for n in ["Autoencoder", "LSTM forecaster"])


def test_labels_of_points_close_together_do_not_overlap():
    fig, ax = plt.subplots(figsize=(4, 3))
    xs, ys = [0.0, 0.1, 0.2, 5.0], [0.0, 0.05, -0.05, 3.0]
    ax.plot(xs, ys, "o")
    anns = plots.place_labels(ax, xs, ys, ["first label", "second label", "third label", "far away"])
    fig.canvas.draw()
    boxes = [a.get_window_extent(fig.canvas.get_renderer()) for a in anns]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (i, j)
    plt.close(fig)
