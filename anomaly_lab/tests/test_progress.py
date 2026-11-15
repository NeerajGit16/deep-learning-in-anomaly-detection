"""Progress bars and the progress log."""
import re

import numpy as np

from adlab import progress, run
from adlab.data import Dataset, Run


def test_times_read_naturally():
    assert [progress.nice_time(s) for s in (5, 90, 4000)] == ["5 s", "1 min 30 s", "1 h 6 min"]


def test_nothing_is_shown_or_written_inside_tests(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ADLAB_PROGRESS_LOG", str(tmp_path / "progress.log"))
    assert progress.in_tests()
    assert list(progress.track(range(5), "counting")) == [0, 1, 2, 3, 4]
    out = capsys.readouterr()
    assert out.out == "" and out.err == "" and not (tmp_path / "progress.log").exists()


def test_the_progress_log_gets_one_line_per_step_with_time_left(tmp_path, monkeypatch):
    """Outside the tests, with the log switched on: time, what, done out of how many, time left."""
    log = tmp_path / "sub" / "progress.log"
    monkeypatch.setenv("ADLAB_PROGRESS_LOG", str(log))
    monkeypatch.setattr(progress, "in_tests", lambda: False)
    monkeypatch.setattr(progress, "LOG_EVERY", 0.0)
    assert list(progress.track(iter("abcd"), "scoring toy / cusum", total=4, verbose=False)) == list("abcd")
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    stamp = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d"
    assert re.fullmatch(stamp + r" \| scoring toy / cusum \| 1/4 \| \d+ s left", lines[0])
    assert re.fullmatch(stamp + r" \| scoring toy / cusum \| 4/4 \| done in \d+ s", lines[-1])
    progress.note("report finished")
    assert log.read_text(encoding="utf-8").splitlines()[-1].endswith("| report finished")


def test_the_log_is_off_unless_asked_for(tmp_path, monkeypatch):
    monkeypatch.delenv("ADLAB_PROGRESS_LOG", raising=False)
    assert progress.log_path() is None
    monkeypatch.setenv("ADLAB_PROGRESS_LOG", "1")
    assert progress.log_path() == progress.LOG and progress.LOG.name == "progress.log"


def test_long_loops_take_verbose_false(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CACHE", tmp_path)
    rng = np.random.default_rng(0)
    fit = rng.normal(size=(200, 2))
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    ds = Dataset("toy", [Run(f"r{i}", rng.normal(size=(100, 2)) + 3 * y[:, None], y, fit, "jump") for i in range(3)],
                 ["a", "b"], 1.0)
    run.score_all(datasets=[ds], methods=["rolling_z", "ewma"], verbose=False)
    res = run.report_dataset(ds, ["rolling_z", "ewma"], 1, 20, verbose=False)
    assert len(res["summary"]) == 2
