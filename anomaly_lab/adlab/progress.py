"""Progress bars for the slow loops, and a progress log for jobs running in the background.

    for run in progress.track(runs, "scoring tep_rieth / cusum"):
        ...

In a terminal or a notebook this shows a bar with the time remaining (tqdm). Pass verbose=False
to switch it off. Bars are always off inside the tests.

For a job running in the background, set the environment variable ADLAB_PROGRESS_LOG=1 before
starting it. Every tracked loop then also writes lines to results/progress.log:

    2026-11-15 09:14:02 | scoring tep_rieth / cusum | 400/1000 | 1 min 10 s left

so the file can be opened at any time to see how far the job is. Set the variable to a path to
write somewhere else.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "results" / "progress.log"
LOG_EVERY = 10.0     # seconds between log lines of one loop (its first and last step are always written)


def in_tests() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def log_path() -> Path | None:
    """Where progress lines go, or None when the progress log is off."""
    value = os.environ.get("ADLAB_PROGRESS_LOG", "")
    if value in ("", "0"):
        return None
    return LOG if value == "1" else Path(value)


def nice_time(seconds: float) -> str:
    """90 -> '1 min 30 s'; 4000 -> '1 h 7 min'."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60} min"


def note(message: str) -> None:
    """Write one line to the progress log (if it is on)."""
    path = log_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")


_awake = 0     # how many tracked loops are running now


def stay_awake(on: bool) -> None:
    """On Windows, ask the computer not to go to sleep by itself while a tracked loop is running.

    Loops inside loops are counted, so the request is dropped only when the outermost one ends.
    It stops the idle timer only: closing the lid or choosing Sleep still puts the computer to
    sleep. Does nothing on other systems.

    Round 5: the screen is kept on too. On laptops with "modern standby" (S0 low-power idle), the
    screen switching off on its idle timer puts the whole computer into standby, and asking only
    for the system to stay on did not stop that (a job stood still from 23:02 to 05:38).
    """
    global _awake
    _awake = max(0, _awake + (1 if on else -1))
    if os.name != "nt":
        return
    try:
        import ctypes
        continuous, system_required, display_required = 0x80000000, 0x00000001, 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            continuous | ((system_required | display_required) if _awake else 0))
    except Exception:           # never let this stop the real work
        pass


def track(items, what: str, total: int | None = None, verbose: bool = True, leave: bool = False):
    """Go through `items`, showing a progress bar and writing to the progress log.

    While the loop runs (outside the tests), Windows is asked not to go to sleep (stay_awake).

    what     a few words saying what is being done, shown next to the bar and in the log
    total    how many items there are, if `items` has no length
    verbose  False switches the bar off (the log, if on, is still written)
    """
    if total is None:
        try:
            total = len(items)
        except TypeError:
            total = None
    show = verbose and not in_tests()
    logging = log_path() is not None and not in_tests()
    if not show and not logging:
        yield from items
        return
    bar = None
    if show:
        from tqdm.auto import tqdm
        bar = tqdm(total=total, desc=what, leave=leave)
    start = last = time.time()
    done = 0
    stay_awake(True)
    try:
        for item in items:
            yield item
            done += 1
            if bar is not None:
                bar.update(1)
            now = time.time()
            if logging and (done == 1 or done == total or now - last >= LOG_EVERY):
                last = now
                if total:
                    left = (now - start) / done * (total - done)
                    tail = "done in " + nice_time(now - start) if done == total else nice_time(left) + " left"
                    note(f"{what} | {done}/{total} | {tail}")
                else:
                    note(f"{what} | {done} done | {nice_time(now - start)} so far")
    finally:
        stay_awake(False)
        if bar is not None:
            bar.close()
