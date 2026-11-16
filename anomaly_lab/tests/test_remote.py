"""Reading parts of a big remote file (adlab/remote.py). No network: the 'server' is a bytes object."""
import io

import numpy as np
import pytest

from adlab.remote import RangeFile


def _served(data: bytes, log: list):
    def fetch(start, end):
        log.append((start, end))
        return data[start: end + 1]
    return fetch


def test_reads_give_the_right_bytes_and_fetch_only_the_blocks_needed():
    data = bytes(range(256)) * 40                                  # 10,240 bytes
    log = []
    f = RangeFile(_served(data, log), size=len(data), block=1000)
    f.seek(2500)
    assert f.read(10) == data[2500:2510] and log == [(2000, 2999)]
    assert f.read(600) == data[2510:3110] and log[-1] == (3000, 3999)   # block 2 is reused, only block 3 fetched
    f.seek(-5, io.SEEK_END)
    assert f.read(100) == data[-5:] and log[-1] == (10000, 10239)       # the last block is short
    assert f.read(10) == b"" and f.tell() == len(data)
    f.seek(0)
    assert f.read(4500) == data[:4500] and log[-1] == (4000, 4999)      # blocks 0-1 in one request, then 4
    assert f.requests == len(log) == 5 and f.fetched_bytes == sum(e - s + 1 for s, e in log)


def test_transfer_limit_stops_the_download():
    data = b"x" * 5000
    f = RangeFile(_served(data, []), size=len(data), block=1000, limit=2500)
    f.read(2000)
    with pytest.raises(OSError):
        f.read(1000)


def test_h5py_can_open_a_file_through_it(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "small.h5"
    values = np.arange(6000.0).reshape(2000, 3)
    with h5py.File(path, "w") as h:
        h.create_group("Mode1/Run1").create_dataset("processdata", data=values)
    data = path.read_bytes()
    f = RangeFile(_served(data, []), size=len(data), block=4096)
    with h5py.File(f, "r") as h:
        assert list(h["Mode1"]) == ["Run1"]
        assert np.array_equal(h["Mode1/Run1/processdata"][:], values)
    assert 0 < f.fetched_bytes <= len(data) + 4096
