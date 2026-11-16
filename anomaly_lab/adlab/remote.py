"""Read parts of a very large remote file without downloading all of it.

The multi-mode Tennessee Eastman files are about 24 GB each, and only a small part of each is
needed. The server accepts requests for a range of bytes, so RangeFile presents the remote file
as an ordinary read-only file that fetches blocks as they are asked for and keeps count of what
it has transferred. h5py can open it directly:

    f = RangeFile("https://ndownloader.figshare.com/files/26003087")
    h5 = h5py.File(f, "r")
    ...
    print(f.fetched_bytes)
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request

AGENT = "adlab-data-loader/0.1"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def http_fetcher(url: str, timeout: float = 120.0, tries: int = 6, pause: float = 2.0):
    """A function fetch(start, end) -> bytes for `url` (both ends included), and the file's size.

    The download address first redirects to a signed address at the storage server that is
    valid for a while. That address is looked up once and reused, and looked up again if the
    storage server starts refusing it. Between tries the wait doubles (pause, 2 x pause, ...),
    so a server that refuses for a moment is not hammered.
    """
    import time
    state = {"direct": None}
    opener = urllib.request.build_opener(_NoRedirect)

    def resolve():
        try:
            opener.open(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=timeout)
            state["direct"] = url                      # no redirect: the address serves the file itself
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise
            state["direct"] = e.headers["Location"]

    def get(start: int, end: int):
        last = None
        for attempt in range(tries):
            if attempt:
                time.sleep(pause * 2 ** (attempt - 1))
            try:
                if state["direct"] is None:
                    resolve()
                req = urllib.request.Request(state["direct"], headers={"User-Agent": AGENT, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    if r.status != 206:
                        raise OSError(f"the server ignored the byte range (status {r.status}); not downloading the whole file")
                    return r.read(), int(r.headers["Content-Range"].rsplit("/", 1)[1])
            except urllib.error.HTTPError as e:
                last = e
                state["direct"] = None                 # the signed address may have run out: look it up again
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last = e
        raise OSError(f"could not read bytes {start}-{end} of {url}: {last}")

    first, size = get(0, 0)
    return (lambda start, end: get(start, end)[0]), size


class RangeFile(io.RawIOBase):
    """A read-only file over fetch(start, end) -> bytes, read in blocks that are kept in memory.

    url_or_fetch   an http(s) address, or a function fetch(start, end) (both ends included)
    size           the file's size in bytes (needed only with a fetch function)
    block          block size in bytes; neighbouring missing blocks are fetched in one request
    keep           how many blocks to keep in memory (oldest dropped first)
    limit          stop with an error once this many bytes have been transferred (None: no limit)
    fetched_bytes, requests   running totals of what has been transferred
    """

    def __init__(self, url_or_fetch, size: int | None = None, block: int = 1 << 20, keep: int = 512,
                 limit: int | None = None):
        super().__init__()
        if callable(url_or_fetch):
            self._fetch, self.size = url_or_fetch, int(size)
        else:
            self._fetch, self.size = http_fetcher(url_or_fetch)
        self.block, self.keep, self.limit = block, keep, limit
        self._blocks: dict[int, bytes] = {}
        self._pos = 0
        self.fetched_bytes = 0
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        self._pos = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence] + offset
        return self._pos

    def _load(self, first: int, last: int):
        """Make sure blocks first..last are in memory, fetching each unbroken run of missing ones at once."""
        b = first
        while b <= last:
            if b in self._blocks:
                b += 1
                continue
            e = b
            while e + 1 <= last and e + 1 not in self._blocks:
                e += 1
            start, end = b * self.block, min((e + 1) * self.block, self.size) - 1
            if self.limit is not None and self.fetched_bytes + (end - start + 1) > self.limit:
                raise OSError(f"transfer limit of {self.limit:,} bytes reached")
            data = self._fetch(start, end)
            self.fetched_bytes += len(data)
            self.requests += 1
            for i in range(b, e + 1):
                self._blocks[i] = data[(i - b) * self.block: (i - b + 1) * self.block]
            b = e + 1

    def readinto(self, buffer):
        n = min(len(buffer), max(self.size - self._pos, 0))
        if n == 0:
            return 0
        first, last = self._pos // self.block, (self._pos + n - 1) // self.block
        self._load(first, last)
        out = b"".join(self._blocks[i] for i in range(first, last + 1))
        offset = self._pos - first * self.block
        buffer[:n] = out[offset: offset + n]
        self._pos += n
        while len(self._blocks) > max(self.keep, last - first + 1):      # drop the oldest blocks
            self._blocks.pop(next(iter(self._blocks)))
        return n
