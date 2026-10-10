"""A cross-platform file lock: an OS lock held through an open handle on
one file, msvcrt.locking on Windows and fcntl.flock elsewhere.

The operating system releases the lock when the handle closes or the
process ends, so it never goes stale the way a pid file does: a holder that
crashed holds nothing. Two FileLock objects on one path exclude each other,
inside one process as across processes, because each holds its own handle.
The lock is advisory, meant for a local file system, and the lock file must
not be deleted while anyone may hold it.

The locked byte sits at LOCK_OFFSET, past a note the holder may write at
the start of the file (who holds it, since when), so a caller that was
refused can still read the note: Windows refuses reads that overlap a
locked range. The holder's note is cleared when it releases.

    with FileLock(path, timeout=5.0) as lock:   # LockTimeout after 5 s
        lock.write_note(f"pid {os.getpid()}")
        ...

acquire() waits at most its timeout and answers False rather than raising;
the context manager raises LockTimeout, naming the holder's note. A
FileLock is not re-entrant and not shared between threads: each thread or
process that wants the lock makes its own FileLock."""
from __future__ import annotations

import errno
import logging
import os
import sys
import time
from pathlib import Path
from types import TracebackType

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

__all__ = ["DEFAULT_TIMEOUT", "LOCK_OFFSET", "FileLock", "LockTimeout"]

log = logging.getLogger("lab.lock")

DEFAULT_TIMEOUT = 10.0    # seconds a FileLock waits for another holder unless told otherwise
LOCK_OFFSET = 4096        # where the locked byte sits, past any note
POLL_SECONDS = 0.005      # pause between attempts while another handle holds the lock

# What a non-blocking attempt raises while another handle holds the lock:
# EACCES or EDEADLOCK from msvcrt.locking, EWOULDBLOCK or EAGAIN from flock.
# Not every name exists on every platform.
_HELD_ERRNOS = frozenset(getattr(errno, name) for name in
                         ("EACCES", "EAGAIN", "EDEADLK", "EDEADLOCK", "EWOULDBLOCK")
                         if hasattr(errno, name))
_O_BINARY = getattr(os, "O_BINARY", 0)


class LockTimeout(TimeoutError):
    """Another handle held the lock for the whole timeout."""


def _try_lock(fd: int) -> bool:
    """One non-blocking attempt on the lock byte: True when it is ours, False
    when another handle holds it. Any other failure is raised."""
    try:
        if sys.platform == "win32":
            os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _HELD_ERRNOS:
            return False
        raise
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """An exclusive lock on the file at path, taken through its own handle.

    timeout is how long acquire() and the context manager wait for another
    holder, in seconds; 0 means a single attempt."""

    def __init__(self, path: str | os.PathLike[str], timeout: float = DEFAULT_TIMEOUT) -> None:
        self.path = Path(path)
        self.timeout = max(0.0, float(timeout))
        self._fd: int | None = None
        self._noted = False

    def __repr__(self) -> str:
        return f"FileLock({str(self.path)!r}, timeout={self.timeout:g}, held={self.held})"

    @property
    def held(self) -> bool:
        """Whether this object holds the lock now."""
        return self._fd is not None

    def acquire(self, timeout: float | None = None) -> bool:
        """True once the lock is ours; holding it already counts. False when
        another handle still holds it after timeout seconds (None: this
        lock's own timeout; 0: one attempt). The file and its directory are
        created when missing; a failure to open them is raised."""
        if self._fd is not None:
            return True
        wait = self.timeout if timeout is None else max(0.0, float(timeout))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | _O_BINARY, 0o644)
        deadline = time.monotonic() + wait
        acquired = False
        try:
            while True:
                if _try_lock(fd):
                    acquired = True
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                time.sleep(min(POLL_SECONDS, left))
        finally:
            if not acquired:
                os.close(fd)
        if acquired:
            self._fd = fd
            self._noted = False
        return acquired

    def release(self) -> None:
        """Clear this holder's note, unlock and close the handle; nothing
        happens when the lock is not held. Closing the handle releases the
        lock even when the unlock call itself fails, so a failure there is
        logged rather than raised."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if self._noted:
                os.ftruncate(fd, 0)
            _unlock(fd)
        except OSError as exc:
            log.error("could not unlock %s cleanly (closing the handle releases it): %s", self.path, exc)
        finally:
            self._noted = False
            os.close(fd)

    def write_note(self, text: str) -> None:
        """Write the holder's note at the start of the file, before the
        locked byte: who holds the lock, for the message a refused caller
        gets. Only the holder may write it; release() clears it."""
        if self._fd is None:
            raise RuntimeError(f"write_note on {self.path} without holding the lock")
        data = text.encode("utf-8")[:LOCK_OFFSET - 1]
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, data)
        self._noted = True

    def read_note(self) -> str:
        """The current holder's note; "" when there is none or the file is
        missing or unreadable. Takes no lock and creates nothing."""
        try:
            fd = os.open(self.path, os.O_RDONLY | _O_BINARY)
        except OSError:
            return ""
        try:
            return os.read(fd, LOCK_OFFSET - 1).decode("utf-8", "replace").strip()
        except OSError:
            return ""
        finally:
            os.close(fd)

    def __enter__(self) -> FileLock:
        if self._fd is not None:
            raise RuntimeError(f"{self.path} is already held through this FileLock, which is not re-entrant")
        if not self.acquire():
            note = self.read_note()
            raise LockTimeout(f"the lock {self.path} stayed held for {self.timeout:g}s"
                              + (f" (holder: {note})" if note else ""))
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.release()
