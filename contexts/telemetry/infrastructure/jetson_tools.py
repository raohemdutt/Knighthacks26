"""The two Jetson command-line tools the power sampler runs: tegrastats (one line per interval) and `nvpmodel -q`.

    tegrastats = TegrastatsProcess(["tegrastats"])
    async for ts, line in tegrastats.lines(interval_ms=100): ...   # ts: when the line arrived, UTC
    mode = await NvpmodelQuery(["/usr/sbin/nvpmodel"])()            # "MAXN_SUPER", or None when it cannot tell

tegrastats runs as `<command> --interval <ms>` with its output on a pipe; stderr is merged into it, so a warning
shows up as a line that does not parse instead of filling a pipe nobody reads. A line is stamped the moment it
is read. When tegrastats exits, `lines()` raises, so the sampler runner starts it again; when the caller stops
early or is cancelled, the process is terminated, and killed if it has not stopped within `stop_timeout_s`.

Both tools print their power figures without root on JetPack 6: the recorded Orin Nano lines run without sudo
(fixtures/telemetry/jetson_tegrastats_orin_nano.txt, third section) carry every rail, and `nvpmodel -q` answers
an ordinary user (fixtures/telemetry/jetson_nvpmodel_q.txt). NVIDIA's guide gives nvpmodel as /usr/sbin/nvpmodel.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime, timezone

from contexts.telemetry.domain.jetson_readings import parse_nvpmodel

log = logging.getLogger("atlas.telemetry")

TEGRASTATS = ("tegrastats",)
NVPMODEL = ("/usr/sbin/nvpmodel",)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TegrastatsProcess:
    def __init__(self, command: Sequence[str] = TEGRASTATS, clock: Callable[[], datetime] = utc_now,
                 stop_timeout_s: float = 2.0) -> None:
        if not command:
            raise ValueError("the tegrastats command is empty")
        self.command = list(command)
        self.clock = clock
        self.stop_timeout_s = stop_timeout_s
        self.pid: int | None = None  # the running tegrastats, for measuring its CPU time
        self.process: asyncio.subprocess.Process | None = None  # the last one started, running or not

    async def lines(self, interval_ms: int) -> AsyncIterator[tuple[datetime, str]]:
        proc = await asyncio.create_subprocess_exec(
            *self.command, "--interval", str(int(interval_ms)),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self.process, self.pid = proc, proc.pid
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                yield self.clock(), raw.decode("utf-8", "replace").rstrip("\r\n")
            code = await proc.wait()
            raise RuntimeError(f"tegrastats exited with code {code}")
        finally:
            self.pid = None
            await stop(proc, self.stop_timeout_s)


class NvpmodelQuery:
    def __init__(self, command: Sequence[str] = NVPMODEL, timeout_s: float = 5.0) -> None:
        self.command = list(command)
        self.timeout_s = timeout_s
        self.failures = 0

    async def __call__(self) -> str | None:
        """The current power mode's name; None (logged on the first failure) when nvpmodel cannot be run or says nothing."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command, "-q", stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
        except OSError as exc:
            return self._failed(f"cannot run {' '.join(self.command)}: {exc}")
        answered = False
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            answered = True
        except asyncio.TimeoutError:
            return self._failed(f"{' '.join(self.command)} -q gave no answer within {self.timeout_s:g} s")
        finally:
            if not answered:  # timed out, or the caller was cancelled while waiting: never leave it running
                await stop(proc, 1.0)
        mode = parse_nvpmodel(out.decode("utf-8", "replace"))
        if mode is None:
            return self._failed(f"{' '.join(self.command)} -q named no power mode (exit {proc.returncode})")
        return mode

    def _failed(self, why: str) -> None:
        self.failures += 1
        if self.failures == 1:
            log.warning("power mode unknown: %s; samples carry nvpmodel=null", why)
        return None


async def stop(proc: asyncio.subprocess.Process, timeout_s: float) -> None:
    """Terminate a child process and wait for it, reading what is left of its output so its pipe closes; kill it
    if it has not ended within `timeout_s`. Never waits forever: if something else still holds the output open
    after the kill, it waits only for the process itself."""
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
    try:
        await asyncio.wait_for(proc.communicate(), timeout_s)
        return
    except asyncio.TimeoutError:
        pass
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    try:
        await asyncio.wait_for(proc.communicate(), timeout_s)
    except asyncio.TimeoutError:
        await proc.wait()
