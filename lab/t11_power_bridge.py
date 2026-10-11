"""Both nodes' power samples in one JSONL file, read from a workstation, for the T11 run off the Jetson.

    python -m lab.t11_power_bridge --out lab/state/t11-power.jsonl --jetson alexander@<the jetson key> \\
        --ssh-key ~/.ssh/wsl_ed25519 --account C [--interval-ms 350] [--minutes 120]

lab/T11_REAL_RUN.md runs the closed loop on the Jetson, reading the power samples the gateway writes into the box's
event log. When the gateway does not log the pods (its restart with ATLAS_TELEMETRY_SOURCE=tegrastats,pods needs
sudo), the loop can run on a workstation instead, with this bridge writing the same events into a file that
`python -m lab routing-closed-loop --power-log <file>` reads:

- telemetry.jetson.power: `tegrastats --interval <ms>` runs on the Jetson over SSH (no sudo needed, as the
  gateway's own sampler runs it) and each line becomes one event, stamped the moment it arrives here, exactly as
  contexts/telemetry/infrastructure/jetson_tools.py stamps it on the box. The rails are the line's current mW
  (contexts/telemetry/domain/jetson_readings.py); the power mode is read once with `nvpmodel -q`. The overcurrent
  counters and the INA3221 settings are not read (they need sysfs reads per sample on the box), so they stay empty.
- telemetry.gpu.power and telemetry.clock.offset for the pod: contexts/telemetry/infrastructure/pods.py's PodReader,
  unchanged, polls the pod sampler (lab/pod_power.py serve) through the hub proxy with the account's keys from the
  environment or the nearest .env, and measures the pod's clock against this machine's from every answer.

One clock. The Jetson's samples carry this machine's arrival time, so in this file "the jetson clock" is this
machine's clock; the pod's offsets are measured against the same clock and say reference "jetson". The closed loop
then runs with --window-clock jetson: its windows are stamped on this machine's clock, a window on the Jetson needs no
move, and a window on the pod is moved by the pod's offset, as on the box.

It only reads: it starts nothing on the pod or the box beyond the tegrastats process of its own SSH session, which
ends with it. Secrets: the hub token goes only in the Authorization header (PodReader); nothing here prints a key.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contexts.telemetry.domain.jetson_readings import parse_nvpmodel, parse_tegrastats_line
from contexts.telemetry.domain.nodes import JETSON, POD_A
from contexts.telemetry.infrastructure.clock import _env_file
from contexts.telemetry.infrastructure.pods import PodReader, hub_user_agent, pod_endpoints
from contracts.events import Event
from contracts.telemetry import JetsonPower

__all__ = ["jetson_event", "JsonlSink", "TegrastatsBridge", "PodBridge", "ssh_command", "main"]

SOURCE = "tegrastats"
DEFAULT_INTERVAL_MS = 350
HISTORY_S = 120.0  # pod samples older than this (on this machine's clock) when the bridge starts are not copied


def jetson_event(line: str, ts: datetime, *, period_ms: float, nvpmodel: str | None,
                 node: str = JETSON) -> JetsonPower | None:
    """One tegrastats line, stamped `ts`, as a telemetry.jetson.power event; None when it carries no power rail."""
    parsed = parse_tegrastats_line(line)
    if parsed is None or not parsed.rails_mw:
        return None
    return JetsonPower(node=node, ts=ts, source=SOURCE, period_ms=period_ms, rails=parsed.rails_mw,
                       nvpmodel=nvpmodel, temps_c=parsed.temps_c)


class JsonlSink:
    """Appends events as JSON lines, one writer at a time, flushed per line so a reader sees whole lines."""

    def __init__(self, path: str | os.PathLike[str], raw: TextIO | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._raw = raw
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def write(self, event: Event) -> None:
        text = event.model_dump_json()
        with self._lock:
            self._fh.write(text + "\n")
            self._fh.flush()
            self.counts[event.type] = self.counts.get(event.type, 0) + 1

    def raw(self, ts: datetime, line: str) -> None:
        if self._raw is None:
            return
        with self._lock:
            self._raw.write(f"{ts.isoformat()}\t{line}\n")
            self._raw.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()
            if self._raw is not None:
                self._raw.close()


def ssh_command(target: str, remote: str, *, key: str | None = None) -> list[str]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ConnectTimeout=20"]
    if key:
        argv += ["-i", os.path.expanduser(key)]
    return argv + [target, remote]


class TegrastatsBridge:
    """tegrastats on the Jetson over SSH, each line an event stamped on arrival here."""

    def __init__(self, sink: JsonlSink, *, target: str, key: str | None = None,
                 interval_ms: int = DEFAULT_INTERVAL_MS, nvpmodel: str | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.sink = sink
        self.target = target
        self.key = key
        self.interval_ms = interval_ms
        self.nvpmodel = nvpmodel
        self.clock = clock
        self.lines = 0
        self.skipped = 0
        self.proc: subprocess.Popen[bytes] | None = None

    def read_mode(self) -> str | None:
        out = subprocess.run(ssh_command(self.target, "/usr/sbin/nvpmodel -q", key=self.key), capture_output=True,
                             timeout=60, check=False)
        self.nvpmodel = parse_nvpmodel(out.stdout.decode("utf-8", "replace"))
        return self.nvpmodel

    def feed(self, lines: Iterable[tuple[datetime, str]]) -> None:
        """Turn (arrival time, line) pairs into events: the part tested on recorded lines."""
        for ts, line in lines:
            self.lines += 1
            self.sink.raw(ts, line)
            event = jetson_event(line, ts, period_ms=self.interval_ms, nvpmodel=self.nvpmodel)
            if event is None:
                self.skipped += 1
                continue
            self.sink.write(event)

    def _arrivals(self) -> Iterable[tuple[datetime, str]]:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            yield self.clock(), raw.decode("utf-8", "replace").rstrip("\r\n")

    def run(self, stop: threading.Event) -> None:
        """Stream until `stop` is set, starting the SSH session again after a drop (every 5 s)."""
        remote = f"exec tegrastats --interval {int(self.interval_ms)}"
        while not stop.is_set():
            self.proc = subprocess.Popen(ssh_command(self.target, remote, key=self.key), stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            try:
                self.feed(self._arrivals())
            finally:
                self.stop_process()
            if not stop.is_set():
                print(json.dumps({"event": "tegrastats stream ended; starting it again in 5 s",
                                  "at": self.clock().isoformat()}), flush=True)
                stop.wait(5.0)

    def stop_process(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class PodBridge:
    """The pod's sampler through the hub proxy (PodReader), its samples and clock offsets into the sink."""

    def __init__(self, sink: JsonlSink, environ: Mapping[str, str], *, account: str = "C", node: str = POD_A,
                 transport: Any = None, rounds: int | None = None, history_s: float = HISTORY_S,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC), started: datetime | None = None) -> None:
        env = {**environ, "ATLAS_POD_ACCOUNTS": f"{node}={account}"}
        start = started or datetime.now(UTC)  # history is cut against this; `clock` times the exchanges only
        self.reader = PodReader(pod_endpoints(env), user_agent=hub_user_agent(env), transport=transport,
                                rounds=rounds, clock=clock,
                                skip_until=lambda _node: start - timedelta(seconds=history_s) if history_s else None)
        self.sink = sink

    async def run_async(self) -> None:
        async for event in self.reader.samples():
            self.sink.write(event)

    def run(self, stop: threading.Event) -> None:
        """Read until `stop` is set; a failed poll loop is started again after 5 s (the reader keeps its cursor)."""
        while not stop.is_set():
            loop = asyncio.new_event_loop()
            task = loop.create_task(self.run_async())

            def watch() -> None:
                stop.wait()
                loop.call_soon_threadsafe(task.cancel)

            threading.Thread(target=watch, daemon=True).start()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - logged and retried; the reader names no secret in it
                print(json.dumps({"event": f"pod reader stopped on {type(exc).__name__}; again in 5 s"}), flush=True)
            finally:
                loop.close()
            if self.reader.rounds is not None:
                return
            stop.wait(5.0)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lab.t11_power_bridge", description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="the JSONL file the events are appended to")
    p.add_argument("--raw", help="also keep each tegrastats line with its arrival time here")
    p.add_argument("--jetson", required=True, help="user@host of the Jetson")
    p.add_argument("--ssh-key", default=None)
    p.add_argument("--account", default="C", help="the teacher pod's hub account letter")
    p.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    p.add_argument("--minutes", type=float, default=None, help="stop after this long (default: until interrupted)")
    args = p.parse_args(argv)
    environ = {**_env_file(Path.cwd()), **os.environ}
    raw = open(args.raw, "a", encoding="utf-8", newline="\n") if args.raw else None
    sink = JsonlSink(args.out, raw)
    stop = threading.Event()
    jetson = TegrastatsBridge(sink, target=args.jetson, key=args.ssh_key, interval_ms=args.interval_ms)
    mode = jetson.read_mode()
    pod = PodBridge(sink, environ, account=args.account.upper())
    threads = [threading.Thread(target=jetson.run, args=(stop,), daemon=True, name="tegrastats"),
               threading.Thread(target=pod.run, args=(stop,), daemon=True, name="pod")]
    for t in threads:
        t.start()
    print(json.dumps({"event": "started", "out": str(sink.path), "nvpmodel": mode,
                      "interval_ms": args.interval_ms}), flush=True)
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    try:
        while not stop.is_set():
            stop.wait(30.0)
            print(json.dumps({"event": "counts", "at": datetime.now(UTC).isoformat(), **sink.counts,
                              "pod": {k: v for k, v in pod.reader.status().get(POD_A, {}).items()
                                      if k in ("fetched", "lost", "rejected", "error", "restarts")}},
                             default=str), flush=True)
            if deadline is not None and time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        jetson.stop_process()
        for t in threads:
            t.join(timeout=10)
        sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
