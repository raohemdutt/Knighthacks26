"""link_speed: how fast a few-GB package crosses the hub to a pod and back, three runs each way, checked by sha256.

    python -m lab link-speed --account D [--sizes-mb 1000,3000] [--repeats 3] [--chunk-mib 64] [--part-mib 64]
                             [--port 8301] [--probe-ceiling-mib 256] [--no-probe] [--scratch r14-link-speed]
                             [--workdir DIR] [--report lab/R14_REPORT.md]

What it does, for each size. A synthetic file is written from a seeded generator (random bytes, so nothing compresses on
the way) and its sha256 taken. Then, `--repeats` times: the file goes up to the pod through the Jupyter contents API in
base64 chunks (lab/transfer.py `push_file`, size read back), a manifest.json beside it, and the box pulls it back
through the hub proxy from R10's file server (lab/package_server.py, started on its own port, serving the folder the push
wrote to) in Range parts (lab/package_pull.py `pull_package`), checks the sha256 twice (the manifest's, and its own
against the generated file) and deletes the copy. Before the sizes, a short probe doubles one single PUT body until the
contents API refuses it and keeps the last size it accepted. Everything lands under one scratch folder on the pod,
which is deleted at the end together with the file server and the script files, so the pod is left as found.

What a run records. One ledger row per run (`link push`, `link pull`) with the metric link.push_mb_s or link.pull_mb_s
(MB of 10^6 bytes a second), its bytes, seconds, chunk_bytes (push) or part_bytes (pull), the node (pod-b), the hub's
host and refs.account; a failed run is a row with the outcome `error` and no metric. One row for the probe
(`link part-probe`, metric link.max_part_mb). The report is a Markdown file with the table for the task page.

Two clocks, said plainly. A push's seconds cover the whole call: folders, every chunk, the size read back. A pull's
seconds are the requests only, from each part's request until its last byte, with the pauses between tries left out
(lab/package_pull.py's definition); its wall seconds are in the row. No MB/s bar applies: this measures, it does not judge.

The pod's side. The file server is lab/package_server.py under another file name (`r14_file_server.py`) in
`~/.r14_link_speed`, so it never touches a standing package server on port 8300. It serves the folder the contents API
wrote to, which `serve_script` finds with `find` (the Jupyter root differs between images). `stop_script` stops it and
removes that directory. Nothing else on the pod is restarted, and no GPU is used.

Tests (tests/test_lab_link_speed.py) run the whole thing at 1 MB and 3 MB against lab/proxy_fake.py and
lab/package_fake.py. Keys are AUP_HUB_USER_<letter> and AUP_HUB_TOKEN_<letter>; no key and no account name is printed,
written to the ledger or put in the report.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import random
import re
import secrets
import shutil
import statistics
import sys
import tempfile
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):          # started as `python lab/link_speed.py`: put the repo root on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import package_pull, transfer
from lab.controller import Controller
from lab.hub import Account, MissingCredentials, PodRoleError, hub_of, load_accounts
from lab.machine import record
from lab.package_pull import PackageClient
from lab.proxy import PodLink, ProxyError
from lab.transfer import PodFiles, TransferError, TransferStats, sha256_file

__all__ = ["FreshLink", "Summary", "Run", "SizeResult", "ProbeStep", "Probe", "Report", "Target", "LiveTarget", "make_payload",
           "size_label", "summarize", "manifest_bytes", "probe_part_limit", "measure", "serve_script", "stop_script",
           "build_parser", "main"]

CLI_VERB = "link-speed"   # registers python -m lab link-speed
KIB = 1 << 10
MIB = 1 << 20
MB = 1_000_000
DEFAULT_SIZES_MB = (1000.0, 3000.0)
DEFAULT_REPEATS = 3
DEFAULT_CHUNK_BYTES = 64 * MIB                 # the most transfer.py sends: about 85 MB on the wire, under Cloudflare's 100 MB
DEFAULT_PART_BYTES = 64 * MIB                  # a Range part the proxy takes (it refuses a response over 100 MiB)
DEFAULT_PORT = 8301                            # not 8300, where R10's standing package server may be running
DEFAULT_SCRATCH = "r14-link-speed"
DEFAULT_REPORT = "lab/R14_REPORT.md"
DEFAULT_SEED = 14
PROBE_START_BYTES = MIB
PROBE_CEILING_BYTES = 256 * MIB
PAYLOAD_BLOCK = MIB
PAYLOAD_NAME = "model.gguf"
PROBE_NAME = "probe.bin"
POD_DIR = "$HOME/.r14_link_speed"
SERVER_FILE = "r14_file_server.py"
NAME_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")
PUSH_ROW, PULL_ROW, PROBE_ROW = "link push", "link pull", "link part-probe"
PUSH_METRIC, PULL_METRIC, PART_METRIC = "link.push_mb_s", "link.pull_mb_s", "link.max_part_mb"
SERVE_TIMEOUT_S = 180.0
PROBE_TRIES = 2                                # a refusal that may be a hiccup (a gateway error) is asked once more
PROBE_PAUSE_S = 5.0
NOTE_CHARS = 300


# --- the synthetic file -------------------------------------------------------------------------------------

def make_payload(path: Path, size: int, seed: int = DEFAULT_SEED) -> str:
    """Write `size` bytes of seeded random data to `path` and return their sha256. The same seed and size always give
    the same bytes; random bytes do not compress, so a link that compresses cannot flatter the speed."""
    if size < 1:
        raise ValueError("a payload is at least one byte")
    rng = random.Random(seed * 1_000_003 + size)
    digest = hashlib.sha256()
    left = size
    with path.open("wb") as out:
        while left > 0:
            block = rng.randbytes(min(PAYLOAD_BLOCK, left))
            out.write(block)
            digest.update(block)
            left -= len(block)
    return digest.hexdigest()


def size_label(size: int) -> str:
    """1000000000 is 1GB, 3000000 is 3MB, 1500000 is 1500KB, 1234 is 1234B: a name for a folder, a row and a table."""
    for unit, factor in (("GB", 10**9), ("MB", 10**6), ("KB", 10**3)):
        if size >= factor and size % factor == 0:
            return f"{size // factor}{unit}"
    if size >= 10**3 and size % 10**3 == 0:
        return f"{size // 10**3}KB"
    return f"{size}B"


def fmt_bytes(count: int) -> str:
    if count >= MIB and count % MIB == 0:
        return f"{count // MIB} MiB"
    if count >= KIB and count % KIB == 0:
        return f"{count // KIB} KiB"
    return f"{count} B"


def manifest_bytes(version: str, sha256: str, now: dt.datetime | None = None) -> bytes:
    """A manifest.json (ModelManifest) for the synthetic package, so R10's pull can read and check it."""
    created = (now or dt.datetime.now(dt.timezone.utc)).isoformat()
    manifest = {"version": version, "base": "synthetic/random-bytes", "task": None, "gguf": PAYLOAD_NAME, "mmproj": None,
                "sha256": {PAYLOAD_NAME: sha256}, "scores": {}, "created": created}
    return json.dumps(manifest, indent=2).encode("utf-8")


# --- numbers ------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    stdev: float                 # sample standard deviation; 0 for a single run
    minimum: float
    maximum: float


def summarize(values: Sequence[float]) -> Summary | None:
    """Mean, sample spread and range of the speeds of the runs that worked; None when there are none."""
    if not values:
        return None
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return Summary(len(values), statistics.fmean(values), spread, min(values), max(values))


@dataclasses.dataclass
class Run:
    """One push or one pull of one repeat."""

    direction: str                       # push or pull
    repeat: int
    size_bytes: int                      # the package file's size
    ok: bool = False
    bytes: int = 0                       # bytes this run moved
    seconds: float = 0.0
    mb_s: float | None = None
    wall_seconds: float | None = None
    parts: int | None = None             # chunks (push) or Range parts (pull)
    retries: int = 0
    sha256_ok: bool | None = None        # pull: did the copy match; None when nothing was compared
    size_ok: bool | None = None          # push: did the pod report the size that was sent
    kind: str | None = None
    error: str | None = None


@dataclasses.dataclass
class SizeResult:
    label: str
    size_bytes: int
    sha256: str
    runs: list[Run] = dataclasses.field(default_factory=list)

    def of(self, direction: str) -> list[Run]:
        return [run for run in self.runs if run.direction == direction]

    def speeds(self, direction: str) -> Summary | None:
        return summarize([run.mb_s for run in self.of(direction) if run.ok and run.mb_s is not None])

    @property
    def pass_rate(self) -> tuple[int, int]:
        """(pulls whose copy matched the generated file, pulls tried)."""
        pulls = self.of("pull")
        return sum(1 for run in pulls if run.sha256_ok is True and run.ok), len(pulls)


@dataclasses.dataclass
class ProbeStep:
    bytes: int
    accepted: bool
    status: int | None = None
    kind: str | None = None
    seconds: float = 0.0
    error: str | None = None

    @property
    def wire_bytes(self) -> int:
        return 4 * ((self.bytes + 2) // 3)           # base64


@dataclasses.dataclass
class Probe:
    steps: list[ProbeStep] = dataclasses.field(default_factory=list)
    largest_accepted_bytes: int | None = None
    first_refused_bytes: int | None = None
    ceiling_bytes: int = PROBE_CEILING_BYTES

    @property
    def ceiling_reached(self) -> bool:
        """True when no size was refused: the probe stopped at its ceiling, and the real limit is higher."""
        return self.first_refused_bytes is None and self.largest_accepted_bytes is not None


@dataclasses.dataclass
class Report:
    run_id: str
    node: str
    hub: str
    account: str
    client: str
    started: str
    repeats: int
    chunk_bytes: int
    part_bytes: int
    port: int
    seed: int
    sizes: list[SizeResult] = dataclasses.field(default_factory=list)
    probe: Probe | None = None
    cleanup_problems: list[str] = dataclasses.field(default_factory=list)
    ledger_errors: list[str] = dataclasses.field(default_factory=list)

    @property
    def runs(self) -> list[Run]:
        return [run for item in self.sizes for run in item.runs]

    @property
    def ok(self) -> bool:
        """Every push and pull of every repeat ran, worked and (for a pull) matched its checksum."""
        if not self.sizes:
            return False
        for item in self.sizes:
            if len(item.of("push")) != self.repeats or len(item.of("pull")) != self.repeats:
                return False
        return all(run.ok for run in self.runs)

    @property
    def largest_file_bytes(self) -> int | None:
        """The largest file the contents API took whole (in chunks), the size read back on the pod."""
        sizes = [run.size_bytes for run in self.runs if run.direction == "push" and run.ok]
        return max(sizes) if sizes else None

    # the table for the task page

    def title(self) -> str:
        names = " and ".join(item.label for item in self.sizes) or "synthetic"
        return f"R14: push and pull speed of a {names} package on {self.node}"

    def markdown(self) -> str:
        lines = [f"# {self.title()}", ""]
        lines += [
            f"Measured {self.started} from a {self.client} machine against hub {self.hub}, account {self.account}, node "
            f"{self.node}. Push goes through the Jupyter contents API in base64 chunks; pull comes back through the hub "
            f"proxy (port {self.port}) from R10's Range-capable file server in parts, and every pulled copy is checked "
            f"against the generated file's sha256. Speeds are MB/s of 10^6 bytes, mean over the runs that worked, with the "
            f"sample standard deviation and the range. A push's seconds cover the whole call (folders, chunks, size read "
            f"back); a pull's cover its requests only. No MB/s bar applies.", ""]
        lines += ["| Package | Direction | Chunk or part | Runs ok | MB/s mean | stdev | min | max | sha256 pass (push: size) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for item in self.sizes:
            for direction in ("push", "pull"):
                runs = item.of(direction)
                done = sum(1 for run in runs if run.ok)
                summary = item.speeds(direction)
                cells = ["n/a"] * 4 if summary is None else [f"{value:.2f}" for value in (summary.mean, summary.stdev, summary.minimum, summary.maximum)]
                if direction == "push":
                    check = f"size {sum(1 for run in runs if run.size_ok)} of {len(runs)}"
                else:
                    passed, tried = item.pass_rate
                    check = f"{passed} of {tried}"
                size = self.chunk_bytes if direction == "push" else self.part_bytes
                lines.append(f"| {item.label} | {direction} | {fmt_bytes(size)} | {done} of {len(runs)} | "
                             + " | ".join(cells) + f" | {check} |")
        lines.append("")
        if self.probe is not None:
            lines += self._probe_lines()
        biggest = self.largest_file_bytes
        if biggest is not None:
            lines += [f"The largest file the contents API accepted whole (in {fmt_bytes(self.chunk_bytes)} chunks, size read "
                      f"back on the pod) was {size_label(biggest)} ({biggest} bytes).", ""]
        lines += ["## Every run", "", "| Package | Direction | Repeat | Bytes | Seconds | MB/s | Result |", "|---|---|---|---|---|---|---|"]
        for item in self.sizes:
            for run in item.runs:
                speed = "n/a" if run.mb_s is None or not run.ok else f"{run.mb_s:.2f}"
                if run.ok:
                    result = "ok"
                else:
                    result = "failed: " + (run.kind or "error") + (f" ({run.error})" if run.error else "")
                lines.append(f"| {item.label} | {run.direction} | {run.repeat} | {run.bytes} | {run.seconds:.1f} | {speed} | "
                             f"{result.replace('|', '/')} |")
        lines.append("")
        if self.cleanup_problems:
            lines += ["Left on the pod: " + "; ".join(self.cleanup_problems), ""]
        else:
            lines += ["The scratch folder, the file server and the script files were removed from the pod after the run.", ""]
        lines += [f"Run id {self.run_id}; one ledger row per run (`{PUSH_ROW}`, `{PULL_ROW}`) and one for the probe "
                  f"(`{PROBE_ROW}`)."]
        return "\n".join(lines) + "\n"

    def _probe_lines(self) -> list[str]:
        probe = self.probe
        assert probe is not None
        lines = ["## Largest single part", ""]
        if probe.largest_accepted_bytes is None:
            lines.append("No single PUT was accepted, not even the smallest tried.")
        else:
            tail = (f"; nothing was refused up to the probe's ceiling of {fmt_bytes(probe.ceiling_bytes)}, so the real limit is higher"
                    if probe.ceiling_reached else f"; {fmt_bytes(probe.first_refused_bytes or 0)} was refused")
            lines.append(f"The contents API accepted a single PUT body of up to {fmt_bytes(probe.largest_accepted_bytes)} of file "
                         f"({probe.largest_accepted_bytes} bytes, about {4 * ((probe.largest_accepted_bytes + 2) // 3)} on the "
                         f"wire after base64){tail}.")
        lines += ["", "| Part (file bytes) | On the wire | Answer | Seconds |", "|---|---|---|---|"]
        for step in probe.steps:
            if step.accepted:
                answer = "accepted"
            else:
                answer = f"refused {step.status}" if step.status else f"refused ({step.kind or 'no answer'})"
            lines.append(f"| {fmt_bytes(step.bytes)} | {step.wire_bytes} | {answer} | {step.seconds:.1f} |")
        lines.append("")
        return lines

    def as_dict(self) -> dict[str, Any]:
        def summary(item: SizeResult, direction: str) -> dict[str, Any] | None:
            found = item.speeds(direction)
            return None if found is None else dataclasses.asdict(found)

        return {
            "run_id": self.run_id, "node": self.node, "hub": self.hub, "account": self.account, "client": self.client,
            "started": self.started, "repeats": self.repeats, "chunk_bytes": self.chunk_bytes, "part_bytes": self.part_bytes,
            "ok": self.ok, "largest_file_bytes": self.largest_file_bytes,
            "largest_part_bytes": None if self.probe is None else self.probe.largest_accepted_bytes,
            "first_refused_part_bytes": None if self.probe is None else self.probe.first_refused_bytes,
            "sizes": [{"label": item.label, "bytes": item.size_bytes, "sha256_passed": item.pass_rate[0],
                       "pulls": item.pass_rate[1], "push": summary(item, "push"), "pull": summary(item, "pull"),
                       "runs": [dataclasses.asdict(run) for run in item.runs]} for item in self.sizes],
            "cleanup_problems": self.cleanup_problems, "ledger_errors": self.ledger_errors}


# --- the pod, as the measurement sees it --------------------------------------------------------------------

class Target(Protocol):
    """What the measurement needs of a pod. LiveTarget is the real one; a test can pass anything with these members."""

    node: str
    hub: str
    account: str

    def push(self, local: Path, remote: str, chunk_bytes: int) -> TransferStats: ...
    def put_once(self, remote: str, data: bytes) -> int | None: ...
    def serve(self, scratch: str, version: str) -> None: ...
    def unserve(self) -> None: ...
    def pull(self, version: str, dest: Path, part_bytes: int) -> dict[str, Any]: ...
    def remove(self, remote: str) -> None: ...
    def cleanup(self, scratch: str) -> list[str]: ...
    def close(self) -> None: ...


def _check_word(text: str, what: str) -> str:
    if not isinstance(text, str) or NAME_OK.fullmatch(text) is None:
        raise ValueError(f"{what} is letters, digits, dot, dash and underscore, not {text!r}")
    return text


def serve_script(port: int, scratch: str, version: str) -> str:
    """The shell script that starts R10's file server (package_server.py under the name r14_file_server.py) on `port`,
    serving the folder the contents API wrote `scratch/version` into. It changes four things in
    lab/package_pull.py's start script, so that it cannot disturb a standing package server on port 8300: its own file
    name (the restart there kills by name), its own directory and log, and a root found on the pod."""
    _check_word(scratch, "the scratch folder")
    _check_word(version, "the version")
    base = package_pull.start_script(int(port), "$HOME")
    find = (f"FOUND=\"$(find / -maxdepth 6 \\( -path /proc -o -path /sys -o -path /dev \\) -prune -o "
            f"-path '*/{scratch}/{version}/manifest.json' -print 2>/dev/null | head -n 1)\"\n"
            f"if [ -z \"$FOUND\" ]; then echo \"{scratch}/{version}/manifest.json is not on this pod\"; exit 1; fi\n"
            f"ROOT=\"$(dirname \"$(dirname \"$FOUND\")\")\"")
    for old, new in (('ROOT="$HOME"', find), ('DIR="$HOME/lab"', f'DIR="{POD_DIR}"'),
                     ('LOG="$HOME/package_server.log"', 'LOG="$DIR/server.log"'), ("package_server.py", SERVER_FILE)):
        if old not in base:
            raise RuntimeError(f"lab/package_pull.py's start script no longer has {old!r}; update lab/link_speed.py")
        base = base.replace(old, new)
    return base


def stop_script() -> str:
    """The shell script that stops that file server and removes its directory, and nothing else."""
    return f"""#!/bin/bash
# Stops the R14 file server and removes its directory. Written by `python -m lab link-speed`.
set -u
pkill -u "$(id -u)" -f '{SERVER_FILE} --root' >/dev/null 2>&1 || true
rm -rf "{POD_DIR}"
echo "r14 file server stopped and removed"
exit 0
"""


class FreshLink(PodLink):
    """A PodLink whose reads of the contents API cannot come from Cloudflare's cache. Measured on the SJC hub on
    2026-10-11: `GET /user/<account>/api/contents/<path>?content=0` is answered `cache-control: public, max-age=14400`
    with `cf-cache-status: HIT`, so the size of a file read back after it was rewritten stays the old size for four
    hours (a changed query string is a MISS). lab/transfer.py reads the size back after every push and before every
    retry; on a repeat of the same path that read would report the previous push's size. Every GET below /api/contents
    here carries a query parameter that is never repeated. (The proxy route /proxy/<port>/ answers `no-store`.)"""

    _tick = itertools.count(1)

    def request(self, method: str, rel: str, **kwargs: Any) -> Any:
        if method.upper() == "GET" and "/api/contents" in rel:
            salt = self.__dict__.setdefault("_salt", secrets.token_hex(3))
            params = dict(kwargs.get("params") or {})
            params["_"] = f"{salt}{next(FreshLink._tick)}"
            kwargs["params"] = params
        return super().request(method, rel, **kwargs)


class LiveTarget:
    """One account's pod reached through the hub: the contents API for the push (lab/transfer.py), the proxy for the pull
    (lab/package_pull.py), and the controller's run_script for the file server. transport and pull_transport replace the
    network (httpx transports) for tests; serve and unserve replace the two pod scripts."""

    def __init__(self, account: Account, *, port: int = DEFAULT_PORT, base_url: str | None = None, transport: Any = None,
                 pull_transport: Any = None, environ: Mapping[str, str] | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.perf_counter,
                 serve: Callable[[str, str], None] | None = None, unserve: Callable[[], None] | None = None) -> None:
        self.account_obj = account
        self.account = account.letter
        self.node = account.node
        self.port = port
        self.sleep = sleep
        self.clock = clock
        self._environ = environ
        self._base_url = base_url
        self._transport = transport
        self._serve = serve
        self._unserve = unserve
        self.link = FreshLink.for_account(account, base_url=base_url, transport=transport, environ=environ, sleep=sleep, clock=clock)
        self.files = PodFiles(self.link)
        self.client = PackageClient(account, port, base_url=base_url, transport=pull_transport or transport, sleep=sleep,
                                    environ=environ)
        self.hub = hub_of(self.link.base_url)
        self._controller: Controller | None = None
        self._leftovers: list[str] = []

    def close(self) -> None:
        self.client.close()
        self.link.close()
        if self._controller is not None:
            self._controller.close()

    # the contents API

    def push(self, local: Path, remote: str, chunk_bytes: int) -> TransferStats:
        return transfer.push_file(self.files, local, remote, chunk_bytes=chunk_bytes)

    def put_once(self, remote: str, data: bytes) -> int | None:
        """One PUT of the whole body, no chunking, and the size the pod reports for it. Raises the ProxyError of a refusal."""
        transfer.clean_remote(remote)
        self.files.ensure_dir(remote.rsplit("/", 1)[0] if "/" in remote else "")
        body = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode("ascii")}
        self.link.request("PUT", self.files.contents_rel(remote), json=body, retry=False)
        return self.files.size_of(remote)

    def remove(self, remote: str) -> None:
        self.files.delete(remote)

    def cleanup(self, scratch: str) -> list[str]:
        """Delete the scratch folder, file by file and folder by folder, then the files the controller left in the Jupyter
        root for its scripts. Returns what could not be removed (empty when the pod is as found)."""
        problems: list[str] = []

        def walk(path: str) -> None:
            reply = self.link.request("GET", self.files.contents_rel(path))
            try:
                entries = reply.json().get("content") or []
            except (ValueError, AttributeError):
                entries = []
            for entry in entries:
                name = f"{path}/{entry.get('name')}"
                if entry.get("type") == "directory":
                    walk(name)
                else:
                    self.files.delete(name)
            self.files.delete(path)

        try:
            if self.files.stat(scratch) is not None:
                walk(scratch)
            if self.files.stat(scratch) is not None:
                problems.append(f"{scratch} is still on the pod")
        except (ProxyError, TransferError) as exc:
            problems.append(f"{scratch}: {type(exc).__name__}")
        for name in self._leftovers:
            try:
                self.files.delete(name)
            except (ProxyError, TransferError) as exc:
                problems.append(f"{name}: {type(exc).__name__}")
        self._leftovers.clear()
        return problems

    # the pod's file server

    def _controller_for(self) -> Controller:
        if self._controller is None:
            self._controller = Controller({self.account: self.account_obj}, base_url=self._base_url, transport=self._transport,
                                          sleep=self.sleep, environ=self._environ)
        return self._controller

    def _run(self, script: str, name: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="r14-script-") as folder:
            path = Path(folder) / name
            path.write_text(script, encoding="utf-8", newline="\n")
            done = self._controller_for().run_script(self.account, path, timeout_s=SERVE_TIMEOUT_S)
        remote = done.get("remote")
        if isinstance(remote, str):
            self._leftovers += [remote, remote[:-3] + ".out" if remote.endswith(".sh") else remote + ".out"]
        return done

    def serve(self, scratch: str, version: str) -> None:
        if self._serve is not None:
            self._serve(scratch, version)
            return
        done = self._run(serve_script(self.port, scratch, version), "r14_serve.sh")
        if not done.get("ok"):
            raise TransferError(f"the file server did not start on the pod: {done.get('outcome')} {done.get('error') or ''}".strip())
        try:
            listing = json.loads(package_pull.fetch_small(self.client, self.client.proxy_path(), sleep=self.sleep))
        except (package_pull.PullError, ValueError) as exc:
            raise TransferError(f"the file server started but the proxy cannot reach it: {type(exc).__name__}") from None
        if version not in (listing.get("versions") or []):
            raise TransferError(f"the file server on port {self.port} does not list {version}")

    def unserve(self) -> None:
        if self._unserve is not None:
            self._unserve()
            return
        done = self._run(stop_script(), "r14_stop.sh")
        if not done.get("ok"):
            raise TransferError(f"the file server could not be stopped: {done.get('outcome')} {done.get('error') or ''}".strip())

    # the proxy

    def pull(self, version: str, dest: Path, part_bytes: int) -> dict[str, Any]:
        return package_pull.pull_package(self.client, version, dest, part_bytes=part_bytes, force=True, clock=self.clock,
                                         sleep=self.sleep, write_row=False)


# --- the measurement ----------------------------------------------------------------------------------------

class _Ledger:
    """Rows of the run, written one at a time; a row that cannot be written is remembered and never stops the run."""

    def __init__(self, target: Target, report: Report, state_dir: Any) -> None:
        self.target = target
        self.report = report
        self.state_dir = state_dir

    def write(self, verb: str, outcome: str, metrics: Mapping[str, float], units: Mapping[str, str], conditions: Mapping[str, Any],
              note: str | None, **fields: Any) -> None:
        extra: dict[str, Any] = {"node": self.target.node, "conditions": dict(conditions),
                                 "hub": self.report.hub, **{k: v for k, v in fields.items() if v is not None}}
        if metrics:
            extra["units"] = dict(units)
        if note:
            extra["note"] = " ".join(note.split())[:NOTE_CHARS]
        try:
            record.record(verb, outcome, dict(metrics), {"account": self.report.account, "run_id": self.report.run_id},
                          state_dir=self.state_dir, **extra)
        except Exception as exc:
            self.report.ledger_errors.append(f"{verb}: {type(exc).__name__}: {exc}")

    def run(self, item: SizeResult, run: Run) -> None:
        push = run.direction == "push"
        metric = PUSH_METRIC if push else PULL_METRIC
        metrics = {metric: run.mb_s} if run.ok and run.mb_s is not None else {}
        conditions = {"path": "contents-api-chunked" if push else "proxy-range", "package": item.label, "bytes": run.bytes,
                      "chunk_bytes" if push else "part_bytes": self.report.chunk_bytes if push else self.report.part_bytes,
                      "hub": self.report.hub, "client": self.report.client, "repeat": run.repeat}
        fields: dict[str, Any] = {"bytes": run.bytes, "seconds": round(run.seconds, 3), "package": item.label,
                                  "package_bytes": item.size_bytes, "repeat": run.repeat, "retries": run.retries,
                                  "parts": run.parts, "kind": run.kind, "wall_seconds": run.wall_seconds,
                                  "size_ok": run.size_ok, "sha256_ok": run.sha256_ok}
        fields["chunk_bytes" if push else "part_bytes"] = self.report.chunk_bytes if push else self.report.part_bytes
        self.write(PUSH_ROW if push else PULL_ROW, "ok" if run.ok else "error", metrics, {metric: "MB/s"}, conditions,
                   None if run.ok else run.error, **fields)


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _push_run(target: Target, source: Path, remote: str, item: SizeResult, repeat: int, chunk_bytes: int) -> Run:
    run = Run("push", repeat, item.size_bytes)
    try:
        stats = target.push(source, remote, chunk_bytes)
    except (ProxyError, TransferError, OSError) as exc:
        run.kind, run.error = getattr(exc, "kind", type(exc).__name__), str(exc)
        return run
    run.bytes, run.seconds, run.mb_s = stats.bytes, stats.seconds, stats.mb_s
    run.parts, run.retries, run.size_ok = stats.requests, stats.retries, bool(stats.verified)
    run.ok = bool(stats.verified) and stats.bytes == item.size_bytes
    if not run.ok:
        run.kind, run.error = "size_mismatch", "the pod does not report the size that was sent"
    return run


def _pull_run(target: Target, item: SizeResult, version: str, dest: Path, repeat: int, part_bytes: int) -> Run:
    run = Run("pull", repeat, item.size_bytes)
    answer = target.pull(version, dest, part_bytes)
    run.bytes = int(answer.get("bytes") or 0)
    run.seconds = float(answer.get("seconds") or 0.0)
    run.mb_s = _finite(answer.get("mb_s"))
    run.wall_seconds = _finite(answer.get("wall_seconds"))
    run.parts = answer.get("parts")
    run.retries = int(answer.get("retries") or 0)
    got = dest / version / PAYLOAD_NAME
    if answer.get("ok"):
        run.sha256_ok = got.is_file() and sha256_file(got) == item.sha256        # the check again, against the generated file
        run.ok = run.sha256_ok and run.bytes == item.size_bytes
        if not run.ok:
            run.kind, run.error = "checksum_mismatch", "the pulled copy does not match the generated file"
    else:
        run.kind, run.error = answer.get("kind") or "error", str(answer.get("error") or answer.get("outcome") or "")
        if run.kind == "checksum_mismatch":
            run.sha256_ok = False
    shutil.rmtree(dest, ignore_errors=True)
    return run


def probe_part_limit(target: Target, scratch: str, *, start_bytes: int = PROBE_START_BYTES,
                     ceiling_bytes: int = PROBE_CEILING_BYTES, seed: int = DEFAULT_SEED, state_dir: Any = None,
                     run_id: str | None = None, report: Report | None = None, progress: Callable[[str], None] | None = None) -> Probe:
    """Double the size of one single PUT until the contents API refuses it (or the ceiling is reached) and keep the last
    size it accepted. A refusal that may be a gateway hiccup is asked once more before it counts."""
    if not 1 <= start_bytes <= ceiling_bytes:
        raise ValueError("the probe starts at one byte or more and ends at or above its start")
    remote = f"{scratch}/{PROBE_NAME}"
    rng = random.Random(seed)
    probe = Probe(ceiling_bytes=ceiling_bytes)
    size = start_bytes
    try:
        while True:
            data = rng.randbytes(size)
            step = ProbeStep(size, False)
            for attempt in range(PROBE_TRIES):
                began = time.perf_counter()
                try:
                    got = target.put_once(remote, data)
                except (ProxyError, TransferError) as exc:
                    step.status, step.kind, step.error = getattr(exc, "status", None), getattr(exc, "kind", type(exc).__name__), str(exc)
                    if getattr(exc, "retryable", False) and attempt + 1 < PROBE_TRIES:
                        time.sleep(PROBE_PAUSE_S)
                        continue
                else:
                    step.accepted = got == size
                    if not step.accepted:
                        step.kind, step.error = "size_mismatch", f"the pod holds {got} bytes of {size}"
                step.seconds = time.perf_counter() - began
                break
            probe.steps.append(step)
            if progress is not None:
                progress(f"probe {fmt_bytes(size)}: {'accepted' if step.accepted else 'refused ' + str(step.status or step.kind)}")
            del data
            if not step.accepted:
                probe.first_refused_bytes = size
                break
            probe.largest_accepted_bytes = size
            if size >= ceiling_bytes:
                break
            size = min(size * 2, ceiling_bytes)
    finally:
        try:
            target.remove(remote)
        except (ProxyError, TransferError):
            pass
    scratch_report = report or Report(run_id or "r14-probe", target.node, target.hub, target.account, package_pull.client_label(),
                                      "", 0, 0, 0, 0, seed)
    if run_id:
        scratch_report.run_id = run_id
    metrics = {PART_METRIC: round(probe.largest_accepted_bytes / 1e6, 6)} if probe.largest_accepted_bytes else {}
    conditions = {"path": "contents-api-single-put", "encoding": "base64", "start_bytes": start_bytes, "ceiling_bytes": ceiling_bytes,
                  "hub": scratch_report.hub, "client": scratch_report.client}
    _Ledger(target, scratch_report, state_dir).write(
        PROBE_ROW, "ok" if probe.largest_accepted_bytes else "error", metrics, {PART_METRIC: "MB"}, conditions,
        None if probe.largest_accepted_bytes else "no single PUT was accepted",
        largest_accepted_bytes=probe.largest_accepted_bytes, first_refused_bytes=probe.first_refused_bytes,
        ceiling_reached=probe.ceiling_reached,
        steps=[{"bytes": s.bytes, "accepted": s.accepted, "status": s.status, "kind": s.kind, "seconds": round(s.seconds, 3)}
               for s in probe.steps])
    return probe


def measure(target: Target, *, sizes: Sequence[int], repeats: int = DEFAULT_REPEATS, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
            part_bytes: int = DEFAULT_PART_BYTES, scratch: str = DEFAULT_SCRATCH, workdir: Path | None = None,
            seed: int = DEFAULT_SEED, probe: bool = True, probe_start_bytes: int = PROBE_START_BYTES,
            probe_ceiling_bytes: int = PROBE_CEILING_BYTES, port: int = DEFAULT_PORT, state_dir: Any = None,
            run_id: str | None = None, progress: Callable[[str], None] | None = None) -> Report:
    """Run the whole measurement against `target` and return the report. The scratch folder on the pod, the file server
    and the local files are removed on the way out, whether the run finished or stopped on an error."""
    _check_word(scratch, "the scratch folder")
    if repeats < 1 or not sizes:
        raise ValueError("at least one size and one repeat")
    say = progress or (lambda line: None)
    started = dt.datetime.now(dt.timezone.utc)
    report = Report(run_id or started.strftime("r14-%Y%m%d-%H%M%S-") + secrets.token_hex(2), target.node, target.hub, target.account,
                    package_pull.client_label(), started.strftime("%Y-%m-%d %H:%M UTC"), repeats, chunk_bytes, part_bytes, port, seed)
    ledger = _Ledger(target, report, state_dir)
    own_dir = workdir is None
    work = Path(tempfile.mkdtemp(prefix="r14-")) if workdir is None else Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    served = False
    try:
        if probe:
            report.probe = probe_part_limit(target, scratch, start_bytes=probe_start_bytes, ceiling_bytes=probe_ceiling_bytes,
                                            seed=seed, state_dir=state_dir, report=report, progress=say)
        for size in sizes:
            label = size_label(size)
            version = f"r14-{label.lower()}"
            source = work / f"{label}.bin"
            say(f"{label}: writing {size} bytes of seeded random data")
            item = SizeResult(label, size, make_payload(source, size, seed))
            report.sizes.append(item)
            manifest = work / f"{label}-manifest.json"
            manifest.write_bytes(manifest_bytes(version, item.sha256))
            remote_dir = f"{scratch}/{version}"
            try:
                for repeat in range(1, repeats + 1):
                    say(f"{label}: push {repeat} of {repeats}")
                    push = _push_run(target, source, f"{remote_dir}/{PAYLOAD_NAME}", item, repeat, chunk_bytes)
                    if push.ok:
                        try:
                            target.push(manifest, f"{remote_dir}/manifest.json", chunk_bytes)
                        except (ProxyError, TransferError, OSError) as exc:
                            push.ok, push.kind, push.error = False, getattr(exc, "kind", type(exc).__name__), f"manifest: {exc}"
                    item.runs.append(push)
                    ledger.run(item, push)
                    if not push.ok:
                        continue
                    if not served:
                        served = True
                        target.serve(scratch, version)
                    say(f"{label}: pull {repeat} of {repeats}")
                    pull = _pull_run(target, item, version, work / f"pull-{label}-{repeat}", repeat, part_bytes)
                    item.runs.append(pull)
                    ledger.run(item, pull)
            finally:
                source.unlink(missing_ok=True)
                manifest.unlink(missing_ok=True)
    finally:
        if served:
            try:
                target.unserve()
            except (ProxyError, TransferError) as exc:
                report.cleanup_problems.append(f"the file server may still run: {type(exc).__name__}")
        report.cleanup_problems += target.cleanup(scratch)
        if own_dir:
            shutil.rmtree(work, ignore_errors=True)
        else:
            for leftover in work.glob("*"):
                shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() else leftover.unlink(missing_ok=True)
            try:
                work.rmdir()
            except OSError:
                pass
    return report


# --- the command line ---------------------------------------------------------------------------------------

def _mib(low: float, high: float) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a number of MiB") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{text} MiB is outside {low} to {high}")
        return int(value * MIB)
    return parse


def _sizes(text: str) -> list[int]:
    try:
        values = [float(piece) for piece in text.split(",") if piece.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is a comma-separated list of megabytes, such as 1000,3000") from None
    if not values or any(value < 0.001 for value in values):
        raise argparse.ArgumentTypeError("every size is at least 0.001 MB")
    return [int(round(value * MB)) for value in values]


def _count(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if not 1 <= value <= 20:
        raise argparse.ArgumentTypeError("repeats are 1 to 20")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab link-speed", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--account", required=True, help="the account letter of the pod, e.g. D")
    parser.add_argument("--sizes-mb", type=_sizes, default=[int(v * MB) for v in DEFAULT_SIZES_MB],
                        help="package sizes in MB (10^6 bytes), comma separated (default 1000,3000)")
    parser.add_argument("--repeats", type=_count, default=DEFAULT_REPEATS)
    parser.add_argument("--chunk-mib", type=_mib(0.001, 64), default=DEFAULT_CHUNK_BYTES, help="push chunk (at most 64)")
    parser.add_argument("--part-mib", type=_mib(0.001, 95.99), default=DEFAULT_PART_BYTES, help="pull Range part (under 96)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="the pod port for the file server")
    parser.add_argument("--probe-ceiling-mib", type=_mib(0.001, 512), default=PROBE_CEILING_BYTES)
    parser.add_argument("--no-probe", action="store_true", help="skip the largest-part probe")
    parser.add_argument("--scratch", default=DEFAULT_SCRATCH, help="the folder on the pod everything goes under")
    parser.add_argument("--workdir", default=None, help="a local folder for the generated files (default: a temp folder)")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="where the Markdown table is written")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, target: Target | None = None, environ: Mapping[str, str] | None = None,
         env_files: Sequence[Path] | None = None, transport: Any = None, state_dir: Any = None,
         sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Measure push and pull speed of a synthetic package on a pod through the hub, and write the table."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if not re.fullmatch(r"[A-Za-z]", args.account):
        raise SystemExit(2)
    letter = args.account.upper()
    own_target = target is None
    if target is None:
        try:
            account = load_accounts([letter], environ=environ, env_files=env_files, require_role=True)[letter]
        except MissingCredentials as exc:
            return {"ok": False, "outcome": "no_credentials", "missing": exc.missing,
                    "error": "keys missing from the environment and the .env files"}
        except PodRoleError as exc:
            return {"ok": False, "outcome": "no_pod_role", "error": str(exc)}
        target = LiveTarget(account, port=args.port, transport=transport, environ=environ, sleep=sleep, clock=clock)
    say = None if args.quiet else (lambda line: print(f"link-speed: {line}", file=sys.stderr, flush=True))
    try:
        report = measure(target, sizes=args.sizes_mb, repeats=args.repeats, chunk_bytes=args.chunk_mib, part_bytes=args.part_mib,
                         scratch=args.scratch, workdir=None if args.workdir is None else Path(args.workdir), seed=args.seed,
                         probe=not args.no_probe, probe_ceiling_bytes=args.probe_ceiling_mib, port=args.port,
                         state_dir=state_dir, progress=say)
    finally:
        if own_target:
            target.close()
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.markdown(), encoding="utf-8", newline="\n")
    return {"ok": report.ok, "outcome": "ok" if report.ok else "incomplete", "run_id": report.run_id, "node": report.node,
            "hub": report.hub, "account": report.account, "runs": len(report.runs), "report": str(path),
            "largest_file_bytes": report.largest_file_bytes,
            "largest_part_bytes": None if report.probe is None else report.probe.largest_accepted_bytes,
            "cleanup_problems": report.cleanup_problems, "ledger_errors": report.ledger_errors,
            "sizes": [{"label": item.label, "push": None if item.speeds("push") is None else round(item.speeds("push").mean, 3),
                       "pull": None if item.speeds("pull") is None else round(item.speeds("pull").mean, 3),
                       "sha256_passed": item.pass_rate[0], "pulls": item.pass_rate[1]} for item in report.sizes]}


if __name__ == "__main__":
    result = main()
    print(json.dumps(result))
    sys.exit(0 if result.get("ok") else 1)
