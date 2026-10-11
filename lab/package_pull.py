"""package_pull: pull a model package from a pod through the hub proxy, in Range parts, and check it.

    python -m lab.package_pull pull --account C --version V --dest DIR [--port 8300] [--part-mib 64] [--force] [--quiet]
    python -m lab.package_pull probe --account C --version V [--port 8300] [--file NAME] [--max-read-mib 116]
    python -m lab.package_pull serve-start --account C [--port 8300] [--root '$HOME/data/packages']
    python -m lab.package_pull serve-stop --account C [--port 8300]
    python -m lab.package_pull curl --account C --version V [--port 8300] [--file NAME]

Run it from the repo root. The pod runs lab/package_server.py (serve-start puts it there and starts it), which
serves ~/data/packages/<version>/ (manifest.json, the GGUF and the vision projector) on a fixed port and answers
HTTP Range requests. The box reaches it at https://<hub>/user/<account>/proxy/<port>/<version>/<file> with the
account's token. Every call answers with one JSON object on stdout and an exit code, as `python -m lab.controller`
does: 0 when it worked, 1 when it did not, 2 for bad arguments and for missing keys. The keys are
AUP_HUB_USER_<letter> and AUP_HUB_TOKEN_<letter>, found as lab/hub.py finds them; no key and no account name is
printed or written to the ledger.

pull. Reads the package's manifest.json (ModelManifest, contracts/learning.py), then pulls each file it names in
Range parts and checks the file's sha256 against the manifest. A part is 64 MiB unless --part-mib says otherwise and
never 96 MiB or more: jupyter-server-proxy 4.6.0 buffers a response and its HTTP client refuses a body over
100 MiB (Tornado's default max_body_size), so one GET of a multi-GB GGUF cannot cross the proxy and python3's
http.server cannot send a Range part. Files land in DIR/<version>/ as NAME.part and are renamed once their sha256
matches; manifest.json is written last, so a half-pulled package is never mistaken for an installed version. A pull
that stops (a dropped connection, a pod relaunch) continues from the bytes already on disk; a copy that fails its
checksum after a resume is fetched once more from the start; a pull whose files are all there and correct sends
nothing. Once every file has matched, the pull writes DIR/<version>/.verified.json (runtime/modelstore.py's
VERIFIED_RECORD): per file the name, size, sha256 and mtime_ns it had when it was hashed, written_by
lab/package_pull.py. The record is removed before the files are touched and written again only after all of them
matched, so it never vouches for a file this pull did not check. The swap (jetson/atlas_swap.py --trust-verified) and
the deploy verb (lab/deploy.py) trust it instead of hashing again while the size and mtime still match; the pull
itself always hashes. The proxy answering with a gateway error (502, 503, 504, 52x, 599, 429) or the connection dropping is
tried again, with growing pauses; a login redirect (the token), 403, 404, 424 (the pod is stopped) and 500 (nothing
listens on the port) are not.

probe. One GET of the whole GGUF through the proxy, once, with no retry, read up to --max-read-mib and then
dropped. It records the status the proxy answered, the Content-Length it announced and how many bytes arrived:
this is the measurement of whether the 100 MiB limit holds on the hub in use, and the reason pull never asks for
the whole file.

serve-start and serve-stop. Run a script on the pod through Controller.run_script (lab/controller.py). The script
carries lab/package_server.py in a heredoc, writes it to ~/lab/ and starts it with nohup, so nothing has to be
uploaded first; serve-start then asks the server's listing through the proxy. They need a ready pod and refuse to
run while lab/state/STOP exists, as the controller's commands do.

curl. Prints the curl lines that prove the path from outside the pod, with the token left as a shell variable.

Speed. link.pull_mb_s is the bytes this run pulled over the seconds the transfers took (each part's request until
its last byte, pauses between tries left out), in MB of 10^6 bytes. A pull that sent nothing records no speed.

The ledger. A command writes a row for the call (verbs `package pull`, `package probe`, `package serve-start`,
`package serve-stop`, through @recorded) and the measurement rows, with the node (pod-a, pod-b, pod-c ...) and
refs.account: `link pull` carries link.pull_mb_s with units MB/s, conditions {path, bytes, part_bytes, hub,
client}, and the fields bytes, seconds, part_bytes, parts, retries, wall_seconds, version and files; `link probe`
carries status, content_length, bytes_read, seconds and cap_bytes. A ledger that cannot be written never undoes
the pull: the answer then carries ledger_error.

Outcomes. pull: ok, already_present, error (kind: auth, missing, pod_stopped, server_down, range_ignored,
oversized_part, bad_answer, file_changed, checksum_mismatch, bad_manifest, unreachable, proxy_error, no_space,
unexpected_status), stopped. probe: whole_file_refused, whole_file_served, error (the same kinds, and
file_too_small), stopped. serve-start: ok, error, and the controller's run-script outcomes.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import platform
import re
import secrets
import shutil
import sys
import tempfile
import time
import traceback
import urllib.parse
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):          # started as `python lab/package_pull.py`: put the repo root on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from contracts.learning import ModelManifest
from lab import hub as hubs
from lab import package_server
from lab.controller import Controller
from lab.hub import Account, Hub, MissingCredentials, PodRoleError, load_accounts, one_line
from lab.machine import record
from lab.machine import stop as kill_switch
from lab.machine.record import recorded
from runtime.modelstore import VERIFIED_BY, VERIFIED_RECORD

__all__ = ["PackageClient", "PullError", "Transient", "pull_package", "probe_whole_file", "fetch_manifest", "package_files",
           "serve_start", "serve_stop", "curl_lines", "run", "main", "build_parser", "write_verified"]

log = logging.getLogger("lab.package_pull")

MIB = 1 << 20
PROXY_LIMIT_BYTES = 100 * MIB             # jupyter-server-proxy 4.6.0 refuses a response body larger than this
MAX_PART_BYTES = 96 * MIB                 # a part stays well under the limit
DEFAULT_PART_BYTES = 64 * MIB
MIN_PART_BYTES = 1024
PROBE_READ_BYTES = PROXY_LIMIT_BYTES + 16 * MIB    # enough bytes of one response to say the limit did not hold
DEFAULT_PORT = package_server.DEFAULT_PORT
DEFAULT_SERVER_ROOT = "$HOME/data/packages"
PATH_RANGE = "proxy-range"
PATH_WHOLE = "proxy-whole-file"
PART_TIMEOUT_S = 120.0                    # silence tolerated while a part arrives
PROBE_TIMEOUT_S = 300.0
MANIFEST_LIMIT_BYTES = 1 << 20
DISK_MARGIN_BYTES = 64 * MIB              # free space kept beyond the bytes still to come
CHUNK_BYTES = 1 << 20
DETAIL_CHARS = 200
NOTE_CHARS = 300
TAIL_CHARS = 1500
MIN_NAME_CHARS = 4                        # a shorter account name would mangle the text it is cut out of
SCRIPT_TIMEOUT_S = 120.0
SENTINEL = "PACKAGE_SERVER_SOURCE_END"
#: What a Range part may be answered with and be tried again: the gateway errors of lab/hub.py, Cloudflare's own 52x
#: (520 to 524), 429, and 599, which jupyter-server-proxy answers for a timeout or a stream its client closed.
TRANSIENT_STATUSES = hubs.TRANSIENT_STATUSES | {429, 520, 521, 522, 523, 524, 599}
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
CONTENT_RANGE = re.compile(r"bytes (?P<first>\d+)-(?P<last>\d+)/(?P<total>\d+|\*)")
UNSATISFIED_RANGE = re.compile(r"bytes \*/(?P<total>\d+)")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
ROOT_RE = re.compile(r"(~|\$HOME|/)[A-Za-z0-9_./-]*")

PULL_ROW, PROBE_ROW = "link pull", "link probe"
PULL_METRIC = "link.pull_mb_s"


# --- what can go wrong ------------------------------------------------------------------------------------------

class PullError(Exception):
    """A pull or probe that cannot go on. kind says why (see the module docstring), status is the HTTP status or None."""

    def __init__(self, kind: str, status: int | None, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.message = message


class Transient(Exception):
    """One try that failed in a way worth trying again: a gateway error, or a connection that dropped."""

    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def backoff_s(failures: int) -> float:
    """The pause after the failures-th failure in a row: 1, 2, 4 seconds, as lab/hub.py pauses between its tries."""
    return min(hubs.BACKOFF_FIRST_S * 2 ** max(0, failures - 1), hubs.BACKOFF_MAX_S)


def client_label() -> str:
    """The kind of machine this runs on, such as windows-amd64 or linux-aarch64 (the Jetson): the link speed depends on
    it, and a host name would say more than the ledger needs."""
    return f"{platform.system()}-{platform.machine()}".lower()


def check_part_bytes(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not MIN_PART_BYTES <= value <= MAX_PART_BYTES:
        raise ValueError(f"a part is {MIN_PART_BYTES} to {MAX_PART_BYTES} bytes (at most {MAX_PART_BYTES // MIB} MiB), "
                         f"not {value!r}: the hub proxy refuses a response body over {PROXY_LIMIT_BYTES // MIB} MiB")
    return value


def check_name(name: str, what: str) -> str:
    if not isinstance(name, str) or not package_server.NAME_RE.match(name):
        raise ValueError(f"{what} is letters, digits, dot, dash and underscore (not starting with a dot), not {name!r}")
    return name


def _guard() -> None:
    """The kill switch: nothing that talks to a pod runs while lab/state/STOP exists."""
    if kill_switch.stopped():
        raise kill_switch.Stopped(kill_switch.stop_reason() or kill_switch.NO_REASON)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


# --- the pod's file server, as the proxy shows it ---------------------------------------------------------------

class PackageClient(Hub):
    """One account's package server, reached through the hub's proxy at /user/<account>/proxy/<port>/.

    A Hub (lab/hub.py) with the same token, browser agent, address rules and redirect rule, plus the calls the hub
    client has no use for: a streamed GET whose body is bytes, never text. A Reply keeps only 2000 characters of
    a body, which is no way to carry a part."""

    def __init__(self, account: Account, port: int = DEFAULT_PORT, *, part_timeout_s: float = PART_TIMEOUT_S,
                 **hub_args: Any) -> None:
        super().__init__(account, **hub_args)
        self.port = port
        self.part_timeout_s = part_timeout_s

    def __repr__(self) -> str:
        return f"PackageClient(account={self.account.letter!r}, port={self.port}, base_url={self.base_url!r})"

    @property
    def host(self) -> str:
        return urllib.parse.urlsplit(self.base_url).netloc

    @property
    def node(self) -> str:
        """The ledger's name for the machine behind the account: pod-b for B, pod-c for C."""
        return self.account.node

    def scrub(self, text: str) -> str:
        """text with the token cut out and the account's name made <account C>."""
        text = super().scrub(text)
        if len(self.account.user) >= MIN_NAME_CHARS:
            text = text.replace(self.account.user, f"<account {self.account.letter}>")
        return text

    def proxy_path(self, *names: str) -> str:
        """The address behind the proxy of the server's root, or of the version or file the names lead to."""
        tail = "/".join(urllib.parse.quote(name, safe="") for name in names)
        return self.pod_url(f"/proxy/{self.port}/{tail}")

    @contextlib.contextmanager
    def opened(self, path: str, *, headers: Mapping[str, str] | None = None,
               timeout_s: float | None = None) -> Iterator[httpx.Response]:
        """One streaming GET. A connection that fails or drops, while the answer is opened or while its body is read
        inside the with block, becomes Transient."""
        timeout = httpx.Timeout(timeout_s or self.part_timeout_s, connect=self.timeout_s)
        try:
            with self._client.stream("GET", path, headers=headers, timeout=timeout) as response:
                yield response
        except httpx.TransportError as exc:
            raise Transient(None, self.scrub(f"{type(exc).__name__}: {exc}")) from None

    def error_text(self, response: httpx.Response) -> str:
        """The start of an error answer's body on one line, markup taken out."""
        raw = b""
        try:
            for block in response.iter_bytes(512):
                raw += block
                if len(raw) >= 2000:
                    break
        except httpx.TransportError:
            pass
        text = re.sub(r"<[^>]+>", " ", raw[:2000].decode("utf-8", errors="replace"))
        return one_line(self.scrub(text), DETAIL_CHARS)

    def problem(self, response: httpx.Response) -> PullError | Transient:
        """What a status that is not a success means: Transient when another try may work, else a PullError."""
        status = response.status_code
        text = self.error_text(response)
        said = f": {text}" if text else ""
        if status in REDIRECT_STATUSES:
            return PullError("auth", status, "redirected to the hub login: the token is wrong or revoked, or belongs "
                                             "to another account")
        if status in (401, 403):
            return PullError("auth", status, f"HTTP {status}, the hub refused the token{said}")
        if status == 404:
            return PullError("missing", status, f"HTTP 404: the file, the version, the server or the proxy route is not there{said}")
        if status == 424:
            return PullError("pod_stopped", status, "HTTP 424: the pod is not running")
        if status == 500:
            return PullError("server_down", status, "HTTP 500 from the proxy: nothing is listening on the port, or the "
                                                    f"proxy failed; start the server with serve-start{said}")
        if status in TRANSIENT_STATUSES:
            return Transient(status, f"HTTP {status}{said}")
        return PullError("unexpected_status", status, f"HTTP {status}{said}")

    def accept(self, response: httpx.Response, *, start: int, part_bytes: int, total: int | None) -> tuple[int, int, int]:
        """(first, last, total) of the bytes a Range answer holds, or the PullError or Transient its status calls for.

        A 206 is checked against the Content-Range it carries. A 200 is accepted only for a file that fits one
        part and was asked for from byte 0: a server that ignores Range would send a multi-GB body whole."""
        status = response.status_code
        if status == 206:
            found = CONTENT_RANGE.fullmatch(response.headers.get("content-range", "").strip())
            if found is None or found["total"] == "*":
                raise PullError("bad_answer", 206, "a 206 answer with no usable Content-Range")
            first, last, size = int(found["first"]), int(found["last"]), int(found["total"])
            if first != start or last < first:
                raise PullError("bad_answer", 206, f"asked for bytes from {start}, the answer holds {first}-{last}")
            if last - first + 1 > part_bytes:
                raise PullError("oversized_part", 206, f"asked for at most {part_bytes} bytes, the answer holds {last - first + 1}")
            if total is not None and size != total:
                raise PullError("file_changed", 206, f"the file was {total} bytes, now {size}")
            return first, last, size
        if status == 200:
            length = _int_header(response, "content-length")
            if start == 0 and length is not None and length <= part_bytes and total in (None, length):
                return 0, length - 1, length
            raise PullError("range_ignored", 200, "the server answered 200 to a Range request for a file larger than a "
                                                  "part: it does not do Range (python3 -m http.server does not); "
                                                  "run lab/package_server.py")
        if status == 416:
            found = UNSATISFIED_RANGE.fullmatch(response.headers.get("content-range", "").strip())
            if found is not None and int(found["total"]) == start and total in (None, start):
                return start, start - 1, start              # nothing left: the file is already on disk, or is empty
            raise PullError("file_changed", 416, f"asked for bytes from {start}, which the file does not have")
        raise self.problem(response)


def _int_header(response: httpx.Response, name: str) -> int | None:
    value = response.headers.get(name, "").strip()
    return int(value) if value.isdigit() else None


# --- the manifest -----------------------------------------------------------------------------------------------

def fetch_small(client: PackageClient, path: str, *, limit: int = MANIFEST_LIMIT_BYTES,
                sleep: Callable[[float], None] = time.sleep, retries: int | None = None) -> bytes:
    """The body of a small file (a manifest, a listing), tried again on a gateway error or a dropped connection
    (client.retries times unless `retries` says otherwise)."""
    allowed = client.retries if retries is None else retries
    failures = 0
    while True:
        try:
            with client.opened(path, headers={"Accept": "application/json", "Accept-Encoding": "identity"}) as response:
                if response.status_code != 200:
                    raise client.problem(response)
                body = bytearray()
                for block in response.iter_bytes():
                    body += block
                    if len(body) > limit:
                        raise PullError("bad_answer", 200, f"the answer is larger than {limit} bytes")
                return bytes(body)
        except Transient as exc:
            failures += 1
            if failures > allowed:
                raise PullError("unreachable" if exc.status is None else "proxy_error", exc.status,
                                f"gave up after {failures} tries: {exc.detail}") from None
            sleep(backoff_s(failures))


def fetch_manifest(client: PackageClient, version: str, *, sleep: Callable[[float], None] = time.sleep) -> tuple[bytes, ModelManifest]:
    """(the manifest.json bytes as the pod has them, the manifest) of a version."""
    raw = fetch_small(client, client.proxy_path(version, package_server.MANIFEST), sleep=sleep)
    try:
        return raw, ModelManifest.model_validate_json(raw)
    except ValueError as exc:
        raise PullError("bad_manifest", 200, f"manifest.json is not a ModelManifest: {one_line(exc, DETAIL_CHARS)}") from None


def package_files(manifest: ModelManifest) -> list[tuple[str, str]]:
    """(file name, sha256) of every file the package holds besides its manifest, the GGUF first, then the projector,
    then any other name the manifest has a checksum for. PullError when a name is not one the server serves or a
    file has no usable checksum: a file that cannot be checked is not pulled."""
    names = [manifest.gguf, *([manifest.mmproj] if manifest.mmproj else [])]
    names += [name for name in sorted(manifest.sha256) if name not in names]
    found = []
    for name in names:
        if name == package_server.MANIFEST:
            continue
        if not package_server.served_name(name):
            raise PullError("bad_manifest", None, f"the manifest names a file the server does not serve: {name!r}")
        checksum = manifest.sha256.get(name, "")
        if not SHA256_RE.fullmatch(checksum):
            raise PullError("bad_manifest", None, f"the manifest has no usable sha256 for {name}")
        found.append((name, checksum.lower()))
    return found


# --- the pull ---------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Tally:
    """What a pull has moved so far."""

    bytes: int = 0             # bytes written to disk by this run
    transfer_s: float = 0.0    # seconds the requests took, pauses between tries left out
    parts: int = 0             # answers read to the end
    retries: int = 0           # tries that failed and were repeated


@dataclasses.dataclass
class FileResult:
    name: str
    size: int
    transferred: int = 0
    parts: int = 0
    resumed_from: int = 0
    outcome: str = "pulled"
    sha256: str = ""
    mtime_ns: int | None = None      # the file's mtime when it was hashed, for the .verified.json record

    def view(self) -> dict[str, Any]:
        return {"name": self.name, "bytes": self.size, "transferred": self.transferred, "parts": self.parts,
                "resumed_from": self.resumed_from, "outcome": self.outcome, "sha256": "ok"}


def _feed(digest: Any, path: Path, size: int) -> None:
    """Add the first `size` bytes of a file to a running digest."""
    left = size
    with path.open("rb") as handle:
        while left > 0:
            block = handle.read(min(CHUNK_BYTES, left))
            if not block:
                break
            digest.update(block)
            left -= len(block)


def _download(client: PackageClient, version: str, name: str, partial: Path, digest: Any, offset: int, *, part_bytes: int,
              clock: Callable[[], float], sleep: Callable[[float], None], progress: Callable[[str], None] | None,
              tally: Tally) -> tuple[int, int]:
    """Read the file's bytes from `offset` to its end into `partial`, one Range part at a time. Returns (total size,
    parts read). Raises PullError."""
    path = client.proxy_path(version, name)
    total: int | None = None
    parts = 0
    failures = 0
    with partial.open("ab") as out:
        while total is None or offset < total:
            began = offset
            error: Transient | None = None
            started = clock()
            try:
                headers = {"Range": f"bytes={offset}-{offset + part_bytes - 1}", "Accept": "*/*", "Accept-Encoding": "identity"}
                with client.opened(path, headers=headers) as response:
                    first, last, size = client.accept(response, start=offset, part_bytes=part_bytes, total=total)
                    if total is None:
                        total = size
                        free = shutil.disk_usage(partial.parent).free
                        if free < total - offset + DISK_MARGIN_BYTES:
                            raise PullError("no_space", None, f"{name} needs {total - offset} more bytes and the disk "
                                                              f"has {free} free")
                    want = last - first + 1            # 0 for an empty file, or nothing left: the body is a message, not data
                    got = 0
                    for block in response.iter_bytes() if want else ():
                        if got + len(block) > want:
                            raise PullError("bad_answer", response.status_code, "the answer holds more bytes than its Content-Range says")
                        out.write(block)
                        digest.update(block)
                        got += len(block)
                        offset += len(block)
                        tally.bytes += len(block)
                    if got < want:
                        raise Transient(None, f"the answer ended after {got} of {want} bytes")
            except Transient as exc:
                error = exc
            finally:
                tally.transfer_s += clock() - started
            out.flush()
            if error is None:
                os.fsync(out.fileno())
                failures = 0
                if offset > began:
                    parts += 1
                    tally.parts += 1
                if progress is not None and total:
                    progress(f"{name}: {offset // MIB} of {total // MIB} MiB")
                continue
            tally.retries += 1
            failures = 0 if offset > began else failures + 1
            if failures > client.retries:
                raise PullError("unreachable" if error.status is None else "proxy_error", error.status,
                                f"{name}: gave up after {failures} tries without a byte: {error.detail}") from None
            if offset == began:
                sleep(backoff_s(failures))
    if total is None or offset != total:
        raise PullError("bad_answer", None, f"{name}: ended at byte {offset} of {total}")
    return total, parts


def pull_file(client: PackageClient, version: str, name: str, checksum: str, folder: Path, *, part_bytes: int, force: bool,
              clock: Callable[[], float], sleep: Callable[[float], None], progress: Callable[[str], None] | None,
              tally: Tally) -> FileResult:
    """One file of a package into `folder`, verified against `checksum` (sha256 hex). A copy already there that
    matches is left alone; a NAME.part from an earlier run is continued from its end."""
    final = folder / name
    partial = folder / f"{name}.part"
    if final.is_file():
        st = final.stat()                  # taken before the hash: a file changed while it is hashed fails the record later
        if not force and sha256_file(final) == checksum:
            return FileResult(name, st.st_size, outcome="present", sha256=checksum, mtime_ns=st.st_mtime_ns)
        final.unlink()
    if force and partial.exists():
        partial.unlink()
    for _ in range(2):
        resumed = partial.stat().st_size if partial.is_file() else 0
        digest = hashlib.sha256()
        if resumed:
            _feed(digest, partial, resumed)
        before = tally.bytes
        try:
            size, parts = _download(client, version, name, partial, digest, resumed, part_bytes=part_bytes, clock=clock,
                                    sleep=sleep, progress=progress, tally=tally)
        except PullError:
            if partial.is_file() and partial.stat().st_size == 0:
                partial.unlink()               # nothing was kept: leave no empty file behind
            raise
        if digest.hexdigest() == checksum:
            os.replace(partial, final)
            return FileResult(name, size, tally.bytes - before, parts, resumed, sha256=checksum,
                              mtime_ns=final.stat().st_mtime_ns)
        partial.unlink()
        if resumed == 0:
            break                          # a clean download from byte 0 that does not match: the pod's copy is wrong
    raise PullError("checksum_mismatch", None, f"{name} does not match the sha256 in the manifest")


def write_verified(folder: Path, version: str, files: Sequence[FileResult]) -> Path:
    """Write <folder>/.verified.json: the files this pull hashed and found matching, with the size and mtime each had
    then. Written to a scratch name and renamed, so a reader sees the old record, none, or the whole new one."""
    record = {"written_by": VERIFIED_BY, "version": version,
              "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "files": [{"name": item.name, "size": item.size, "sha256": item.sha256, "mtime_ns": item.mtime_ns}
                        for item in files]}
    target = folder / VERIFIED_RECORD
    scratch = folder / f"{VERIFIED_RECORD}.tmp"
    scratch.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(scratch, target)
    return target


def _row(client: PackageClient, verb: str, outcome: str, *, metrics: Mapping[str, Any] | None = None,
         units: Mapping[str, str] | None = None, conditions: Mapping[str, Any] | None = None,
         run_id: str | None = None, note: str | None = None, **fields: Any) -> str | None:
    """Append a ledger row about the pod, with its node. Returns why it could not be written, else None."""
    extra: dict[str, Any] = {"node": client.node}
    if metrics and units:
        extra["units"] = {name: unit for name, unit in units.items() if name in metrics}
    if conditions:
        extra["conditions"] = dict(conditions)
    if note:
        extra["note"] = one_line(client.scrub(str(note)), NOTE_CHARS)
    refs: dict[str, Any] = {"account": client.account.letter}
    if run_id:
        refs["run_id"] = run_id
    fields = {name: value for name, value in fields.items() if value is not None}
    try:
        record.record(verb, outcome, dict(metrics) if metrics else None, refs, **extra, **fields)
    except Exception as exc:
        log.warning("no ledger row written for %s (%s): %s: %s", verb, outcome, type(exc).__name__, exc)
        return f"{type(exc).__name__}: {exc}"
    return None


def _new_run_id() -> str:
    return time.strftime("pkg-%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(2)


def _base(client: PackageClient, command: str, version: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    """The start of every answer: what was asked of which pod. refs is for the ledger row of the call."""
    answer: dict[str, Any] = {"command": command, "account": client.account.letter, "node": client.node,
                              "hub": client.host, "port": client.port}
    refs: dict[str, Any] = {"account": client.account.letter}
    if version is not None:
        answer["version"] = version
    if run_id is not None:
        answer["run_id"] = run_id
        refs["run_id"] = run_id
    answer["refs"] = refs
    return answer


def pull_package(client: PackageClient, version: str, dest: str | os.PathLike[str], *, part_bytes: int = DEFAULT_PART_BYTES,
                 force: bool = False, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep, progress: Callable[[str], None] | None = None,
                 run_id: str | None = None, write_row: bool = True) -> dict[str, Any]:
    """Pull a package version into dest/<version>/ and say how it went; see the module docstring.

    The answer holds ok, outcome, the bytes and seconds of the transfer, mb_s and a line per file. A pull that
    failed is an answer, not an exception; only a bad argument raises (ValueError) and the kill switch (Stopped).
    write_row=False leaves the `link pull` ledger row out, for a caller that writes its own."""
    part_bytes = check_part_bytes(part_bytes)
    check_name(version, "a version")
    _guard()
    run_id = run_id or _new_run_id()
    folder = Path(dest) / version
    answer = {**_base(client, "pull", version, run_id), "path": PATH_RANGE, "part_bytes": part_bytes, "dest": str(folder)}
    tally = Tally()
    files: list[FileResult] = []
    wall_began = clock()
    try:
        raw, manifest = fetch_manifest(client, version, sleep=sleep)
        wanted = package_files(manifest)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / VERIFIED_RECORD).unlink(missing_ok=True)    # vouches only for files this pull checks
        for name, checksum in wanted:
            files.append(pull_file(client, version, name, checksum, folder, part_bytes=part_bytes, force=force,
                                   clock=clock, sleep=sleep, progress=progress, tally=tally))
        write_verified(folder, version, files)
        target = folder / package_server.MANIFEST
        if not target.is_file() or target.read_bytes() != raw:
            scratch = folder / f"{package_server.MANIFEST}.tmp"
            scratch.write_bytes(raw)
            os.replace(scratch, target)
        outcome = "ok" if any(item.outcome == "pulled" for item in files) else "already_present"
        failure: PullError | None = None
    except PullError as err:
        outcome, failure = "error", err
    except OSError as exc:
        outcome, failure = "error", PullError("disk", None, f"{type(exc).__name__}: {exc}")
    wall_s = clock() - wall_began
    mb_s = round(tally.bytes / tally.transfer_s / 1e6, 3) if tally.bytes and tally.transfer_s > 0 else None
    answer.update({
        "ok": failure is None, "outcome": outcome, "bytes": tally.bytes, "seconds": round(tally.transfer_s, 3),
        "wall_seconds": round(wall_s, 3), "mb_s": mb_s, "parts": tally.parts, "retries": tally.retries,
        "package_bytes": sum(item.size for item in files), "files": [item.view() for item in files]})
    if failure is not None:
        answer.update({"kind": failure.kind, "error": client.scrub(failure.message)})
        if failure.status is not None:
            answer["status"] = failure.status
    if write_row:
        # a speed counts only for a package that was checked: the metric is defined as bytes over seconds, sha256 checked
        metrics = {PULL_METRIC: mb_s} if mb_s is not None and failure is None else None
        error = _row(client, PULL_ROW, outcome, metrics=metrics, units={PULL_METRIC: "MB/s"},
                     conditions={"path": PATH_RANGE, "bytes": tally.bytes, "part_bytes": part_bytes, "hub": client.host,
                                 "client": client_label()},
                     run_id=run_id, note=answer.get("error"), bytes=tally.bytes, seconds=round(tally.transfer_s, 3),
                     wall_seconds=round(wall_s, 3), part_bytes=part_bytes, parts=tally.parts, retries=tally.retries,
                     version=version, files=[item.name for item in files], kind=answer.get("kind"))
        if error:
            answer["ledger_error"] = error
    return answer


# --- the one whole-file GET -------------------------------------------------------------------------------------

def probe_whole_file(client: PackageClient, version: str, name: str | None = None, *, max_read_bytes: int = PROBE_READ_BYTES,
                     limit_bytes: int = PROXY_LIMIT_BYTES, clock: Callable[[], float] = time.perf_counter,
                     sleep: Callable[[float], None] = time.sleep, run_id: str | None = None,
                     write_row: bool = True) -> dict[str, Any]:
    """GET a whole package file (the GGUF unless a name is given) through the proxy once, with no Range and no
    retry, read at most max_read_bytes of it, and record what the proxy did. limit_bytes is the size above which
    the proxy is expected to refuse (100 MiB), which decides whether the file is big enough to say anything.
    See the module docstring."""
    check_name(version, "a version")
    if max_read_bytes < 1:
        raise ValueError(f"the probe reads at least one byte, not {max_read_bytes!r}")
    _guard()
    run_id = run_id or _new_run_id()
    answer = {**_base(client, "probe", version, run_id), "path": PATH_WHOLE, "cap_bytes": max_read_bytes}
    status: int | None = None
    length: int | None = None
    got = 0
    capped = False
    failure: PullError | None = None
    dropped: str | None = None
    seconds = 0.0
    try:
        if name is None:
            name = _manifest_gguf(client, version, sleep)
            if not package_server.served_name(name):
                raise PullError("bad_manifest", None, f"the manifest names a file the server does not serve: {name!r}")
        check_name(name, "a file name")
        answer["file"] = name
        answer["file_bytes"] = _size_on_pod(client, version, name, sleep)
        started = clock()
        try:
            with client.opened(client.proxy_path(version, name), timeout_s=PROBE_TIMEOUT_S,
                               headers={"Accept": "*/*", "Accept-Encoding": "identity"}) as response:
                status = response.status_code
                if 200 <= status < 300:
                    length = _int_header(response, "content-length")
                    for block in response.iter_bytes():
                        got += len(block)
                        if got >= max_read_bytes:
                            capped = True
                            break
                else:
                    problem = client.problem(response)
                    if isinstance(problem, PullError):
                        failure = problem
        except Transient as exc:
            dropped = exc.detail
        finally:
            seconds = clock() - started
    except PullError as err:
        failure = err
    if failure is None and status is None:
        failure = PullError("unreachable", None, dropped or "the proxy gave no answer")
    outcome, note = _probe_outcome(status, answer.get("file_bytes"), length, got, capped, dropped, failure, limit_bytes)
    ok = failure is None and outcome.startswith("whole_file_")
    if not ok and failure is None:
        failure = PullError("file_too_small", status, note)
    answer.update({"ok": ok, "outcome": outcome if ok else "error", "status": status, "content_length": length,
                   "bytes_read": got, "seconds": round(seconds, 3), "note": note})
    if failure is not None:
        answer.update({"kind": failure.kind, "error": client.scrub(failure.message)})
    if write_row:
        error = _row(client, PROBE_ROW, answer["outcome"],
                     conditions={"path": PATH_WHOLE, "hub": client.host, "client": client_label()},
                     run_id=run_id, note=note if ok else answer.get("error"), status=status, content_length=length,
                     bytes_read=got, seconds=round(seconds, 3), cap_bytes=max_read_bytes, file=answer.get("file"),
                     file_bytes=answer.get("file_bytes"), kind=answer.get("kind"))
        if error:
            answer["ledger_error"] = error
    return answer


def _manifest_gguf(client: PackageClient, version: str, sleep: Callable[[float], None]) -> str:
    return fetch_manifest(client, version, sleep=sleep)[1].gguf


def _size_on_pod(client: PackageClient, version: str, name: str, sleep: Callable[[float], None]) -> int | None:
    """The size of a file on the pod from the server's listing of the version; None when it cannot be read."""
    try:
        listing = json.loads(fetch_small(client, client.proxy_path(version), sleep=sleep, retries=0))
        return next(item["size"] for item in listing["files"] if item["name"] == name)
    except (PullError, ValueError, KeyError, TypeError, StopIteration):
        return None


def _probe_outcome(status: int | None, file_bytes: int | None, length: int | None, got: int, capped: bool,
                   dropped: str | None, failure: PullError | None, limit_bytes: int) -> tuple[str, str]:
    """(outcome, one line about it) for the whole-file GET. file_bytes is the file's size on the pod and length the
    Content-Length of a successful answer."""
    limit = f"{limit_bytes // MIB} MiB" if limit_bytes % MIB == 0 else f"{limit_bytes} bytes"
    size = f"{file_bytes} bytes" if file_bytes is not None else "an unknown size"
    small = file_bytes is not None and file_bytes <= limit_bytes
    why = (f"the file is {size}, not over the {limit} limit, so the refusal is not that limit" if small
           else f"as predicted for a response over {limit}: pull in Range parts")
    if failure is not None:
        return "error", failure.message
    if status is not None and not 200 <= status < 300:
        return "whole_file_refused", f"the proxy answered {status} to the whole file ({size} on the pod); {why}"
    if capped:
        return "whole_file_served", (f"{got} bytes of one response came through the proxy (the file is {size}): the "
                                     f"{limit} limit did not hold here")
    if dropped is not None:
        return "whole_file_refused", f"the proxy cut the whole file off after {got} of {length} bytes ({dropped}); {why}"
    if got <= limit_bytes:
        return "inconclusive", f"the file is {got} bytes, not larger than {limit}, so it says nothing about the limit"
    return "whole_file_served", f"the whole file ({got} bytes) came through the proxy in one response"


# --- starting and stopping the server on the pod ----------------------------------------------------------------

START_SCRIPT = """#!/bin/bash
# Starts the package server (lab/package_server.py) on this pod. Written by `python -m lab.package_pull serve-start`;
# the server's source is the heredoc below, so nothing has to be uploaded first. Safe to run again: it restarts it.
set -u
export PATH="$HOME/.local/bin:$PATH"
PORT=@PORT@
ROOT="@ROOT@"
DIR="$HOME/lab"
LOG="$HOME/package_server.log"
mkdir -p "$DIR"
cat > "$DIR/package_server.py" <<'@SENTINEL@'
@SOURCE@
@SENTINEL@
pkill -u "$(id -u)" -f 'package_server.py --root' >/dev/null 2>&1 || true
sleep 1
LAUNCH=""
command -v setsid >/dev/null 2>&1 && LAUNCH="setsid"
nohup $LAUNCH python3 "$DIR/package_server.py" --root "$ROOT" --port "$PORT" --log "$LOG" >/dev/null 2>&1 </dev/null &
for i in 1 2 3 4 5 6 7 8 9 10; do
  if python3 -c "import sys, urllib.request; sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:$PORT/', timeout=3).read().decode())" 2>/dev/null; then
    echo "package server up on port $PORT, serving $ROOT"
    exit 0
  fi
  sleep 1
done
echo "package server did not answer on port $PORT; the end of $LOG follows"
tail -n 20 "$LOG" 2>&1
exit 1
"""

STOP_SCRIPT = """#!/bin/bash
# Stops the package server on this pod. Written by `python -m lab.package_pull serve-stop`.
set -u
if pkill -u "$(id -u)" -f 'package_server.py --root'; then echo "package server stopped"; else echo "no package server was running"; fi
exit 0
"""


def start_script(port: int, root: str = DEFAULT_SERVER_ROOT) -> str:
    """The shell script serve-start runs on the pod."""
    if not ROOT_RE.fullmatch(root):
        raise ValueError(f"the server root is a path under ~, $HOME or /, with letters, digits, dot, dash, underscore and slash, not {root!r}")
    source = Path(package_server.__file__).read_text(encoding="utf-8")
    if SENTINEL in source:
        raise ValueError("package_server.py contains the line that ends its own heredoc")
    text = START_SCRIPT.replace("@PORT@", str(int(port))).replace("@ROOT@", root).replace("@SENTINEL@", SENTINEL)
    return text.replace("@SOURCE@", source.rstrip("\n"))


def _run_on_pod(controller: Controller, client: PackageClient, script: str, *, timeout_s: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="package-script-") as folder:
        path = Path(folder) / "package_server.sh"
        path.write_text(script, encoding="utf-8", newline="\n")
        return controller.run_script(client.account.letter, path, timeout_s=timeout_s)


def _on_pod_view(done: Mapping[str, Any]) -> dict[str, Any]:
    keep = ("outcome", "exit_code", "output_tail", "kind", "error")
    return {key: done[key] for key in keep if key in done}


def serve_start(controller: Controller, client: PackageClient, *, root: str = DEFAULT_SERVER_ROOT,
                timeout_s: float = SCRIPT_TIMEOUT_S, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Start (or restart) the package server on the account's pod and ask its listing through the proxy."""
    _guard()
    answer = {**_base(client, "serve-start"), "root": root}
    done = _run_on_pod(controller, client, start_script(client.port, root), timeout_s=timeout_s)
    answer["on_pod"] = _on_pod_view(done)
    if not done.get("ok"):
        answer.update({"ok": False, "outcome": "error", "kind": done.get("kind") or done.get("outcome", "script_failed"),
                       "error": done.get("error") or f"the start script ended with {done.get('outcome')}"})
        return answer
    try:
        listing = json.loads(fetch_small(client, client.proxy_path(), sleep=sleep))
    except PullError as err:
        hint = {"missing": "jupyter-server-proxy is not loaded on this pod: run the proxy step of the pod scripts and relaunch once",
                "server_down": "the server started but the proxy cannot reach it"}.get(err.kind, "")
        answer.update({"ok": False, "outcome": "error", "kind": err.kind, "error": client.scrub(err.message), "hint": hint})
        return answer
    except ValueError:
        answer.update({"ok": False, "outcome": "error", "kind": "bad_answer", "error": "the listing is not JSON"})
        return answer
    answer.update({"ok": True, "outcome": "ok", "versions": listing.get("versions", []) if isinstance(listing, dict) else []})
    return answer


def serve_stop(controller: Controller, client: PackageClient, *, timeout_s: float = SCRIPT_TIMEOUT_S) -> dict[str, Any]:
    """Stop the package server on the account's pod."""
    _guard()
    answer = _base(client, "serve-stop")
    done = _run_on_pod(controller, client, STOP_SCRIPT, timeout_s=timeout_s)
    answer["on_pod"] = _on_pod_view(done)
    if not done.get("ok"):
        answer.update({"ok": False, "outcome": "error", "kind": done.get("kind") or done.get("outcome", "script_failed"),
                       "error": done.get("error") or f"the stop script ended with {done.get('outcome')}"})
    else:
        answer.update({"ok": True, "outcome": "ok"})
    return answer


# --- the curl lines ---------------------------------------------------------------------------------------------

def curl_lines(client: PackageClient, version: str, name: str | None = None) -> list[str]:
    """Curl commands that check the path from outside the pod, with the hub's address spelled out and the account and
    token left as the shell variables AUP_HUB_USER_<letter> and AUP_HUB_TOKEN_<letter> (load the workspace .env first)."""
    letter = client.account.letter
    base = f"{client.base_url}/user/${{AUP_HUB_USER_{letter}}}/proxy/{client.port}"
    auth = f'-H "Authorization: token ${{AUP_HUB_TOKEN_{letter}}}"'
    file_url = f"{base}/{version}/{name or '<file>.gguf'}"
    return [
        f'curl -sS {auth} "{base}/"',
        f'curl -sS {auth} "{base}/{version}/{package_server.MANIFEST}"',
        f'curl -sS {auth} -D - -o /dev/null -r 0-1048575 "{file_url}"',
        f'curl -sS {auth} -D - -o /dev/null -r -1048576 "{file_url}"',
        f'curl -sS {auth} -D - -o /dev/null -m 30 "{file_url}"',
    ]


# --- the command line -------------------------------------------------------------------------------------------

def _account_letter(text: str) -> str:
    letter = text.strip().upper()
    if not re.fullmatch(r"[A-Z]", letter):
        raise argparse.ArgumentTypeError(f"{text!r} is not an account: use its letter, such as B or C")
    return letter


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a port number") from None
    if not 1024 <= value <= 65535:
        raise argparse.ArgumentTypeError("a port from 1024 to 65535")
    return value


def _name(text: str) -> str:
    try:
        return check_name(text, "the name")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _mib(low: float, high: float, reason: str = "") -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"from {low:g} to {high:g} MiB{reason}")
        return int(value * MIB)
    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab.package_pull", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def common(sub: argparse.ArgumentParser, *, version: bool) -> None:
        sub.add_argument("--account", required=True, type=_account_letter, help="the account's letter, such as B or C")
        sub.add_argument("--port", type=_port, default=DEFAULT_PORT, help=f"the package server's port (default {DEFAULT_PORT})")
        if version:
            sub.add_argument("--version", required=True, type=_name, help="the package version: its folder on the pod")

    pull = commands.add_parser("pull", help="pull a package in Range parts and check its sha256")
    common(pull, version=True)
    pull.add_argument("--dest", required=True, help="the folder that will hold <version>/")
    pull.add_argument("--part-mib", dest="part_bytes", metavar="MIB", type=_mib(MIN_PART_BYTES / MIB, MAX_PART_BYTES / MIB,
                                f": the hub proxy refuses a response body over {PROXY_LIMIT_BYTES // MIB} MiB"),
                      default=DEFAULT_PART_BYTES, help=f"size of a Range part in MiB (default {DEFAULT_PART_BYTES // MIB}, "
                                                       f"at most {MAX_PART_BYTES // MIB})")
    pull.add_argument("--force", action="store_true", help="pull again even when the files are already there and correct")
    pull.add_argument("--quiet", action="store_true", help="no progress lines on stderr")

    probe = commands.add_parser("probe", help="GET the whole GGUF through the proxy once and record the status")
    common(probe, version=True)
    probe.add_argument("--file", type=_name, help="the file to GET (default: the manifest's GGUF)")
    probe.add_argument("--max-read-mib", dest="max_read_bytes", metavar="MIB", type=_mib(0.001, 4096), default=PROBE_READ_BYTES,
                       help=f"read at most this much of the answer (default {PROBE_READ_BYTES // MIB})")

    start = commands.add_parser("serve-start", help="start the package server on the pod")
    common(start, version=False)
    start.add_argument("--root", default=DEFAULT_SERVER_ROOT, help=f"the packages folder on the pod (default {DEFAULT_SERVER_ROOT})")

    stop = commands.add_parser("serve-stop", help="stop the package server on the pod")
    common(stop, version=False)

    curl = commands.add_parser("curl", help="print the curl lines that check the path from outside the pod")
    common(curl, version=True)
    curl.add_argument("--file", type=_name, help="the file to ask for (default: a placeholder)")
    return parser


class Context:
    """What a command needs: the account, the way to reach the hub, and the clocks."""

    def __init__(self, accounts: Mapping[str, Account], *, transport: Any, sleep: Callable[[float], None],
                 clock: Callable[[], float] | None, environ: Mapping[str, str]) -> None:
        self.accounts = dict(accounts)
        self.transport = transport
        self.sleep = sleep
        self.clock = clock
        self.environ = environ
        self._open: list[Any] = []

    def files(self, letter: str, port: int) -> PackageClient:
        client = PackageClient(self.accounts[letter], port, transport=self.transport, sleep=self.sleep, environ=self.environ)
        self._open.append(client)
        return client

    def controller(self) -> Controller:
        controller = Controller(self.accounts, transport=self.transport, sleep=self.sleep, environ=self.environ,
                                **({"clock": self.clock} if self.clock is not None else {}))
        self._open.append(controller)
        return controller

    def close(self) -> None:
        for item in self._open:
            item.close()
        self._open.clear()


def _progress(args: argparse.Namespace) -> Callable[[str], None] | None:
    if getattr(args, "quiet", False):
        return None
    return lambda line: print(f"package_pull: {line}", file=sys.stderr, flush=True)


@recorded("package pull")
def cmd_pull(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    _guard()
    client = ctx.files(args.account, args.port)
    return pull_package(client, args.version, args.dest, part_bytes=args.part_bytes, force=args.force,
                        clock=ctx.clock or time.perf_counter, sleep=ctx.sleep, progress=_progress(args))


@recorded("package probe")
def cmd_probe(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    _guard()
    client = ctx.files(args.account, args.port)
    return probe_whole_file(client, args.version, args.file, max_read_bytes=args.max_read_bytes,
                            clock=ctx.clock or time.perf_counter, sleep=ctx.sleep)


@recorded("package serve-start")
def cmd_serve_start(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    _guard()
    client = ctx.files(args.account, args.port)
    return serve_start(ctx.controller(), client, root=args.root, sleep=ctx.sleep)


@recorded("package serve-stop")
def cmd_serve_stop(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    _guard()
    client = ctx.files(args.account, args.port)
    return serve_stop(ctx.controller(), client)


def cmd_curl(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    client = ctx.files(args.account, args.port)
    return {"ok": True, "command": "curl", "outcome": "ok", "account": args.account, "commands": curl_lines(client, args.version, args.file),
            "note": f"load the workspace .env first so AUP_HUB_USER_{args.account} and AUP_HUB_TOKEN_{args.account} are set"}


HANDLERS: dict[str, Callable[[Context, argparse.Namespace], dict[str, Any]]] = {
    "pull": cmd_pull, "probe": cmd_probe, "serve-start": cmd_serve_start, "serve-stop": cmd_serve_stop, "curl": cmd_curl,
}


def run(argv: Sequence[str], *, environ: Mapping[str, str] | None = None, env_files: Sequence[Path] | None = None,
        transport: Any = None, sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None) -> tuple[dict[str, Any], int]:
    """One command: (the answer, the exit code). environ and env_files say where the keys are looked up, and
    transport, sleep and clock replace the network and the time, for rehearsals."""
    args = build_parser().parse_args(list(argv))
    environ = os.environ if environ is None else environ
    try:
        accounts = load_accounts([args.account], environ=environ, env_files=env_files, require_role=True)
    except PodRoleError as exc:
        return ({"ok": False, "command": args.command, "outcome": "no_pod_role", "error": str(exc)}, 2)
    except MissingCredentials as exc:
        return ({"ok": False, "command": args.command, "outcome": "no_credentials", "missing": exc.missing,
                 "error": "keys missing from the environment and the .env files",
                 "hint": "set them in the workspace .env (see config/.env.example); the values are never printed"}, 2)
    ctx = Context(accounts, transport=transport, sleep=sleep or time.sleep, clock=clock, environ=environ)
    try:
        answer = HANDLERS[args.command](ctx, args)
    except kill_switch.Stopped as exc:
        answer = {"ok": False, "command": args.command, "outcome": "stopped", "reason": exc.reason,
                  "file": str(kill_switch.stop_file())}
    finally:
        ctx.close()
    answer = {key: value for key, value in answer.items() if key != "refs"}
    return answer, 0 if answer.get("ok", True) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Pull a model package from a pod through the hub proxy: pull, probe, serve-start, serve-stop, curl."""
    try:
        answer, code = run(list(sys.argv[1:] if argv is None else argv))
    except KeyboardInterrupt:
        answer, code = {"ok": False, "outcome": "interrupted"}, 1
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        answer, code = {"ok": False, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}, 1
    print(json.dumps(answer))
    return code


if __name__ == "__main__":
    sys.exit(main())
