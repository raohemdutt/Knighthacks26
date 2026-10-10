"""transfer: move files between the box and a pod through the hub, the two ways the plumbing research settled on.

    from lab import hub, proxy, transfer

    account = hub.load_accounts(["D"])["D"]
    with proxy.PodLink.for_account(account) as link:
        files = transfer.PodFiles(link)
        pushed = transfer.push_dataset(files, Path("data/exports/ds-0007"))             # lands in datasets/ds-0007
        got = transfer.pull_package(files, "packages/atlas-v3", Path("data/models"))    # data/models/atlas-v3/
        got.version          # then: python jetson/atlas_swap.py swap <version>

    python -m lab.transfer push-dataset --account D data/exports/ds-0007 [--to datasets] [--chunk-mib 8]
    python -m lab.transfer push-file --account D LOCAL REMOTE [--chunk-mib 8]
    python -m lab.transfer pull-file --account D REMOTE LOCAL [--via files|proxy --port N] [--part-mib 64] [--sha256 HEX]
    python -m lab.transfer pull-package --account D REMOTE_DIR [--models-dir data/models] [--via files|proxy --port N]

Facts are from lab/refine/research/plumbing/transfers.md and proxy-path.md; none has been tried on a live pod.

Push, through the Jupyter contents API: PUT /user/<account>/api/contents/<path> with the file as base64. A file up to
one chunk goes in one PUT. A bigger one goes in chunks (`chunk` 1 creates the file, 2 and up append, -1 appends the
last), each a third bigger on the wire. Cloudflare refuses a body over 100 MB, so a chunk is at most 64 MiB of the file
(about 85 MB sent) and 8 MiB unless told otherwise; with 140 ms to every round trip a bigger chunk is a faster push, and
which size is best is what the queued push experiment measures. The contents API will not make folders, so they are
made first. A chunk is an append, so it is never sent twice blindly: when its answer is lost (a gateway error, a dropped
link) the file's size is read back and the chunk is sent again only if the size says it did not land. A chunked push
first removes a file already at the path, so that the size says what landed. When the last byte is in, the size on the
pod is checked against the file.

Where a push lands. On the DUB pods the Jupyter root was /opt/workspace, which a relaunch wipes: what is pushed there
must be moved under ~/ on the pod before the pod stops. The SJC pods run another image and have not been looked at, so
push_dataset reports `persistent`, whether the root listing shows a `home` folder (a newer image's does), and the caller
knows which case it is in. Nothing here moves a file on the pod.

Datasets. push_dataset sends one approved export, the folder `data/exports/<dataset_id>/`. It refuses a folder whose
manifest.json is not a DatasetManifest or carries no `action_id`: that field is the approval that allowed the export,
and a push is an outside effect. Symbolic links are not followed or sent. manifest.json goes last, so a half-sent dataset
has none and a reader on the pod can take its presence for "complete".

Pull, with HTTP byte ranges: the Jupyter server's own /files/<path> route (nothing to run on the pod; it serves only
what is under the Jupyter root and puts no limit on a response), or /proxy/<port>/<path> for a file server on a pod
port such as the package server of R10 (the proxy refuses one response over 100 MiB, so no part is larger than 96 MiB).
A file comes in parts of 64 MiB, each `Range: bytes=a-b`; a part that breaks off is continued from the byte it reached;
the bytes go into `<name>.part`, which a later run continues; the sha256 is checked; and the file appears under its own
name only when it is whole and matches. A server that ignores Range answers 200 with the whole file, which is accepted
only as the first and only request.

Packages. pull_package fetches `<remote_dir>/manifest.json` (a ModelManifest), refuses a file name that is not a plain
name and a file the manifest has no checksum for, pulls each file into `<models_dir>/.<version>.part/`, writes the
manifest last, renames the folder to `<models_dir>/<version>` and has runtime.modelstore check it as the switch does. It
does not switch to it, and it will not replace the version that is serving.

Time. Durations come from the link's clock and pauses from its sleep, both handed in by tests.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import httpx

if __package__ in (None, ""):          # started as `python lab/transfer.py`: put the repo root on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.hub import MissingCredentials, load_accounts
from lab.proxy import NotFound, PodLink, ProxyError

__all__ = ["DEFAULT_CHUNK_BYTES", "MAX_CHUNK_BYTES", "DEFAULT_PART_BYTES", "MAX_PART_BYTES", "TransferError",
           "UnsafePath", "NotApproved", "ChecksumMismatch", "RangeUnsupported", "TransferStats", "Layout", "PodFiles",
           "DatasetPush", "PackageInstalled", "clean_remote", "clean_name", "sha256_file", "push_file", "push_dataset",
           "pull_file", "pull_package", "main", "build_parser"]

MIB = 1 << 20
DEFAULT_CHUNK_BYTES = 8 * MIB
MAX_CHUNK_BYTES = 64 * MIB            # base64 makes it about 85 MB on the wire, under Cloudflare's 100 MB
DEFAULT_PART_BYTES = 64 * MIB
MAX_PART_BYTES = 96 * MIB             # the proxy refuses one response over 100 MiB
RETRY_PAUSES_S = (1.0, 2.0, 4.0)
MANIFEST_LIMIT_BYTES = MIB
READ_BLOCK = MIB
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,199}")
RESERVED_VERSIONS = frozenset({"current", "previous", "manifest.json"})   # what data/models keeps beside the versions


class TransferError(Exception):
    """A push or a pull that did not complete. The message names the file and what was found, never the token."""


class UnsafePath(TransferError, ValueError):
    """A path or name that could reach outside the folder it is meant for."""


class NotApproved(TransferError):
    """A dataset folder that is not an approved export, so it is not pushed."""


class ChecksumMismatch(TransferError):
    """A pulled file whose sha256 is not the one expected; the damaged copy is removed."""


class RangeUnsupported(TransferError):
    """A server that answered a byte range with the whole file."""


def clean_remote(path: str) -> str:
    """A path on the pod, relative to its Jupyter root, with forward slashes and no empty, `.` or `..` part."""
    if not isinstance(path, str) or not path:
        raise UnsafePath("a remote path is needed")
    if "\\" in path or any(ord(char) < 32 for char in path):
        raise UnsafePath(f"not a plain remote path: {path!r}")
    parts = path.split("/")
    if path.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise UnsafePath(f"not a plain remote path: {path!r}")
    return path


def clean_name(name: str) -> str:
    """One plain file or folder name, the kind a package manifest and a dataset id hold."""
    if not isinstance(name, str) or NAME_PATTERN.fullmatch(name) is None or name.endswith("."):
        raise UnsafePath(f"not a plain name: {name!r}")
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(READ_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _quote(path: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))


@dataclass
class TransferStats:
    """What a transfer moved and how fast: bytes over the wire this run, the seconds it took, the requests it made
    (retries are the ones that repeated an earlier one) and, when it was checked, whether the checksum or the size
    matched. mb_s is in megabytes (10**6 bytes) a second, the unit of the link.push_mb_s and link.pull_mb_s metrics."""

    direction: str
    route: str
    files: int = 0
    bytes: int = 0
    seconds: float = 0.0
    requests: int = 0
    retries: int = 0
    part_bytes: int = 0
    skipped: int = 0
    verified: bool | None = None

    @property
    def mb_s(self) -> float | None:
        return round(self.bytes / 1e6 / self.seconds, 3) if self.seconds > 0 and self.bytes else None

    def metrics(self) -> dict[str, float]:
        name = "link.push_mb_s" if self.direction == "push" else "link.pull_mb_s"
        return {name: self.mb_s} if self.mb_s is not None else {}

    def add(self, other: "TransferStats") -> None:
        self.files += other.files
        self.bytes += other.bytes
        self.seconds += other.seconds
        self.requests += other.requests
        self.retries += other.retries
        self.skipped += other.skipped
        self.part_bytes = max(self.part_bytes, other.part_bytes)
        if other.verified is not None:
            self.verified = other.verified if self.verified is None else (self.verified and other.verified)

    def as_dict(self) -> dict[str, Any]:
        return {"direction": self.direction, "route": self.route, "files": self.files, "bytes": self.bytes,
                "seconds": round(self.seconds, 3), "mb_s": self.mb_s, "requests": self.requests,
                "retries": self.retries, "part_bytes": self.part_bytes, "skipped": self.skipped,
                "verified": self.verified}


@dataclass(frozen=True)
class Layout:
    """What the pod's Jupyter root shows: has_home is true when it lists a `home` folder, which means a pushed file can
    be put in the persistent home directly; false means the root is the ephemeral /opt/workspace."""

    has_home: bool
    names: tuple[str, ...]


class PodFiles:
    """The files of one pod as its Jupyter server shows them, reached through a PodLink."""

    def __init__(self, link: PodLink, *, retries: int = len(RETRY_PAUSES_S)) -> None:
        self.link = link
        self.retries = max(0, min(retries, len(RETRY_PAUSES_S)))
        self._dirs: set[str] = {""}

    @property
    def clock(self) -> Callable[[], float]:
        return self.link.clock

    @property
    def sleep(self) -> Callable[[float], None]:
        return self.link.sleep

    # addresses

    @staticmethod
    def contents_rel(path: str) -> str:
        return "/api/contents/" + (_quote(path) if path else "")

    @staticmethod
    def read_rel(path: str, via: str, port: int | None) -> str:
        """The path below the account that serves `path` for reading: /files/<path> or /proxy/<port>/<path>."""
        if via == "files":
            return "/files/" + _quote(path)
        if via == "proxy":
            if port is None:
                raise ValueError("--via proxy needs a port")
            return f"/proxy/{int(port)}/" + _quote(path)
        raise ValueError(f"unknown route {via!r}: files or proxy")

    # the contents API

    def stat(self, path: str) -> Mapping[str, Any] | None:
        """The contents API's model of a file or folder without its content, or None when there is none."""
        try:
            reply = self.link.request("GET", self.contents_rel(path), params={"content": "0"})
        except NotFound:
            return None
        try:
            model = reply.json()
        except ValueError:
            raise TransferError(f"the pod's answer about {path} was not JSON") from None
        return model if isinstance(model, Mapping) else None

    def size_of(self, path: str) -> int | None:
        model = self.stat(path)
        if model is None or model.get("type") != "file":
            return None
        size = model.get("size")
        return size if isinstance(size, int) else None

    def layout(self) -> Layout:
        reply = self.link.request("GET", self.contents_rel(""))
        try:
            entries = reply.json().get("content") or []
        except (ValueError, AttributeError):
            raise TransferError("the pod's root listing was not JSON") from None
        names = tuple(sorted(str(item.get("name")) for item in entries if isinstance(item, Mapping)))
        has_home = any(isinstance(item, Mapping) and item.get("name") == "home" and item.get("type") == "directory"
                       for item in entries)
        return Layout(has_home, names)

    def ensure_dir(self, path: str) -> None:
        """The folder and every folder above it exist on the pod (the contents API will not make them on its own)."""
        walked = ""
        for part in [p for p in path.split("/") if p]:
            walked = f"{walked}/{part}" if walked else part
            if walked in self._dirs:
                continue
            model = self.stat(walked)
            if model is None:
                self.link.request("PUT", self.contents_rel(walked), json={"type": "directory"}, retry=True)
            elif model.get("type") != "directory":
                raise TransferError(f"{walked} is a file on the pod, so {path} cannot be a folder")
            self._dirs.add(walked)

    def delete(self, path: str) -> None:
        self.link.request("DELETE", self.contents_rel(path), also_ok=(404,))

    def put_whole(self, path: str, data: bytes, stats: TransferStats) -> None:
        """One file in one PUT. It replaces whatever is there, so it is safe to repeat."""
        stats.requests += 1
        self.link.request("PUT", self.contents_rel(path), json={
            "type": "file", "format": "base64", "content": base64.b64encode(data).decode("ascii")}, retry=True)

    def put_chunk(self, path: str, data: bytes, marker: int, *, before: int, stats: TransferStats) -> None:
        """One chunk of a chunked upload: marker 1 creates the file, 2 and up append, -1 appends the last. `before` is how
        many bytes the file should hold already. A failure that may have lost only the answer is settled by reading the
        size back: if the chunk landed the push goes on, if nothing did it is sent again, and anything else is an error."""
        body = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode("ascii"), "chunk": marker}
        for attempt in range(self.retries + 1):
            stats.requests += 1
            try:
                self.link.request("PUT", self.contents_rel(path), json=body, retry=False)
                return
            except ProxyError as exc:
                if not exc.retryable or attempt == self.retries:
                    raise
                landed = self.size_of(path)
                if landed == before + len(data):
                    return
                if landed not in (None, before) or (marker != 1 and landed is None):
                    raise TransferError(f"{path} holds {landed} bytes after a failed chunk; it should hold {before} or "
                                        f"{before + len(data)}, so the push was stopped") from exc
                stats.retries += 1
                self.sleep(RETRY_PAUSES_S[attempt])


def push_file(files: PodFiles, local: Path | str, remote: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
              progress: Callable[[int, int], None] | None = None) -> TransferStats:
    """Send one file to `remote` (relative to the pod's Jupyter root), replacing what is there, and check its size on
    the pod. Raises TransferError when the size does not match, and the ProxyError of any call that fails."""
    local = Path(local)
    remote = clean_remote(remote)
    if not 1 <= chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError(f"chunk_bytes is 1 to {MAX_CHUNK_BYTES}")
    size = local.stat().st_size
    stats = TransferStats("push", "contents", files=1, bytes=size, part_bytes=chunk_bytes)
    started = files.clock()
    files.ensure_dir(remote.rsplit("/", 1)[0] if "/" in remote else "")
    if size <= chunk_bytes:
        files.put_whole(remote, local.read_bytes(), stats)
        if progress is not None:
            progress(size, size)
    else:
        if files.stat(remote) is not None:
            files.delete(remote)
        sent = number = 0
        with local.open("rb") as handle:
            while sent < size:
                data = handle.read(min(chunk_bytes, size - sent))
                if not data:
                    raise TransferError(f"{local.name} got shorter while it was being sent")
                number += 1
                last = sent + len(data) >= size
                files.put_chunk(remote, data, -1 if last else number, before=sent, stats=stats)
                sent += len(data)
                if progress is not None:
                    progress(sent, size)
    got = files.size_of(remote)
    stats.verified = got == size
    stats.seconds = files.clock() - started
    if got != size:
        raise TransferError(f"{remote} is {got} bytes on the pod and {size} here")
    return stats


@dataclass
class DatasetPush:
    dataset_id: str
    remote_dir: str
    files: int
    bytes: int
    persistent: bool
    skipped: list[str]
    stats: TransferStats

    def as_dict(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "remote_dir": self.remote_dir, "files": self.files, "bytes": self.bytes,
                "persistent": self.persistent, "skipped": self.skipped, "stats": self.stats.as_dict()}


def _walk(root: Path) -> tuple[list[str], list[str]]:
    """The relative paths (forward slashes) of the regular files below root, and of the links that were left out."""
    found: list[str] = []
    skipped: list[str] = []
    for folder, names, entries in os.walk(root, followlinks=False):
        here = Path(folder)
        for name in list(names):
            if (here / name).is_symlink():
                skipped.append((here / name).relative_to(root).as_posix())
                names.remove(name)
        for name in entries:
            path = here / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                skipped.append(rel)
            else:
                found.append(rel)
    found.sort(key=lambda rel: (rel == "manifest.json", rel))
    return found, sorted(skipped)


def push_dataset(files: PodFiles, local_dir: Path | str, *, remote_root: str = "datasets",
                 chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> DatasetPush:
    """Send one approved export folder to `<remote_root>/<dataset_id>` on the pod. Raises NotApproved for a folder that
    is not an export or carries no approval id."""
    from contracts.learning import DatasetManifest

    root = Path(local_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise NotApproved(f"{root.name} has no manifest.json, so it is not an export")
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise NotApproved(f"{root.name}/manifest.json is not a dataset manifest: {str(exc).splitlines()[0]}") from None
    if not manifest.action_id.strip():
        raise NotApproved(f"{root.name} carries no approval id (action_id), so it was not approved for export")
    dataset_id = clean_name(manifest.dataset_id)
    top = remote_root.strip("/")
    remote_dir = clean_remote(f"{top}/{dataset_id}" if top else dataset_id)
    names, skipped = _walk(root)
    total = TransferStats("push", "contents", part_bytes=chunk_bytes)
    for rel in names:
        total.add(push_file(files, root / rel, f"{remote_dir}/{rel}", chunk_bytes=chunk_bytes))
    layout = files.layout()
    return DatasetPush(dataset_id, remote_dir, len(names), total.bytes, layout.has_home, skipped, total)


# --- pulling ------------------------------------------------------------------------------------

@dataclass
class _Cursor:
    offset: int
    total: int | None


def _content_range(header: str) -> tuple[int, int, int | None] | None:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", header.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), None if match.group(3) == "*" else int(match.group(3))


def _fetch_part(link: PodLink, rel: str, cursor: _Cursor, end: int, out: Any, stats: TransferStats,
                progress: Callable[[int, int], None] | None) -> None:
    """One ranged GET written to `out`, moving the cursor by every byte that lands."""
    with link.stream("GET", rel, headers={"Range": f"bytes={cursor.offset}-{end}"}, accept="*/*",
                     also_ok=(416,)) as reply:
        stats.requests += 1
        if reply.status == 416:
            match = re.fullmatch(r"bytes \*/(\d+)", reply.headers.get("content-range", "").strip())
            known = int(match.group(1)) if match else None
            if known is not None and (cursor.offset == known):
                cursor.total = known
                return
            raise TransferError(f"the pod cannot serve bytes {cursor.offset}-{end} of {rel.rsplit('/', 1)[-1]}"
                                f"{'' if known is None else f' ({known} bytes)'}")
        whole = reply.status == 200
        if whole:
            if cursor.offset != 0:
                raise RangeUnsupported(f"the server ignored the byte range for {rel.rsplit('/', 1)[-1]} and sent the "
                                       "whole file; remove the .part file and pull again")
            expected = None
        else:
            parsed = _content_range(reply.headers.get("content-range", ""))
            if parsed is None or parsed[0] != cursor.offset:
                raise TransferError(f"the pod answered the wrong range for {rel.rsplit('/', 1)[-1]}")
            expected = parsed[1] - parsed[0] + 1
            if parsed[2] is not None:
                cursor.total = parsed[2]
        written = 0
        for block in reply.iter_bytes():
            out.write(block)
            cursor.offset += len(block)
            written += len(block)
            if progress is not None and cursor.total is not None:
                progress(cursor.offset, cursor.total)
        if whole:
            cursor.total = cursor.offset
        elif written != expected:
            raise reply.interrupted(f"{written} of {expected} bytes arrived")


def pull_file(files: PodFiles, remote: str, local: Path | str, *, via: str = "files", port: int | None = None,
              part_bytes: int = DEFAULT_PART_BYTES, sha256: str | None = None,
              progress: Callable[[int, int], None] | None = None) -> TransferStats:
    """Fetch `remote` into `local` in byte-range parts. With sha256 the copy is checked (and a file already there that
    matches is left alone). The file appears under its name only when it is whole. Raises ChecksumMismatch, and the
    ProxyError of a call that fails; a part that breaks off is continued from the byte it reached."""
    remote = clean_remote(remote)
    local = Path(local)
    rel = files.read_rel(remote, via, port)
    if not 1 <= part_bytes <= MAX_PART_BYTES:
        raise ValueError(f"part_bytes is 1 to {MAX_PART_BYTES}")
    want = sha256.lower() if sha256 else None
    stats = TransferStats("pull", "files" if via == "files" else f"proxy:{port}", files=1, part_bytes=part_bytes)
    started = files.clock()
    if want and local.is_file() and sha256_file(local) == want:
        stats.skipped, stats.verified, stats.seconds = 1, True, files.clock() - started
        return stats
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + ".part")
    for attempt in (1, 2):
        before = part.stat().st_size if part.is_file() else 0
        cursor = _Cursor(before, None)
        failures = 0
        with part.open("ab") as out:
            while cursor.total is None or cursor.offset < cursor.total:
                end = cursor.offset + part_bytes - 1
                if cursor.total is not None:
                    end = min(end, cursor.total - 1)
                try:
                    _fetch_part(files.link, rel, cursor, end, out, stats, progress)
                except ProxyError as exc:
                    out.flush()
                    failures += 1
                    if not exc.retryable or failures > files.retries:
                        raise
                    stats.retries += 1
                    files.sleep(RETRY_PAUSES_S[failures - 1])
        stats.bytes += cursor.offset - before
        if want is None or sha256_file(part) == want:
            break
        part.unlink()      # a continued copy that does not match may have a bad start: once more from the first byte
        if attempt == 2 or before == 0:
            raise ChecksumMismatch(f"{local.name} does not match its sha256 (a damaged copy was removed)")
    os.replace(part, local)
    stats.verified = True if want else None
    stats.seconds = files.clock() - started
    return stats


@dataclass
class PackageInstalled:
    version: str
    directory: Path
    files: dict[str, int]
    already_installed: bool
    stats: TransferStats

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "directory": str(self.directory), "files": self.files,
                "already_installed": self.already_installed, "stats": self.stats.as_dict()}


def _fetch_small(files: PodFiles, remote: str, via: str, port: int | None) -> bytes:
    reply = files.link.request("GET", files.read_rel(remote, via, port), accept="*/*")
    if len(reply.content) > MANIFEST_LIMIT_BYTES:
        raise TransferError(f"{remote} is larger than a manifest should be")
    return reply.content


def pull_package(files: PodFiles, remote_dir: str, models_dir: Path | str, *, via: str = "files",
                 port: int | None = None, part_bytes: int = DEFAULT_PART_BYTES, replace: bool = False,
                 progress: Callable[[str, int, int], None] | None = None) -> PackageInstalled:
    """Install the model package in `remote_dir` on the pod as `<models_dir>/<version>`, checked as the model switch
    checks it, and return it without switching to it. A version already installed and intact is left alone; a damaged
    one is replaced only with replace=True, and the version that is serving never is."""
    from contracts.learning import ModelManifest
    from runtime.modelstore import ModelStore, ModelStoreError

    remote_dir = clean_remote(remote_dir)
    manifest_bytes = _fetch_small(files, f"{remote_dir}/manifest.json", via, port)
    try:
        manifest = ModelManifest.model_validate_json(manifest_bytes)
    except ValueError as exc:
        raise TransferError(f"{remote_dir}/manifest.json is not a model manifest: {str(exc).splitlines()[0]}") from None
    version = clean_name(manifest.version)
    if version in RESERVED_VERSIONS:
        raise UnsafePath(f"{version!r} is a name the model store keeps for itself")
    names = [clean_name(manifest.gguf)] + ([clean_name(manifest.mmproj)] if manifest.mmproj else [])
    for name in names:
        if name not in manifest.sha256:
            raise TransferError(f"the manifest has no checksum for {name}, so the package is not accepted")
    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)
    store = ModelStore(models)
    final = models / version
    if final.is_symlink():
        raise TransferError(f"{final.name} in {models.name} is a link, so it is not replaced")
    route = "files" if via == "files" else f"proxy:{port}"
    stats = TransferStats("pull", route, part_bytes=part_bytes)
    if final.exists():
        try:
            problems = store.verify(version)
        except (ModelStoreError, ValueError) as exc:
            problems = [str(exc).splitlines()[0]]
        if not problems:
            sizes = {name: (final / name).stat().st_size for name in names}
            return PackageInstalled(version, final, sizes, True, stats)
        if version == store.current():
            raise TransferError(f"{version} is the version that is serving and is damaged: {'; '.join(problems)}; "
                                "it is not replaced")
        if not replace:
            raise TransferError(f"{version} is installed but damaged ({'; '.join(problems)}); pull again with replace")
    staging = models / f".{version}.part"
    staging.mkdir(exist_ok=True)
    sizes: dict[str, int] = {}
    for name in names:
        notify = None if progress is None else (lambda done, total, name=name: progress(name, done, total))
        stats.add(pull_file(files, f"{remote_dir}/{name}", staging / name, via=via, port=port, part_bytes=part_bytes,
                            sha256=manifest.sha256[name], progress=notify))
        sizes[name] = (staging / name).stat().st_size
    (staging / "manifest.json").write_bytes(manifest_bytes)
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    problems = store.verify(version)
    if problems:
        raise TransferError(f"{version} was pulled but does not verify: {'; '.join(problems)}")
    return PackageInstalled(version, final, sizes, False, stats)


# --- the command line ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab.transfer", description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    push_file_p = commands.add_parser("push-file", help="send one file to the pod's Jupyter root")
    push_file_p.add_argument("local")
    push_file_p.add_argument("remote")
    push_dataset_p = commands.add_parser("push-dataset", help="send one approved export folder")
    push_dataset_p.add_argument("local_dir")
    push_dataset_p.add_argument("--to", default="datasets", help="the folder on the pod it goes into")
    for pusher in (push_file_p, push_dataset_p):
        pusher.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_BYTES // MIB)
    pull_file_p = commands.add_parser("pull-file", help="fetch one file in byte-range parts")
    pull_file_p.add_argument("remote")
    pull_file_p.add_argument("local")
    pull_file_p.add_argument("--sha256", default=None)
    pull_package_p = commands.add_parser("pull-package", help="install a model package under data/models")
    pull_package_p.add_argument("remote_dir")
    pull_package_p.add_argument("--models-dir", default="data/models")
    pull_package_p.add_argument("--replace", action="store_true")
    for puller in (pull_file_p, pull_package_p):
        puller.add_argument("--via", choices=("files", "proxy"), default="files")
        puller.add_argument("--port", type=int, default=None)
        puller.add_argument("--part-mib", type=int, default=DEFAULT_PART_BYTES // MIB)
    for command in commands.choices.values():
        command.add_argument("--account", required=True, help="the account letter, e.g. D")
    return parser


def _emit(out: TextIO, payload: Mapping[str, Any]) -> None:
    out.write(json.dumps(payload, sort_keys=True) + "\n")
    out.flush()


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_files: Sequence[Any] | None = None, transport: httpx.BaseTransport | None = None,
         sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.perf_counter,
         stdout: TextIO | None = None) -> int:
    """Exit 0 when the transfer worked, 1 when it did not, 2 for bad arguments and for missing keys. One JSON object on
    stdout; no token and no account name is printed."""
    out = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z]", args.account):
        _emit(out, {"ok": False, "kind": "usage", "error": "--account is one letter"})
        return 2
    letter = args.account.upper()
    try:
        account = load_accounts([letter], environ=environ, env_files=env_files)[letter]
    except MissingCredentials as exc:
        _emit(out, {"ok": False, "kind": "missing_keys", "error": str(exc), "missing": exc.missing})
        return 2
    try:
        with PodLink.for_account(account, transport=transport, environ=environ, sleep=sleep, clock=clock) as link:
            pod = PodFiles(link)
            if args.command == "push-file":
                stats = push_file(pod, args.local, args.remote, chunk_bytes=args.chunk_mib * MIB)
                _emit(out, {"ok": True, **stats.as_dict(), "metrics": stats.metrics()})
            elif args.command == "push-dataset":
                pushed = push_dataset(pod, args.local_dir, remote_root=args.to, chunk_bytes=args.chunk_mib * MIB)
                _emit(out, {"ok": True, **pushed.as_dict(), "metrics": pushed.stats.metrics()})
            elif args.command == "pull-file":
                stats = pull_file(pod, args.remote, args.local, via=args.via, port=args.port,
                                  part_bytes=args.part_mib * MIB, sha256=args.sha256)
                _emit(out, {"ok": True, **stats.as_dict(), "metrics": stats.metrics()})
            else:
                got = pull_package(pod, args.remote_dir, args.models_dir, via=args.via, port=args.port,
                                   part_bytes=args.part_mib * MIB, replace=args.replace)
                _emit(out, {"ok": True, **got.as_dict(), "metrics": got.stats.metrics()})
        return 0
    except ProxyError as exc:
        _emit(out, {"ok": False, "kind": exc.kind, "status": exc.status, "retryable": exc.retryable, "error": str(exc)})
        return 1
    except (TransferError, ValueError, OSError) as exc:
        _emit(out, {"ok": False, "kind": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
