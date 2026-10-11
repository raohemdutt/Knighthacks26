"""Versioned model packages on disk and the switch between them (R7).

    data/models/<version>/{manifest.json, <model>.gguf, <mmproj>.gguf}   one folder per version
    data/models/current  ->  <version>                                    what the box serves
    data/models/previous                                                  text file: the version before the last switch

`manifest.json` follows `ModelManifest` (contracts/learning.py). A version is only switched to after every
file named in its manifest exists and matches its sha256. The `current` link is replaced atomically, so
a reader (the launcher, `config/models.yaml`'s manifest path) sees the old version or the new one, never neither.

    data/models/<version>/.verified.json                                  the record of the files last hashed

`.verified.json` is written by lab/package_pull.py after every file of the version matched its sha256, and by
`verify(version, record=True)` (jetson/atlas_swap.py's swap, so a version that was never pulled, such as atlas-base,
gets one on its first full check): per file the name, size, sha256 and mtime (st_mtime_ns) the file had when it was
hashed, always with written_by lab/package_pull.py so each trusts the other's record (`hashed_by` names the swap when
the swap wrote it). `verify(version, trust_verified=True)`
takes a file as checked without hashing it again only when that record names it with the manifest's sha256 and the
file still has the recorded size and mtime; anything else (no record, a record someone else wrote, a size or mtime
that moved, a manifest whose sha256 changed) hashes the file as before. The default, trust_verified=False, always
hashes.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from contracts.learning import ModelManifest


VERIFIED_RECORD = ".verified.json"
VERIFIED_BY = "lab/package_pull.py"


class ModelStoreError(Exception):
    pass


def read_verified(folder: Path) -> dict[str, dict]:
    """The files the pull's record in `folder` vouches for, by name; empty when there is no record, it cannot be read,
    or it was not written by the pull."""
    try:
        record = json.loads((Path(folder) / VERIFIED_RECORD).read_text(encoding="utf-8"))
        if record.get("written_by") != VERIFIED_BY:
            return {}
        return {str(item["name"]): item for item in record["files"] if isinstance(item, dict)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


RECORDED_BY_SWAP = "jetson/atlas_swap.py"


def write_verified_record(folder: Path, version: str, files: list[dict], *, hashed_by: str | None = None) -> Path:
    """Write <folder>/.verified.json: per file the name, size, sha256 and mtime_ns it had when it was checked, with
    written_by VERIFIED_BY whoever checked it, so the pull and the swap trust each other's record; `hashed_by` says
    which one did when it was not the pull. Written to a scratch name and renamed, so a reader sees the old record,
    none, or the whole new one."""
    record = {"written_by": VERIFIED_BY, "version": version,
              "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              **({"hashed_by": hashed_by} if hashed_by else {}),
              "files": [{k: item.get(k) for k in ("name", "size", "sha256", "mtime_ns")} for item in files]}
    target = Path(folder) / VERIFIED_RECORD
    scratch = Path(folder) / f"{VERIFIED_RECORD}.tmp"
    scratch.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(scratch, target)
    return target


def record_vouches(entry: dict | None, path: Path, sha256: str) -> bool:
    """True when a record entry says `path` hashed to `sha256` and the file still has the size and mtime it had then."""
    if not entry or str(entry.get("sha256", "")).lower() != str(sha256).lower():
        return False
    try:
        st = path.stat()
    except OSError:
        return False
    return entry.get("size") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class ModelStore:
    def __init__(self, root: str | Path = "data/models") -> None:
        self.root = Path(root)

    def manifest(self, version: str) -> ModelManifest:
        path = self.root / version / "manifest.json"
        if not path.is_file():
            raise ModelStoreError(f"no model version {version!r} in {self.root}")
        return ModelManifest.model_validate_json(path.read_text())

    def versions(self) -> list[ModelManifest]:
        """Every installed version, newest first. Folders without a readable manifest are skipped."""
        found = []
        for d in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if d.is_symlink() or not d.is_dir():
                continue  # `current` is an alias, not a version
            try:
                found.append(self.manifest(d.name))
            except (ModelStoreError, ValueError):
                continue
        return sorted(found, key=lambda m: m.created, reverse=True)

    def current(self) -> str | None:
        link = self.root / "current"
        return Path(os.readlink(link)).name if link.is_symlink() else None

    def previous(self) -> str | None:
        f = self.root / "previous"
        return f.read_text().strip() or None if f.is_file() else None

    def verify(self, version: str, *, trust_verified: bool = False, record: bool = False) -> list[str]:
        """What is wrong with the package, as plain sentences; empty means every file is there and matches.
        trust_verified: a file the pull's record vouches for (see the module docstring) is not hashed again.
        record (the swap's, jetson/atlas_swap.py): when every file matched and at least one was hashed, write the record
        (write_verified_record) with the size and mtime each file had before it was checked, so a later trusting
        check of an unchanged file hashes nothing."""
        m = self.manifest(version)
        vouched = read_verified(self.root / version) if trust_verified else {}
        problems, checked, hashed = [], [], False
        for name in [m.gguf, *([m.mmproj] if m.mmproj else [])]:
            path = self.root / version / name
            st = path.stat() if record and path.is_file() else None  # before the hash: a file changed meanwhile goes stale
            if not path.is_file():
                problems.append(f"{name} is missing")
                continue
            elif name not in m.sha256:
                problems.append(f"the manifest has no checksum for {name}")
                continue
            elif not record_vouches(vouched.get(name), path, m.sha256[name]):
                hashed = True
                if sha256_file(path) != m.sha256[name]:
                    problems.append(f"{name} does not match its sha256 (a damaged or partial copy)")
                    continue
            if st is not None:
                checked.append({"name": name, "size": st.st_size, "sha256": m.sha256[name], "mtime_ns": st.st_mtime_ns})
        if record and hashed and not problems:  # a record that vouched for every file stays as it is
            write_verified_record(self.root / version, version, checked, hashed_by=RECORDED_BY_SWAP)
        return problems

    def activate(self, version: str, *, trust_verified: bool = False, record: bool = False) -> None:
        """Point `current` at `version` (after verifying it) and remember what it replaced."""
        problems = self.verify(version, trust_verified=trust_verified, record=record)
        if problems:
            raise ModelStoreError(f"{version} is not safe to switch to: " + "; ".join(problems))
        before = self.current()
        tmp = self.root / "current.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(version, tmp)
        try:
            os.replace(tmp, self.root / "current")  # atomic on POSIX
        except PermissionError:  # Windows cannot rename a directory symlink over another one: remove, then rename
            link = self.root / "current"
            try:
                link.unlink()
            except OSError:  # a directory symlink on Windows is removed with rmdir
                os.rmdir(link)
            os.replace(tmp, link)
        if before and before != version:
            (self.root / "previous").write_text(before + "\n")

    def set_previous(self, version: str | None) -> None:
        f = self.root / "previous"
        if version:
            f.write_text(version + "\n")
        elif f.exists():
            f.unlink()

    def rollback(self, *, trust_verified: bool = False) -> str:
        """Switch back to the version before the last switch; returns it."""
        target = self.previous()
        if not target:
            raise ModelStoreError("there is no earlier version to go back to")
        self.activate(target, trust_verified=trust_verified)
        return target
