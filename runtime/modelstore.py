"""Versioned model packages on disk and the switch between them (R7).

    data/models/<version>/{manifest.json, <model>.gguf, <mmproj>.gguf}   one folder per version
    data/models/current  ->  <version>                                    what the box serves
    data/models/previous                                                  text file: the version before the last switch

`manifest.json` follows `ModelManifest` (contracts/learning.py). A version is only switched to after every
file named in its manifest exists and matches its sha256. The `current` link is replaced atomically, so
a reader (the launcher, `config/models.yaml`'s manifest path) sees the old version or the new one, never neither.
"""

import hashlib
import os
from pathlib import Path

from contracts.learning import ModelManifest


class ModelStoreError(Exception):
    pass


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

    def verify(self, version: str) -> list[str]:
        """What is wrong with the package, as plain sentences; empty means every file is there and matches."""
        m = self.manifest(version)
        problems = []
        for name in [m.gguf, *([m.mmproj] if m.mmproj else [])]:
            path = self.root / version / name
            if not path.is_file():
                problems.append(f"{name} is missing")
            elif name not in m.sha256:
                problems.append(f"the manifest has no checksum for {name}")
            elif sha256_file(path) != m.sha256[name]:
                problems.append(f"{name} does not match its sha256 (a damaged or partial copy)")
        return problems

    def activate(self, version: str) -> None:
        """Point `current` at `version` (after verifying it) and remember what it replaced."""
        problems = self.verify(version)
        if problems:
            raise ModelStoreError(f"{version} is not safe to switch to: " + "; ".join(problems))
        before = self.current()
        tmp = self.root / "current.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(version, tmp)
        os.replace(tmp, self.root / "current")  # atomic on POSIX
        if before and before != version:
            (self.root / "previous").write_text(before + "\n")

    def set_previous(self, version: str | None) -> None:
        f = self.root / "previous"
        if version:
            f.write_text(version + "\n")
        elif f.exists():
            f.unlink()

    def rollback(self) -> str:
        """Switch back to the version before the last switch; returns it."""
        target = self.previous()
        if not target:
            raise ModelStoreError("there is no earlier version to go back to")
        self.activate(target)
        return target
