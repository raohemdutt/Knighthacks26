"""P9: where face templates and the on/off switch live.

    store = IdentityStore(data_dir, vault)        # vault: contexts.vault.Vault
    store.enabled                                 # False until someone switches it on
    store.save(Member(...))                       # <data_dir>/vault/identity/<member_id>.face
    store.members()                               # decrypted, in memory only

A template file is one `vault.encrypt("identity", ...)` blob holding the member's id, name, vector and
enrolment time: nothing about a person is on disk in plain text, and nothing here is ever written to the
event log, Memory or an export. The switch is <data_dir>/vault/identity.json: {"identity_enabled": false}.
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTEXT = "identity"
MEMBER_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}\Z")  # \Z, not $: "$" also matches before a trailing newline


@dataclass(frozen=True)
class Member:
    member_id: str
    name: str
    enrolled_at: str  # ISO time
    model: str  # which recognizer made the vector; a template only matches vectors from the same one
    frames: int
    vector: list[float]


class IdentityStore:
    def __init__(self, data_dir: str | Path, vault: Any) -> None:
        self.root = Path(data_dir) / "vault"
        self.dir = self.root / "identity"
        self.settings_path = self.root / "identity.json"
        self.vault = vault
        self._cache: tuple[tuple, list[Member]] | None = None
        self.enabled = self._read_enabled()  # read once: the camera threads ask many times a second

    # --- the switch ---

    def _read_enabled(self) -> bool:
        try:
            return json.loads(self.settings_path.read_text()).get("identity_enabled") is True
        except (OSError, ValueError, AttributeError):
            return False  # no file, or an unreadable one: off

    def set_enabled(self, on: bool) -> None:
        self._write(self.settings_path, (json.dumps({"identity_enabled": bool(on)}, indent=2) + "\n").encode())
        self.enabled = bool(on)

    # --- templates ---

    def save(self, member: Member) -> None:
        if not MEMBER_ID.match(member.member_id):
            raise ValueError("member_id must be 1 to 32 of a-z, 0-9, - and _")
        blob = self.vault.encrypt(CONTEXT, json.dumps(asdict(member)).encode())
        self._write(self.dir / f"{member.member_id}.face", blob)

    def members(self) -> list[Member]:
        """Every enrolled member, oldest first. A file that does not decrypt is skipped, not guessed at."""
        files = sorted(self.dir.glob("*.face")) if self.dir.is_dir() else []
        stamp = tuple((p.name, p.stat().st_mtime_ns) for p in files)
        if self._cache is None or self._cache[0] != stamp:
            found = []
            for path in files:
                try:
                    found.append(Member(**json.loads(self.vault.decrypt(CONTEXT, path.read_bytes()))))
                except Exception:  # wrong key, damaged or foreign file
                    continue
            self._cache = (stamp, sorted(found, key=lambda m: (m.enrolled_at, m.member_id)))
        return list(self._cache[1])

    def delete(self, member_id: str) -> bool:
        path = self.dir / f"{member_id}.face"
        if not MEMBER_ID.match(member_id) or not path.is_file():
            return False
        path.unlink()
        return True

    def _write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        tmp = path.with_name(f".{path.name}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)  # never half a template
