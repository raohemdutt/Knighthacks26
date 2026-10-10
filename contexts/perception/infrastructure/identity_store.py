"""P9: where face templates and the on/off switch live.

    store = IdentityStore(data_dir, vault)        # vault: contexts.vault.Vault
    store.enabled                                 # False until someone switches it on
    store.save(Member(...))                       # <data_dir>/vault/identity/<member_id>.face
    store.members()                               # decrypted, in memory only

A template file is one `vault.encrypt("identity", ...)` blob holding the member's id, name, vector and
enrolment time: nothing about a person is on disk in plain text, and nothing here is ever written to the
event log, Memory or an export. The switch is <data_dir>/vault/identity.json: {"identity_enabled": false}.

A blob may also hold `vectors` (up to MAX_EXEMPLARS exemplar vectors: straight on, slight turns, with and
without glasses; `vector` stays their mean), `expressions` (this member turned "read my expressions" on),
`updated_at`, `source` (live, scan or photos), `consent_at` and `circle_member_id`. A blob written before those existed has none of
them and loads as before: its one vector is its only exemplar. identity.json holds `"everyone": true` only
while the anonymous room mode is on. The switch file is read again when it changes on disk, so a wipe of
everything switches the running box off without a restart.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

CONTEXT = "identity"
MEMBER_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}\Z")  # \Z, not $: "$" also matches before a trailing newline
CIRCLE_MEMBER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,63}\Z")
MAX_EXEMPLARS = 12


@dataclass(frozen=True)
class Member:
    member_id: str
    name: str
    enrolled_at: str  # ISO time
    model: str  # which recognizer made the vector; a template only matches vectors from the same one
    frames: int
    vector: list[float]  # the mean of the exemplars
    vectors: list[list[float]] = field(default_factory=list)  # the exemplars; empty in a blob from before they existed
    expressions: bool = False  # this member switched "read my expressions" on
    updated_at: str | None = None  # the last time looks or photos were added
    source: str = "live"  # live (the camera) or photos (uploaded)
    consent_at: str | None = None  # when the person being enrolled was confirmed to agree
    circle_member_id: str | None = None  # which Circle member this is, when someone said so: an id, nothing else

    @property
    def exemplars(self) -> list[list[float]]:
        """What a look is matched against: every exemplar, or the one vector of an older template."""
        return self.vectors or [self.vector]


_MEMBER_FIELDS = {f.name for f in fields(Member)}


class IdentityStore:
    def __init__(self, data_dir: str | Path, vault: Any) -> None:
        self.root = Path(data_dir) / "vault"
        self.dir = self.root / "identity"
        self.settings_path = self.root / "identity.json"
        self.vault = vault
        self._cache: tuple[tuple, list[Member]] | None = None
        self._switches: tuple[int | None, bool, bool] | None = None  # (the file's mtime, enabled, everyone)

    # --- the switches ---

    def _read(self) -> tuple[int | None, bool, bool]:
        """(mtime, enabled, everyone). The camera threads ask many times a second, so the file is only read
        again when its time changed (one stat); a file that is gone or unreadable is off."""
        try:
            stamp = self.settings_path.stat().st_mtime_ns
        except OSError:
            self._switches = (None, False, False)
            return self._switches
        if self._switches is None or self._switches[0] != stamp:
            try:
                raw = json.loads(self.settings_path.read_text())
                on = raw.get("identity_enabled") is True
                self._switches = (stamp, on, on and raw.get("everyone") is True)
            except (OSError, ValueError, AttributeError):
                self._switches = (stamp, False, False)
        return self._switches

    @property
    def enabled(self) -> bool:
        return self._read()[1]

    @property
    def everyone(self) -> bool:
        """The anonymous room mode. Never on while recognition is off."""
        return self._read()[2]

    def set_enabled(self, on: bool) -> None:
        self._save_switches(bool(on), self.everyone and bool(on))

    def set_everyone(self, on: bool) -> None:
        self._save_switches(self.enabled, bool(on) and self.enabled)

    def _save_switches(self, enabled: bool, everyone: bool) -> None:
        raw = {"identity_enabled": enabled, **({"everyone": True} if everyone else {})}
        self._write(self.settings_path, (json.dumps(raw, indent=2) + "\n").encode())
        self._switches = None

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
                    raw = json.loads(self.vault.decrypt(CONTEXT, path.read_bytes()))
                    found.append(Member(**{k: v for k, v in raw.items() if k in _MEMBER_FIELDS}))
                except Exception:  # wrong key, damaged or foreign file
                    continue
            self._cache = (stamp, sorted(found, key=lambda m: (m.enrolled_at, m.member_id)))
        return list(self._cache[1])

    def get(self, member_id: str) -> Member | None:
        return next((m for m in self.members() if m.member_id == member_id), None)

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
