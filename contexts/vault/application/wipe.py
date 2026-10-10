"""Wipe (V4): a confirmation token, then delete what Atlas has stored.

    tokens = WipeTokens()
    token, expires_at = tokens.request("all", now)   # good for two minutes, once
    scope = tokens.redeem(token, now)                # "all", or None

    wiper = Wiper(data_dir)
    bus.subscribe("vault.wipe.requested", wiper.on_wipe_requested)   # LIVE ONLY: a replayed wipe would wipe again

What each scope removes, all under data_dir:
- keyframes: everything in frames/
- memory:    memory.sqlite and its -wal and -shm files
- all:       both, plus everything in exports/ and events/, and the face templates in vault/identity/ (P9)

Directories are emptied, not removed. Nothing outside data_dir is touched: a
target that is a symlink, or that resolves outside data_dir, is refused and
reported. Links inside a wiped directory are unlinked, never followed. The
vault's own folder (master key, policy) is left alone unless
`wipe(..., include_master_key=True)` is called directly; the event cannot ask for that.
"""

import hmac
import logging
import os
import secrets
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from contexts.vault.infrastructure.keyring import VaultError
from contracts.vault import WipeRequested

log = logging.getLogger("atlas.vault")

TOKEN_TTL = timedelta(minutes=2)
SCOPES = {"keyframes": ("frames",), "memory": ("memory",), "all": ("frames", "memory", "exports", "events")}


class WipeTokens:
    """One pending confirmation at a time, held in memory: a restart or a new request cancels it."""

    def __init__(self, ttl: timedelta = TOKEN_TTL) -> None:
        self.ttl = ttl
        self._pending: tuple[str, str, datetime] | None = None  # token, scope, expires_at

    def request(self, scope: str, now: datetime) -> tuple[str, datetime]:
        token, expires_at = secrets.token_urlsafe(16), now + self.ttl
        self._pending = (token, scope, expires_at)
        return token, expires_at

    def redeem(self, token: str, now: datetime) -> str | None:
        """The scope the token confirms, or None if it is wrong, used or expired. A right token works once."""
        if self._pending is None:
            return None
        expected, scope, expires_at = self._pending
        if not hmac.compare_digest(token.encode(), expected.encode()):
            return None
        self._pending = None
        return scope if now < expires_at else None


class OutsideDataDir(VaultError):
    """A wipe target that is not plainly inside data_dir."""


@dataclass
class WipeReport:
    scope: str
    removed: dict[str, int] = field(default_factory=dict)  # target -> files deleted
    refused: list[str] = field(default_factory=list)  # paths left alone
    master_key_removed: bool = False
    identity_removed: int = 0  # face templates (P9) deleted from vault/identity/; only scope "all" removes them


class Wiper:
    def __init__(self, data_dir: str | Path = "data", memory_db: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.memory_db = Path(memory_db) if memory_db else self.data_dir / "memory.sqlite"
        self.last: WipeReport | None = None

    def on_wipe_requested(self, e: WipeRequested) -> None:
        self.wipe(e.scope)

    def wipe(self, scope: str, include_master_key: bool = False) -> WipeReport:
        report = WipeReport(scope=scope)
        for target in SCOPES[scope]:
            report.removed[target] = 0
            for path in self._paths(target):
                try:
                    report.removed[target] += self._remove(path)
                except OutsideDataDir as exc:
                    log.warning("wipe refused: %s", exc)
                    report.refused.append(str(path))
        if scope == "all":  # "everything" includes who the box can recognise; the key and the policy stay
            report.identity_removed = self._remove(self.data_dir / "vault" / "identity", vault_ok=True)
        if include_master_key:  # everything encrypted with it becomes unreadable for good
            report.master_key_removed = self._remove(self.data_dir / "vault" / "master.key", vault_ok=True) > 0
        log.warning("wipe %s: removed %s, refused %s", scope, report.removed, report.refused)
        self.last = report
        return report

    def _paths(self, target: str) -> list[Path]:
        if target == "memory":
            db = self.memory_db
            return [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")]
        return [self.data_dir / target]

    def _remove(self, path: Path, vault_ok: bool = False) -> int:
        """Delete a file, or everything inside a directory (the directory stays). Returns files deleted."""
        self._check(path, vault_ok)
        if not path.exists():
            return 0
        if path.is_file():
            path.unlink()
            return 1
        count = 0
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                count += sum(len(files) for _, _, files in os.walk(child))  # os.walk does not follow links
                shutil.rmtree(child)  # nor does rmtree
            else:
                child.unlink()  # a link is removed, never what it points at
                count += 1
        return count

    def _check(self, path: Path, vault_ok: bool) -> None:
        if path.is_symlink():
            raise OutsideDataDir(f"{path} is a symlink")
        root, real = self.data_dir.resolve(), path.resolve()
        if root not in real.parents:
            raise OutsideDataDir(f"{path} is not inside {self.data_dir}")
        vault = root / "vault"
        if not vault_ok and (real == vault or vault in real.parents):
            raise OutsideDataDir(f"{path} is the vault's own folder")
