"""Key ring (V1): one master key on disk, one data key per context derived from it.

    keys = KeyRing(data_dir)
    keys.key_for("memory")     # 32 bytes, HKDF-SHA256(master, info="atlas.vault.v1:memory")

The master key is 32 random bytes in <data_dir>/vault/master.key (mode 600,
directory 700), written on first use and only ever read after that. It is
never used to encrypt data, never logged and never returned.
"""

import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEY_BYTES = 32
INFO_PREFIX = b"atlas.vault.v1:"  # bump the version to rotate every derived key at once


class VaultError(Exception):
    """Anything the vault refuses to do. Messages never contain key material."""


class KeyRing:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.dir = Path(data_dir) / "vault"
        self.path = self.dir / "master.key"
        self._master: bytes | None = None  # read on first use

    def key_for(self, context: str) -> bytes:
        """The data key for one context: the same every time, different for every context."""
        if not context:
            raise VaultError("a context name is needed to derive a key")
        if self._master is None:
            self._master = self._load_or_create()
        hkdf = HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None, info=INFO_PREFIX + context.encode())
        return hkdf.derive(self._master)

    def _load_or_create(self) -> bytes:
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        try:  # O_EXCL: an existing key is never overwritten, which would orphan everything encrypted with it
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return self._load()
        key = os.urandom(KEY_BYTES)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
        return key

    def _load(self) -> bytes:
        os.chmod(self.path, 0o600)
        key = self.path.read_bytes()
        if len(key) != KEY_BYTES:  # do not regenerate: the old data would be lost without anyone noticing
            raise VaultError(f"{self.path} is not a {KEY_BYTES}-byte key; restore it or remove it by hand")
        return key
