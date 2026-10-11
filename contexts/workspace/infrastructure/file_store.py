"""Inbox files on disk, encrypted with the Privacy Vault (W1).

Each file is one AES-256-GCM blob under `<data>/files/<file_id>.bin`, sealed with the vault key for the
"workspace.files" context (V2). The folder is 700 and each blob 600; a write lands through a temp file and a
rename, so a crash never leaves half a file. The vault's full wipe removes the folder.
"""

import os
import re
from pathlib import Path

from contexts.vault import Vault

CONTEXT = "workspace.files"
FILE_ID = re.compile(r"^[0-9a-f]{32}$")


class EncryptedFileStore:
    def __init__(self, root: str | Path, vault: Vault) -> None:
        self.root = Path(root)
        self.vault = vault

    def path(self, file_id: str) -> Path:
        if not FILE_ID.match(file_id):  # ids are uuid4 hex; anything else could walk out of the folder
            raise KeyError(file_id)
        return self.root / f"{file_id}.bin"

    def save(self, file_id: str, data: bytes) -> None:
        path = self.path(file_id)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        blob = self.vault.encrypt(CONTEXT, data)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)

    def load(self, file_id: str) -> bytes:
        path = self.path(file_id)
        if not path.exists():
            raise KeyError(file_id)
        return self.vault.decrypt(CONTEXT, path.read_bytes())

    def exists(self, file_id: str) -> bool:
        try:
            return self.path(file_id).exists()
        except KeyError:
            return False
