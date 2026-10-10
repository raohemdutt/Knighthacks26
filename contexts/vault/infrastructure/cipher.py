"""Encrypt and decrypt helpers (V2): AES-256-GCM under a context's data key.

    vault = Vault(KeyRing(data_dir))
    blob = vault.encrypt("memory", b"...")
    vault.decrypt("memory", blob)

Blob layout: 1 version byte | 12-byte random nonce | ciphertext + 16-byte tag.
The context name is the associated data, so a blob made for one context does
not decrypt as another. Random nonces are safe for about 2^32 blobs per key.
"""

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from contexts.vault.infrastructure.keyring import INFO_PREFIX, KeyRing, VaultError

VERSION = b"\x01"
NONCE_BYTES = 12
TAG_BYTES = 16


class DecryptError(VaultError):
    """The blob is not ours, was made for another context, or has been changed."""


class Vault:
    def __init__(self, keys: KeyRing) -> None:
        self.keys = keys

    def encrypt(self, context: str, data: bytes) -> bytes:
        nonce = os.urandom(NONCE_BYTES)
        return VERSION + nonce + AESGCM(self.keys.key_for(context)).encrypt(nonce, data, _aad(context))

    def decrypt(self, context: str, blob: bytes) -> bytes:
        if len(blob) < 1 + NONCE_BYTES + TAG_BYTES or blob[:1] != VERSION:
            raise DecryptError(f"not a vault blob for context {context!r}: too short or unknown version")
        nonce, sealed = blob[1 : 1 + NONCE_BYTES], blob[1 + NONCE_BYTES :]
        try:
            return AESGCM(self.keys.key_for(context)).decrypt(nonce, sealed, _aad(context))
        except InvalidTag:
            raise DecryptError(f"cannot decrypt blob for context {context!r}: wrong context, wrong key or tampered data") from None


def _aad(context: str) -> bytes:
    return INFO_PREFIX + context.encode()
