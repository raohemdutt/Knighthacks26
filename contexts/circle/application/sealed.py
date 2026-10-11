"""Sealed envelopes for the Circle's face share (version 1). Pure: keys and bytes in, dicts out. No I/O.

Every primitive comes from the `cryptography` package (already a dependency; the vault uses it too). Nothing here
is a cipher of our own: this file only puts reviewed pieces together, the way a sealed box does.

Each box has two long-term keys:
- an Ed25519 key that signs everything the box sends (so a reader can check who it is from), and
- an X25519 key that others encrypt to (so only this box can read what is meant for it).
The private halves stay in the box's vault. The public halves are published to the Circle in a `box_key`
record, signed with the Ed25519 key so the two belong together.

Sealing a template for one friend (`seal`):
1. a fresh X25519 key pair is made for this one message (ephemeral; its private half is dropped at once);
2. X25519(ephemeral private, friend's public) gives a shared secret only the friend's box can also compute;
3. HKDF-SHA256 turns it into a 32-byte key, bound to this protocol, both public keys and the sender's signing key;
4. ChaCha20-Poly1305 encrypts the template with that key. The header (who, to whom, which Circle, a counter,
   when it was made and when it expires) is the associated data: changing any of it makes decryption fail;
5. Ed25519 signs the header, the ephemeral public key, the nonce and the ciphertext.
A revoke is a signed header with no ciphertext.

What this gives: the shared database and every other member see only ciphertext; a share cannot be re-addressed,
re-dated or passed off as someone else's; a changed bit is refused. What it does not give is in the README
("Limits"): first-use key trust, no forward secrecy, and metadata (who shared with whom, when) is visible.

`SealError` messages are fixed words. They never hold key material, plaintext or ciphertext.
"""

import base64
import binascii
import hashlib
import json
import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

VERSION = 1
KEY_BYTES = 32
NONCE_BYTES = 12
SIG_BYTES = 64
# Domain separation: a signature or a key made for one purpose is useless for another.
DOMAIN_KEY = b"atlas.circle.box-key.v1\x00"
DOMAIN_SHARE = b"atlas.circle.face-share.v1\x00"
DOMAIN_REVOKE = b"atlas.circle.face-revoke.v1\x00"
DOMAIN_FINGERPRINT = b"atlas.circle.fingerprint.v1\x00"
FINGERPRINT_HEX = 20  # 80 bits, shown as five groups of four

_RAW = (serialization.Encoding.Raw, serialization.PublicFormat.Raw)


class SealError(Exception):
    """The envelope is not acceptable: not signed by the key we know, changed on the way, not for us, or not
    well formed. The message is one of a few fixed sentences and safe to show or log."""


@dataclass(frozen=True)
class PublicKeys:
    sign: bytes  # Ed25519 public key, 32 bytes
    enc: bytes  # X25519 public key, 32 bytes

    @property
    def fingerprint(self) -> str:
        """What two friends compare: 80 bits of SHA-256 over both public keys, as "ab12 cd34 ef56 0a1b 2c3d"."""
        digest = hashlib.sha256(DOMAIN_FINGERPRINT + self.sign + self.enc).hexdigest()[:FINGERPRINT_HEX]
        return " ".join(digest[i:i + 4] for i in range(0, FINGERPRINT_HEX, 4))


class BoxKeys:
    """A box's two private keys. `to_bytes` is what the vault encrypts; nothing else ever takes them out."""

    def __init__(self, sign: Ed25519PrivateKey, enc: X25519PrivateKey) -> None:
        self._sign, self._enc = sign, enc
        self.public = PublicKeys(sign.public_key().public_bytes(*_RAW), enc.public_key().public_bytes(*_RAW))

    @classmethod
    def generate(cls) -> "BoxKeys":
        return cls(Ed25519PrivateKey.generate(), X25519PrivateKey.generate())

    @classmethod
    def from_bytes(cls, raw: bytes) -> "BoxKeys":
        if len(raw) != 2 * KEY_BYTES:
            raise SealError("the box key is damaged")
        return cls(Ed25519PrivateKey.from_private_bytes(raw[:KEY_BYTES]), X25519PrivateKey.from_private_bytes(raw[KEY_BYTES:]))

    def to_bytes(self) -> bytes:
        private = (serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
        return self._sign.private_bytes(*private) + self._enc.private_bytes(*private)

    def __repr__(self) -> str:  # never the keys, also not by accident in a log line or a traceback
        return f"BoxKeys({self.public.fingerprint})"


# --- small helpers ---


def canonical(value: dict) -> bytes:
    """One byte string per dict, whatever order its keys were written in."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _raw(text: object, length: int | None = None) -> bytes:
    if not isinstance(text, str):
        raise SealError("the envelope is not well formed")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise SealError("the envelope is not well formed") from None
    if length is not None and len(data) != length:
        raise SealError("the envelope is not well formed")
    return data


def _framed(header: dict) -> bytes:
    """The header with its length in front, so nothing after it can be read as part of it."""
    h = canonical(header)
    return len(h).to_bytes(4, "big") + h


def _check(public_sign: bytes, signature: bytes, message: bytes) -> None:
    try:
        Ed25519PublicKey.from_public_bytes(public_sign).verify(signature, message)
    except (InvalidSignature, ValueError):
        raise SealError("the signature does not match the key this box knows for that member") from None


def _header(body: object) -> dict:
    header = body.get("header") if isinstance(body, dict) else None
    if not isinstance(header, dict):
        raise SealError("the envelope is not well formed")
    return header


# --- the public key record ---


def key_record(keys: BoxKeys, circle_id: str, member_id: str, created_at: str) -> dict:
    """What a box publishes about itself: both public keys, signed by its signing key."""
    header = {"v": VERSION, "kind": "box_key", "circle_id": circle_id, "member_id": member_id,
              "sign_pub": b64(keys.public.sign), "enc_pub": b64(keys.public.enc), "created_at": created_at}
    return {"header": header, "sig": b64(keys._sign.sign(DOMAIN_KEY + _framed(header)))}


def read_key_record(body: object, circle_id: str, member_id: str) -> PublicKeys:
    """The keys in a record, once it is signed by its own signing key and says it is `member_id`'s in this
    Circle. That proves the two keys belong together; it does not prove whose they are (see the README)."""
    header = _header(body)
    if header.get("v") != VERSION or header.get("kind") != "box_key":
        raise SealError("the envelope is not well formed")
    if header.get("circle_id") != circle_id or header.get("member_id") != member_id:
        raise SealError("the envelope is addressed to someone else")
    keys = PublicKeys(_raw(header.get("sign_pub"), KEY_BYTES), _raw(header.get("enc_pub"), KEY_BYTES))
    _check(keys.sign, _raw(body.get("sig"), SIG_BYTES), DOMAIN_KEY + _framed(header))
    return keys


# --- a sealed template ---


def _message_key(shared: bytes, ephemeral: bytes, recipient_enc: bytes, sender_sign: bytes) -> bytes:
    info = DOMAIN_SHARE + ephemeral + recipient_enc + sender_sign
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None, info=info).derive(shared)


def _signed_share(header: dict, ephemeral: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    return DOMAIN_SHARE + _framed(header) + ephemeral + nonce + ciphertext  # fixed lengths, then the rest


def seal(keys: BoxKeys, header: dict, plaintext: bytes, recipient: PublicKeys) -> dict:
    """`plaintext` readable only by the box that holds `recipient.enc`'s private half, signed by `keys`."""
    once = X25519PrivateKey.generate()  # used for this message and dropped when this function returns
    ephemeral = once.public_key().public_bytes(*_RAW)
    try:
        shared = once.exchange(X25519PublicKey.from_public_bytes(recipient.enc))
    except ValueError:  # a public key that is not a usable point
        raise SealError("that member's box key cannot be used") from None
    nonce = os.urandom(NONCE_BYTES)
    key = _message_key(shared, ephemeral, recipient.enc, keys.public.sign)
    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, DOMAIN_SHARE + _framed(header))
    return {"header": header, "eph_pub": b64(ephemeral), "nonce": b64(nonce), "ciphertext": b64(ciphertext),
            "sig": b64(keys._sign.sign(_signed_share(header, ephemeral, nonce, ciphertext)))}


def verify_share(body: object, sender: PublicKeys) -> dict:
    """The header of a sealed share, once the whole envelope is signed by `sender`. Decrypts nothing."""
    header = _header(body)
    ephemeral, nonce = _raw(body.get("eph_pub"), KEY_BYTES), _raw(body.get("nonce"), NONCE_BYTES)
    _check(sender.sign, _raw(body.get("sig"), SIG_BYTES), _signed_share(header, ephemeral, nonce, _raw(body.get("ciphertext"))))
    return header


def open_sealed(keys: BoxKeys, body: object, sender: PublicKeys) -> bytes:
    """The plaintext of a share sealed to this box and signed by `sender`. Anything else is a SealError."""
    header = verify_share(body, sender)
    ephemeral, nonce = _raw(body["eph_pub"], KEY_BYTES), _raw(body["nonce"], NONCE_BYTES)
    try:
        shared = keys._enc.exchange(X25519PublicKey.from_public_bytes(ephemeral))
        key = _message_key(shared, ephemeral, keys.public.enc, sender.sign)
        return ChaCha20Poly1305(key).decrypt(nonce, _raw(body["ciphertext"]), DOMAIN_SHARE + _framed(header))
    except (InvalidTag, ValueError):
        raise SealError("this box cannot open the envelope: it was sealed for another box or changed on the way") from None


# --- a revoke: a signed header, nothing else ---


def sign_revoke(keys: BoxKeys, header: dict) -> dict:
    return {"header": header, "sig": b64(keys._sign.sign(DOMAIN_REVOKE + _framed(header)))}


def verify_revoke(body: object, sender: PublicKeys) -> dict:
    header = _header(body)
    _check(sender.sign, _raw(body.get("sig"), SIG_BYTES), DOMAIN_REVOKE + _framed(header))
    return header
