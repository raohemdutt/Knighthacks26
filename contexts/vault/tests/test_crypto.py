import logging
import stat

import pytest

from contexts.vault import DecryptError, KeyRing, Vault, VaultError


def mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_master_key_is_created_once_with_mode_600_and_reused(tmp_path):
    keys = KeyRing(tmp_path)
    assert not keys.path.exists()  # nothing on disk until first use
    memory_key = keys.key_for("memory")
    assert keys.path == tmp_path / "vault" / "master.key"
    assert mode(keys.path) == 0o600 and mode(keys.dir) == 0o700
    master = keys.path.read_bytes()
    assert len(master) == 32

    again = KeyRing(tmp_path)  # a restart
    assert again.key_for("memory") == memory_key
    assert keys.path.read_bytes() == master  # not regenerated


def test_each_context_gets_its_own_key_and_never_the_master(tmp_path):
    keys = KeyRing(tmp_path)
    memory, frames = keys.key_for("memory"), keys.key_for("frames")
    master = keys.path.read_bytes()
    assert len(memory) == len(frames) == 32
    assert len({memory, frames, master}) == 3
    assert KeyRing(tmp_path / "other-box").key_for("memory") != memory  # another master, other keys


def test_a_loose_key_file_is_tightened_and_a_damaged_one_is_refused(tmp_path):
    keys = KeyRing(tmp_path)
    keys.key_for("memory")
    keys.path.chmod(0o644)
    KeyRing(tmp_path).key_for("memory")
    assert mode(keys.path) == 0o600

    keys.path.write_bytes(b"short")
    with pytest.raises(VaultError, match="not a 32-byte key"):
        KeyRing(tmp_path).key_for("memory")
    assert keys.path.read_bytes() == b"short"  # left for a person to sort out, not replaced
    with pytest.raises(VaultError):
        keys.key_for("")


def test_round_trip_and_fresh_nonce_every_time(tmp_path):
    vault = Vault(KeyRing(tmp_path))
    data = b"keys are on the desk, left of the laptop"
    a, b = vault.encrypt("memory", data), vault.encrypt("memory", data)
    assert a != b and a[1:13] != b[1:13]  # same bytes, different nonce and ciphertext
    assert data not in a
    assert len(a) == 1 + 12 + len(data) + 16  # version, nonce, ciphertext, tag
    assert vault.decrypt("memory", a) == vault.decrypt("memory", b) == data
    assert vault.decrypt("memory", vault.encrypt("memory", b"")) == b""
    assert Vault(KeyRing(tmp_path)).decrypt("memory", a) == data  # after a restart


def test_wrong_context_and_tampered_blobs_fail_clearly(tmp_path):
    vault = Vault(KeyRing(tmp_path))
    blob = vault.encrypt("memory", b"secret")
    with pytest.raises(DecryptError, match="wrong context, wrong key or tampered"):
        vault.decrypt("frames", blob)
    for i in (1, 13, len(blob) - 1):  # nonce, ciphertext, tag
        bent = blob[:i] + bytes([blob[i] ^ 1]) + blob[i + 1 :]
        with pytest.raises(DecryptError):
            vault.decrypt("memory", bent)
    for junk in (b"", blob[:20], b"\x02" + blob[1:]):  # empty, cut short, unknown version
        with pytest.raises(DecryptError, match="too short or unknown version"):
            vault.decrypt("memory", junk)
    with pytest.raises(DecryptError):  # a different box's key
        Vault(KeyRing(tmp_path / "other-box")).decrypt("memory", blob)


def test_same_key_under_another_context_name_still_fails(tmp_path):
    """The context is bound as associated data, not only through the derived key."""

    class OneKey(KeyRing):
        def key_for(self, context: str) -> bytes:
            return super().key_for("shared")

    vault = Vault(OneKey(tmp_path))
    with pytest.raises(DecryptError):
        vault.decrypt("frames", vault.encrypt("memory", b"secret"))


def test_key_material_is_never_logged_or_in_errors(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    keys = KeyRing(tmp_path)
    vault = Vault(keys)
    with pytest.raises(DecryptError) as err:
        vault.decrypt("frames", vault.encrypt("memory", b"secret"))
    secrets = [keys.path.read_bytes(), keys.key_for("memory"), keys.key_for("frames")]
    text = caplog.text + str(err.value) + repr(keys) + repr(vault)
    for s in secrets:
        assert s.hex() not in text and repr(s) not in text
