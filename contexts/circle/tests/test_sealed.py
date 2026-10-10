"""The sealed envelope (contexts/circle/application/sealed.py): only the box it is for can open it, anything
changed on the way is refused, and nobody can pass one off as someone else's. Made-up bytes only."""

import base64
import copy
import json

import pytest

from contexts.circle.application import sealed
from contexts.circle.application.sealed import BoxKeys, SealError

SECRET = json.dumps({"name": "Zebulon Quartermaine", "vector": [0.125, -0.5, 0.75] * 40}).encode()
HEADER = {"v": 1, "kind": "face_share", "circle_id": "team", "from": "ana", "to": "bob", "share_id": "s1", "seq": 7,
          "issued_at": "2026-10-11T12:00:00+00:00", "expires_at": "2026-11-10T12:00:00+00:00", "model": "fake"}


@pytest.fixture
def people():
    return BoxKeys.generate(), BoxKeys.generate(), BoxKeys.generate()  # ana, bob, mal


def flip(text: str, at: int = 0) -> str:
    raw = bytearray(base64.b64decode(text))
    raw[at] ^= 0x01
    return base64.b64encode(bytes(raw)).decode()


def test_only_the_box_it_was_sealed_for_can_open_it(people):
    ana, bob, mal = people
    body = sealed.seal(ana, HEADER, SECRET, bob.public)
    assert sealed.open_sealed(bob, body, ana.public) == SECRET
    for other in (mal, ana):  # another member, and even the sender herself, cannot
        with pytest.raises(SealError, match="cannot open the envelope"):
            sealed.open_sealed(other, body, ana.public)


def test_the_envelope_holds_ciphertext_not_the_template(people):
    ana, bob, _ = people
    one, two = sealed.seal(ana, HEADER, SECRET, bob.public), sealed.seal(ana, HEADER, SECRET, bob.public)
    assert set(one) == {"header", "eph_pub", "nonce", "ciphertext", "sig"}
    raw = base64.b64decode(one["ciphertext"])
    assert len(raw) == len(SECRET) + 16  # the text plus the authentication tag: it is not padded, packed or encoded text
    assert b"Zebulon" not in raw and b"0.125" not in raw and SECRET[:16] not in raw
    assert b"Zebulon" not in json.dumps(one).encode() and base64.b64encode(SECRET)[:24].decode() not in json.dumps(one)
    # the same template sealed twice shares nothing: a fresh key pair and nonce every time
    assert one["eph_pub"] != two["eph_pub"] and one["nonce"] != two["nonce"] and one["ciphertext"] != two["ciphertext"]


@pytest.mark.parametrize("field", ["ciphertext", "nonce", "eph_pub", "sig"])
def test_one_changed_bit_anywhere_is_refused(people, field):
    ana, bob, _ = people
    body = sealed.seal(ana, HEADER, SECRET, bob.public)
    body[field] = flip(body[field])
    with pytest.raises(SealError):
        sealed.open_sealed(bob, body, ana.public)
    with pytest.raises(SealError):
        sealed.verify_share(body, ana.public)


def test_a_changed_tag_or_last_byte_of_the_ciphertext_is_refused(people):
    ana, bob, _ = people
    body = sealed.seal(ana, HEADER, SECRET, bob.public)
    body["ciphertext"] = flip(body["ciphertext"], at=-1)
    with pytest.raises(SealError):
        sealed.open_sealed(bob, body, ana.public)


@pytest.mark.parametrize("change", [{"to": "mal"}, {"from": "mal"}, {"expires_at": "2027-11-10T12:00:00+00:00"},
                                    {"seq": 8}, {"circle_id": "other"}, {"share_id": "s2"}, {"extra": 1}])
def test_a_share_cannot_be_readdressed_redated_or_renumbered(people, change):
    ana, bob, _ = people
    body = sealed.seal(ana, HEADER, SECRET, bob.public)
    body["header"] = {**body["header"], **change}
    with pytest.raises(SealError, match="signature"):
        sealed.open_sealed(bob, body, ana.public)


def test_someone_else_cannot_sign_it_as_theirs_or_as_the_sender(people):
    ana, bob, mal = people
    body = sealed.seal(ana, HEADER, SECRET, bob.public)
    with pytest.raises(SealError, match="signature"):  # mal's key did not sign it
        sealed.open_sealed(bob, body, mal.public)
    # mal strips ana's signature and signs the very same envelope: the signature now checks against mal's key ...
    stolen = copy.deepcopy(body)
    signed = sealed._signed_share(stolen["header"], *(base64.b64decode(stolen[k]) for k in ("eph_pub", "nonce", "ciphertext")))
    stolen["sig"] = sealed.b64(mal._sign.sign(signed))
    assert sealed.verify_share(stolen, mal.public)["from"] == "ana"  # ... but the header still names ana (the protocol checks that),
    with pytest.raises(SealError, match="cannot open"):  # and the message key was bound to ana's signing key: it does not open as mal's
        sealed.open_sealed(bob, stolen, mal.public)
    # and mal cannot make a new one that says it is from ana: bob checks it against ana's key
    forged = sealed.seal(mal, HEADER, SECRET, bob.public)
    with pytest.raises(SealError, match="signature"):
        sealed.open_sealed(bob, forged, ana.public)


def test_a_box_key_record_is_signed_by_itself_and_names_its_member(people):
    ana, _, mal = people
    record = sealed.key_record(ana, "team", "ana", "2026-10-11T12:00:00+00:00")
    assert sealed.read_key_record(record, "team", "ana") == ana.public
    assert set(record) == {"header", "sig"} and set(record["header"]) == {"v", "kind", "circle_id", "member_id", "sign_pub",
                                                                         "enc_pub", "created_at"}
    with pytest.raises(SealError, match="someone else"):
        sealed.read_key_record(record, "team", "bob")  # posted under another member's name
    with pytest.raises(SealError, match="someone else"):
        sealed.read_key_record(record, "other-circle", "ana")
    swapped = copy.deepcopy(record)
    swapped["header"]["enc_pub"] = sealed.b64(mal.public.enc)  # someone else's encryption key under ana's signature
    with pytest.raises(SealError, match="signature"):
        sealed.read_key_record(swapped, "team", "ana")
    for junk in (None, {}, {"header": "x"}, {"header": {**record["header"], "sign_pub": "!!"}, "sig": record["sig"]},
                 {"header": record["header"], "sig": "AAAA"}):
        with pytest.raises(SealError):
            sealed.read_key_record(junk, "team", "ana")


def test_a_revoke_is_a_signed_header_and_a_share_signature_is_not_a_revoke(people):
    ana, bob, mal = people
    header = {"v": 1, "kind": "face_revoke", "circle_id": "team", "from": "ana", "to": "bob", "seq": 9,
              "issued_at": "2026-10-12T12:00:00+00:00"}
    body = sealed.sign_revoke(ana, header)
    assert sealed.verify_revoke(body, ana.public) == header and set(body) == {"header", "sig"}
    with pytest.raises(SealError):
        sealed.verify_revoke(body, mal.public)
    with pytest.raises(SealError):
        sealed.verify_revoke({**body, "header": {**header, "to": "mal"}}, ana.public)
    share = sealed.seal(ana, HEADER, SECRET, bob.public)
    with pytest.raises(SealError):  # a signature made for one purpose does nothing for another
        sealed.verify_revoke({"header": share["header"], "sig": share["sig"]}, ana.public)


def test_fingerprints_are_short_stable_and_different_per_box(people):
    ana, bob, _ = people
    print_ = ana.public.fingerprint
    assert len(print_.split()) == 5 and all(len(g) == 4 and set(g) <= set("0123456789abcdef") for g in print_.split())
    assert print_ == BoxKeys.from_bytes(ana.to_bytes()).public.fingerprint != bob.public.fingerprint


def test_the_private_keys_only_come_out_as_bytes_for_the_vault(people):
    ana, bob, _ = people
    raw = ana.to_bytes()
    assert len(raw) == 64 and BoxKeys.from_bytes(raw).public == ana.public
    assert raw.hex() not in repr(ana) and "BoxKeys(" in repr(ana)
    again = BoxKeys.from_bytes(raw)  # the same box after a restart still opens what was sealed to it
    assert sealed.open_sealed(again, sealed.seal(bob, HEADER, SECRET, ana.public), bob.public) == SECRET
    with pytest.raises(SealError):
        BoxKeys.from_bytes(raw[:40])


def test_errors_never_quote_what_they_were_handling(people):
    ana, bob, mal = people
    body = sealed.seal(ana, HEADER, SECRET, bob.public)
    for attempt in (lambda: sealed.open_sealed(mal, body, ana.public), lambda: sealed.open_sealed(bob, body, mal.public),
                    lambda: sealed.open_sealed(bob, {**body, "ciphertext": "not base64"}, ana.public)):
        with pytest.raises(SealError) as caught:
            attempt()
        text = str(caught.value) + repr(caught.value.args)
        assert body["ciphertext"][:16] not in text and "Zebulon" not in text and caught.value.__cause__ is None
