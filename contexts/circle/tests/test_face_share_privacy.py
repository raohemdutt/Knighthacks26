"""Face share's privacy promises, each as a test. What may be in the Circle because of a face is ciphertext, only
in a `face_share` row, only addressed to one member; the plain vectors and the name kept with them are nowhere but
inside the two boxes' own vault blobs: not in a Circle row, an event, a log line, an error or any other file."""

import asyncio
import base64
import json
import os
import logging

import pytest

from contexts.circle.application import face_share as face_share_module
from contexts.circle.application import sealed
from contexts.circle.application.face_share import TEMPLATE_FIELDS, FaceShareError, InMemorySealedStore
from contexts.circle.application.protocol import InMemoryCircleStore
from contexts.circle.infrastructure.outbox import CircleOutbox
from contexts.circle.tests.face_share_helpers import NAMES, TEMPLATE_NAME, Box, Wall, needles, share_and_accept
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.infrastructure.store import MemoryStore
from contexts.perception.application.identity import shared_member_id
from contexts.perception.infrastructure.identity_store import MAX_EXEMPLARS, Member
from contexts.perception.infrastructure.recognizer import FakeRecognizer
from contexts.vault import KeyRing, Vault
from contracts.circle import SealedMessage

SHARE_BODY = {"header", "eph_pub", "nonce", "ciphertext", "sig"}
SHARE_HEADER = {"v", "kind", "circle_id", "from", "to", "share_id", "seq", "issued_at", "expires_at", "model"}
REVOKE_HEADER = {"v", "kind", "circle_id", "from", "to", "seq", "issued_at"}
KEY_HEADER = {"v", "kind", "circle_id", "member_id", "sign_pub", "enc_pub", "created_at"}
BINARY = ("eph_pub", "nonce", "ciphertext", "sig")


class Recorded(InMemorySealedStore):
    """The in-memory store, remembering every row ever written to it (also the ones later deleted)."""

    def __init__(self) -> None:
        super().__init__()
        self.ever: list[SealedMessage] = []

    def post(self, msg: SealedMessage) -> None:
        self.ever.append(msg.model_copy(deep=True))
        super().post(msg)


def found_in(hay: dict[str, bytes], data: bytes) -> list[str]:
    return [what for what, needle in hay.items() if needle in data]


def test_the_template_is_nowhere_in_the_clear_outside_the_two_vaults(tmp_path, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    store, wall, plans = Recorded(), Wall(), InMemoryCircleStore()
    ana, bob, mal = (Box(tmp_path, m, store, wall) for m in ("ana", "bob", "mal"))
    for box in (ana, bob):  # Memory and the Circle's own outbox, as on a real box
        MemoryProjection(MemoryStore(box.dir / "memory.sqlite")).subscribe(box.bus)
    outbox = CircleOutbox(bob.dir / "circle.sqlite")
    ana.enrol_self()
    bob.identity.set_enabled(True)

    # The whole life of a share: keys, card, send, card, accept, recognised, renewed, a failed open, forgotten, revoked.
    share_and_accept(ana, bob)
    assert bob.sees(b"ana") == {7: shared_member_id("ana")}
    mal.poll(), ana.poll()
    ana.ask_to_share("mal"), ana.decide(), mal.poll()  # a second friend: its own row, its own ciphertext, a card never answered
    wall.pass_(days=10)
    ana.ask_to_share("bob"), ana.decide(), bob.poll()
    renewal = next(r for r in store.rows if r.kind == "face_share" and r.to_member == "bob")
    store.rows[store.rows.index(renewal)] = renewal.model_copy(update={"body": {**renewal.body, "nonce": sealed.b64(b"\x01" * 12)}})
    assert [r.status for r in bob.decide()] == ["failed"]  # an error path: what does the failure say?
    store.rows[[r.id for r in store.rows].index(renewal.id)] = renewal
    bob.publish([bob.share.review("ana")])
    assert [r.status for r in bob.decide()] == ["done"]
    bob.publish(bob.share.forget("ana"))
    ana.publish(ana.share.revoke("bob"))
    ana.publish(ana.share.revoke("mal"))
    bob.poll(), mal.poll(), ana.poll()

    mine = ana.identity.store.get("m-ana")
    hay = needles(TEMPLATE_NAME["ana"], [mine.vector, *mine.vectors])
    assert len(hay) > 60

    # Positive control: the same search finds all of it inside a sealed share once bob's box opens it, so the
    # scan can see; and what was inside is the template and nothing else.
    shares = [r for r in store.ever if r.kind == "face_share"]
    assert [(r.from_member, r.to_member) for r in shares] == [("ana", "bob"), ("ana", "mal"), ("ana", "bob")]
    plain = sealed.open_sealed(bob.share._box(), shares[0].body, ana.share._box().public)
    assert set(found_in(hay, plain)) >= {"name", "vector 0 as compact JSON", "vector 1 as compact JSON", "vector 0[0] as text"}
    inside = json.loads(plain)
    assert set(inside) == TEMPLATE_FIELDS == {"v", "circle_member_id", "name", "model", "vector", "vectors"}
    assert (inside["circle_member_id"], inside["vector"], inside["vectors"]) == ("ana", mine.vector, mine.vectors)
    with pytest.raises(sealed.SealError):  # mal's box cannot open bob's, and bob's cannot open mal's
        sealed.open_sealed(mal.share._box(), shares[0].body, ana.share._box().public)
    with pytest.raises(sealed.SealError):
        sealed.open_sealed(bob.share._box(), shares[1].body, ana.share._box().public)

    # 1. The Circle store: every row ever written.
    assert {r.kind for r in store.ever} == {"box_key", "face_share", "face_revoke"}
    for r in store.ever:
        text = r.model_dump_json().encode()
        assert not found_in(hay, text), f"row {r.kind} {r.id} holds {found_in(hay, text)} in plain text"
        assert r.circle_id == "team" and r.v == 1
        if r.kind == "face_share":
            # the only identity-derived payload, addressed to exactly one member who is not the sender
            assert r.to_member in NAMES and r.to_member != r.from_member == "ana"
            assert set(r.body) == SHARE_BODY and set(r.body["header"]) == SHARE_HEADER
            assert r.body["header"]["to"] == r.to_member and r.body["header"]["from"] == "ana"
            for field in BINARY:
                raw = base64.b64decode(r.body[field], validate=True)
                assert not found_in(hay, raw), f"{field} of {r.id} holds {found_in(hay, raw)}"
            assert len(base64.b64decode(r.body["ciphertext"])) < 64 * 1024
        elif r.kind == "face_revoke":
            assert r.to_member in NAMES and set(r.body) == {"header", "sig"} and set(r.body["header"]) == REVOKE_HEADER
            assert len(text) < 1024
        else:  # box_key: two public keys and a signature, for everyone
            assert r.to_member is None and set(r.body) == {"header", "sig"} and set(r.body["header"]) == KEY_HEADER
            assert len(text) < 1024
    # no two shares carry the same bytes, also not the two to bob
    assert len({r.body["ciphertext"] for r in shares}) == 3 and len({r.body["eph_pub"] for r in shares}) == 3
    # and nothing of it went the Circle's other ways out: the plans' messages and the summaries' outbox
    assert plans.messages == [] and outbox.pending() == [] and outbox.stats() == []
    assert [r.kind for r in store.rows] == ["box_key"] * 3  # in the end only the public keys are left

    # 2. Every file under all three data directories: the event logs, Memory, the vault blobs, the keys.
    files = [p for p in sorted(tmp_path.rglob("*")) if p.is_file()]
    logs = [p for p in files if p.parent.name == "events"]
    assert {p.parent.parent.name for p in logs} == {"ana", "bob", "mal"} and any(p.name == "face_share.bin" for p in files) and any(p.suffix == ".face" for p in files)
    scanned = 0
    for path in files:
        data = path.read_bytes()
        scanned += len(data)
        assert not found_in(hay, data), f"{path.relative_to(tmp_path)} holds {found_in(hay, data)} in plain text"
        for row in shares:  # the ciphertext is not kept anywhere either: not in a log, not in the state file
            assert row.body["ciphertext"].encode() not in data and row.body["ciphertext"][:64].encode() not in data
    logged = [json.loads(line) for p in logs for line in p.read_text().splitlines()]
    types = {e["type"] for e in logged}
    assert {"circle.face_share.changed", "agent.action.proposed", "workspace.action.completed",
            "perception.identity.changed", "perception.person.identified"} <= types
    for e in logged:
        if e["type"] == "circle.face_share.changed":
            assert set(e) == {"event_id", "type", "version", "ts", "circle_id", "member_id", "peer", "direction", "state",
                              "share_id", "expires_at", "reason"}
        if e["type"] == "agent.action.proposed" and e["params"].get("kind", "").startswith("face_"):
            assert set(e["params"]) in ({"kind", "to", "fingerprint"}, {"kind", "from", "share_id"})  # ids and a fingerprint
    states = [(e["member_id"], e["direction"], e["peer"], e["state"]) for e in logged if e["type"] == "circle.face_share.changed"]
    assert ("bob", "in", "ana", "received") in states and ("bob", "in", "ana", "forgotten") in states
    assert ("ana", "out", "bob", "revoked") in states and ("mal", "in", "ana", "revoked") in states
    failed = [e["result"] for e in logged if e["type"] == "workspace.action.completed" and e["status"] == "failed"]
    assert failed == ["the signature does not match the key this box knows for that member"]

    # 3. Log lines, and anything printed.
    said = "\n".join(r.getMessage() for r in caplog.records).encode() + capsys.readouterr().out.encode()
    assert not found_in(hay, said) and not any(row.body["ciphertext"][:64].encode() in said for row in shares)

    # 4. And the vault blobs are where it IS: ana's own template on her box (bob forgot his copy).
    blob = (ana.dir / "vault" / "identity" / "m-ana.face").read_bytes()
    assert "name" in found_in(hay, Vault(KeyRing(ana.dir)).decrypt("identity", blob))
    assert not (bob.dir / "vault" / "identity" / f"{shared_member_id('ana')}.face").exists()

    with capsys.disabled():
        print(f"\n  face share leak scan: {len(store.ever)} Circle rows ever written ({len(shares)} sealed shares), "
              f"{len(files)} files in 3 boxes ({scanned} bytes: {len(logs)} event log files, Memory, vault blobs, keys), "
              f"{len(logged)} events, {len(caplog.records)} log lines, {len(hay)} needles (the name, "
              f"{len(mine.vectors) + 1} vectors in 10 encodings): none in plain text; found only inside the opened envelope "
              f"and the owner's vault blob")


def test_while_it_sits_on_the_friends_box_it_is_a_vault_blob_only_that_box_can_open(tmp_path):
    store, wall = InMemorySealedStore(), Wall()
    ana, bob = Box(tmp_path, "ana", store, wall), Box(tmp_path, "bob", store, wall)
    ana.enrol_self()
    share_and_accept(ana, bob)
    path = bob.dir / "vault" / "identity" / f"{shared_member_id('ana')}.face"
    mine = ana.identity.store.get("m-ana")
    hay = needles(TEMPLATE_NAME["ana"], [mine.vector, *mine.vectors])
    blob = path.read_bytes()
    assert not found_in(hay, blob)
    if os.name == "posix":  # Windows has no owner-only mode
        assert path.stat().st_mode & 0o777 == 0o600
    kept = json.loads(Vault(KeyRing(bob.dir)).decrypt("identity", blob))
    assert (kept["source"], kept["shared_by"], kept["circle_member_id"], kept["vector"]) == ("circle", "ana", "ana", mine.vector)
    assert set(kept) == {f.name for f in Member.__dataclass_fields__.values()} and kept["expressions"] is False
    with pytest.raises(Exception):
        Vault(KeyRing(ana.dir)).decrypt("identity", blob)  # another box's vault key does not open it


def test_only_vectors_and_a_name_leave_and_the_size_is_bounded(tmp_path, monkeypatch):
    store, wall = Recorded(), Wall()
    ana, bob = Box(tmp_path, "ana", store, wall), Box(tmp_path, "bob", store, wall)
    ana.enrol_self()
    # The largest template a box can hold: twelve exemplars of SFace's 128 numbers, a 60-character name.
    fake, old = FakeRecognizer(dims=128), ana.identity.store.get("m-ana")
    vectors = [fake.embed(b"ana#%d" % i) for i in range(MAX_EXEMPLARS)]
    ana.identity.store.save(Member(**{**old.__dict__, "name": "N" * 60, "vector": vectors[0], "vectors": vectors, "expressions": True}))
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob")
    assert [r.status for r in ana.decide()] == ["done"]
    row = store.ever[-1]
    plain = sealed.open_sealed(bob.share._box(), row.body, ana.share._box().public)
    assert 30_000 < len(plain) < 60_000 < face_share_module.MAX_PLAINTEXT  # about 37 kB at the most
    assert len(sealed.canonical(row.body)) < 100_000 < face_share_module.MAX_SEALED
    sent = json.loads(plain)
    assert set(sent) == TEMPLATE_FIELDS and len(sent["vectors"]) == MAX_EXEMPLARS
    for word in (b"expressions", b"photo", b"frame", b"enrolled_at", b"consent", b"member_id\"", b"m-ana"):
        assert word not in plain.replace(b"circle_member_id", b"")  # no history, no picture, not even the local id

    monkeypatch.setattr(face_share_module, "MAX_PLAINTEXT", 10_000)
    ana.ask_to_share("bob")
    done = ana.decide()
    assert done[0].status == "failed" and done[0].result == "the template is too large to share"
    assert len([r for r in store.ever if r.kind == "face_share"]) == 1


def test_an_approved_card_that_fails_never_says_what_it_was_handling(tmp_path):
    store, wall = InMemorySealedStore(), Wall()
    ana, bob = Box(tmp_path, "ana", store, wall), Box(tmp_path, "bob", store, wall)
    ana.enrol_self()
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide(), bob.poll()
    mine = ana.identity.store.get("m-ana")

    def explode(*a, **kw):  # a bug somewhere below, whose message quotes the template
        raise RuntimeError(f"could not store {kw!r} {a!r}")

    bob.identity.receive_shared = explode
    done = bob.decide()
    assert done[0].status == "failed" and done[0].result == "the face share could not be completed: nothing was changed"
    hay = needles(TEMPLATE_NAME["ana"], [mine.vector, *mine.vectors])
    for path in (p for p in bob.dir.rglob("*") if p.is_file()):
        assert not found_in(hay, path.read_bytes())
    with pytest.raises(FaceShareError) as caught:
        asyncio.run(bob.share.executor(type("Req", (), {"params": {"kind": "face_accept", "from": "ana", "share_id": "nope"}})()))
    assert caught.value.__cause__ is None and "no longer there" in str(caught.value)
