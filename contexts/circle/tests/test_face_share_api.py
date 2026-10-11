"""The face share endpoints: 409 with a plain reason while it is off, cards (not sends) when it is on, and no answer
that holds a template, a key or ciphertext. In-memory store only."""

import json

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from contexts.circle.application.face_share import EXPLANATION, InMemorySealedStore
from contexts.circle.application.sharing import CircleOffline
from contexts.circle.face_share_api import face_share_router
from contexts.circle.tests.face_share_helpers import TEMPLATE_NAME, Box, Wall, needles, share_and_accept

BASE = "/api/circle/face-share"
WRITES = ["/bob/share", "/bob/revoke", "/bob/review", "/bob/forget", "/bob/trust-key"]


def http(box: Box | None, token: str | None = None, share=..., bus=None) -> TestClient:
    def guard(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(face_share_router(box.share if share is ... else share, bus or (box.bus if box else None), guard=guard))
    return TestClient(app)


@pytest.fixture
def two(tmp_path):
    store, wall = InMemorySealedStore(), Wall()
    ana, bob = Box(tmp_path, "ana", store, wall), Box(tmp_path, "bob", store, wall)
    ana.enrol_self()
    bob.identity.set_enabled(True)
    return ana, bob, store


def every_call(c: TestClient):
    yield c.get(BASE)
    for path in WRITES:
        yield c.post(BASE + path, json={"fingerprint": "abcd"} if path.endswith("trust-key") else None)


def test_switched_off_every_endpoint_answers_409_with_the_plain_reason(tmp_path):
    store = InMemorySealedStore()
    off = Box(tmp_path, "ana", store, Wall(), enabled=False)
    for r in every_call(http(off)):
        assert r.status_code == 409, r.request.url
        assert r.json()["detail"].startswith("face sharing between friends' boxes is switched off on this box")
        assert "circle.face_share" in r.json()["detail"] and "ATLAS_CIRCLE_FACE_SHARE=1" in r.json()["detail"]
    assert store.rows == [] and off.events == [] and not (off.dir / "vault" / "circle").exists()


def test_without_a_circle_every_endpoint_answers_409_too(tmp_path):
    for r in every_call(http(None, share=None)):
        assert r.status_code == 409 and r.json()["detail"] == "this box has not joined a Circle"


def test_the_status_says_who_can_recognise_whom_and_nothing_about_a_face(two):
    ana, bob, _ = two
    share_and_accept(ana, bob)
    mine, theirs = http(ana).get(BASE), http(bob).get(BASE)
    assert mine.status_code == theirs.status_code == 200
    body = mine.json()
    assert set(body) == {"enabled", "circle_id", "member_id", "expiry_days", "recognition_enabled", "recognition_available",
                         "me", "friends", "explanation"}
    assert body["explanation"] == EXPLANATION == ("Your face template goes only to the friends you choose, encrypted so only "
                                                  "their box can read it. You can take it back at any time.")
    assert (body["enabled"], body["circle_id"], body["member_id"], body["expiry_days"]) == (True, "team", "ana", 30)
    assert body["me"]["can_share"] and body["me"]["member"] == {"member_id": "m-ana", "name": TEMPLATE_NAME["ana"]}
    bob_row, mal_row = body["friends"]
    assert set(bob_row) == {"member_id", "display_name", "key", "outgoing", "incoming", "renewal_offered"}
    assert (bob_row["display_name"], bob_row["outgoing"]["state"], bob_row["incoming"]["state"]) == ("Bob", "shared", "none")
    assert bob_row["outgoing"]["expires_at"] == "2026-11-10T12:00:00+00:00"
    assert mal_row["key"] == {"known": False, "fingerprint": None, "changed": False, "new_fingerprint": None}
    ana_row = theirs.json()["friends"][0]
    assert (ana_row["member_id"], ana_row["incoming"]["state"], ana_row["incoming"]["usable"]) == ("ana", "accepted", True)
    # each side shows the other's fingerprint and its own: the two the friends compare
    assert bob_row["key"]["fingerprint"] == theirs.json()["me"]["fingerprint"]
    assert ana_row["key"]["fingerprint"] == body["me"]["fingerprint"]

    stored = bob.identity.store.members()[0]
    hay = needles(TEMPLATE_NAME["ana"], [stored.vector, *stored.vectors])
    for name, text in (("ana's status", mine.text), ("bob's status", theirs.text)):
        found = [what for what, needle in hay.items() if needle in text.encode()]
        allowed = ["name", "first name", "last name"] if name == "ana's status" else []  # her own name, on her own box only
        assert found == allowed, f"{name} holds {found}"
        for word in ("vector", "ciphertext", "eph_pub", "sign_pub", "enc_pub", "nonce", '"sig"', "keys"):
            assert word not in text


def test_share_is_a_card_and_revoke_review_forget_do_what_they_say(two):
    ana, bob, store = two
    a, b = http(ana), http(bob)
    r = a.post(BASE + "/bob/share")
    assert r.status_code == 409 and r.json()["detail"].startswith("Bob's box has not published a key yet")
    ana.poll(), bob.poll(), ana.poll()

    r = a.post(BASE + "/bob/share")
    assert r.status_code == 200 and set(r.json()) == {"action_id"}
    assert [c.id for c in ana.cards()] == [r.json()["action_id"]] and len(store.rows) == 2  # a card; nothing sent
    assert a.get(BASE).json()["friends"][0]["outgoing"] == {"state": "waiting_approval", "share_id": None, "shared_at": None,
                                                           "expires_at": None, "action_id": r.json()["action_id"]}
    ana.decide(), bob.poll()
    assert b.get(BASE).json()["friends"][0]["incoming"]["state"] == "offered"
    assert [c.status for c in bob.decide("deny")] == ["denied"]
    r = b.post(BASE + "/ana/review")
    assert r.status_code == 200 and [c.id for c in bob.cards()] == [r.json()["action_id"]]
    bob.decide()
    assert b.get(BASE).json()["friends"][0]["incoming"]["state"] == "accepted"

    assert b.post(BASE + "/ana/forget").json() == {"forgotten": "ana"} and bob.identity.members() == []
    assert b.post(BASE + "/ana/forget").status_code == 404
    assert a.post(BASE + "/bob/revoke").json() == {"revoked": "bob"}
    assert a.post(BASE + "/bob/revoke").status_code == 404
    assert [r.kind for r in store.rows] == ["box_key", "box_key", "face_revoke"]
    assert ana.states() == [("out", "bob", "shared"), ("out", "bob", "revoked")]
    assert bob.states("in")[-1] == ("in", "ana", "forgotten")


def test_unknown_friends_bad_bodies_and_refusals(two):
    ana, bob, _ = two
    ana.poll(), bob.poll(), ana.poll()
    a = http(ana)
    for path in ("/zed/share", "/ana/share", "/bob/review", "/bob/forget", "/bob/revoke", "/zed/revoke"):
        assert a.post(BASE + path).status_code == 404, path
    assert a.post(BASE + "/bob/trust-key", json={"fingerprint": "abcd"}).status_code == 404  # no changed key to trust
    assert a.post(BASE + "/bob/trust-key", json={}).status_code == 422
    assert a.post(BASE + "/bob/trust-key", json={"fingerprint": "abcd", "more": 1}).status_code == 422
    r = http(bob).post(BASE + "/ana/share")  # bob is not enrolled on his own box
    assert r.status_code == 409 and "nobody enrolled on this box is linked to you" in r.json()["detail"]


def test_writes_need_the_token_and_reads_do_not(two):
    ana, bob, _ = two
    ana.poll(), bob.poll(), ana.poll()
    c, auth = http(ana, token="s3cret"), {"Authorization": "Bearer s3cret"}
    assert c.get(BASE).status_code == 200
    for path in WRITES:
        assert c.post(BASE + path, json={"fingerprint": "x"}).status_code == 401, path
    assert c.post(BASE + "/bob/share", headers=auth).status_code == 200


def test_an_offline_circle_is_a_503_not_a_traceback(two):
    ana, bob, _ = two
    share_and_accept(ana, bob)

    class Offline:
        def post(self, msg):
            raise CircleOffline("Neon is unreachable (ConnectionError)")

    ana.share.store = Offline()
    r = http(ana).post(BASE + "/bob/revoke")
    assert r.status_code == 503 and r.json()["detail"] == "the Circle is offline: Neon is unreachable (ConnectionError)"
    assert ana.share.status()["friends"][0]["outgoing"]["state"] == "shared"  # still there to take back when it is reachable


def test_no_answer_and_no_card_ever_holds_the_template(two):
    ana, bob, _ = two
    ana.poll(), bob.poll(), ana.poll()
    a, b = http(ana), http(bob)
    answers = [a.post(BASE + "/bob/share")]
    ana.decide(), bob.poll(), bob.decide()
    answers += [a.get(BASE), b.get(BASE), b.post(BASE + "/ana/forget"), a.post(BASE + "/bob/revoke"), a.get(BASE), b.get(BASE)]
    mine = ana.identity.store.get("m-ana")
    hay = {k: v for k, v in needles(TEMPLATE_NAME["ana"], [mine.vector, *mine.vectors]).items() if "name" not in k}
    for r in answers:
        assert r.status_code == 200 and not [what for what, needle in hay.items() if needle in r.content]
    cards = json.dumps([e.model_dump(mode="json") for box in (ana, bob) for e in box.events
                        if e.type in ("agent.action.proposed", "workspace.approval.requested", "workspace.action.completed")])
    assert cards.count("face_share") == 1 and cards.count("face_accept") == 1
    assert not [what for what, needle in needles(TEMPLATE_NAME["ana"], [mine.vector, *mine.vectors]).items() if needle in cards.encode()]
