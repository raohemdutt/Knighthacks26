"""Face share between two boxes (three when someone has to misbehave), on one in-memory sealed store. Each box has
its own data folder, vault, identity service and approvals; they never touch each other's objects."""

import asyncio
import copy
import json
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest

from contexts.agent.application.memory_port import FakeMemory
from contexts.circle.application import sealed
from contexts.circle.application.face_share import FaceShare, FaceShareError, FaceShareOff, InMemorySealedStore
from contexts.circle.application.protocol import CircleAgent, InMemoryCircleStore
from contexts.circle.tests.face_share_helpers import NAMES, TEMPLATE_NAME, Box, Down, Wall, share_and_accept
from contexts.perception.application.identity import shared_member_id
from contexts.perception.tests.identity_helpers import Frame
from contexts.vault import Wiper
from contexts.vault.tests.helpers import T0
from contracts.circle import SealedMessage
from contracts.vault import WipeRequested

ANA_AT_BOB = shared_member_id("ana")  # the id bob's box keeps ana's template under


@pytest.fixture
def world(tmp_path):
    store, wall = InMemorySealedStore(), Wall()
    return SimpleNamespace(store=store, wall=wall, root=tmp_path,
                           box=lambda member, **kw: Box(tmp_path, member, store, wall, **kw))


@pytest.fixture
def pair(world):
    ana, bob = world.box("ana"), world.box("bob")
    ana.enrol_self()
    bob.identity.set_enabled(True)
    world.ana, world.bob = ana, bob
    return world


def kinds(store):
    return [(r.kind, r.from_member, r.to_member) for r in store.rows]


# --- the two-box flow ---


def test_share_card_accept_card_and_then_the_friends_box_recognises_them(pair):
    ana, bob, store = pair.ana, pair.bob, pair.store
    assert bob.sees(b"ana") == {}  # before anything: ana is a stranger to bob's box
    ana.poll(), bob.poll(), ana.poll()
    assert kinds(store) == [("box_key", "ana", None), ("box_key", "bob", None)]  # public keys only so far

    # 1. ana asks. A card on HER box; nothing has left it.
    card = ana.ask_to_share("bob")
    assert card.kind == "circle_share" and card.params == {"kind": "face_share", "to": "bob",
                                                           "fingerprint": bob.share.status()["me"]["fingerprint"]}
    assert "Let Bob's Atlas recognise you" in card.reason and "until Nov 10" in card.reason
    assert [c.id for c in ana.cards()] == [card.action_id] and len(store.rows) == 2
    assert ana.share.status()["friends"][0]["outgoing"]["state"] == "waiting_approval"
    bob.poll()
    assert bob.cards() == [] and bob.identity.members() == []

    # 2. ana approves: one sealed row, addressed to bob alone.
    done = ana.decide()
    assert done[0].status == "done" and "can recognise you until Nov 10 once they accept" in done[0].result
    assert kinds(store)[2:] == [("face_share", "ana", "bob")]
    assert ana.states() == [("out", "bob", "shared")]

    # 3. bob's box raises its own card. Still nothing decrypted or stored there.
    events = bob.poll()
    offer = bob.cards()
    assert len(offer) == 1 and offer[0].reason.startswith("Ana wants your Atlas to recognise them. Accept?")
    assert offer[0].params == {"kind": "face_accept", "from": "ana", "share_id": store.rows[2].id}
    assert bob.identity.members() == [] and bob.sees(b"ana") == {}
    assert [e.type for e in events] == ["circle.face_share.changed", "agent.action.proposed"]
    assert bob.poll() == []  # the same offer does not raise a second card
    friend = bob.share.status()["friends"][0]
    assert (friend["member_id"], friend["incoming"]["state"], friend["outgoing"]["state"]) == ("ana", "offered", "none")

    # 4. bob accepts: decrypted, stored in bob's own vault as a member that says where it came from.
    done = bob.decide()
    assert done[0].status == "done" and done[0].result == "Ana will be recognised by this box until Nov 10"
    [member] = bob.identity.members()
    assert member == {"member_id": ANA_AT_BOB, "name": TEMPLATE_NAME["ana"], "enrolled_at": T0.isoformat(),
                      "updated_at": T0.isoformat(), "photos": 5, "source": "circle", "expressions": False,
                      "circle_member_id": "ana", "shared_by": "ana", "received_at": T0.isoformat(),
                      "expires_at": (T0 + timedelta(days=30)).isoformat()}
    assert bob.states() == [("in", "ana", "offered"), ("in", "ana", "received")]
    changed = [e for e in bob.events if e.type == "perception.identity.changed"][-1]
    assert changed.received_from == ["ana"] and changed.members == [ANA_AT_BOB]
    assert kinds(store) == [("box_key", "ana", None), ("box_key", "bob", None)]  # the ciphertext is gone from the Circle

    # 5. ana walks up to bob's box, where she never enrolled.
    assert bob.sees(b"ana") == {7: ANA_AT_BOB}
    seen = [e for e in bob.events if e.type == "perception.person.identified"]
    assert seen and seen[0].member_id == ANA_AT_BOB and seen[0].circle_member_id == "ana"
    both = ana.share.status()["friends"][0], bob.share.status()["friends"][0]
    assert both[0]["outgoing"]["state"] == "shared" and both[1]["incoming"] == {
        "state": "accepted", "received_at": T0.isoformat(), "expires_at": (T0 + timedelta(days=30)).isoformat(),
        "offered_at": None, "usable": True}
    assert both[0]["key"]["fingerprint"] == bob.share.status()["me"]["fingerprint"]  # what the two friends compare
    assert both[1]["key"]["fingerprint"] == ana.share.status()["me"]["fingerprint"]


def test_a_stranger_stays_a_stranger_on_the_friends_box(pair):
    share_and_accept(pair.ana, pair.bob)
    assert pair.bob.sees(b"mal") == {}


def test_declining_stores_nothing_and_the_same_offer_does_not_come_back(pair):
    ana, bob = pair.ana, pair.bob
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide()
    bob.poll()
    assert [r.status for r in bob.decide("deny")] == ["denied"]
    assert bob.identity.members() == [] and bob.poll() == []  # denied: still waiting, no new card on its own
    again = bob.share.review("ana")  # the screen can raise the card again ...
    bob.publish([again])
    assert again.params["kind"] == "face_accept" and [c.id for c in bob.cards()] == [again.action_id]
    bob.publish(bob.share.forget("ana"))  # ... or turn the offer down for good
    assert bob.states()[-1] == ("in", "ana", "declined") and pair.store.rows[2:] == []
    assert bob.poll() == [] and bob.share.status()["friends"][0]["incoming"]["state"] == "none"
    with pytest.raises(KeyError):
        bob.share.review("ana")
    assert [r.status for r in bob.decide()] == ["failed"] and bob.identity.members() == []  # a card left over opens nothing


def test_nothing_is_sent_while_the_card_is_only_waiting_or_denied(pair):
    ana, bob = pair.ana, pair.bob
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob")
    assert [r.status for r in ana.decide("deny")] == ["denied"]
    assert kinds(pair.store) == [("box_key", "ana", None), ("box_key", "bob", None)]
    bob.poll()
    assert bob.cards() == [] and ana.share.status()["friends"][0]["outgoing"]["state"] in ("none", "waiting_approval")


# --- only one's own template ---


def test_a_box_cannot_share_someone_who_merely_enrolled_on_it(world):
    ana, bob = world.box("ana"), world.box("bob")
    ana.rig.enroll("m-ben", "Ben Visitor", b"ben")  # a visitor enrolled on ana's box, with consent, as nobody in the Circle
    ana.poll(), bob.poll(), ana.poll()
    with pytest.raises(FaceShareError, match="nobody enrolled on this box is linked to you"):
        ana.ask_to_share("bob")
    ana.identity.update_member("m-ben", circle_member_id="bob")  # linked to a Circle member, but not THIS box's member
    with pytest.raises(FaceShareError, match="nobody enrolled"):
        ana.ask_to_share("bob")
    assert ana.cards() == [] and ana.share.status()["me"] == {
        "fingerprint": ana.share.status()["me"]["fingerprint"], "member": None, "can_share": False,
        "why_not": ana.share.status()["me"]["why_not"]}
    ana.enrol_self()
    assert ana.share.status()["me"]["member"] == {"member_id": "m-ana", "name": TEMPLATE_NAME["ana"]}
    ana.ask_to_share("bob"), ana.decide(), bob.poll(), bob.decide()
    [got] = bob.identity.store.members()
    mine = ana.identity.store.get("m-ana")
    assert (got.vector, got.vectors, got.name) == (mine.vector, mine.vectors, mine.name)  # ana's, not the visitor's
    assert got.vector != ana.identity.store.get("m-ben").vector


def test_two_people_linked_as_the_box_owner_is_refused_rather_than_guessed(world):
    ana, bob = world.box("ana"), world.box("bob")
    ana.enrol_self()
    ana.rig.enroll("m-ben", "Ben Visitor", b"ben")
    ana.identity.update_member("m-ben", circle_member_id="ana")
    ana.poll(), bob.poll(), ana.poll()
    with pytest.raises(FaceShareError, match="more than one"):
        ana.ask_to_share("bob")


def test_a_received_template_cannot_be_shared_on_or_changed(pair, world):
    ana, bob = pair.ana, pair.bob
    mal = world.box("mal")
    share_and_accept(ana, bob)
    mal.poll(), bob.poll()
    with pytest.raises(FaceShareError, match="nobody enrolled"):  # bob himself is not enrolled: ana's template is not his
        bob.ask_to_share("mal")
    # Even with the link forced to say "this is bob", a template that came from the Circle is never handed out.
    stored = bob.identity.store.get(ANA_AT_BOB)
    bob.identity.store.save(stored.__class__(**{**stored.__dict__, "circle_member_id": "bob"}))
    with pytest.raises(FaceShareError, match="nobody enrolled"):
        bob.ask_to_share("mal")
    for change in (lambda: bob.identity.update_member(ANA_AT_BOB, name="Someone Else"),
                   lambda: bob.identity.update_member(ANA_AT_BOB, circle_member_id="bob"),
                   lambda: bob.identity.start_enrollment(ANA_AT_BOB, "Someone Else", consent_ack=True)):
        with pytest.raises(Exception, match="shared by a friend's box"):
            change()


def test_a_member_cannot_pass_off_a_template_as_someone_elses(world):
    ana, bob, mal = world.box("ana"), world.box("bob"), world.box("mal")
    mal.enrol_self()
    for box in (ana, bob, mal, ana, bob):
        box.poll()
    # 1. mal's box seals mal's face to bob, but the template inside says it is ana.
    mal.identity.own_template = lambda member: {**type(mal.identity).own_template(mal.identity, "mal"), "name": "Ana"}
    real_canonical = sealed.canonical
    try:
        sealed.canonical = lambda v: real_canonical({**v, "circle_member_id": "ana"} if "vectors" in v else v)
        mal.ask_to_share("bob"), mal.decide()
    finally:
        sealed.canonical = real_canonical
    bob.poll()
    assert "Mal wants your Atlas" in bob.cards()[0].reason  # the card names who it is really from
    done = bob.decide()
    assert done[0].status == "failed" and done[0].result == "the shared template is not well formed: nothing was stored"
    assert bob.identity.members() == []

    # 2. mal writes a row that claims to come from ana (the database lets anyone write any name).
    forged = sealed.seal(mal.share._box(), {"v": 1, "kind": "face_share", "circle_id": "team", "from": "ana", "to": "bob",
                                            "share_id": "forged", "seq": 10 ** 15, "issued_at": T0.isoformat(),
                                            "expires_at": (T0 + timedelta(days=5)).isoformat(), "model": "fake"},
                         b"{}", bob.share._box().public)
    world.store.post(SealedMessage(id="forged", circle_id="team", from_member="ana", to_member="bob", kind="face_share",
                                   body=forged, created_at=world.wall()))
    bob.poll()
    assert bob.states()[-1] == ("in", "ana", "ignored") and bob.events[-1].reason == "bad_signature"
    assert len(bob.cards()) == 0 and bob.identity.members() == []
    assert bob.poll() == []  # reported once


# --- taking it back ---


def test_revoke_deletes_it_on_the_friends_box_at_its_next_poll(pair):
    ana, bob = pair.ana, pair.bob
    share_and_accept(ana, bob)
    assert bob.sees(b"ana") == {7: ANA_AT_BOB}
    ana.publish(ana.share.revoke("bob"))
    assert ana.states()[-1] == ("out", "bob", "revoked") and kinds(pair.store)[2:] == [("face_revoke", "ana", "bob")]
    assert ana.share.status()["friends"][0]["outgoing"]["state"] == "none"
    assert len(bob.identity.members()) == 1  # bob's box was not asked yet (it may be off): nothing changed there
    pair.wall.pass_(days=3)  # ... and it obeys whenever it next polls
    events = bob.poll()
    assert bob.identity.members() == [] and not (bob.dir / "vault" / "identity" / f"{ANA_AT_BOB}.face").exists()
    assert bob.states()[-1] == ("in", "ana", "revoked")
    assert [e.type for e in events] == ["perception.identity.changed", "circle.face_share.changed"] and events[0].received_from == []
    bob.publish(bob.rig.look(Frame(), []))
    assert bob.identity.tagged().get("cam-desk", {}) == {} and kinds(pair.store)[2:] == []  # the tag ended with the template
    with pytest.raises(KeyError):
        ana.share.revoke("bob")


def test_an_old_share_played_back_after_the_revoke_is_refused(pair):
    ana, bob, store = pair.ana, pair.bob, pair.store
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide()
    captured = copy.deepcopy(store.rows[2])  # whoever runs the database keeps a copy of the sealed share
    bob.poll(), bob.decide()
    ana.publish(ana.share.revoke("bob"))
    bob.poll()
    assert bob.identity.members() == []

    store.post(captured)  # ... and puts it back after ana took it back
    events = bob.poll()
    assert [(e.state, e.reason) for e in events] == [("ignored", "replayed")] and bob.cards() == []
    assert bob.identity.members() == [] and bob.poll() == []
    # with a new id and a new date on the row it is still the same signed share, with the same counter
    store.post(captured.model_copy(update={"id": "again", "created_at": pair.wall() + timedelta(hours=1)}))
    assert [(e.state, e.reason) for e in bob.poll()] == [("ignored", "malformed")] and bob.cards() == []
    # a revoke played back does nothing either, and a fresh share by ana works again
    ana.ask_to_share("bob"), ana.decide(), bob.poll()
    assert [r.status for r in bob.decide()] == ["done"] and len(bob.identity.members()) == 1


def test_a_replay_before_any_revoke_cannot_bring_back_what_was_forgotten(pair):
    ana, bob, store = pair.ana, pair.bob, pair.store
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide()
    captured = copy.deepcopy(store.rows[2])
    bob.poll(), bob.decide()
    bob.publish(bob.share.forget("ana"))  # bob's own "forget"
    assert bob.identity.members() == [] and bob.states()[-1] == ("in", "ana", "forgotten")
    store.post(captured)
    assert [(e.state, e.reason) for e in bob.poll()] == [("ignored", "replayed")] and bob.cards() == []


def test_only_the_sharers_key_can_make_a_box_delete_something(pair, world):
    ana, bob = pair.ana, pair.bob
    mal = world.box("mal")
    share_and_accept(ana, bob)
    mal.poll()
    header = {"v": 1, "kind": "face_revoke", "circle_id": "team", "from": "ana", "to": "bob", "seq": 10 ** 15,
              "issued_at": T0.isoformat()}
    world.store.post(SealedMessage(id="fake-revoke", circle_id="team", from_member="ana", to_member="bob", kind="face_revoke",
                                   body=sealed.sign_revoke(mal.share._box(), header), created_at=world.wall()))
    bob.poll()
    assert len(bob.identity.members()) == 1


def test_forgetting_yourself_on_your_own_box_takes_your_shares_back(pair):
    ana, bob = pair.ana, pair.bob
    share_and_accept(ana, bob)
    ana.publish(ana.identity.remove("m-ana"))
    assert ana.states()[-1] != ("out", "bob", "revoked")
    ana.poll()
    assert ana.states()[-1] == ("out", "bob", "revoked")
    bob.poll()
    assert bob.identity.members() == []


# --- running out ---


def test_a_share_expires_and_is_deleted_also_when_the_circle_cannot_be_reached(pair):
    ana, bob = pair.ana, pair.bob
    share_and_accept(ana, bob)
    pair.wall.pass_(days=29, hours=23)
    bob.poll()
    assert len(bob.identity.members()) == 1 and bob.sees(b"ana") == {7: ANA_AT_BOB}

    pair.wall.pass_(hours=2)  # past the thirty days, and the box has not polled yet:
    bob.identity._tags.clear()
    assert bob.sees(b"ana") == {}  # already not used for matching ...
    bob.share.store = down = Down()  # ... and the next poll deletes it, with the Circle unreachable
    with pytest.raises(ConnectionError):
        bob.share.poll()
    assert down.calls == 1 and bob.identity.members() == []
    bob.share.store = pair.store
    events = bob.poll()  # what the failed poll did is said by the next one
    assert [(e.type, getattr(e, "state", None)) for e in events] == [("perception.identity.changed", None),
                                                                      ("circle.face_share.changed", "expired")]
    ana.poll()
    assert ana.states()[-1] == ("out", "bob", "expired") and ana.share.status()["friends"][0]["outgoing"]["state"] == "none"


def test_a_box_that_was_switched_off_for_a_month_deletes_at_start_up(pair):
    ana, bob = pair.ana, pair.bob
    share_and_accept(ana, bob)
    pair.wall.pass_(days=31)
    assert bob.identity.announce()[0].received_from == [] and bob.identity.members() == []  # the gateway calls this at start


def test_renewing_moves_the_expiry_and_needs_the_friends_ok_again(pair):
    ana, bob = pair.ana, pair.bob
    share_and_accept(ana, bob)
    pair.wall.pass_(days=20)
    ana.ask_to_share("bob"), ana.decide()
    assert [r.kind for r in pair.store.rows].count("face_share") == 1
    bob.poll()
    friend = bob.share.status()["friends"][0]
    assert friend["renewal_offered"] and friend["incoming"]["state"] == "accepted"
    assert friend["incoming"]["expires_at"] == (T0 + timedelta(days=30)).isoformat()  # until bob says yes, the old date stands
    assert [r.status for r in bob.decide()] == ["done"]
    assert bob.identity.members()[0]["expires_at"] == (T0 + timedelta(days=50)).isoformat()
    pair.wall.pass_(days=15)
    bob.poll()
    assert len(bob.identity.members()) == 1  # day 35: the first share would have run out
    pair.wall.pass_(days=16)
    bob.poll()
    assert bob.identity.members() == []


def test_an_offer_nobody_answered_runs_out_too(pair):
    ana, bob = pair.ana, pair.bob
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide(), bob.poll()
    pair.wall.pass_(days=31)
    bob.poll()
    assert bob.states()[-1] == ("in", "ana", "expired") and bob.share.status()["friends"][0]["incoming"]["state"] == "none"
    assert pair.store.rows[2:] == [] and bob.poll() == []  # said once, and the ciphertext is gone from the Circle
    assert [r.status for r in bob.decide()] == ["expired"] and bob.identity.members() == []  # the card ran out long ago


def test_a_card_tapped_after_the_offer_was_taken_back_opens_nothing(pair):
    ana, bob = pair.ana, pair.bob
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide(), bob.poll()
    ana.publish(ana.share.revoke("bob"))  # ana changes her mind while bob's card is still on his phone
    done = bob.decide()  # bob taps Accept before his box polled again: the ciphertext is no longer there
    assert done[0].status == "failed" and "no longer there" in done[0].result and bob.identity.members() == []
    bob.poll()
    assert bob.states()[-1] == ("in", "ana", "revoked") and bob.share.status()["friends"][0]["incoming"]["state"] == "none"


# --- forget and wipe ---


def test_wiping_everything_on_the_friends_box_removes_the_shared_template(pair):
    ana, bob = pair.ana, pair.bob
    share_and_accept(ana, bob)
    report = Wiper(bob.dir).wipe("all")
    assert report.identity_removed == 1 and bob.identity.members() == [] and bob.identity.shared_members() == []
    assert asyncio.run(bob.share.on_wipe(WipeRequested(scope="all"))) == []
    assert bob.poll() == [] and bob.share.status()["friends"][0]["incoming"]["state"] == "none"


def test_wiping_everything_on_your_own_box_takes_your_shares_back(pair):
    ana, bob = pair.ana, pair.bob
    share_and_accept(ana, bob)
    assert asyncio.run(ana.share.on_wipe(WipeRequested(scope="keyframes"))) == []
    Wiper(ana.dir).wipe("all")
    events = asyncio.run(ana.share.on_wipe(WipeRequested(scope="all")))
    assert [(e.direction, e.peer, e.state) for e in events] == [("out", "bob", "revoked")]
    bob.poll()
    assert bob.identity.members() == []


def test_the_identity_screens_own_delete_forgets_a_shared_template_for_good(pair):
    ana, bob = pair.ana, pair.bob
    share_and_accept(ana, bob)
    bob.publish(bob.identity.remove(ANA_AT_BOB))  # DELETE /identity/members/{id}
    assert bob.poll() == [] and bob.cards() == [] and bob.identity.members() == []


# --- the switch ---


def test_switched_off_and_never_used_a_box_touches_nothing(world):
    down = Down()
    off = Box(world.root, "bob", down, world.wall, enabled=False)
    assert off.poll() == [] and down.calls == 0
    assert not (off.dir / "vault" / "circle").exists() and not (off.dir / "vault" / "master.key").exists()
    for call in (off.share.status, lambda: off.share.propose_share("ana"), lambda: off.share.revoke("ana"),
                 lambda: off.share.forget("ana"), lambda: off.share.review("ana"), lambda: off.share.trust_key("ana", "x")):
        with pytest.raises(FaceShareOff, match="switched off"):
            call()
    assert asyncio.run(off.share.on_wipe(WipeRequested(scope="all"))) == [] and down.calls == 0


def test_a_box_that_has_it_off_publishes_no_key_so_nobody_can_share_with_it(world):
    ana, bob = world.box("ana"), world.box("bob", enabled=False)
    ana.enrol_self()
    ana.poll(), bob.poll(), ana.poll()
    assert kinds(world.store) == [("box_key", "ana", None)]
    with pytest.raises(FaceShareError, match="Bob's box has not published a key yet"):
        ana.ask_to_share("bob")


def test_switching_it_off_takes_everything_back_and_ignores_what_still_arrives(world, pair):
    ana, bob, store = pair.ana, pair.bob, pair.store
    bob.enrol_self()
    share_and_accept(ana, bob)
    share_and_accept(bob, ana)
    assert len(bob.identity.members()) == 2 and len(ana.identity.members()) == 2

    bob.share.enabled = False  # the one switch, at bob's box (in the app: the config or the env, and a restart)
    events = bob.poll()
    assert [(e.direction, e.peer, e.state) for e in events if e.type == "circle.face_share.changed"] == [
        ("in", "ana", "forgotten"), ("out", "ana", "revoked")]
    assert [m["member_id"] for m in bob.identity.members()] == ["m-bob"]  # ana's template is gone, bob's own enrolment stays
    assert ("box_key", "bob", None) not in kinds(store)  # and bob's key is no longer published
    ana.poll()
    assert [m["member_id"] for m in ana.identity.members()] == ["m-ana"]  # bob's template left ana's box

    ana.ask_to_share("bob"), ana.decide()  # ana still knows bob's key and tries again
    events = bob.poll()
    assert [(e.state, e.reason) for e in events] == [("ignored", "switched_off")] and bob.cards() == []
    assert bob.poll() == []  # reported once
    assert len(bob.identity.members()) == 1

    bob.share.enabled = True  # switched on again: the share that was waiting is offered now
    bob.poll()
    assert "Ana wants your Atlas" in bob.cards()[0].reason


def test_recognition_off_on_the_friends_box_the_card_says_so_and_it_is_not_used(pair):
    ana, bob = pair.ana, pair.bob
    bob.identity.set_enabled(False)
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide(), bob.poll()
    assert "Recognition is switched off on this box, so it will not be used until you switch it on." in bob.cards()[0].reason
    done = bob.decide()
    assert done[0].result == "Ana will be recognised by this box until Nov 10 once recognition is switched on"
    assert bob.share.status()["friends"][0]["incoming"]["usable"] is False and bob.share.status()["recognition_enabled"] is False
    assert bob.sees(b"ana") == {}
    bob.identity.set_enabled(True)
    assert bob.sees(b"ana") == {7: ANA_AT_BOB}


def test_a_box_without_the_face_model_can_hold_the_card_but_says_what_it_cannot_do(world):
    ana, bob = world.box("ana"), world.box("bob", recognizer=None)
    ana.enrol_self()
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide(), bob.poll()
    assert "Recognition is switched off on this box" in bob.cards()[0].reason
    assert bob.share.status()["recognition_available"] is False


# --- keys: trust on first use ---


def test_a_changed_key_is_never_used_silently(pair, world):
    ana, bob, store = pair.ana, pair.bob, pair.store
    ana.poll(), bob.poll(), ana.poll()
    first = ana.share.status()["friends"][0]["key"]
    assert first == {"known": True, "fingerprint": bob.share.status()["me"]["fingerprint"], "changed": False,
                     "new_fingerprint": None}
    # Someone who can write to the database publishes another key under bob's name.
    other = sealed.BoxKeys.generate()
    store.post(SealedMessage(id="key:bob:other", circle_id="team", from_member="bob", kind="box_key",
                             body=sealed.key_record(other, "team", "bob", world.wall().isoformat()),
                             created_at=world.wall() + timedelta(minutes=1)))
    ana.poll()
    key = ana.share.status()["friends"][0]["key"]
    assert key["fingerprint"] == first["fingerprint"] and key["changed"] and key["new_fingerprint"] == other.public.fingerprint
    with pytest.raises(FaceShareError, match="Bob's box key changed"):
        ana.ask_to_share("bob")
    with pytest.raises(FaceShareError, match="not the fingerprint"):
        ana.share.trust_key("bob", first["fingerprint"])
    # bob's own box takes down keys under its name that are not its own; the warning goes with it
    bob.poll(), ana.poll()
    assert ana.share.status()["friends"][0]["key"] == first
    # bob really did set his box up again: after comparing fingerprints ana trusts the new key
    store.post(SealedMessage(id="key:bob:other", circle_id="team", from_member="bob", kind="box_key",
                             body=sealed.key_record(other, "team", "bob", world.wall().isoformat()),
                             created_at=world.wall() + timedelta(minutes=1)))
    ana.poll()
    ana.share.trust_key("bob", other.public.fingerprint)
    assert ana.share.status()["friends"][0]["key"]["fingerprint"] == other.public.fingerprint


def test_a_key_changed_between_the_card_and_the_approval_sends_nothing(pair, world):
    ana, bob, store = pair.ana, pair.bob, pair.store
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob")
    other = sealed.BoxKeys.generate()
    store.post(SealedMessage(id="key:bob:other", circle_id="team", from_member="bob", kind="box_key",
                             body=sealed.key_record(other, "team", "bob", world.wall().isoformat()), created_at=world.wall()))
    ana.poll()
    done = ana.decide()
    assert done[0].status == "failed" and "box key changed" in done[0].result
    assert "face_share" not in [r.kind for r in store.rows]


def test_a_key_record_under_someone_elses_name_is_not_a_key(world):
    ana, mal = world.box("ana"), world.box("mal")
    mal.poll()
    record = world.store.rows[0]
    world.store.post(record.model_copy(update={"id": "key:bob:mal", "from_member": "bob"}))  # mal's key, posted as bob's
    ana.poll()
    keys = {f["member_id"]: f["key"]["known"] for f in ana.share.status()["friends"]}
    assert keys == {"bob": False, "mal": True}


def test_the_box_key_survives_a_restart_and_is_kept_in_the_vault(pair, world):
    ana = pair.ana
    ana.poll()
    before = ana.share.status()["me"]["fingerprint"]
    blob = (ana.dir / "vault" / "circle" / "face_share.bin").read_bytes()
    state = json.loads(ana.share.vault.decrypt("circle.face_share", blob))
    assert set(state) == {"keys", "seq", "pins", "out", "in", "ignored"} and state["keys"].encode() not in blob
    if os.name == "posix":  # Windows has no owner-only mode
        assert (ana.dir / "vault" / "circle" / "face_share.bin").stat().st_mode & 0o777 == 0o600
    again = Box(world.root, "ana", world.store, world.wall)  # the same data folder, a new process
    assert again.share.status()["me"]["fingerprint"] == before


# --- the protocol is additive ---


def test_rows_of_a_kind_or_version_this_box_does_not_know_are_left_alone(pair):
    ana, bob, store = pair.ana, pair.bob, pair.store
    ana.poll(), bob.poll()
    store.post(SealedMessage(id="x1", circle_id="team", from_member="ana", to_member="bob", kind="voice_share",
                             body={"anything": [1, 2, 3]}, created_at=pair.wall()))
    store.post(SealedMessage(id="x2", v=2, circle_id="team", from_member="ana", to_member="bob", kind="face_share",
                             body={"header": {"v": 2}, "new_field": "?"}, created_at=pair.wall()))
    store.post(SealedMessage(id="x3", circle_id="team", from_member="ana", to_member="bob", kind="face_share",
                             body={"not": "an envelope"}, created_at=pair.wall()))
    events = bob.poll()
    assert bob.share.skipped == 2 and bob.cards() == [] and bob.identity.members() == []
    assert [(e.state, e.reason) for e in events] == [("ignored", "bad_signature")]  # x3 claims to be version 1 and is not
    assert {r.id for r in store.rows} >= {"x1", "x2", "x3"}  # nothing of someone else's was deleted
    share_and_accept(ana, bob)  # and the real thing still works next to them
    assert len(bob.identity.members()) == 1


def test_rows_with_oversized_ids_or_names_are_not_read_into_the_log_or_the_state(pair):
    ana, bob, store = pair.ana, pair.bob, pair.store
    ana.poll(), bob.poll()
    store.post(SealedMessage(id="i" * 5000, circle_id="team", from_member="ana", to_member="bob", kind="face_share",
                             body={}, created_at=pair.wall()))
    store.post(SealedMessage(id="x9", circle_id="team", from_member="n" * 5000, to_member="bob", kind="face_share",
                             body={}, created_at=pair.wall()))
    assert bob.poll() == [] and bob.share.skipped == 2
    assert len((bob.dir / "vault" / "circle" / "face_share.bin").read_bytes()) < 4000


def test_the_plans_protocol_never_sees_a_face_share(pair):
    """`CircleAgent.poll` builds a `circle.message.received` for every agent_messages row, whose `kind` is a closed
    list: a new kind in that table would stop an older box's poll. So face share has its own channel, and a box
    that only knows plans (or has face share off) reads nothing of it."""
    ana, bob = pair.ana, pair.bob
    plans = InMemoryCircleStore()
    agents = [CircleAgent(m, "team", plans, FakeMemory(items=[]), pair.wall, members=list(NAMES)) for m in ("ana", "bob", "mal")]
    share_and_accept(ana, bob)
    ana.publish(ana.share.revoke("bob"))
    bob.poll()
    assert plans.messages == [] and plans.events == {}
    assert [a.poll() for a in agents] == [[], [], []]
    assert not hasattr(plans, "inbox") and not hasattr(pair.store, "since")  # two stores, two ports


def test_other_circle_shares_still_reach_their_own_executor(world):
    ana = world.box("ana")
    seen = []
    ana.share.fallback = lambda req: seen.append(req.params["kind"]) or "proposed to the circle: Study"
    req = SimpleNamespace(params={"kind": "event_proposal"})
    assert asyncio.run(ana.share.executor(req)) == "proposed to the circle: Study" and seen == ["event_proposal"]

    async def league(req):
        return "League stake set"

    ana.share.fallback = league
    assert asyncio.run(ana.share.executor(SimpleNamespace(params={"kind": "league_stake"}))) == "League stake set"
    ana.share.fallback = None
    with pytest.raises(ValueError, match="unsupported share"):
        asyncio.run(ana.share.executor(req))


# --- bounds ---


@pytest.mark.parametrize("change, reason", [
    (lambda body: {**body, "padding": "A" * 500_000}, "too_large"),
    (lambda body: {**body, "ciphertext": body["ciphertext"][:-8] + "AAAAAAA="}, "bad_signature"),
    (lambda body: {**body, "header": {**body["header"], "expires_at": "2027-06-01T00:00:00+00:00"}}, "bad_signature"),
])
def test_an_oversized_or_tampered_row_is_ignored_before_anything_is_opened(pair, change, reason):
    ana, bob, store = pair.ana, pair.bob, pair.store
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide()
    real = store.rows[2]
    assert len(sealed.canonical(real.body)) < 20_000  # 6 vectors of 32 numbers and a name: small
    store.rows[2] = real.model_copy(update={"body": change(real.body)})
    assert [(e.state, e.reason) for e in bob.poll()] == [("ignored", reason)]
    assert bob.poll() == [] and bob.cards() == [] and bob.identity.members() == []


def test_a_share_changed_after_the_card_was_raised_is_not_opened(pair):
    ana, bob, store = pair.ana, pair.bob, pair.store
    ana.poll(), bob.poll(), ana.poll()
    ana.ask_to_share("bob"), ana.decide(), bob.poll()
    real = store.rows[2]
    store.rows[2] = real.model_copy(update={"body": {**real.body, "nonce": sealed.b64(b"\x00" * 12)}})
    done = bob.decide()
    assert done[0].status == "failed" and "signature" in done[0].result and bob.identity.members() == []


def test_only_roster_members_and_never_oneself(pair):
    ana = pair.ana
    ana.poll()
    for nobody in ("ana", "zed"):
        with pytest.raises(KeyError):
            ana.share.propose_share(nobody)
    assert [f["member_id"] for f in ana.share.status()["friends"]] == ["bob", "mal"]


def test_days_are_bounded(world):
    assert world.box("ana", days=400).share.days == 90 and FaceShare("team", "x", None, None, None, world.root, days=0).days == 1
