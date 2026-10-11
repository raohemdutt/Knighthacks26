"""The four doors face share uses on the identity service (contexts/circle/application/face_share.py is the only
caller): what may be handed out, what may be taken in, and how it goes away. Synthetic vectors, the fake recogniser."""

from datetime import timedelta

import pytest

from contexts.perception.application.identity import IdentityError, shared_member_id
from contexts.perception.domain.identity import average
from contexts.perception.infrastructure.recognizer import FakeRecognizer
from contexts.perception.tests.identity_helpers import T0, Frame, Rig, face_in, person

LATER = T0 + timedelta(days=30)


def template(who: bytes = b"zoe", looks: int = 3) -> dict:
    fake = FakeRecognizer()
    vectors = [fake.embed(who + b"#%d" % i) for i in range(looks)]
    return {"name": "Zoe Example", "model": "fake", "vector": average(vectors), "vectors": vectors}


def take(rig: Rig, shared_by: str = "zoe", **change):
    return rig.service.receive_shared(shared_by, **{**template(), **change}, expires_at=LATER, share_id="s1")


def test_only_the_person_linked_as_this_boxes_own_member_can_be_handed_out(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(IdentityError, match="nobody enrolled"):
        rig.service.own_template("ana")
    rig.enroll("m-ana", "Ana Example", b"ana")
    rig.enroll("m-ben", "Ben Visitor", b"ben")
    with pytest.raises(IdentityError, match="nobody enrolled"):  # enrolled, but nobody said which Circle member they are
        rig.service.own_template("ana")
    rig.service.update_member("m-ana", circle_member_id="ana")
    rig.service.update_member("m-ben", circle_member_id="ben")
    own = rig.service.own_template("ana")
    stored = rig.store.get("m-ana")
    assert own == {"member_id": "m-ana", "name": "Ana Example", "model": "fake", "vector": stored.vector, "vectors": stored.vectors}
    assert set(own) == {"member_id", "name", "model", "vector", "vectors"}  # vectors and a name: nothing else exists to give
    rig.service.update_member("m-ben", circle_member_id="ana")
    with pytest.raises(IdentityError, match="more than one"):
        rig.service.own_template("ana")


def test_a_template_without_recorded_consent_is_not_handed_out(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    rig.service.update_member("m-ana", circle_member_id="ana")
    old = rig.store.get("m-ana")
    rig.store.save(old.__class__(**{**old.__dict__, "consent_at": None}))  # a blob from before consent was recorded
    with pytest.raises(IdentityError, match="nobody enrolled"):
        rig.service.own_template("ana")


def test_a_received_template_is_a_member_that_says_where_it_came_from(tmp_path):
    rig = Rig(tmp_path)
    rig.service.set_enabled(True)
    [changed] = take(rig)
    assert changed.received_from == ["zoe"] and changed.members == [shared_member_id("zoe")] and changed.enrolled_member is None
    [row] = rig.service.members()
    assert row == {"member_id": shared_member_id("zoe"), "name": "Zoe Example", "enrolled_at": row["enrolled_at"],
                   "updated_at": row["enrolled_at"], "photos": 3, "source": "circle", "expressions": False,
                   "circle_member_id": "zoe", "shared_by": "zoe", "received_at": row["enrolled_at"],
                   "expires_at": LATER.isoformat(), "scanned": True, "recognised": True}  # made from their own scan, on their box
    assert shared_member_id("zoe").startswith("c-") and shared_member_id("zoe") != shared_member_id("Zoe")
    zoe = person(7)
    for i in range(3):
        rig.look(Frame(face_in(zoe, b"zoe#desk%d" % i)), [zoe])
    assert rig.service.tagged() == {"cam-desk": {7: shared_member_id("zoe")}}
    take(rig)  # a newer share from the same friend replaces it: still one member
    assert len(rig.service.members()) == 1


def test_rows_of_people_enrolled_here_have_nothing_of_a_share_in_them(tmp_path):
    rig = Rig(tmp_path)
    events = rig.enroll("m-ana", "Ana Example", b"ana")
    assert set(rig.service.members()[0]) == {"member_id", "name", "enrolled_at", "updated_at", "photos", "source",
                                              "expressions", "circle_member_id", "scanned", "recognised"}
    assert all(e.received_from == [] for e in events if e.type == "perception.identity.changed")


@pytest.mark.parametrize("bad", [
    {"vector": [0.1] * 8}, {"vector": [0.0] * 32}, {"vector": [float("nan")] * 32}, {"vector": [1] * 32},
    {"vectors": []}, {"vectors": [[0.5] * 31]}, {"vectors": [[0.5] * 32] * 13}, {"vectors": "abc"}, {"vector": None},
    {"name": ""}, {"name": "x" * 61}, {"model": ""}, {"model": 7},
])
def test_a_malformed_template_is_refused_and_never_quoted(tmp_path, bad):
    rig = Rig(tmp_path)
    with pytest.raises(ValueError) as caught:
        take(rig, **bad)
    assert rig.service.members() == [] and "0.5" not in str(caught.value) and "nan" not in str(caught.value)


def test_a_face_already_enrolled_here_is_not_taken_in_under_a_second_id(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-zoe", "Zoe Enrolled Here", b"zoe")  # the friend enrolled on this box earlier (the old way)
    with pytest.raises(IdentityError, match="already enrolled on this box"):
        take(rig)
    assert [m["member_id"] for m in rig.service.members()] == ["m-zoe"]


def test_it_can_only_be_forgotten_never_edited_or_added_to(tmp_path):
    rig = Rig(tmp_path)
    rig.service.set_enabled(True)
    take(rig)
    member_id = shared_member_id("zoe")
    for change in (lambda: rig.service.update_member(member_id, expressions=True),
                   lambda: rig.service.update_member(member_id, name="Renamed"),
                   lambda: rig.service.start_enrollment(member_id, "Renamed", consent_ack=True),
                   lambda: rig.service.own_template("zoe")):
        with pytest.raises(IdentityError):
            change()
    assert rig.service.forget_shared("nobody") == []
    [gone] = rig.service.forget_shared("zoe")
    assert gone.members == [] and gone.received_from == [] and rig.service.members() == []


def test_past_its_expiry_it_is_not_matched_and_the_next_sweep_deletes_it(tmp_path):
    rig = Rig(tmp_path)
    rig.service.set_enabled(True)
    rig.enroll("m-ana", "Ana Example", b"ana")
    take(rig)
    assert rig.service.expire_shared(T0 + timedelta(days=29)) == ([], [])
    rig.service._now = lambda: LATER + timedelta(seconds=1)
    zoe, ana = person(7), person(8, 0.55, 0.1, 0.95, 0.95)
    for i in range(3):
        rig.look(Frame(face_in(zoe, b"zoe#%d" % i), face_in(ana, b"ana#%d" % i)), [zoe, ana])
    assert rig.service.tagged() == {"cam-desk": {8: "m-ana"}}  # the person enrolled here is still recognised; the friend is not
    whose, events = rig.service.expire_shared(LATER + timedelta(seconds=1))
    assert whose == ["zoe"] and events[0].members == ["m-ana"] and [m["member_id"] for m in rig.service.members()] == ["m-ana"]


def test_an_expiry_that_cannot_be_read_counts_as_run_out(tmp_path):
    rig = Rig(tmp_path)
    take(rig)
    stored = rig.store.get(shared_member_id("zoe"))
    rig.store.save(stored.__class__(**{**stored.__dict__, "expires_at": "soon"}))
    assert rig.service.expire_shared(T0)[0] == ["zoe"] and rig.service.members() == []
