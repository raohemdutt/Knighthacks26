"""Who "you" is on this box, and which faces in a live frame are yours ("Only me clear", the training opt-in).

The owner is the one person who did the face scan here. `owner_faces` answers from the person tags recognition
already keeps: it recognises nothing itself, and every way of not being sure is False (so the face is blurred).
"""

from dataclasses import replace

from contexts.perception.application.identity import OWNER_NO_MODEL, OWNER_OFF, OWNER_SCAN_FIRST, OWNER_WHICH
from contexts.perception.infrastructure.identity_store import Member
from contexts.perception.tests.identity_helpers import H, W, Frame, Rig, face_in, kinds, person, row

ANA, BEN = person(1, 0.05, 0.1, 0.45, 0.95), person(2, 0.55, 0.1, 0.95, 0.95)


def tagged(rig: Rig) -> None:
    """Ana (track 1) and a stranger (track 2) in view until Ana's track carries her tag."""
    for i in range(4):
        rig.look(Frame(face_in(ANA, b"ana#%d" % i), face_in(BEN, b"ben#%d" % i)), [ANA, BEN])
    assert rig.service.tagged() == {"cam-desk": {1: "m-ana"}}


def faces(rig: Rig, *who_and_track, tracks=(ANA, BEN), camera="cam-desk") -> list[bool]:
    rows = [face_in(track, who)[1] for who, track in who_and_track]
    return rig.service.owner_faces(camera, rows, list(tracks), W, H)


def test_nobody_is_the_owner_until_someone_has_done_the_face_scan(tmp_path):
    rig = Rig(tmp_path)
    assert rig.service.owner() is None
    assert rig.service.owner_status() == {"scanned": False, "ready": False, "reason": OWNER_SCAN_FIRST, "demo": False}
    assert faces(rig, (b"ana#1", ANA)) == [False]

    rig.enroll("m-ana", "Ana", b"ana")
    assert rig.service.owner().member_id == "m-ana"
    assert rig.service.owner_status() == {"scanned": True, "ready": True, "reason": None, "demo": False}


def test_only_a_face_in_a_track_tagged_as_the_owner_is_the_owners(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana", b"ana")
    assert faces(rig, (b"ana#1", ANA), (b"ben#1", BEN)) == [False, False]  # in view, but no look has tagged her yet
    rig.look(Frame(face_in(ANA, b"ana#0"), face_in(BEN, b"ben#0")), [ANA, BEN])
    assert faces(rig, (b"ana#1", ANA), (b"ben#1", BEN)) == [False, False]  # one match is not a tag: still blurred
    tagged(rig)

    calls = rig.finder.calls
    assert faces(rig, (b"ana#1", ANA), (b"ben#1", BEN)) == [True, False]
    assert faces(rig, (b"ben#1", BEN), (b"ana#1", ANA)) == [False, True]
    assert rig.finder.calls == calls  # nothing was looked for or recognised to answer: the tags are enough

    assert faces(rig, (b"ana#1", ANA), camera="cam-door") == [False]  # her tag is on the desk camera only
    assert rig.service.owner_faces("cam-desk", [row(0.47 * W, 0.5 * H)], [ANA, BEN], W, H) == [False]  # a face in nobody's box
    assert rig.service.owner_faces("cam-desk", [], [ANA, BEN], W, H) == []
    assert faces(rig, (b"ana#1", ANA), tracks=()) == [False]  # no person tracks to go by


def test_two_faces_in_the_owners_box_are_both_blurred(tmp_path):
    """Someone leans in close behind the owner: the box cannot say which face is whose, so neither is left clear."""
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana", b"ana")
    tagged(rig)
    hers, behind = face_in(ANA, b"ana#1")[1], row(0.3 * W, 0.3 * H)
    assert rig.service.owner_faces("cam-desk", [hers, behind, face_in(BEN, b"ben#1")[1]], [ANA, BEN], W, H) == [False, False, False]
    assert rig.service.owner_faces("cam-desk", [hers], [ANA, BEN], W, H) == [True]


def test_recognition_off_forgetting_or_a_missing_model_ends_it(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana", b"ana")
    tagged(rig)
    rig.service.set_enabled(False)
    assert faces(rig, (b"ana#1", ANA)) == [False]
    assert rig.service.owner_status() == {"scanned": True, "ready": False, "reason": OWNER_OFF, "demo": False}

    rig.service.set_enabled(True)
    tagged(rig)
    assert faces(rig, (b"ana#1", ANA)) == [True]
    rig.service.remove("m-ana")  # "Forget me"
    assert faces(rig, (b"ana#1", ANA)) == [False] and rig.service.owner() is None
    assert rig.service.owner_status()["reason"] == OWNER_SCAN_FIRST

    bare = Rig(tmp_path / "bare", recognizer=None)
    bare.store.save(replace(rig_member(), member_id="m-ana"))
    assert bare.service.owner_status() == {"scanned": True, "ready": False, "reason": OWNER_NO_MODEL, "demo": False}
    assert faces(bare, (b"ana#1", ANA)) == [False]


def rig_member(**changes) -> Member:
    base = Member(member_id="m-x", name="X", enrolled_at="2026-10-11T12:00:00+00:00", model="fake", frames=1, vector=[1.0, 0.0],
                  vectors=[[1.0, 0.0]], updated_at="2026-10-11T12:00:00+00:00", source="scan", consent_at="2026-10-11T12:00:00+00:00",
                  scanned=True)
    return replace(base, **changes)


def test_photos_alone_a_friends_template_and_someone_elses_link_are_not_the_owner(tmp_path):
    rig = Rig(tmp_path)
    rig.service.set_enabled(True)
    rig.store.save(rig_member(member_id="m-photos", source="photos", scanned=False))
    assert rig.service.owner() is None and rig.service.owner_status()["reason"] == OWNER_SCAN_FIRST
    rig.store.save(rig_member(member_id="c-friend", source="circle", shared_by="maya", circle_member_id="maya", scanned=None))
    assert rig.service.owner() is None  # a friend's box shared it: they are not this box's "you"

    rig.store.save(rig_member(member_id="m-me"))
    assert rig.service.owner().member_id == "m-me"
    rig.store.save(rig_member(member_id="m-guest", vector=[0.0, 1.0], vectors=[[0.0, 1.0]]))
    assert rig.service.owner() is None  # two people scanned here and nothing says which one is "you"
    assert rig.service.owner_status() == {"scanned": False, "ready": False, "reason": OWNER_WHICH, "demo": False}

    rig.service.owner_circle_id = "hemdutt"  # the box has a Circle: "you" is its own member
    assert rig.service.owner() is None
    rig.store.save(rig_member(member_id="m-me", circle_member_id="hemdutt"))
    assert rig.service.owner().member_id == "m-me"
    rig.store.delete("m-me")
    rig.store.save(rig_member(member_id="m-guest", circle_member_id="maya", vector=[0.0, 1.0], vectors=[[0.0, 1.0]]))
    assert rig.service.owner() is None  # the only scan left is linked to someone else


def test_the_demo_owner_is_shown_as_scanned_and_never_matched_to_a_face(tmp_path):
    rig = Rig(tmp_path)
    rig.service.set_enabled(True)
    rig.store.save(rig_member(member_id="m-demo", model="demo-synthetic", demo=True))
    assert rig.service.owner_status() == {"scanned": True, "ready": True, "reason": None, "demo": True}
    for i in range(4):
        assert kinds(rig.look(Frame(face_in(ANA, b"ana#%d" % i)), [ANA])) in ([], [("lost", None, None)])
    assert faces(rig, (b"ana#1", ANA)) == [False]  # made-up numbers: no real face is ever the demo owner's
