import pytest

from contexts.perception.application.identity import IdentityError
from contexts.perception.domain.identity import TrackTags, average, best_match, cosine, face_box, track_for_face
from contexts.perception.infrastructure.identity_store import Member
from contexts.perception.infrastructure.recognizer import FakeRecognizer, SFaceRecognizer, build_recognizer
from contexts.perception.tests.identity_helpers import H, W, Frame, Rig, face_in, kinds, person, row

# --- the pure parts ---


def test_cosine_average_and_the_threshold():
    assert cosine([1, 0], [2, 0]) == pytest.approx(1.0) and cosine([1, 0], [0, 3]) == pytest.approx(0.0)
    assert cosine([1, 0], [1, 0, 0]) == -1.0  # vectors from different models never match
    assert average([[2, 0], [0, 5]]) == pytest.approx([0.7071, 0.7071], abs=1e-3)  # each look counts the same
    templates = {"m-ana": [1.0, 0.0], "m-ben": [0.0, 1.0]}
    assert best_match([0.9, 0.1], templates, 0.8) == "m-ana"
    assert best_match([0.7, 0.7], templates, 0.8) is None  # close to both, enough like neither
    assert best_match([1.0, 0.0], {}, 0.1) is None


def test_a_face_goes_to_the_person_whose_box_contains_it():
    left, right = person(1, 0.05, 0.1, 0.45, 0.95), person(2, 0.55, 0.1, 0.95, 0.95)
    cup = person(3, 0.0, 0.0, 1.0, 1.0).model_copy(update={"label": "cup"})
    face = face_box(row(0.7 * W, 0.15 * H), W, H)
    assert track_for_face(face, [left, right, cup]).track_id == 2  # never a thing that is not a person
    assert track_for_face(face_box(row(10, 0.99 * H - 80), W, H), [right]) is None  # a face in nobody's box
    behind = person(4, 0.5, 0.3, 1.0, 1.0)  # overlaps `right`, but its head would be lower
    assert track_for_face(face, [behind, right]).track_id == 2


def test_the_fake_recognizer_knows_the_same_crop_and_no_other():
    r = FakeRecognizer()
    ana, again, ben = r.embed(b"ana#1"), r.embed(b"ana#2"), r.embed(b"ben#1")
    assert r.embed(b"ana#1") == ana and cosine(ana, again) > r.threshold > cosine(ana, ben)
    assert r.match(again, {"m-ana": ana}) == "m-ana" and r.match(ben, {"m-ana": ana}) is None


def test_sface_is_built_only_when_its_model_file_is_here(tmp_path):
    assert build_recognizer(tmp_path / "missing.onnx") is None  # nothing is downloaded: no file, no recognition

    class Net:  # cv2.FaceRecognizerSF's two calls
        def alignCrop(self, frame, face_row):
            return ("aligned", frame, tuple(face_row[:4]))

        def feature(self, crop):
            return type("M", (), {"flatten": lambda self: [3.0, 4.0]})()

    sface = SFaceRecognizer(net=Net())
    assert sface.crop("frame", row(1, 2)) == ("aligned", "frame", (1, 2, 60, 80))
    assert sface.embed("crop") == pytest.approx([0.6, 0.8])
    assert sface.threshold == pytest.approx(0.363) and sface.name.startswith("sface")


def test_a_tag_needs_two_matches_in_a_row_and_is_checked_again_later():
    tags = TrackTags("cam-desk", confirm=2, verify_every_s=10, strikes=2)
    tags.seen([person(1)], 0.0)
    assert tags.due(0.0) == {1}
    assert tags.looked(1, "m-ana", 0.0) == [] and tags.looked(1, None, 1.0) == []  # one match, then a miss: start over
    assert tags.looked(1, "m-ana", 2.0) == [] and tags.looked(1, "m-ben", 3.0) == []  # two different people: still nobody
    assert kinds(tags.looked(1, "m-ben", 4.0)) == [("identified", 1, "m-ben")]
    tags.seen([person(1)], 5.0)
    assert tags.due(5.0) == set()  # tagged: not looked at again until the check is due
    tags.seen([person(1)], 14.0)
    assert tags.due(14.0) == {1}
    assert tags.looked(1, "m-ana", 14.0) == []  # one look that disagrees is not enough to drop the tag
    assert kinds(tags.looked(1, None, 24.0)) == [("lost", 1, None)] and tags.members == {}


# --- off by default ---


def test_recognition_is_off_until_switched_on_and_then_does_nothing_without_a_template(tmp_path):
    rig = Rig(tmp_path)
    assert rig.service.settings() == {"identity_enabled": False, "available": True, "recognizer": "fake", "everyone": False,
                                      "expressions_available": False, "expression_reader": None, "photos_available": False}
    assert rig.look(Frame(face_in(person(1), b"ana#1")), [person(1)]) == [] and rig.finder.calls == 0
    with pytest.raises(IdentityError):
        rig.service.start_enrollment("m-ana", "Ana")  # enrolling needs the switch
    assert not (tmp_path / "vault").exists()  # off: nothing was written, not even a key

    changed = rig.service.set_enabled(True)
    assert [(e.type, e.enabled, e.members) for e in changed] == [("perception.identity.changed", True, [])]
    first = rig.look(Frame(face_in(person(1), b"ana#1")), [person(1)])
    assert kinds(first) == [("lost", None, None)]  # the first look forgets track ids from before the start
    assert rig.finder.calls == 0  # nobody is enrolled: no face is even looked for


def test_without_the_model_the_feature_says_so_and_cannot_be_switched_on(tmp_path):
    rig = Rig(tmp_path, recognizer=None)
    assert rig.service.settings()["available"] is False and "not installed" in rig.service.settings()["reason"]
    with pytest.raises(IdentityError):
        rig.service.set_enabled(True)
    rig.store.set_enabled(True)  # even with the file saying yes
    assert rig.service.enabled is False and rig.look(Frame(), [person(1)]) == []


# --- enrolment ---


def test_enrolling_takes_five_frames_with_exactly_one_face_looking_at_the_camera(tmp_path):
    rig = Rig(tmp_path)
    rig.service.set_enabled(True)
    status = rig.service.start_enrollment("m-ana", "Ana Example", consent_ack=True)
    assert (status["state"], status["frames_done"], status["frames_needed"]) == ("collecting", 0, 5)
    assert "name" not in status

    ana = lambda i, **kw: (b"ana#%d" % i, row(300, 100, **kw))  # noqa: E731
    rig.look(Frame())
    assert rig.service.enrollment_status()["hint"] == "no_face"
    rig.look(Frame(ana(0), (b"ben#0", row(100, 100))))
    assert rig.service.enrollment_status()["hint"] == "one_person_only"
    rig.look(Frame(ana(0, facing=False)))
    assert rig.service.enrollment_status()["hint"] == "look_at_camera"
    rig.look(Frame(ana(0, score=0.5)))  # YuNet is not sure it is a face
    assert rig.service.enrollment_status()["frames_done"] == 0

    out = []
    for i in range(4):
        out += rig.look(Frame(ana(i)))
    out += rig.look(Frame((b"ben#9", row(300, 100))))  # someone else steps in: not part of this template
    assert rig.service.enrollment_status()["frames_done"] == 4 and not [e for e in out if e.type.endswith("changed")]
    out += rig.look(Frame(ana(4)))

    status = rig.service.enrollment_status()
    assert (status["state"], status["member_id"], status["camera_id"]) == ("done", "m-ana", "cam-desk")
    assert [(e.enabled, e.members) for e in out if e.type == "perception.identity.changed"] == [(True, ["m-ana"])]
    assert rig.service.members() == [{"member_id": "m-ana", "name": "Ana Example", "enrolled_at": "2026-10-11T12:00:10+00:00",
                                      "updated_at": "2026-10-11T12:00:10+00:00", "photos": 5, "source": "live", "expressions": False,
                                      "circle_member_id": None, "scanned": True, "recognised": True}]
    template = rig.store.members()[0].vector
    assert cosine(template, FakeRecognizer().embed(b"ana#77")) > 0.9  # the average still is her


def test_an_enrolment_nobody_shows_up_for_times_out_and_bad_input_is_refused(tmp_path):
    rig = Rig(tmp_path, timeout_s=30)
    rig.service.set_enabled(True)
    for bad_id, name in (("Ana Example", "Ana"), ("m-ana", "  "), ("../x", "Ana")):
        with pytest.raises(ValueError):
            rig.service.start_enrollment(bad_id, name)
    made_up = rig.service.start_enrollment(None, "Ana", consent_ack=True)["member_id"]
    assert made_up.startswith("m-") and "ana" not in made_up  # the box picks an id that is not the name
    rig.look(Frame((b"ana#0", row(300, 100))))
    rig.look(Frame(), step=31)
    status = rig.service.enrollment_status()
    assert (status["state"], status["error"], status["frames_done"]) == ("failed", "timeout", 0)
    assert rig.service.members() == []


def test_a_member_id_that_ends_in_a_newline_is_refused_everywhere(tmp_path):
    """`$` matches before a trailing newline, so "m-ana\\n" passed and would have named a file "m-ana\\n.face"."""
    rig = Rig(tmp_path)
    rig.service.set_enabled(True)
    with pytest.raises(ValueError):
        rig.service.start_enrollment("m-ana\n", "Ana")
    with pytest.raises(ValueError):
        rig.store.save(Member(member_id="m-ana\n", name="Ana", enrolled_at="2026-10-11T12:00:00+00:00", model="fake", frames=5, vector=[1.0]))
    rig.enroll("m-ana", "Ana", b"ana")
    assert rig.store.delete("m-ana\n") is False and [m.member_id for m in rig.store.members()] == ["m-ana"]


# --- tagging ---


def test_a_matched_face_tags_its_track_and_a_stranger_never_gets_one(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    ana, guest = person(7, 0.05, 0.1, 0.45, 0.95), person(8, 0.55, 0.1, 0.95, 0.95)
    both = lambda i: Frame(face_in(ana, b"ana#%d" % i), face_in(guest, b"guest#%d" % i))  # noqa: E731

    assert rig.look(both(1), [ana, guest]) == []  # one look is not enough
    assert kinds(rig.look(both(2), [ana, guest])) == [("identified", 7, "m-ana")]
    for i in range(3, 30):
        assert rig.look(both(i), [ana, guest]) == []  # the guest is looked at again and again and stays nobody
    assert rig.service.tagged() == {"cam-desk": {7: "m-ana"}}


def test_a_track_keeps_its_tag_when_the_face_turns_away_and_loses_it_when_the_track_ends(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    ana = person(7)
    rig.look(Frame(face_in(ana, b"ana#1")), [ana])
    assert kinds(rig.look(Frame(face_in(ana, b"ana#2")), [ana])) == [("identified", 7, "m-ana")]
    for _ in range(40):  # 40 s with her back to the camera: no face at all
        assert rig.look(Frame(), [ana]) == []
    assert rig.service.tagged()["cam-desk"] == {7: "m-ana"}
    assert rig.look(Frame(), [], step=2) == []  # out of view for a moment
    assert kinds(rig.look(Frame(), [], step=4)) == [("lost", 7, None)]
    back = person(9)  # she walks back in: a new track, tagged only after it was matched again
    assert rig.look(Frame(face_in(back, b"ana#3")), [back]) == []


def test_two_enrolled_people_are_tagged_separately_on_each_camera(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    rig.enroll("m-ben", "Ben Example", b"ben")
    assert [m["member_id"] for m in rig.service.members()] == ["m-ana", "m-ben"]
    ana, ben = person(1, 0.05, 0.1, 0.45, 0.95), person(2, 0.55, 0.1, 0.95, 0.95)
    out = []
    for i in range(2):
        out += rig.look(Frame(face_in(ana, b"ana#%d" % i), face_in(ben, b"ben#%d" % i)), [ana, ben])
        out += rig.look(Frame(face_in(ana, b"ben#%d" % i)), [ana], camera="cam-door", step=0)
    assert sorted(kinds([e for e in out if e.type.endswith("identified")])) == [
        ("identified", 1, "m-ana"), ("identified", 1, "m-ben"), ("identified", 2, "m-ben")]
    assert rig.service.tagged() == {"cam-desk": {1: "m-ana", 2: "m-ben"}, "cam-door": {1: "m-ben"}}


def test_deleting_a_member_or_switching_off_ends_the_tags(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    rig.enroll("m-ben", "Ben Example", b"ben")
    ana, ben = person(1, 0.05, 0.1, 0.45, 0.95), person(2, 0.55, 0.1, 0.95, 0.95)
    for i in range(2):
        rig.look(Frame(face_in(ana, b"ana#%d" % i), face_in(ben, b"ben#%d" % i)), [ana, ben])

    changed = rig.service.remove("m-ben")
    assert [(e.enabled, e.members) for e in changed] == [(True, ["m-ana"])]
    assert not (tmp_path / "vault" / "identity" / "m-ben.face").exists()
    assert kinds(rig.look(Frame(), [ana, ben])) == [("lost", 2, None)]
    with pytest.raises(KeyError):
        rig.service.remove("m-ben")

    assert [(e.enabled, e.members) for e in rig.service.set_enabled(False)] == [(False, ["m-ana"])]
    assert kinds(rig.look(Frame(), [ana, ben])) == [("lost", None, None)]
    assert rig.look(Frame(face_in(ana, b"ana#5")), [ana]) == [] and rig.service.tagged() == {}


def test_the_hook_returns_the_detectors_tracks_and_never_lets_a_failure_through(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    ana, published = person(7), []
    detect = rig.service.watch("cam-desk", lambda frame: [ana], rig.finder, published.append)
    for i in range(2):
        rig.clock.t += 1
        assert detect(Frame(face_in(ana, b"ana#%d" % i))) == [ana]
    assert kinds(published)[-1] == ("identified", 7, "m-ana")

    class Broken:
        def faces(self, frame):
            raise RuntimeError("no model")

    bad = rig.service.watch("cam-door", lambda frame: [ana], Broken(), published.append)
    rig.clock.t += 1
    assert bad(Frame()) == [ana] and rig.service.errors == 1  # counted; the camera carries on

    def failing(frame):
        raise RuntimeError("engine hiccup")

    with pytest.raises(RuntimeError):  # the detector's own errors still reach the pipeline, which counts them
        rig.service.watch("cam-desk", failing, rig.finder, published.append)(Frame())


def test_at_start_up_the_state_is_announced_only_when_the_log_does_not_have_it(tmp_path):
    rig = Rig(tmp_path)
    assert rig.service.announce() == []  # off, nobody enrolled, nothing in the log: nothing to say
    events = rig.enroll("m-ana", "Ana Example", b"ana")

    restarted = Rig(tmp_path)  # the same vault folder
    for e in events:
        if e.type == "perception.identity.changed":
            restarted.service.on_changed(e)  # replay
    assert restarted.service.enabled and restarted.service.announce() == []
    wiped = Rig(tmp_path)  # the log was wiped; the vault folder was not
    assert [(e.enabled, e.members) for e in wiped.service.announce()] == [(True, ["m-ana"])]
