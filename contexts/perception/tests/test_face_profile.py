"""The face profile on top of P9: enrolling from photos, exemplars, apparent expression, the "everyone" mode.
Fakes stand in for the detector, the recognizer, the photo decoder and the expression model: no model file,
no OpenCV and no network is needed here."""

import asyncio
import base64
import json
import logging
import math
import random
import struct
import time
from datetime import timedelta

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.perception.api import MAX_BODY_BYTES, identity_router, parse_photos
from contexts.perception.application.identity import MIN_FACE_PX, SCAN_FRAMES, IdentityError
from contexts.perception.application.tracking import TrackingPipeline
from contexts.perception.domain.expression import LABELS, ExpressionRun, head_pose
from contexts.perception.domain.identity import average, consistent, cosine, normalize
from contexts.perception.infrastructure.expression import FakeExpressionReader, build_expression_reader, softmax
from contexts.perception.infrastructure.identity_store import MAX_EXEMPLARS, IdentityStore, Member
from contexts.perception.infrastructure.photo import build_photo_reader, image_size
from contexts.perception.infrastructure.recognizer import FakeRecognizer
from contexts.perception.tests.identity_helpers import T0, FakePhotoReader, Frame, Mood, Rig, face_in, person, photo, row
from contexts.vault import KeyRing, Vault, Wiper
from contracts.events import parse_event

NAME = "Zebulon Quartermaine"
FAST = {"window_s": 30.0, "heartbeat_s": 180.0, "min_samples": 5, "idle_s": 60.0}


def box(tmp_path, **kw) -> Rig:
    """A box with the photo decoder and the expression model "installed", recognition switched on."""
    rig = Rig(tmp_path, photo_reader=FakePhotoReader(), expression_reader=FakeExpressionReader(), **kw)
    rig.service._photo_finder = rig.finder
    rig.service.set_enabled(True)
    return rig


def http(rig: Rig, bus: EventBus, token: str | None = None) -> TestClient:
    def guard(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(identity_router(rig.service, bus, guard=guard))
    return TestClient(app)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def good_photos(who="ana", n=5):
    return [photo(who, i) for i in range(n)]


# --- the store: old templates still load, exemplars, switches ---


def test_a_template_written_before_exemplars_existed_still_loads_and_matches(tmp_path):
    vault = Vault(KeyRing(tmp_path))
    old = {"member_id": "m-ana", "name": "Ana Example", "enrolled_at": "2026-10-01T09:00:00+00:00", "model": "fake",
           "frames": 5, "vector": average([FakeRecognizer().embed(b"ana#%d" % i) for i in range(5)])}
    folder = tmp_path / "vault" / "identity"
    folder.mkdir(parents=True)
    (folder / "m-ana.face").write_bytes(vault.encrypt("identity", json.dumps(old).encode()))  # exactly what P9 wrote

    rig = box(tmp_path)
    member = rig.store.get("m-ana")
    assert member.vectors == [] and member.exemplars == [old["vector"]] and member.expressions is False
    assert rig.service.members()[0]["photos"] == 1
    ana = person(7)
    for i in range(2):
        rig.look(Frame(face_in(ana, b"ana#live%d" % i)), [ana])
    assert rig.service.tagged() == {"cam-desk": {7: "m-ana"}}  # recognised from the old template

    result, _ = rig.service.enroll_photos("m-ana", None, good_photos(n=3), consent_ack=True)  # and it can be added to
    assert result["state"] == "done" and result["exemplars"] == 4
    assert rig.store.get("m-ana").name == "Ana Example" and rig.store.get("m-ana").enrolled_at == old["enrolled_at"]


def test_a_blob_from_a_newer_version_loads_and_the_switch_file_is_read_again_when_it_changes(tmp_path):
    vault = Vault(KeyRing(tmp_path))
    store = IdentityStore(tmp_path, vault)
    store.save(Member(member_id="m-ana", name="Ana", enrolled_at="2026-10-01T09:00:00+00:00", model="fake", frames=1, vector=[1.0]))
    raw = json.loads(vault.decrypt("identity", (store.dir / "m-ana.face").read_bytes())) | {"a_field_from_the_future": 1}
    (store.dir / "m-ana.face").write_bytes(vault.encrypt("identity", json.dumps(raw).encode()))
    assert [m.member_id for m in IdentityStore(tmp_path, vault).members()] == ["m-ana"]  # the unknown field is ignored

    assert (store.enabled, store.everyone) == (False, False)
    store.set_everyone(True)
    assert store.everyone is False  # never while recognition is off
    store.set_enabled(True)
    assert json.loads(store.settings_path.read_text()) == {"identity_enabled": True}  # "everyone" is written only while on
    store.set_everyone(True)
    assert json.loads(store.settings_path.read_text()) == {"identity_enabled": True, "everyone": True}
    other = IdentityStore(tmp_path, vault)
    assert (other.enabled, other.everyone) == (True, True)
    store.set_enabled(False)  # off takes the room mode with it
    assert (store.everyone, other.enabled, other.everyone) == (False, False, False)  # the other reader saw the file change
    store.set_enabled(True)
    assert store.everyone is False
    store.settings_path.unlink()
    assert store.enabled is False


# --- vectors ---


def test_photos_that_disagree_with_the_rest_are_dropped():
    fake = FakeRecognizer()
    ana = [fake.embed(b"ana#%d" % i) for i in range(4)]
    assert consistent([*ana[:2], fake.embed(b"ben#0"), *ana[2:]], fake.threshold) == [0, 1, 3, 4]
    assert consistent(ana, fake.threshold) == [0, 1, 2, 3]
    assert consistent([ana[0]], fake.threshold) == [0] and consistent([], fake.threshold) == []
    assert consistent([ana[0], fake.embed(b"ben#0"), fake.embed(b"cy#0")], fake.threshold) == []  # three strangers: no majority


def _unit(rng, dims):
    return normalize([rng.gauss(0, 1) for _ in range(dims)])


def test_matching_the_mean_of_the_exemplars_is_not_worse_than_before_and_the_closest_exemplar_would_be():
    """The evidence behind the choice of matcher, on synthetic 128-number vectors made the way SFace's behave:
    a person is a direction, a pose pushes it (partly the same way for everyone), every look adds noise.
    Old: the mean of five frontal live looks. New: the mean of eight photos (frontal, turns, glasses).
    Not taken: the closest single exemplar, which rejects more of the right person and scores strangers higher."""
    rng, dims, threshold, noise = random.Random(26), 128, 0.363, 1.5
    poses = {"front": 0.0, "left": 0.8, "right": 0.8, "glasses": 0.6, "far_left": 1.2, "far_right": 1.2, "down": 1.0}
    shared = {k: _unit(rng, dims) for k in poses}

    def someone():
        centre, own = _unit(rng, dims), {k: _unit(rng, dims) for k in poses}

        def look(pose):
            b, eps = poses[pose], _unit(rng, dims)
            return normalize([c + b * (0.8 * o + 0.6 * g) + noise * e for c, o, g, e in zip(centre, own[pose], shared[pose], eps, strict=True)])

        return look

    people = [someone() for _ in range(40)]
    live = [[p("front") for _ in range(5)] for p in people]
    photos = [[p(k) for k in ("front", "front", "front", "left", "right", "glasses", "left", "right")] for p in people]
    rules = {"old": lambda x, i: cosine(x, average(live[i])), "mean": lambda x, i: cosine(x, average(photos[i])),
             "closest": lambda x, i: max(cosine(x, v) for v in photos[i])}
    genuine, impostor = {k: [] for k in rules}, {k: [] for k in rules}
    for i, p in enumerate(people):
        for pose in poses:
            x = p(pose)
            for name, score in rules.items():
                genuine[name].append(score(x, i))
                impostor[name] += [score(x, (i + j) % len(people)) for j in (1, 2, 3)]
    frr = {k: sum(s < threshold for s in v) / len(v) for k, v in genuine.items()}
    far = {k: sum(s >= threshold for s in v) / len(v) for k, v in impostor.items()}
    top = {k: sorted(v)[int(0.99 * len(v))] for k, v in impostor.items()}  # the 99th percentile stranger
    assert frr["mean"] <= frr["old"] and far["mean"] <= 0.005  # the right person is let in at least as often; strangers are not
    assert sum(genuine["mean"]) > sum(genuine["old"])  # and with more room to spare
    assert frr["closest"] > frr["mean"] and top["closest"] > top["mean"]  # the closest-exemplar rule is worse both ways


# --- enrolling from photos ---


def test_each_photo_is_judged_and_told_why_and_the_good_ones_become_the_template(tmp_path):
    rig = box(tmp_path)
    photos = [photo("ana", 0), photo("ana", 1, others=("ben",)), photo("ana", 2, brightness=20), photo("ana", 3, w=MIN_FACE_PX - 10),
              photo("ana", 4, sharpness=5), photo("ana", 5, score=0.5), photo("ana", 6, facing=False), b"not a picture",
              photo("ana", 8, brightness=250), photo("ben", 9), photo("ana", 10), photo("ana", 11)]
    result, events = rig.service.enroll_photos(None, "Ana Example", photos, consent_ack=True)

    assert [p["reason"] for p in result["photos"]] == [None, "two_faces", "too_dark", "too_small", "blurry", "no_face", "turned",
                                                       "unreadable", "too_bright", "different", None, None]
    assert [p["accepted"] for p in result["photos"]] == [True] + [False] * 9 + [True, True]
    messages = {p["reason"]: p["message"] for p in result["photos"] if p["reason"]}
    assert messages["two_faces"].startswith("two faces") and messages["too_dark"].startswith("too dark")
    assert messages["too_small"] == "too small: move closer" and messages["blurry"].startswith("blurry")
    assert (result["state"], result["accepted"], result["rejected"], result["exemplars"], result["error"]) == ("done", 3, 9, 3, None)
    assert set(result) == {"state", "member_id", "accepted", "rejected", "needed", "photos", "exemplars", "error"}

    member = rig.store.get(result["member_id"])
    assert (member.name, member.source, member.frames, len(member.vectors)) == ("Ana Example", "photos", 3, 3)
    assert member.consent_at == member.enrolled_at == member.updated_at and member.vector == average(member.vectors)
    assert [(e.type, e.members, e.enrolled_member, e.enrolled_via, e.consent_ack) for e in events] == [
        ("perception.identity.changed", [member.member_id], member.member_id, "photos", True)]
    ana, ben = person(7, 0.05, 0.1, 0.45, 0.95), person(8, 0.55, 0.1, 0.95, 0.95)
    for i in range(2):
        rig.look(Frame(face_in(ana, b"ana#desk%d" % i), face_in(ben, b"ben#desk%d" % i)), [ana, ben])
    assert rig.service.tagged() == {"cam-desk": {7: member.member_id}}  # the photos are enough to know her; Ben stays nobody


def test_fewer_than_three_photos_that_agree_stores_nothing(tmp_path):
    rig = box(tmp_path)
    result, events = rig.service.enroll_photos("m-ana", "Ana", [photo("ana", 0), photo("ana", 1), photo("ana", 2, sharpness=1)], consent_ack=True)
    assert (result["state"], result["error"], result["accepted"], result["needed"]) == ("failed", "not_enough_photos", 2, 3)
    assert events == [] and rig.service.members() == [] and not (tmp_path / "vault" / "identity").exists()
    three_people, _ = rig.service.enroll_photos("m-x", "X", [photo("ana", 0), photo("ben", 1), photo("cy", 2)], consent_ack=True)
    assert three_people["state"] == "failed" and {p["reason"] for p in three_people["photos"]} == {"different"}


def test_photo_enrolment_needs_the_switch_consent_a_decoder_and_a_sane_request(tmp_path):
    rig = box(tmp_path)
    for kw, error in (({"consent_ack": False}, ValueError), ({}, ValueError)):
        with pytest.raises(error, match="consent_ack"):
            rig.service.enroll_photos(None, "Ana", good_photos(), **kw)
    with pytest.raises(ValueError, match="3 to 12"):
        rig.service.enroll_photos(None, "Ana", good_photos(n=2), consent_ack=True)
    with pytest.raises(ValueError, match="3 to 12"):
        rig.service.enroll_photos(None, "Ana", good_photos(n=13), consent_ack=True)
    with pytest.raises(ValueError, match="name"):
        rig.service.enroll_photos(None, "  ", good_photos(), consent_ack=True)
    with pytest.raises(ValueError, match="member_id"):
        rig.service.enroll_photos("Not An Id", "Ana", good_photos(), consent_ack=True)
    with pytest.raises(ValueError, match="consent_ack"):
        rig.service.start_enrollment("m-ana", "Ana")  # the live path asks for it too
    assert rig.service.members() == []
    assert rig.service.photo_reader.decoded == 0  # refused before a single picture was opened

    rig.service.set_enabled(False)
    with pytest.raises(IdentityError, match="switched off"):
        rig.service.enroll_photos(None, "Ana", good_photos(), consent_ack=True)
    no_opencv = Rig(tmp_path / "other")
    no_opencv.service.set_enabled(True)
    assert no_opencv.service.settings()["photos_available"] is False
    with pytest.raises(IdentityError, match="cannot read photos"):
        no_opencv.service.enroll_photos(None, "Ana", good_photos(), consent_ack=True)


def test_adding_photos_later_is_how_the_box_learns_a_face_and_only_the_same_face_is_added(tmp_path):
    rig = box(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")  # live, five looks
    before = rig.store.get("m-ana")
    rig.clock.t += 3600
    result, events = rig.service.enroll_photos("m-ana", None, [photo("ana", 0), photo("ben", 1), photo("ana", 2, brightness=10)],
                                               consent_ack=True)
    assert [p["reason"] for p in result["photos"]] == [None, "not_them", "too_dark"]
    assert (result["state"], result["accepted"], result["needed"], result["exemplars"]) == ("done", 1, 1, 6)  # one good photo is enough to add
    after = rig.store.get("m-ana")
    assert after.vectors[:5] == before.vectors and len(after.vectors) == 6 and after.vector == average(after.vectors)
    assert (after.name, after.enrolled_at, after.source) == (before.name, before.enrolled_at, "live") and after.updated_at > before.updated_at
    assert [(e.enrolled_member, e.enrolled_via) for e in events] == [("m-ana", "photos")]

    for round_ in range(2):  # the newest twelve are kept
        rig.service.enroll_photos("m-ana", None, [photo("ana", 100 + round_ * 10 + i) for i in range(6)], consent_ack=True)
    kept = rig.store.get("m-ana")
    assert len(kept.vectors) == MAX_EXEMPLARS == 12 and kept.vectors[-1] == FakeRecognizer().embed(b"ana#photo115")
    assert before.vectors[0] not in kept.vectors

    only_ben, _ = rig.service.enroll_photos("m-ana", None, [photo("ben", 0)], consent_ack=True)  # nothing of hers in it: nothing changes
    assert only_ben["state"] == "failed" and rig.store.get("m-ana").vectors == kept.vectors
    twice, events = rig.service.enroll_photos("m-ana2", "Ana Again", good_photos("ana"), consent_ack=True)
    assert (twice["state"], twice["error"]) == ("failed", "already_enrolled") and events == []  # one face, one member
    assert [m["member_id"] for m in rig.service.members()] == ["m-ana"]


def test_the_live_path_keeps_its_looks_as_exemplars_and_a_members_own_switch_survives_enrolling_again(tmp_path):
    rig = box(tmp_path)
    events = rig.enroll("m-ana", "Ana Example", b"ana")
    changed = [e for e in events if e.type == "perception.identity.changed"]
    assert [(e.enrolled_member, e.enrolled_via, e.consent_ack) for e in changed] == [("m-ana", "live", True)]
    assert len(rig.store.get("m-ana").vectors) == 5
    rig.service.update_member("m-ana", expressions=True)
    rig.enroll("m-ana", "Ana Example", b"ana")
    assert rig.store.get("m-ana").expressions is True


# --- photos over HTTP: JSON and multipart, never kept ---


def multipart(fields: dict[str, str], files: list[bytes], boundary: str = "----atlasTestBoundary7MA4YWxk") -> tuple[bytes, str]:
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="photos"; filename="p{i}.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
              + data + b"\r\n" for i, data in enumerate(files)]
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def test_a_multipart_body_and_a_json_body_are_read_the_same_way_in_memory():
    binary = [bytes(range(256)) * 3 + b"\r\n--not-the-boundary\r\n", b"\xff\xd8\xff\xe0 jpeg-ish \r\n\r\n tail", b""]
    body, content_type = multipart({"name": "Ana Example", "consent_ack": "true", "member_id": "m-ana", "ignored": "x"}, binary)
    fields, photos = parse_photos(content_type, body)
    assert fields == {"name": "Ana Example", "member_id": "m-ana", "consent_ack": True, "circle_member_id": None}
    assert photos == binary  # byte for byte

    as_json = json.dumps({"name": "Ana Example", "member_id": "m-ana", "consent_ack": True,
                          "photos": [b64(binary[0]), "data:image/jpeg;base64," + b64(binary[1]), ""]}).encode()
    assert parse_photos("application/json; charset=utf-8", as_json) == (fields, binary)
    assert parse_photos("application/json", json.dumps({"photos": [], "consent_ack": "true"}).encode())[0]["consent_ack"] is True
    assert parse_photos("application/json", json.dumps({"photos": [], "circle_member_id": "maya"}).encode())[0] == {
        "name": None, "member_id": None, "consent_ack": False, "circle_member_id": "maya"}
    for content_type, bad in (("application/json", b"{not json"), ("application/json", b"[]"), ("text/plain", b"hello"),
                              ("application/json", json.dumps({"photos": ["!!not base64!!"]}).encode()),
                              ("application/json", json.dumps({"photos": [1]}).encode()),
                              ("application/json", json.dumps({"photos": [], "path": "/etc"}).encode()),
                              ("application/json", json.dumps({"photos": [], "name": 7}).encode()),
                              ("application/json", json.dumps({"photos": [""] * 13}).encode()),
                              ("multipart/form-data", b"--x--")):
        with pytest.raises(ValueError) as refused:
            parse_photos(content_type, bad)
        assert "not base64!!" not in str(refused.value) and "hello" not in str(refused.value)  # nothing from the body in the message


def needles(photos: list[bytes], vectors: list[list[float]]) -> dict[str, bytes]:
    """Every way a photo, the name or a vector could sit somewhere in plain text."""
    out = {"name": NAME.encode(), "first name": b"Zebulon", "last name": b"Quartermaine", "any photo's pixels": b"PIXELS-OF-"}
    for i, data in enumerate(photos):
        out[f"photo {i}"] = data
        out[f"photo {i}, start"] = data[:48]
        out[f"photo {i} as base64"] = base64.b64encode(data)
        out[f"photo {i} as base64, start"] = base64.b64encode(data)[:48]
    for i, v in enumerate(vectors):
        out[f"vector {i} as float32"] = struct.pack(f"<{len(v)}f", *v)
        out[f"vector {i} as float64, start"] = struct.pack("<2d", *v[:2])
        out[f"vector {i} as JSON"] = json.dumps(v).encode()
        out[f"vector {i}[0] as text"] = repr(v[0]).encode()
    return out


def test_uploaded_photos_are_never_written_logged_published_or_returned(tmp_path, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    bus = EventBus(tmp_path / "events")
    rig = box(tmp_path)
    c = http(rig, bus)
    photos = [photo("zeb", i) for i in range(6)] + [photo("zeb", 6, brightness=5), b"\x00garbage\x00PIXELS-OF-NOTHING"]
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    answers = {}

    r = c.post("/identity/enroll/photos", json={"name": NAME, "consent_ack": True, "photos": [b64(p) for p in photos[:4]]})
    assert r.status_code == 200 and r.json()["state"] == "done", r.text
    member_id = r.json()["member_id"]
    answers["json enrol"] = r.text
    body, content_type = multipart({"member_id": member_id, "consent_ack": "true"}, photos[4:])
    r = c.post("/identity/enroll/photos", content=body, headers={"content-type": content_type})
    assert r.status_code == 200 and [p["reason"] for p in r.json()["photos"]] == [None, None, "too_dark", "unreadable"], r.text
    assert r.json()["exemplars"] == 6
    answers["multipart add"] = r.text
    answers["members"] = c.get("/identity/members").text
    answers["settings"] = c.get("/identity/settings").text

    member = rig.store.get(member_id)
    hay = needles(photos, [member.vector, *member.vectors])
    blob = tmp_path / "vault" / "identity" / f"{member_id}.face"
    plain = Vault(KeyRing(tmp_path)).decrypt("identity", blob.read_bytes())
    assert hay["name"] in plain and hay["vector 0 as JSON"] in plain  # the scan can see: the plain template holds them
    assert not [what for what, needle in hay.items() if what.startswith(("photo", "any photo")) and needle in plain]  # but no photo

    # 1. Files: the only new ones are the template, the vault's key that seals it, and the event log.
    new = sorted(p.relative_to(tmp_path).as_posix() for p in {p for p in tmp_path.rglob("*") if p.is_file()} - before)
    assert [p for p in new if not p.startswith("events/")] == [f"vault/identity/{member_id}.face", "vault/master.key"]
    scanned = 0
    for path in (p for p in tmp_path.rglob("*") if p.is_file()):
        data = path.read_bytes()
        scanned += len(data)
        assert not [what for what, needle in hay.items() if needle in data], f"{path.name} holds something in plain text"
    # 2. The event log: ids, how and consent; nothing else.
    logged = [json.loads(line) for p in (tmp_path / "events").glob("*.jsonl") for line in p.read_text().splitlines()]
    assert [(e["enrolled_member"], e["enrolled_via"], e["consent_ack"]) for e in logged] == [(member_id, "photos", True)] * 2
    # 3. Answers: the name only from GET /identity/members; never a photo or a vector.
    for which, text in answers.items():
        found = [what for what, needle in hay.items() if needle in text.encode()]
        assert found == (["name", "first name", "last name"] if which == "members" else []), f"{which} returned {found}"
    # 4. Log lines.
    lines = "\n".join(r.getMessage() for r in caplog.records).encode()
    assert not [what for what, needle in hay.items() if needle in lines]
    with capsys.disabled():
        print(f"\n  photo-leak scan: {len(photos)} uploaded photos (JSON and multipart), {scanned} bytes of files, {len(answers)} API "
              f"answers, {len(caplog.records)} log lines, {len(hay)} needles: no photo anywhere; only the encrypted template was written")


def test_the_photo_endpoint_refuses_what_it_should(tmp_path):
    bus = EventBus(tmp_path / "events")
    rig = box(tmp_path)
    c, auth = http(rig, bus, token="s3cret"), {"Authorization": "Bearer s3cret"}
    good = {"name": "Ana", "consent_ack": True, "photos": [b64(p) for p in good_photos()]}
    assert c.post("/identity/enroll/photos", json=good).status_code == 401
    assert c.put("/identity/members/m-ana", json={"expressions": True}).status_code == 401
    no_consent = c.post("/identity/enroll/photos", json=good | {"consent_ack": False}, headers=auth)
    assert no_consent.status_code == 422 and "consent_ack" in no_consent.json()["detail"]
    assert c.post("/identity/enroll/start", json={"name": "Ana"}, headers=auth).status_code == 422  # the live path too
    assert c.post("/identity/enroll/start", json={"name": "Ana", "consent_ack": True}, headers=auth).status_code == 200
    assert c.post("/identity/enroll/photos", json=good | {"photos": good["photos"][:2]}, headers=auth).status_code == 422
    assert c.post("/identity/enroll/photos", content=b"{", headers=auth | {"content-type": "application/json"}).status_code == 422
    assert c.post("/identity/enroll/photos", content=b"x", headers=auth | {"content-type": "image/jpeg"}).status_code == 422
    too_big = c.post("/identity/enroll/photos", content=b"x", headers=auth | {"content-length": str(MAX_BODY_BYTES + 1)})
    assert too_big.status_code == 413
    assert rig.service.photo_reader.decoded == 0 and rig.service.members() == []

    c.put("/identity/settings", json={"identity_enabled": False}, headers=auth)
    off = c.post("/identity/enroll/photos", json=good, headers=auth)
    assert off.status_code == 409 and "switched off" in off.json()["detail"]  # like the live path
    assert rig.service.photo_reader.decoded == 0


# --- a guided scan on the phone: one still at a time, the same checks, live feedback ---


def test_a_scan_takes_stills_one_at_a_time_says_what_to_change_and_wants_more_than_one_pose(tmp_path):
    rig = box(tmp_path)
    with pytest.raises(ValueError, match="consent_ack"):
        rig.service.scan_start(None, "Ana Example")
    with pytest.raises(ValueError, match="name"):
        rig.service.scan_start(None, None, consent_ack=True)
    start = rig.service.scan_start(None, "Ana Example", consent_ack=True)
    session = start["session"]
    assert (start["count"], start["needed"], start["want"], start["prompt"], start["done"]) == (0, SCAN_FRAMES, "front", "look straight at the camera", False)
    assert "name" not in start

    def send(n, **kw):
        return rig.service.scan_frame(session, photo("ana", n, **kw))

    dark = send(0, brightness=10)
    assert (dark["accepted"], dark["reason"], dark["message"], dark["count"]) == (False, "too_dark", "too dark: find better light", 0)
    assert send(1, w=40)["message"] == "too small: move closer" and send(2, sharpness=3)["reason"] == "blurry"
    assert rig.service.scan_frame(session, photo("ana", 3, others=("ben",)))["reason"] == "two_faces"
    assert rig.service.scan_frame(session, b"junk")["reason"] == "unreadable"

    got = [send(10 + i) for i in range(4)]
    assert [f["accepted"] for f in got] == [True] * 4 and got[-1]["poses"] == {"front": 4, "left": 0, "right": 0, "up": 0}
    assert (got[2]["want"], got[2]["prompt"]) == ("left", "turn a little to your left")  # three straight on: now a turn
    same = send(20)  # a fifth straight on: the template would learn nothing from it
    assert (same["accepted"], same["reason"], same["count"]) == (False, "enough_like_this", 4)
    assert same["message"] == "got enough like this: turn a little to your left"
    stranger = rig.service.scan_frame(session, photo("ben", 21, yaw=0.2))
    assert (stranger["accepted"], stranger["reason"]) == (False, "different")  # someone else leaned in

    left = [send(30 + i, yaw=0.2) for i in range(2)]
    assert [f["pose"] for f in left] == ["left", "left"] and left[-1]["want"] == "right"
    right = send(40, yaw=-0.2)
    up = send(41, drop=0.3)
    assert (right["pose"], up["pose"], up["count"], up["done"], up["prompt"]) == ("right", "up", 8, True, None)
    assert send(42, yaw=-0.2)["reason"] == "enough_like_this"  # complete: nothing more is taken
    assert rig.service.members() == [] and not (tmp_path / "vault" / "identity").exists()  # nothing is stored until it is finished

    result, events = rig.service.scan_finish(session)
    assert (result["state"], result["accepted"], result["exemplars"], result["error"]) == ("done", 8, 8, None)
    member = rig.store.get(result["member_id"])
    assert (member.name, member.source, len(member.vectors)) == ("Ana Example", "scan", 8) and member.consent_at
    assert [(e.enrolled_member, e.enrolled_via, e.consent_ack) for e in events] == [(member.member_id, "scan", True)]
    with pytest.raises(KeyError):
        rig.service.scan_frame(session, photo("ana", 50))  # the scan is closed
    with pytest.raises(KeyError):
        rig.service.scan_finish(session)
    ana = person(7)
    for i in range(2):
        rig.look(Frame(face_in(ana, b"ana#desk%d" % i)), [ana])
    assert rig.service.tagged() == {"cam-desk": {7: member.member_id}}


def test_a_scan_can_add_to_someone_enrolled_times_out_and_too_few_stills_store_nothing(tmp_path):
    rig = box(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    session = rig.service.scan_start("m-ana", None, consent_ack=True)["session"]
    assert rig.service.scan_frame(session, photo("ben", 0))["reason"] == "not_them"
    assert rig.service.scan_frame(session, photo("ana", 1))["accepted"] is True
    result, events = rig.service.scan_finish(session)  # one good still is enough to add
    assert (result["state"], result["exemplars"]) == ("done", 6) and rig.store.get("m-ana").source == "live"
    assert [e.enrolled_via for e in events] == ["scan"]

    session = rig.service.scan_start(None, "Ben Example", consent_ack=True)["session"]
    for i in range(2):
        rig.service.scan_frame(session, photo("ben", i))
    result, events = rig.service.scan_finish(session)
    assert (result["state"], result["error"], events) == ("failed", "not_enough_photos", []) and len(rig.service.members()) == 1

    old = rig.service.scan_start(None, "Ben Example", consent_ack=True)["session"]
    new = rig.service.scan_start(None, "Cy Example", consent_ack=True)["session"]  # a new scan replaces an unfinished one
    with pytest.raises(KeyError):
        rig.service.scan_frame(old, photo("ben", 0))
    rig.clock.t += 181
    with pytest.raises(IdentityError, match="too long"):
        rig.service.scan_frame(new, photo("cy", 0))
    again = rig.service.scan_start(None, "Cy Example", consent_ack=True)["session"]
    rig.service.scan_cancel(again)
    with pytest.raises(KeyError):
        rig.service.scan_finish(again)
    rig.service.set_enabled(False)
    with pytest.raises(IdentityError, match="switched off"):
        rig.service.scan_start(None, "Cy Example", consent_ack=True)


def test_a_scan_over_http_never_keeps_or_echoes_a_still(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    bus = EventBus(tmp_path / "events")
    rig = box(tmp_path)
    c, auth = http(rig, bus, token="s3cret"), {"Authorization": "Bearer s3cret"}
    assert c.post("/identity/enroll/scan/start", json={"name": NAME, "consent_ack": True}).status_code == 401
    assert c.post("/identity/enroll/scan/start", json={"name": NAME}, headers=auth).status_code == 422
    session = c.post("/identity/enroll/scan/start", json={"name": NAME, "consent_ack": True}, headers=auth).json()["session"]
    assert c.post("/identity/enroll/scan/frame", json={"session": session, "photo": "x"}).status_code == 401
    stills = [photo("zeb", i, yaw=y, drop=d) for i, (y, d) in enumerate([(0, 0.5)] * 3 + [(0.2, 0.5)] * 2 + [(-0.2, 0.5)] * 2 + [(0, 0.3)])]
    stills.append(photo("zeb", 99, brightness=4))
    answers = []
    for still in [stills[-1], *stills[:-1]]:
        r = c.post("/identity/enroll/scan/frame", json={"session": session, "photo": "data:image/jpeg;base64," + b64(still)}, headers=auth)
        assert r.status_code == 200, r.text
        answers.append(r.text)
    assert json.loads(answers[0])["message"].startswith("too dark") and json.loads(answers[-1])["done"] is True
    bad = [c.post("/identity/enroll/scan/frame", content=b"{", headers=auth),
           c.post("/identity/enroll/scan/frame", json={"session": session, "photo": "!!not base64!!"}, headers=auth),
           c.post("/identity/enroll/scan/frame", json={"session": session, "photo": b64(stills[0]), "extra": 1}, headers=auth)]
    assert [r.status_code for r in bad] == [422, 422, 422]
    assert c.post("/identity/enroll/scan/frame", json={"session": "nope", "photo": b64(stills[0])}, headers=auth).status_code == 404
    done = c.post("/identity/enroll/scan/finish", json={"session": session}, headers=auth)
    assert done.status_code == 200 and (done.json()["state"], done.json()["exemplars"]) == ("done", 8)
    assert c.post("/identity/enroll/scan/finish", json={"session": session}, headers=auth).status_code == 404
    assert c.post("/identity/enroll/scan/cancel", json={"session": session}, headers=auth).json() == {"cancelled": True}
    answers += [r.text for r in bad] + [done.text]

    member = rig.store.get(done.json()["member_id"])
    hay = needles(stills, [member.vector, *member.vectors])
    for path in (p for p in tmp_path.rglob("*") if p.is_file()):
        assert not [what for what, needle in hay.items() if needle in path.read_bytes()], f"{path.name} holds something in plain text"
    lines = "\n".join(r.getMessage() for r in caplog.records).encode()
    for text in [*answers, lines.decode()]:
        assert not [what for what, needle in hay.items() if needle in text.encode()]
    assert "not base64!!" not in "".join(answers)
    logged = [json.loads(line) for p in (tmp_path / "events").glob("*.jsonl") for line in p.read_text().splitlines()]
    assert [(e["type"], e["enrolled_via"]) for e in logged] == [("perception.identity.changed", "scan")]  # one fact, at the end


# --- a person can be linked to a Circle member: an id, nothing else ---


def test_a_member_can_be_linked_to_a_circle_member_by_id_and_seen_as_here_now(tmp_path):
    bus = EventBus(tmp_path / "events")
    rig = box(tmp_path)
    app = FastAPI()
    app.include_router(identity_router(rig.service, bus, circle_members=lambda: ["hemdutt", "maya", "riley"]))
    c = TestClient(app)
    good = {"name": "Maya Example", "consent_ack": True, "photos": [b64(p) for p in good_photos("maya")]}
    refused = c.post("/identity/enroll/photos", json=good | {"circle_member_id": "nobody"})
    assert refused.status_code == 422 and "no such member" in refused.json()["detail"] and rig.service.members() == []
    assert c.post("/identity/enroll/scan/start", json={"name": "X", "consent_ack": True, "circle_member_id": "nobody"}).status_code == 422
    assert c.post("/identity/enroll/start", json={"name": "X", "consent_ack": True, "circle_member_id": "nobody"}).status_code == 422
    assert c.post("/identity/enroll/photos", json=good | {"circle_member_id": "../etc"}).status_code == 422
    member_id = c.post("/identity/enroll/photos", json=good | {"circle_member_id": "maya"}).json()["member_id"]
    assert c.get("/identity/members").json()[0]["circle_member_id"] == "maya"
    assert rig.store.get(member_id).circle_member_id == "maya"

    rig.enroll("m-ana", "Ana Example", b"ana")  # someone with no link
    assert c.get("/identity/present").json() == {"present": []}
    maya, ana = person(7, 0.05, 0.1, 0.45, 0.95), person(8, 0.55, 0.1, 0.95, 0.95)
    events = []
    for i in range(2):
        events += rig.look(Frame(face_in(maya, b"maya#%d" % i), face_in(ana, b"ana#%d" % i)), [maya, ana])
    seen = {e.member_id: e.circle_member_id for e in events if e.type == "perception.person.identified"}
    assert seen == {member_id: "maya", "m-ana": None}  # the event that names her carries the link; an id, nothing else
    assert sorted(c.get("/identity/present").json()["present"], key=lambda p: p["circle_member_id"] or "") == [
        {"member_id": "m-ana", "circle_member_id": None, "cameras": ["cam-desk"]},
        {"member_id": member_id, "circle_member_id": "maya", "cameras": ["cam-desk"]}]

    assert c.put("/identity/members/m-ana", json={"circle_member_id": "nobody"}).status_code == 422
    assert c.put("/identity/members/m-ana", json={"circle_member_id": "riley"}).json()["circle_member_id"] == "riley"
    assert c.put("/identity/members/m-ana", json={"expressions": True}).json()["circle_member_id"] == "riley"  # other changes keep it
    assert c.put("/identity/members/m-ana", json={"circle_member_id": ""}).json()["circle_member_id"] is None  # unlinked
    rig.service.enroll_photos(member_id, None, [photo("maya", 50)], consent_ack=True)  # adding photos keeps the link
    assert rig.store.get(member_id).circle_member_id == "maya"

    logged = "".join(p.read_text() for p in (tmp_path / "events").glob("*.jsonl"))
    rows = [json.loads(line) for line in logged.splitlines()]
    assert [e["circle_member_id"] for e in rows if e.get("enrolled_member") == member_id] == ["maya"]
    assert "vector" not in logged and "Maya Example" not in logged  # the link is an id; nothing of the face or the name with it

    rig.service.set_enabled(False)
    assert c.get("/identity/present").json() == {"present": []}
    no_circle = TestClient(FastAPI())
    no_circle.app.include_router(identity_router(rig.service, bus, circle_members=lambda: None))
    rig.service.set_enabled(True)
    assert no_circle.put("/identity/members/m-ana", json={"circle_member_id": "anyone"}).json()["circle_member_id"] == "anyone"


def test_a_template_from_before_the_link_existed_loads_without_one(tmp_path):
    rig = box(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    vault = Vault(KeyRing(tmp_path))
    path = tmp_path / "vault" / "identity" / "m-ana.face"
    raw = json.loads(vault.decrypt("identity", path.read_bytes()))
    del raw["circle_member_id"]
    path.write_bytes(vault.encrypt("identity", json.dumps(raw).encode()))
    assert IdentityStore(tmp_path, vault).get("m-ana").circle_member_id is None


# --- apparent expression: the pure part ---


def test_the_head_pose_comes_from_the_five_landmarks():
    straight = head_pose(row(100, 100))
    assert not straight.away and not straight.down and straight.nose_drop == pytest.approx(0.375)
    assert head_pose(row(100, 100, facing=False)).away
    r = row(100, 100)
    r[9] = r[5] + 0.75 * (r[11] - r[5])  # the nose tip three quarters of the way from the eye line to the mouth line
    assert head_pose(r).down and not head_pose(r).away
    r[8] += 40  # and the nose far off to one side: turned away
    assert head_pose(r).away
    assert sum(softmax([0.5, -1.0, 3.5, 0.0])) == pytest.approx(1.0) and softmax([1000.0, 0.0])[0] == pytest.approx(1.0)


def summaries(run: ExpressionRun, looks: list[tuple[float, str | None]], confidence: float = 0.9):
    out = []
    for at, label in looks:
        out += [(round((s.start - T0).total_seconds()), round((s.end - T0).total_seconds()), s.dominant, s.samples)
                for s in run.add(T0 + timedelta(seconds=at), label, confidence)]
    return out


def test_looks_become_one_fact_on_a_change_and_one_every_few_minutes_never_one_per_look():
    run = ExpressionRun(**FAST)
    neutral = [(t, "neutral") for t in range(0, 400, 2)]
    out = summaries(run, neutral)
    # the first window is news (nothing was known); after it the same look is reported once per three minutes
    assert out == [(0, 30, "neutral", 16), (32, 222, "neutral", 96)]  # 200 looks, two facts
    assert sum(o[3] for o in out) + (run.window.samples if run.window else 0) + (run.pending.samples if run.pending else 0) == 200

    run = ExpressionRun(**FAST)
    mixed = [(t, "neutral") for t in range(0, 100, 2)] + [(t, "sad") for t in range(100, 200, 2)]
    out = summaries(run, mixed)
    assert [o[2] for o in out][:1] == ["neutral"] and "sad" in [o[2] for o in out]
    changed = next(o for o in out if o[2] == "sad")
    assert changed[0] < 130  # reported within a window or two of the change, not at the next heartbeat
    counted = sum(o[3] for o in out) + sum(s.samples for s in run.finish())
    assert 95 <= counted <= 100  # no look is counted twice; only a last window of fewer than five looks is dropped


def test_too_few_or_too_unsure_looks_say_nothing_and_an_idle_stretch_is_closed():
    run = ExpressionRun(**FAST)
    assert summaries(run, [(t, "happy") for t in (0, 10, 20, 31)]) == []  # four looks in a window: dropped
    assert summaries(run, [(t, "happy") for t in range(40, 80, 2)], confidence=0.2) == []  # unsure: looks, not samples
    assert summaries(run, [(t, None) for t in range(80, 120, 2)]) == []  # turned away
    run = ExpressionRun(**FAST)
    assert summaries(run, [(t, "happy") for t in range(0, 20, 2)]) == []
    assert summaries(run, [(100, "neutral")]) == [(0, 18, "happy", 10)]  # nobody for over a minute: what there was goes out
    s = ExpressionRun(**FAST)
    for i, (label, away, down) in enumerate([("happy", False, False), ("happy", False, True), (None, True, False), ("sad", False, True)] * 5):
        assert s.add(T0 + timedelta(seconds=i), label, 0.8, away=away, down=down, people=1 + i % 3) == []
    got = s.finish()
    assert (got[0].looks, got[0].samples, got[0].away, got[0].down, got[0].people) == (20, 15, 5, 10, 3)
    assert got[0].shares == {"happy": 0.667, "sad": 0.333} and got[0].confidence == 0.8 and got[0].dominant == "happy"


# --- apparent expression: on the box ---


def watch(rig: Rig, seconds: int, tracks, faces, moods=None, step: float = 1.0) -> list:
    """`seconds` of looks, one per `step`: every face keeps its place, `moods` says how each looks."""
    out = []
    for i in range(int(seconds / step)):
        out += rig.look(Mood(*[(who + b"#%d" % i, r) for who, r in faces], moods=moods or {}), tracks, step=step)
    return out


def observed(events):
    return [e for e in events if e.type == "perception.expression.observed"]


def test_expressions_are_read_only_for_a_member_who_switched_it_on_and_published_as_stretches(tmp_path):
    rig = box(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    rig.enroll("m-ben", "Ben Example", b"ben")
    ana, ben = person(7, 0.05, 0.1, 0.45, 0.95), person(8, 0.55, 0.1, 0.95, 0.95)
    faces = [(b"ana", face_in(ana, b"")[1]), (b"ben", face_in(ben, b"")[1])]
    reader = rig.service.expression_reader

    assert observed(watch(rig, 120, [ana, ben], faces, {b"ana": "sad"})) == [] and reader.calls == 0  # nobody switched it on
    assert rig.service.tagged() == {"cam-desk": {7: "m-ana", 8: "m-ben"}}

    changed = rig.service.update_member("m-ana", expressions=True)
    assert [(e.type, e.expressions, e.members) for e in changed] == [("perception.identity.changed", ["m-ana"], ["m-ana", "m-ben"])]
    events = watch(rig, 400, [ana, ben], faces, {b"ana": "sad", b"ben": "happy"})
    got = observed(events)
    assert {e.member_id for e in got} == {"m-ana"} and {e.dominant for e in got} == {"sad"}  # Ben did not switch it on: never read
    assert reader.calls == 200  # one face every two seconds, not every frame
    assert 2 <= len(got) <= 3  # the first window, then one per three minutes: not 400, not 200
    first = got[0]
    assert (first.camera_id, first.shares, first.confidence, first.people) == ("cam-desk", {"sad": 1.0}, 0.9, None)
    assert first.samples == first.looks >= 5 and first.head_down == 0 and first.turned_away == 0
    assert timedelta(seconds=28) <= first.window_end - first.window_start <= timedelta(seconds=32) and first.ts >= first.window_end
    assert sum(e.samples for e in got) <= 200

    brighter = observed(watch(rig, 60, [ana, ben], faces, {b"ana": "happy"}))  # a change is reported within the minute
    assert [e.dominant for e in brighter][-1] == "happy" and len(brighter) <= 2

    for e in got + brighter:  # the fact, and nothing of the face
        raw = json.loads(e.model_dump_json())
        assert not {"track_id", "frame_ref", "vector", "landmarks", "box", "name"} & set(raw)
        assert parse_event(raw).type == "perception.expression.observed" and set(raw["shares"]) <= set(LABELS)

    calls = reader.calls
    rig.service.update_member("m-ana", expressions=False)  # off again: reading stops, and the half-counted window is dropped
    assert observed(watch(rig, 200, [ana, ben], faces, {b"ana": "angry"})) == [] and reader.calls == calls


def test_a_face_turned_away_is_counted_as_a_look_but_not_read(tmp_path):
    rig = box(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    rig.service.update_member("m-ana", expressions=True)
    ana = person(7)
    watch(rig, 4, [ana], [(b"ana", face_in(ana, b"")[1])])
    calls = rig.service.expression_reader.calls
    events = watch(rig, 100, [ana], [(b"ana", face_in(ana, b"", facing=False)[1])])
    assert observed(events) == [] and rig.service.expression_reader.calls == calls  # a profile is not put through the model


def test_the_everyone_mode_needs_a_confirmation_and_counts_strangers_together_as_nobody(tmp_path):
    rig = box(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    with pytest.raises(ValueError, match="confirm"):
        rig.service.set_everyone(True)
    assert rig.service.everyone is False
    changed = rig.service.set_everyone(True, confirm=True)
    assert [(e.enabled, e.everyone, e.members) for e in changed] == [(True, True, ["m-ana"])]
    assert rig.service.settings()["everyone"] is True

    ana, guest1, guest2 = person(7, 0.02, 0.1, 0.3, 0.95), person(8, 0.35, 0.1, 0.62, 0.95), person(9, 0.68, 0.1, 0.98, 0.95)
    faces = [(b"ana", face_in(ana, b"")[1]), (b"guest1", face_in(guest1, b"")[1]), (b"guest2", face_in(guest2, b"")[1])]
    events = watch(rig, 300, [ana, guest1, guest2], faces, {b"ana": "angry", b"guest1": "happy", b"guest2": "happy"})
    got = observed(events)
    assert got and {e.member_id for e in got} == {None}  # Ana is enrolled and did not switch hers on: she is not read at all
    assert {e.dominant for e in got} == {"happy"} and {e.people for e in got} == {2}
    assert rig.service.tagged() == {"cam-desk": {7: "m-ana"}}  # no guest was tagged, tracked as a person or given an id
    assert [e.type for e in events if e.type.startswith("perception.person")] == ["perception.person.identified"]
    for e in got:
        assert not {"track_id", "vector", "frame_ref", "name"} & set(json.loads(e.model_dump_json()))
    assert sorted(p.name for p in (tmp_path / "vault" / "identity").iterdir()) == ["m-ana.face"]  # no template of a guest
    assert [m.member_id for m in rig.store.members()] == ["m-ana"]

    rig.service.update_member("m-ana", expressions=True)  # with hers on: hers is hers, and never part of the room's
    both = observed(watch(rig, 300, [ana, guest1, guest2], faces, {b"ana": "angry", b"guest1": "happy", b"guest2": "happy"}))
    assert {(e.member_id, e.dominant, e.people) for e in both} == {("m-ana", "angry", None), (None, "happy", 2)}

    off = rig.service.set_everyone(False)
    assert off[0].everyone is False
    after = observed(watch(rig, 200, [ana, guest1, guest2], faces, {b"ana": "angry", b"guest1": "sad", b"guest2": "sad"}))
    assert {e.member_id for e in after} == {"m-ana"}  # the room is no longer counted


def test_the_everyone_mode_cannot_outlive_recognition_and_needs_the_expression_model(tmp_path):
    rig = box(tmp_path)
    rig.service.set_everyone(True, confirm=True)
    guest = person(8)
    got = observed(watch(rig, 100, [guest], [(b"guest", face_in(guest, b"")[1])], {b"guest": "surprised"}))
    assert got and got[0].member_id is None and got[0].people == 1  # nobody is enrolled: every face is nobody
    rig.service.set_enabled(False)
    assert rig.service.everyone is False and rig.service.settings()["everyone"] is False
    assert observed(watch(rig, 100, [guest], [(b"guest", face_in(guest, b"")[1])])) == []
    with pytest.raises(IdentityError, match="switched off"):
        rig.service.set_everyone(True, confirm=True)
    rig.service.set_enabled(True)
    assert rig.service.everyone is False  # switching recognition back on does not bring it back

    plain = Rig(tmp_path / "plain")  # no expression model on this box
    plain.service.set_enabled(True)
    plain.enroll("m-ana", "Ana Example", b"ana")
    assert plain.service.settings()["expressions_available"] is False
    with pytest.raises(IdentityError, match="expression model"):
        plain.service.set_everyone(True, confirm=True)
    with pytest.raises(IdentityError, match="expression model"):
        plain.service.update_member("m-ana", expressions=True)
    assert build_expression_reader(tmp_path / "missing.onnx") is None  # nothing is downloaded: no file, no feature


# --- the API around it, forgetting, wiping ---


def test_the_switches_the_series_and_forget_me_over_http(tmp_path):
    bus = EventBus(tmp_path / "events")
    rig = box(tmp_path)
    bus.subscribe("perception.identity", rig.service.on_changed, replay=True)
    bus.subscribe("perception.expression", rig.service.on_expression, replay=True)
    c = http(rig, bus)
    assert c.get("/identity/settings").json() == {
        "identity_enabled": True, "available": True, "recognizer": "fake", "everyone": False, "expressions_available": True,
        "expression_reader": "fake", "photos_available": True}
    member_id = c.post("/identity/enroll/photos", json={"name": NAME, "consent_ack": True,
                                                         "photos": [b64(p) for p in good_photos("zeb")]}).json()["member_id"]
    assert c.put("/identity/settings", json={"everyone": True}).status_code == 422  # not without the confirmation
    assert c.put("/identity/settings", json={}).status_code == 422
    assert c.put("/identity/settings", json={"everyone": True, "confirm": True}).json()["everyone"] is True
    row_ = c.put(f"/identity/members/{member_id}", json={"expressions": True}).json()
    assert (row_["expressions"], row_["photos"], row_["source"], row_["name"]) == (True, 5, "photos", NAME)
    assert c.put("/identity/members/m-nobody", json={"expressions": True}).status_code == 404
    assert c.put(f"/identity/members/{member_id}", json={"name": " "}).status_code == 422

    zeb, guest = person(7, 0.05, 0.1, 0.45, 0.95), person(8, 0.55, 0.1, 0.95, 0.95)
    faces = [(b"zeb", face_in(zeb, b"")[1]), (b"guest", face_in(guest, b"")[1])]
    asyncio.run(bus.publish_all(watch(rig, 300, [zeb, guest], faces, {b"zeb": "angry", b"guest": "happy"})))
    asyncio.run(bus.publish_all(watch(rig, 100, [zeb, guest], faces, {b"zeb": "happy", b"guest": "happy"})))

    mine = c.get("/identity/expressions", params={"member_id": member_id, "hours": 6}).json()
    assert mine["member_id"] == member_id and mine["samples"] > 100 and mine["windows"]
    assert [w["looked"] for w in mine["windows"]][0] == "tense" and mine["windows"][-1]["looked"] == "upbeat"
    assert set(mine["looked"]) == {"tense", "upbeat"} and mine["looked"]["tense"] > 0.5 and 0.5 < mine["strained_share"] < 1
    assert "not an emotion" in mine["note"] and "people" not in mine["windows"][0]
    room = c.get("/identity/expressions").json()
    assert room["member_id"] is None and room["windows"][0]["people"] == 1 and room["looked"] == {"upbeat": 1.0}
    assert c.get("/identity/expressions", params={"member_id": member_id, "hours": 0}).status_code == 422
    assert c.get("/identity/expressions", params={"member_id": "m-nobody"}).status_code == 404

    text = json.dumps([mine, room])
    assert "vector" not in text and "track" not in text and NAME not in text

    assert c.delete(f"/identity/members/{member_id}").json() == {"removed": member_id}  # forget me
    assert c.get("/identity/expressions", params={"member_id": member_id}).status_code == 404
    assert not [o for o in rig.service._observed if o.member_id == member_id]  # gone from what the app can answer ...
    in_log = "".join(p.read_text() for p in (tmp_path / "events").glob("*.jsonl"))
    assert in_log.count('"perception.expression.observed"') >= 4 and member_id in in_log  # ... and, honestly, still in the log
    assert not (tmp_path / "vault" / "identity" / f"{member_id}.face").exists()

    replayed = box(tmp_path / "again")  # a restart replays the log: the forgotten member's stretches do not come back
    bus2 = EventBus(tmp_path / "events")
    bus2.subscribe("perception.identity", replayed.service.on_changed, replay=True)
    bus2.subscribe("perception.expression", replayed.service.on_expression, replay=True)
    asyncio.run(bus2.replay_into())
    assert replayed.service._observed and {o.member_id for o in replayed.service._observed} == {None}


def test_wiping_everything_removes_exemplars_switches_and_the_series(tmp_path):
    rig = box(tmp_path)
    rig.service.enroll_photos("m-ana", "Ana Example", good_photos(), consent_ack=True)
    rig.service.update_member("m-ana", expressions=True)
    rig.service.set_everyone(True, confirm=True)
    ana = person(7)
    for e in observed(watch(rig, 100, [ana], [(b"ana", face_in(ana, b"")[1])])):
        rig.service.on_expression(e)
    assert rig.service._observed and rig.service.everyone

    wiper = Wiper(tmp_path)
    report = wiper.wipe("all")
    rig.service.on_wipe(type("Wipe", (), {"scope": "all"})())
    assert report.identity_removed == 1 and report.identity_switched_off and "recognition switches" in report.summary()
    assert not (tmp_path / "vault" / "identity.json").exists() and list((tmp_path / "vault" / "identity").iterdir()) == []
    assert (tmp_path / "vault" / "master.key").exists()
    assert rig.service.settings() | {"reason": None} == {
        "identity_enabled": False, "available": True, "recognizer": "fake", "everyone": False, "expressions_available": True,
        "expression_reader": "fake", "photos_available": True, "reason": None}
    assert rig.service.members() == [] and rig.service._observed == [] and rig.service._runs == {}
    assert observed(watch(rig, 100, [ana], [(b"ana", face_in(ana, b"")[1])])) == []  # the running box reads nobody
    assert Wiper(tmp_path).wipe("memory").identity_switched_off is False


def test_the_state_is_announced_when_the_log_misses_a_switch(tmp_path):
    rig = box(tmp_path)
    events = rig.enroll("m-ana", "Ana Example", b"ana") + rig.service.update_member("m-ana", expressions=True)
    restarted = box(tmp_path)
    for e in events:
        if e.type == "perception.identity.changed":
            restarted.service.on_changed(e)  # replay
    assert restarted.service.announce() == []
    restarted.service.store.set_everyone(True)  # the vault folder says more than the log does
    said = restarted.service.announce()
    assert [(e.enabled, e.members, e.everyone, e.expressions, e.enrolled_member) for e in said] == [(True, ["m-ana"], True, ["m-ana"], None)]


# --- blur is untouched, with everything switched on ---


class TwoPeople:
    """A 20 fps camera: Ana (enrolled) in person 7's box and a guest in person 8's. The pixels hold both faces."""

    def __init__(self):
        self.t0 = time.monotonic()

    def latest(self):
        n = int((time.monotonic() - self.t0) / 0.05) + 1
        return n, 1_700_000_000_000 + n * 50, two_people(n)


PEOPLE = [person(7, 0.05, 0.1, 0.45, 0.95), person(8, 0.55, 0.1, 0.95, 0.95)]


def two_people(n: int) -> Mood:
    a, g = b"ana#%04d" % n, b"guest#%04d" % n
    return Mood(face_in(PEOPLE[0], a), face_in(PEOPLE[1], g), pixels=b"desk" * 8 + a + b"wall" * 8 + g + b"door" * 4,
                moods={b"ana": "sad", b"guest": "happy"})


def blurred(frame: Frame) -> bytes:
    """What a keyframe store writes for this frame: every face scrambled (stands in for FaceBlur + JPEG)."""
    out = bytes(frame.pixels)
    for who, _ in frame.faces:
        out = out.replace(who, b"#" * len(who))
    return out


def run_pipeline(data_dir, rig: Rig, hooked: bool, seconds: float = 1.2) -> dict[int, bytes]:
    folder = data_dir / "frames" / "cam-desk"
    folder.mkdir(parents=True)

    def store(camera_id, ts_ms, frame):
        (folder / f"{ts_ms}.jpg").write_bytes(blurred(frame))
        return f"frames:{camera_id}:{ts_ms}"

    class Counter:
        finder = rig.finder

        def __call__(self, frame):
            return len(frame.faces), len(frame.faces)

    async def main():
        bus = EventBus(data_dir / "events")
        pipe = TrackingPipeline("cam-desk", TwoPeople(), lambda frame: list(PEOPLE), bus, asyncio.get_running_loop(),
                                store=store, detect_fps=20, observe_every_s=0.1, keyframe_every_s=0.1, faces=Counter())
        if hooked:
            rig.service.attach({"cam-desk": pipe}, asyncio.get_running_loop(), bus)
        pipe.start()
        await asyncio.sleep(seconds)
        pipe.stop()
        await asyncio.sleep(0.05)
        assert pipe.errors == 0 and rig.service.errors == 0

    asyncio.run(main())
    return {int(p.stem): p.read_bytes() for p in folder.glob("*.jpg")}


def test_stored_frames_are_blurred_the_same_with_recognition_expressions_and_everyone_all_on(tmp_path):
    quick = {"window_s": 0.2, "heartbeat_s": 0.5, "min_samples": 2, "idle_s": 5.0}
    rig = box(tmp_path / "box", retry_s=0.05, expression_every_s=0.03, expression_settings=quick)
    rig.enroll("m-ana", "Ana Example", b"ana")
    rig.service.update_member("m-ana", expressions=True)
    rig.service.set_everyone(True, confirm=True)
    assert rig.service.settings() | {"recognizer": 0, "expression_reader": 0} == {
        "identity_enabled": True, "available": True, "recognizer": 0, "everyone": True, "expressions_available": True,
        "expression_reader": 0, "photos_available": True}
    rig.service._clock = time.monotonic  # the pipeline runs on the real clock
    rig.service._now = lambda: T0 + timedelta(seconds=time.monotonic())
    rig.service._tags.clear()

    off = run_pipeline(tmp_path / "off", rig, hooked=False)
    on = run_pipeline(tmp_path / "on", rig, hooked=True)

    # everything really ran: Ana was recognised, her expression and the room's were read and published
    assert rig.service.tagged() == {"cam-desk": {7: "m-ana"}} and rig.service.expression_reader.calls > 6
    log = [json.loads(line) for p in (tmp_path / "on" / "events").glob("*.jsonl") for line in p.read_text().splitlines()]
    seen = {(e["member_id"], e["dominant"]) for e in log if e["type"] == "perception.expression.observed"}
    assert seen == {("m-ana", "sad"), (None, "happy")}
    assert not [e for e in (json.loads(line) for p in (tmp_path / "off" / "events").glob("*.jsonl") for line in p.read_text().splitlines())
                if e["type"].startswith(("perception.expression", "perception.person"))]

    assert len(off) >= 3 and len(on) >= 3
    for written in (off, on):
        for ts_ms, data in written.items():
            n = (ts_ms - 1_700_000_000_000) // 50
            assert data == blurred(two_people(n))  # frame n's keyframe is the blurred frame n, whatever is switched on
            assert b"ana#" not in data and b"guest#" not in data  # no face in a stored frame: not hers, not the guest's
    assert all(on[ts] == off[ts] for ts in on.keys() & off.keys())
    raw = "".join(p.read_text() for p in (tmp_path / "on" / "events").glob("*.jsonl"))
    assert "ana#" not in raw and "guest#" not in raw  # and none in the log


def test_the_real_keyframe_store_writes_the_same_blurred_jpeg_with_everything_on(tmp_path):
    pytest.importorskip("cv2")
    import numpy as np

    from contexts.perception.infrastructure.face_blur import FaceBlur
    from contexts.perception.infrastructure.keyframe_store import KeyframeStore

    faces = np.array([row(96, 58), row(420, 58)], np.float32)

    class YuNet:  # two faces: person 7's head and person 8's
        def detect(self, frame):
            return 1, faces

    frame = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    blur = FaceBlur(detector=YuNet())
    store = KeyframeStore(tmp_path, blur=blur)
    path = tmp_path / "frames" / "cam-desk" / "1.jpg"
    store("cam-desk", 1, frame)
    before = path.read_bytes()

    quick = {"window_s": 4.0, "heartbeat_s": 10.0, "min_samples": 2, "idle_s": 30.0}
    rig = box(tmp_path / "box", expression_settings=quick)
    rig.service.start_enrollment("m-ana", "Ana Example", consent_ack=True)
    published = []
    detect = rig.service.watch("cam-desk", lambda f: list(PEOPLE), blur, published.append)
    frame[58:138, 96:156] = 200  # so the two crops differ: person 7 enrols, person 8 is a stranger
    for step in range(40):
        rig.clock.t += 1
        if step == 0:
            frame[58:138, 96:156] = original[58:138, 96:156]
            frame[58:138, 420:480] = 0  # only one face is "seen" clearly while enrolling: hide the other from the fake
            faces_now = faces
            faces = faces[:1]
        if step == 8:
            frame[58:138, 420:480] = original[58:138, 420:480]
            faces = faces_now
            rig.service.update_member("m-ana", expressions=True)
            rig.service.set_everyone(True, confirm=True)
        detect(frame)
    assert rig.service.errors == 0 and rig.service.tagged() == {"cam-desk": {7: "m-ana"}}
    assert {e.member_id for e in published if e.type == "perception.expression.observed"} == {"m-ana", None}

    store("cam-desk", 1, frame)
    assert path.read_bytes() == before  # byte for byte the same file
    assert (frame == original).all()  # the live frame was only read
    ok, unblurred = __import__("cv2").imencode(".jpg", original, [1, 85])
    assert ok and unblurred.tobytes() != before  # and what is stored is the blurred picture, not the live one


# --- the real decoder and the real model's wrapper (only where OpenCV is installed; no model file is needed) ---


def test_the_size_of_a_picture_is_read_from_its_header_without_decoding_it():
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 4000, 3000, 8, 2, 0, 0, 0)
    assert image_size(png) == (4000, 3000)
    jpeg = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
            + b"\xff\xdb" + struct.pack(">H", 4) + b"\x00\x00"
            + b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, 3024, 4032, 3) + b"\x00" * 9)
    assert image_size(jpeg) == (4032, 3024)
    bomb = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 60000, 60000, 8, 2, 0, 0, 0)
    assert image_size(bomb) == (60000, 60000)  # known before a single pixel is decoded, so it can be refused
    for junk in (b"", b"GIF89a....", b"\xff\xd8", b"\xff\xd8\x00\x00\x00\x00\x00\x00\x00\x00\x00", b"<svg/>"):
        assert image_size(junk) is None


def test_the_real_decoder_measures_brightness_and_sharpness_and_refuses_what_it_cannot_read():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from contexts.perception.application.identity import MAX_BRIGHTNESS, MIN_BRIGHTNESS, MIN_SHARPNESS
    from contexts.perception.infrastructure.photo import DETECT_SIDE, MAX_SIDE

    reader = build_photo_reader()
    rng = np.random.default_rng(3)
    # A made-up "face": blobs and edges at the scale of eyes and a mouth. Not a person.
    sharp = cv2.GaussianBlur(rng.integers(0, 256, (600, 600, 3), dtype=np.uint8), (0, 0), 3)
    sharp = cv2.normalize(sharp, None, 40, 215, cv2.NORM_MINMAX)
    cv2.circle(sharp, (220, 250), 22, (20, 20, 20), -1)
    cv2.circle(sharp, (380, 250), 22, (20, 20, 20), -1)
    cv2.ellipse(sharp, (300, 420), (80, 25), 0, 0, 180, (30, 30, 160), 8)
    picture = np.full((1500, 2000, 3), 128, np.uint8)
    picture[400:1000, 700:1300] = sharp

    def encode(image, ext=".jpg"):
        ok, data = cv2.imencode(ext, image)
        assert ok
        return data.tobytes()

    image = reader.decode(encode(picture))
    assert image.shape[:2] == (1200, 1600) and max(image.shape[:2]) == MAX_SIDE  # shrunk in memory
    small, scale = reader.small(image)
    assert max(small.shape[:2]) == DETECT_SIDE and scale == pytest.approx(0.4)
    face = [560, 320, 480, 480]  # where the "face" is after the shrink
    brightness, sharpness = reader.measure(image, face)
    assert MIN_BRIGHTNESS < brightness < MAX_BRIGHTNESS and sharpness > MIN_SHARPNESS

    soft = picture.copy()
    soft[400:1000, 700:1300] = cv2.GaussianBlur(sharp, (0, 0), 14)  # out of focus
    assert reader.measure(reader.decode(encode(soft)), face)[1] < MIN_SHARPNESS
    assert reader.measure(reader.decode(encode((picture * 0.2).astype(np.uint8))), face)[0] < MIN_BRIGHTNESS
    assert reader.measure(reader.decode(encode(cv2.add(picture, 140))), face)[0] > MAX_BRIGHTNESS
    assert reader.measure(image, [5000, 5000, 10, 10]) == (0.0, 0.0)  # a box outside the picture
    assert reader.decode(encode(picture, ".png")).shape == image.shape
    for junk in (b"", b"not a picture", encode(picture)[:200], b"GIF89a" + b"\x00" * 64):
        assert reader.decode(junk) is None
    bomb = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 60000, 60000, 8, 2, 0, 0, 0)
    assert reader.decode(bomb) is None  # refused on its header: 3.6 gigapixels are never allocated


def test_the_expression_models_wrapper_aligns_the_face_and_turns_scores_into_probabilities():
    pytest.importorskip("cv2")
    import numpy as np

    from contexts.perception.infrastructure.expression import FACE_112, FerExpressionReader, alignment

    turn = math.radians(12)
    spin = 2.3 * np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
    landmarks = np.asarray(FACE_112) @ spin.T + [300, 140]  # the template, larger, turned and moved
    moved = np.hstack((landmarks, np.ones((5, 1)))) @ alignment(landmarks.ravel()).T
    assert np.abs(moved - np.asarray(FACE_112)).max() < 1e-6  # the landmarks land where the model expects them

    class Net:
        def setInput(self, blob, name):
            self.blob, self.name = blob, name

        def forward(self, name):
            assert name == "label"
            return np.array([[-0.75, -3.2, -1.9, 0.5, 3.5, 0.6, 1.5]], np.float32)

    net = Net()
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    original = frame.copy()
    probabilities = FerExpressionReader(net=net).read(frame, [280, 130, 260, 300, *landmarks.ravel(), 0.95])
    assert (net.name, net.blob.shape) == ("data", (1, 3, 112, 112)) and -1.0 <= net.blob.min() and net.blob.max() <= 1.0
    assert list(probabilities) == list(LABELS) and sum(probabilities.values()) == pytest.approx(1.0)
    assert max(probabilities, key=probabilities.get) == "neutral" and probabilities["neutral"] == pytest.approx(0.797, abs=0.01)
    assert (frame == original).all()  # the frame is only read
