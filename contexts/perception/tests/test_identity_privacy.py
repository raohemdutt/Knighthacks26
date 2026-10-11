import os
"""P9's privacy promises, each as a test: nothing about a face in the log, in Memory, in an API answer or in a
log line; templates only as vault blobs; stored keyframes exactly as blurred as before."""

import asyncio
import json
import logging
import struct
import time

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.infrastructure.store import MemoryStore
from contexts.perception.api import identity_router
from contexts.perception.application.identity import IdentityService
from contexts.perception.application.tracking import TrackingPipeline
from contexts.perception.infrastructure.recognizer import FakeRecognizer
from contexts.perception.tests.identity_helpers import Frame, Rig, face_in, person, row
from contexts.vault import KeyRing, Vault, Wiper
from contracts.perception import ObservationRecorded

NAME = "Zebulon Quartermaine"


def http(rig: Rig, bus: EventBus, token: str | None = None) -> TestClient:
    def guard(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(identity_router(rig.service, bus, guard=guard))
    return TestClient(app)


def needles(vectors: list[list[float]], crops: list[bytes]) -> dict[str, bytes]:
    """Every way a name, a face crop or a template could sit somewhere in plain text."""
    out = {"name": NAME.encode(), "name, lower case": NAME.lower().encode(), "first name": b"Zebulon",
           "last name": b"Quartermaine"}
    for i, crop in enumerate(crops):
        out[f"face crop {i}"] = crop
    for i, v in enumerate(vectors):
        out[f"vector {i} as float32"] = struct.pack(f"<{len(v)}f", *v)
        out[f"vector {i} as float64"] = struct.pack(f"<{len(v)}d", *v)
        out[f"vector {i} as float32, start"] = struct.pack("<4f", *v[:4])
        out[f"vector {i} as float64, start"] = struct.pack("<2d", *v[:2])
        out[f"vector {i} as JSON"] = json.dumps(v).encode()
        for j, x in enumerate(v[:6]):
            out[f"vector {i}[{j}] as text"] = repr(x).encode()
            out[f"vector {i}[{j}] to 6 places"] = f"{x:.6f}".encode()
    return out


def test_no_template_face_crop_or_name_is_anywhere_in_plain_text(tmp_path, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    bus = EventBus(tmp_path / "events")
    MemoryProjection(MemoryStore(tmp_path / "memory.sqlite")).subscribe(bus)
    rig = Rig(tmp_path)
    bus.subscribe("perception.identity", rig.service.on_changed, replay=True)
    c = http(rig, bus)
    answers: dict[str, str] = {}

    def call(method: str, path: str, **kw) -> dict:
        r = c.request(method, path, **kw)
        assert r.status_code == 200, r.text
        answers[f"{method} {path} #{len(answers)}"] = r.text
        return r.json()

    # Switch on and enrol through the API, with the camera hook publishing to the real bus.
    call("PUT", "/identity/settings", json={"identity_enabled": True})
    member_id = call("POST", "/identity/enroll/start", json={"name": NAME, "consent_ack": True})["member_id"]
    owner, guest = person(7, 0.05, 0.1, 0.45, 0.95), person(8, 0.55, 0.1, 0.95, 0.95)
    crops = [b"zeb#enrol%d" % i for i in range(5)] + [b"zeb#desk1", b"zeb#desk2", b"guest#1", b"guest#2"]
    published = []
    detect = rig.service.watch("cam-desk", lambda frame: frame.tracks, rig.finder, published.append)

    def see(frame, tracks=()):
        frame.tracks = list(tracks)
        rig.clock.t += 1
        detect(frame)
        call("GET", "/identity/enroll/status")
        events = [ObservationRecorded(camera_id="cam-desk", tracks=frame.tracks), *published]
        published.clear()
        asyncio.run(bus.publish_all(events))

    for crop in crops[:5]:
        see(Frame((crop, row(300, 100))))
    assert call("GET", "/identity/enroll/status")["state"] == "done"
    see(Frame(face_in(owner, crops[5]), face_in(guest, crops[7])), [owner, guest])
    see(Frame(face_in(owner, crops[6]), face_in(guest, crops[8])), [owner, guest])
    assert rig.service.tagged() == {"cam-desk": {7: member_id}}
    call("GET", "/identity/settings")
    members = call("GET", "/identity/members")
    assert members == [{"member_id": member_id, "name": NAME, "enrolled_at": members[0]["enrolled_at"], "source": "live",
                        "updated_at": members[0]["enrolled_at"], "photos": 5, "expressions": False, "circle_member_id": None}]
    assert rig.service.errors == 0

    # What must not be found: the name, the crops, the template and every vector that was ever computed.
    fake = FakeRecognizer()
    template = rig.store.members()[0].vector
    hay = needles([template] + [fake.embed(crop) for crop in crops], crops)

    # Positive control: the same search does find them in the template's plain text, so the scan can see.
    blob_path = tmp_path / "vault" / "identity" / f"{member_id}.face"
    plain = Vault(KeyRing(tmp_path)).decrypt("identity", blob_path.read_bytes())
    assert hay["name"] in plain and hay["vector 0 as JSON"] in plain and hay["vector 0[0] as text"] in plain

    # 1. The event log, and every other file under the data directory (the vault blobs, Memory's database).
    files = [p for p in sorted(tmp_path.rglob("*")) if p.is_file()]
    log_files = [p for p in files if p.parent.name == "events"]
    assert log_files and blob_path in files and (tmp_path / "memory.sqlite") in files
    scanned = 0
    for path in files:
        data = path.read_bytes()
        scanned += len(data)
        found = [what for what, needle in hay.items() if needle in data]
        assert not found, f"{path.relative_to(tmp_path)} holds {found} in plain text"
    types = {json.loads(line)["type"] for p in log_files for line in p.read_text().splitlines()}
    assert {"perception.identity.changed", "perception.person.identified", "perception.observation.recorded"} <= types
    assert member_id in "".join(p.read_text() for p in log_files) and "zeb" not in member_id  # ids only

    # 2. API answers: the name only from GET /identity/members, the template from none.
    for path, text in answers.items():
        found = [what for what, needle in hay.items() if needle in text.encode()]
        allowed = ["name", "first name", "last name"] if "/identity/members" in path else []
        assert found == allowed, f"{path} returned {found}"
        assert "vector" not in text

    # 3. Log lines.
    logged = "\n".join(r.getMessage() for r in caplog.records).encode()
    assert not [what for what, needle in hay.items() if needle in logged]

    with capsys.disabled():
        print(f"\n  plaintext-leak scan: {len(files)} files ({scanned} bytes: {len(log_files)} event log, "
              f"1 template blob, Memory's database, the key and settings), {len(answers)} API answers, "
              f"{len(caplog.records)} log lines, {len(hay)} needles (name, {len(crops)} face crops, "
              f"{len(crops) + 1} vectors in 7 encodings): none found in plain text; "
              f"the name appears only in GET /identity/members")


def test_the_template_file_is_a_vault_blob_only_this_box_can_open(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    path = tmp_path / "vault" / "identity" / "m-ana.face"
    if os.name == "posix":  # Windows has no owner-only mode
        assert path.stat().st_mode & 0o777 == 0o600 and path.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads((tmp_path / "vault" / "identity.json").read_text()) == {"identity_enabled": True}
    blob = path.read_bytes()
    assert blob[:1] == b"\x01" and b"Ana" not in blob
    assert json.loads(Vault(KeyRing(tmp_path)).decrypt("identity", blob))["name"] == "Ana Example"
    with pytest.raises(Exception):
        Vault(KeyRing(tmp_path)).decrypt("memory", blob)  # sealed for the identity context only
    (tmp_path / "vault" / "identity" / "m-eve.face").write_bytes(b"\x01" + b"not ours" * 8)
    assert [m.member_id for m in rig.store.members()] == ["m-ana"]  # a file that does not decrypt is nobody


def test_wiping_everything_removes_the_templates_and_keeps_the_key(tmp_path):
    rig = Rig(tmp_path)
    rig.enroll("m-ana", "Ana Example", b"ana")
    assert Wiper(tmp_path).wipe("keyframes").identity_removed == 0 and len(rig.service.members()) == 1
    report = Wiper(tmp_path).wipe("all")
    assert report.identity_removed == 1 and report.refused == [] and rig.service.members() == []
    assert (tmp_path / "vault" / "master.key").exists()
    assert report.identity_switched_off and rig.service.enabled is False  # the switches went too: the running box is off
    rig.service.set_enabled(True)
    ana = person(7)
    for i in range(3):
        rig.look(Frame(face_in(ana, b"ana#%d" % i)), [ana])
    assert rig.service.tagged() == {"cam-desk": {}}  # no template: she is a stranger again


def test_writes_need_the_token_and_reads_never_return_a_template(tmp_path):
    rig = Rig(tmp_path)
    bus = EventBus(tmp_path / "events")
    c, auth = http(rig, bus, token="s3cret"), {"Authorization": "Bearer s3cret"}
    assert c.put("/identity/settings", json={"identity_enabled": True}).status_code == 401
    assert c.post("/identity/enroll/start", json={"name": "Ana"}).status_code == 401
    assert c.delete("/identity/members/m-ana").status_code == 401
    assert c.post("/identity/enroll/start", json={"name": "Ana"}, headers=auth).status_code == 409  # still off
    assert c.put("/identity/settings", json={"identity_enabled": True}, headers=auth).json()["identity_enabled"] is True
    assert c.post("/identity/enroll/start", json={"name": "Ana", "member_id": "Not An Id"}, headers=auth).status_code == 422
    assert c.get("/identity/enroll/status").json()["state"] == "idle"
    rig.enroll("m-ana", "Ana Example", b"ana")
    assert set(c.get("/identity/members").json()[0]) == {"member_id", "name", "enrolled_at", "updated_at", "photos", "source", "expressions",
                                                      "circle_member_id"}
    assert c.delete("/identity/members/m-ana", headers=auth).json() == {"removed": "m-ana"}
    assert c.delete("/identity/members/m-ana", headers=auth).status_code == 404
    assert c.get("/identity/members").json() == []
    logged = [json.loads(line) for p in (tmp_path / "events").glob("*.jsonl") for line in p.read_text().splitlines()]
    assert [(e["enabled"], e["members"]) for e in logged] == [(True, []), (True, [])]  # switched on; deleted

    off = Rig(tmp_path / "other", recognizer=None)
    r = http(off, bus).put("/identity/settings", json={"identity_enabled": True})
    assert r.status_code == 409 and "not installed" in r.json()["detail"]


# --- blur is untouched ---


class Reader:
    """A 20 fps camera: frame n shows Ana's face in person 7's box. Its pixels hold the face, as a real frame would."""

    def __init__(self):
        self.t0 = time.monotonic()

    def latest(self):
        n = int((time.monotonic() - self.t0) / 0.05) + 1
        return n, 1_700_000_000_000 + n * 50, frame_number(n)


def frame_number(n: int) -> Frame:
    crop = b"ana#%04d" % n
    return Frame(face_in(person(7), crop), pixels=b"desk" * 8 + crop + b"wall" * 8)


def blurred(frame: Frame) -> bytes:
    """What a keyframe store writes for this frame: every face scrambled (stands in for FaceBlur + JPEG)."""
    out = bytes(frame.pixels)
    for who, _ in frame.faces:
        out = out.replace(who, b"#" * len(who))
    return out


def run_pipeline(data_dir, rig: Rig, hooked: bool, seconds: float = 0.9) -> dict[int, bytes]:
    """The real tracking pipeline for a moment, with a store that blurs and writes like KeyframeStore does."""
    folder = data_dir / "frames" / "cam-desk"
    folder.mkdir(parents=True)

    def store(camera_id, ts_ms, frame):
        (folder / f"{ts_ms}.jpg").write_bytes(blurred(frame))
        return f"frames:{camera_id}:{ts_ms}"

    class Counter:  # the pipeline's face counter (S3); the hook borrows its YuNet, as it does on the box
        finder = rig.finder

        def __call__(self, frame):
            return len(frame.faces), len(frame.faces)

    async def main():
        bus = EventBus(data_dir / "events")
        pipe = TrackingPipeline("cam-desk", Reader(), lambda frame: [person(7)], bus, asyncio.get_running_loop(),
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


def test_enabling_recognition_does_not_change_what_is_written_for_a_keyframe(tmp_path):
    rig = Rig(tmp_path / "box", retry_s=0.05)
    rig.enroll("m-ana", "Ana Example", b"ana")
    rig.service._clock = time.monotonic  # the pipeline runs on the real clock
    rig.service._tags.clear()

    off = run_pipeline(tmp_path / "off", rig, hooked=False)
    on = run_pipeline(tmp_path / "on", rig, hooked=True)

    assert rig.service.tagged() == {"cam-desk": {7: "m-ana"}} and rig.finder.calls > 5  # recognition really ran
    log = "".join(p.read_text() for p in (tmp_path / "on" / "events").glob("*.jsonl"))
    assert "perception.person.identified" in log and "perception.person.identified" not in "".join(
        p.read_text() for p in (tmp_path / "off" / "events").glob("*.jsonl"))
    assert len(off) >= 3 and len(on) >= 3
    for written in (off, on):
        for ts_ms, data in written.items():
            n = (ts_ms - 1_700_000_000_000) // 50
            assert data == blurred(frame_number(n))  # frame n's keyframe is the blurred frame n, recognition or not
            assert b"ana" not in data
    assert all(on[ts] == off[ts] for ts in on.keys() & off.keys())


def test_the_real_keyframe_store_writes_the_same_blurred_jpeg_with_recognition_on(tmp_path):
    pytest.importorskip("cv2")
    import numpy as np

    from contexts.perception.infrastructure.face_blur import FaceBlur
    from contexts.perception.infrastructure.keyframe_store import KeyframeStore

    class YuNet:  # one face, where person 7's head is
        def detect(self, frame):
            return 1, np.array([row(96, 58)], np.float32)

    frame = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    blur = FaceBlur(detector=YuNet())
    store = KeyframeStore(tmp_path, blur=blur)
    path = tmp_path / "frames" / "cam-desk" / "1.jpg"

    store("cam-desk", 1, frame)
    before = path.read_bytes()

    rig = Rig(tmp_path / "box")
    rig.service.set_enabled(True)
    rig.service.start_enrollment("m-ana", "Ana Example", consent_ack=True)
    detect = rig.service.watch("cam-desk", lambda f: [person(7)], blur, lambda e: None)
    for _ in range(12):  # enrols from the live frame, then tags the track
        rig.clock.t += 1
        detect(frame)
    assert rig.service.tagged() == {"cam-desk": {7: "m-ana"}} and rig.service.errors == 0

    store("cam-desk", 1, frame)
    assert path.read_bytes() == before  # byte for byte the same file
    assert (frame == original).all()  # the live frame was only read
    ok, unblurred = __import__("cv2").imencode(".jpg", original, [1, 85])
    assert ok and unblurred.tobytes() != before  # and what is stored is the blurred picture, not the live one
