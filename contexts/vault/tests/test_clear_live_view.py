"""The clear live view changes one thing: the live picture a request with the owner's view ticket gets back.

Proved here on the real CaptureService, TrackingPipeline, KeyframeStore, keyframe loader and DatasetExporter, wired
the way the gateway wires them, with three stand-ins so the run is exact and needs neither a camera nor OpenCV:
- the camera reader hands out scripted frames (a "frame" is bytes with a face in it: b"... face=ALICE");
- the face blur replaces the face, so a blurred picture and a clear one differ visibly, and the two blurs differ
  from each other: one pass writes ALICE -> #####, the strict repeat-until-clear blur writes ALICE -> @@@@@;
- the JPEG encoder is `b"JPEG<quality>:" + frame`, so "the same bytes as today" can be written down.

The same frames are run twice, once with nobody holding a ticket and once with the owner's screen asking for the
clear picture of every frame. Everything except those ticketed answers has to be byte for byte the same.

What "the same as before" means since the blur rework: the live picture without a ticket is exactly what it was
(one pass, quality 70). A saved keyframe is still always blurred, and now more strictly: with the strict settings
(score 0.5, margin 0.6), repeated until no face is found. So the saved bytes here are `saved()` (@@@@@), where
before the rework they were the one-pass bytes (#####); the files, their names, the events and the export are
otherwise as they were, and the ticket still changes none of them.
"""

import asyncio
import sys
import types
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.agent.application.memory_port import FakeMemory
from contexts.capture.api import capture_router
from contexts.capture.application.service import CaptureService
from contexts.learning.application.export import DatasetExporter
from contexts.perception.infrastructure.frames import keyframe_loader
from contexts.vault import Policy, ViewTickets
from contexts.vault.api import vault_router
from contexts.workspace.domain.approvals import ActionRequest
from contracts.learning import TaskSpec
from contracts.memory import Episode
from contracts.types import Box, Track

NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)
T0_MS = 1_791_000_000_000
FRAMES = [f"frame {i} desk mug face=ALICE".encode() for i in range(6)]


def blurred(frame: bytes) -> bytes:
    """One pass: the live picture for a screen without the ticket."""
    return frame.replace(b"ALICE", b"#####")


def blurred_until_clear(frame: bytes) -> bytes:
    """The strict blur: what is saved, and what ?blur=1 answers."""
    return frame.replace(b"ALICE", b"@@@@@")


def today_live(frame: bytes) -> bytes:
    """What GET /capture/<camera>/snapshot.jpg answered before the clear view existed: blurred, quality 70."""
    return b"JPEG70:" + blurred(frame)


def saved(frame: bytes) -> bytes:
    """What the keyframe store writes: blurred until clear with the strict settings, quality 85."""
    return b"JPEG85:" + blurred_until_clear(frame)


class Blur:
    def __init__(self, **settings):
        self.settings = settings

    def __call__(self, frame):
        assert self.settings == {}  # the one-pass blur is the default detector
        return blurred(frame)

    def scrambled_until_clear(self, frame):
        assert self.settings == {"score": 0.5, "margin": 0.6}  # only the strict detector repeats
        return blurred_until_clear(frame), frame.count(b"face=")

    def faces(self, frame):  # the face counter's question: counts only
        return [[0.0] * 15] * frame.count(b"face=")


class Reader:
    """A camera whose frames the test hands over one at a time."""

    def __init__(self, camera_id, source, **_):
        self.item = None

    def start(self):
        pass

    def stop(self):
        pass

    def push(self, i):
        self.item = (i + 1, T0_MS + i * 1000, FRAMES[i])

    def latest(self):
        return self.item

    def stats(self):
        return {"status": "ok", "fps_actual": 15.0, "dropped": 0}


def detector(frame):
    return [Track(label="person", confidence=0.9, track_id=int(frame.split()[1]), box=Box(x1=0.1, y1=0.1, x2=0.4, y2=0.9))]


def stand_ins(monkeypatch):
    def imencode(ext, frame, params):
        return True, types.SimpleNamespace(tobytes=lambda: b"JPEG%d:" % params[1] + frame)

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(IMWRITE_JPEG_QUALITY=1, imencode=imencode))
    monkeypatch.setattr("contexts.capture.application.service.FaceBlur", Blur)
    monkeypatch.setattr("contexts.capture.application.service.CameraReader", Reader)


def box(data_dir, service_policy=None):
    """The vault and the capture routers on one app, as contexts/gateway/app.py puts them together."""
    policy = Policy(data_dir)
    tickets = ViewTickets()
    bus = EventBus(persist=False)
    policy.subscribe(bus)
    cameras = {"cam-desk": {"source": "scripted", "detect_fps": 1000, "observe_every_s": 0, "keyframe_every_s": 0}}
    service = CaptureService(cameras, bus, lambda: detector, data_dir / "data", service_policy or policy)

    app = FastAPI()
    app.include_router(vault_router(policy, bus, view_tickets=tickets, clock=lambda: NOW))
    app.include_router(capture_router(service, clear=lambda ticket: tickets.valid(ticket, NOW)))
    return policy, tickets, bus, service, TestClient(app)


def run(data_dir, monkeypatch, owner_watches: bool) -> dict:
    stand_ins(monkeypatch)
    out = {"live": [], "clear": [], "stream": [], "events": []}

    async def main():
        policy, tickets, bus, service, http = box(data_dir)
        bus.subscribe("perception.", out["events"].append)
        service.start(asyncio.get_running_loop())
        reader, pipeline = service.readers["cam-desk"], service.pipelines["cam-desk"]
        ticket = http.post("/vault/view-ticket").json()["ticket"]
        for i in range(len(FRAMES)):
            reader.push(i)
            for _ in range(400):
                if pipeline.detected > i:
                    break
                await asyncio.sleep(0.005)
            assert pipeline.detected == i + 1 and pipeline.errors == 0
            if owner_watches:  # the owner's signed-in screen looks at this very frame, clear, by both routes
                out["clear"].append(http.get(f"/capture/cam-desk/snapshot.jpg?ticket={ticket}").content)
                out["stream"].append(http.get(f"/capture/live/cam-desk?max=1&fps=15&ticket={ticket}").content)
            out["live"].append(http.get("/capture/cam-desk/snapshot.jpg").content)  # anyone else, at the same moment
            out["live"].append(http.get("/capture/live/cam-desk?max=1&fps=15").content)
        await asyncio.sleep(0.1)  # the pipeline's last publishes
        service.stop()

    asyncio.run(main())
    frames_dir = data_dir / "data" / "frames"
    out["saved"] = {p.relative_to(frames_dir).as_posix(): p.read_bytes() for p in sorted(frames_dir.rglob("*")) if p.is_file()}
    refs = [e.frame_ref for e in out["events"] if getattr(e, "frame_ref", None)]
    out["loaded"] = [keyframe_loader(data_dir / "data")(ref) for ref in refs]

    # The approved training export of that hour, through the real exporter and the real keyframe loader.
    memory = FakeMemory([Episode(id="e1", camera_id="cam-desk", start=NOW - timedelta(hours=1), captions=["at the desk"],
                                 keyframe_refs=refs)])
    exports = data_dir / "exports"
    spec = TaskSpec(task="desk-objects", instruction="List what is on the desk.", camera="cam-desk")
    DatasetExporter(memory, keyframe_loader(data_dir / "data"), lambda e: None, out_dir=exports)(ActionRequest(
        id="a1", kind="export_training_data", target="lab", reason="test", expires_at=NOW + timedelta(hours=1),
        params={"spec": spec.model_dump(mode="json"), "start": (NOW - timedelta(hours=2)).isoformat(), "end": NOW.isoformat()}))
    [dataset] = exports.iterdir()
    out["export"] = {p.relative_to(dataset).as_posix(): p.read_bytes() for p in sorted(dataset.rglob("*"))
                     if p.is_file() and p.name != "manifest.json"}  # the manifest holds a random dataset id
    out["events"] = [e.model_dump(mode="json", exclude={"ts", "event_id"}) for e in out["events"]]
    return out


def mjpeg(jpeg: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"


def test_without_a_ticket_everything_is_byte_for_byte_what_it_was(tmp_path, monkeypatch):
    alone = run(tmp_path / "alone", monkeypatch, owner_watches=False)
    watched = run(tmp_path / "watched", monkeypatch, owner_watches=True)

    # The live picture for anyone without a ticket: today's bytes, whether or not the owner is watching clear.
    expected = [b for f in FRAMES for b in (today_live(f), mjpeg(today_live(f)))]
    assert alone["live"] == expected and watched["live"] == expected

    # Saved keyframes: the same files with the same bytes, every one blurred (the strict blur, not the live view's one
    # pass), also for the frames the owner saw clear.
    assert alone["saved"] == {f"cam-desk/{T0_MS + i * 1000}.jpg": saved(f) for i, f in enumerate(FRAMES)}
    assert watched["saved"] == alone["saved"]
    assert watched["loaded"] == alone["loaded"] == [saved(f) for f in FRAMES]
    assert not [name for name, data in watched["saved"].items() if b"ALICE" in data or b"#####" in data]

    # Detection and tracking: the same events (observations with their tracks and frame refs, and the face counts).
    assert watched["events"] == alone["events"]
    observations = [e for e in alone["events"] if e["type"] == "perception.observation.recorded"]
    assert [e["tracks"][0]["track_id"] for e in observations] == list(range(len(FRAMES)))
    assert [e["frame_ref"] for e in observations] == [f"frames:cam-desk:{T0_MS + i * 1000}" for i in range(len(FRAMES))]
    assert {e["type"] for e in alone["events"]} == {"perception.observation.recorded", "perception.faces.counted"}

    # The training export: the same pictures and the same examples, all blurred.
    assert watched["export"] == alone["export"]
    assert [alone["export"][f"images/{i:04d}.jpg"] for i in range(len(FRAMES))] == [saved(f) for f in FRAMES]
    assert b"ALICE" not in b"".join(watched["export"].values())

    # And what the ticket did change: the owner's own live picture, and nothing else.
    assert watched["clear"] == [b"JPEG70:" + f for f in FRAMES] and all(b"ALICE" in c for c in watched["clear"])
    assert watched["stream"] == [mjpeg(b"JPEG70:" + f) for f in FRAMES]
    assert alone["clear"] == [] and b"ALICE" not in b"".join(alone["live"])


def frozen(tmp_path, monkeypatch, service_policy=None):
    """A box holding one frame, no pipeline running: for asking the same picture in different ways."""
    stand_ins(monkeypatch)
    policy, tickets, bus, service, http = box(tmp_path, service_policy)
    service.readers["cam-desk"] = Reader("cam-desk", "scripted")
    service.readers["cam-desk"].push(0)
    return policy, tickets, bus, service, http


def test_a_wrong_or_expired_ticket_gets_the_blurred_picture_and_no_explanation(tmp_path, monkeypatch):
    policy, tickets, bus, service, http = frozen(tmp_path, monkeypatch)
    good = tickets.issue(NOW)[0]
    expired = tickets.issue(NOW - timedelta(minutes=11))[0]
    plain = http.get("/capture/cam-desk/snapshot.jpg")
    assert plain.content == today_live(FRAMES[0])
    for ticket in ("", "nonsense", good[:-2], good + "a", expired, ViewTickets().issue(NOW)[0], "a" * 5000):
        r = http.get("/capture/cam-desk/snapshot.jpg", params={"ticket": ticket})
        assert (r.status_code, r.content, dict(r.headers)) == (200, plain.content, dict(plain.headers)), ticket
        r = http.get("/capture/live/cam-desk", params={"ticket": ticket, "max": 1, "fps": 15})
        assert r.status_code == 200 and r.content == mjpeg(today_live(FRAMES[0])), ticket
    assert http.get("/capture/cam-desk/snapshot.jpg", params={"ticket": good}).content == b"JPEG70:" + FRAMES[0]
    assert http.get("/capture/cam-desk/snapshot.jpg").content == today_live(FRAMES[0])  # the clear one did not take its place


def test_an_older_box_shows_its_owner_the_clear_view_without_a_switch_and_nobody_else(tmp_path, monkeypatch):
    """A box whose policy was written before `live_faces` existed: clear for the ticket, today's bytes without one."""
    (tmp_path / "vault").mkdir(parents=True)
    (tmp_path / "vault" / "policy.json").write_text('{"blur_faces": true, "paused_cameras": [], "offline": false, "keep_days": 30}')
    policy, tickets, bus, service, http = frozen(tmp_path, monkeypatch)
    assert http.get("/vault/policy").json()["live_faces"] == "clear"
    ticket = {"ticket": http.post("/vault/view-ticket").json()["ticket"]}
    assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket).content == b"JPEG70:" + FRAMES[0]
    assert http.get("/capture/live/cam-desk", params=ticket | {"max": 1}).content == mjpeg(b"JPEG70:" + FRAMES[0])
    assert http.get("/capture/cam-desk/snapshot.jpg").content == today_live(FRAMES[0])  # no ticket: as before
    assert http.get("/capture/live/cam-desk", params={"max": 1}).content == mjpeg(today_live(FRAMES[0]))
    assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket | {"blur": 1}).content == b"JPEG70:" + blurred_until_clear(FRAMES[0])


def test_the_ticket_shows_faces_unless_the_owner_switched_the_clear_view_off(tmp_path, monkeypatch):
    policy, tickets, bus, service, http = frozen(tmp_path, monkeypatch)
    ticket = {"ticket": http.post("/vault/view-ticket").json()["ticket"]}
    assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket).content == b"JPEG70:" + FRAMES[0]

    assert http.put("/vault/policy", json={"live_faces": "blurred"}).json()["live_faces"] == "blurred"
    assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket).content == today_live(FRAMES[0])  # at once, same frame
    assert http.get("/capture/live/cam-desk", params=ticket | {"max": 1}).content == mjpeg(today_live(FRAMES[0]))

    http.put("/vault/policy", json={"paused_cameras": ["cam-desk"]})  # a pause, then the camera back: still off
    assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket).status_code == 404
    assert http.put("/vault/policy", json={"paused_cameras": []}).json()["live_faces"] == "blurred"
    assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket).content == today_live(FRAMES[0])
    assert not Policy(tmp_path).live_clear  # and after a restart, from the file

    http.put("/vault/policy", json={"live_faces": "clear", "paused_cameras": ["cam-desk"]})
    assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket).status_code == 404  # a paused camera shows nobody anything
    assert http.get("/capture/live/cam-desk", params=ticket).status_code == 404


def shared(frame: bytes) -> bytes:
    """What ?blur=1 answers: the live encoding of the strictly blurred frame."""
    return b"JPEG70:" + blurred_until_clear(frame)


def test_blur_1_is_blurred_until_clear_even_with_a_valid_ticket(tmp_path, monkeypatch):
    """A recorder (jetson/record_clip.py) asks for a picture it will share: no ticket and no policy makes it clear."""
    policy, tickets, bus, service, http = frozen(tmp_path, monkeypatch)
    ticket = http.post("/vault/view-ticket").json()["ticket"]
    for query in ({"blur": 1}, {"blur": 1, "ticket": ticket}, {"blur": "true", "ticket": ticket}, {"blur": 1, "ticket": "nonsense"}):
        assert http.get("/capture/cam-desk/snapshot.jpg", params=query).content == shared(FRAMES[0]), query
        assert http.get("/capture/live/cam-desk", params=query | {"max": 1}).content == mjpeg(shared(FRAMES[0])), query
    # asking for it did not take the place of the other two pictures of the same frame
    assert http.get("/capture/cam-desk/snapshot.jpg", params={"ticket": ticket}).content == b"JPEG70:" + FRAMES[0]
    assert http.get("/capture/cam-desk/snapshot.jpg").content == today_live(FRAMES[0])
    assert http.get("/capture/cam-desk/snapshot.jpg", params={"blur": 0, "ticket": ticket}).content == b"JPEG70:" + FRAMES[0]

    http.put("/vault/policy", json={"blur_faces": False})  # a box switched to saving unblurred pictures
    service.readers["cam-desk"].push(1)  # its next frame
    assert http.get("/capture/cam-desk/snapshot.jpg").content == b"JPEG70:" + FRAMES[1]
    assert http.get("/capture/cam-desk/snapshot.jpg", params={"blur": 1}).content == shared(FRAMES[1])
    assert http.get("/capture/live/cam-desk", params={"blur": 1, "max": 1}).content == mjpeg(shared(FRAMES[1]))

    http.put("/vault/policy", json={"paused_cameras": ["cam-desk"]})
    assert http.get("/capture/cam-desk/snapshot.jpg", params={"blur": 1}).status_code == 404  # paused: nobody gets a picture


def test_a_camera_set_to_fast_saves_one_pass_keyframes_and_they_are_still_blurred(tmp_path, monkeypatch):
    """`keyframe_blur: fast` in config/cameras.yaml is the way back if a box cannot afford the second detector pass."""
    stand_ins(monkeypatch)
    out = {}

    async def main():
        policy = Policy(tmp_path)
        cameras = {"cam-desk": {"source": "scripted", "detect_fps": 1000, "observe_every_s": 0, "keyframe_every_s": 0,
                                "keyframe_blur": "fast"}}
        service = CaptureService(cameras, EventBus(persist=False), lambda: detector, tmp_path / "data", policy)
        service.start(asyncio.get_running_loop())
        service.readers["cam-desk"].push(0)
        for _ in range(400):
            if service.pipelines["cam-desk"].detected:
                break
            await asyncio.sleep(0.005)
        service.stop()
        out["errors"] = service.pipelines["cam-desk"].errors

    asyncio.run(main())
    [file] = (tmp_path / "data" / "frames" / "cam-desk").glob("*.jpg")
    assert out["errors"] == 0 and file.read_bytes() == b"JPEG85:" + blurred(FRAMES[0])


def test_a_policy_that_knows_nothing_of_the_clear_view_is_served_as_before(tmp_path, monkeypatch):
    class OldPolicy:  # what contexts/capture's own tests pass in: blur and pauses, not the Vault's policy
        blur_faces = True

        def is_paused(self, camera_id):
            return False

    for policy in (OldPolicy(), None):
        _, tickets, _, service, http = frozen(tmp_path / str(type(policy).__name__), monkeypatch, policy)
        service.policy = policy
        ticket = {"ticket": tickets.issue(NOW)[0]}
        assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket).content == today_live(FRAMES[0])
        assert service.snapshot("cam-desk", clear=True) == service.snapshot("cam-desk") == today_live(FRAMES[0])


def test_a_box_that_already_saved_unblurred_pictures_answers_as_it_did(tmp_path, monkeypatch):
    """`blur_faces` off is an older choice some box may hold: its live picture was unblurred for everyone, and still is."""
    policy, tickets, bus, service, http = frozen(tmp_path, monkeypatch)
    http.put("/vault/policy", json={"blur_faces": False, "live_faces": "blurred"})
    ticket = {"ticket": tickets.issue(NOW)[0]}
    plain = http.get("/capture/cam-desk/snapshot.jpg").content
    assert plain == b"JPEG70:" + FRAMES[0] == http.get("/capture/cam-desk/snapshot.jpg", params=ticket).content
