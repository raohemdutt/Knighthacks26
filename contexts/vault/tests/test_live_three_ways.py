"""The owner's signed-in live view has three settings, and each changes only the picture a ticket holder gets.

    "clear"    everyone as they are (the default)
    "owner"    only the scanned owner's own face as it is; every other face blurred, also on the owner's screen
    "blurred"  everyone blurred: the same bytes as without a ticket

Run on the real CaptureService, TrackingPipeline, KeyframeStore, keyframe loader and DatasetExporter with the
stand-ins of test_clear_live_view.py, and frames that hold two faces: ALICE (who did the face scan) and BOBBY (who
did not). The one-pass blur writes a face it scrambles as #####, the strict one as @@@@@.

"Nothing else changed" is proved the same way as there: the same frames are run with nobody watching and with the
owner watching in each setting, and everything except the ticketed answers has to be byte for byte the same.
"""

import asyncio
from datetime import timedelta

import pytest

from contexts.agent.application.memory_port import FakeMemory
from contexts.learning.application.export import DatasetExporter
from contexts.perception.infrastructure.frames import keyframe_loader
from contexts.vault import Policy
from contexts.vault.tests import test_clear_live_view as base
from contexts.workspace.domain.approvals import ActionRequest
from contracts.learning import TaskSpec
from contracts.memory import Episode
from contracts.vault import PolicyChanged

NOW, T0_MS, mjpeg = base.NOW, base.T0_MS, base.mjpeg
NAMES = (b"ALICE", b"BOBBY")


class F(bytes):
    """A frame the size of a picture, as far as the code that asks for its shape can tell."""

    shape = (480, 640, 3)


FRAMES = [F(b"frame %d desk face=ALICE face=BOBBY" % i) for i in range(4)]


def hide(frame: bytes, names=NAMES, mark: bytes = b"#####") -> bytes:
    for name in names:
        frame = frame.replace(name, mark)
    return bytes(frame)


def live(frame: bytes) -> bytes:
    """Without a ticket: both faces blurred with one pass, quality 70. What the box answered before this setting."""
    return b"JPEG70:" + hide(frame)


def only_alice(frame: bytes) -> bytes:
    return b"JPEG70:" + hide(frame, names=(b"BOBBY",))


def saved(frame: bytes) -> bytes:
    return b"JPEG85:" + hide(frame, mark=b"@@@@@")


class Blur(base.Blur):
    """test_clear_live_view's stand-in, with two faces per frame and the `keep` the owner-only view passes."""

    def __call__(self, frame):
        assert self.settings == {}
        return hide(frame)

    def scrambled(self, frame, keep=None):
        assert self.settings == {}  # the owner-only view is one pass, like the blurred live view
        rows = self.faces(frame)
        spare = keep(rows) if keep else [False] * len(rows)
        return hide(frame, names=[NAMES[int(r[0])] for r, k in zip(rows, spare, strict=True) if not k]), len(rows)

    def scrambled_until_clear(self, frame):
        assert self.settings == {"score": 0.5, "margin": 0.6}
        return hide(frame, mark=b"@@@@@"), frame.count(b"face=")

    def faces(self, frame):
        return [[float(i)] + [0.0] * 14 for i, name in enumerate(NAMES) if name in frame]  # row[0] says whose face it is


class Owner:
    """Stands in for IdentityService.owner_faces: Alice did the face scan, and her person track is tagged."""

    def __init__(self, tagged: bool = True):
        self.tagged, self.asked, self.fail = tagged, 0, False

    def __call__(self, camera_id, rows, tracks, width, height):
        self.asked += 1
        assert camera_id == "cam-desk" and (width, height) == (640, 480) and tracks and tracks[0].label == "person"
        if self.fail:
            raise RuntimeError("the tags changed under us")
        return [self.tagged and NAMES[int(row[0])] == b"ALICE" for row in rows]


def stand_ins(monkeypatch):
    base.stand_ins(monkeypatch)
    monkeypatch.setattr("contexts.capture.application.service.FaceBlur", Blur)
    monkeypatch.setattr(base, "FRAMES", FRAMES)


def run(data_dir, monkeypatch, live_faces: str | None, owner: Owner | None = None) -> dict:
    """The frames through the box; with `live_faces`, the owner's screen asks for every frame with its ticket."""
    stand_ins(monkeypatch)
    out = {"live": [], "ticket": [], "stream": [], "events": []}

    async def main():
        policy, tickets, bus, service, http = base.box(data_dir)
        service.owner_faces = owner
        if live_faces:
            await bus.publish(PolicyChanged(blur_faces=True, live_faces=live_faces))
            assert policy.live_faces == live_faces
        bus.subscribe("perception.", out["events"].append)
        service.start(asyncio.get_running_loop())
        reader, pipeline = service.readers["cam-desk"], service.pipelines["cam-desk"]
        ticket = tickets.issue(NOW)[0]
        for i in range(len(FRAMES)):
            reader.push(i)
            for _ in range(400):
                if pipeline.detected > i:
                    break
                await asyncio.sleep(0.005)
            assert pipeline.detected == i + 1 and pipeline.errors == 0
            if live_faces:
                out["ticket"].append(http.get(f"/capture/cam-desk/snapshot.jpg?ticket={ticket}").content)
                out["stream"].append(http.get(f"/capture/live/cam-desk?max=1&fps=15&ticket={ticket}").content)
            out["live"].append(http.get("/capture/cam-desk/snapshot.jpg").content)  # anyone else, at the same moment
            out["live"].append(http.get("/capture/live/cam-desk?max=1&fps=15").content)
            out["live"].append(http.get(f"/capture/cam-desk/snapshot.jpg?blur=1&ticket={ticket}").content)  # a recorder
        await asyncio.sleep(0.1)
        service.stop()

    asyncio.run(main())
    frames_dir = data_dir / "data" / "frames"
    out["saved"] = {p.relative_to(frames_dir).as_posix(): p.read_bytes() for p in sorted(frames_dir.rglob("*")) if p.is_file()}
    refs = [e.frame_ref for e in out["events"] if getattr(e, "frame_ref", None)]
    memory = FakeMemory([Episode(id="e1", camera_id="cam-desk", start=NOW - timedelta(hours=1), captions=["at the desk"],
                                 keyframe_refs=refs)])
    spec = TaskSpec(task="desk-objects", instruction="List what is on the desk.", camera="cam-desk")
    DatasetExporter(memory, keyframe_loader(data_dir / "data"), lambda e: None, out_dir=data_dir / "exports")(ActionRequest(
        id="a1", kind="export_training_data", target="lab", reason="test", expires_at=NOW + timedelta(hours=1),
        params={"spec": spec.model_dump(mode="json"), "start": (NOW - timedelta(hours=2)).isoformat(), "end": NOW.isoformat()}))
    [dataset] = (data_dir / "exports").iterdir()
    out["export"] = {p.relative_to(dataset).as_posix(): p.read_bytes() for p in sorted(dataset.rglob("*"))
                     if p.is_file() and p.name != "manifest.json"}
    out["events"] = [e.model_dump(mode="json", exclude={"ts", "event_id"}) for e in out["events"]]
    return out


def test_each_choice_changes_only_what_the_ticket_holder_sees(tmp_path, monkeypatch):
    alone = run(tmp_path / "alone", monkeypatch, None)
    owner = Owner()
    seen = {choice: run(tmp_path / choice, monkeypatch, choice, owner) for choice in ("clear", "owner", "blurred")}

    # Without a ticket (and for a recorder, ?blur=1): the bytes the box always answered, whatever the choice.
    expected = [b for f in FRAMES for b in (live(f), mjpeg(live(f)), b"JPEG70:" + hide(f, mark=b"@@@@@"))]
    assert alone["live"] == expected
    for choice, got in seen.items():
        assert got["live"] == expected, choice
        # Saved keyframes, the events and the approved export: the same files and bytes, every face strictly blurred.
        assert got["saved"] == alone["saved"] == {f"cam-desk/{T0_MS + i * 1000}.jpg": saved(f) for i, f in enumerate(FRAMES)}, choice
        assert got["events"] == alone["events"] and got["export"] == alone["export"], choice
        assert not [n for n in NAMES if n in b"".join(got["saved"].values()) + b"".join(got["export"].values())], choice

    # And what the ticket holder sees, in each of the three.
    assert seen["clear"]["ticket"] == [b"JPEG70:" + f for f in FRAMES]
    assert seen["owner"]["ticket"] == [only_alice(f) for f in FRAMES]
    assert seen["owner"]["stream"] == [mjpeg(only_alice(f)) for f in FRAMES]
    assert all(b"ALICE" in p and b"BOBBY" not in p for p in seen["owner"]["ticket"])
    assert seen["blurred"]["ticket"] == [live(f) for f in FRAMES]  # byte for byte the picture without a ticket
    assert owner.asked == len(FRAMES)  # asked once per frame, only in "owner": the two routes share the picture


@pytest.mark.parametrize("why", ["no port", "not tagged yet", "the port fails"])
def test_only_me_is_the_blurred_picture_whenever_the_box_cannot_tell(tmp_path, monkeypatch, why):
    """Nobody scanned, recognition off or the model missing (no port, or it answers nobody); the first second or two
    before a tag exists; the tags changing under the request: every face is blurred, on the owner's screen too."""
    owner = None if why == "no port" else Owner(tagged=why != "not tagged yet")
    if why == "the port fails":
        owner.fail = True
    got = run(tmp_path, monkeypatch, "owner", owner)
    assert got["ticket"] == [live(f) for f in FRAMES] and got["stream"] == [mjpeg(live(f)) for f in FRAMES]
    assert not [n for n in NAMES if n in b"".join(got["ticket"])]


def test_only_me_blurs_everyone_until_a_detection_has_placed_the_people(tmp_path, monkeypatch):
    """A frame the tracker has not looked at yet has no person boxes to go by: cannot tell, so blur."""
    stand_ins(monkeypatch)
    policy, tickets, bus, service, http = base.box(tmp_path)
    owner = service.owner_faces = Owner()
    service.readers["cam-desk"] = base.Reader("cam-desk", "scripted")
    service.readers["cam-desk"].push(0)
    asyncio.run(bus.publish(PolicyChanged(blur_faces=True, live_faces="owner")))
    ticket = {"ticket": tickets.issue(NOW)[0]}
    assert http.get("/capture/cam-desk/snapshot.jpg", params=ticket).content == live(FRAMES[0]) and owner.asked == 0
    assert http.get("/capture/cam-desk/snapshot.jpg", params={"ticket": "nonsense"}).content == live(FRAMES[0])
    assert Policy(tmp_path).live_owner_only and not Policy(tmp_path).live_clear  # and after a restart, from the file


def test_only_me_changes_nothing_on_a_box_that_saves_unblurred_pictures(tmp_path, monkeypatch):
    """`blur_faces` off is an older choice some box may hold: its live picture was unblurred for everyone, and still is."""
    stand_ins(monkeypatch)
    policy, tickets, bus, service, http = base.box(tmp_path)
    service.owner_faces = Owner()
    service.readers["cam-desk"] = base.Reader("cam-desk", "scripted")
    service.readers["cam-desk"].push(0)
    asyncio.run(bus.publish(PolicyChanged(blur_faces=False, live_faces="owner")))
    for query in ({}, {"ticket": tickets.issue(NOW)[0]}):
        assert http.get("/capture/cam-desk/snapshot.jpg", params=query).content == b"JPEG70:" + FRAMES[0]
