"""jetson/record_clip.py: recording a clip through the gateway, and the dataset the lab reads from recorded clips.

The gateway is the network boundary: a stand-in serves its HTTP API from this machine (the same routes and answers
as contexts/capture/api.py and contexts/vault/api.py) and GStreamer, which the laptop and CI do not have, is the
`runner` the recorder is handed. Everything else is the real script, and the dataset is read back with the lab's own
reader, lab/dataset.py.
"""
import hashlib
import http.server
import importlib.util
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lab.dataset import load, write_target

spec = importlib.util.spec_from_file_location("record_clip", Path(__file__).parent.parent / "jetson" / "record_clip.py")
record_clip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record_clip)


def fake_jpeg(tag: str, width: int = 1280, height: int = 720) -> bytes:
    """Bytes the recorder reads as a 1280x720 JPEG: start, a comment that makes the frame unique, a frame header, end."""
    comment = b"\xff\xfe" + (2 + len(tag)).to_bytes(2, "big") + tag.encode()
    frame_header = (b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
                    + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01")
    return b"\xff\xd8" + comment + frame_header + b"\xff\xd9"


class FakeClock:
    """Time that moves only when the recorder sleeps, so a ten second clip takes no time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class StandIn:
    """The gateway's API as the recorder uses it. A new frame appears every `every` snapshot asks."""

    def __init__(self) -> None:
        self.blur_faces, self.camera_status, self.every, self.freeze_after = True, "ok", 4, None
        self.asks, self.served, self.track_samples, self.track_asks = 0, [], [[]], 0
        self.url = ""

    def snapshot(self) -> bytes:
        self.asks += 1
        frame_no = (self.asks if self.freeze_after is None else min(self.asks, self.freeze_after)) // self.every
        jpeg = fake_jpeg(f"frame {frame_no}")
        if not self.served or self.served[-1] != jpeg:
            self.served.append(jpeg)
        return jpeg

    def tracks(self) -> list[dict]:
        sample = self.track_samples[min(self.track_asks, len(self.track_samples) - 1)]
        self.track_asks += 1
        return sample

    def answer(self, path: str) -> tuple[int, bytes, str]:
        if path == "/health":
            body: object = {"ok": True, "model": "atlas-base"}
        elif path == "/capture/status":
            body = {"cameras": {"cam-desk": {"status": self.camera_status, "fps_actual": 14.9, "dropped": 0,
                                             "detected": 10, "errors": 0}}}
        elif path == "/vault/policy":
            body = {"blur_faces": self.blur_faces, "paused_cameras": [], "offline": False, "keep_days": None}
        elif path == "/capture/cam-desk/tracks":
            body = {"camera_id": "cam-desk", "ts_ms": 1, "tracks": self.tracks()}
        elif path == "/capture/cam-desk/snapshot.jpg?blur=1":  # the recorder always asks for blurred faces
            return 200, self.snapshot(), "image/jpeg"
        else:
            return 404, b"{}", "application/json"
        return 200, json.dumps(body).encode(), "application/json"


@pytest.fixture
def gateway():
    stand_in = StandIn()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            status, body, kind = stand_in.answer(self.path)
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    stand_in.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = record_clip.Gateway(stand_in.url)
    stand_in.client = client
    yield stand_in
    client.close()
    server.shutdown()
    server.server_close()
    thread.join()


def person(track_id: int, x1: float, x2: float, label: str = "person") -> dict:
    return {"label": label, "confidence": 0.65, "box": {"x1": x1, "y1": 0.2, "x2": x2, "y2": 0.9}, "track_id": track_id}


def fake_gst(written: list) -> object:
    """Stands in for GStreamer: remembers the command and writes the clip file the command names."""

    def run(command: list[str]) -> None:
        written.append(command)
        Path(command[-1].removeprefix("location=")).write_bytes(b"an h264 clip")

    return run


def record_one(gateway, out: Path, name: str = "keys-on-desk", seconds: float = 2.0, **options) -> dict:
    clock, written = FakeClock(), []
    return record_clip.record(gateway.client, name, out, seconds=seconds, clock=clock, sleep=clock.sleep,
                              runner=fake_gst(written), **{"fps": 12, **options})


# --- frames ---------------------------------------------------------------------------------

def test_grab_frames_keeps_each_new_frame_once_with_its_offset():
    pictures = iter([b"a", b"a", b"b", None, b"b", b"c"])
    clock = FakeClock()
    frames = record_clip.grab_frames(lambda: next(pictures), 1.5, clock=clock, sleep=clock.sleep, poll_s=0.25)
    assert frames == [(0.0, b"a"), (0.5, b"b"), (1.25, b"c")]


def test_grab_frames_treats_a_dropped_connection_as_no_picture():
    answers = iter([b"a", ConnectionResetError("reset"), b"b", b"b"])

    def fetch():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    clock = FakeClock()
    assert record_clip.grab_frames(fetch, 1.0, clock=clock, sleep=clock.sleep, poll_s=0.25) == [(0.0, b"a"), (0.5, b"b")]


def test_jpeg_size_reads_the_frame_header_and_refuses_what_is_not_a_jpeg():
    assert record_clip.jpeg_size(fake_jpeg("x", 640, 360)) == (640, 360)
    with pytest.raises(ValueError, match="not a JPEG"):
        record_clip.jpeg_size(b"GIF89a....")
    with pytest.raises(ValueError, match="no start-of-frame"):
        record_clip.jpeg_size(b"\xff\xd8" + b"\xff\xfe\x00\x04ab" + b"\xff\xd9" * 10)


def test_pick_spreads_frames_evenly_with_the_first_and_last_included():
    assert record_clip.pick(10, 4) == [0, 3, 6, 9]
    assert record_clip.pick(3, 6) == [0, 1, 2]
    assert record_clip.pick(10, 1) == [5]


# --- a person arriving or moving ---------------------------------------------------------------

def test_walk_detector_ignores_a_person_sitting_still_or_drifting_slowly():
    detector = record_clip.WalkDetector()
    still = [detector.update(t * 0.5, [person(2, 0.2, 0.9)]) for t in range(20)]
    drifting = record_clip.WalkDetector()
    slow = [drifting.update(float(t), [person(2, 0.2 + 0.05 * t, 0.7 + 0.05 * t)]) for t in range(10)]
    assert (any(still), any(slow)) == (False, False)


def test_walk_detector_fires_when_a_person_arrives_who_was_not_there():
    detector = record_clip.WalkDetector()
    assert [detector.update(0.0, []), detector.update(0.25, []), detector.update(0.5, [person(7, 0.0, 0.2)])] == [False, False, True]


def test_walk_detector_fires_when_a_person_who_was_there_crosses_the_view():
    detector = record_clip.WalkDetector()
    seen = [detector.update(0.0, [person(2, 0.1, 0.3)]), detector.update(0.75, [person(2, 0.25, 0.45)]),
            detector.update(1.5, [person(2, 0.45, 0.65)])]
    assert seen == [False, False, True]


def test_walk_detector_looks_only_at_people():
    detector = record_clip.WalkDetector()
    seen = [detector.update(0.0, [person(1, 0.0, 0.1, label="keys")]), detector.update(0.5, [person(1, 0.7, 0.8, label="keys")]),
            detector.update(1.0, [person(9, 0.0, 0.1, label="package")])]
    assert seen == [False, False, False]


# --- recording ------------------------------------------------------------------------------

def test_record_keeps_every_frame_the_gateway_served_and_joins_them_with_the_pipeline(gateway, tmp_path):
    clock, written = FakeClock(), []
    meta = record_clip.record(gateway.client, "keys-on-desk", tmp_path, seconds=2.0, fps=12, clock=clock, sleep=clock.sleep,
                              runner=fake_gst(written))
    frames_dir = tmp_path / "keys-on-desk" / "frames"
    kept = [path.read_bytes() for path in sorted(frames_dir.glob("*.jpg"))]
    assert kept == gateway.served and len(kept) == meta["frames"] > 10
    assert [path.name for path in sorted(frames_dir.glob("*.jpg"))][:2] == ["00001.jpg", "00002.jpg"]
    assert written == [record_clip.gst_command(frames_dir, 12, tmp_path / "keys-on-desk.mp4")]
    assert meta["mp4_sha256"] == hashlib.sha256(b"an h264 clip").hexdigest() and meta["mp4_bytes"] == len(b"an h264 clip")
    assert (meta["width"], meta["height"], meta["fps_clip"], meta["blur_faces"], meta["when"]) == (1280, 720, 12, True, "now")
    assert meta["fps_measured"] == round(meta["frames"] / 2.0, 2)
    assert json.loads((tmp_path / "keys-on-desk" / "clip.json").read_text()) == meta


def test_record_asks_for_blurred_frames_even_on_a_box_that_saves_unblurred_pictures(gateway, tmp_path):
    gateway.blur_faces = False  # not the default, but a box can hold it: the gateway still blurs what ?blur=1 asks for
    meta = record_one(gateway, tmp_path)
    assert meta["blur_faces"] is True and gateway.asks > 0


def test_record_refuses_a_camera_that_is_not_delivering(gateway, tmp_path):
    gateway.camera_status = "down"
    with pytest.raises(record_clip.RecordError, match="not delivering frames"):
        record_one(gateway, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_record_keeps_nothing_when_the_picture_stalls(gateway, tmp_path):
    gateway.freeze_after = 20
    with pytest.raises(record_clip.RecordError, match="stalled"):
        record_one(gateway, tmp_path, seconds=5.0)
    assert list(tmp_path.iterdir()) == []


def test_record_refuses_a_camera_slower_than_the_clip_rate_and_says_how_to_record_it_as_it_is(gateway, tmp_path):
    gateway.every = 8  # 6.5 frames a second reach the recorder
    with pytest.raises(record_clip.RecordError, match=r"6\.5 frames a second.*--fps auto"):
        record_one(gateway, tmp_path, fps=15)
    assert list(tmp_path.iterdir()) == []
    gateway.asks = 0  # the same stretch of frames again
    assert record_one(gateway, tmp_path, fps=6)["fps_clip"] == 6


def test_record_with_auto_rate_joins_the_clip_at_the_rate_the_camera_gave(gateway, tmp_path):
    gateway.every = 8
    meta = record_one(gateway, tmp_path, fps=None)
    assert meta["fps_clip"] == meta["fps_measured"] == 6.5
    assert "framerate=13/2" in " ".join(meta["gst"])


def test_record_accepts_a_degraded_camera_and_says_so(gateway, tmp_path):
    gateway.camera_status = "degraded"
    meta = record_one(gateway, tmp_path)
    assert meta["gateway_before"]["status"] == "degraded" and "error" not in meta["gateway_after"]


def test_record_keeps_nothing_when_the_pipeline_fails(gateway, tmp_path):
    clock = FakeClock()

    def failing(command: list[str]) -> None:
        Path(command[-1].removeprefix("location=")).write_bytes(b"half a clip")
        raise record_clip.RecordError("GStreamer failed (1): no element jpegdec")

    with pytest.raises(record_clip.RecordError, match="GStreamer failed"):
        record_clip.record(gateway.client, "keys-on-desk", tmp_path, seconds=2.0, fps=12, clock=clock, sleep=clock.sleep, runner=failing)
    assert list(tmp_path.iterdir()) == []


def test_record_keeps_a_good_clip_and_says_so_when_the_gateway_looks_bad_afterwards(gateway, tmp_path):
    clock = FakeClock()

    def encode_then_lose_the_camera(command: list[str]) -> None:
        fake_gst([])(command)
        gateway.camera_status = "down"

    meta = record_clip.record(gateway.client, "keys-on-desk", tmp_path, seconds=2.0, fps=12, clock=clock, sleep=clock.sleep,
                              runner=encode_then_lose_the_camera)
    assert (tmp_path / "keys-on-desk.mp4").is_file() and (tmp_path / "keys-on-desk" / "clip.json").is_file()
    assert "not delivering frames" in meta["gateway_after"]["error"]
    assert meta["gateway_before"] == {"status": "ok", "fps_actual": 14.9, "dropped": 0, "blur_faces": True}


def test_record_will_not_overwrite_an_earlier_clip_of_the_same_name(gateway, tmp_path):
    record_one(gateway, tmp_path)
    before = (tmp_path / "keys-on-desk.mp4").read_bytes(), sorted(path.name for path in (tmp_path / "keys-on-desk").iterdir())
    with pytest.raises(record_clip.RecordError, match="already exists"):
        record_one(gateway, tmp_path)
    assert before == ((tmp_path / "keys-on-desk.mp4").read_bytes(), sorted(path.name for path in (tmp_path / "keys-on-desk").iterdir()))


def test_record_refuses_a_name_that_could_leave_the_clip_folder(gateway, tmp_path):
    with pytest.raises(record_clip.RecordError, match="not a clip name"):
        record_one(gateway, tmp_path, name="../keys")


def test_record_waits_for_a_person_and_then_records(gateway, tmp_path):
    gateway.track_samples = [[], [], [person(4, 0.0, 0.2)]]
    meta = record_one(gateway, tmp_path, name="person-walks-through", when="person-moves", wait_max=30.0)
    assert meta["when"] == "person-moves" and gateway.track_asks == 3 and meta["frames"] > 10


def test_record_gives_up_when_nobody_comes_and_writes_nothing(gateway, tmp_path):
    with pytest.raises(record_clip.NoPerson):
        record_one(gateway, tmp_path, name="person-walks-through", when="person-moves", wait_max=2.0)
    assert list(tmp_path.iterdir()) == [] and gateway.asks == 0


def test_the_command_line_gives_exit_code_1_for_a_refusal_and_3_for_no_person(gateway, tmp_path, capsys):
    assert record_clip.main(["clip", "Not A Name", "--gateway", gateway.url, "--out", str(tmp_path)]) == 1
    assert "not a clip name" in capsys.readouterr().err
    assert record_clip.main(["clip", "walk", "--gateway", gateway.url, "--out", str(tmp_path), "--when", "person-moves",
                             "--wait-max", "0.3"]) == 3
    assert "nobody arrived or moved" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [] and gateway.asks == 0


# --- the dataset the lab reads ---------------------------------------------------------------

def two_recorded_clips(gateway, out: Path) -> dict:
    return {name: record_one(gateway, out, name=name) for name in ("keys-on-desk", "package-at-door")}


def test_the_dataset_is_read_by_the_labs_own_reader_with_the_frames_the_clips_were_made_from(gateway, tmp_path):
    clips = two_recorded_clips(gateway, tmp_path)
    dest = tmp_path / "desk-clips-fixture"
    now = datetime.fromisoformat("2026-10-10T20:00:00+00:00")
    manifest = record_clip.build_dataset(tmp_path, ["keys-on-desk", "package-at-door"], dest, per_clip=4, now=now)
    data = load(dest)
    assert manifest["examples"] == len(data.examples) == 8
    assert (data.manifest.dataset_id, data.manifest.action_id, data.manifest.spec.task, data.manifest.spec.camera) == (
        "desk-clips-fixture", "fixture:desk-clips", "desk-objects", "cam-desk")
    assert data.manifest.created == now and data.manifest.spec.instruction == "Name the small things on the desk."
    assert [example.id for example in data.pending()] == [example.id for example in data.examples]
    first = data.examples[0]
    first_frame = (tmp_path / "keys-on-desk" / "frames" / "00001.jpg").read_bytes()
    assert (first.id, first.frame_ref, first.image, first.context, first.verdict) == (
        "keys-on-desk-00001", "fixture:clip:keys-on-desk:00001", "images/0000.jpg", None, None)
    assert data.image_path(first).read_bytes() == first_frame
    last_of_keys = data.examples[3]
    count = clips["keys-on-desk"]["frames"]
    assert last_of_keys.frame_ref == f"fixture:clip:keys-on-desk:{count:05d}"
    assert data.image_path(last_of_keys).read_bytes() == (tmp_path / "keys-on-desk" / "frames" / f"{count:05d}.jpg").read_bytes()
    started = datetime.fromisoformat(clips["keys-on-desk"]["started"])
    assert last_of_keys.ts == started + timedelta(seconds=clips["keys-on-desk"]["offsets_s"][count - 1])
    assert data.examples[4].frame_ref.startswith("fixture:clip:package-at-door:")


def test_the_dataset_will_not_be_rebuilt_over_labels_the_teacher_already_wrote(gateway, tmp_path):
    two_recorded_clips(gateway, tmp_path)
    dest = tmp_path / "desk-clips-fixture"
    record_clip.build_dataset(tmp_path, ["keys-on-desk"], dest, per_clip=3)
    write_target(dest, "keys-on-desk-00001", "no objects on the desk")
    before = (dest / "examples.jsonl").read_bytes()
    with pytest.raises(record_clip.RecordError, match="already holds labels"):
        record_clip.build_dataset(tmp_path, ["keys-on-desk", "package-at-door"], dest, per_clip=3)
    assert (dest / "examples.jsonl").read_bytes() == before


def test_a_rebuilt_dataset_has_no_images_left_over_from_the_one_before(gateway, tmp_path):
    two_recorded_clips(gateway, tmp_path)
    dest = tmp_path / "desk-clips-fixture"
    record_clip.build_dataset(tmp_path, ["keys-on-desk", "package-at-door"], dest, per_clip=4)
    record_clip.build_dataset(tmp_path, ["keys-on-desk"], dest, per_clip=2)
    assert sorted(path.name for path in (dest / "images").iterdir()) == ["0000.jpg", "0001.jpg"]
    assert len(load(dest).examples) == 2


def test_the_command_line_builds_the_dataset_and_prints_where(gateway, tmp_path, capsys):
    two_recorded_clips(gateway, tmp_path)
    assert record_clip.main(["dataset", "keys-on-desk", "package-at-door", "--per-clip", "3", "--out", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"dataset": str(tmp_path / "desk-clips-fixture"), "examples": 6, "dataset_id": "desk-clips-fixture"}
    assert len(load(tmp_path / "desk-clips-fixture").examples) == 6
