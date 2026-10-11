"""jetson/reblur_stored.py: find stored keyframes that still show a face, and blur them again in place.

The first part runs everywhere: a "picture" is bytes with `face=NAME` in it, the detector and the blur are stand-ins,
and decode / encode are the identity, so every byte on disk can be written down. The second part needs OpenCV: real
JPEGs, the real decode / scramble / encode, a faked YuNet for the face (the repo holds no picture of a real face),
and the real YuNet on a picture without one.
"""

import importlib.util
import os
import socket
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("reblur_stored", Path(__file__).parent.parent / "jetson" / "reblur_stored.py")
reblur_stored = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reblur_stored)

DAY1 = datetime(2026, 10, 10, 12, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)


def ms(when: datetime) -> int:
    return int(when.timestamp() * 1000)


class Blur:
    """`face=ALICE` is a face. One pass hides a face; a `STUBBORN` face is still found however often it is blurred."""

    def __init__(self):
        self.looked, self.blurred = 0, 0

    def faces(self, frame: bytes) -> list:
        self.looked += 1
        return [[0.0] * 15] * (frame.count(b"face=ALICE") + frame.count(b"STUBBORN"))

    def scrambled_until_clear(self, frame: bytes, passes: int = 5) -> tuple[bytes, int]:
        self.blurred += 1
        return frame.replace(b"face=ALICE", b"face=#####").replace(b"clear STUBBORN", b"blurred STUBBORN"), len(self.faces(frame))


def same(data):
    return data


def store(tmp_path: Path, pictures: dict[str, bytes]) -> Path:
    for name, data in pictures.items():
        path = tmp_path / "frames" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return tmp_path


def snapshot(data_dir: Path) -> dict[str, tuple[bytes, int, int]]:
    """Every file under frames/: its bytes, its modification time and its inode (a rewritten file gets a new one)."""
    return {p.relative_to(data_dir).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino)
            for p in sorted((data_dir / "frames").rglob("*")) if p.is_file()}


@pytest.fixture
def box(tmp_path):
    return store(tmp_path, {
        f"cam-desk/{ms(DAY1)}.jpg": b"desk mug face=ALICE",
        f"cam-desk/{ms(DAY1) + 5000}.jpg": b"desk mug",
        f"cam-desk/{ms(DAY2)}.jpg": b"desk face=ALICE and face=ALICE",
        f"cam-door/{ms(DAY2)}.jpg": b"door mat",
        f"cam-door/{ms(DAY2) + 5000}.jpg": b"door face=ALICE",
        "cam-door/notes.jpg": b"not a keyframe face=ALICE",  # not named by a timestamp: not a keyframe, left alone
        f"cam-door/.{ms(DAY2) + 9000}.tmp": b"a writer's half file face=ALICE",
    })


def run(data_dir, blur=None, **kwargs):
    return reblur_stored.reblur(data_dir, blur or Blur(), tz=UTC, decode=same, encode=same, **kwargs)


@pytest.fixture
def plain(monkeypatch):
    """For the command line, which reads and writes pictures with OpenCV: here a picture is its bytes."""
    monkeypatch.setattr(reblur_stored, "read_picture", same)
    monkeypatch.setattr(reblur_stored, "write_picture", same)


def counts(report) -> dict:
    return {key: (row.scanned, row.with_face, row.reblurred, row.still_found, row.unreadable) for key, row in report.rows.items()}


def test_a_dry_run_counts_per_camera_and_per_day_and_changes_nothing(box):
    before = snapshot(box)
    blur = Blur()
    report = run(box, blur)
    assert counts(report) == {("cam-desk", "2026-10-10"): (2, 1, 0, 0, 0), ("cam-desk", "2026-10-11"): (1, 1, 0, 0, 0),
                              ("cam-door", "2026-10-11"): (2, 1, 0, 0, 0)}
    assert [p.relative_to(box).as_posix() for p in report.with_face] == [
        f"frames/cam-desk/{ms(DAY1)}.jpg", f"frames/cam-desk/{ms(DAY2)}.jpg", f"frames/cam-door/{ms(DAY2) + 5000}.jpg"]
    assert blur.blurred == 0 and snapshot(box) == before  # bytes, dates and the files themselves
    total = report.total
    assert (total.scanned, total.with_face, total.reblurred) == (5, 3, 0)


def test_apply_blurs_the_files_with_a_face_and_never_touches_one_without(box):
    before = snapshot(box)
    report = run(box, apply=True)
    after = snapshot(box)
    assert counts(report) == {("cam-desk", "2026-10-10"): (2, 1, 1, 0, 0), ("cam-desk", "2026-10-11"): (1, 1, 1, 0, 0),
                              ("cam-door", "2026-10-11"): (2, 1, 1, 0, 0)}
    assert after[f"frames/cam-desk/{ms(DAY1)}.jpg"][0] == b"desk mug face=#####"
    assert after[f"frames/cam-desk/{ms(DAY2)}.jpg"][0] == b"desk face=##### and face=#####"
    assert after[f"frames/cam-door/{ms(DAY2) + 5000}.jpg"][0] == b"door face=#####"
    untouched = [f"frames/cam-desk/{ms(DAY1) + 5000}.jpg", f"frames/cam-door/{ms(DAY2)}.jpg", "frames/cam-door/notes.jpg",
                 f"frames/cam-door/.{ms(DAY2) + 9000}.tmp"]
    assert all(after[name] == before[name] for name in untouched)  # same bytes, same date, the very same file
    assert set(after) == set(before)  # no temporary file is left, none is gone
    assert report.still_found == [] and report.problems == []


def test_a_second_apply_does_nothing(box):
    run(box, apply=True)
    once = snapshot(box)
    blur = Blur()
    again = run(box, blur, apply=True)
    assert snapshot(box) == once and blur.blurred == 0
    assert (again.total.scanned, again.total.with_face, again.total.reblurred) == (5, 0, 0)


def test_since_limits_it_to_pictures_taken_at_or_after_that_time(box):
    before = snapshot(box)
    report = run(box, apply=True, since_ms=ms(DAY2))
    assert set(counts(report)) == {("cam-desk", "2026-10-11"), ("cam-door", "2026-10-11")}
    after = snapshot(box)
    assert after[f"frames/cam-desk/{ms(DAY1)}.jpg"] == before[f"frames/cam-desk/{ms(DAY1)}.jpg"]  # older: still as it was
    assert after[f"frames/cam-desk/{ms(DAY2)}.jpg"][0] == b"desk face=##### and face=#####"  # exactly at the time: included
    assert reblur_stored.since_ms("2026-10-11T12:00:00+00:00") == ms(DAY2) == reblur_stored.since_ms("2026-10-11T12:00:00Z")
    assert reblur_stored.since_ms("2026-10-11T08:00:00-04:00") == ms(DAY2)
    assert reblur_stored.since_ms("2026-10-11T12:00", UTC) == ms(DAY2)  # no offset: the zone it is given (the box's own)


def test_days_are_counted_in_the_zone_given(tmp_path):
    late = datetime(2026, 10, 10, 23, 30, tzinfo=UTC)
    store(tmp_path, {f"cam-desk/{ms(late)}.jpg": b"face=ALICE"})
    east = timezone(timedelta(hours=9))
    report = reblur_stored.reblur(tmp_path, Blur(), tz=east, decode=same, encode=same)
    assert list(report.rows) == [("cam-desk", "2026-10-11")]


def test_a_face_still_found_after_blurring_is_counted_and_named(tmp_path):
    name = f"cam-desk/{ms(DAY1)}.jpg"
    store(tmp_path, {name: b"clear STUBBORN", f"cam-desk/{ms(DAY1) + 1}.jpg": b"face=ALICE"})
    report = run(tmp_path, apply=True)
    assert (tmp_path / "frames" / name).read_bytes() == b"blurred STUBBORN"  # it was blurred all the same
    assert report.still_found == [tmp_path / "frames" / name]
    assert counts(report) == {("cam-desk", "2026-10-10"): (2, 2, 2, 1, 0)}


def test_a_file_that_cannot_be_read_is_named_and_the_run_goes_on(tmp_path):
    bad, good = f"cam-desk/{ms(DAY1)}.jpg", f"cam-desk/{ms(DAY1) + 1}.jpg"
    store(tmp_path, {bad: b"BROKEN", good: b"face=ALICE"})

    def decode(data):
        if data == b"BROKEN":
            raise ValueError("not a picture")
        return data

    report = reblur_stored.reblur(tmp_path, Blur(), apply=True, tz=UTC, decode=decode, encode=same)
    assert (tmp_path / "frames" / bad).read_bytes() == b"BROKEN" and (tmp_path / "frames" / good).read_bytes() == b"face=#####"
    assert [(p.name, why) for p, why in report.problems] == [(Path(bad).name, "ValueError: not a picture")]
    assert counts(report) == {("cam-desk", "2026-10-10"): (2, 1, 1, 0, 1)}


def test_a_write_that_fails_leaves_the_old_file_whole_and_no_temporary_file(tmp_path, monkeypatch):
    name = f"cam-desk/{ms(DAY1)}.jpg"
    store(tmp_path, {name: b"face=ALICE"})
    before = snapshot(tmp_path)

    def no_rename(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(reblur_stored.os, "replace", no_rename)
    report = run(tmp_path, apply=True)
    assert snapshot(tmp_path) == before and [why for _, why in report.problems] == ["OSError: disk full"]


def test_the_replacement_keeps_the_file_mode_and_links_are_not_followed(tmp_path):
    name = f"cam-desk/{ms(DAY1)}.jpg"
    store(tmp_path, {name: b"face=ALICE"})
    path = tmp_path / "frames" / name
    path.chmod(0o640)
    outside = tmp_path / "elsewhere.jpg"
    outside.write_bytes(b"face=ALICE somewhere else")
    (path.parent / f"{ms(DAY1) + 1}.jpg").symlink_to(outside)
    report = run(tmp_path, apply=True)
    assert path.read_bytes() == b"face=#####" and path.stat().st_mode & 0o777 == 0o640
    assert outside.read_bytes() == b"face=ALICE somewhere else" and report.total.scanned == 1


def test_the_command_line_dry_run_is_the_default_and_says_what_it_found(box, plain, capsys):
    before = snapshot(box)
    assert reblur_stored.main([str(box)], blur=Blur(), tz=UTC) == 2  # faces are there
    out = capsys.readouterr().out
    table = [line.split() for line in out.splitlines()]
    assert ["camera", "day", "scanned", "with", "a", "face"] in table
    assert ["cam-desk", "2026-10-10", "2", "1"] in table and ["cam-door", "2026-10-11", "2", "1"] in table and ["total", "5", "3"] in table
    assert "dry run: nothing was changed. Run again with --apply to blur the 3 files with a face again." in out
    assert "face:" not in out and snapshot(box) == before
    assert reblur_stored.main([str(box), "--dry-run", "--names", "--since", "2026-10-11T00:00:00+00:00"], blur=Blur(), tz=UTC) == 2
    out = capsys.readouterr().out
    assert out.count("face: ") == 2 and f"face: {box / 'frames' / 'cam-door' / (str(ms(DAY2) + 5000) + '.jpg')}" in out
    assert "2026-10-10" not in out.split("\n", 1)[1] and snapshot(box) == before


def test_the_command_line_apply_then_a_dry_run_that_finds_nothing(box, plain, capsys):
    assert reblur_stored.main([str(box), "--apply"], blur=Blur(), tz=UTC) == 0
    out = capsys.readouterr().out
    assert "re-blurred  still found" in out and "3 re-blurred; no stored keyframe has a face the detector finds." in out
    assert reblur_stored.main([str(box)], blur=Blur(), tz=UTC) == 0
    assert "No stored keyframe has a face the detector finds." in capsys.readouterr().out


def test_the_command_line_refuses_what_it_cannot_do(tmp_path, plain, capsys):
    assert reblur_stored.main([str(tmp_path / "nowhere")], blur=Blur()) == 1
    assert "no frames folder" in capsys.readouterr().err
    (tmp_path / "frames").mkdir()
    assert reblur_stored.main([str(tmp_path), "--since", "last tuesday"], blur=Blur()) == 1
    assert "--since is not an ISO time" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        reblur_stored.main([str(tmp_path), "--dry-run", "--apply"], blur=Blur())
    assert reblur_stored.main([str(tmp_path)], blur=Blur()) == 0  # an empty box: nothing to do


def test_it_opens_no_network_connection(box, plain, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert reblur_stored.main([str(box), "--apply"], blur=Blur(), tz=UTC) == 0
    source = (Path(__file__).parent.parent / "jetson" / "reblur_stored.py").read_text()
    assert not [word for word in ("socket", "urllib", "http", "requests", "httpx", "subprocess") if word in source]


# --- with OpenCV: real JPEGs through the real decode, scramble and encode -----------------------------------------------

FACE = (120, 60, 80, 80)


def noise_jpeg() -> bytes:
    import cv2
    import numpy as np

    noise = np.random.default_rng(7).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return jpeg.tobytes()


def detail(jpeg: bytes, region) -> float:
    """How much fine detail is left in a region: the mean step between neighbouring pixels."""
    import cv2
    import numpy as np

    x, y, w, h = region
    grey = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)[y:y + h, x:x + w].astype(np.int16)
    return float(np.abs(np.diff(grey, axis=1)).mean())


class FaceWhileSharp:
    """A YuNet stand-in that reports the face only while that region of the frame still has detail in it."""

    def detect(self, frame):
        import numpy as np

        x, y, w, h = FACE
        patch = frame[y:y + h, x:x + w].astype(np.int16)
        sharp = float(np.abs(np.diff(patch, axis=1)).mean()) > 20
        return 1, (np.array([[*FACE, *([0.0] * 10), 0.95]], dtype=np.float32) if sharp else None)


def test_with_opencv_the_face_region_is_scrambled_in_place_and_a_second_run_changes_nothing(tmp_path):
    pytest.importorskip("cv2")
    from contexts.perception.infrastructure.face_blur import STRICT, FaceBlur

    face, plain = f"cam-desk/{ms(DAY1)}.jpg", f"cam-desk/{ms(DAY1) + 1}.jpg"
    store(tmp_path, {face: noise_jpeg(), plain: noise_jpeg()})
    os.utime(tmp_path / "frames" / plain, ns=(1, 1))
    blur = FaceBlur(**STRICT, detector=FaceWhileSharp())

    report = reblur_stored.reblur(tmp_path, blur, tz=UTC)
    assert (report.total.scanned, report.total.with_face) == (2, 2) and (tmp_path / "frames" / face).read_bytes() == noise_jpeg()

    report = reblur_stored.reblur(tmp_path, blur, apply=True, tz=UTC)
    assert (report.total.reblurred, report.total.still_found, report.problems) == (2, 0, [])
    written = (tmp_path / "frames" / face).read_bytes()
    assert written[:2] == b"\xff\xd8" and detail(noise_jpeg(), FACE) > 40 and detail(written, FACE) < 5
    assert detail(written, (0, 180, 320, 60)) > 40  # the rest of the picture is still the picture
    assert sorted(p.name for p in (tmp_path / "frames" / "cam-desk").iterdir()) == [Path(face).name, Path(plain).name]

    once = snapshot(tmp_path)
    again = reblur_stored.reblur(tmp_path, blur, apply=True, tz=UTC)
    assert (again.total.with_face, again.total.reblurred) == (0, 0) and snapshot(tmp_path) == once


def test_with_the_real_detector_a_picture_without_a_face_is_never_rewritten(tmp_path, capsys):
    pytest.importorskip("cv2")
    name = f"cam-desk/{ms(DAY1)}.jpg"
    store(tmp_path, {name: noise_jpeg(), f"cam-desk/{ms(DAY1) + 1}.jpg": b"not a jpeg at all"})
    before = snapshot(tmp_path)
    assert reblur_stored.main([str(tmp_path), "--apply"], tz=UTC) == 1  # the real YuNet, the real decode; one bad file
    out = capsys.readouterr().out
    assert "problem: " in out and "not a picture OpenCV can read" in out
    assert snapshot(tmp_path) == before
