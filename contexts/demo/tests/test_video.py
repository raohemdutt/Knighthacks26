import shutil
from pathlib import Path
import subprocess

import pytest

from contexts.demo import video
from contexts.demo.__main__ import main


def test_every_judge_watches_three_minutes_with_at_most_one_key_press():
    for track in video.TRACKS:
        stretches = video.path(track)
        assert sum(b - a for a, b in stretches) == pytest.approx(3 * video.SLOT_S), track.name
        assert len(stretches) <= 2, track.name
        assert stretches[0][0] == 0, track.name


def test_keys_are_distinct_and_land_on_slot_starts():
    keys = [t.key for t in video.TRACKS if t.key is not None]
    assert len(keys) == len(set(keys))
    assert all(1 <= k <= 8 for k in keys)  # key 9 would leave a judge only one slot
    assert [s.key for s in video.SLOTS] == list(range(10))


def test_every_slot_after_the_intro_has_someone_watching_it():
    for slot in video.SLOTS:
        assert video.watchers(slot.key), slot.title
    assert len(video.watchers(0)) == len(video.TRACKS)
    assert video.watchers(9) == ["Full Sail"]


def test_neighbouring_tracks_share_the_slot_between_them():
    assert video.watchers(3) == ["Hugging Face", "AMD"]
    assert video.watchers(8) == ["Neon", "Full Sail"]


def test_chapters_start_at_zero_and_name_every_key():
    text = video.chapters()
    chapter_lines = [line for line in text.splitlines() if line[:1].isdigit() and "[" in line]
    assert chapter_lines[0].startswith("0:00 [0]")
    assert len(chapter_lines) == 10
    assert "AMD: press 3" in text and "just keep watching" in text


def test_the_video_stays_under_the_ten_minute_cap():
    assert video.DURATION_S + video.TOLERANCE_S < 600


def test_report_passes_an_exact_export_and_says_how_to_fix_a_long_one():
    ok, text = video.report(598.04)
    assert ok and "PASS" in text
    ok, text = video.report(606.0)
    assert not ok
    assert "key 9 lands at  545.40 s" in text  # 7.2 s late
    assert "-8.00 s" in text


def test_chapter_stamps_never_start_after_the_key_lands():
    assert video.stamp(8 * video.SLOT_S) == "7:58"  # key 8 lands at 478.4 s


def test_cards_has_one_section_per_slot(tmp_path):
    out = tmp_path / "cards.html"
    assert main(["video", "cards", str(out)]) == 0
    assert out.read_text(encoding="utf-8").count("<section") == 10


def test_check_needs_a_file(capsys):
    assert main(["video", "check"]) == 2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_check_reads_a_real_file(tmp_path, capsys):
    clip = tmp_path / "short.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2", str(clip)], check=True)
    assert main(["video", "check", str(clip)]) == 1
    assert "FAIL  duration 2.00 s" in capsys.readouterr().out


def test_the_remotion_timeline_is_frame_exact_and_uses_clips_that_exist(tmp_path):
    (tmp_path / "slot-5.mp4").write_bytes(b"")
    t = video.timeline(clips_dir=tmp_path)
    assert (t["fps"], t["slot_frames"], t["duration_frames"]) == (30, 1794, 17940)
    assert [s["start_frame"] for s in t["slots"]] == [k * 1794 for k in range(10)]
    assert t["slots"][5]["clip"] == "clips/slot-5.mp4" and t["slots"][4]["clip"] is None


def test_the_committed_remotion_timeline_matches_the_layout():
    """video/src/timeline.json is generated (`npm run timeline`); a layout change must regenerate it."""
    import json
    from pathlib import Path

    committed = json.loads((Path(__file__).parents[3] / "video" / "src" / "timeline.json").read_text(encoding="utf-8"))
    fresh = video.timeline(clips_dir=Path("no-such-dir"))
    strip = lambda t: {**t, "slots": [{**s, "clip": None} for s in t["slots"]]}  # clips differ per machine
    assert strip(committed) == strip(fresh)


def test_every_slot_has_an_opening_line_that_fits_a_caption():
    for s in video.SLOTS:
        assert s.line and len(s.line) <= 140, s.title
    assert all(s["line"] for s in video.timeline(clips_dir=Path("no-such-dir"))["slots"])
