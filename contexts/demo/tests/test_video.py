import shutil
import subprocess

import pytest

from contexts.demo import video
from contexts.demo.__main__ import main


def test_every_judge_watches_three_minutes_with_at_most_one_key_press():
    for track in video.TRACKS:
        stretches = video.path(track)
        assert sum(b - a for a, b in stretches) == 180, track.name
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


def test_report_passes_an_exact_export_and_says_how_to_fix_a_long_one():
    ok, text = video.report(600.04)
    assert ok and "PASS" in text
    ok, text = video.report(606.0)
    assert not ok
    assert "key 9 lands at  545.40 s" in text  # 5.4 s late
    assert "-6.00 s" in text


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
