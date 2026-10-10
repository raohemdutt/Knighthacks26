"""The submission video's layout (D10): ten 60 s slots that YouTube's number keys land on.

YouTube's keys 0-9 seek to 0%, 10%, ... 90% of the video, so a video of exactly 10:00 puts key k on
minute k. Every judge watches slot 0 (the intro), presses their key, and watches two slots: 1 + 2 = 3:00.
Each slot after the intro is shared by two neighbouring tracks, which is why the order below matters.
The general judges press nothing and watch slots 0 to 2.

    python -m contexts.demo video check atlas.mp4   # duration within tolerance, and where each key lands
    python -m contexts.demo video chapters          # the YouTube description: index card + chapters
    python -m contexts.demo video cards             # one title card per slot, as HTML to screen-capture
"""

import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DURATION_S = 600.0
TOLERANCE_S = 0.1  # key 9 lands at 90%, so a long export shifts it by 0.9 x the overrun
SLOT_S = DURATION_S / 10
WATCH_SLOTS = 2  # after the intro, every judge watches two slots


@dataclass(frozen=True)
class Slot:
    key: int
    title: str
    shows: str


@dataclass(frozen=True)
class Track:
    name: str
    key: int | None  # None: keeps watching from the intro, no key press


SLOTS = (
    Slot(0, "Atlas", "Index card (5 s), then the hook: it remembers, it watches, nothing leaves the box. Index card again at 0:55."),
    Slot(1, "It runs your day", "Knows your week (Canvas, calendar), the Day loop plan, one approval card."),
    Slot(2, "It learns", "The big Qwen teaches the small Qwen; before/after scores; our adapter on Hugging Face; quick unplug."),
    Slot(3, "The learning lab", "Qwen3.8-27B teacher on R9700 A, LoRA on R9700 B, ROCm, loss curve, publishing to the Hub."),
    Slot(4, "Box and lab", "The Q4_K_M package hot-swapped onto the Jetson; watts from amd-smi and tegrastats; joules per answer."),
    Slot(5, "On the Jetson", "Cameras, the on-device Qwen VL, face blur, FPS and tok/s, then unplug the internet."),
    Slot(6, "Data into counts", "Frames to events to episodes; Waste Watch counting items and the bin zone they end in."),
    Slot(7, "Impact, shared", "Diversion rate before vs after nudges; the leaderboard served from Neon."),
    Slot(8, "Friends' Atlases", "Two Atlases agree on a time through Neon; the event lands in the Day Run plan; the League."),
    Slot(9, "Day Run", "A bad plan fails, a good plan wins three stars, hard mode re-routes, Play mode, a Sort It round started by a face."),
)

TRACKS = (
    Track("Best Overall / Best AI / Best Embedded", None),
    Track("Hugging Face", 2),
    Track("AMD", 3),
    Track("NVIDIA", 4),
    Track("Encompass", 5),
    Track("OneEthos", 6),
    Track("Neon", 7),
    Track("Full Sail", 8),
)


def stamp(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def path(track: Track, duration: float = DURATION_S) -> list[tuple[float, float]]:
    """The stretches a track's judge watches, in seconds: the intro, then two slots from their key."""
    slot = duration / 10
    start = (1 if track.key is None else track.key) * slot
    stretches = [(0.0, slot), (start, min(start + WATCH_SLOTS * slot, duration))]
    return [(0.0, stretches[1][1])] if track.key is None else stretches


def watchers(key: int) -> list[str]:
    """The tracks whose path includes this slot."""
    return [t.name for t in TRACKS if any(a <= key * SLOT_S < b for a, b in path(t))]


def index_card() -> str:
    lines = ["ATLAS: watch 1 minute, press your number key, watch 2 minutes."]
    for t in TRACKS:
        lines.append(f"{t.name}: " + ("just keep watching" if t.key is None else f"press {t.key}"))
    return "\n".join(lines)


def chapters() -> str:
    """The YouTube description: the index card, then chapters (first at 0:00, each over 10 s)."""
    rows = [f"{stamp(s.key * SLOT_S)} [{s.key}] {s.title}: {'everyone' if s.key == 0 else ' · '.join(watchers(s.key))}"
            for s in SLOTS]
    return index_card() + "\n\nChapters (number keys jump to the same places on a computer):\n" + "\n".join(rows)


def probe(file: Path) -> float:
    """The file's duration in seconds, from ffprobe."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is not installed (it comes with ffmpeg)")
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(file)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def report(duration: float) -> tuple[bool, str]:
    """Whether the export is the right length, and where each key really lands in it."""
    ok = abs(duration - DURATION_S) <= TOLERANCE_S
    lines = [f"{'PASS' if ok else 'FAIL'}  duration {duration:.2f} s (want {DURATION_S:.1f} ± {TOLERANCE_S} s)"]
    for s in SLOTS:
        lands = s.key * duration / 10
        drift = lands - s.key * SLOT_S
        lines.append(f"  key {s.key} lands at {lands:7.2f} s ({stamp(lands)}), drift {drift:+.2f} s: {s.title}")
    if not ok:
        lines.append(f"  fix: trim or pad the edit by {DURATION_S - duration:+.2f} s, then export again")
    return ok, "\n".join(lines)


CARD = """<section style="width:1920px;height:1080px;display:flex;flex-direction:column;justify-content:center;
padding:0 160px;box-sizing:border-box;background:#14161a;color:#f3f1ec;font-family:Inter,Arial,sans-serif;page-break-after:always">
<div style="font-size:40px;color:#9aa3ad">{stamp} · key {key}</div>
<div style="font-size:120px;font-weight:700;margin:24px 0">{title}</div>
<div style="font-size:44px;color:#f2a65a">{watchers}</div>
</section>"""


def cards() -> str:
    """One 1920x1080 title card per slot (show 1 to 2 s at the start of each slot), as a single HTML page."""
    body = "\n".join(CARD.format(stamp=stamp(s.key * SLOT_S), key=s.key, title=html.escape(s.title),
                                 watchers=html.escape(" · ".join(watchers(s.key))))
                     for s in SLOTS)
    return f"<!doctype html><meta charset=utf-8><title>Atlas video title cards</title><body style=margin:0>{body}</body>"
