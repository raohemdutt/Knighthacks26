"""The submission video's layout (D10): ten 60 s slots that YouTube's number keys land on.

YouTube's keys 0-9 seek to 0%, 10%, ... 90% of the video, so key k lands at k tenths of it. Devpost caps
the video at 10 minutes, so it is 9:58 (a margin under the cap) and each slot is 59.8 s. Every judge watches slot 0 (the intro), presses their key, and watches two slots: 1 + 2 = 3:00.
Each slot after the intro is shared by two neighbouring tracks, which is why the order below matters.
The general judges press nothing and watch slots 0 to 2.

    python -m contexts.demo video check atlas.mp4   # duration within tolerance, and where each key lands
    python -m contexts.demo video chapters          # the YouTube description: index card + chapters
    python -m contexts.demo video cards             # one title card per slot, as HTML to screen-capture
    python -m contexts.demo video json              # the timeline for the Remotion project (video/src/timeline.json)
"""

import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FPS = 30
DURATION_S = 598.0  # Devpost cap is 10:00; stay under it
TOLERANCE_S = 0.1  # key 9 lands at 90%, so a long export shifts it by 0.9 x the overrun
SLOT_S = DURATION_S / 10
WATCH_SLOTS = 2  # after the intro, every judge watches two slots


@dataclass(frozen=True)
class Slot:
    key: int
    title: str
    shows: str
    line: str = ""  # the narrator's opening line, shown as a caption for the slot's first seconds


@dataclass(frozen=True)
class Track:
    name: str
    key: int | None  # None: keeps watching from the intro, no key press


SLOTS = (
    Slot(0, "Atlas", "Index card (5 s), then the hook: it remembers, it watches, nothing leaves the box. Index card again at 0:55.",
         "This is Atlas: a private AI box that sees, remembers, talks and learns. Your video never leaves it."),
    Slot(1, "It runs your day", "Knows your week (Canvas, calendar), the Day loop plan, one approval card.",
         "Point it at your week: it reads your Canvas and calendar, plans the day around what the cameras see, and asks before it acts."),
    Slot(2, "It learns", "The big Qwen teaches the small Qwen; before/after scores; our adapter on Hugging Face; quick unplug.",
         "When it needs a new skill, a big open model teaches the small one on the box."),
    Slot(3, "The learning lab", "Qwen3.8-27B teacher on R9700 A, LoRA on R9700 B, ROCm, loss curve, publishing to the Hub.",
         "Here is where that teaching happens: two AMD Radeon R9700s on ROCm."),
    Slot(4, "Box and lab", "The Q4_K_M package hot-swapped onto the Jetson; watts from amd-smi and tegrastats; joules per answer.",
         "The lesson comes back as a smaller, smarter model, hot-swapped on the Jetson. We measure every joule both ways."),
    Slot(5, "On the Jetson", "Cameras and face blur; the on-device Qwen VL understands the scene; FPS and tok/s; then unplug the internet.",
         "Everything you see live runs on this Jetson, with the internet unplugged."),
    Slot(6, "Data into counts", "Frames to events to episodes; Waste Watch counting items and the bin zone they end in.",
         "The same cameras turn a room into data: events, episodes and counts."),
    Slot(7, "Impact, shared", "Diversion rate before vs after nudges; the leaderboard served from Neon.",
         "Point it at a recycling bin, and a campus can measure what changed."),
    Slot(8, "Friends' Atlases", "Two Atlases agree on a time through Neon; the event lands in the Day Run plan; the League.",
         "Point it at your friends: Atlases agree on a time through Neon, and only free or busy leaves each box."),
    Slot(9, "Day Run", "A bad plan fails, a good plan wins three stars, hard mode re-routes, Play mode, a Sort It round started by a face.",
         "Point it at your habits: play tomorrow before you live it."),
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
    whole = int(seconds)  # floor: a YouTube chapter must not start after the key's landing point
    return f"{whole // 60}:{whole % 60:02d}"


def path(track: Track, duration: float = DURATION_S) -> list[tuple[float, float]]:
    """The stretches a track's judge watches, in seconds: the intro, then two slots from their key."""
    slot = duration / 10
    start = (1 if track.key is None else track.key) * slot
    stretches = [(0.0, slot), (start, min(start + WATCH_SLOTS * slot, duration))]
    return [(0.0, stretches[1][1])] if track.key is None else stretches


def slots_of(track: Track) -> set[int]:
    """The slots a track's judge watches: the intro, then two slots from their key."""
    start = 1 if track.key is None else track.key
    return {0, *range(start, min(start + WATCH_SLOTS, 10))}


def watchers(key: int) -> list[str]:
    """The tracks whose path includes this slot."""
    return [t.name for t in TRACKS if key in slots_of(t)]


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


def timeline(clips_dir: Path = Path("video/public/clips")) -> dict:
    """The Remotion project's timeline (D12): frame-exact slots, so the number keys land where the edit puts them.
    A slot's screen recording is used when `video/public/clips/slot-<key>.mp4` exists."""
    slot_frames = round(SLOT_S * FPS)
    assert slot_frames * 10 == round(DURATION_S * FPS), "a slot must be a whole number of frames"
    return {
        "fps": FPS,
        "duration_frames": slot_frames * 10,
        "slot_frames": slot_frames,
        "index_card": index_card(),
        "tracks": [{"name": t.name, "key": t.key} for t in TRACKS],
        "slots": [{"key": s.key, "title": s.title, "shows": s.shows, "line": s.line, "watchers": watchers(s.key),
                   "start_frame": s.key * slot_frames, "stamp": stamp(s.key * SLOT_S),
                   "clip": f"clips/slot-{s.key}.mp4" if (clips_dir / f"slot-{s.key}.mp4").exists() else None}
                  for s in SLOTS],
    }
