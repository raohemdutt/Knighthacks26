"""Tool-calling dataset v3: v2's generator, its scenes taken from the box's own memory instead of a made-up night.

    python lab/fixture_dataset_v3.py --events <box events.jsonl.gz> --out <folder>          # writes <folder>/tool-calling-v3/
    python lab/fixture_dataset_v3.py --events ... --snapshot <memory.sqlite> --out ... --tokenizer ~/models/Qwen3.5-4B

What changes from v2 (lab/fixture_dataset_v2.py), and what does not:

    the memory     v2 asked its questions of fixtures/golden/memory.yaml plus a made-up sighting per object. v3 replays
                   the box's event log (the perception and memory events, in the order the box logged them) through
                   the memory context's own MemoryProjection into an SQLite store, and stops at each example's clock:
                   every reference call and every tool result is what the agent's code returns against what the box
                   knew at that moment. With --snapshot, the build first replays everything and checks that it lands
                   on the box's memory.sqlite exactly (episodes, their ends and labels, sightings, tracks, notes).
    the scenes     a group is one real thing asked: an object the box saw (last_seen-<object>), or one real episode
                   (scene-<episode id>) asked about with windows that contain its start. Object names are the labels
                   the box's object lists and detector wrote, said the way ASKABLE says them.
    the clocks     every now is the time of a real logged event (the box's clock as it printed it: UTC, microseconds),
                   or that time at -04:00 cut to the whole minute (the fixture's style), half and half by seed, as v2.
    the split      by whole group, and by time: train groups are asked before VALIDATION_EPISODES episodes from the
                   end, validation groups between that and HELD_OUT_EPISODES from the end, held-out groups after. So no
                   train or validation tool result holds anything the box saw in a later part's stretch.
    reserved       train and validation hold nothing of the held-out side of tool-calling-fixture (keys, wallet, a 60
                   minute window, overnight words) or of tool-calling-v2 (its held-out objects and window lengths);
                   those may appear on v3's held-out side, which never trains. Overnight words appear nowhere.
    long results   the box's real tool results run to tens of thousands of characters (every caption of every hour);
                   a row shows them shortened by clip() (LINE_CHARS, RESULT_CHARS) and its answer is written from what
                   is shown, so rows fit the 4,096 tokens the box serves with. The call is never touched.
    the teacher    questions the rules cannot label ("when did I last see my X?", "what happened to my X?") are
                   asked of the teacher by lab/fixture_dataset_v3_teacher.py; with --teacher its kept lines join
                   the train side as teacher-labeled rows (groups teacher-<thing>), listed in source.json.
    unchanged      the labels are reference_trace() of lab/fixture_dataset.py (the agent's own pre-fetch, no teacher);
                   the question templates and stems are v2's; rows are lab/format_traces.py rows checked the same way;
                   the folder's files and their meaning are v2's; the build reads no wall clock and no random source.

Not taken from the box: injected events. Two scripted test runs wrote events on Sat 10-10 (18:36 and 19:13 UTC: a
cam-door the box does not have, confidence 0.9, frame refs frames:cam-desk:9 and :12, whole-second times). An
event is dropped as injected when its time has no fraction of a second and its frame_ref is missing or not a real
frame (frames:<camera>:<13-digit ms>). source.json counts them.

The entry point writes a ledger row (verb fixture-dataset-v3) and refuses to run while lab/state/STOP exists.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
from dataclasses import replace
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds contracts/, contexts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contexts.agent.application.qa import SYSTEM, TOOLS, SleepSchedule, time_window
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.infrastructure.store import MemoryStore
from contracts.events import parse_event
from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
from lab import fixture_dataset_v2 as v2
from lab import format_traces as ft
from lab.dataset import EXAMPLES, IMAGES, MANIFEST, SPLIT, load_split
from lab.fixture_dataset import (AGENT_CONFIG, EXPECTED, TASK, BuildError, Expected, ExpectedLabel, Trace,
                                 Unsupported, _label,
                                 _normal, _sentence, _write, instruction, output_schema, reference_trace)
from lab.machine import stop
from lab.machine.record import recorded

__all__ = ["DATASET_ID", "ACTION_ID", "SOURCE", "ASKABLE", "HELD_OUT_OBJECTS", "VALIDATION_OBJECTS",
           "HELD_OUT_EPISODES", "VALIDATION_EPISODES", "TRAIN_MINUTES", "VALIDATION_MINUTES", "HELD_OUT_MINUTES",
           "Plan", "Built", "read_events", "injected", "replay", "verify_snapshot", "plan", "build", "main"]

DATASET_ID = "tool-calling-v3"
ACTION_ID = "fixture:tool-calling-v3"
SOURCE = "source.json"
TEACHER = "teacher.jsonl"
REPLAYED = ("perception.observation.recorded", "perception.zone.event", "perception.scene.described",
            "memory.episode.closed", "agent.note.written")
REAL_FRAME = re.compile(r"^frames:[a-z0-9-]+:\d{13}$")

# Labels the box's object lists or detector wrote -> (how a person asks for it, plural). A label not here is never
# asked about: the detector's misses on this desk (surfboard, airplane, toilet, skis, bus...) are nobody's question.
ASKABLE = {
    "laptop": ("laptop", False), "bottle": ("bottle", False), "charger": ("charger", False),
    "phone": ("phone", False), "cell phone": ("cell phone", False), "can": ("can", False), "cup": ("cup", False),
    "keys": ("keys", True), "remote": ("remote", False), "book": ("book", False), "backpack": ("backpack", False),
    "umbrella": ("umbrella", False), "keyboard": ("keyboard", False), "mouse": ("mouse", False),
    "toothbrush": ("toothbrush", False), "vase": ("vase", False), "bowl": ("bowl", False),
    "suitcase": ("suitcase", False), "tie": ("tie", False), "potted plant": ("potted plant", False),
    "package": ("package", False), "scissors": ("scissors", True), "handbag": ("handbag", False),
}
# Held out: the first fixture's keys and v2's backpack (both held out there too, so they may only be tested here),
# and two objects v3 alone holds out. Validation: three more. Every other askable object the box saw trains.
HELD_OUT_OBJECTS = ("keys", "backpack", "cup", "suitcase")
VALIDATION_OBJECTS = ("can", "umbrella", "toothbrush")
HELD_OUT_EPISODES = 16       # the last 16 real episodes are held-out scenes
VALIDATION_EPISODES = 12     # the 12 before them are validation scenes
# Window lengths per part. Train: v2's train lengths and its named windows. Validation: v2's validation lengths.
# Held out: v2's held-out lengths and the first fixture's 60 minutes, none of which v3's train or validation shows.
TRAIN_MINUTES = tuple(m for m in v2.WINDOW_MINUTES if m not in v2.HELD_OUT_WINDOWS + v2.VALIDATION_WINDOWS)
TRAIN_NAMED = tuple(name for name, _ in v2.NAMED_WINDOWS)
VALIDATION_MINUTES = v2.VALIDATION_WINDOWS
HELD_OUT_MINUTES = v2.HELD_OUT_WINDOWS + (60,)
ROWS_PER_OBJECT = {"train": 10, "validation": 10, "held_out": 12}
ROWS_PER_SCENE = {"train": 8, "validation": 8, "held_out": 12}
NO_EVIDENCE_EVERY = 5        # one object row in five may be asked at any time of its part, seen yet or not
OPTIONS_PER_ROW = 3          # options planned per row: an option whose row is too long or whose scene the box did
                             # not have yet at its clock is passed over for the next one
# The box's real tool results are long: a window over a busy stretch returns every caption of every hour (median
# 22,000 characters for a train window, 35,000 for a held-out one, before any answer). A row shows them shortened
# by rule (clip()), so it fits the 4,096 tokens the box and pod D serve with; the call is never touched.
LINE_CHARS = 300             # a result line longer than this is cut at a word boundary and ends with "..."
RESULT_CHARS = 2_400         # whole lines are kept while the result stays within this; then "(N more lines)"
MAX_ROW_CHARS = 12_000       # a row longer than this (it should not happen after clip()) is passed over
MARGIN = timedelta(seconds=5)  # a window's now is at least this long after its scene's start, so the box had it
EDT = timezone(timedelta(hours=-4))
PARTS = ("train", "validation", "held_out")
# Train and validation never hold these (held out by an earlier dataset); v3's held-out side may.
EARLIER_HELD_OUT = {"items": set(v2.RESERVED["items"]) | set(v2.HELD_OUT_ITEMS),
                    "minutes": set(v2.RESERVED["minutes"]) | set(v2.HELD_OUT_WINDOWS)}


@dataclass(frozen=True)
class Scene:
    episode_id: str
    camera_id: str
    start: datetime


@dataclass
class Plan:
    """What the box's log supports, from one full replay: the scenes in time order, the parts' time ranges, the
    clock candidates (every logged time), and when each askable label was first seen."""
    scenes: list[Scene]
    parts: dict[str, tuple[datetime, datetime]]          # part -> [from, to): the clocks its examples may have
    scene_part: dict[str, str]
    times: list[datetime]
    first_seen: dict[str, datetime]
    objects: dict[str, str]                              # askable label the box saw -> part
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Built:
    root: Path
    examples: int
    groups: int
    parts: dict[str, dict[str, int]]
    rows: dict[str, int]
    sha256: dict[str, str]
    tokenizer: str
    dropped: dict[str, int]


# --- the box's log ----------------------------------------------------------------------------------------------

def _ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def injected(row: dict[str, Any]) -> bool:
    """A scripted test event, not the camera: a whole-second time and no real frame."""
    stamp = row.get("ts") or ""
    whole = "." not in stamp
    ref = row.get("frame_ref")
    return whole and (ref is None or not REAL_FRAME.match(str(ref)))


def read_events(path: str | Path) -> tuple[list[dict[str, Any]], Counter]:
    """The replayed event types in the order the box logged them, injected ones dropped; and what was dropped."""
    opener: Callable[..., Any] = gzip.open if str(path).endswith(".gz") else open
    kept: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") not in REPLAYED:
                    continue
                if injected(row):
                    dropped[row["type"]] += 1
                    continue
                kept.append(row)
    except (OSError, ValueError) as exc:
        raise BuildError(f"cannot read the box's events: {type(exc).__name__}") from None
    return kept, dropped


def replay(rows: Iterable[dict[str, Any]], checkpoints: Sequence[datetime] = (),
           visit: Callable[[datetime, SqliteMemory], None] | None = None, path: str = ":memory:") -> MemoryStore:
    """Project the rows into a new store in log order. Before the first row stamped after a checkpoint, `visit`
    sees the memory as it stood (checkpoints in time order; any left at the end see the final memory)."""
    store = MemoryStore(path)
    projection = MemoryProjection(store)
    handlers = {"perception.observation.recorded": projection.on_observation,
                "perception.zone.event": projection.on_zone_event,
                "perception.scene.described": projection.on_scene,
                "memory.episode.closed": projection.on_episode_closed,
                "agent.note.written": projection.on_note}
    pending = sorted(checkpoints)
    memory = SqliteMemory(store)
    index = 0
    for row in rows:
        stamp = _ts(row["ts"])
        while visit and index < len(pending) and stamp > pending[index]:
            visit(pending[index], memory)
            index += 1
        handlers[row["type"]](parse_event(row))
    while visit and index < len(pending):
        visit(pending[index], memory)
        index += 1
    return store


def _state(db: sqlite3.Connection) -> dict[str, Any]:
    db.row_factory = sqlite3.Row
    return {"episodes": sorted((r["id"], r["end_ts"], r["labels"]) for r in db.execute(
                "SELECT id, end_ts, labels FROM memory_episodes")),
            "sightings": db.execute("SELECT count(*) FROM memory_sightings").fetchone()[0],
            "tracks": sorted((r["label"], r["ts"], r["zone_id"]) for r in db.execute(
                "SELECT label, ts, zone_id FROM memory_tracks")),
            "notes": sorted(r["id"] for r in db.execute("SELECT id FROM memory_notes"))}


def verify_snapshot(events: str | Path, snapshot: str | Path) -> dict[str, Any]:
    """Replay every replayed event, injected ones included, and compare with the box's memory.sqlite."""
    opener: Callable[..., Any] = gzip.open if str(events).endswith(".gz") else open
    with opener(events, "rt", encoding="utf-8") as handle:
        rows = [row for line in handle if line.strip() and (row := json.loads(line)).get("type") in REPLAYED]
    ours = _state(replay(rows).db)
    box = _state(sqlite3.connect(f"file:{Path(snapshot).as_posix()}?mode=ro", uri=True))
    # the snapshot may be a moment later than the copy of the log: compare what both hold
    last = max(_ts(row["ts"]) for row in rows)
    box_eps = [e for e in box["episodes"] if e[0] in {o[0] for o in ours["episodes"]}
               or int(e[0].rsplit("-", 1)[1]) / 1000 <= last.timestamp()]
    same = {"episodes": ours["episodes"] == box_eps, "tracks": ours["tracks"] == box["tracks"],
            "notes": ours["notes"] == box["notes"], "sightings": ours["sightings"] == box["sightings"]}
    return {"same": same, "episodes": len(ours["episodes"]), "box_episodes": len(box["episodes"]),
            "sightings": ours["sightings"], "box_sightings": box["sightings"], "tracks": len(ours["tracks"]),
            "notes": len(ours["notes"])}


# --- the plan: scenes, parts, objects --------------------------------------------------------------------------

def plan(rows: Sequence[dict[str, Any]], *, held_out_episodes: int = HELD_OUT_EPISODES,
         validation_episodes: int = VALIDATION_EPISODES, askable: dict[str, tuple[str, bool]] | None = None,
         held_out_objects: Sequence[str] = HELD_OUT_OBJECTS,
         validation_objects: Sequence[str] = VALIDATION_OBJECTS) -> Plan:
    askable = ASKABLE if askable is None else askable
    store = replay(rows)
    scenes = [Scene(r[0], r[1], _ts(r[2])) for r in store.db.execute(
        "SELECT id, camera_id, start_ts FROM memory_episodes ORDER BY start_ts, id")]
    if len(scenes) < held_out_episodes + validation_episodes + 1:
        raise BuildError(f"{len(scenes)} episodes cannot be split into {held_out_episodes} held out, "
                         f"{validation_episodes} validation and some train")
    times = sorted({_ts(row["ts"]) for row in rows})
    held_start = scenes[-held_out_episodes].start
    val_start = scenes[-held_out_episodes - validation_episodes].start
    parts = {"train": (times[0], val_start), "validation": (val_start, held_start),
             "held_out": (held_start, times[-1] + timedelta(microseconds=1))}
    scene_part = {}
    for number, scene in enumerate(scenes):
        from_end = len(scenes) - number
        scene_part[scene.episode_id] = ("held_out" if from_end <= held_out_episodes else
                                        "validation" if from_end <= held_out_episodes + validation_episodes else "train")
    first_seen: dict[str, datetime] = {}
    for row in rows:
        stamp = _ts(row["ts"])
        labels = ([t["label"] for t in row.get("tracks") or []] if row["type"] == "perception.observation.recorded"
                  else [o["label"] for o in row.get("objects") or []] if row["type"] == "perception.scene.described"
                  else [row["label"]] if row["type"] == "perception.zone.event" else [])
        for label in labels:
            if label in askable and label not in first_seen:
                first_seen[label] = stamp
    objects = {}
    for label in sorted(first_seen):
        part = ("held_out" if label in held_out_objects else "validation" if label in validation_objects
                else "train")
        if first_seen[label] < parts[part][1]:      # the box saw it before its part's stretch ends
            objects[label] = part
    return Plan(scenes, parts, scene_part, times, first_seen, objects,
                {"episodes": len(scenes), "first_time": times[0].isoformat(), "last_time": times[-1].isoformat(),
                 "parts": {p: [a.isoformat(), b.isoformat()] for p, (a, b) in parts.items()}})


# --- the questions --------------------------------------------------------------------------------------------

def _number(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")


def _clock(real: datetime, seed: str) -> datetime:
    """A logged time as the box prints it (UTC, microseconds), or at -04:00 cut to the minute, by seed."""
    if _number(seed) % 2:
        return real.astimezone(timezone.utc)
    return real.astimezone(EDT).replace(second=0, microsecond=0)


def _candidates(times: Sequence[datetime], low: datetime, high: datetime, seed: str, count: int = 48) -> list[datetime]:
    """Up to `count` logged times in (low, high), in an order fixed by the seed."""
    lo, hi = bisect.bisect_right(times, low), bisect.bisect_left(times, high)
    if hi <= lo:
        return []
    span = hi - lo
    start = _number(seed) % span
    step = max(1, span // count)
    return [times[lo + (start + k * step) % span] for k in range(min(count, span))]


def _whens(part: str, rows: int, seed: str) -> list[tuple[str, dict[str, Any]]]:
    """(wording of the window, its shape) for one scene's rows: each length of the part before any twice."""
    pool: list[tuple[str, dict[str, Any]]] = []
    for minutes in {"train": TRAIN_MINUTES, "validation": VALIDATION_MINUTES, "held_out": HELD_OUT_MINUTES}[part]:
        pool += [(spelling, {"minutes": minutes}) for spelling in v2._spellings(minutes)]
    if part == "train":
        pool += [(spelling, {"named": name}) for name, spellings in v2.NAMED_WINDOWS for spelling in spellings]
    order = sorted(pool, key=lambda item: _number(f"{seed}|{item[0]}"))
    lengths: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for item in order:
        lengths.setdefault(json.dumps(item[1], sort_keys=True), []).append(item)
    keys = sorted(lengths, key=lambda key: _number(f"{seed}|{key}"))
    out = []
    round_ = 0
    while len(out) < rows * OPTIONS_PER_ROW * 2 and round_ < 40:   # more than needed: some do not fit the scene
        for key in keys:
            options = lengths[key]
            out.append(options[round_ % len(options)])
        round_ += 1
    return out


@dataclass(frozen=True)
class Ask:
    part: str
    group: str
    question: str
    shape: dict[str, Any]
    now: datetime
    anchor: str | None          # the scene's episode id for a window question
    want: int                   # rows its group takes: the first options, in this order, that pass the checks


def asks(p: Plan, sleep: SleepSchedule) -> tuple[list[Ask], Counter]:
    """Every option of the set (a question with its clock), in a fixed order; each group takes its first `want`
    options that pass the build's checks. Also how many wordings found no clock that fits their scene."""
    out: list[Ask] = []
    misfit: Counter = Counter()
    for label, part in p.objects.items():
        name, plural = ASKABLE[label]
        group = "last_seen-" + name.replace(" ", "-")
        low, high = p.parts[part]
        templates = v2._pick(v2.LAST_SEEN_TEMPLATES, OPTIONS_PER_ROW * ROWS_PER_OBJECT[part], group)
        for number, template in enumerate(templates):
            question = template.format(item=name, be="are" if plural else "is")
            seed = f"{group}|{number}|{question}"
            since = low if number % NO_EVIDENCE_EVERY == NO_EVIDENCE_EVERY - 1 else max(low, p.first_seen[label])
            picks = _candidates(p.times, since, high, seed, count=1)
            if not picks:
                misfit[part] += 1
                continue
            now = _clock(picks[0], seed)
            out.append(Ask(part, group, question, {"name": "last_seen", "label": name}, now, None,
                           ROWS_PER_OBJECT[part]))
    for scene in p.scenes:
        part = p.scene_part[scene.episode_id]
        group = f"scene-{scene.episode_id}"
        _, high = p.parts[part]
        stems = v2._pick(v2.WINDOW_STEMS, 10_000, group)
        for number, (when, shape) in enumerate(_whens(part, ROWS_PER_SCENE[part], group)):
            question = stems[number].format(when=when)
            seed = f"{group}|{number}|{question}"
            top = min(high, scene.start + timedelta(minutes=shape["minutes"])) if "minutes" in shape else high
            for real in _candidates(p.times, scene.start + MARGIN, top, seed):
                now = _clock(real, seed)
                window = time_window(question.lower(), now, sleep)
                if (window and now - MARGIN >= scene.start and window[0] <= scene.start < window[1]
                        and p.parts[part][0] <= now < high):
                    out.append(Ask(part, group, question, {"name": "episodes_between", **shape}, now, scene.episode_id,
                                   ROWS_PER_SCENE[part]))
                    break
            else:
                misfit[part] += 1
    return out, misfit


def _problems(listed: Sequence[Ask]) -> list[str]:
    """Train and validation hold nothing an earlier dataset held out; no part uses an overnight word."""
    problems = []
    for ask in listed:
        folded = _normal(ask.question)
        if any(word in folded for word in v2.RESERVED["words"]):
            problems.append(f"{ask.question!r} uses a reserved wording")
        if ask.part == "held_out":
            continue
        if ask.shape.get("label") in EARLIER_HELD_OUT["items"] or ask.shape.get("minutes") in EARLIER_HELD_OUT["minutes"]:
            problems.append(f"{ask.question!r} ({ask.part}) asks a thing an earlier dataset holds out")
    return problems


# --- the build ------------------------------------------------------------------------------------------------

def _cut(line: str) -> str:
    if len(line) <= LINE_CHARS:
        return line
    head = line[:LINE_CHARS - 3]
    return (head[:head.rfind(" ")] if " " in head else head).rstrip(" ,.;:") + "..."


def clip(trace: Trace) -> tuple[Trace, bool]:
    """The trace a row shows: each tool result's lines cut to LINE_CHARS, kept while the result stays within
    RESULT_CHARS, the rest counted on one last line; and the answer written from the kept lines the way
    reference_trace() writes it. A trace that is already short comes back unchanged (and False)."""
    results, kept_lines, clipped = [], [], False
    for text in trace.tool_results:
        lines = text.split("\n")
        if len(text) <= RESULT_CHARS and all(len(line) <= LINE_CHARS for line in lines):
            results.append(text)
            kept_lines += lines
            continue
        clipped = True
        kept: list[str] = []
        for line in lines:
            line = _cut(line)
            if kept and len("\n".join(kept + [line])) > RESULT_CHARS:
                break
            kept.append(line)
        rest = len(lines) - len(kept)
        results.append("\n".join(kept + ([f"({rest} more lines)"] if rest else [])))
        kept_lines += kept
    if not clipped:
        return trace, False
    answer = " ".join(_sentence(line) for line in kept_lines) if trace.cited else trace.answer
    return replace(trace, tool_results=tuple(results), answer=answer), True

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(out_dir: str | Path, events: str | Path, *, snapshot: str | Path | None = None,
          dataset_id: str = DATASET_ID, tokenizer: ft.Tokenizer | None = None, reference: ft.Reference | None = None,
          sleep: SleepSchedule | None = None, source_note: dict[str, Any] | None = None,
          teacher: str | Path | None = None, min_held_out: int | None = None, **plan_options: Any) -> Built:
    """Write <out_dir>/<dataset_id>/: v2's files plus source.json (where the scenes came from). With `teacher`, the
    kept lines of lab/fixture_dataset_v3_teacher.py's file join the train side as teacher-labeled rows, and the file
    is copied in as teacher.jsonl."""
    root = Path(out_dir) / dataset_id
    if root.exists() and any(root.iterdir()):
        raise BuildError(f"{root.name} already exists and is not empty; the dataset is frozen once built",
                         outcome="refused")
    sleep = sleep or SleepSchedule.load(AGENT_CONFIG)
    verified = verify_snapshot(events, snapshot) if snapshot else None
    if verified and not all(verified["same"].values()):
        raise BuildError(f"the replay does not land on the box's memory: {verified['same']}")
    rows_in, injected_counts = read_events(events)
    p = plan(rows_in, **plan_options)
    listed, misfit = asks(p, sleep)
    problems = _problems(listed)
    if problems:
        raise BuildError(f"{len(problems)} reserved questions, for example: {problems[0]}")

    results: dict[int, tuple[Any, Any, bool | None]] = {}
    by_now: dict[datetime, list[int]] = {}
    for number, ask in enumerate(listed):
        by_now.setdefault(ask.now, []).append(number)

    def visit(now: datetime, memory: SqliteMemory) -> None:
        for number in by_now[now]:
            ask = listed[number]
            try:
                trace = reference_trace(ask.question, memory, now, sleep)
            except Unsupported as why:
                raise BuildError(f"{ask.question!r} is not answered with a tool: {why}") from None
            shown = None
            if ask.part != "held_out":
                shown = list(ft.tool_loop_results(memory, now, sleep)(
                    ask.question, [ft.as_call(call) for call in trace.tool_calls]))
            inside = None
            if ask.anchor:      # the scene is among the episodes the call's window returns (the evidence may list
                arguments = trace.tool_calls[0]["arguments"]   # only the first MAX_EPISODE_LINES of them)
                inside = ask.anchor in {e.id for e in memory.episodes(_ts(arguments["start"]), _ts(arguments["end"]))}
            results[number] = (trace, shown, inside)

    with tempfile.TemporaryDirectory() as scratch:   # a file, not :memory:, so the tool loop's threads share it
        replay(rows_in, sorted(by_now), visit, path=str(Path(scratch) / "replay.sqlite")).db.close()

    examples: list[DatasetExample] = []
    expected: list[Expected] = []
    parts: dict[str, list[str]] = {part: [] for part in PARTS}
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    tokens: list[int] = []
    seen: set[str] = set()
    dropped: Counter = Counter()
    taken: Counter = Counter()
    clipped_count: Counter = Counter()
    for number, ask in enumerate(listed):
        if taken[ask.group] >= ask.want:
            continue
        trace, shown, inside = results[number]
        v2._check_call(ask.question, ask.shape, trace.tool_calls, ask.now)
        if ask.anchor and not inside:
            dropped["scene not yet in the box's memory at that clock"] += 1
            continue
        example_id = ("o-" if ask.shape["name"] == "last_seen" else "w-") + hashlib.sha1(
            f"{_normal(ask.question)}|{ask.now.isoformat()}".encode("utf-8")).hexdigest()[:10]
        if example_id in seen:
            dropped["same question at the same clock"] += 1
            continue
        if shown is not None and list(shown) != list(trace.tool_results):
            raise BuildError(f"{ask.question!r}: the tool loop shows other results than the reference trace")
        trace, was_clipped = clip(trace)
        label = _label(trace)
        example = DatasetExample(id=example_id, image="", frame_ref=f"box:group:{ask.group}", ts=ask.now,
                                 context=ask.question, target=label.model_dump_json())
        made = ft._trace(example, instruction(ask.now), lambda _q, _c, r=list(trace.tool_results): r, TOOLS)
        row = ft.build_row(made, TOOLS)
        if len(row["text"]) > MAX_ROW_CHARS:
            dropped["row longer than MAX_ROW_CHARS"] += 1
            continue
        seen.add(example_id)
        taken[ask.group] += 1
        clipped_count[ask.part] += was_clipped
        examples.append(example)
        expected.append(Expected(id=example_id, question=ask.question, origin="variant", group=ask.group,
                                 label=label, tool_results=list(trace.tool_results), cited=list(trace.cited),
                                 keywords=[], no_evidence=not trace.cited))
        parts[ask.part].append(example_id)
        if ask.part == "held_out":
            continue
        checked = ft.check_trace(made, TOOLS, tokenizer, reference)
        if checked.problems:
            raise BuildError(f"example {example_id}: its row does not check out: {'; '.join(checked.problems)}")
        if checked.tokens is not None:
            tokens.append(checked.tokens)
        rows[ask.part].append(row)

    teacher_ids: list[str] = []
    teacher_outcomes: Counter = Counter()
    teacher_lines = [json.loads(line) for line in Path(teacher).read_text(encoding="utf-8").splitlines()
                     if line.strip()] if teacher else []
    for line in teacher_lines:
        teacher_outcomes[line["outcome"]] += 1
        if line["outcome"] != "kept" or not line["group"].startswith("teacher-") or line["id"] in seen:
            continue
        now = _ts(line["ts"])
        target = {"tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in line["tool_calls"]],
                  "answer": line["answer"]}
        example = DatasetExample(id=line["id"], image="", frame_ref=f"box:group:{line['group']}", ts=now,
                                 context=line["question"], target=json.dumps(target, ensure_ascii=False))
        shown = [c["result"] for c in line["tool_calls"]]
        made = ft._trace(example, instruction(now), lambda _q, _c, r=shown: r, TOOLS)
        checked = ft.check_trace(made, TOOLS, tokenizer, reference)
        row = ft.build_row(made, TOOLS)
        if checked.problems or len(row["text"]) > MAX_ROW_CHARS:
            teacher_outcomes["kept but the row does not check out"] += 1
            continue
        if checked.tokens is not None:
            tokens.append(checked.tokens)
        seen.add(line["id"])
        examples.append(example)
        expected.append(Expected(id=line["id"], question=line["question"], origin="variant", group=line["group"],
                                 label=ExpectedLabel.model_validate(target), tool_results=shown, cited=[],
                                 keywords=[], no_evidence=False))
        parts["train"].append(line["id"])
        rows["train"].append(row)
        teacher_ids.append(line["id"])
    if teacher:
        root.mkdir(parents=True, exist_ok=True)
        _write(root / TEACHER, "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in teacher_lines))

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        spec=TaskSpec(task=TASK, instruction=f"{SYSTEM} The time now is {v2.INSTRUCTION_NOW}. Use tools to find evidence.",
                      output_schema=output_schema()),
        created=p.times[-1], examples=len(examples), action_id=ACTION_ID)
    (root / IMAGES).mkdir(parents=True, exist_ok=True)
    _write(root / IMAGES / ".gitkeep", "")
    _write(root / MANIFEST, json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n")
    _write(root / EXAMPLES, "".join(example.model_dump_json() + "\n" for example in examples))
    _write(root / EXPECTED, "".join(row.model_dump_json() + "\n" for row in expected))
    _write(root / SPLIT, json.dumps({
        "version": 1, "dataset_id": dataset_id,
        "rule": "By whole group and by time, fixed in lab/fixture_dataset_v3.py before any training. A group is one "
                "object the box saw (last_seen-<object>) or one real episode of the box (scene-<episode id>), "
                "whatever the wording and the clock; frame_ref is box:group:<group>. Held out: the last "
                "HELD_OUT_EPISODES episodes as scenes, the objects of HELD_OUT_OBJECTS, the window lengths of "
                "HELD_OUT_MINUTES, all asked at clocks after the first held-out episode starts. Train: every other "
                "group; of these, the VALIDATION_EPISODES episodes before the held-out ones and the objects of "
                "VALIDATION_OBJECTS are listed in validation.json, get no training row, and are asked at clocks "
                "between the first validation episode's start and the first held-out one's. Train groups are asked "
                "before the first validation episode starts. Train and validation hold no held-out object or window "
                "of tool-calling-fixture or tool-calling-v2.",
        "train": parts["train"] + parts["validation"], "held_out": parts["held_out"]}, indent=2) + "\n")
    _write(root / v2.VALIDATION, json.dumps({
        "version": 1, "dataset_id": dataset_id,
        "rule": "Whole groups of the split's train side that no training row shows; a sweep chooses by these and "
                "never by the held-out side.",
        "train": parts["train"], "validation": parts["validation"]}, indent=2) + "\n")
    written = {v2.ROWS_TRAIN: len(rows["train"]), v2.ROWS_VALIDATION: len(rows["validation"])}
    ft._write_rows(root / v2.ROWS_TRAIN, rows["train"])
    ft._write_rows(root / v2.ROWS_VALIDATION, rows["validation"])
    split = load_split(root, require_images=False,      # a test's small log may hold out fewer than a gate needs
                       **({} if min_held_out is None else {"min_held_out": min_held_out}))
    texts = {row["id"]: row["text"] for part in rows.values() for row in part}
    for example in split.data.examples:
        if example.id in texts and instruction(example.ts) not in texts[example.id]:
            raise BuildError(f"example {example.id}: its ts does not read back as the time its row was written with")

    by_id = {row.id: row for row in expected}
    counts = {part: {"examples": len(ids), "groups": len({by_id[i].group for i in ids})} for part, ids in parts.items()}
    kinds = {part: dict(Counter(by_id[i].group.split("-")[0] for i in ids)) for part, ids in parts.items()}
    source = {
        "dataset_id": dataset_id, "from": "the Atlas box (Jetson Orin Nano), data/events and data/memory.sqlite",
        "events_sha256": _file_sha256(events), "snapshot_sha256": _file_sha256(snapshot) if snapshot else None,
        "replayed_events": len(rows_in), "replayed_types": dict(Counter(row["type"] for row in rows_in)),
        "injected_dropped": dict(injected_counts), "replay_matches_box_memory": verified,
        "span": p.facts, "objects": {label: [part, p.first_seen[label].isoformat()] for label, part in p.objects.items()},
        "scenes": {part: sum(1 for v in p.scene_part.values() if v == part) for part in PARTS},
        "labels": "reference: lab/fixture_dataset.py reference_trace() against the replayed memory at each clock",
        "teacher_labeled": len(teacher_ids), "teacher_ids": teacher_ids,
        "teacher": "pod C, model name teacher, by lab/fixture_dataset_v3_teacher.py" if teacher else None,
        **(source_note or {})}
    _write(root / SOURCE, json.dumps(source, indent=2) + "\n")
    clocks = Counter("box (UTC, microseconds)" if example.ts.utcoffset() == timedelta(0)
                     else "fixture (-04:00, whole minute)" for example in examples)
    files = (MANIFEST, EXAMPLES, EXPECTED, SPLIT, v2.VALIDATION, v2.ROWS_TRAIN, v2.ROWS_VALIDATION, SOURCE) + (
        (TEACHER,) if teacher else ())
    digests = {name: _sha256(root / name) for name in files}
    facts = {"dataset_id": dataset_id, "examples": len(examples), "groups": len({row.group for row in expected}),
             "parts": counts, "group_kinds": kinds, "rows": written, "clocks": dict(clocks),
             "distinct_clocks": len({example.ts.isoformat() for example in examples}),
             "distinct_questions": len({_normal(example.context or "") for example in examples}),
             "tools": dict(Counter(row.label.tool_calls[0].name for row in expected)),
             "no_evidence": {part: sum(by_id[i].no_evidence for i in ids) for part, ids in parts.items()},
             "rule_labeled": len(examples) - len(teacher_ids), "teacher_labeled": len(teacher_ids),
             "teacher_outcomes": dict(teacher_outcomes),
             "clipped": {"rows": dict(clipped_count), "line_chars": LINE_CHARS, "result_chars": RESULT_CHARS},
             "misfit_wordings": dict(misfit), "dropped": dict(dropped),
             "short_groups": {group: [taken[group], want] for group, want in sorted(
                 {ask.group: ask.want for ask in listed}.items()) if taken[group] < want},
             "max_row_chars": max((len(row["text"]) for part in rows.values() for row in part), default=0),
             "tokenizer": "checked" if tokenizer else "not checked",
             "reference": "checked" if reference else "not checked",
             "tokens": {"total": sum(tokens), "max": max(tokens), "min": min(tokens)} if tokens else None,
             "template": f"{ft.TEMPLATE_REPO}@{ft.TEMPLATE_REVISION}", "sha256": digests}
    _write(root / v2.BUILD, json.dumps(facts, indent=2) + "\n")
    return Built(root, len(examples), facts["groups"], counts, written, digests, facts["tokenizer"], dict(dropped))


# --- the entry point ------------------------------------------------------------------------------------------

@recorded("fixture-dataset-v3")
@stop.guarded
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Build tool-calling dataset v3 from the box's own event log (reference labels, real scenes and clocks)."""
    parser = argparse.ArgumentParser(prog="python lab/fixture_dataset_v3.py", description=main.__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--events", required=True, help="the box's events (JSONL, or .gz): the replayed types at least")
    parser.add_argument("--snapshot", help="the box's memory.sqlite at the time of the copy: the replay must match it")
    parser.add_argument("--out", required=True, help="the folder that gets <dataset id>/")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--tokenizer", help="a folder with tokenizer.json (and chat_template.jinja): check every row")
    parser.add_argument("--teacher", help="lab/fixture_dataset_v3_teacher.py's file: its kept lines become rows")
    args = parser.parse_args(argv)
    try:
        tokenizer, reference = ft.load_tokenizer(args.tokenizer) if args.tokenizer else (None, None)
        built = build(args.out, args.events, snapshot=args.snapshot, dataset_id=args.dataset_id,
                      tokenizer=tokenizer, reference=reference, teacher=args.teacher)
    except BuildError as err:
        return {"ok": False, "outcome": err.outcome, "error": str(err)}
    except ft.FormatError as err:
        return {"ok": False, "outcome": "refused", "error": str(err)}
    return {"ok": True, "outcome": "built", "dataset_id": args.dataset_id, "path": str(built.root),
            "examples": built.examples, "groups": built.groups, "parts": built.parts, "rows": built.rows,
            "tokenizer": built.tokenizer, "dropped": built.dropped, "sha256": built.sha256,
            "refs": {"dataset_id": args.dataset_id, "split_sha256": built.sha256[SPLIT],
                     "rows_sha256": built.sha256[v2.ROWS_TRAIN]}}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
