"""Tool-calling dataset v2: lab/fixture_dataset.py's approach, widened, with reference labels as targets.

    python lab/fixture_dataset_v2.py --out <folder>                       # writes <folder>/tool-calling-v2/
    python lab/fixture_dataset_v2.py --out <folder> --tokenizer ~/models/Qwen3.5-4B   # and checks every row with it

Why it exists. The first fine-tune (L16, run train-20261011T032149Z-7dd098) trained on 84 rows that reword 12
questions, all at one clock time, with teacher labels. It memorized them and broke a held-out pattern ("in the last
60 minutes": wrong start time). This dataset keeps the fixture's method and changes what the method is fed:

    reference labels   every target is what the agent's own pre-fetch does for the question (reference_trace() of
                       lab/fixture_dataset.py, unchanged): the tool call and the answer written from its evidence.
                       No teacher is asked. The answer is the evidence written out, not natural speech.
    many objects       "where did I leave my X" for ITEMS (one group per object), asked LAST_SEEN_TEMPLATES ways.
                       Half the objects have a sighting in the night (SEEN_POSITIONS), so last_seen also returns
                       something; the fixture's memory had seen only the keys.
    many windows       "what happened in the last N" for WINDOW_MINUTES (one group per length), plus today and
                       yesterday, asked WINDOW_STEMS ways with several spellings of each length.
    a clock per row    the agent's tool loop writes "The time now is <now>" into the system prompt, so each example
                       has its own now, stored as its `ts`: half in the fixture's style (a whole minute, -04:00), half
                       as the box's clock really prints it (contexts/gateway/bootstrap.py: datetime.now(UTC), so
                       seconds, microseconds and +00:00). The expected start and end follow from that now.

A group is one thing asked (an object, a window length): frame_ref is "fixture:group:<group>", whatever the wording
and the clock. The split is by whole group and is written by the build, frozen, before anything trains:

    held_out     HELD_OUT_ITEMS and HELD_OUT_WINDOWS: objects and window lengths no other part shows
    validation   VALIDATION_ITEMS and VALIDATION_WINDOWS: whole groups on the split's train side that the trainer
                 never gets rows for (validation.json lists them); what the sweep chooses by
    train        every other group

Nothing of the first fixture's held-out side is in any part: no question about the keys or the wallet, no 60 minute
or one hour window, no overnight question (RESERVED). So tool-call agreement on that fixture's 58 held-out examples
stays a test of questions this dataset never trained on. The cost is that the overnight shape (its start is the
household's bedtime, which no prompt states) is not taught here.

The folder: manifest.json (action_id "fixture:tool-calling-v2"; spec.instruction holds the prompt with the time left
as INSTRUCTION_NOW, since each example has its own), examples.jsonl (targets filled), expected.jsonl, images/,
split.json, validation.json, rows-train.jsonl and rows-validation.jsonl (lab/format_traces.py rows: each trace is
checked exactly as format_dataset() checks one, with the tokenizer when one is given), and build.json (the counts
and the sha256 of every file). The build reads no clock and no random source: the same commit gives the same bytes.

The entry point writes a ledger row (verb fixture-dataset-v2) and refuses to run while lab/state/STOP exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds contracts/, contexts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import SYSTEM, TOOLS, SleepSchedule
from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
from contracts.memory import Episode, Item, Sighting
from lab import format_traces as ft
from lab.dataset import EXAMPLES, IMAGES, MANIFEST, SPLIT, load_split
from lab.fixture_dataset import (AGENT_CONFIG, EXPECTED, GOLDEN_DIR, TASK, BuildError, Expected, Unsupported,
                                 _label, _normal, _write, instruction, output_schema, reference_trace)
from lab.machine import stop
from lab.machine.record import recorded

__all__ = ["DATASET_ID", "ACTION_ID", "VALIDATION", "ROWS_TRAIN", "ROWS_VALIDATION", "BUILD", "INSTRUCTION_NOW",
           "ITEMS", "HELD_OUT_ITEMS", "VALIDATION_ITEMS", "WINDOW_MINUTES", "HELD_OUT_WINDOWS", "VALIDATION_WINDOWS",
           "RESERVED", "LAST_SEEN_TEMPLATES", "WINDOW_STEMS", "Built", "night_memory", "clock_for", "system_for",
           "questions", "build", "load_validation", "main"]

DATASET_ID = "tool-calling-v2"
ACTION_ID = "fixture:tool-calling-v2"
VALIDATION = "validation.json"
ROWS_TRAIN = "rows-train.jsonl"
ROWS_VALIDATION = "rows-validation.jsonl"
BUILD = "build.json"
INSTRUCTION_NOW = "<the example's ts>"   # where spec.instruction leaves the time: system_for() writes each example's
OLD_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "learning" / "tool-calling-fixture"

# (name, plural). One group per object. Every other object (an even index) has a sighting in the night.
ITEMS = (
    ("phone", False), ("glasses", True), ("passport", False), ("umbrella", False), ("laptop", False),
    ("laptop charger", False), ("earbuds", True), ("watch", False), ("ring", False), ("jacket", False),
    ("hat", False), ("scarf", False), ("gloves", True), ("shoes", True), ("bag", False), ("purse", False),
    ("lunch box", False), ("coffee mug", False), ("thermos", False), ("remote", False), ("book", False),
    ("textbook", False), ("calculator", False), ("pen", False), ("pencil case", False), ("stapler", False),
    ("scissors", True), ("flash drive", False), ("hard drive", False), ("mouse", False), ("keyboard", False),
    ("camera", False), ("tripod", False), ("student id", False), ("credit card", False), ("bus pass", False),
    ("parking pass", False), ("tickets", True), ("mail", False), ("package", False), ("envelope", False),
    ("medication", False), ("vitamins", True), ("inhaler", False), ("toothbrush", False), ("towel", False),
    ("blanket", False), ("hoodie", False), ("sweater", False), ("coat", False), ("boots", True), ("socks", True),
    ("belt", False), ("necklace", False), ("bracelet", False), ("earrings", True), ("guitar pick", False),
    ("screwdriver", False), ("tape measure", False), ("flashlight", False), ("power bank", False),
    ("game controller", False), ("yoga mat", False), ("gym bag", False), ("lanyard", False), ("checkbook", False),
    # held out and validation objects follow; the lists below say which is which
    ("backpack", False), ("headphones", True), ("library card", False), ("bike helmet", False),
    ("sunglasses", True), ("tablet", False), ("notebook", False), ("water bottle", False), ("badge", False),
    ("slippers", True),
    ("stylus", False), ("lip balm", False), ("mittens", True), ("planner", False), ("usb cable", False),
    ("retainer", False), ("binoculars", True), ("cardigan", False),
)
HELD_OUT_ITEMS = ("backpack", "headphones", "library card", "bike helmet", "sunglasses", "tablet", "notebook",
                  "water bottle", "badge", "slippers")
VALIDATION_ITEMS = ("stylus", "lip balm", "mittens", "planner", "usb cable", "retainer", "binoculars", "cardigan")
# Window lengths in minutes. One group per length; a whole number of hours is also asked in hours.
WINDOW_MINUTES = (5, 10, 15, 20, 30, 35, 40, 50, 55, 70, 80, 90, 100, 110, 120, 150, 180, 240, 300, 360, 480, 600,
                  720, 1080, 1440,
                  25, 45, 75, 105, 420, 540, 840, 2160,       # held out
                  65, 85, 130, 660, 1200)                     # validation
HELD_OUT_WINDOWS = (25, 45, 75, 105, 420, 540, 840, 2160)
VALIDATION_WINDOWS = (65, 85, 130, 660, 1200)
NAMED_WINDOWS = (("today", ("today", "so far today", "earlier today")),
                 ("yesterday", ("yesterday", "all day yesterday", "during the day yesterday")))   # both train
# The first fixture's held-out side: its objects, its window and its overnight wording never appear here.
RESERVED = {"items": ("keys", "wallet"), "minutes": (60,),
            "words": ("overnight", "last night", "last hour", "past hour", "asleep", "sleeping", "recently")}

LAST_SEEN_TEMPLATES = (
    "Where did I leave my {item}?", "Where did I put my {item}?", "Where {be} my {item}?",
    "Where {be} my {item} right now?", "Where did I leave my {item} again?", "Where {be} the {item}?",
    "Have you seen my {item}?", "I can't find my {item}.", "Where'd I put my {item}?", "Where'd I leave my {item}?",
    "Where did you last see my {item}?", "Where was the last place you saw my {item}?", "Have you seen the {item}?",
    "I cannot find my {item}", "Where {be} my {item} at the moment?", "Where did I last put my {item}?",
    "where {be} our {item} now", "Where did I last leave my {item}?")
WINDOW_STEMS = (
    "What happened {when}?", "What happened at our table {when}?", "Tell me what the cameras saw {when}.",
    "Was there any activity {when}?", "Did anyone come to the door {when}?", "Was anyone at the desk {when}?",
    "Who came to the door {when}?", "Any visitors at the door {when}?", "Did someone come by {when}?",
    "What did the cameras pick up {when}?", "Anything happen at the table {when}?", "Was anybody around {when}?",
    "Show me what went on {when}.", "Did anything move at the desk {when}?", "Has anyone been at the table {when}?",
    "Was there any movement {when}?", "what went on {when}", "Was the door used {when}?")
MINUTE_SPELLINGS = ("in the last {n} minutes", "in the past {n} minutes", "over the last {n} minutes",
                    "in the last {n} mins", "during the past {n} minutes", "within the last {n} minutes")
HOUR_SPELLINGS = ("in the last {n} hours", "in the past {n} hours", "over the last {n} hours",
                  "during the last {n} hours", "within the past {n} hours")
ROWS_PER_ITEM = {"train": 7, "validation": 7, "held_out": 10}
ROWS_PER_WINDOW = {"train": 14, "validation": 14, "held_out": 16}
ROWS_PER_NAMED = 14

# Clocks. Every now falls in the fixture night and the morning after it, so windows find the night's episodes.
CLOCK_START = datetime(2026, 10, 10, 19, 0, tzinfo=timezone(timedelta(hours=-4)))
CLOCK_SPAN_MINUTES = 19 * 60      # to 2:00 PM on the 11th
SIGHTINGS_START = datetime(2026, 10, 10, 16, 0, tzinfo=timezone(timedelta(hours=-4)))   # all before the first clock
SEEN_POSITIONS = (("cam-desk", "desk", "next to the laptop"), ("cam-desk", "desk", "on the left side of the table"),
                  ("cam-door", None, "on the shelf by the door"), ("cam-desk", "desk", "under the monitor"),
                  ("cam-desk", "desk", "on the chair"), ("cam-door", None, "on the floor by the door mat"),
                  ("cam-desk", "desk", "behind the coffee mug"), ("cam-desk", "desk", None),
                  ("cam-door", None, "hanging on the hook"), ("cam-desk", "desk", "at the edge of the table"))


@dataclass(frozen=True)
class Built:
    root: Path
    examples: int
    groups: int
    parts: dict[str, dict[str, int]]     # part -> {"examples", "groups"}
    rows: dict[str, int]                 # rows file -> rows written
    sha256: dict[str, str]               # file -> sha256
    tokenizer: str                       # "checked" or "not checked"


# --- the night, and each example's clock ------------------------------------------------------------------------

def night_memory(golden_dir: str | Path = GOLDEN_DIR) -> tuple[FakeMemory, datetime]:
    """The fixture night (fixtures/golden/memory.yaml, read as contexts/agent/eval.py load_fixture reads it) with one
    sighting added for every other object of ITEMS; and the fixture's own now."""
    try:
        raw = yaml.safe_load((Path(golden_dir) / "memory.yaml").read_text(encoding="utf-8"))
        sightings = [Sighting.model_validate(s) for s in raw["sightings"]]
        for index, (name, _) in enumerate(ITEMS):
            if index % 2 == 0:
                camera, zone, position = SEEN_POSITIONS[(index // 2) % len(SEEN_POSITIONS)]
                sightings.append(Sighting(label=name, camera_id=camera, zone_id=zone, position=position,
                                          ts=SIGHTINGS_START + timedelta(minutes=2 * index),
                                          frame_ref=f"frames:{camera}:s{index:03d}"))
        memory = FakeMemory(episodes=[Episode.model_validate(e) for e in raw["episodes"]], sightings=sightings,
                            items=[Item.model_validate(i) for i in raw.get("items", [])])
        return memory, datetime.fromisoformat(raw["now"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BuildError(f"cannot read the fixture night in {Path(golden_dir).name}: {type(exc).__name__}") from None


def _number(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")


def clock_for(seed: str) -> datetime:
    """The now of one example, fixed by its seed. Even draws are the fixture's style (a whole minute at -04:00);
    odd draws are the box's (UTC with seconds and microseconds, as datetime.now(UTC).isoformat() prints)."""
    draw = _number(seed)
    local = CLOCK_START + timedelta(minutes=(draw >> 8) % CLOCK_SPAN_MINUTES)
    if draw % 2 == 0:
        return local
    moved = local + timedelta(seconds=(draw >> 24) % 60, microseconds=(draw >> 32) % 1_000_000)
    return moved.astimezone(timezone.utc)


def system_for(manifest: DatasetManifest, example: DatasetExample) -> str:
    """The system prompt of one example: this dataset writes the example's own ts as the time now; any other
    dataset has one instruction for all its examples."""
    return instruction(example.ts) if manifest.action_id == ACTION_ID else manifest.spec.instruction


# --- the questions ------------------------------------------------------------------------------------------------

def _part_of_item(name: str) -> str:
    return "held_out" if name in HELD_OUT_ITEMS else "validation" if name in VALIDATION_ITEMS else "train"


def _part_of_window(minutes: int) -> str:
    return "held_out" if minutes in HELD_OUT_WINDOWS else "validation" if minutes in VALIDATION_WINDOWS else "train"


def _spellings(minutes: int) -> list[str]:
    out = [spelling.format(n=minutes) for spelling in MINUTE_SPELLINGS] if minutes < 600 else []
    if minutes % 60 == 0:
        out += [spelling.format(n=minutes // 60) for spelling in HOUR_SPELLINGS]
    return out


def _pick(options: Sequence[str], count: int, seed: str) -> list[str]:
    """`count` of the options in an order fixed by the seed; each is used once before any is used twice."""
    order = sorted(options, key=lambda option: _number(f"{seed}|{option}"))
    return [order[index % len(order)] for index in range(count)]


def questions() -> list[tuple[str, str, str, dict[str, Any]]]:
    """Every (part, group, question, expected call shape) of the dataset, in a fixed order. The call shape is
    {"name": "last_seen", "label": ...} or {"name": "episodes_between", "minutes": ... or "named": ...}."""
    out: list[tuple[str, str, str, dict[str, Any]]] = []
    for name, plural in ITEMS:
        part, group = _part_of_item(name), "last_seen-" + name.replace(" ", "-")
        for template in _pick(LAST_SEEN_TEMPLATES, ROWS_PER_ITEM[part], group):
            out.append((part, group, template.format(item=name, be="are" if plural else "is"),
                        {"name": "last_seen", "label": name}))
    for minutes in WINDOW_MINUTES:
        part, group = _part_of_window(minutes), f"episodes_between-{minutes}m"
        whens = [stem.format(when=when) for stem in WINDOW_STEMS for when in _spellings(minutes)]
        for question in _pick(whens, ROWS_PER_WINDOW[part], group):
            out.append((part, group, question, {"name": "episodes_between", "minutes": minutes}))
    for named, spellings in NAMED_WINDOWS:
        group = f"episodes_between-{named}"
        whens = [stem.format(when=when) for stem in WINDOW_STEMS for when in spellings]
        for question in _pick(whens, ROWS_PER_NAMED, group):
            out.append(("train", group, question, {"name": "episodes_between", "named": named}))
    return out


def _check_call(question: str, shape: dict[str, Any], calls: Sequence[dict[str, Any]], now: datetime) -> None:
    """The agent must answer the question with the one call its group stands for, or the build fails."""
    if len(calls) != 1 or calls[0]["name"] != shape["name"]:
        raise BuildError(f"{question!r} is answered with {[c['name'] for c in calls]}, not one {shape['name']}")
    arguments = calls[0]["arguments"]
    if shape["name"] == "last_seen" and arguments != {"label": shape["label"]}:
        raise BuildError(f"{question!r} asks last_seen for {arguments}, not {shape['label']!r}")
    if "minutes" in shape:
        want = {"start": (now - timedelta(minutes=shape["minutes"])).isoformat(), "end": now.isoformat()}
        if arguments != want:
            raise BuildError(f"{question!r} asks for {arguments}, not the last {shape['minutes']} minutes of {now}")


def _reserved_problems(rows: Sequence[tuple[str, str, str, dict[str, Any]]]) -> list[str]:
    """Questions that belong to the first fixture's held-out side: its wordings (when its folder is beside this
    one), its objects, its window length, its overnight words."""
    problems = []
    old: set[str] = set()
    if (OLD_FIXTURE / SPLIT).is_file() and (OLD_FIXTURE / EXAMPLES).is_file():
        split = json.loads((OLD_FIXTURE / SPLIT).read_text(encoding="utf-8"))
        held = set(split.get("held_out", []))
        for line in (OLD_FIXTURE / EXAMPLES).read_text(encoding="utf-8").split("\n"):
            if line.strip() and (row := json.loads(line))["id"] in held:
                old.add(_normal(row["context"]))
    for _, _, question, shape in rows:
        folded = _normal(question)
        if folded in old:
            problems.append(f"{question!r} is a held-out question of the first fixture")
        if shape.get("label") in RESERVED["items"] or shape.get("minutes") in RESERVED["minutes"]:
            problems.append(f"{question!r} asks a reserved thing")
        if any(word in folded for word in RESERVED["words"]):
            problems.append(f"{question!r} uses a reserved wording")
    return problems


# --- the build ------------------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out_dir: str | Path, *, golden_dir: str | Path = GOLDEN_DIR, dataset_id: str = DATASET_ID,
          tokenizer: ft.Tokenizer | None = None, reference: ft.Reference | None = None,
          sleep: SleepSchedule | None = None) -> Built:
    """Write <out_dir>/<dataset_id>/ (see the module docstring). BuildError when a question is not answered with
    its group's call, when a reserved question got in, when a row does not check out, or when the folder exists."""
    root = Path(out_dir) / dataset_id
    if root.exists() and any(root.iterdir()):
        raise BuildError(f"{root.name} already exists and is not empty; the dataset is frozen once built",
                         outcome="refused")
    memory, fixture_now = night_memory(golden_dir)
    sleep = sleep or SleepSchedule.load(AGENT_CONFIG)
    listed = questions()
    problems = _reserved_problems(listed)
    if problems:
        raise BuildError(f"{len(problems)} reserved questions, for example: {problems[0]}")

    examples: list[DatasetExample] = []
    expected: list[Expected] = []
    parts: dict[str, list[str]] = {"train": [], "validation": [], "held_out": []}
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    tokens: list[int] = []
    seen: set[str] = set()
    for number, (part, group, question, shape) in enumerate(listed):
        now = clock_for(f"{group}|{number}|{question}")
        try:
            trace = reference_trace(question, memory, now, sleep)
        except Unsupported as why:
            raise BuildError(f"{question!r} is not answered with a tool: {why}") from None
        _check_call(question, shape, trace.tool_calls, now)
        example_id = ("o-" if shape["name"] == "last_seen" else "w-") + hashlib.sha1(
            f"{_normal(question)}|{now.isoformat()}".encode("utf-8")).hexdigest()[:10]
        if example_id in seen:
            raise BuildError(f"two examples got the id {example_id}")
        seen.add(example_id)
        label = _label(trace)
        example = DatasetExample(id=example_id, image="", frame_ref=f"fixture:group:{group}", ts=now,
                                 context=question, target=label.model_dump_json())
        examples.append(example)
        expected.append(Expected(id=example_id, question=question, origin="variant", group=group, label=label,
                                 tool_results=list(trace.tool_results), cited=list(trace.cited), keywords=[],
                                 no_evidence=not trace.cited))
        parts[part].append(example_id)
        if part == "held_out":
            continue    # a held-out example never gets a row
        made = ft._trace(example, instruction(now), ft.tool_loop_results(memory, now, sleep), TOOLS)
        if list(made.rounds[0].results) != list(trace.tool_results):
            raise BuildError(f"example {example_id}: the tool loop shows other results than the reference trace")
        checked = ft.check_trace(made, TOOLS, tokenizer, reference)
        if checked.problems:
            raise BuildError(f"example {example_id}: its row does not check out: {'; '.join(checked.problems)}")
        if checked.tokens is not None:
            tokens.append(checked.tokens)
        rows[part].append(ft.build_row(made, TOOLS))

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        spec=TaskSpec(task=TASK, instruction=f"{SYSTEM} The time now is {INSTRUCTION_NOW}. Use tools to find evidence.",
                      output_schema=output_schema()),
        created=fixture_now, examples=len(examples), action_id=ACTION_ID)
    (root / IMAGES).mkdir(parents=True, exist_ok=True)
    _write(root / IMAGES / ".gitkeep", "")
    _write(root / MANIFEST, json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n")
    _write(root / EXAMPLES, "".join(example.model_dump_json() + "\n" for example in examples))
    _write(root / EXPECTED, "".join(row.model_dump_json() + "\n" for row in expected))
    _write(root / SPLIT, json.dumps({
        "version": 1, "dataset_id": dataset_id,
        "rule": "By whole group, fixed in lab/fixture_dataset_v2.py before any training. Held out: the objects of "
                "HELD_OUT_ITEMS and the window lengths of HELD_OUT_WINDOWS, which no train or validation example "
                "shows. Train: every other group; of these, the groups of VALIDATION_ITEMS and VALIDATION_WINDOWS "
                "are listed in validation.json and get no training row. A group is one object or one window length, "
                "whatever the wording and the clock, and frame_ref is fixture:group:<group>. No part holds a "
                "held-out question, object or window of tool-calling-fixture.",
        "train": parts["train"] + parts["validation"], "held_out": parts["held_out"]}, indent=2) + "\n")
    _write(root / VALIDATION, json.dumps({
        "version": 1, "dataset_id": dataset_id,
        "rule": "Whole groups of the split's train side that no training row shows; the sweep chooses by these and "
                "never by the held-out side.",
        "train": parts["train"], "validation": parts["validation"]}, indent=2) + "\n")
    written = {ROWS_TRAIN: len(rows["train"]), ROWS_VALIDATION: len(rows["validation"])}
    ft._write_rows(root / ROWS_TRAIN, rows["train"])
    ft._write_rows(root / ROWS_VALIDATION, rows["validation"])
    split = load_split(root, require_images=False)   # the split must hold up exactly as the trainer will read it
    for example in split.data.examples:              # and each ts must come back printing the same time now
        if instruction(example.ts) not in next(
                (row["text"] for part in rows.values() for row in part if row["id"] == example.id),
                instruction(example.ts)):
            raise BuildError(f"example {example.id}: its ts does not read back as the time its row was written with")

    by_id = {row.id: row for row in expected}
    counts = {part: {"examples": len(ids), "groups": len({by_id[i].group for i in ids})} for part, ids in parts.items()}
    clocks = Counter("box (UTC, microseconds)" if example.ts.utcoffset() == timedelta(0)
                     else "fixture (-04:00, whole minute)" for example in examples)
    files = (MANIFEST, EXAMPLES, EXPECTED, SPLIT, VALIDATION, ROWS_TRAIN, ROWS_VALIDATION)
    digests = {name: _sha256(root / name) for name in files}
    facts = {"dataset_id": dataset_id, "examples": len(examples), "groups": len({row.group for row in expected}),
             "parts": counts, "rows": written, "clocks": dict(clocks),
             "distinct_clocks": len({example.ts.isoformat() for example in examples}),
             "distinct_questions": len({_normal(example.context or "") for example in examples}),
             "tools": dict(Counter(row.label.tool_calls[0].name for row in expected)),
             "no_evidence": {part: sum(by_id[i].no_evidence for i in ids) for part, ids in parts.items()},
             "tokenizer": "checked" if tokenizer else "not checked",
             "reference": "checked" if reference else "not checked",
             "tokens": {"total": sum(tokens), "max": max(tokens), "min": min(tokens)} if tokens else None,
             "template": f"{ft.TEMPLATE_REPO}@{ft.TEMPLATE_REVISION}", "sha256": digests}
    _write(root / BUILD, json.dumps(facts, indent=2) + "\n")
    return Built(root, len(examples), facts["groups"], counts, written, digests, facts["tokenizer"])


def load_validation(root: str | Path) -> dict[str, list[str]]:
    """validation.json of a built folder: {"train": ids that get training rows, "validation": ids that do not}."""
    listed = json.loads((Path(root) / VALIDATION).read_text(encoding="utf-8"))
    return {"train": list(listed["train"]), "validation": list(listed["validation"])}


# --- the entry point ----------------------------------------------------------------------------------------------

@recorded("fixture-dataset-v2")
@stop.guarded
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Build tool-calling dataset v2 (reference labels, many objects and windows, a clock per example)."""
    parser = argparse.ArgumentParser(prog="python lab/fixture_dataset_v2.py", description=main.__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--out", required=True, help="the folder that gets <dataset id>/")
    parser.add_argument("--golden", default=str(GOLDEN_DIR), help="memory.yaml of the fixture night")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--tokenizer", help="a folder with tokenizer.json (and chat_template.jinja): encode and decode "
                                            "every row, and compare with the template's own rendering")
    args = parser.parse_args(argv)
    try:
        tokenizer, reference = ft.load_tokenizer(args.tokenizer) if args.tokenizer else (None, None)
        built = build(args.out, golden_dir=args.golden, dataset_id=args.dataset_id, tokenizer=tokenizer,
                      reference=reference)
    except BuildError as err:
        return {"ok": False, "outcome": err.outcome, "error": str(err)}
    except ft.FormatError as err:
        return {"ok": False, "outcome": "refused", "error": str(err)}
    return {"ok": True, "outcome": "built", "dataset_id": args.dataset_id, "path": str(built.root),
            "examples": built.examples, "groups": built.groups, "parts": built.parts, "rows": built.rows,
            "tokenizer": built.tokenizer, "sha256": built.sha256,
            "refs": {"dataset_id": args.dataset_id, "split_sha256": built.sha256[SPLIT],
                     "rows_sha256": built.sha256[ROWS_TRAIN]}}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
