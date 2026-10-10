"""A tool-calling dataset with no cameras, built from the golden questions and the fixture night.

    python lab/fixture_dataset.py                 # writes fixtures/learning/tool-calling-fixture/
    python lab/fixture_dataset.py --out exports --no-variants

The folder has the box's export layout (contracts/learning.py), so lab/dataset.py reads it like a real one:

    manifest.json    DatasetManifest. task "tool-calling"; instruction = the agent's own tool-loop system prompt
                     at the fixture's "now"; output_schema = the JSON schema a label (`target`) must satisfy;
                     action_id "fixture:golden-questions" (nobody approved this: it is fixture data, not an export)
    examples.jsonl   DatasetExample per question: id, image "" (text-only), frame_ref "fixture:group:<group>",
                     ts = the fixture's "now", context = the question, target empty for the teacher to fill
    images/          empty (a .gitkeep), as there are no pictures
    expected.jsonl   one Expected per example: the tool calls and answer a correct label holds, what each call
                     returns, what the evidence cites. The teacher never sees it; it is for checking labels.

A label is the JSON text {"tool_calls": [{"name": ..., "arguments": {...}}], "answer": "..."}. Its schema is built
from the agent's own TOOLS (contexts/agent/application/qa.py), so a call to a tool that does not exist, a missing
argument or an invented one fails validation.

Where the expected values come from. The agent already knows how to answer these questions: QA.prefetch() turns
the common shapes into a memory lookup (or two) and answers from it. reference_trace() runs that real code against
fixtures/golden/memory.yaml with the three tool entry points logged, so the expected tool calls are what the agent
does, not a second copy of its rules, and a change to the agent shows up when the dataset is rebuilt. The expected
answer is the evidence the pre-fetch gathered, written out (an abstention is the agent's "I didn't see that."). It
is a reference to check labels against, not a training target: the teacher writes those. What each call returns
(`tool_results`) is what the agent's tool loop hands the model for that call, which can differ from the evidence
the answer is written from: the pre-fetch knows things the model's call does not, such as which camera a
question names. A golden question must still pass the golden eval's own checks (every `cite` cited, a `keywords`
word in the answer), or the build fails.

What is in it:

    golden    every golden question the agent answers with a tool. "What is due this week?" is left out, with the
              reason in the build's `skipped`: the agent answers it from Canvas and calendar items it pre-fetches,
              and none of the three TOOLS reads items, so there is no tool call to teach.
    variant   the same shapes asked other ways about the fixture night: where an item is (ITEMS, of which the
              memory has seen only the keys, so the rest teach "I didn't see that"), what happened in a window
              (WINDOWS spelled in several ways, under WINDOW_STEMS) and what happened to an item (STORY_ITEMS, the
              one shape the agent answers with two calls: where it was last seen, then what the memory says about
              it). Each variant is run through reference_trace() like a golden question, and the build fails if
              the agent does not answer it with a tool.

Examples whose expected tool calls are identical ask the same thing in different words, so they share a `group`,
and frame_ref is "fixture:group:<group>": a split by frame_ref or by group keeps every wording of a question on one
side. The golden questions are the scorer's (contexts/agent/eval.py); a split that holds out their groups trains
on questions the scorer never asks, and `origin` in expected.jsonl says which examples they are.

The build is deterministic (nothing reads the clock), so rebuilding gives the same folder apart from `target`. It
never deletes: it writes its four files, and refuses to touch a folder whose examples already have targets unless
force is set, so a rerun cannot lose labels.

The entry point writes a ledger row (verb fixture-dataset) and refuses to run while lab/state/STOP exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

if not __package__:  # run by path: the repo root, one folder up, is what holds contracts/, contexts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import NO_EVIDENCE, QA, SYSTEM, TOOLS, Evidence, SleepSchedule
from contexts.agent.eval import Golden, load_fixture
from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
from lab.dataset import EXAMPLES, IMAGES, MANIFEST, DatasetError, load
from lab.machine import stop
from lab.machine.record import recorded
from runtime import FakeRuntime

__all__ = ["TASK", "DATASET_ID", "FIXTURE_ACTION_ID", "EXPECTED", "GOLDEN_DIR", "OUT_DIR", "ITEMS", "WINDOWS",
           "WINDOW_STEMS", "LAST_SEEN_TEMPLATES", "STORY_ITEMS", "STORY_TEMPLATES", "BuildError", "Unsupported", "Trace", "Expected", "BuildResult",
           "reference_trace", "output_schema", "instruction", "variant_questions", "build", "load_expected",
           "main"]

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "fixtures" / "golden"
OUT_DIR = REPO_ROOT / "fixtures" / "learning"
AGENT_CONFIG = REPO_ROOT / "config" / "agent.yaml"

TASK = "tool-calling"
DATASET_ID = "tool-calling-fixture"
FIXTURE_ACTION_ID = "fixture:golden-questions"
EXPECTED = "expected.jsonl"
NOTHING_FOUND = "nothing found"   # what the agent's tool loop hands the model when a tool finds nothing

# Variant questions. Each must be a shape QA.prefetch() answers with one tool; the build checks that it is.
ITEMS = (("keys", True), ("wallet", False), ("phone", False), ("glasses", True), ("passport", False),
         ("umbrella", False))   # (name, plural): the memory has seen the keys and none of the rest
LAST_SEEN_TEMPLATES = ("Where did I leave my {item}?", "Where did I put my {item}?", "Where {be} my {item}?",
                       "Where {be} my {item} right now?", "Where did I leave my {item} again?",
                       "Where {be} the {item}?")
WINDOWS = (("overnight", "last night"), ("today",), ("in the last hour", "in the past hour", "in the last 60 minutes"),
           ("in the last 2 hours", "in the past 2 hours"), ("in the last 3 hours",), ("in the last 6 hours",),
           ("in the last 12 hours",), ("in the last 30 minutes", "in the past 30 minutes"),
           ("in the last 10 minutes",), ("in the last 90 minutes",))   # spellings of one window per tuple
WINDOW_STEMS = ("What happened {when}?", "What happened at our table {when}?",
                "Tell me what the cameras saw {when}.", "Was there any activity {when}?",
                "Did anyone come to the door {when}?", "Was anyone at the desk {when}?")   # the last two name a camera
STORY_ITEMS = ("keys", "laptop", "coffee mug", "phone charger", "package", "wallet")   # what happened to ...
STORY_TEMPLATES = ("What happened to my {item}?", "What happened to the {item}?", "What became of my {item}?",
                   "What became of the {item}?")


class BuildError(RuntimeError):
    """The dataset could not be built, or was refused. outcome is the word the ledger row gets."""

    def __init__(self, message: str, outcome: str = "error") -> None:
        super().__init__(message)
        self.outcome = outcome


class Unsupported(Exception):
    """The agent has no tool call for this question; str() is the reason."""


@dataclass(frozen=True)
class Trace:
    """What the agent does for a question: the tool calls, what each returned, what they cite, the answer."""

    tool_calls: tuple[dict[str, Any], ...]   # {"name": ..., "arguments": {...}}, in the order made
    tool_results: tuple[str, ...]            # per call, as the tool loop shows the model ("nothing found" if empty)
    cited: tuple[str, ...]                   # episode ids, frame refs and links the evidence cites, sorted
    answer: str


class ExpectedCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class ExpectedLabel(BaseModel):
    tool_calls: list[ExpectedCall]
    answer: str


class Expected(BaseModel):
    """A row of expected.jsonl: what a correct label for one example holds."""

    id: str
    question: str
    origin: Literal["golden", "variant"]
    group: str
    label: ExpectedLabel
    tool_results: list[str]
    cited: list[str]
    keywords: list[str] = []
    no_evidence: bool


@dataclass(frozen=True)
class _Entry:
    id: str
    question: str
    origin: Literal["golden", "variant"]
    trace: Trace
    keywords: list[str]


@dataclass(frozen=True)
class BuildResult:
    root: Path
    examples: int
    golden: int
    variants: int
    groups: int
    skipped: tuple[dict[str, str], ...]   # golden questions with no tool call, and why


# --- what the agent does ------------------------------------------------------------------

class _RecordingQA(QA):
    """The agent's own QA with its tool entry points logged: whatever prefetch() calls is the reference policy.

    Each call is logged with the arguments the model controls and with what the tool loop would hand the model
    for them, which is the same tool called with only those arguments: prefetch can pass more (a question that
    names a camera, a newest-first sort) that the model's call cannot."""

    def __init__(self, memory: FakeMemory, now: datetime, sleep: SleepSchedule) -> None:
        super().__init__(FakeRuntime(), memory, clock=lambda: now, sleep=sleep)
        self.calls: list[tuple[str, dict[str, str], list[str]]] = []
        self.no_tool: str | None = None

    def last_seen(self, label: str, ev: Evidence, *extra: Any, **named: Any) -> None:
        super().last_seen(label, ev, *extra, **named)
        shown = Evidence()
        super().last_seen(label, shown)
        self.calls.append(("last_seen", {"label": label}, shown.lines))

    def episodes_between(self, start: datetime, end: datetime, ev: Evidence, *extra: Any, **named: Any) -> None:
        super().episodes_between(start, end, ev, *extra, **named)
        shown = Evidence()
        super().episodes_between(start, end, shown)
        self.calls.append(("episodes_between", {"start": start.isoformat(), "end": end.isoformat()}, shown.lines))

    def search_memory(self, q: str, ev: Evidence, *extra: Any, **named: Any) -> None:
        super().search_memory(q, ev, *extra, **named)
        shown = Evidence()
        super().search_memory(q, shown)
        self.calls.append(("search_memory", {"q": q}, shown.lines))

    def upcoming_items(self, q: str, ev: Evidence) -> None:
        self.no_tool = ("the agent answers it from Canvas and calendar items it pre-fetches, and none of its "
                        "TOOLS reads items")


def _sentence(line: str) -> str:
    text = line.strip()
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def reference_trace(question: str, memory: FakeMemory, now: datetime, sleep: SleepSchedule) -> Trace:
    """The tool calls the agent makes for the question against this memory, and the answer they support.

    Runs QA.prefetch() with the tool entry points logged. Unsupported, with the reason, for a question the agent
    does not pre-fetch (it would use the open tool loop) or answers without a tool."""
    qa = _RecordingQA(memory, now, sleep)
    evidence = qa.prefetch(question)
    if qa.no_tool:
        raise Unsupported(qa.no_tool)
    if evidence is None or not qa.calls:
        raise Unsupported("the agent has no pre-fetch for this kind of question, so it would use the open tool loop")
    return Trace(
        tool_calls=tuple({"name": name, "arguments": arguments} for name, arguments, _ in qa.calls),
        tool_results=tuple("\n".join(shown) or NOTHING_FOUND for _, _, shown in qa.calls),
        cited=tuple(sorted({ref for c in evidence.citations for ref in (c.episode_id, c.frame_ref, c.url) if ref})),
        answer=" ".join(_sentence(line) for line in evidence.lines) if evidence.lines else NO_EVIDENCE,
    )


# --- the task -------------------------------------------------------------------------------

def instruction(now: datetime) -> str:
    """The system prompt of the agent's tool loop at this time (QA.tool_loop builds the same text)."""
    return f"{SYSTEM} The time now is {now.isoformat()}. Use tools to find evidence."


def output_schema() -> dict[str, Any]:
    """JSON schema of a label: a list of calls to the agent's TOOLS with exactly their arguments, and an answer."""
    calls = []
    for tool in TOOLS:
        function = tool["function"]
        calls.append({
            "type": "object",
            "properties": {"name": {"const": function["name"]},
                           "arguments": {**function["parameters"], "additionalProperties": False}},
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    return {
        "type": "object",
        "properties": {"tool_calls": {"type": "array", "items": {"oneOf": calls}},
                       "answer": {"type": "string", "minLength": 1}},
        "required": ["tool_calls", "answer"],
        "additionalProperties": False,
    }


def variant_questions() -> list[str]:
    """The variant questions in a fixed order: where an item is, what happened in a window, what happened to an item."""
    questions = []
    for name, plural in ITEMS:
        for template in LAST_SEEN_TEMPLATES:
            questions.append(template.format(item=name, be="are" if plural else "is"))
    for spellings in WINDOWS:
        for when in spellings:
            questions.extend(stem.format(when=when) for stem in WINDOW_STEMS)
    for item in STORY_ITEMS:
        questions.extend(template.format(item=item) for template in STORY_TEMPLATES)
    return questions


# --- the build ------------------------------------------------------------------------------

def _normal(question: str) -> str:
    return " ".join(question.lower().split()).rstrip("?.! ")


def _group(trace: Trace) -> str:
    digest = hashlib.sha1(json.dumps(trace.tool_calls, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"{trace.tool_calls[0]['name']}-{digest}"


def _check_golden(number: int, golden: Golden, trace: Trace) -> None:
    """The golden eval's own pass test, applied to the reference: citations and keywords, or an abstention."""
    where = f"golden question {number}"
    if golden.no_evidence:
        if trace.cited or trace.answer != NO_EVIDENCE:
            raise BuildError(f"{where} expects no evidence, but the agent cites {list(trace.cited)} and answers "
                             f"{trace.answer!r}")
        return
    missing = sorted(set(golden.cite) - set(trace.cited))
    if missing:
        raise BuildError(f"{where}: the agent's tool call does not cite {missing}")
    if golden.keywords and not any(word.lower() in trace.answer.lower() for word in golden.keywords):
        raise BuildError(f"{where}: the answer holds none of the keywords {golden.keywords}")


def _label(trace: Trace) -> ExpectedLabel:
    return ExpectedLabel(tool_calls=[ExpectedCall(**call) for call in trace.tool_calls], answer=trace.answer)


def _guard_labels(root: Path, force: bool) -> None:
    """Refuse to rebuild over targets: they cost teacher time, and a rebuild would drop them."""
    if force or not (root / EXAMPLES).exists():
        return
    try:
        labeled = len(load(root, require_images=False).labeled())
    except DatasetError as err:
        raise BuildError(f"{root.name} exists and cannot be read ({err}); remove it or pass force",
                         outcome="refused") from None
    if labeled:
        raise BuildError(f"{root.name} already has {labeled} labeled examples; rebuilding would drop them "
                         "(pass force to rebuild anyway)", outcome="refused")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def build(out_dir: str | Path = OUT_DIR, *, golden_dir: str | Path = GOLDEN_DIR, dataset_id: str = DATASET_ID,
          variants: bool = True, force: bool = False, sleep: SleepSchedule | None = None) -> BuildResult:
    """Write <out_dir>/<dataset_id>/ (manifest.json, examples.jsonl, images/, expected.jsonl).

    BuildError when a golden question no longer passes its own checks against the agent, when a variant is not a
    question the agent answers with a tool, or (outcome "refused") when the folder already holds labels."""
    try:
        memory, now, questions = load_fixture(Path(golden_dir))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BuildError(f"cannot read the golden fixtures in {Path(golden_dir).name}: {type(exc).__name__}") from None
    sleep = sleep or SleepSchedule.load(AGENT_CONFIG)
    root = Path(out_dir) / dataset_id
    _guard_labels(root, force)

    entries: list[_Entry] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for number, golden in enumerate(questions, 1):
        seen.add(_normal(golden.q))
        try:
            trace = reference_trace(golden.q, memory, now, sleep)
        except Unsupported as why:
            skipped.append({"question": golden.q, "reason": str(why)})
            continue
        _check_golden(number, golden, trace)
        entries.append(_Entry(f"golden-{number:02d}", golden.q, "golden", trace, golden.keywords))
    if variants:
        for question in variant_questions():
            if _normal(question) in seen:
                continue
            seen.add(_normal(question))
            try:
                trace = reference_trace(question, memory, now, sleep)
            except Unsupported as why:
                raise BuildError(f"variant question {question!r} is not answered with a tool: {why}") from None
            digest = hashlib.sha1(_normal(question).encode("utf-8")).hexdigest()[:8]
            entries.append(_Entry(f"v-{digest}", question, "variant", trace, []))
    if len({entry.id for entry in entries}) != len(entries):
        raise BuildError("two questions got the same example id")

    examples, expected = [], []
    for entry in entries:
        group = _group(entry.trace)
        examples.append(DatasetExample(id=entry.id, image="", frame_ref=f"fixture:group:{group}", ts=now,
                                       context=entry.question))
        expected.append(Expected(id=entry.id, question=entry.question, origin=entry.origin, group=group,
                                 label=_label(entry.trace), tool_results=list(entry.trace.tool_results),
                                 cited=list(entry.trace.cited), keywords=entry.keywords,
                                 no_evidence=not entry.trace.cited))
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        spec=TaskSpec(task=TASK, instruction=instruction(now), output_schema=output_schema()),
        created=now, examples=len(examples), action_id=FIXTURE_ACTION_ID)

    (root / IMAGES).mkdir(parents=True, exist_ok=True)
    _write(root / IMAGES / ".gitkeep", "")
    _write(root / MANIFEST, json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n")
    _write(root / EXAMPLES, "".join(example.model_dump_json() + "\n" for example in examples))
    _write(root / EXPECTED, "".join(row.model_dump_json() + "\n" for row in expected))
    return BuildResult(root=root, examples=len(examples), golden=sum(e.origin == "golden" for e in entries),
                       variants=sum(e.origin == "variant" for e in entries), groups=len({row.group for row in expected}),
                       skipped=tuple(skipped))


def load_expected(root: str | Path) -> dict[str, Expected]:
    """expected.jsonl of a built folder, by example id."""
    text = (Path(root) / EXPECTED).read_bytes().decode("utf-8")
    rows = [Expected.model_validate_json(line) for line in text.split("\n") if line.strip()]
    return {row.id: row for row in rows}


# --- the entry point ----------------------------------------------------------------------

@recorded("fixture-dataset")
@stop.guarded
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Build the no-camera tool-calling dataset from the golden questions and memory."""
    parser = argparse.ArgumentParser(prog="python lab/fixture_dataset.py", description=main.__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--out", default=str(OUT_DIR), help="the folder that gets <dataset id>/ (default: fixtures/learning)")
    parser.add_argument("--golden", default=str(GOLDEN_DIR), help="questions.yaml and memory.yaml (default: fixtures/golden)")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--no-variants", action="store_true", help="only the golden questions")
    parser.add_argument("--force", action="store_true", help="rebuild over a folder whose examples already have targets")
    args = parser.parse_args(argv)
    try:
        result = build(args.out, golden_dir=args.golden, dataset_id=args.dataset_id, variants=not args.no_variants,
                       force=args.force)
    except BuildError as err:
        return {"ok": False, "outcome": err.outcome, "error": str(err)}
    return {"ok": True, "outcome": "built", "dataset_id": args.dataset_id, "path": str(result.root),
            "examples": result.examples, "golden": result.golden, "variants": result.variants,
            "groups": result.groups, "skipped": list(result.skipped), "refs": {"dataset_id": args.dataset_id}}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
