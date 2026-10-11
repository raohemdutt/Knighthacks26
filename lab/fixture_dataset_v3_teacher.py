"""The teacher's part of tool-calling v3: the questions the agent's rules cannot label, asked of the teacher on pod C.

    python lab/fixture_dataset_v3_teacher.py --events <box events.jsonl.gz> --account C --out <teacher.jsonl>

Which questions. lab/fixture_dataset_v3.py labels a question by the agent's own pre-fetch (reference_trace()). Two
kinds of question the box really gets have no such label, because the agent answers them from evidence no single
tool call returns: "when did I last see my X?" (QA.last_time ranks search hits itself) and "what happened to my X?"
(QA.trace adds the detector's tagged episodes). TEACHER_TEMPLATES asks them about the train objects of v3, and
ACTIVITY asks them about the desk itself, each at a clock taken from the box's log in v3's train stretch. A question
is sent only when reference_trace() at that clock really raises Unsupported; one the rules can label never goes.

How it asks. Requests only: the teacher's server is never started, stopped or changed. Each question is one tool
loop as lab/label.py runs one (the teacher's model name "teacher", thinking off, greedy, at most MAX_TOOL_CALLS
rounds), with the example's own time in the system prompt and v3's TOOLS. Each call the teacher makes is run by the
agent's own tool code (QA._run_tool) against the box's memory as it stood at that clock (a copy of the replayed store
taken at the checkpoint), and the result goes back shortened by v3's clip() rule, exactly as a v3 row shows it.
vLLM may return a call as markup in the text (served without --tool-call-parser): lab/format_traces.py
parse_tool_calls() reads it, the same parser the scorer uses. WORKERS questions are asked at the same time.

What is written. One JSON line per question asked: id, group, question, ts, the calls with their (clipped) results,
the answer, the served checkpoint, and outcome (kept, or why not). Only kept lines become v3 rows (train side only,
groups named teacher-<thing>), and the build marks them teacher-labeled.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
import tempfile
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contexts.agent.application.qa import MAX_TOOL_CALLS, QA, TOOLS, Evidence, SleepSchedule
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.infrastructure.store import MemoryStore
from contracts.types import ToolCall
from lab import fixture_dataset_v3 as v3
from lab import format_traces as ft
from lab.fixture_dataset import AGENT_CONFIG, NOTHING_FOUND, BuildError, Unsupported, _normal, instruction, reference_trace
from runtime import FakeRuntime

TEACHER_TEMPLATES = ("When did I last see my {item}?", "When was the last time you saw my {item}?",
                     "What happened to my {item}?", "What time did you last see the {item}?")
ACTIVITY = ("When did someone last sit at the desk?", "When was the last time anyone was at the desk?",
            "When did someone last leave the table?", "When was the last time someone used the laptop?")
WORKERS = 16


@dataclass(frozen=True)
class TeacherAsk:
    group: str
    question: str
    now: datetime


def teacher_asks(p: v3.Plan) -> list[TeacherAsk]:
    """The teacher's questions: the train objects of the plan, and the desk, each at clocks of v3's train stretch."""
    out: list[TeacherAsk] = []
    low, high = p.parts["train"]
    for label, part in p.objects.items():
        if part != "train":
            continue
        name, _ = v3.ASKABLE[label]
        group = "teacher-" + name.replace(" ", "-")
        for number, template in enumerate(TEACHER_TEMPLATES):
            question = template.format(item=name)
            seed = f"{group}|{number}|{question}"
            picks = v3._candidates(p.times, max(low, p.first_seen[label]), high, seed, count=1)
            if picks:
                out.append(TeacherAsk(group, question, v3._clock(picks[0], seed)))
    for number, question in enumerate(ACTIVITY):
        seed = f"teacher-desk|{number}|{question}"
        picks = v3._candidates(p.times, low, high, seed, count=1)
        if picks:
            out.append(TeacherAsk("teacher-desk", question, v3._clock(picks[0], seed)))
    return out


def example_id(question: str, now: datetime) -> str:
    return "t-" + hashlib.sha1(f"{_normal(question)}|{now.isoformat()}".encode("utf-8")).hexdigest()[:10]


def _tool_runner(path: str, now: datetime, sleep: SleepSchedule):
    store = MemoryStore(path)
    qa = QA(FakeRuntime(), SqliteMemory(store), clock=lambda: now, sleep=sleep)

    def run(name: str, arguments: dict[str, Any]) -> str:
        found = Evidence()
        asyncio.run(qa._run_tool(ToolCall(id="", name=name, arguments=arguments), found))
        text = "\n".join(found.lines) or NOTHING_FOUND
        clipped, _ = v3.clip(v3.Trace(tool_calls=({},), tool_results=(text,), cited=("x",), answer=""))
        return clipped.tool_results[0]
    return run, store


def ask_one(teacher: Any, ask: TeacherAsk, path: str, sleep: SleepSchedule, decode: Any) -> dict[str, Any]:
    run, store = _tool_runner(path, ask.now, sleep)
    row: dict[str, Any] = {"id": example_id(ask.question, ask.now), "group": ask.group, "question": ask.question,
                           "ts": ask.now.isoformat(), "tool_calls": [], "answer": None, "outcome": "kept"}
    messages: list[dict[str, Any]] = [{"role": "system", "content": instruction(ask.now)},
                                      {"role": "user", "content": ask.question}]
    try:
        for round_number in range(MAX_TOOL_CALLS + 1):
            completion = teacher.chat(messages, decode=decode, tools=TOOLS)
            if completion.finish_reason == "length":
                row["outcome"] = "truncated"
                return row
            calls = [(c.id or f"call-{round_number}-{n}", c.name, c.arguments) for n, c in enumerate(completion.calls)]
            content = completion.content or ""
            if not calls and "<tool_call>" in content:
                content, parsed = ft.parse_tool_calls(content, TOOLS)
                calls = [(f"call-{round_number}-{n}", c.name, json.dumps(dict(c.arguments)))
                         for n, c in enumerate(parsed)]
            if not calls:
                row["answer"] = content.strip()
                break
            if round_number == MAX_TOOL_CALLS:
                row["outcome"] = "tool_rounds"
                return row
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
                                            for cid, name, args in calls]})
            for cid, name, args in calls:
                try:
                    arguments = json.loads(args) if str(args).strip() else {}
                except ValueError:
                    arguments = None
                if not isinstance(arguments, dict) or name not in {t["function"]["name"] for t in TOOLS}:
                    row["outcome"] = "bad_call"
                    return row
                result = run(name, arguments)
                row["tool_calls"].append({"name": name, "arguments": arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": cid, "content": result})
        if not row["answer"]:
            row["outcome"] = "empty"
        elif not row["tool_calls"]:
            row["outcome"] = "no_call"
    except Exception as exc:   # one question's failure is recorded and does not stop the others
        row["outcome"] = f"failed: {type(exc).__name__}"
    finally:
        store.db.close()
    return row


def ask_teacher(events: str | Path, out: str | Path, teacher: Any, *, workers: int = WORKERS,
                sleep: SleepSchedule | None = None, decode: Any = None) -> dict[str, Any]:
    from lab import label
    sleep = sleep or SleepSchedule.load(AGENT_CONFIG)
    decode = decode or label.Decode()
    rows, _ = v3.read_events(events)
    p = v3.plan(rows)
    asks = teacher_asks(p)
    by_now: dict[datetime, list[TeacherAsk]] = {}
    for ask in asks:
        by_now.setdefault(ask.now, []).append(ask)
    ruled: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        copies: dict[datetime, str] = {}

        def visit(now: datetime, memory: SqliteMemory) -> None:
            for ask in by_now[now]:
                try:
                    reference_trace(ask.question, memory, now, sleep)
                    ruled.append(ask.question)       # the rules label it: v3 proper covers this kind
                except Unsupported:
                    pass
            path = str(Path(scratch) / f"m{len(copies):04d}.sqlite")
            target = sqlite3.connect(path)
            memory.db.backup(target)
            target.close()
            copies[now] = path

        v3.replay(rows, sorted(by_now), visit, path=str(Path(scratch) / "replay.sqlite")).db.close()
        if ruled:
            raise BuildError(f"{len(ruled)} teacher questions have a rule label, for example {ruled[0]!r}")
        checkpoint = teacher.served_checkpoint()
        target = Path(out)
        earlier = {json.loads(line)["id"]: json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
                   if line.strip()} if target.exists() else {}
        todo = [ask for ask in asks if example_id(ask.question, ask.now) not in earlier]   # a rerun asks only the rest
        lock = threading.Lock()

        def one(ask: TeacherAsk) -> dict[str, Any]:
            row = ask_one(teacher, ask, copies[ask.now], sleep, decode)
            row["checkpoint"] = checkpoint
            with lock, target.open("a", encoding="utf-8", newline="\n") as handle:   # each line as it comes
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return row

        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = list(earlier.values()) + list(pool.map(one, todo))
        order = {example_id(ask.question, ask.now): n for n, ask in enumerate(asks)}   # the file in plan order at the end
        done.sort(key=lambda row: order.get(row["id"], len(order)))
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in done), encoding="utf-8",
                          newline="\n")
    outcomes: dict[str, int] = {}
    for row in done:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    return {"asked": len(done), "outcomes": outcomes, "checkpoint": checkpoint, "out": str(out)}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="python lab/fixture_dataset_v3_teacher.py", description=__doc__.split("\n")[0])
    parser.add_argument("--events", required=True)
    parser.add_argument("--account", type=str.upper, default="C")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    from lab import hub, label
    accounts = hub.load_accounts([args.account], require_role=False)
    with label.Teacher.from_hub(accounts[args.account]) as teacher:
        return ask_teacher(args.events, args.out, teacher, workers=args.workers)


if __name__ == "__main__":
    print(json.dumps(main()))
