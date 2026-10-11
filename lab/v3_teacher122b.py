"""Label tool-calling v3's training rows with the 122B pair, the way lab/fixture_dataset_v3_teacher.py asks the teacher.

    python lab/v3_teacher122b.py --events <box events.jsonl.gz> --account C --out <answers.jsonl> [--workers 16]
    python lab/v3_teacher122b.py --summarize <answers.jsonl> --rows-out fixtures/learning/tool-calling-v3/rows-train-teacher122b.jsonl

What it asks. Every row of fixtures/learning/tool-calling-v3/rows-train.jsonl: its question (the user turn) at its own
clock (the time in its system prompt), as one tool loop of ask_one() in lab/fixture_dataset_v3_teacher.py: the
teacher's model name "teacher" on the account's pod (the pair's head, pod C port 8000), thinking off, greedy, v3's
TOOLS, each call run by the agent's own tool code against the box's memory replayed to that clock and shortened by
v3's clip(). The served checkpoint must be the Qwen3.5-122B pair. Requests only: no server is started or changed.

What it writes. --out: one JSON line per question as ask_one() writes it (id, question, ts, tool_calls with results,
answer, outcome, checkpoint), plus row_id. --summarize turns it into rows in rows-train.jsonl's schema (messages,
tools) labeled_by "Qwen3.5-122B pair", each with rule_agreement against the rule row (the truth): calls (the same
tool calls with the same arguments, in order) and answer_times (the same clock times named in the answer).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contexts.agent.application.qa import SleepSchedule, TOOLS
from lab import fixture_dataset_v3 as v3
from lab.fixture_dataset import AGENT_CONFIG
from lab.fixture_dataset_v3_teacher import TeacherAsk, ask_one

ROWS = Path(__file__).resolve().parents[1] / "fixtures" / "learning" / "tool-calling-v3" / "rows-train.jsonl"
EXPECT = "Qwen3.5-122B"
LABELED_BY = "Qwen3.5-122B pair"
NOW = re.compile(r"The time now is (\S+?)\. Use tools")
TIMES = re.compile(r"\b\d{1,2}:\d{2}\s?[AP]M\b", re.I)


def rule_rows(path: Path = ROWS) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row["messages"]
        now = datetime.fromisoformat(NOW.search(msgs[0]["content"]).group(1))
        calls = [(c["function"]["name"], c["function"]["arguments"]) for m in msgs if m["role"] == "assistant"
                 for c in (m.get("tool_calls") or [])]
        answer = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), "")
        out.append({"row": row, "id": row["id"], "question": msgs[1]["content"], "now": now, "calls": calls,
                    "answer": answer, "group": row.get("frame_ref", "")})
    return out


def ask_all(events: str, out: str, teacher: Any, workers: int) -> dict[str, Any]:
    from lab import label
    sleep = SleepSchedule.load(AGENT_CONFIG)
    decode = label.Decode()
    rules = rule_rows()
    checkpoint = teacher.served_checkpoint(EXPECT)
    target = Path(out)
    done_ids = {json.loads(l)["row_id"] for l in target.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if target.exists() else set()
    todo = [r for r in rules if r["id"] not in done_ids]
    rows, _ = v3.read_events(events)
    clocks = sorted({r["now"] for r in todo})
    lock = threading.Lock()
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        copies: dict[datetime, str] = {}

        def visit(now: datetime, memory: Any) -> None:
            path = str(Path(scratch) / f"m{len(copies):04d}.sqlite")
            target_db = sqlite3.connect(path)
            memory.db.backup(target_db)
            target_db.close()
            copies[now] = path

        v3.replay(rows, clocks, visit, path=str(Path(scratch) / "replay.sqlite")).db.close()
        print(json.dumps({"replayed_clocks": len(copies), "todo": len(todo), "replay_s": round(time.monotonic() - t0, 1)}),
              flush=True)

        def one(r: dict[str, Any]) -> dict[str, Any]:
            got = ask_one(teacher, TeacherAsk(r["group"], r["question"], r["now"]), copies[r["now"]], sleep, decode)
            got.update(row_id=r["id"], checkpoint=checkpoint)
            with lock, target.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(got, ensure_ascii=False) + "\n")
            return got

        t1 = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(pool.map(one, todo))
    outcomes: dict[str, int] = {}
    for a in answers:
        outcomes[a["outcome"]] = outcomes.get(a["outcome"], 0) + 1
    return {"asked": len(answers), "outcomes": outcomes, "checkpoint": checkpoint, "ask_s": round(time.monotonic() - t1, 1),
            "s_per_example": round((time.monotonic() - t1) / max(1, len(answers)), 3)}


def _norm_args(a: Any) -> Any:
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return a
    return json.dumps(a, sort_keys=True)


def summarize(answers_path: str, rows_out: str, extra: dict[str, Any]) -> dict[str, Any]:
    rules = {r["id"]: r for r in rule_rows()}
    answers = {}
    for line in Path(answers_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            answers[a["row_id"]] = a
    out_rows, counts = [], {"accepted": 0, "rejected": 0, "failed": 0}
    agree_calls = agree_times = agree_both = 0
    for rid, r in rules.items():
        a = answers.get(rid)
        if a is None:
            continue
        outcome = a["outcome"]
        kind = "accepted" if outcome == "kept" else ("failed" if outcome.startswith("failed") else "rejected")
        counts[kind] += 1
        t_calls = [(c["name"], _norm_args(c["arguments"])) for c in a.get("tool_calls", [])]
        r_calls = [(n, _norm_args(args)) for n, args in r["calls"]]
        same_calls = t_calls == r_calls
        same_times = sorted(x.upper().replace(" ", "") for x in TIMES.findall(a.get("answer") or "")) == \
            sorted(x.upper().replace(" ", "") for x in TIMES.findall(r["answer"] or ""))
        both = same_calls and same_times and kind == "accepted"
        agree_calls += same_calls and kind == "accepted"
        agree_times += same_times and kind == "accepted"
        agree_both += both
        msgs = [r["row"]["messages"][0], r["row"]["messages"][1]]
        for n, c in enumerate(a.get("tool_calls", [])):
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}]})
            msgs.append({"role": "tool", "content": c["result"]})
        msgs.append({"role": "assistant", "content": a.get("answer") or ""})
        out_rows.append({"id": rid, "frame_ref": r["row"].get("frame_ref"), "messages": msgs, "tools": r["row"]["tools"],
                         "labeled_by": LABELED_BY, "outcome": outcome, "checkpoint": a.get("checkpoint"),
                         "rule_agreement": {"calls": same_calls, "answer_times": same_times, "both": both}})
    Path(rows_out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows), encoding="utf-8",
                              newline="\n")
    total = len(out_rows)
    summary = {"rows": total, **counts, "agree_calls": agree_calls, "agree_answer_times": agree_times,
               "agree_both": agree_both, "agree_both_share": round(agree_both / total, 4) if total else None, **extra}
    Path(rows_out).with_name("rows-train-teacher122b.summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                               encoding="utf-8", newline="\n")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(prog="python lab/v3_teacher122b.py", description=__doc__.split("\n")[0])
    p.add_argument("--events")
    p.add_argument("--account", type=str.upper, default="C")
    p.add_argument("--out")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--summarize")
    p.add_argument("--rows-out")
    p.add_argument("--extra", default="{}", help="JSON merged into the summary (the run's joules and seconds)")
    a = p.parse_args(argv)
    if a.summarize:
        return summarize(a.summarize, a.rows_out, json.loads(a.extra))
    from lab import hub, label
    accounts = hub.load_accounts([a.account], require_role=False)
    with label.Teacher.from_hub(accounts[a.account]) as teacher:
        return ask_all(a.events, a.out, teacher, a.workers)


if __name__ == "__main__":
    print(json.dumps(main()))
