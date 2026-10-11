"""A stand-in scorer for lab/eval_gate.py: canned per-item results, shaped like `contexts.agent.eval --json` output.

    from lab.eval_gate_fake import FakeClient

    client = FakeClient("fixtures/learning/tool-calling-fixture", {
        "http://current": [passing_run_1, passing_run_2],   # one entry per call; the last one repeats
        "http://candidate": [passing],
        "http://teacher": [passing],
    })
    gate.run_gate(client=client, ...)

Each entry is the set of held-out questions (by their text, or by example id with by_id=True) that pass on that
call; every other question the fake answers fails. By default it answers every held-out question of the dataset's
split.json; questions= narrows or reorders that, and extra_questions= adds questions the split does not hold as
held-out (a training question, or one outside the dataset), so a test can show what the gate does with them.

A result has the scorer's own fields (contexts.agent.eval.Result: q, answer, cited, citations_ok, answer_ok,
latency_ms, shape); a passing item has citations_ok and answer_ok true, a failing one answer_ok false. With by_id
each result also carries the example id, as an image-task scorer would.

decode() answers DECODE for every server unless decode= names another setting (or None, for a server whose
setting cannot be read) for a base URL. scorer_commit() answers commit and counts its calls. Nothing here opens a
socket or reads the clock.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from contexts.agent.eval import Result
from lab.dataset import load_split
from lab.eval_gate import DECODE

__all__ = ["FAKE_COMMIT", "FakeClient", "scorer_output"]

FAKE_COMMIT = "fa4e0000000000000000000000000000000000c0"


def scorer_output(questions: Sequence[str], passing: Iterable[str], model: str,
                  ids: Sequence[str] | None = None) -> dict[str, Any]:
    """What `python -m contexts.agent.eval --json` writes, for these questions, with these passing."""
    passing = set(passing)
    results = []
    for index, question in enumerate(questions):
        key = ids[index] if ids is not None else question
        ok = key in passing
        row = Result(q=question, answer="fake answer" if ok else "fake miss", cited=[], citations_ok=True,
                     answer_ok=ok, latency_ms=0).model_dump()
        if ids is not None:
            row["id"] = key
        results.append(row)
    score = sum(r["citations_ok"] and r["answer_ok"] is not False for r in results) / len(results) if results else 0.0
    return {"score": score, "model": model, "results": results}


class FakeClient:
    """The gate's client (score, decode, scorer_commit) with canned answers; calls lists every scored base URL."""

    def __init__(self, dataset: str | os.PathLike[str], outcomes: Mapping[str, Sequence[Iterable[str]]], *,
                 questions: Sequence[str] | None = None, extra_questions: Sequence[str] = (), by_id: bool = False,
                 commit: str = FAKE_COMMIT, decode: Mapping[str, Mapping[str, Any] | None] | None = None) -> None:
        held_out = load_split(dataset, require_images=False, min_held_out=1).held_out
        if questions is None:
            questions = [example.context or "" for example in held_out]
        self.questions = [*questions, *extra_questions]
        self.ids = [example.id for example in held_out] if by_id else None
        self.outcomes = {url: [set(entry) for entry in entries] for url, entries in outcomes.items()}
        self.commit = commit
        self.decode_by_url = dict(decode or {})
        self.calls: list[str] = []
        self.commit_calls = 0

    def score(self, base_url: str, *, model_version: str | None = None) -> dict[str, Any]:
        entries = self.outcomes[base_url]
        passing = entries[min(self.calls.count(base_url), len(entries) - 1)]
        self.calls.append(base_url)
        return scorer_output(self.questions, passing, model_version or "fake-model", self.ids)

    def decode(self, base_url: str) -> dict[str, Any] | None:
        if base_url in self.decode_by_url:
            setting = self.decode_by_url[base_url]
            return None if setting is None else dict(setting)
        return dict(DECODE)

    def scorer_commit(self) -> str:
        self.commit_calls += 1
        return self.commit
