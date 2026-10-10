"""Golden-question eval (A5).

    python -m contexts.agent.eval                                  # fake model: checks retrieval only
    python -m contexts.agent.eval --runtime llama --base-url http://127.0.0.1:8080
    python -m contexts.agent.eval --runtime llama --json out.json  # for before/after numbers

Runs the golden questions against fixture memory and scores citations (did
the agent fetch the right evidence?) and, with a real model, the answer text.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import NO_EVIDENCE, QA, SleepSchedule
from contracts.memory import Episode, Sighting
from contracts.types import ModelVersion
from runtime import FakeRuntime
from runtime.interface import Runtime

GOLDEN = Path("fixtures/golden")


class Golden(BaseModel):
    q: str
    cite: list[str] = []
    keywords: list[str] = []
    no_evidence: bool = False


class Result(BaseModel):
    q: str
    answer: str
    cited: list[str]
    citations_ok: bool
    answer_ok: bool | None  # None when the model is fake
    latency_ms: int

    @property
    def passed(self) -> bool:
        return self.citations_ok and self.answer_ok is not False


def load_fixture(root: Path = GOLDEN) -> tuple[FakeMemory, datetime, list[Golden]]:
    mem = yaml.safe_load((root / "memory.yaml").read_text())
    memory = FakeMemory(
        episodes=[Episode.model_validate(e) for e in mem["episodes"]],
        sightings=[Sighting.model_validate(s) for s in mem["sightings"]],
    )
    now = datetime.fromisoformat(mem["now"])
    questions = [Golden.model_validate(g) for g in yaml.safe_load((root / "questions.yaml").read_text())]
    return memory, now, questions


def run(runtime: Runtime, root: Path = GOLDEN, judge_answers: bool = True) -> list[Result]:
    memory, now, questions = load_fixture(root)
    qa = QA(runtime, memory, clock=lambda: now, sleep=SleepSchedule.load())
    results = []
    for g in questions:
        start = time.monotonic()
        a = asyncio.run(qa.ask(g.q))
        latency = int((time.monotonic() - start) * 1000)
        cited = sorted({ref for c in a.citations for ref in (c.episode_id, c.frame_ref) if ref})
        if g.no_evidence:
            citations_ok = not a.citations
            answer_ok = a.text == NO_EVIDENCE if judge_answers or not a.citations else None
        else:
            citations_ok = set(g.cite) <= set(cited)
            answer_ok = any(k.lower() in a.text.lower() for k in g.keywords) if judge_answers else None
        results.append(Result(q=g.q, answer=a.text, cited=cited, citations_ok=citations_ok, answer_ok=answer_ok, latency_ms=latency))
    return results


def score(results: list[Result]) -> float:
    return sum(r.passed for r in results) / len(results)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runtime", choices=["fake", "llama"], default="fake")
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model-version", default="atlas-base")
    p.add_argument("--json", type=Path, help="write results here")
    args = p.parse_args(argv)

    if args.runtime == "fake":
        runtime: Runtime = FakeRuntime()
    else:
        from runtime.llama import LlamaServerRuntime

        runtime = LlamaServerRuntime(ModelVersion(version=args.model_version, base="Qwen/Qwen3.5-4B"), base_url=args.base_url)

    results = run(runtime, judge_answers=args.runtime != "fake")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {r.latency_ms:>6} ms  {r.q}\n      -> {r.answer}\n      cited: {', '.join(r.cited) or '-'}")
    total = score(results)
    print(f"\nscore {total:.2f} ({sum(r.passed for r in results)}/{len(results)}) model={runtime.model_info().version}")
    if args.json:
        args.json.write_text(json.dumps({"score": total, "model": runtime.model_info().version, "results": [r.model_dump() for r in results]}, indent=2))
    return 0 if total == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
