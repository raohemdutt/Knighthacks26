"""Score a model on the held-out SPOT questions: python -m lab spot-score split|score.

    python -m contexts.spot.dataset --out <dir>                      # spot-qa-train/ and spot-qa-heldout/
    python -m lab spot-score split <dir> [--out <dir>/spot-qa]        # one gate-ready folder with a frozen split.json
    python -m lab spot-score score <dir>/spot-qa --base-url http://127.0.0.1:8080 \\
        [--runs 2] [--json out.json] [--node pod-a] [--commit <40 hex>] [--model <name>] [--min-held-out 50]

The set is the one contexts/spot/dataset.py writes, in the box's export layout (contracts/learning.py): text only,
each example's context is the question with the evidence a tool would hand the model, and its target is the label
{"answer": str, "citations": [row ids]}. That command writes the train and the held-out side as two folders, split by
anchor mission (held-out anchors M022 to M027, M021 left unused between them). split joins them into one folder the
way lab/dataset.py load_split() and the eval gate read a dataset: manifest.json (dataset id spot-qa), examples.jsonl
(train first, then held out, ids unchanged) and split.json listing each folder's ids on its own side. It refuses a
folder that already has a split.json and records the file's sha256 in a ledger row (verb spot-split).

The fair score. A reply is the JSON object the instruction asks for. It scores correct when both hold:

    citations   the set of its citations equals the expected set (ids compared exactly, order and repeats ignored)
    numbers     every number in the expected answer appears among the numbers of its answer (numbers below)

A reply that is not such an object (no JSON object, answer not a string, citations not a list of strings) is
malformed and scores wrong on both. A thinking block and a ``` fence around the object are removed first.

Numbers (numbers()): a number is a run of digits, with an optional decimal part and an optional sign glued to it,
read as its value, so 10.0, 10.00 and 10 are one number and 1,234 is 1234 (a comma counts as a thousands separator
only before exactly three digits). A unit or a percent sign beside it is not part of it (24min, 24 min and 86% read
24 and 86). A digit run that touches a letter, an underscore or another number's point before it, or a letter and a
hyphen (M022, WP12, A0213, PANEL-218, V2, x2), is part of an id or a word and is not a number. A minus or a Unicode
minus right before the digits, with no letter or digit before it, makes the number negative (-3.2, p=-0.4); between
two numbers it is a range (2-40 and 2 to 40 are 2 and 40), and +3.2 is 3.2. A date or a time is its parts
(2026-09-08 18:00 is 2026, 9, 8, 18 and 0). The numbers are compared as a set.

score sends one chat request per held-out example to an OpenAI-compatible server (llama-server for current and
candidate, vLLM for the teacher): the dataset's instruction as the system message and the example's context as the
user message, with the eval gate's one decode setting (thinking off, greedy, the gate's maximum output length) and
the served model's id. The answer is the shape contexts/agent/eval.py writes:

    {"score": <share correct>, "model": <served model id>, "results": [{id, frame_ref, group, q, answer, cited,
     citations_ok, answer_ok, latency_ms, shape, malformed, missing_numbers, expected, raw}, ...],
     "spot.citations_ok": ..., "spot.numbers_ok": ..., "spot.malformed": ..., "spot.n_items": ...,
     "spot.n_groups": <anchor missions>}

each result carrying contexts.agent.eval.Result's fields, so the gate counts a pass as that scorer's Result.passed,
and the example id, so the gate matches results by id. The gate reaches it by the dataset's task (spot-qa): its
registry (lab/eval_gate.py SCORERS) builds GateClient for the dataset and pairs on group_of(example), the anchor
mission, because every question asked as of one mission draws on the same history. A server that answers with an
error stops the run: a failed request is never counted as a miss.

Each run of the verb writes one ledger row (verb spot-score, task L26): metrics spot.score and the dotted keys; refs
split_sha256, commit (--commit, else git rev-parse HEAD; a run with neither is refused), model and run_id; conditions
with the decode setting, the instruction's sha256, the dataset id and whether the held-out side meets the deploy
floor. --runs 2 scores twice and answers the spread as noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

if not __package__:  # run by path on a pod: the repo root, one folder up, is what holds contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from contracts.learning import DatasetExample, DatasetManifest
from lab import dataset
from lab.dataset import MIN_HELD_OUT, DatasetError, Split
from lab.desk_score import repo_commit
from lab.eval_gate import DECODE as GATE_DECODE
from lab.eval_gate import GateError, RealClient
from lab.machine.record import record

__all__ = ["TASK", "DATASET_TASK", "DATASET_ID", "DECODE", "SpotError", "SpotExample", "numbers", "parse_reply",
           "judge", "group_of", "join", "held_out_examples", "SpotClient", "score", "GateClient", "main"]

TASK = "L26"
DATASET_TASK = "spot-qa"          # the task contexts/spot/dataset.py names in its manifests; the gate's registry key
DATASET_ID = "spot-qa"
TRAIN_DIR, HELD_OUT_DIR = "spot-qa-train", "spot-qa-heldout"
PREFIX = "spot"
TIMEOUT_S = 300.0
DECODE: dict[str, Any] = {  # the gate's setting (thinking off, greedy, its max tokens), as a chat request sends it
    "temperature": 0, "top_k": 1, "seed": 0, "max_tokens": GATE_DECODE["max_tokens"],
    "chat_template_kwargs": {"enable_thinking": False},
}
SPLIT_RULE = ("By anchor mission, as contexts.spot.dataset writes it: train is every example of spot-qa-train (anchors "
              "M006 to M020), held_out every example of spot-qa-heldout (anchors M022 to M027); M021 is unused between "
              "them. Joined once by python -m lab spot-score split.")

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(?<![A-Za-z][-_])([-+−]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\d)")
_ANCHOR = re.compile(r"\(through mission (M\d+)\)")
_QUESTION = re.compile(r"^Question: (.*)$", re.M)


class SpotError(GateError):
    """The SPOT scorer cannot go on: a split that exists already, a label missing or malformed, a server error. A
    GateError, so the gate refuses rather than scoring around it."""


@dataclass(frozen=True)
class SpotExample:
    """One held-out SPOT question: what the model is asked and the label it should give."""

    id: str
    frame_ref: str
    group: str                 # the anchor mission, the gate's unit
    question: str
    context: str               # the question with its evidence: the user message
    answer: str                # the expected answer
    citations: tuple[str, ...]  # the expected row ids


# --- the fair score -----------------------------------------------------------------------

def numbers(text: str) -> set[str]:
    """The numbers in text, each as its value in plain decimal form (see the module docstring)."""
    found = set()
    for sign, whole, part in _NUMBER.findall(text or ""):
        value = Decimal(whole.replace(",", "") + (part or ""))
        if sign in ("-", "−"):
            value = -value
        found.add("0" if value == 0 else format(value.normalize(), "f"))
    return found


def parse_reply(text: str) -> dict[str, Any] | None:
    """The reply's {"answer": str, "citations": [str, ...]}, or None when it is malformed."""
    body = _FENCE.sub("", _THINK.sub("", text or "").strip()).strip()
    candidates = [body]
    if "{" in body and "}" in body:
        candidates.append(body[body.index("{"):body.rindex("}") + 1])
    for candidate in candidates:
        try:
            reply = json.loads(candidate)
        except ValueError:
            continue
        if (isinstance(reply, dict) and isinstance(reply.get("answer"), str)
                and isinstance(reply.get("citations"), list)
                and all(isinstance(c, str) for c in reply["citations"])):
            return {"answer": reply["answer"], "citations": [c.strip() for c in reply["citations"]]}
        return None
    return None


def judge(reply: str, answer: str, citations: Sequence[str]) -> dict[str, Any]:
    """A reply against the label: citations_ok, answer_ok (every expected number present), malformed and the
    expected numbers it lacks. Correct is citations_ok and answer_ok."""
    parsed = parse_reply(reply)
    want = numbers(answer)
    if parsed is None:
        return {"answer": "", "cited": [], "citations_ok": False, "answer_ok": False, "malformed": True,
                "missing_numbers": sorted(want, key=Decimal)}
    missing = want - numbers(parsed["answer"])
    return {"answer": parsed["answer"], "cited": sorted(set(parsed["citations"])),
            "citations_ok": set(parsed["citations"]) == set(citations), "answer_ok": not missing,
            "malformed": False, "missing_numbers": sorted(missing, key=Decimal)}


# --- the dataset --------------------------------------------------------------------------

def group_of(example: DatasetExample) -> str:
    """The anchor mission an example is asked as of, as the gate's group: mission:M022."""
    found = _ANCHOR.search(example.context or "")
    if found is None:
        raise SpotError(f"example {example.id!r} names no anchor mission in its context")
    return f"mission:{found.group(1)}"


def _label(example: DatasetExample) -> tuple[str, tuple[str, ...]]:
    try:
        label = json.loads(example.target or "")
        answer, cited = label["answer"], label["citations"]
    except (ValueError, KeyError, TypeError):
        raise SpotError(f"example {example.id!r} has no {{answer, citations}} label") from None
    if not isinstance(answer, str) or not isinstance(cited, list):
        raise SpotError(f"example {example.id!r} has no {{answer, citations}} label")
    return answer, tuple(cited)


def join(root: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    """Join <root>/spot-qa-train and <root>/spot-qa-heldout into one folder with a frozen split.json (once)."""
    root = Path(root)
    out = Path(out) if out else root / DATASET_ID
    if (out / dataset.SPLIT).exists():
        raise SpotError(f"{dataset.SPLIT} already exists in {out.name}; the split is written once")
    train = dataset.load(root / TRAIN_DIR, require_images=False)
    held = dataset.load(root / HELD_OUT_DIR, require_images=False)
    for side in (train, held):
        if side.manifest.spec.task != DATASET_TASK:
            raise SpotError(f"{side.dataset_id}: task {side.manifest.spec.task!r}, not {DATASET_TASK!r}")
    if held.manifest.spec != train.manifest.spec:
        raise SpotError("the train and held-out folders were written with different task specs")
    examples = [*train.examples, *held.examples]
    out.mkdir(parents=True, exist_ok=True)
    (out / dataset.IMAGES).mkdir(exist_ok=True)
    manifest = DatasetManifest(dataset_id=DATASET_ID, spec=held.manifest.spec, created=held.manifest.created,
                               examples=len(examples), action_id=held.manifest.action_id)
    (out / dataset.MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / dataset.EXAMPLES).write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8",
                                        newline="\n")
    body = {"version": 1, "dataset_id": DATASET_ID, "rule": SPLIT_RULE,
            "train": [e.id for e in train.examples], "held_out": [e.id for e in held.examples]}
    text = json.dumps(body, indent=2) + "\n"
    with open(out / dataset.SPLIT, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    split = dataset.load_split(out, require_images=False)   # the gate's own check, before anyone relies on it
    return {"dataset_id": DATASET_ID, "folder": str(out), "sha256": split.sha256, "train": len(split.train),
            "held_out": len(split.held_out), "groups": len({group_of(e) for e in split.held_out})}


def held_out_examples(split: Split) -> list[SpotExample]:
    """The held-out side as SPOT examples, in split.json order. SpotError when one has no label."""
    out = []
    for e in split.held_out:
        answer, cited = _label(e)
        question = _QUESTION.search(e.context or "")
        out.append(SpotExample(e.id, e.frame_ref, group_of(e), question.group(1) if question else (e.context or ""),
                               e.context or "", answer, cited))
    return out


# --- the client ---------------------------------------------------------------------------

def request_body(example: SpotExample, instruction: str, model: str | None) -> dict[str, Any]:
    """The one chat request for an example: the instruction as system, the context as user, with DECODE."""
    body: dict[str, Any] = {"messages": [{"role": "system", "content": instruction},
                                         {"role": "user", "content": example.context}],
                            "stream": False, **json.loads(json.dumps(DECODE))}
    if model:
        body["model"] = model
    return body


class SpotClient:
    """An OpenAI-compatible chat API, as much as the scorer uses. transport replaces the network in tests."""

    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None,
                 timeout_s: float = TIMEOUT_S) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), transport=transport, timeout=timeout_s)

    def close(self) -> None:
        self.http.close()

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self.http.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise SpotError(f"{path}: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            raise SpotError(f"{path}: HTTP {response.status_code}")
        return response.json()

    def model(self) -> str:
        """The id the server names its model by (GET /v1/models, the first entry)."""
        data = self._call("GET", "/v1/models").get("data") or []
        return str(data[0].get("id", "")) if data else ""

    def ask(self, example: SpotExample, instruction: str, model: str | None) -> str:
        reply = self._call("POST", "/v1/chat/completions", request_body(example, instruction, model))
        try:
            return reply["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise SpotError(f"/v1/chat/completions: an answer without choices[0].message ({exc})") from exc


def score(base_url: str, examples: Sequence[SpotExample], *, instruction: str,
          transport: httpx.BaseTransport | None = None, model: str | None = None) -> dict[str, Any]:
    """Score the model behind base_url on the examples: the eval.py shape with dotted spot.* keys."""
    client = SpotClient(base_url, transport=transport)
    try:
        served = client.model()
        results = []
        for example in examples:
            start = time.monotonic()
            raw = client.ask(example, instruction, served or None)
            verdict = judge(raw, example.answer, example.citations)
            results.append({"id": example.id, "frame_ref": example.frame_ref, "group": example.group,
                            "q": example.question, **verdict,
                            "latency_ms": int((time.monotonic() - start) * 1000), "shape": "prefetch",
                            "expected": {"answer": example.answer, "citations": list(example.citations)},
                            "raw": raw})
    finally:
        client.close()
    n = len(results)

    def share(flag: str) -> float:
        return sum(bool(r[flag]) for r in results) / n if n else 0.0

    correct = [r["citations_ok"] and r["answer_ok"] for r in results]
    return {"score": sum(correct) / n if n else 0.0, "model": model or served, "results": results,
            f"{PREFIX}.citations_ok": share("citations_ok"), f"{PREFIX}.numbers_ok": share("answer_ok"),
            f"{PREFIX}.malformed": share("malformed"), f"{PREFIX}.n_items": n,
            f"{PREFIX}.n_groups": len({r["group"] for r in results})}


class GateClient(RealClient):
    """The eval gate's client for the spot-qa task: score() scores the held-out side of the gate's dataset; the
    scorer's commit and the decode attestation are the gate's own (RealClient)."""

    def __init__(self, dataset_root: str | Path, *, transport: httpx.BaseTransport | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.transport = transport
        self._examples: list[SpotExample] | None = None
        self._instruction = ""

    def score(self, base_url: str, *, model_version: str | None = None) -> dict[str, Any]:
        if self._examples is None:
            split = dataset.load_split(self.dataset_root, require_images=False, min_held_out=1)
            self._examples = held_out_examples(split)
            self._instruction = split.data.manifest.spec.instruction
        return score(base_url, self._examples, instruction=self._instruction, transport=self.transport,
                     model=model_version)


def metrics_of(scores: Mapping[str, Any]) -> dict[str, float | int]:
    """The ledger metrics of a score: spot.score and every dotted key."""
    return {f"{PREFIX}.score": scores["score"], **{k: v for k, v in scores.items() if k.startswith(f"{PREFIX}.")}}


# --- the verb -----------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab spot-score", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    split = sub.add_parser("split", help="join spot-qa-train and spot-qa-heldout into one folder with split.json")
    split.add_argument("root", type=Path, help="the folder contexts.spot.dataset --out wrote")
    split.add_argument("--out", type=Path, help="default: <root>/spot-qa")
    run = sub.add_parser("score", help="score the model behind --base-url on the held-out side")
    run.add_argument("export", type=Path)
    run.add_argument("--base-url", required=True, help="an OpenAI-compatible server, e.g. http://127.0.0.1:8080")
    run.add_argument("--runs", type=int, default=1, help="score this many times (2 for run-to-run noise)")
    run.add_argument("--json", type=Path, help="write the last run's scores here, as contexts/agent/eval.py does")
    run.add_argument("--node", help="where the model ran: jetson, pod-a, ...")
    run.add_argument("--commit", help="the scorer's commit, where this is not a git checkout")
    run.add_argument("--model", help="the model's name for the row (default: the id the server gives)")
    run.add_argument("--min-held-out", type=int, default=MIN_HELD_OUT,
                     help="read a shorter held-out side to report it; it never meets the deploy floor")
    return parser


def _split(args: argparse.Namespace) -> dict[str, Any]:
    try:
        joined = join(args.root, args.out)
    except SpotError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc)}
    except DatasetError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc), "problems": exc.problems}
    record("spot-split", "ok", refs={"split_sha256": joined["sha256"]}, task=TASK,
           note=f"dataset {DATASET_ID}: {joined['train']} train, {joined['held_out']} held out")
    return {"ok": True, "outcome": "ok", **joined}


def _score(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> dict[str, Any]:
    if args.runs < 1:
        return {"ok": False, "outcome": "refused", "error": "--runs must be at least 1"}
    commit = args.commit or repo_commit()
    if not commit:
        return {"ok": False, "outcome": "no_commit",
                "error": "the scorer's commit is unknown: run from a git checkout or pass --commit"}
    try:
        split = dataset.load_split(args.export, require_images=False, min_held_out=args.min_held_out)
        examples = held_out_examples(split)
    except DatasetError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc), "problems": exc.problems}
    except SpotError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc)}
    instruction = split.data.manifest.spec.instruction
    conditions = {"decode": DECODE, "prompt_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                  "dataset_id": split.dataset_id, "meets_deploy_floor": split.meets_deploy_floor}
    runs = []
    for _ in range(args.runs):
        try:
            scores = score(args.base_url, examples, instruction=instruction, transport=transport, model=args.model)
        except SpotError as exc:
            return {"ok": False, "outcome": "error", "error": str(exc), "runs": runs}
        run_id = f"spot-{uuid.uuid4().hex[:12]}"
        extra = {"node": args.node} if args.node else {}
        record("spot-score", "ok", metrics_of(scores),
               {"split_sha256": split.sha256, "commit": commit, "model": scores["model"], "run_id": run_id},
               task=TASK, conditions=conditions, **extra)
        runs.append({"run_id": run_id, "scores": scores})
        if args.json:
            args.json.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    tops = [r["scores"]["score"] for r in runs]
    last = runs[-1]["scores"]
    return {"ok": True, "outcome": "ok", "scores": {k: v for k, v in last.items() if k != "results"},
            "runs": [{"run_id": r["run_id"], "score": r["scores"]["score"]} for r in runs],
            "noise": max(tops) - min(tops) if len(tops) > 1 else None,
            "split_sha256": split.sha256, "commit": commit, "meets_deploy_floor": split.meets_deploy_floor}


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Score a model on the held-out SPOT questions (L26): split joins the set's two folders, score scores."""
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    return _split(args) if args.action == "split" else _score(args, transport)


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)

CLI_VERB = "spot-score"
