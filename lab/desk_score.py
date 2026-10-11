"""Score a model on the held-out desk-object examples (L24): python -m lab desk-score split|score.

    python -m lab desk-score split data/exports/<dataset_id> [--held-out 50]
    python -m lab desk-score score data/exports/<dataset_id> --base-url http://127.0.0.1:8080 \\
        [--runs 2] [--json out.json] [--node jetson] [--commit <40 hex>] [--model <name>] [--min-held-out 50]

The export is a desk-object dataset from the box, read the way lab/dataset.py reads every export: manifest.json,
examples.jsonl (one DatasetExample per line: id, image, frame_ref, ts, context, target) and the blurred keyframes the
examples point at. A desk example is the picture, its frame_ref, the box's caption (context) and the object label the
box should name (target); DeskExample holds those fields.

split writes the frozen split.json once, by frame_ref group, in the format lab/dataset.py load_split() reads: groups in
the order of the sha256 of their frame_ref, held out until the held-out side holds at least --held-out examples, the
rest train. It refuses a folder that already has a split.json (a new split starts a new baseline), and records the
file's sha256 in a ledger row (verb desk-split).

score reads the held-out side only from that split.json (load_split, so a leaky or short split is refused before a
request is sent) and sends one chat request per held-out example, the way the box serves its model: the picture as a
data URL and the export's instruction as the one user message, to llama-server's /v1/chat/completions with DECODE
(thinking off, greedy, one seed, MAX_TOKENS). The answer and the label are normalised the same way (normalize: case,
punctuation, the articles a, an and the, the plural of the last word) and compared exactly. The answer is:

    {"score": <match rate>, "model": <served model id>, "results": [{id, frame_ref, label, answer, match}, ...],
     "desk.<object>": <match rate of that object>, ..., "desk.n_items": <items>, "desk.n_groups": <frame_refs>}

the shape contexts/agent/eval.py writes ({"score", "model", "results"}) with the per-metric detail as dotted keys
beside the top-line score, so the eval gate (L4) reads it as it reads that scorer's file. Every result keeps its
frame_ref, so the gate's paired test takes its unit as the group, never the single item. score(base_url, examples)
is the client interface for the gate; no gate module exists on origin yet, so this is the interface defined here.

Each run writes one ledger row (verb desk-score, task L24): metrics desk.score, desk.<object>, desk.n_items and
desk.n_groups; refs split_sha256, commit (the scorer's: --commit, else git rev-parse HEAD of this checkout; a run
with neither is refused), model and run_id; conditions with the decode setting, the prompt's sha256, the dataset id
and whether the held-out side meets the gate's deploy floor. --runs 2 scores twice for run-to-run noise and answers
the spread as noise. A server that answers with an error stops the run: a failed request is never counted as a miss.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if not __package__:  # run by path on a pod: the repo root, one folder up, is what holds contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from lab import dataset
from lab.dataset import MIN_HELD_OUT, DatasetError, Split
from lab.machine.metrics import _METRIC_RE
from lab.machine.record import record

__all__ = ["TASK", "SEED", "MAX_TOKENS", "DECODE", "RESERVED", "DeskError", "DeskExample", "normalize", "is_match",
           "slug", "write_split", "load_desk_split", "held_out_examples", "DeskClient", "score", "repo_commit",
           "main"]

CLI_VERB = "desk-score"   # registers python -m lab desk-score
TASK = "L24"
SEED = 0
MAX_TOKENS = 24            # an object's name is a few tokens; a longer answer is cut, and then it does not match
DECODE: dict[str, Any] = {  # one decode setting for every model scored, written into every row
    "temperature": 0, "top_k": 1, "seed": SEED, "max_tokens": MAX_TOKENS,
    "chat_template_kwargs": {"enable_thinking": False},
}
PREFIX = "desk"
RESERVED = ("score", "n_items", "n_groups")   # desk.<name> keys that are not an object
TIMEOUT_S = 120.0
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
SPLIT_RULE = ("Held out: whole frame_ref groups, taken in the order of the sha256 of their frame_ref until the held-out "
              "side holds at least {n} examples; train: every other group. No frame_ref is on both sides. Written "
              "once by python -m lab desk-score split; a new split starts a new baseline.")

ARTICLES = frozenset({"a", "an", "the"})
_THINK = re.compile(r"<think>.*?</think>", re.S)
_NOT_WORD = re.compile(r"[^a-z0-9\s]")


class DeskError(RuntimeError):
    """The desk scorer cannot go on: a split that exists already, a label missing, a server error."""


@dataclass(frozen=True)
class DeskExample:
    """One held-out desk example: the box's keyframe and the object label it should name."""

    id: str
    frame_ref: str
    image: Path      # the blurred keyframe, inside the export folder
    caption: str     # the box's own caption (the export's context); not sent to the model
    label: str       # the object the box should name (the export's target)


# --- normalisation ------------------------------------------------------------------------

def _singular(word: str) -> str:
    """The singular of an English noun by its ending, enough for object names: batteries, boxes, glasses, keys."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if len(word) > 1 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize(text: str) -> str:
    """An answer or a label as compared: no thinking block, lowercase, punctuation as spaces, the articles dropped,
    spacing folded and the last word singular. "The Keys." and "a key" are both "key"."""
    words = [w for w in _NOT_WORD.sub(" ", _THINK.sub(" ", text or "").lower()).split() if w not in ARTICLES]
    if words:
        words[-1] = _singular(words[-1])
    return " ".join(words)


def is_match(answer: str, label: str) -> bool:
    """Exact match after normalize(); an empty label matches nothing."""
    want = normalize(label)
    return bool(want) and normalize(answer) == want


def slug(normal_label: str) -> str:
    """A normalised label as the last part of a dotted metric name: phone charger is phone_charger."""
    name = re.sub(r"[^a-z0-9]+", "_", normal_label).strip("_")
    return name if name[:1].isalpha() else f"o_{name}"


# --- the frozen split ---------------------------------------------------------------------

def _group_order(frame_ref: str) -> str:
    return hashlib.sha256(frame_ref.encode("utf-8")).hexdigest()


def write_split(root: str | Path, *, held_out: int = MIN_HELD_OUT) -> dict[str, Any]:
    """Write split.json for the export once, by frame_ref group (see the module docstring); its ids and sha256."""
    root = Path(root)
    path = root / dataset.SPLIT
    if path.exists():
        raise DeskError(f"{dataset.SPLIT} already exists in {root.name}; the split is written once")
    data = dataset.load(root)
    groups: dict[str, list[str]] = {}
    for example in data.examples:
        if not example.frame_ref.strip():
            raise DeskError(f"example {example.id!r} has no frame_ref, so it cannot be kept on one side")
        groups.setdefault(example.frame_ref, []).append(example.id)
    held: list[str] = []
    train: list[str] = []
    for ref in sorted(groups, key=_group_order):
        (held if len(held) < held_out else train).extend(sorted(groups[ref]))
    if not train:
        raise DeskError(f"holding out {held_out} examples leaves nothing to train on ({len(held)} in all)")
    body = {"version": 1, "dataset_id": data.dataset_id, "rule": SPLIT_RULE.format(n=held_out),
            "train": train, "held_out": held}
    text = json.dumps(body, indent=2) + "\n"
    with open(path, "x", encoding="utf-8", newline="\n") as handle:   # "x": never over a split written meanwhile
        handle.write(text)
    return {"dataset_id": data.dataset_id, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "train": train, "held_out": held}


def load_desk_split(root: str | Path, *, min_held_out: int = MIN_HELD_OUT) -> Split:
    """The export's frozen split, as lab/dataset.py load_split() reads and checks it (pictures required)."""
    return dataset.load_split(root, require_images=True, min_held_out=min_held_out)


def held_out_examples(split: Split) -> list[DeskExample]:
    """The held-out side as desk examples, in split.json order. DeskError when one has no label."""
    bare = [e.id for e in split.held_out if not dataset.has_target(e)]
    if bare:
        raise DeskError(f"held-out examples with no label ({len(bare)}): {', '.join(repr(b) for b in bare[:5])}")
    return [DeskExample(e.id, e.frame_ref, split.data.image_path(e), e.context or "", (e.target or "").strip())
            for e in split.held_out]


# --- the client ---------------------------------------------------------------------------

def _image_url(path: Path) -> str:
    mime = MIME.get(path.suffix.lower())
    if mime is None:
        raise DeskError(f"{path.name}: not a picture the scorer sends ({', '.join(MIME)})")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def request_body(example: DeskExample, prompt: str) -> dict[str, Any]:
    """The one chat request for an example: its picture and the prompt as one user message, with DECODE."""
    content = [{"type": "image_url", "image_url": {"url": _image_url(example.image)}},
               {"type": "text", "text": prompt}]
    return {"messages": [{"role": "user", "content": content}], "stream": False, **json.loads(json.dumps(DECODE))}


class DeskClient:
    """llama-server's chat API, as much as the scorer uses. transport replaces the network in tests."""

    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None,
                 timeout_s: float = TIMEOUT_S) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), transport=transport, timeout=timeout_s)

    def close(self) -> None:
        self.http.close()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.http.post(path, json=body)
        except httpx.HTTPError as exc:
            raise DeskError(f"{path}: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            raise DeskError(f"{path}: HTTP {response.status_code}")
        return response.json()

    def model(self) -> str:
        """The id the server names its model by (GET /v1/models, the first entry)."""
        try:
            response = self.http.get("/v1/models")
        except httpx.HTTPError as exc:
            raise DeskError(f"/v1/models: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            raise DeskError(f"/v1/models: HTTP {response.status_code}")
        data = response.json().get("data") or []
        return str(data[0].get("id", "")) if data else ""

    def ask(self, example: DeskExample, prompt: str) -> str:
        reply = self._post("/v1/chat/completions", request_body(example, prompt))
        try:
            return reply["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise DeskError(f"/v1/chat/completions: an answer without choices[0].message ({exc})") from exc


# --- the score ----------------------------------------------------------------------------

def score(base_url: str, examples: Sequence[DeskExample], *, prompt: str,
          transport: httpx.BaseTransport | None = None, model: str | None = None) -> dict[str, Any]:
    """Score the model behind base_url on the examples: the eval.py shape with dotted per-object keys."""
    by_object: dict[str, list[bool]] = {}
    for example in examples:
        name = slug(normalize(example.label)) if normalize(example.label) else ""
        if name in RESERVED:
            raise DeskError(f"object label {example.label!r} would be the reserved key {PREFIX}.{name}")
        if name and not _METRIC_RE.fullmatch(f"{PREFIX}.{name}"):
            raise DeskError(f"object label {example.label!r} makes no metric name")
    client = DeskClient(base_url, transport=transport)
    try:
        served = client.model()
        results = []
        for example in examples:
            answer = client.ask(example, prompt)
            match = is_match(answer, example.label)
            results.append({"id": example.id, "frame_ref": example.frame_ref, "label": example.label,
                            "answer": answer, "match": match})
            by_object.setdefault(slug(normalize(example.label)), []).append(match)
    finally:
        client.close()
    hits = [r["match"] for r in results]
    out: dict[str, Any] = {"score": sum(hits) / len(hits) if hits else 0.0, "model": model or served,
                           "results": results}
    for name in sorted(by_object):
        out[f"{PREFIX}.{name}"] = sum(by_object[name]) / len(by_object[name])
    out[f"{PREFIX}.n_items"] = len(results)
    out[f"{PREFIX}.n_groups"] = len({r["frame_ref"] for r in results})
    return out


def metrics_of(scores: dict[str, Any]) -> dict[str, float | int]:
    """The ledger metrics of a score: desk.score and every dotted key."""
    metrics: dict[str, float | int] = {f"{PREFIX}.score": scores["score"]}
    metrics.update({k: v for k, v in scores.items() if k.startswith(f"{PREFIX}.")})
    return metrics


def repo_commit() -> str | None:
    """HEAD of the checkout this file is in, or None outside a git checkout (a pod's copy of lab/)."""
    try:
        found = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = found.stdout.strip()
    return commit if found.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else None


# --- the verb -----------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab desk-score", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    split = sub.add_parser("split", help="write the export's frozen split.json once, by frame_ref group")
    split.add_argument("export", type=Path)
    split.add_argument("--held-out", type=int, default=MIN_HELD_OUT, help="examples to hold out, at least")
    run = sub.add_parser("score", help="score the model behind --base-url on the held-out side")
    run.add_argument("export", type=Path)
    run.add_argument("--base-url", required=True, help="llama-server, e.g. http://127.0.0.1:8080")
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
        written = write_split(args.export, held_out=args.held_out)
    except DeskError as exc:
        exists = (args.export / dataset.SPLIT).exists()
        return {"ok": False, "outcome": "exists" if exists else "refused", "error": str(exc)}
    except DatasetError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc), "problems": exc.problems}
    record("desk-split", "ok", refs={"split_sha256": written["sha256"]}, task=TASK,
           note=f"dataset {written['dataset_id']}: {len(written['train'])} train, {len(written['held_out'])} held out")
    return {"ok": True, "outcome": "ok", "dataset_id": written["dataset_id"], "sha256": written["sha256"],
            "train": len(written["train"]), "held_out": len(written["held_out"])}


def _score(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> dict[str, Any]:
    if args.runs < 1:
        return {"ok": False, "outcome": "refused", "error": "--runs must be at least 1"}
    commit = args.commit or repo_commit()
    if not commit:
        return {"ok": False, "outcome": "no_commit",
                "error": "the scorer's commit is unknown: run from a git checkout or pass --commit"}
    try:
        split = load_desk_split(args.export, min_held_out=args.min_held_out)
        examples = held_out_examples(split)
    except DatasetError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc), "problems": exc.problems}
    except DeskError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc)}
    prompt = split.data.manifest.spec.instruction
    conditions = {"decode": DECODE, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                  "dataset_id": split.dataset_id, "meets_deploy_floor": split.meets_deploy_floor}
    runs = []
    for _ in range(args.runs):
        try:
            scores = score(args.base_url, examples, prompt=prompt, transport=transport, model=args.model)
        except DeskError as exc:
            return {"ok": False, "outcome": "error", "error": str(exc), "runs": runs}
        run_id = f"desk-{uuid.uuid4().hex[:12]}"
        extra = {"node": args.node} if args.node else {}
        record("desk-score", "ok", metrics_of(scores),
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
    """Score a model on the held-out desk-object examples (L24): split writes the frozen split, score scores."""
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    return _split(args) if args.action == "split" else _score(args, transport)


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
