"""The eval gate (L4): score the current model, the candidate and the teacher on the frozen held-out set, then deploy
or reject the candidate.

    python -m lab eval-gate --dataset fixtures/learning/tool-calling-fixture \\
        --current-url http://127.0.0.1:8080 --candidate-url http://127.0.0.1:8081 --teacher-url http://127.0.0.1:8000 \\
        --candidate-manifest data/models/atlas-v3-tool-calling/manifest.json --run-id run-3 \\
        [--current-runs 2] [--preview-url URL] [--episodes FILE] [--events-dir data/events] [--decode-attested]

    from lab.eval_gate import run_gate
    result = run_gate(client=FakeClient(...), dataset=..., ...)   # lab/eval_gate_fake.py for tests

What is scored. The held-out examples are the ones the dataset's frozen split.json lists as held_out, read through
lab.dataset.load_split() and nothing else; its sha256 goes into every ledger row and the saved scores. The scorer is
`python -m contexts.agent.eval --runtime llama --base-url <server> --json out.json`, run once per score through the
client's score(base_url). Its per-question results are matched to held-out examples by example id when a result
carries one (an image task), else by the question text folded the way the split folds it. A result that matches a
training example stops the gate (the scorer would be grading on what the candidate trained on); a result that
matches no example of the split is counted as outside_split and ignored; a held-out example that no result matches
is counted as unscored and left out. An item passes when the scorer's own Result.passed says so.

How. One scorer commit for every score (the client's scorer_commit(), asked once) and one decode setting, DECODE:
thinking off, greedy decoding and one fixed maximum output length (512 tokens, what runtime/llama.py sends for
chat). The client's decode(base_url) reads a server's setting; a server that reports another setting stops the gate,
and one whose setting cannot be read (None) is accepted only with decode_attested, which the row records. The
current model is scored at least MIN_CURRENT_RUNS times; the spread of its scores (max minus min) is the run-to-run
noise. The candidate must be the packaged Q4_K_M GGUF (its ModelManifest.gguf, from L5); a bf16 score of it may be
added with preview_url and is kept under the preview key "preview.candidate_bf16", never in the decision.

The decision. The items are those every current run, the candidate and the teacher scored. The unit of the paired
test is the question group: an example's frame_ref, or for an image example its episode (from the episodes map,
frame_ref -> episode id; an image example with no episode stops the gate). Each item's current value is its mean
over the current runs; each group's paired difference is the mean of (candidate - current) over its items. The test
is the exact two-sided sign-flip test over the groups' differences (sign_flip_p), which is the exact McNemar test
when every group holds one item. Deploy only when the held-out set holds at least MIN_ITEMS items (the split's
deploy floor and the scored items both), the candidate's score beats the current one's by more than the noise, and
the p-value is below ALPHA; otherwise reject, with every reason listed. A set under MIN_ITEMS items is scored and
reported, never deployed.

What it leaves. One ledger row per score (verb eval-gate.score) and one for the decision (verb eval-gate, outcome
deploy, reject or refused), each with the decode setting, the scorer commit and the split sha256 in its conditions;
the decision row's metrics hold the three scores, n as items and as groups, the noise and the p-value. A
learning.run.evaluated event (contracts/learning.py) with the scores and the decision, its model_version the
candidate's (runtime.registry.model_version of its manifest), written to <out>/learning.run.evaluated.jsonl and,
with events_dir, published to that event log. <out>/scores.json for the demo slide, and <out>/<role>.json per score
in the shape lab/slide_numbers.py reads (score is the held-out score; scorer_score is the scorer's own top line).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from bus import EventBus
from contexts.agent.eval import Result
from contracts.learning import DatasetExample, ModelManifest, RunEvaluated
from lab.dataset import MIN_HELD_OUT, DatasetError, Split, _question_key, load_split
from lab.machine.record import record
from runtime.registry import load_models_config, model_version

__all__ = ["DECODE", "MIN_ITEMS", "ALPHA", "MIN_CURRENT_RUNS", "PACKAGED_QUANT", "PREVIEW_KEY", "SCORES_FILE",
           "EVENT_FILE", "GateError", "Client", "RealClient", "make_client", "sign_flip_p", "run_gate", "main"]

CLI_VERB = "eval-gate"   # registers python -m lab eval-gate
REPO_ROOT = Path(__file__).resolve().parents[1]
DECODE: dict[str, Any] = {"thinking": "off", "sampling": "greedy", "max_tokens": 512}
MIN_ITEMS = MIN_HELD_OUT       # held-out items a deploy needs (the owner's ruling, the same floor as the split's)
ALPHA = 0.05
MIN_CURRENT_RUNS = 2           # scores of the current model, for its run-to-run noise
PACKAGED_QUANT = "Q4_K_M"      # what L5 packages and the box serves
PREVIEW_KEY = "preview.candidate_bf16"
SCORES_FILE = "scores.json"
EVENT_FILE = "learning.run.evaluated.jsonl"
SCORER_TIMEOUT_S = 3600
SCORER_PATHS = ("contexts", "runtime", "fixtures/golden", "config")   # what a scorer commit must hold unchanged


class GateError(RuntimeError):
    """The gate cannot score or decide fairly; str() says why. The verb answers outcome refused."""


class Client(Protocol):
    def score(self, base_url: str, *, model_version: str | None = None) -> dict[str, Any]:
        """The scorer's --json output for the server at base_url."""

    def decode(self, base_url: str) -> dict[str, Any] | None:
        """The server's decode setting in DECODE's shape, or None when it cannot be read."""

    def scorer_commit(self) -> str:
        """The commit the scorer runs at."""


class RealClient:
    """Runs Dominick's scorer as a subprocess of this checkout. Only the real run uses it; no test calls a server."""

    def command(self, base_url: str, out: Path, *, model_version: str | None = None) -> list[str]:
        return [sys.executable, "-m", "contexts.agent.eval", "--runtime", "llama", "--base-url", base_url,
                "--model-version", model_version or "atlas-base", "--json", str(out)]

    def score(self, base_url: str, *, model_version: str | None = None) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="eval-gate-") as folder:
            out = Path(folder) / "out.json"
            done = subprocess.run(self.command(base_url, out, model_version=model_version), cwd=REPO_ROOT,
                                  stdout=sys.stderr, timeout=SCORER_TIMEOUT_S, check=False)
            if done.returncode not in (0, 1) or not out.exists():   # 1 only means "not every question passed"
                raise GateError(f"the scorer exited {done.returncode} for {base_url} and wrote no results")
            return json.loads(out.read_text(encoding="utf-8"))

    def decode(self, base_url: str) -> dict[str, Any] | None:
        return None   # neither llama-server nor vLLM reports all three settings; the operator attests them

    def scorer_commit(self) -> str:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        if git("status", "--porcelain", "--", *SCORER_PATHS):
            raise GateError(f"uncommitted changes under {', '.join(SCORER_PATHS)}: the scorer must run at a commit")
        return git("rev-parse", "HEAD")


def make_client() -> Client:
    return RealClient()


# --- the paired test ------------------------------------------------------------------------------------------------

def sign_flip_p(diffs: Sequence[Fraction | int]) -> Fraction:
    """Exact two-sided sign-flip p-value: the share of the 2**n sign patterns whose |sum| is at least the observed one.

    Exact, in integers: the differences are scaled to a common denominator and the null distribution of the sum is
    counted by dynamic programming. With differences of +1, -1 and 0 it is the exact McNemar test."""
    values = [Fraction(d) for d in diffs]
    if not values:
        return Fraction(1)
    scale = math.lcm(*(v.denominator for v in values))
    scaled = [int(v * scale) for v in values]
    observed = abs(sum(scaled))
    counts: dict[int, int] = {0: 1}
    for x in scaled:
        nxt: dict[int, int] = defaultdict(int)
        for total, ways in counts.items():
            nxt[total + x] += ways
            nxt[total - x] += ways
        counts = nxt
    hits = sum(ways for total, ways in counts.items() if abs(total) >= observed)
    return Fraction(hits, 2 ** len(values))


# --- matching the scorer's results to the split ---------------------------------------------------------------------

def _passed(row: Mapping[str, Any]) -> bool:
    return Result.model_validate({k: row[k] for k in Result.model_fields if k in row}).passed


def _match(split: Split, output: Mapping[str, Any], role: str) -> tuple[dict[str, Mapping[str, Any]], int]:
    """{held-out example id: its result} and how many results fall outside the split. GateError on a leak."""
    held_out = {e.id for e in split.held_out}
    by_id = {e.id: e for e in split.data.examples}
    by_question: dict[str, list[DatasetExample]] = defaultdict(list)
    for example in split.data.examples:
        if not example.image:
            by_question[_question_key(example.context)].append(example)
    found: dict[str, Mapping[str, Any]] = {}
    outside = 0
    for row in output.get("results", []):
        if row.get("id") is not None:
            example = by_id.get(row["id"])
        else:
            hits = by_question.get(_question_key(row.get("q")), [])
            if len(hits) > 1:
                raise GateError(f"{role}: a scored question matches {len(hits)} examples of the split")
            example = hits[0] if hits else None
        if example is None:
            outside += 1
        elif example.id not in held_out:
            raise GateError(f"{role}: the scorer graded training example {example.id!r}; the gate scores only "
                            "the held-out side of split.json")
        elif example.id in found:
            raise GateError(f"{role}: the scorer graded held-out example {example.id!r} twice")
        else:
            found[example.id] = row
    return found, outside


def _groups(split: Split, episodes: Mapping[str, str] | None) -> dict[str, str]:
    """{held-out example id: its group}: the frame_ref, or for an image example its episode."""
    groups = {}
    for example in split.held_out:
        if not example.image:
            groups[example.id] = example.frame_ref
        elif episodes is None or example.frame_ref not in episodes:
            raise GateError(f"image example {example.id!r} has no episode; pass an episode map (frame_ref -> "
                            "episode id) so the test unit is the episode, never the single item")
        else:
            groups[example.id] = f"episode:{episodes[example.frame_ref]}"
    return groups


def _mean(values: Sequence[Fraction | int | float]) -> Fraction:
    return Fraction(sum(Fraction(v) for v in values), len(values)) if values else Fraction(0)


# --- the gate ---------------------------------------------------------------------------------------------------------

def _candidate(path: str | os.PathLike[str]) -> tuple[str, ModelManifest]:
    try:
        manifest = ModelManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"cannot read the candidate's package manifest: {type(exc).__name__}") from None
    if PACKAGED_QUANT not in manifest.gguf.upper():
        raise GateError(f"the candidate must be the packaged {PACKAGED_QUANT} GGUF the box serves, not "
                        f"{manifest.gguf!r}; a bf16 score goes in preview_url")
    return model_version({"manifest": str(path)}).version, manifest


def _current_version() -> str:
    try:
        return model_version(load_models_config(REPO_ROOT / "config" / "models.yaml").get("atlas_model", {})).version
    except (OSError, ValueError):
        return "atlas-base"


def _write_json(path: Path, body: Any) -> None:
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_gate(*, client: Client, dataset: str | os.PathLike[str], current_url: str, candidate_url: str,
             teacher_url: str, candidate_manifest: str | os.PathLike[str], run_id: str,
             out_dir: str | os.PathLike[str], current_version: str | None = None,
             current_runs: int = MIN_CURRENT_RUNS, preview_url: str | None = None,
             episodes: str | os.PathLike[str] | None = None, events_dir: str | os.PathLike[str] | None = None,
             decode_attested: bool = False, min_held_out: int = MIN_ITEMS, task: str | None = None) -> dict[str, Any]:
    """Score, test and decide (see the module docstring). GateError when it cannot do so fairly."""
    if current_runs < MIN_CURRENT_RUNS:
        raise GateError(f"the current model must be scored at least twice (asked for {current_runs}) to measure "
                        "its run-to-run noise")
    try:
        split = load_split(dataset, require_images=False, min_held_out=min_held_out)
    except DatasetError as err:
        raise GateError(f"the frozen split does not hold up: {err}") from None
    candidate_version, manifest = _candidate(candidate_manifest)
    current_version = current_version or _current_version()
    episode_map = json.loads(Path(episodes).read_text(encoding="utf-8")) if episodes else None
    groups = _groups(split, episode_map)
    task = task or split.data.manifest.spec.task
    commit = client.scorer_commit()

    roles = [(f"current-{i}", "current", i, current_url, current_version) for i in range(1, current_runs + 1)]
    roles += [("candidate", "candidate", 1, candidate_url, candidate_version),
              ("teacher", "teacher", 1, teacher_url, "teacher")]
    if preview_url:
        roles.append(("preview", "preview", 1, preview_url, f"{candidate_version}-bf16"))
    decode_source: dict[str, str] = {}
    for _, role, _, url, _ in roles:
        setting = client.decode(url)
        if setting is None:
            if not decode_attested:
                raise GateError(f"the {role} server's decode setting cannot be read; launch it as "
                                "lab/L4_REAL_RUN.md says and attest it (--decode-attested)")
            decode_source[role] = "attested"
        elif dict(setting) != DECODE:
            raise GateError(f"the {role} server decodes with {dict(setting)}, not the one decode setting {DECODE}")
        else:
            decode_source[role] = "reported"
    base = {"decode": dict(DECODE), "scorer_commit": commit, "split_sha256": split.sha256,
            "dataset_id": split.dataset_id}
    refs = {"run_id": run_id, "commit": commit, "split_sha256": split.sha256}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored: dict[str, dict[str, Mapping[str, Any]]] = {}
    raw: dict[str, Mapping[str, Any]] = {}
    outside = 0
    for name, role, _, url, version in roles:
        output = client.score(url, model_version=version)
        scored[name], outside_here = _match(split, output, name)
        raw[name] = output
        outside = max(outside, outside_here)

    decisive = [name for name, role, *_ in roles if role != "preview"]
    common = [e.id for e in split.held_out if all(e.id in scored[name] for name in decisive)]
    if not common:
        raise GateError("no held-out example was scored by every model")

    def score_of(name: str, ids: Sequence[str]) -> Fraction:
        return _mean([_passed(scored[name][i]) for i in ids])

    def detail(name: str, ids: Sequence[str]) -> dict[str, float]:
        rows = [scored[name][i] for i in ids]
        return {"citations_ok": float(_mean([bool(r.get("citations_ok")) for r in rows])),
                "answer_ok": float(_mean([r.get("answer_ok") is True for r in rows])),
                "scorer_score": float(raw[name].get("score", 0.0))}

    current_names = [name for name, role, *_ in roles if role == "current"]
    run_scores = [score_of(name, common) for name in current_names]
    current = _mean(run_scores)
    candidate = score_of("candidate", common)
    teacher = score_of("teacher", common)
    noise = max(run_scores) - min(run_scores)
    delta = candidate - current

    by_group: dict[str, list[Fraction]] = defaultdict(list)
    for i in common:
        now = _mean([_passed(scored[name][i]) for name in current_names])
        by_group[groups[i]].append(Fraction(int(_passed(scored["candidate"][i]))) - now)
    p_value = sign_flip_p([_mean(diffs) for diffs in by_group.values()])
    n_items, n_groups = len(common), len(by_group)

    reasons = []
    if n_items < MIN_ITEMS or not split.meets_deploy_floor:
        reasons.append(f"the held-out set has {n_items} scored items ({len(split.held_out)} in split.json), under "
                       f"the {MIN_ITEMS} a deploy needs: scored and reported only")
    if delta <= noise:
        reasons.append(f"the candidate beats current by {float(delta):.4f}, not more than the run-to-run noise "
                       f"{float(noise):.4f}")
    if p_value >= ALPHA:
        reasons.append(f"the paired p-value {float(p_value):.4f} over {n_groups} groups is not below {ALPHA}")
    decision = "reject" if reasons else "deploy"

    scores: dict[str, float] = {"current": float(current), "candidate": float(candidate), "teacher": float(teacher)}
    for name, role, attempt, *_ in roles:
        if role == "preview":
            continue
        prefix = role if role != "current" else f"current.run_{attempt}"
        for key, value in detail(name, common).items():
            scores[f"{prefix}.{key}"] = value
        if role == "current":
            scores[f"current.run_{attempt}"] = float(run_scores[attempt - 1])
    preview_ids: list[str] = []
    if preview_url:
        preview_ids = [i for i in common if i in scored["preview"]]
        scores[PREVIEW_KEY] = float(score_of("preview", preview_ids)) if preview_ids else 0.0
    scores.update({"gate.noise": float(noise), "gate.p_value": float(p_value), "gate.delta": float(delta),
                   "gate.n_items": float(n_items), "gate.n_groups": float(n_groups)})

    for name, role, attempt, url, version in roles:
        ids = preview_ids if role == "preview" else common
        value = float(score_of(name, ids)) if ids else 0.0
        _write_json(out / f"{name}.json", {"score": value, "model": version, "role": role, "held_out_items": len(ids),
                                           "scorer_score": raw[name].get("score"), "split_sha256": split.sha256,
                                           "results": list(raw[name].get("results", []))})
        metrics = {"eval.score": value, "eval.n_items": len(ids), "eval.n_groups": len({groups[i] for i in ids}),
                   **{f"eval.{k}": v for k, v in detail(name, ids).items()}} if ids else {"eval.n_items": 0}
        record("eval-gate.score", "ok", metrics, refs, source="contexts.agent.eval",
               conditions={**base, "role": role, "attempt": attempt, "base_url": url, "model_version": version,
                           "decode_source": decode_source[role]})

    event = RunEvaluated(task=task, run_id=run_id, scores=scores, decision=decision, model_version=candidate_version)
    (out / EVENT_FILE).write_text(event.model_dump_json() + "\n", encoding="utf-8", newline="\n")
    if events_dir is not None:
        asyncio.run(EventBus(log_dir=events_dir).publish(event))

    result: dict[str, Any] = {
        "ok": True, "outcome": decision, "decision": decision, "reasons": reasons, "run_id": run_id, "task": task,
        "model_version": candidate_version, "current_version": current_version, "candidate_gguf": manifest.gguf,
        "scores": scores, "noise": float(noise), "delta": float(delta), "p_value": float(p_value),
        "n_items": n_items, "n_groups": n_groups, "held_out": len(split.held_out),
        "unscored": len(split.held_out) - n_items, "outside_split": outside, "split_sha256": split.sha256,
        "dataset_id": split.dataset_id, "scorer_commit": commit, "decode": dict(DECODE),
        "decode_source": decode_source, "current_runs": [float(s) for s in run_scores],
        "event": json.loads(event.model_dump_json()), "out": str(out),
    }
    _write_json(out / SCORES_FILE, {k: v for k, v in result.items() if k not in ("ok", "outcome", "out")})
    record("eval-gate", decision,
           {"eval.current": float(current), "eval.candidate": float(candidate), "eval.teacher": float(teacher),
            "eval.noise": float(noise), "eval.delta": float(delta), "eval.p_value": float(p_value),
            "eval.n_items": n_items, "eval.n_groups": n_groups, "eval.unscored": result["unscored"]},
           refs, conditions={**base, "decode_source": decode_source}, source="contexts.agent.eval",
           note="; ".join(reasons) or None)
    return result


# --- the verb --------------------------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Score current, candidate and teacher on the frozen held-out set and decide deploy or reject (L4)."""
    parser = argparse.ArgumentParser(prog="python -m lab eval-gate", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", required=True, help="the dataset folder whose split.json is frozen")
    parser.add_argument("--current-url", required=True, help="llama-server serving the current Jetson model")
    parser.add_argument("--candidate-url", required=True, help="llama-server serving the packaged Q4_K_M candidate")
    parser.add_argument("--teacher-url", required=True, help="the teacher's OpenAI-compatible server")
    parser.add_argument("--candidate-manifest", required=True, help="the L5 package's manifest.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--current-version", help="default: config/models.yaml through runtime.registry")
    parser.add_argument("--current-runs", type=int, default=MIN_CURRENT_RUNS)
    parser.add_argument("--preview-url", help="a bf16 server of the candidate; recorded, never decides")
    parser.add_argument("--episodes", help="JSON {frame_ref: episode id}, for image tasks")
    parser.add_argument("--events-dir", help="also publish learning.run.evaluated to this event log")
    parser.add_argument("--decode-attested", action="store_true",
                        help="the servers were launched with the one decode setting (lab/L4_REAL_RUN.md)")
    parser.add_argument("--min-held-out", type=int, default=MIN_ITEMS,
                        help="read a shorter held-out set to report it; it never deploys")
    parser.add_argument("--task", help="default: the dataset manifest's task")
    parser.add_argument("--out", help="default: lab/state/eval-gate/<run id>")
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else REPO_ROOT / "lab" / "state" / "eval-gate" / args.run_id
    try:
        return run_gate(client=make_client(), dataset=args.dataset, current_url=args.current_url,
                        candidate_url=args.candidate_url, teacher_url=args.teacher_url,
                        candidate_manifest=args.candidate_manifest, run_id=args.run_id, out_dir=out,
                        current_version=args.current_version, current_runs=args.current_runs,
                        preview_url=args.preview_url, episodes=args.episodes, events_dir=args.events_dir,
                        decode_attested=args.decode_attested, min_held_out=args.min_held_out, task=args.task)
    except GateError as err:
        record("eval-gate", "refused", refs={"run_id": args.run_id}, note=str(err)[:500],
               conditions={"decode": dict(DECODE)})
        return {"ok": False, "outcome": "refused", "error": str(err), "run_id": args.run_id}
