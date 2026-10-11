"""Tool-call agreement of a served student on the frozen split of a tool-calling dataset (L4, the second measure).

    python -m lab tool-call-agreement --dataset fixtures/learning/tool-calling-fixture --run-id <run id> \\
        --before-url http://127.0.0.1:8100 --after-url http://127.0.0.1:8101 --out <folder> \\
        [--side held_out|train] [--before-runs 2] [--after-runs 2] [--limit N]

The golden-question scorer the gate runs (contexts/agent/eval.py) asks questions that all take the agent's
prefetch path, so it never scores a tool call. This asks the question the fine-tune was trained on: given the
prompt the trainer gave the student for an example, does the student write the reference tool call?

The scoring rule is RULE below. It is written into every file this leaves, before any model is asked.

What is reused, not rewritten: the prompt is lab/format_traces.py render() (the text the trainer's rows hold up to
the first assistant span); the calls are read by its parse_tool_calls(); an example is right when
lab/routing_closed_loop.py agreement() says so against expected.jsonl; the conservative test is lab/eval_gate.py
sign_flip_p() over the frame_ref groups, as the gate pairs them.

What it leaves in --out: raw/<side>.<arm>.jsonl (a header line, then one line per example: id, group, the prompt's
sha256, the output text, the parsed calls, the reference calls, the three marks), <side>.summary.json (the gate's
scores.json fields where they fit: correct, noise_runs, alpha, test, items with id, group, question, expected,
before, after) and <side>.md. The verb also writes one ledger row (verb tool-call-agreement: the two rates, the
difference, the noise, both p-values, the flips, n as items and as groups; refs.run_id and the split's sha256).
Nothing is sent anywhere but the two servers, and nothing is imported into the box: the summary is a file.

The first real run (2026-10-11, the L16 adapter against the base rebuilt on pod D) is in
lab/docs/tool-call-agreement.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

if not __package__:   # run by path: the repo root, one folder up, is what holds contexts/, contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import format_traces as ft
from lab.dataset import load_split
from lab.eval_gate import ALPHA, sign_flip_p
from lab.fixture_dataset import Expected, load_expected
from lab.fixture_dataset_v2 import system_for
from lab.machine.record import record
from lab.routing_closed_loop import agreement

__all__ = ["RULE", "MAX_TOKENS", "TEMPERATURE", "Completer", "LlamaCompletion", "prompt_for", "score_output",
           "mcnemar_exact", "run_arm", "summarize", "run", "main"]

CLI_VERB = "tool-call-agreement"   # registers python -m lab tool-call-agreement
MAX_TOKENS = 512        # the gate's one output length (lab/eval_gate.py DECODE)
TEMPERATURE = 0.2       # what runtime/llama.py sends; greedy comes from the server's --top-k 1
TEST_MCNEMAR = "exact two-sided McNemar test on the paired examples (first before run against first after run)"
TEST_SIGN_FLIP = "exact two-sided sign-flip test over question groups"
RULE = (
    "Input: the training text of the example up to its first assistant span, that is lab/format_traces.py render() "
    "of [system = the dataset's instruction, user = the example's context] with the agent's TOOLS and the generation "
    "prompt (<|im_start|>assistant, a newline, the empty thinking block), sent unchanged as the prompt of "
    "llama-server POST /completion (n_predict 512, temperature 0.2 as the box's client sends, sampling left to the "
    "server's --top-k 1). Both arms get the identical prompt text. "
    "Output: the generated tokens as text (the server's /detokenize of the returned token ids when it returns them, "
    "else the content field), cut at the first <|im_end|>; nothing else is stripped or repaired. "
    "Calls: lab/format_traces.py parse_tool_calls() on that text with the same TOOLS; a FormatError is not parsed. "
    "Score 1 when lab/routing_closed_loop.py agreement() says so: the calls' names and argument objects equal the "
    "reference's (expected.jsonl label.tool_calls), in order and in number; else 0. "
    "Secondary marks: names (the list of tool names equals the reference's, in order) and parsed (the parser read "
    "at least one call). "
    "Pairing: the flip table and McNemar use the first before run and the first after run; the sign-flip test pairs "
    "each example's mean over the before runs with its mean over the after runs and tests the frame_ref groups' "
    "mean differences, as lab/eval_gate.py does. Noise is the spread (max minus min) of the before runs' rates."
)


class Completer(Protocol):
    def complete(self, prompt: str) -> dict[str, Any]:
        """{"text": what the model wrote, ...whatever else the server said worth keeping}."""


class LlamaCompletion:
    """llama-server's raw completion endpoint: the prompt text is tokenized as it stands, special tokens read."""

    def __init__(self, base_url: str, timeout_s: float = 180.0, transport: Any = None) -> None:
        import httpx

        self.base_url = base_url
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def props(self) -> dict[str, Any]:
        body = self.http.get("/props").raise_for_status().json()
        settings = body.get("default_generation_settings") or {}
        params = settings.get("params", settings)
        return {"model": str(body.get("model_path") or "").replace("\\", "/").rsplit("/", 1)[-1],
                "top_k": params.get("top_k"), "n_ctx": settings.get("n_ctx")}

    def complete(self, prompt: str) -> dict[str, Any]:
        reply = self.http.post("/completion", json={"prompt": prompt, "n_predict": MAX_TOKENS,
                                                    "temperature": TEMPERATURE, "return_tokens": True})
        body = reply.raise_for_status().json()
        tokens = body.get("tokens") or []
        if tokens:
            text = self.http.post("/detokenize", json={"tokens": tokens}).raise_for_status().json()["content"]
            source = "detokenize"
        else:
            text, source = body.get("content") or "", "content"
        return {"text": text, "text_source": source, "tokens_predicted": body.get("tokens_predicted"),
                "tokens_evaluated": body.get("tokens_evaluated"), "stop_type": body.get("stop_type"),
                "truncated": body.get("truncated")}


# --- one example ------------------------------------------------------------------------------------------------

def prompt_for(system: str, question: str, tools: Sequence[Mapping[str, Any]] = ft.TOOLS) -> str:
    """What the trainer's row holds before the first assistant span (lab/format_traces.py: text[:spans[0][0]])."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return ft.render(messages, tools, add_generation_prompt=True)


def _gold(expected: Expected) -> list[dict[str, Any]]:
    return [{"name": call.name, "arguments": call.arguments} for call in expected.label.tool_calls]


def score_output(output: str, expected: Expected, tools: Sequence[Mapping[str, Any]] = ft.TOOLS) -> dict[str, Any]:
    """The marks of one output under RULE: answer (the text scored), calls, problem, parsed, names, correct."""
    answer = output.split(ft.IM_END, 1)[0]
    problem = None
    try:
        made: list[dict[str, Any]] | None = [{"name": call.name, "arguments": call.arguments}
                                             for call in ft.parse_tool_calls(answer, tools)[1]]
    except ft.FormatError as why:
        made, problem = None, str(why)
    label = json.dumps({"tool_calls": made}) if made is not None else None
    gold = _gold(expected)
    return {"answer": answer, "calls": made, "problem": problem, "parsed": bool(made),
            "names": made is not None and [call["name"] for call in made] == [call["name"] for call in gold],
            "correct": agreement(label, expected) == 1.0}


# --- the tests --------------------------------------------------------------------------------------------------

def mcnemar_exact(gained: int, lost: int) -> Fraction:
    """Exact two-sided McNemar p for the discordant pairs (0 to 1: gained, 1 to 0: lost): the binomial test of
    min(gained, lost) successes in gained + lost fair trials, doubled and capped at 1. No discordant pair is p 1."""
    n = gained + lost
    if n == 0:
        return Fraction(1)
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1))
    return min(Fraction(1), Fraction(2 * tail, 2 ** n))


# --- an arm, and the summary ----------------------------------------------------------------------------------------

def run_arm(name: str, completer: Completer, examples: Sequence[Any], expected: Mapping[str, Expected],
            system: str | Callable[[Any], str],
            write: Callable[[dict[str, Any]], None] | None = None,
            clock: Callable[[], float] = time.monotonic) -> dict[str, dict[str, Any]]:
    """Ask every example once, in order; {example id: its row}. A server that fails stops the arm (never a 0).
    system is the dataset's one instruction, or a function giving each example's own (a dataset whose examples each
    have their own time now, lab/fixture_dataset_v2.py)."""
    rows: dict[str, dict[str, Any]] = {}
    for example in examples:
        prompt = prompt_for(system(example) if callable(system) else system, example.context or "")
        started = clock()
        reply = completer.complete(prompt)
        marks = score_output(str(reply.get("text") or ""), expected[example.id])
        row = {"arm": name, "id": example.id, "group": example.frame_ref, "question": example.context,
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "output": reply.get("text"),
               "calls": marks["calls"], "problem": marks["problem"], "expected": _gold(expected[example.id]),
               "correct": marks["correct"], "names": marks["names"], "parsed": marks["parsed"],
               "answer": marks["answer"], "seconds": round(clock() - started, 3),
               **{key: value for key, value in reply.items() if key != "text"}}
        rows[example.id] = row
        if write:
            write(row)
    return rows


def _rate(rows: Mapping[str, Mapping[str, Any]], ids: Sequence[str], mark: str = "correct") -> Fraction:
    return Fraction(sum(bool(rows[i][mark]) for i in ids), len(ids)) if ids else Fraction(0)


def summarize(arms: Mapping[str, Mapping[str, Mapping[str, Any]]], examples: Sequence[Any],
              expected: Mapping[str, Expected]) -> dict[str, Any]:
    """The numbers of one side from its arms' rows (arms are named before-<n> and after-<n>)."""
    ids = [example.id for example in examples]
    group = {example.id: example.frame_ref for example in examples}
    before = sorted(name for name in arms if name.startswith("before-"))
    after = sorted(name for name in arms if name.startswith("after-"))
    if not before or not after:
        raise ValueError("a summary needs at least one before arm and one after arm")
    rates = {name: _rate(arms[name], ids) for name in (*before, *after)}
    current = sum(rates[name] for name in before) / len(before)
    candidate = sum(rates[name] for name in after) / len(after)
    noise = max(rates[name] for name in before) - min(rates[name] for name in before)
    first_before, first_after = arms[before[0]], arms[after[0]]
    gained = [i for i in ids if not first_before[i]["correct"] and first_after[i]["correct"]]
    lost = [i for i in ids if first_before[i]["correct"] and not first_after[i]["correct"]]
    by_group: dict[str, list[Fraction]] = defaultdict(list)
    for i in ids:
        was = Fraction(sum(bool(arms[name][i]["correct"]) for name in before), len(before))
        now = Fraction(sum(bool(arms[name][i]["correct"]) for name in after), len(after))
        by_group[group[i]].append(now - was)
    group_diffs = {name: sum(diffs) / len(diffs) for name, diffs in by_group.items()}
    p_sign_flip = sign_flip_p(list(group_diffs.values()))
    p_mcnemar = mcnemar_exact(len(gained), len(lost))
    scores: dict[str, float] = {"current": float(current), "candidate": float(candidate)}
    for role, names in (("current", before), ("candidate", after)):
        for number, name in enumerate(names, start=1):
            scores[f"{role}.run_{number}"] = float(rates[name])
            scores[f"{role}.run_{number}.names"] = float(_rate(arms[name], ids, "names"))
            scores[f"{role}.run_{number}.parsed"] = float(_rate(arms[name], ids, "parsed"))
    scores.update({"gate.noise": float(noise), "gate.delta": float(candidate - current),
                   "gate.p_value": float(p_sign_flip), "mcnemar.p_value": float(p_mcnemar),
                   "gate.n_items": float(len(ids)), "gate.n_groups": float(len(by_group))})
    items = [{"id": i, "group": group[i], "question": first_before[i]["question"],
              "expected": json.dumps({"tool_calls": _gold(expected[i])}, ensure_ascii=False),
              "before": {"answer": first_before[i]["answer"], "correct": bool(first_before[i]["correct"])},
              "after": {"answer": first_after[i]["answer"], "correct": bool(first_after[i]["correct"])}} for i in ids]
    return {
        "scores": scores, "n_items": len(ids), "n_groups": len(by_group),
        "correct": {"current": sum(bool(first_before[i]["correct"]) for i in ids),
                    "candidate": sum(bool(first_after[i]["correct"]) for i in ids)},
        "noise_runs": [sum(bool(arms[name][i]["correct"]) for i in ids) for name in before],
        "after_runs": [sum(bool(arms[name][i]["correct"]) for i in ids) for name in after],
        "names": {name: sum(bool(arms[name][i]["names"]) for i in ids) for name in (*before, *after)},
        "parsed": {name: sum(bool(arms[name][i]["parsed"]) for i in ids) for name in (*before, *after)},
        "noise": float(noise), "delta": float(candidate - current),
        "p_value": float(p_sign_flip), "alpha": ALPHA, "test": TEST_SIGN_FLIP,
        "mcnemar": {"test": TEST_MCNEMAR, "gained_0_to_1": len(gained), "lost_1_to_0": len(lost),
                    "p_value": float(p_mcnemar), "p_exact": str(p_mcnemar)},
        "sign_flip": {"test": TEST_SIGN_FLIP, "p_value": float(p_sign_flip), "p_exact": str(p_sign_flip),
                      "smallest_possible_p": float(Fraction(2, 2 ** len(by_group))) if by_group else 1.0,
                      "group_differences": {name: float(value) for name, value in group_diffs.items()},
                      "group_sizes": {name: len(diffs) for name, diffs in by_group.items()}},
        "flips": {"gained_0_to_1": gained, "lost_1_to_0": lost},
        "identical_outputs": {"before_runs": _same_outputs(arms, before, ids),
                              "after_runs": _same_outputs(arms, after, ids)},
        "items": items,
    }


def _same_outputs(arms: Mapping[str, Mapping[str, Mapping[str, Any]]], names: Sequence[str],
                  ids: Sequence[str]) -> int | None:
    """How many examples got the same output text in every run of the arm; None with one run."""
    if len(names) < 2:
        return None
    return sum(len({arms[name][i]["output"] for name in names}) == 1 for i in ids)


def _markdown(summary: Mapping[str, Any]) -> str:
    s, m, f = summary["scores"], summary["mcnemar"], summary["sign_flip"]
    n = summary["n_items"]
    lines = [f"# Tool-call agreement, {summary['side']} side ({n} examples, {summary['n_groups']} groups)", "",
             f"Run {summary['run_id']}, dataset {summary['dataset_id']}, split sha256 {summary['split_sha256'][:12]}, "
             f"made {summary['made']}.", "",
             "| arm | exact call | tool name | parsed as a call |", "| --- | --- | --- | --- |"]
    for arm in sorted(summary["names"]):
        role, number = "current" if arm.startswith("before") else "candidate", arm.split("-")[1]
        exact = round(s[f"{role}.run_{number}"] * n)
        lines.append(f"| {arm} | {exact}/{n} = {s[f'{role}.run_{number}']:.3f} | {summary['names'][arm]}/{n} | "
                     f"{summary['parsed'][arm]}/{n} |")
    lines += ["", f"Difference (after minus before, means over runs): {summary['delta']:+.3f}; "
                  f"before run-to-run noise {summary['noise']:.3f}.",
              f"Flips, first before run to first after run: {m['gained_0_to_1']} from 0 to 1, "
              f"{m['lost_1_to_0']} from 1 to 0.",
              f"McNemar, exact two-sided, on the {n} paired examples: p = {m['p_value']:.4g}. Descriptive: the "
              "examples are rewordings of a few questions, so they are not independent draws.",
              f"Sign-flip, exact two-sided, over the {summary['n_groups']} question groups (the gate's test): "
              f"p = {f['p_value']:.4g}; its smallest possible value with {summary['n_groups']} groups is "
              f"{f['smallest_possible_p']:.4g}.", "",
              "## The rule", "", summary["rule"], "",
              "## Models", ""]
    for arm, facts in summary["servers"].items():
        lines.append(f"- {arm}: {json.dumps(facts, ensure_ascii=False)}")
    lines += ["", "## Flips", "", "0 to 1: " + (", ".join(summary["flips"]["gained_0_to_1"]) or "none"), "",
              "1 to 0: " + (", ".join(summary["flips"]["lost_1_to_0"]) or "none"), ""]
    return "\n".join(lines)


def run(*, dataset: str | Path, completers: Mapping[str, Completer], out: str | Path, run_id: str,
        side: str = "held_out", before_runs: int = 2, after_runs: int = 1, limit: int | None = None,
        servers: Mapping[str, Any] | None = None,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)) -> dict[str, Any]:
    """Score one side of the dataset's frozen split with the before and after completers and write the files."""
    if side not in ("held_out", "train"):
        raise ValueError("side is held_out or train")
    split = load_split(dataset, require_images=False)
    expected = load_expected(Path(dataset))
    examples = list(split.held_out if side == "held_out" else split.train)[:limit]
    missing = [example.id for example in examples if example.id not in expected]
    if missing:
        raise ValueError(f"{len(missing)} examples have no reference in expected.jsonl, e.g. {missing[0]}")
    manifest = split.data.manifest

    def system(example: Any) -> str:   # one instruction for a dataset, or each example's own (its ts as the time now)
        return system_for(manifest, example)
    header = {"header": True, "rule": RULE, "run_id": run_id, "side": side, "dataset_id": split.dataset_id,
              "split_sha256": split.sha256, "n_examples": len(examples), "max_tokens": MAX_TOKENS,
              "temperature": TEMPERATURE, "servers": dict(servers or {}), "started": now().isoformat()}
    folder = Path(out)
    (folder / "raw").mkdir(parents=True, exist_ok=True)
    plan = {"before": [f"before-{n}" for n in range(1, before_runs + 1)],
            "after": [f"after-{n}" for n in range(1, after_runs + 1)]}
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    failures: list[str] = []

    def work(role: str) -> None:      # one server answers one request at a time: its arms run one after another
        for name in plan[role]:
            path = folder / "raw" / f"{side}.{name}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({**header, "arm": name}, ensure_ascii=False) + "\n")

                def write(row: dict[str, Any]) -> None:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                try:
                    arms[name] = run_arm(name, completers[role], examples, expected, system, write)
                except Exception as exc:   # the arm is not scored; the run says which and why
                    failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
                    return

    threads = [threading.Thread(target=work, args=(role,)) for role in plan]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("; ".join(failures))
    summary = {"run_id": run_id, "task": split.data.manifest.spec.task, "side": side, "dataset_id": split.dataset_id,
               "split_sha256": split.sha256, "measure": "tool-call agreement", "rule": RULE,
               "decode": {"thinking": "off", "sampling": "greedy", "max_tokens": MAX_TOKENS},
               "servers": dict(servers or {}), "made": now().isoformat(), **summarize(arms, examples, expected)}
    (folder / f"{side}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    (folder / f"{side}.md").write_text(_markdown(summary), encoding="utf-8", newline="\n")
    return summary


def main(argv: Sequence[str] | None = None, *, transport: Any = None) -> dict[str, Any]:
    """Tool-call agreement of two served models (before, after) on one side of a dataset's frozen split (L4)."""
    parser = argparse.ArgumentParser(prog="python -m lab tool-call-agreement", description=main.__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--dataset", required=True, help="the dataset folder: split.json and expected.jsonl")
    parser.add_argument("--before-url", required=True, help="llama-server serving the model before the fine-tune")
    parser.add_argument("--after-url", required=True, help="llama-server serving the candidate")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, help="the folder the files are written to")
    parser.add_argument("--side", choices=("held_out", "train"), default="held_out")
    parser.add_argument("--before-runs", type=int, default=2)
    parser.add_argument("--after-runs", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only the first N examples (a smoke run)")
    parser.add_argument("--before-sha256", help="of the before GGUF, for the record")
    parser.add_argument("--after-sha256", help="of the candidate GGUF, for the record")
    args = parser.parse_args(argv)
    completers = {"before": LlamaCompletion(args.before_url, transport=transport),
                  "after": LlamaCompletion(args.after_url, transport=transport)}
    servers = {role: {"url": getattr(args, f"{role}_url"), "sha256": getattr(args, f"{role}_sha256"),
                      **completers[role].props()} for role in completers}
    for role, facts in servers.items():
        if facts["top_k"] != 1:
            return {"ok": False, "outcome": "refused", "error": f"the {role} server's top_k is {facts['top_k']}, not 1"}
    summary = run(dataset=args.dataset, completers=completers, out=args.out, run_id=args.run_id, side=args.side,
                  before_runs=args.before_runs, after_runs=args.after_runs, limit=args.limit, servers=servers)
    scores, mcnemar = summary["scores"], summary["mcnemar"]
    record(CLI_VERB, "ok",
           {"learning.tool_call_agreement.before": scores["current"],
            "learning.tool_call_agreement.after": scores["candidate"],
            "learning.tool_call_agreement.delta": summary["delta"],
            "learning.tool_call_agreement.noise": summary["noise"],
            "learning.tool_call_agreement.p_sign_flip": summary["p_value"],
            "learning.tool_call_agreement.p_mcnemar": mcnemar["p_value"],
            "learning.tool_call_agreement.gained": mcnemar["gained_0_to_1"],
            "learning.tool_call_agreement.lost": mcnemar["lost_1_to_0"],
            "learning.tool_call_agreement.n_items": summary["n_items"],
            "learning.tool_call_agreement.n_groups": summary["n_groups"]},
           {"run_id": args.run_id, "split_sha256": summary["split_sha256"]}, source="lab.tool_call_agreement",
           conditions={"side": args.side, "decode": summary["decode"], "before_runs": args.before_runs,
                       "after_runs": args.after_runs, "dataset_id": summary["dataset_id"],
                       "rule_sha256": hashlib.sha256(RULE.encode("utf-8")).hexdigest(),
                       "before_sha256": args.before_sha256, "after_sha256": args.after_sha256})
    return {"ok": True, "outcome": "ok", **{key: summary[key] for key in (
        "run_id", "side", "n_items", "n_groups", "correct", "noise_runs", "after_runs", "names", "parsed", "noise",
        "delta", "p_value", "mcnemar", "flips", "identical_outputs")}, "out": str(Path(args.out))}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer, ensure_ascii=False))
    sys.exit(0 if answer.get("ok") else 1)
