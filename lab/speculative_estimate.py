"""How many tokens n-gram speculative decoding would accept on our own conversations, measured offline.

    python -m lab speculative-estimate                                   # the committed estimate, proxy tokenizer
    python -m lab speculative-estimate --tokenizer ~/models/Qwen3.5-4B   # the same with the model's own tokenizer
    python -m lab speculative-estimate --out lab/refine/research/kernels/data/speculative-acceptance-<date>.json

vLLM's `ngram` method (prompt lookup) drafts the next tokens by finding the longest n-gram, between
`prompt_lookup_min` and `prompt_lookup_max` tokens long, that ends the request's tokens so far (prompt plus
output) and also occurs earlier in them; the k tokens that followed its earliest earlier occurrence are the draft
(`vllm/v1/spec_decode/ngram_proposer.py`, `_find_longest_matched_ngram_and_propose_tokens`, at the commit the pods
run, b1388b1). The target model then checks the whole draft in one forward pass: with greedy sampling it keeps the
longest prefix of the draft that equals its own next tokens and adds one token of its own, so a step emits
accepted + 1 tokens.

The draft rule needs no model, so the acceptance is computed here from text the teacher really produced: each
assistant turn of each conversation is one request whose prompt is everything rendered before it (the Qwen3.5
chat template with thinking off, `lab/format_traces.py`) and whose output is the turn itself. Two sources:

    * the teacher's labels of the tool-calling fixture (`fixtures/learning/tool-calling-labels/`): the question,
      the calls the teacher made, the tool results it was shown and its answer, as recorded in traces.jsonl;
    * the 16 conversations rendered by the real template (`fixtures/lab/qwen35-template/rendered.jsonl`).

Without --tokenizer the text is split by a proxy tokenizer (words with their leading space, single digits,
punctuation runs, whitespace runs, the chat template's special tokens), which is coarser than the model's BPE, so
the proxy estimate is a stand-in until the run with the real tokenizer is committed. The output is a JSON file:
for each setting of (prompt_lookup_min, prompt_lookup_max, num_speculative_tokens), per source and per turn kind,
the tokens generated, the target forward passes they took, tokens per pass, and the draft acceptance rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contexts.agent.application.qa import TOOLS
from lab.format_traces import IM_END, IM_START, Call, Round, Trace, as_call, load_tokenizer, messages_of, render_with_spans

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO_ROOT / "fixtures" / "learning" / "tool-calling-labels"
TEMPLATE_CASES = REPO_ROOT / "fixtures" / "lab" / "qwen35-template" / "rendered.jsonl"
VLLM_COMMIT = "b1388b1fbf5aaef47937fabe98931211684666a6"
RULE = "vllm/v1/spec_decode/ngram_proposer.py:_find_longest_matched_ngram_and_propose_tokens"

# (prompt_lookup_min, prompt_lookup_max): vLLM's default when neither is given is (5, 5); the rest are the
# settings its docs and the field reports use.
LOOKUPS: tuple[tuple[int, int], ...] = ((1, 3), (2, 4), (3, 5), (5, 5))
DRAFT_LENGTHS: tuple[int, ...] = (2, 4, 8)

_PROXY = re.compile(r"<\|[A-Za-z_]+\|>|\d| ?[A-Za-z]+| ?[^\sA-Za-z\d]+|\n|[ \t]+")


class EstimateError(ValueError):
    """The inputs cannot be read. str() says which file and why, never the text of a conversation."""


# --- the draft rule ------------------------------------------------------------------------------

def propose_reference(tokens: Sequence[int], min_n: int, max_n: int, k: int) -> list[int]:
    """vLLM's prompt-lookup draft, a line-for-line port of the KMP search at b1388b1 (without the max_model_len
    cap): the k tokens after the earliest earlier occurrence of the longest suffix n-gram in [min_n, max_n]."""
    total = len(tokens)
    if total < min_n or k <= 0:
        return []
    rev = list(tokens)[::-1]
    lps = [0] * max_n
    longest = position = prev = 0
    i = 1
    while i < total:
        if rev[prev] == rev[i]:
            prev += 1
            if prev >= longest:
                longest, position = prev, i
            if i < max_n:
                lps[i] = prev
            if prev == max_n:
                prev = lps[max_n - 1]
            i += 1
        elif prev != 0:
            prev = lps[prev - 1]
        else:
            i += 1
    if longest < min_n:
        return []
    start = total - 1 - position + longest
    return list(tokens[start:start + min(k, total - start)])


def _as_text(tokens: Sequence[int]) -> str:
    return "".join(map(chr, tokens))


def propose(text: str, min_n: int, max_n: int, k: int) -> str:
    """The same draft as propose_reference() for tokens held one per character (chr(token id)), found with
    str.find: for n from max_n down to min_n, the earliest occurrence of the last n tokens that ends before the
    last token; the first n that has one wins."""
    total = len(text)
    if total < min_n or k <= 0:
        return ""
    for n in range(min(max_n, total - 1), min_n - 1, -1):
        start = text.find(text[total - n:], 0, total - 1)
        if start >= 0:
            return text[start + n:start + n + k]
    return ""


@dataclass
class Tally:
    """What speculative decoding did over some requests: tokens generated, target forward passes taken after the
    first token (the prefill emits that one), tokens drafted, and draft tokens accepted."""

    requests: int = 0
    tokens: int = 0
    steps: int = 0
    drafted: int = 0
    accepted: int = 0

    def add(self, other: Tally) -> None:
        self.requests += other.requests
        self.tokens += other.tokens
        self.steps += other.steps
        self.drafted += other.drafted
        self.accepted += other.accepted

    def summary(self) -> dict[str, Any]:
        decoded = self.tokens - self.requests
        return {
            "requests": self.requests, "tokens": self.tokens, "decode_steps": self.steps,
            "baseline_decode_steps": decoded, "drafted": self.drafted, "accepted": self.accepted,
            "tokens_per_step": round(decoded / self.steps, 4) if self.steps else None,
            "acceptance_rate": round(self.accepted / self.drafted, 4) if self.drafted else None,
            "accepted_per_step": round(self.accepted / self.steps, 4) if self.steps else None,
        }


def simulate(prompt: Sequence[int], output: Sequence[int], min_n: int, max_n: int, k: int) -> Tally:
    """Greedy speculative decoding of `output` after `prompt` with the prompt-lookup draft: the prefill emits the
    first output token, then each step drafts from everything so far, keeps the draft's longest prefix that equals
    the next output tokens, and emits that plus one token (fewer only when the output ends)."""
    tally = Tally(requests=1, tokens=len(output))
    if not output:
        tally.requests = 0
        return tally
    whole = _as_text(list(prompt) + list(output))
    base = len(prompt)
    pos = 1
    while pos < len(output):
        draft = propose(whole[:base + pos], min_n, max_n, k)
        accepted = 0
        for guess, actual in zip(draft, whole[base + pos:]):
            if guess != actual:
                break
            accepted += 1
        tally.steps += 1
        tally.drafted += len(draft)
        tally.accepted += accepted
        pos += min(accepted + 1, len(output) - pos)
    return tally


# --- the requests ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    """One generation: where it came from, which kind of turn it is ("call" or "answer"), the text the model was
    given and the text it wrote."""

    source: str
    kind: str
    prompt: str
    output: str


def _requests(name: str, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> list[Request]:
    text, spans = render_with_spans(messages, tools)
    out = []
    for start, end in spans:
        output = text[start:end]
        kind = "call" if "<tool_call>" in output else "answer"
        out.append(Request(name, kind, text[:start], output))
    return out


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise EstimateError(f"cannot read {path.name} ({type(exc).__name__})") from None


def label_requests(folder: str | Path = LABELS_DIR) -> list[Request]:
    """The teacher's labeled conversations: each example's question (examples.jsonl) with the calls, tool results
    and answer the teacher produced (traces.jsonl), under the dataset's instruction as the system prompt."""
    folder = Path(folder)
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EstimateError(f"cannot read manifest.json in {folder.name} ({type(exc).__name__})") from None
    system = (manifest.get("spec") or {}).get("instruction") or ""
    questions = {row["id"]: row.get("context") or "" for row in _jsonl(folder / "examples.jsonl")}
    out: list[Request] = []
    for row in _jsonl(folder / "traces.jsonl"):
        question = questions.get(row.get("id"))
        if not question:
            raise EstimateError(f"traces.jsonl names an example examples.jsonl does not have in {folder.name}")
        calls = tuple(as_call(raw) for raw in row.get("tool_calls") or [])
        results = tuple(str(raw.get("result", "")) for raw in row.get("tool_calls") or [])
        rounds = (Round(calls, results),) if calls else ()
        trace = Trace(row["id"], system, question, rounds, row["answer"])
        out.extend(_requests("teacher_labels", messages_of(trace), TOOLS))
    return out


def template_requests(path: str | Path = TEMPLATE_CASES) -> list[Request]:
    """The conversations the real chat template rendered: every assistant turn after the question is a request."""
    out: list[Request] = []
    for row in _jsonl(Path(path)):
        out.extend(_requests("template_cases", row["messages"], row.get("tools") or []))
    return out


# --- tokenizing ------------------------------------------------------------------------------------

@dataclass
class ProxyTokenizer:
    """A stand-in for the model's tokenizer: words with their leading space, single digits, punctuation runs,
    newlines, space runs and the template's special tokens, each piece one id. Coarser than BPE."""

    vocab: dict[str, int] = field(default_factory=dict)

    def encode(self, text: str) -> list[int]:
        return [self.vocab.setdefault(piece, len(self.vocab)) for piece in _PROXY.findall(text)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(requests: Iterable[Request], tokenizer: Any, lookups: Sequence[tuple[int, int]] = LOOKUPS,
             draft_lengths: Sequence[int] = DRAFT_LENGTHS) -> list[dict[str, Any]]:
    """For each (min, max, k): the tally over all requests, per source, and per source and turn kind."""
    encoded = [(r, tokenizer.encode(r.prompt), tokenizer.encode(r.output)) for r in requests]
    rows = []
    for min_n, max_n in lookups:
        for k in draft_lengths:
            groups: dict[str, Tally] = {}
            for request, prompt, output in encoded:
                tally = simulate(prompt, output, min_n, max_n, k)
                for key in ("all", request.source, f"{request.source}/{request.kind}"):
                    groups.setdefault(key, Tally()).add(tally)
            rows.append({"prompt_lookup_min": min_n, "prompt_lookup_max": max_n, "num_speculative_tokens": k,
                         "groups": {key: groups[key].summary() for key in sorted(groups)}})
    return rows


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Estimate n-gram (prompt lookup) speculative decoding acceptance offline on the teacher's real outputs."""
    parser = argparse.ArgumentParser(prog="python -m lab speculative-estimate", description=main.__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--labels", default=str(LABELS_DIR), help="the labeled dataset folder with traces.jsonl")
    parser.add_argument("--template-cases", default=str(TEMPLATE_CASES), help="rendered.jsonl of the template cases")
    parser.add_argument("--tokenizer", help="a tokenizer folder (tokenizer.json); without it the proxy is used")
    parser.add_argument("--out", help="write the full estimate to this JSON file")
    args = parser.parse_args(argv)

    labels, cases = Path(args.labels), Path(args.template_cases)
    requests = label_requests(labels) + template_requests(cases)
    if args.tokenizer:
        tokenizer, _ = load_tokenizer(args.tokenizer)
        tokenizer_name = f"model tokenizer ({Path(args.tokenizer).name})"
    else:
        tokenizer, tokenizer_name = ProxyTokenizer(), "proxy (lab/speculative_estimate.py ProxyTokenizer)"
    rows = estimate(requests, tokenizer)
    best = max(rows, key=lambda row: row["groups"]["all"]["tokens_per_step"] or 0)
    report = {
        "format": "atlas.speculative-acceptance/1",
        "rule": {"vllm_commit": VLLM_COMMIT, "function": RULE, "sampling": "greedy"},
        "tokenizer": tokenizer_name,
        "inputs": {
            "teacher_labels": {"path": str(labels.relative_to(REPO_ROOT)) if labels.is_relative_to(REPO_ROOT)
                               else labels.name, "traces_sha256": _sha(labels / "traces.jsonl")},
            "template_cases": {"path": str(cases.relative_to(REPO_ROOT)) if cases.is_relative_to(REPO_ROOT)
                               else cases.name, "sha256": _sha(cases)},
        },
        "requests": len(requests),
        "settings": rows,
        "best": {key: best[key] for key in ("prompt_lookup_min", "prompt_lookup_max", "num_speculative_tokens")}
        | {"tokens_per_step": best["groups"]["all"]["tokens_per_step"],
           "acceptance_rate": best["groups"]["all"]["acceptance_rate"]},
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"requests": report["requests"], "tokenizer": tokenizer_name, "best": report["best"],
            "out": args.out}


if __name__ == "__main__":
    print(json.dumps(main()))
