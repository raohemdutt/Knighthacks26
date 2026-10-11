"""Writes the calibration set the teacher's low-bit quantization reads (calibration.jsonl), deterministically.

    python fixtures/lab/quant/make_calibration.py           # rewrite calibration.jsonl
    python fixtures/lab/quant/make_calibration.py --check   # exit 1 when the committed file is stale

GPTQ and AutoRound fit each layer's rounding to the activations of a calibration set, so the set should look like
what the teacher is asked at serve time: Atlas's own tool-calling turns, the golden questions and their phrasings,
the rendered chat-template cases, and the text prompts of the benchmark set. Every row is text the repository
already holds under fixtures/, rendered in the Qwen3.5 chat format (thinking off, tool calls in the XML form the
template renders), so no camera frame, no image and no private data enters it. Rows are ordered by source and then
by the source's own order; running it again writes the same bytes.

Each line is {"id", "source", "text", "chars"}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "calibration.jsonl"

TOOL_CALLING = ROOT / "fixtures" / "learning" / "tool-calling-fixture"
GOLDEN = ROOT / "fixtures" / "golden" / "questions.yaml"
TEMPLATE = ROOT / "fixtures" / "lab" / "qwen35-template" / "rendered.jsonl"
BENCH = ROOT / "fixtures" / "lab" / "bench" / "prompts.json"

SOURCES = ("tool-calling", "golden", "qwen35-template", "bench")
EMPTY_THINK = "<think>\n\n</think>\n\n"


def turn(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def tool_call(call: dict) -> str:
    """One call in the XML form the Qwen3.5 template renders (see fixtures/lab/qwen35-template)."""
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in call.get("arguments", {}).items())
    return f"<tool_call>\n<function={call['name']}>\n{params}</function>\n</tool_call>"


def tool_calling_rows() -> list[dict]:
    """The labelled variant turns of the tool-calling fixture (its golden-* rows repeat the golden questions)."""
    manifest = json.loads((TOOL_CALLING / "manifest.json").read_text(encoding="utf-8"))
    system = manifest["spec"]["instruction"]
    rows = []
    for line in (TOOL_CALLING / "expected.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        if ex.get("origin") == "golden":
            continue
        label = ex["label"]
        calls = "\n".join(tool_call(c) for c in label.get("tool_calls", []))
        text = turn("system", system) + turn("user", ex["question"]) + turn("assistant", EMPTY_THINK + calls)
        for result in ex.get("tool_results", []):
            text += turn("user", f"<tool_response>\n{result}\n</tool_response>")
        text += turn("assistant", EMPTY_THINK + label["answer"])
        rows.append({"id": ex["id"], "source": "tool-calling", "text": text})
    return rows


def golden_rows() -> list[dict]:
    """Every golden question and each of its other phrasings, as a user turn under the fixture's system prompt."""
    manifest = json.loads((TOOL_CALLING / "manifest.json").read_text(encoding="utf-8"))
    system = manifest["spec"]["instruction"]
    rows = []
    for i, q in enumerate(yaml.safe_load(GOLDEN.read_text(encoding="utf-8")), start=1):
        for j, phrasing in enumerate([q["q"], *q.get("also", [])]):
            text = turn("system", system) + turn("user", phrasing) + "<|im_start|>assistant\n" + EMPTY_THINK
            rows.append({"id": f"golden-{i:02d}-{j}", "source": "golden", "text": text})
    return rows


def template_rows() -> list[dict]:
    """The chat-template cases exactly as the tokenizer rendered them."""
    rows = []
    for line in TEMPLATE.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        rows.append({"id": f"template-{case['name']}", "source": "qwen35-template", "text": case["text"]})
    return rows


def bench_rows() -> list[dict]:
    """The text prompts of the benchmark set (its image prompts are skipped: calibration is text only)."""
    rows = []
    for p in json.loads(BENCH.read_text(encoding="utf-8"))["prompts"]:
        if p["kind"] != "text":
            continue
        text = turn("system", p["system"]) + turn("user", p["user"]) + "<|im_start|>assistant\n" + EMPTY_THINK
        rows.append({"id": f"bench-{p['id']}", "source": "bench", "text": text})
    return rows


def build() -> list[dict]:
    rows = tool_calling_rows() + golden_rows() + template_rows() + bench_rows()
    for row in rows:
        row["chars"] = len(row["text"])
    return rows


def render(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing it")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    text = render(build())
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is stale: run python fixtures/lab/quant/make_calibration.py", file=sys.stderr)
            return 1
        print(f"{args.out} is current")
        return 0
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({text.count(chr(10))} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
