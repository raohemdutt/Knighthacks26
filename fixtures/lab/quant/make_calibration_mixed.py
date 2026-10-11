"""Writes the 512-row mixed calibration set (calibration-mixed-512.jsonl) the A100 quantization runs used.

    python fixtures/lab/quant/make_calibration_mixed.py     # needs `datasets` and `transformers`, and the network

The committed Atlas set (calibration.jsonl, 184 rows from make_calibration.py) is narrow: tool-calling turns, the
golden questions and a few rendered template cases. GPTQ fitted on that alone can over-fit the rounding to one kind
of text and lose general language, which the teacher still needs. So this set keeps all 184 Atlas rows and pads them
to 512 with general text from two public datasets, picked deterministically:

  264 rows  HuggingFaceH4/ultrachat_200k, split train_sft, shuffled with seed 42, rendered with the base model's own
            chat template (thinking off), each cut to MAX_CHARS
   64 rows  Salesforce/wikitext wikitext-103-raw-v1, split train: the first paragraphs of at least MIN_WIKI_CHARS,
            in file order, each cut to MAX_CHARS. The train split only: the held-out perplexity text is the
            wikitext-2 *test* split, which shares no text with train.

Each line is {"id", "source", "text", "chars"}, Atlas rows first, as in calibration.jsonl. The script runs where the
quantization runs (the Colab VM), and the file it wrote there is the one committed; running it again with the same
dataset revisions writes the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATLAS = HERE / "calibration.jsonl"
OUT = HERE / "calibration-mixed-512.jsonl"

BASE_MODEL = "Qwen/Qwen3.8-27B"
TOTAL = 512
N_WIKI = 64
SEED = 42
MAX_CHARS = 12000
MIN_WIKI_CHARS = 1500


def row(source: str, text: str) -> dict:
    text = text[:MAX_CHARS]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return {"chars": len(text), "id": f"{source[:2]}-{digest}", "source": source, "text": text}


def atlas_rows(path: Path = ATLAS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wiki_rows(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    out: list[dict] = []
    for item in ds:
        text = item["text"].strip()
        if len(text) >= MIN_WIKI_CHARS and not text.startswith("="):
            out.append(row("wikitext-103-train", text))
            if len(out) == n:
                break
    return out


def chat_rows(n: int) -> list[dict]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=SEED)
    out: list[dict] = []
    for item in ds:
        text = tok.apply_chat_template(item["messages"], tokenize=False, enable_thinking=False)
        out.append(row("ultrachat-200k", text))
        if len(out) == n:
            break
    return out


def build() -> list[dict]:
    atlas = atlas_rows()
    n_chat = TOTAL - len(atlas) - N_WIKI
    if n_chat <= 0:
        raise SystemExit(f"{ATLAS} already holds {len(atlas)} rows; nothing left to pad to {TOTAL}")
    return atlas + chat_rows(n_chat) + wiki_rows(N_WIKI)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    rows = build()
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                              encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(json.dumps({"rows": len(rows), "by_source": counts, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
