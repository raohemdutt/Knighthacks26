"""Held-out perplexity of a teacher checkpoint under vLLM, so low-bit weights can be compared on the same text.

    python lab/lowbit_ppl.py --model /content/models/Qwen3.8-27B-INT4 --label int4 --out ppl-int4.json
    python lab/lowbit_ppl.py --model /content/quant/<name>-gptq --dtype float16 --label w3a16 --out ppl-w3.json

The text is the wikitext-2-raw-v1 *test* split (the calibration set uses only wikitext-103 *train*), joined with
blank lines, tokenized with the base model's tokenizer, and cut into non-overlapping windows of --window tokens
(the first --windows of them). Each window is one prompt; vLLM's prompt_logprobs give log p(token | the window's
earlier tokens), and perplexity is exp of the mean negative log-likelihood over every token but each window's first.
Every checkpoint scored this way sees the same token ids, so their perplexities compare directly.

Runs where vLLM runs (the Colab A100). Standard library at import time; vllm, transformers and datasets inside.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

TOKENIZER = "Qwen/Qwen3.8-27B"


def windows(ids: list[int], size: int, count: int) -> list[list[int]]:
    """Non-overlapping windows of `size` ids, at most `count` of them; a short tail is dropped."""
    out = [ids[i:i + size] for i in range(0, len(ids) - size + 1, size)]
    return out[:count] if count > 0 else out


def perplexity(prompt_logprobs: list[list[float | None]]) -> tuple[float, int]:
    """exp(mean NLL) over every scored token; None (the first token of a window) is skipped."""
    nll, n = 0.0, 0
    for row in prompt_logprobs:
        for lp in row:
            if lp is not None:
                nll -= lp
                n += 1
    if not n:
        raise ValueError("no scored tokens")
    return math.exp(nll / n), n


def heldout_ids() -> list[int]:  # pragma: no cover - needs the network
    from datasets import load_dataset
    from transformers import AutoTokenizer

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return AutoTokenizer.from_pretrained(TOKENIZER)(text, add_special_tokens=False)["input_ids"]


def score(model: str, dtype: str, wins: list[list[int]], max_len: int, gpu_util: float) -> dict:  # pragma: no cover
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    started = time.time()
    # max_num_batched_tokens bounds the prompt-logprob logits (tokens x 248k vocab in float32) to a few GB.
    llm = LLM(model=model, dtype=dtype, max_model_len=max_len, gpu_memory_utilization=gpu_util,
              max_num_batched_tokens=max_len, max_num_seqs=8, limit_mm_per_prompt={"image": 0, "video": 0},
              tokenizer=TOKENIZER)
    loaded = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins],
                        SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))
    rows = []
    for w, o in zip(wins, outs):
        lps = o.prompt_logprobs or []
        rows.append([None if d is None else d[tok].logprob for tok, d in zip(w, lps)])
    ppl, n = perplexity(rows)
    engine = llm.llm_engine
    cfg = getattr(engine, "model_config", None) or getattr(getattr(engine, "vllm_config", None), "model_config", None)
    import vllm

    return {"ppl": round(ppl, 4), "tokens": n, "windows": len(wins), "quantization": getattr(cfg, "quantization", None),
            "dtype": str(getattr(cfg, "dtype", dtype)), "vllm": vllm.__version__, "load_s": round(loaded - started, 1),
            "score_s": round(time.time() - loaded, 1)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - runs on the GPU
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=0, help="0 = every full window of the test split")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    wins = windows(heldout_ids(), args.window, args.windows)
    result = {"label": args.label, "model": args.model, "text": "wikitext-2-raw-v1 test", "window": args.window,
              **score(args.model, args.dtype, wins, args.window + 16, args.gpu_util)}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
