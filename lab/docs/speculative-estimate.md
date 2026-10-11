# Speculative decoding acceptance, offline: `python -m lab speculative-estimate` (R54)

Arguments: `python -m lab speculative-estimate -h`. The full reasoning, the predictions and the kill criterion are in
`lab/refine/research/kernels/speculative_gfx1151.md`.

How many tokens vLLM's `ngram` speculative decoding (prompt lookup) would accept on our own conversations, measured
without a model. The verb applies vLLM's draft rule at the pinned commit b1388b1
(`vllm/v1/spec_decode/ngram_proposer.py`, `_find_longest_matched_ngram_and_propose_tokens`: the k tokens after the
earliest earlier occurrence of the longest n-gram that ends the tokens so far) to every assistant turn of the
teacher's labeled tool-calling conversations (`fixtures/learning/tool-calling-labels/`) and of the 16 rendered
template cases (`fixtures/lab/qwen35-template/rendered.jsonl`), and counts the greedy target passes each turn takes:
a pass emits the accepted draft tokens plus one of the model's own.

```
python -m lab speculative-estimate                                   # the committed estimate, proxy tokenizer
python -m lab speculative-estimate --tokenizer ~/models/Qwen3.5-4B   # the same with the model's own tokenizer
python -m lab speculative-estimate --out lab/refine/research/kernels/data/speculative-acceptance-<date>.json
```

- **What it writes.** One JSON line on stdout, and with `--out` the full estimate: for each setting of
  (`prompt_lookup_min`, `prompt_lookup_max`, `num_speculative_tokens`), per source and per turn kind, the tokens
  generated, the target passes they took, tokens per pass and the draft acceptance rate. The committed result is
  `lab/refine/research/kernels/data/speculative-acceptance-2026-10-11.json`: 314 requests, 17,940 output tokens,
  16 settings; n-gram 1 to 3 with k 4 gives 2.38 tokens per pass (call turns 2.74, answers 1.80), k 8 gives 2.73,
  vLLM's default 5 to 5 gives 1.52.
- **The tokenizer.** Without `--tokenizer` the text is split by a proxy (words with their leading space, single
  digits, punctuation runs, the chat template's special tokens), about 1.18 times finer than the teacher's own
  token count in the label ledger, so the committed estimate is a stand-in until the run with the real tokenizer is
  committed. That run needs only a tokenizer folder, no GPU.
- **Serving it.** `bash lab/pod_a_teacher.sh serve --speculative off|ngram|draft` starts the teacher with the
  matching `--speculative-config` (default off; `SPEC_K`, `SPEC_MIN`, `SPEC_MAX` and `DRAFT` override the defaults;
  `draft` refuses to start without the draft checkpoint). Not yet measured on a pod.

Tests: `tests/test_lab_speculative_estimate.py` (the fast draft against the port of vLLM's search, the pass count,
the requests built from `fixtures/lab/speculative-tiny/`, the report, the registered verb) and
`tests/test_lab_pod_a_teacher.py` (the vLLM command line in each mode, in bash with stand-ins).
