# Speculative decoding on the gfx1151 teacher: what it buys, which mode, and when to drop it

How vLLM's speculative decoding works on the build the pods run (b1388b1 with `lab/vllm-gfx1201.diff`, Radeon 8060S,
gfx1151), what acceptance the Atlas workloads give it (measured offline on the teacher's real outputs), what
tokens per second and joules per token it should reach at batch 1 and 4 from the chip sheet, the serve flags for each
mode, and the kill criterion. The pod measurement itself is not here: it waits on the dispatcher (experiment SP1).

## Why more tokens per weight read is nearly free on this chip

- **[measured]** On pod-b (gfx1151, torch 2.9.1+rocm7.13.0) the time of a 27B decode-shaped matmul does not grow from
  1 to 16 rows: bf16 `x[m,5120] @ w[248320,5120]` (the lm_head shape) ran at 0.219 TFLOP/s at 1 row and 3.45 at 16,
  and the int4 dequant matmul `w4a16 [m,5120] x [5120,6144]` at 0.594 and 9.08, so 16 rows take 1.02 and 1.05 times
  the time of one (derived: 16 x rate(1) / rate(16)). Copy bandwidth above 32 MiB is about 210 GB/s; loaded watts
  run 75 to 128 W against 24 W idle. Source: [S185] chip sheet, sections "fp16 and bf16 matmul, 27B decode shapes",
  "int4-to-fp16 dequantize plus matmul" and "Device-to-device copy bandwidth".
  - Implication: a target pass that checks k drafted tokens for b requests costs about one pass as long as
    b x (k + 1) is at most 16 rows, so decode tokens per second scale with tokens emitted per pass, and joules per
    token fall in the same proportion at near-constant watts. Batch 1 allows k up to 15; batch 4 allows k up to 3.
  - Candidate experiment: SP1, as specified in "Predicted tokens per second and joules per token" below.
    Expected movement: see SP1. Minimum sample size: see SP1. Confidence: high for the GEMM part, medium for the
    whole pass (attention and the Gated DeltaNet state update were not on the chip sheet).

- **[measured]** The teacher decodes at 2.88 tok/s and 120.3 W average at batch 1 on pod C (vLLM b1388b1,
  `RedHatAI/Qwen3.8-27B-INT4`), which is 347 ms per pass and about 41.8 J per token (derived), and about
  44.6 GB/s against the 210 GB/s copy ceiling at about 15.5 GB per token. Source: [L8] `lab/refine/metrics.jsonl`
  lines 24, 29 and 34; [B58] for the 15.5 GB per token.
  - Implication: the pass is kernel-bound, not bandwidth-bound, today; speculative decoding multiplies whatever the
    pass time is, so it stacks with the kernel work (R40, R53) instead of competing with it.
  - Candidate experiment: none beyond SP1. Expected movement: see SP1. Minimum sample size: see SP1.
    Confidence: high.

## Prompt lookup (ngram) on our own transcripts

- **[practitioner]** vLLM's `ngram` method drafts with no second model: for each request it finds the longest
  n-gram of `prompt_lookup_min` to `prompt_lookup_max` tokens that ends the tokens so far (prompt plus output) and
  occurs earlier in them, and proposes the `num_speculative_tokens` tokens that followed its earliest earlier
  occurrence (a KMP search over the reversed tokens, compiled with numba, on the CPU). With neither bound set, both
  default to 5. Source: [S183] `vllm/v1/spec_decode/ngram_proposer.py::_find_longest_matched_ngram_and_propose_tokens`
  and `vllm/config/speculative.py` at b1388b1
  ([permalink](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/spec_decode/ngram_proposer.py)).
  - Implication: the rule is deterministic and model-free, so its acceptance on our traffic can be measured offline
    from recorded outputs, and the drafting runs on the CPU and costs no GPU time on the pod.
  - Candidate experiment: SP3: rerun `python -m lab speculative-estimate --tokenizer <Qwen3.5 tokenizer folder>`
    with the real tokenizer instead of the proxy. Expected movement: tokens per pass within 20 percent of the proxy
    figure below. Minimum sample size: all 314 turns, 1 run (it is deterministic). Confidence: medium.

- **[measured]** Applied to the teacher's 142 labeled tool-calling conversations (two turns each: the call turn
  and the answer turn) and the 16 rendered template cases, 314 greedy requests and 17,940 output tokens, the rule
  emits on average 2.38 tokens per target pass at `prompt_lookup_min` 1, `prompt_lookup_max` 3, k 4 (draft
  acceptance 0.40), 2.17 at k 3, 2.73 at k 8, and 1.52 at vLLM's default 5 to 5 with k 4. Call turns gain most
  (2.74 tokens per pass at 1 to 3, k 4: argument values and times repeat from the question and the tool list), the
  teacher's prose answers least (1.80). These counts use a proxy tokenizer; the teacher's own ledger counts 14,343
  completion tokens for the label set where the proxy counts 16,960, so the proxy is about 1.18 times finer.
  Source: [S184] `lab/refine/research/kernels/data/speculative-acceptance-2026-10-11.json`
  (`python -m lab speculative-estimate`), `fixtures/learning/tool-calling-labels/label_ledger.jsonl`.
  - Implication: prompt lookup roughly doubles tokens per pass on the labeling and agent traffic, which is mostly
    tool calls; the default 5 to 5 setting leaves most of it unused, so serve with 1 to 3.
  - Candidate experiment: SP1. Expected movement: decode tok/s at batch 1 up 2.0 to 2.4 times on the label prompts.
    Minimum sample size: see SP1. Confidence: medium (proxy tokenizer; acceptance on the pod is the real number).

- **[anecdote]** On a Radeon 8060S (gfx1151) with ROCm 7.1, one llama.cpp user reports a model's native MTP head
  plus an n-gram drafter raising decode from 16.8 to 47.1 tok/s on a file rewrite at 8k context and from about 15
  to 28.6 tok/s at 24k; other posters in the same thread report draft acceptance of 0.86 to 0.89 on short code but
  0.35 to 0.39 on prose, a slowdown from 36.4 to 31.9 tok/s at small context, and that MTP was a net loss without
  a checkpoint mechanism in their fork. No shared benchmark script; each post is one setup. Source: [P18]
  ([llama.cpp discussion 27950](https://github.com/ggml-org/llama.cpp/discussions/27950)), posts of 2026-08-29 to
  2026-09-20.
  - Implication: on this chip speculation pays on repetitive output (code, tool calls) and much less on free prose,
    the same split our offline estimate shows between call turns and answers; a slowdown is possible, so the pod
    measurement must include short prompts and the kill criterion below.
  - Candidate experiment: none beyond SP1. Expected movement: see SP1. Minimum sample size: see SP1.
    Confidence: low (single reports, other engine).

## A small Qwen3.5 draft model

- **[practitioner]** Speculation with a separate draft model (`method: draft_model`) is in the pinned commit: it
  landed on 2026-01-19 and b1388b1 is from 2026-04-15. vLLM refuses a draft whose vocabulary size differs from the
  target's, and the draft runs k sequential forward passes per step before the target checks them. Source: [S181]
  ([commit 4a5299c](https://github.com/vllm-project/vllm/commit/4a5299c93ff97c26def537b92562df5ada530fea)),
  [S183] `vllm/config/speculative.py::verify_equal_vocab_size_if_draft_model` and
  `docs/features/speculative_decoding/draft_model.md` at b1388b1.
  - Implication: the draft must be a Qwen3.5 model with the teacher's 248,320-token vocabulary (the lm_head width
    on the chip sheet); `Qwen/Qwen3.5-0.8B` is the default the serve script expects, and its vocabulary size is
    checked by vLLM at start. Each draft token costs one small-model pass, so a draft only beats ngram if its higher
    acceptance outweighs k of those passes.
  - Candidate experiment: SP2: `bash lab/pod_a_teacher.sh serve --speculative draft` after
    `hf download Qwen/Qwen3.5-0.8B --local-dir ~/models/Qwen3.5-0.8B`, then the SP1 bench and correctness runs.
    Expected movement: decode tok/s at batch 1 at least 10 percent above the ngram arm, or the mode is dropped.
    Minimum sample size: 5 runs per batch size. Confidence: low (no offline acceptance for a draft model).

- **[anecdote]** Our fast-optimization note records a blog report of vLLM serving `Qwen3.5-0.8B` at 86.3 tok/s for
  one client on a Ryzen AI Max+ 395 (ROCm 7.14, vLLM 0.26 and 0.27), about 11.6 ms per token; the post itself could
  not be reopened from this session. Source: `lab/refine/research/kernels/fast_optimization.md` (its finding citing
  B35).
  - Implication: with k 4 a draft adds about 46 ms to a 347 ms target pass (13 percent), so at an assumed per-token
    acceptance of 0.7 it emits (1 - 0.7^5) / 0.3 = 2.77 tokens per 393 ms, about 7.0 tok/s; at 0.5 it is 1.94
    tokens, about 4.9 tok/s, below ngram. Assumptions, not measurements.
  - Candidate experiment: none beyond SP2. Expected movement: see SP2. Minimum sample size: see SP2.
    Confidence: low.

## EAGLE and MTP

- **[practitioner]** At b1388b1 vLLM also has `eagle`, `eagle3`, `mtp`, `suffix` and `ngram_gpu`; MTP model types
  include `qwen3_5_mtp` (`vllm/model_executor/models/qwen3_5_mtp.py`), and the Gated DeltaNet attention backend keeps
  per-draft-token state indices, so the teacher's hybrid layers can be verified several tokens at a time. EAGLE-3 is
  allowed for Qwen-family targets but needs a trained EAGLE head for this exact target. Source: [S183]
  `vllm/config/speculative.py`, `vllm/v1/attention/backends/gdn_attn.py` at b1388b1.
  - Implication: MTP is the best candidate if the served checkpoint still carries its MTP weights (an INT4 export can
    drop them); EAGLE needs a head nobody has trained for Qwen3.8-27B, so it is out of scope for the weekend.
  - Candidate experiment: SP4: on pod A, list the MTP tensors in the checkpoint index without loading it:
    `python3 -c "import json; m=json.load(open('/home/jovyan/models/Qwen3.8-27B-INT4/model.safetensors.index.json'))['weight_map']; print(sum('mtp' in k for k in m))"`.
    Expected movement: none to tok/s; a count above 0 opens an MTP arm for SP1. Minimum sample size: 1 run.
    Confidence: high (the command reads a file).

- **[practitioner]** ROCm support in the pin, per mode: ngram drafts on the CPU and draft_model reuses the target's
  model runner, so neither needs a ROCm-specific kernel. For unquantized bf16 layers (the lm_head), the gfx11 and
  gfx12 skinny GEMM path (`wvSplitK`) is taken only for 1 to 4 rows; 5 rows or more go to `torch.nn.functional.linear`.
  A main-branch commit of 2026-05-28 raises that bound to 5 rows "for speculative decoding" and is not in the pin;
  the later ROCm speculative commits on main are for DeepSeek V4 and Kimi drafters. Source: [S183]
  `vllm/model_executor/layers/utils.py::rocm_unquantized_gemm_impl` at b1388b1, [S182]
  ([commit a9ec46d](https://github.com/vllm-project/vllm/commit/a9ec46d4b7db6ae1d0cb0f27ce1a7ddb0f749ef8)).
  - Implication: at batch 1 with k 4 or more the lm_head runs through torch's linear, which is what the chip sheet
    measured flat to 16 rows, so the prediction below holds on the pinned build; nothing in the pin blocks ngram or
    draft on gfx1151.
  - Candidate experiment: none beyond SP1. Expected movement: see SP1. Minimum sample size: see SP1.
    Confidence: medium.

## Serve flags for each mode

- **[practitioner]** `bash lab/pod_a_teacher.sh serve --speculative off|ngram|draft` (PR #323) keeps every flag the
  teacher already starts with (tool calling with `qwen3_coder`, priority scheduling, two served names) and adds:
  off, nothing; ngram, `--speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 1, "prompt_lookup_max": 3}'`;
  draft, `--speculative-config '{"method": "draft_model", "model": "~/models/Qwen3.5-0.8B", "num_speculative_tokens": 4}'`,
  with `SPEC_K`, `SPEC_MIN`, `SPEC_MAX` and `DRAFT` as overrides. The JSON keys are the ones `SpeculativeConfig`
  defines at b1388b1. Source: `lab/pod_a_teacher.sh::serve`, [S183] `vllm/config/speculative.py`.
  - Implication: one restart of the teacher switches modes; batch-4 arms use `SPEC_K=3` to stay inside 16 rows.
  - Candidate experiment: none beyond SP1. Expected movement: see SP1. Minimum sample size: see SP1.
    Confidence: high.

## Predicted tokens per second and joules per token

- **[opinion]** Prediction from the findings above, at the measured 120.3 W (the chip sheet's loaded maximum of
  128 W gives the upper joules in brackets), with the GEMM-flat pass time of 347 ms and tokens per pass from the
  offline estimate (proxy tokenizer, ngram 1 to 3):

  | Arm | Batch | k | Tokens per pass | Decode tok/s (aggregate) | J per token |
  | --- | --- | --- | --- | --- | --- |
  | off | 1 | - | 1.00 | 2.88 | 41.8 (44.4) |
  | ngram | 1 | 4 | 2.38 | 6.85 | 17.6 (18.7) |
  | ngram | 1 | 8 | 2.73 | 7.87 | 15.3 (16.3) |
  | off | 4 | - | 1.00 | 11.5 | 10.4 (11.1) |
  | ngram | 4 | 3 | 2.17 | 25.0 | 4.8 (5.1) |
  | draft, acceptance 0.7 assumed | 1 | 4 | 2.77 | 7.0 | 17.1 |

  The batch-4 rows assume the pass stays at 347 ms for 4 to 16 rows, which the chip sheet shows for the GEMMs only.
  At the 210 GB/s ceiling (15.5 GB per token, 73.8 ms per pass) the same factors give 32 tok/s at batch 1, k 4.
  Source: [S185], [L8], [S184].
  - Implication: speculative decoding is the largest single lever left on the teacher at the same watts, about
    2.4 times at batch 1, independent of the kernel work.
  - Candidate experiment: SP1: after the R53 rebuild, on pod A, one slot: for each of off, ngram (k 4 at batch 1,
    `SPEC_K=3` at batch 4) and draft when SP4 or SP2 allows: `bash lab/pod_a_teacher.sh serve --speculative <mode>`,
    then `python -m lab correctness compare --url http://127.0.0.1:8000 --model teacher --node pod-a --baseline <the off recording> --set speculative=<mode>`,
    then `python -m lab bench attach --url http://127.0.0.1:8000 --model teacher --node pod-a --batches 1,4 --no-64 --metrics-file --metric-prefix speculative.<mode> --out lab/refine/research/kernels/data/speculative-gfx1151-<date>.json`
    (the off arm records the baseline first with `python -m lab correctness record`). Expected movement: decode
    tok/s up 2.0 to 2.4 times and joules per token down 50 to 58 percent at batch 1; greedy outputs identical.
    Minimum sample size: 5 runs of 256 tokens per batch size and mode. Confidence: medium.

## Kill criterion

- **[opinion]** Keep a speculative mode on the teacher only if, on the SP1 run, all hold: every greedy output in the
  correctness set equals the off arm's (a speculative mode that changes output is a bug, since rejection sampling
  is lossless at temperature 0); batch-1 decode is at least 1.3 times the off arm (3.7 tok/s or more); batch-4
  aggregate decode is not below the off arm; joules per token is at least 20 percent lower. A draft model must
  also beat ngram by 10 percent at batch 1, or it is not worth its memory and second model. Otherwise serve
  `--speculative off` and record the numbers. Source: [S184], [L8], `lab/correctness.py`.
  - Implication: the decision is mechanical from one slot's ledger rows; no judgment call is left for the pod.
  - Candidate experiment: none beyond SP1. Expected movement: see SP1. Minimum sample size: see SP1.
    Confidence: high.

## Open questions

- Does `RedHatAI/Qwen3.8-27B-INT4` keep the MTP weights (SP4)? If yes, an `mtp` arm (`{"method": "mtp", "num_speculative_tokens": 1}`) joins SP1.
- Does `Qwen/Qwen3.5-0.8B` have the 248,320-token vocabulary? vLLM checks at start; not verified from this session.
- How much do attention and the Gated DeltaNet state update add to a 5- or 16-row pass? Not on the chip sheet.
- Is there a 2025 or 2026 paper measuring prompt lookup on tool-calling traffic? arXiv and Hugging Face did not
  resolve from this session, so none was opened.

## Search log

- 2026-10-11, budget 3 h, used about 1 h 30 min (two sessions). Starting points: vLLM at b1388b1 and main (blobless
  clone, `git show` of `vllm/config/speculative.py`, `vllm/v1/spec_decode/`, the speculative decoding docs,
  `gdn_attn.py`, `layers/utils.py`; `git log` for ROCm speculative commits), our chip sheet, metrics file, label
  ledger and the kernels SOURCES index; llama.cpp discussion 27950 reopened. arXiv, huggingface.co and the
  soothill.io blog did not resolve (ENOTFOUND). Stopped when the vLLM sources answered every mode's mechanism and the
  remaining questions needed the pod.
