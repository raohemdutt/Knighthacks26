# Correctness harness: `python -m lab correctness` (R21)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab correctness -h`.

A speedup that changes outputs is a bug (kernels rubric K4). `lab/correctness.py` records the greedy outputs of a
running server on the R11 prompt set (`fixtures/lab/bench/prompts.json`: two text prompts and two drawn images, sent
with lab/bench.py's own `PromptSet`, `Settings` and endpoints) as a baseline fixture, and later compares a run with it.
Requests only: it never spawns, restarts or changes anything on the pod, and it sends one request at a time.

```
python -m lab correctness record  --account C --engine vllm --chip gfx1151 --node pod-a --engine-commit b1388b1 \
    --patch lab/vllm-gfx1201.diff --quantization int4 --rocm 7.13 --torch 2.9.1+rocm7.13.0 --triton 3.5.1 \
    --set enforce_eager=false --set serve_flags="..."
python -m lab correctness compare --account C <the same condition flags> \
    --baseline fixtures/lab/correctness/vllm-gfx1151-2026-10-11.json [--allow-condition-drift] [--metrics-file]
```

- **Requests.** Greedy (temperature 0, top_p 1, seed 0), `--max-tokens` 128 with `ignore_eos`, thinking off, no
  per-request nonce (the prompt must be the same bytes every run), and `logprobs` with `return_tokens_as_token_ids`,
  so an output is token ids plus the logprob of each chosen token. A server that sends no logprobs is recorded by its
  streamed chunks (`token_kind` `chunk_text`).
- **Fixture** `fixtures/lab/correctness/<engine>-<chip>-<date>.json` (date in UTC): the conditions (engine, engine
  commit, patch sha256, checkpoint and max model length as `/v1/models` reports them, quantization, served model name,
  node, chip, ROCm, torch, Triton, the sampling settings, the prompt-set hash, anything given with `--set`) and per
  prompt the text, token ids, logprobs, finish reason and token counts. What the box cannot read from the pod is given
  with flags; what is neither given nor served is null with its reason under `unavailable`.
- **Compare.** Per prompt: exact token match, first divergence index (first differing id, or where one output ends),
  the baseline's and the run's logprob there and their difference (K4 d), and a normalised-text match (NFC,
  whitespace collapsed, case kept). Summary: prompts, matches, divergences, match rate, first diverging prompt. Exit 1
  with outcome `diverged` when any prompt diverges. The full result goes to `lab/state/correctness/<run id>.json`.
- **Conditions.** The compare is refused before a request is sent when engine, engine commit, checkpoint, quantization,
  max model length, served model, chip, ROCm, torch, Triton, token kind, the sampling settings, the prompt set or any
  `--set` value that both sides know differ, unless `--allow-condition-drift` (the row then says `condition_drift`). A
  changed patch hash is the change under test, so it is reported as `under_test`, not refused. A condition the run
  does not know is listed under `unverified`.
- **Rows.** One `correctness` ledger row per call (`@recorded`; `LAB_STATE_DIR` names the ledger) with
  `teacher.output_matches`, `.output_divergences`, `.output_prompts`, `.output_match_rate` and the first diverging
  prompt in its refs; with `--metrics-file` the match rate is appended to `lab/refine/metrics.jsonl` as
  `teacher.output_match_rate` (low confidence: one compare).

The baseline in the repo was recorded 2026-10-10 8:53 to 8:58 PM ET (2026-10-11 00:53 to 00:58 UTC) from the teacher
on pod C after its restart with `--enable-auto-tool-choice --tool-call-parser qwen3_coder` (vLLM 0.19.2+gfx1151 at
b1388b1 plus `lab/vllm-gfx1201.diff`, RedHatAI/Qwen3.8-27B-INT4, bfloat16, compiled not eager, node pod-a), with R26's
batching sweep possibly loading the same server. Both commands as run, with their output:

```
$ python -m lab correctness record --account C --engine vllm --engine-commit b1388b1 --patch lab/vllm-gfx1201.diff \
    --quantization int4 --chip gfx1151 --node pod-a --rocm 7.13 --torch 2.9.1+rocm7.13.0 --triton 3.5.1 \
    --set vllm_version=0.19.2+gfx1151 --set enforce_eager=false --set dtype=bfloat16 --set serve_flags="..."
{"ok": true, "outcome": "ok", "action": "record", "fixture": ".../fixtures/lab/correctness/vllm-gfx1151-2026-10-11.json", "prompts": 4, "tokens": [128, 128, 128, 128], "token_kind": "token_id", ...}

$ python -m lab correctness compare <the same flags> --baseline fixtures/lab/correctness/vllm-gfx1151-2026-10-11.json --metrics-file
{"action": "compare", "prompts": 4, "matches": 2, "divergences": 2, "match_rate": 0.5, "normalised_text_matches": 2, "first_divergence_prompt": "image-bars", "first_divergence_index": 102, "condition_drift": false, "diverged": [{"prompt": "image-bars", "first_divergence": 102, "logprob_difference": 0.060536}, {"prompt": "text-schedule", "first_divergence": 106, "logprob_difference": 0.06076}], "ok": false, "outcome": "diverged", ...}
```

**The stock server does not match itself.** Compared with its own baseline minutes later, the unchanged teacher
reproduced two of four 128-token outputs and diverged on the other two at tokens 102 and 106, by logprob gaps of 0.06
(near-ties; a first compare before the restart diverged at token 113 on two other prompts, not kept). That is the K4 (c)
floor: until the cause is known (batch composition under R26's load, compiled-graph kernel choice, or the scheduler),
a changed build is held to 0.5 here, not 1.0, and K4's `--enforce-eager` arm and a quiet-server repeat are the next
steps. The row in `lab/refine/metrics.jsonl` records it as low confidence.

Tested against `lab/bench_fake.py` (spawned as a real process; `--diverge-at`, `--retoken-at` and `--diverge-match`
make a second server answer differently from a chosen position): record, compare equal, compare divergent, a
same-text-different-token divergence, conditions that differ (refused, nothing sent, and `--allow-condition-drift`).
