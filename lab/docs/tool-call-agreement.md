# Tool-call agreement: `python -m lab tool-call-agreement` (L4)

Arguments: `python -m lab tool-call-agreement -h`. The module is `lab/tool_call_agreement.py`.

The second measure of a fine-tune, beside the eval gate. The gate's scorer (`contexts/agent/eval.py`) asks the golden
questions, and every one of them takes the agent's prefetch path, so the gate never scores a tool call. This verb asks
what the fine-tune was trained on: given the prompt the trainer gave the student for an example, does the student
write the reference tool call?

```
python -m lab tool-call-agreement --dataset fixtures/learning/tool-calling-fixture --run-id <run id> \
    --before-url http://127.0.0.1:8100 --after-url http://127.0.0.1:8101 --out <folder> \
    [--side held_out|train] [--before-runs 2] [--after-runs 1] [--limit N] [--before-sha256 H] [--after-sha256 H]
```

## The rule

The rule is the constant `RULE` of the module, written into every file a run leaves before any model is asked:

- **Input.** The training text of the example up to its first assistant span: `lab/format_traces.py` `render()` of
  the dataset's instruction (system) and the example's context (user), with the agent's `TOOLS` and the generation
  prompt (`<|im_start|>assistant`, a newline, the empty thinking block). It is sent unchanged as the prompt of
  llama-server `POST /completion` (512 tokens, temperature 0.2 as the box's client sends; greedy comes from the
  server's `--top-k 1`, and the verb refuses a server whose `/props` reports another `top_k`). The completion endpoint
  is used because the trainer trains on rendered text, not on chat messages; both arms get the identical text.
- **Output.** The generated tokens as text (the server's `/detokenize` of the returned token ids, so the template's
  own tokens are kept), cut at the first `<|im_end|>`. Nothing else is stripped or repaired.
- **Calls.** `lab/format_traces.py` `parse_tool_calls()` on that text, the parser that proves the training rows.
- **Score.** 1 when `lab/routing_closed_loop.py` `agreement()` says so: the calls' names and argument objects equal
  the reference's (`expected.jsonl`, `label.tool_calls`), in order and in number. Else 0.
- **Secondary marks.** `names`: the list of tool names equals the reference's. `parsed`: the parser read at least one
  call.

The before model is asked `--before-runs` times (default 2): the spread of its rates is the run-to-run noise, as in
the gate.

## The two tests, and what each supports

- **Exact McNemar** on the paired examples (first before run against first after run): descriptive. It says whether
  the change on this set is more than a coin flip between the discordant examples. The examples are rewordings of a
  few questions, so they are not independent draws and this p-value is not a claim about new questions.
- **Exact sign-flip over the question groups** (`lab/eval_gate.py` `sign_flip_p`, groups by `frame_ref`, each
  example's mean over the before runs against its mean over the after runs): conservative, the gate's own test. With
  the committed split's 4 held-out groups its smallest possible value is 0.125.

## What it leaves

In `--out`: `raw/<side>.<arm>.jsonl` (a header line with the rule, then one line per example: id, group, the prompt's
sha256, the output text, the parsed calls, the reference calls, the three marks), `<side>.summary.json` and
`<side>.md`. The summary follows the gate's `scores.json` where the fields fit (`correct`, `noise_runs`, `alpha`,
`test`, `p_value`, `items` with `id`, `group`, `question`, `expected`, `before`, `after`) and adds `mcnemar`,
`sign_flip`, `flips`, `names`, `parsed` and `after_runs`. It carries no `learning.run.evaluated` event and nothing
imports it into the box. One ledger row (verb `tool-call-agreement`): `learning.tool_call_agreement.before`, `.after`,
`.delta`, `.noise`, `.p_sign_flip`, `.p_mcnemar`, `.gained`, `.lost`, `.n_items`, `.n_groups`, with `refs.run_id` and
the split's sha256.

Tests run on fakes: `tests/test_lab_tool_call_agreement.py`.

## The first real run (Sun 2026-10-11, 2:35 to 2:45 AM ET, pod D)

Run `train-20261011T032149Z-7dd098` (the L16 adapter, packaged by L5 as Q4_K_M). Both models on pod D's Strix Halo,
llama-server (HIP build, llama.cpp 23b0202a) with the box's flags of `jetson/atlas-llm.sh` plus `--top-k 1`. The
before model is the base Qwen3.5-4B rebuilt on pod D through the package's own path (loaded in bf16 and saved, the
same converter and quantizer; sha256 `84284aa2899f112244a6735d9fbb0c6f94085fb59eb719c95ec376e421eee04f`), not the
file on the Jetson. The after model is the package's GGUF (sha256
`b48886c0655005f59fbbee9097215a64549b230cb00dcf5b744cd5d19b18ea6d`).

| side | arm | exact call | tool name | parsed |
| --- | --- | --- | --- | --- |
| held out (58, 4 groups) | before, run 1 | 42/58 = 0.724 | 55 | 58 |
| held out | before, run 2 | 42/58 = 0.724 | 54 | 58 |
| held out | after, run 1 | 34/58 = 0.586 | 56 | 58 |
| held out | after, run 2 | 33/58 = 0.569 | 55 | 58 |
| train (84, 12 groups) | before | 56/84 = 0.667 | 76 | 84 |
| train | after | 57/84 = 0.679 | 63 | 84 |

Held out: difference -0.147, before noise 0.000; flips 0 from wrong to right and 8 from right to wrong; McNemar
p = 1/128 = 0.0078; sign-flip over the 4 groups p = 0.25. Train: 16 up, 15 down; McNemar p = 1; sign-flip over 12
groups p = 0.93.

What makes the number weaker than it looks:

- Seven of the eight losses are one question in several wordings ("in the last 60 minutes": the after model opens
  the window at 9:00 or 9:10 instead of 8:30). The eighth adds a second call.
- The reference is `expected.jsonl`, the fixture's own rule. The adapter was trained on the teacher's labels of the
  train examples, which were deleted with the dataset when the run ended. On the train side the after model adds a
  `search_memory` call to 19 of the 24 `last_seen` questions and agrees with `expected.jsonl` no more often than the
  base, so the labels it learned from very likely differ from this reference. The measure then shows that the
  adapter moved away from `expected.jsonl`, not that it failed to learn its labels.
- The score is an exact match: 38 of the 58 held-out references are time windows, and a window that differs by any
  amount is 0.
- Greedy decoding was not fully repeatable: 56 of 58 before outputs and 55 of 58 after outputs were the same text in
  both runs.
- 4 held-out groups, two tools (`episodes_between`, `last_seen`), one call per reference, both models at Q4_K_M.
