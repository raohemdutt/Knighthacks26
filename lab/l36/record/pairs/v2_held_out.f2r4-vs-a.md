# Tool-call agreement, held_out side (228 examples, 18 groups)

Run l36-v2_held_out-f2r4-vs-a, dataset tool-calling-v2, split sha256 4776164b103c, made 2026-10-11T12:43:32.371519+00:00.

| arm | exact call | tool name | parsed as a call |
| --- | --- | --- | --- |
| after-1 | 179/228 = 0.785 | 228/228 | 228/228 |
| before-1 | 174/228 = 0.763 | 228/228 | 228/228 |

Difference (after minus before, means over runs): +0.022; before run-to-run noise 0.000.
Flips, first before run to first after run: 8 from 0 to 1, 3 from 1 to 0.
McNemar, exact two-sided, on the 228 paired examples: p = 0.2266. Descriptive: the examples are rewordings of a few questions, so they are not independent draws.
Sign-flip, exact two-sided, over the 18 question groups (the gate's test): p = 0.125; its smallest possible value with 18 groups is 7.629e-06.

## The rule

Input: the training text of the example up to its first assistant span, that is lab/format_traces.py render() of [system = the dataset's instruction, user = the example's context] with the agent's TOOLS and the generation prompt (<|im_start|>assistant, a newline, the empty thinking block), sent unchanged as the prompt of llama-server POST /completion (n_predict 512, temperature 0.2 as the box's client sends, sampling left to the server's --top-k 1). Both arms get the identical prompt text. Output: the generated tokens as text (the server's /detokenize of the returned token ids when it returns them, else the content field), cut at the first <|im_end|>; nothing else is stripped or repaired. Calls: lab/format_traces.py parse_tool_calls() on that text with the same TOOLS; a FormatError is not parsed. Score 1 when lab/routing_closed_loop.py agreement() says so: the calls' names and argument objects equal the reference's (expected.jsonl label.tool_calls), in order and in number; else 0. Secondary marks: names (the list of tool names equals the reference's, in order) and parsed (the parser read at least one call). Pairing: the flip table and McNemar use the first before run and the first after run; the sign-flip test pairs each example's mean over the before runs with its mean over the after runs and tests the frame_ref groups' mean differences, as lab/eval_gate.py does. Noise is the spread (max minus min) of the before runs' rates.

## Models

- before: {"model": "Qwen3.5-4B bf16 transformers A100, arm f2r4"}
- after: {"model": "Qwen3.5-4B bf16 transformers A100, arm a"}

## Flips

0 to 1: w-3c9cb972d4, w-a31e177be8, w-81dc3da01d, w-fe20e436f9, w-e9805c22bb, w-406848aaf1, w-87d5d27bc5, w-028e736144

1 to 0: w-c975567521, w-20b0fc9d4a, w-935d3801b6
