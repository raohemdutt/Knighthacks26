# Tool-call agreement, held_out side (228 examples, 18 groups)

Run l36-v2_held_out-base-vs-a, dataset tool-calling-v2, split sha256 4776164b103c, made 2026-10-11T12:43:32.235443+00:00.

| arm | exact call | tool name | parsed as a call |
| --- | --- | --- | --- |
| after-1 | 179/228 = 0.785 | 228/228 | 228/228 |
| before-1 | 136/228 = 0.596 | 221/228 | 228/228 |

Difference (after minus before, means over runs): +0.189; before run-to-run noise 0.000.
Flips, first before run to first after run: 48 from 0 to 1, 5 from 1 to 0.
McNemar, exact two-sided, on the 228 paired examples: p = 7.077e-10. Descriptive: the examples are rewordings of a few questions, so they are not independent draws.
Sign-flip, exact two-sided, over the 18 question groups (the gate's test): p = 0.0004883; its smallest possible value with 18 groups is 7.629e-06.

## The rule

Input: the training text of the example up to its first assistant span, that is lab/format_traces.py render() of [system = the dataset's instruction, user = the example's context] with the agent's TOOLS and the generation prompt (<|im_start|>assistant, a newline, the empty thinking block), sent unchanged as the prompt of llama-server POST /completion (n_predict 512, temperature 0.2 as the box's client sends, sampling left to the server's --top-k 1). Both arms get the identical prompt text. Output: the generated tokens as text (the server's /detokenize of the returned token ids when it returns them, else the content field), cut at the first <|im_end|>; nothing else is stripped or repaired. Calls: lab/format_traces.py parse_tool_calls() on that text with the same TOOLS; a FormatError is not parsed. Score 1 when lab/routing_closed_loop.py agreement() says so: the calls' names and argument objects equal the reference's (expected.jsonl label.tool_calls), in order and in number; else 0. Secondary marks: names (the list of tool names equals the reference's, in order) and parsed (the parser read at least one call). Pairing: the flip table and McNemar use the first before run and the first after run; the sign-flip test pairs each example's mean over the before runs with its mean over the after runs and tests the frame_ref groups' mean differences, as lab/eval_gate.py does. Noise is the spread (max minus min) of the before runs' rates.

## Models

- before: {"model": "Qwen3.5-4B bf16 transformers A100, arm base"}
- after: {"model": "Qwen3.5-4B bf16 transformers A100, arm a"}

## Flips

0 to 1: o-c884fd302f, o-23e6f7935e, o-89cdc3e202, o-5ac2ecb0a7, o-4e334e5366, o-f248f121ea, o-cb786f76a8, w-15b25c0eb9, w-dca51bc6a0, w-f0daeb168d, w-58a5e1fdd7, w-0517941efc, w-0def67e89e, w-cca9a9fd46, w-81dc3da01d, w-6d4e92fb62, w-f75948349d, w-9027593373, w-fe20e436f9, w-df069b1648, w-3bca195684, w-85911d9dbf, w-e9805c22bb, w-48e69b9101, w-406848aaf1, w-469da9df29, w-eb0764f1ba, w-f0af522104, w-4710ef35da, w-dc932e34b9, w-87d5d27bc5, w-028e736144, w-50b0911619, w-538407f287, w-581b217979, w-1709fb66cc, w-b14e1b8c26, w-3d09858222, w-6611bae8d7, w-3130e5945a, w-2bc3522bfd, w-691f388ba4, w-a62db50c6d, w-7df1ecbfbe, w-c5a8fb4f36, w-4a378cfa1a, w-c32ed45833, w-751a1bcc3f

1 to 0: w-e8f07de679, w-0f69550e26, w-27d804ff01, w-055f1a7393, w-1567cfd4eb
