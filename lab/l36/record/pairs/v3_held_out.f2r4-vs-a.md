# Tool-call agreement, held_out side (240 examples, 20 groups)

Run l36-v3_held_out-f2r4-vs-a, dataset tool-calling-v3, split sha256 a054c84f4c20, made 2026-10-11T12:43:31.952714+00:00.

| arm | exact call | tool name | parsed as a call |
| --- | --- | --- | --- |
| after-1 | 180/240 = 0.750 | 240/240 | 240/240 |
| before-1 | 179/240 = 0.746 | 240/240 | 240/240 |

Difference (after minus before, means over runs): +0.004; before run-to-run noise 0.000.
Flips, first before run to first after run: 10 from 0 to 1, 9 from 1 to 0.
McNemar, exact two-sided, on the 240 paired examples: p = 1. Descriptive: the examples are rewordings of a few questions, so they are not independent draws.
Sign-flip, exact two-sided, over the 20 question groups (the gate's test): p = 1; its smallest possible value with 20 groups is 1.907e-06.

## The rule

Input: the training text of the example up to its first assistant span, that is lab/format_traces.py render() of [system = the dataset's instruction, user = the example's context] with the agent's TOOLS and the generation prompt (<|im_start|>assistant, a newline, the empty thinking block), sent unchanged as the prompt of llama-server POST /completion (n_predict 512, temperature 0.2 as the box's client sends, sampling left to the server's --top-k 1). Both arms get the identical prompt text. Output: the generated tokens as text (the server's /detokenize of the returned token ids when it returns them, else the content field), cut at the first <|im_end|>; nothing else is stripped or repaired. Calls: lab/format_traces.py parse_tool_calls() on that text with the same TOOLS; a FormatError is not parsed. Score 1 when lab/routing_closed_loop.py agreement() says so: the calls' names and argument objects equal the reference's (expected.jsonl label.tool_calls), in order and in number; else 0. Secondary marks: names (the list of tool names equals the reference's, in order) and parsed (the parser read at least one call). Pairing: the flip table and McNemar use the first before run and the first after run; the sign-flip test pairs each example's mean over the before runs with its mean over the after runs and tests the frame_ref groups' mean differences, as lab/eval_gate.py does. Noise is the spread (max minus min) of the before runs' rates.

## Models

- before: {"model": "Qwen3.5-4B bf16 transformers A100, arm f2r4"}
- after: {"model": "Qwen3.5-4B bf16 transformers A100, arm a"}

## Flips

0 to 1: w-437b482137, w-e91cb56aa6, w-63e04cf207, w-cdeab265fc, w-f095b29533, w-ddad699ea2, w-8cd3705411, w-b8f9f525f5, w-4b701017f4, w-bd47b45bd8

1 to 0: w-a92ab1d30d, w-e9bf6e97e6, w-b68b745804, w-9b03e4c6a8, w-a105bae14a, w-d0dad962e5, w-94c5ff81c8, w-5168dc8bc2, w-2e03c1af61
