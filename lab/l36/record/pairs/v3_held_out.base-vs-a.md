# Tool-call agreement, held_out side (240 examples, 20 groups)

Run l36-v3_held_out-base-vs-a, dataset tool-calling-v3, split sha256 a054c84f4c20, made 2026-10-11T12:43:31.830707+00:00.

| arm | exact call | tool name | parsed as a call |
| --- | --- | --- | --- |
| after-1 | 180/240 = 0.750 | 240/240 | 240/240 |
| before-1 | 109/240 = 0.454 | 235/240 | 240/240 |

Difference (after minus before, means over runs): +0.296; before run-to-run noise 0.000.
Flips, first before run to first after run: 79 from 0 to 1, 8 from 1 to 0.
McNemar, exact two-sided, on the 240 paired examples: p = 8.378e-16. Descriptive: the examples are rewordings of a few questions, so they are not independent draws.
Sign-flip, exact two-sided, over the 20 question groups (the gate's test): p = 3.052e-05; its smallest possible value with 20 groups is 1.907e-06.

## The rule

Input: the training text of the example up to its first assistant span, that is lab/format_traces.py render() of [system = the dataset's instruction, user = the example's context] with the agent's TOOLS and the generation prompt (<|im_start|>assistant, a newline, the empty thinking block), sent unchanged as the prompt of llama-server POST /completion (n_predict 512, temperature 0.2 as the box's client sends, sampling left to the server's --top-k 1). Both arms get the identical prompt text. Output: the generated tokens as text (the server's /detokenize of the returned token ids when it returns them, else the content field), cut at the first <|im_end|>; nothing else is stripped or repaired. Calls: lab/format_traces.py parse_tool_calls() on that text with the same TOOLS; a FormatError is not parsed. Score 1 when lab/routing_closed_loop.py agreement() says so: the calls' names and argument objects equal the reference's (expected.jsonl label.tool_calls), in order and in number; else 0. Secondary marks: names (the list of tool names equals the reference's, in order) and parsed (the parser read at least one call). Pairing: the flip table and McNemar use the first before run and the first after run; the sign-flip test pairs each example's mean over the before runs with its mean over the after runs and tests the frame_ref groups' mean differences, as lab/eval_gate.py does. Noise is the spread (max minus min) of the before runs' rates.

## Models

- before: {"model": "Qwen3.5-4B bf16 transformers A100, arm base"}
- after: {"model": "Qwen3.5-4B bf16 transformers A100, arm a"}

## Flips

0 to 1: o-1a9e54e9a0, o-c8a5b5dc53, o-fe8b7cedc6, w-7922e80f3a, w-0521b95849, w-5d272b5590, w-98047d81e2, w-a83015d099, w-9ebe3aba23, w-af825bf3df, w-69ce182303, w-c972e29bac, w-c0f778be15, w-437b482137, w-c2c8872d2c, w-af3249b078, w-450597207e, w-90a1a1e17b, w-805a297f5b, w-e4477a174f, w-185327ba12, w-a4f8a33f7e, w-63e04cf207, w-66263a7b26, w-5441d8bd33, w-2061643d33, w-71f795be09, w-cdeab265fc, w-40d6a34fb3, w-cd1f99d963, w-c470a619c8, w-c6462c298e, w-f095b29533, w-47582af68d, w-3fd0d93bc9, w-ca01fd7060, w-97b71079cc, w-a736949087, w-7f44d17d41, w-2967bae34d, w-e7572f07ff, w-ffb23dd738, w-ed7a2e1735, w-acf36a274f, w-b7f914e6cf, w-84fa1a1d90, w-1e86c29154, w-ddad699ea2, w-f91a7f06cf, w-ad48a89b53, w-180688cfd3, w-8396b93ad0, w-8cd3705411, w-0cd2155cd6, w-f5f589e1f5, w-3322cab73d, w-a46c871a86, w-5ae59dcf4e, w-6aa9bd8061, w-9762f3633a, w-449f8a95be, w-b8f9f525f5, w-1ccf10b9e7, w-b2fc56d18a, w-421478dc93, w-2e04842f6b, w-52c7f9f780, w-ca43e425ad, w-4b701017f4, w-c898d312dd, w-ebf8ab85b4, w-04d49fed53, w-645bcca21d, w-ddfb27d632, w-62483a60db, w-e0c4a14c29, w-422a92043b, w-641815989d, w-b688fbba6d

1 to 0: w-a92ab1d30d, w-e9bf6e97e6, w-b68b745804, w-f4f918c53f, w-94c5ff81c8, w-5168dc8bc2, w-01dd8d4fa7, w-2e03c1af61
