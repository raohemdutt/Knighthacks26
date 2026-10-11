# Request-side settings sweep on gfx1151 (pod-b), 2026-10-11

Run `sweep-20261011T052418Z-e9c2` from 2026-10-11T05:24:18.210Z to 2026-10-11T06:07:07.471Z: one server spawn, the pinned launch, ready after 391.675 s; 16 cells, 3 repeats each in serpentine order, one run of a cell being B concurrent requests of the cell's settings. Every value is the median of the repeats, with the lowest and highest in brackets. Grid `lab/refine/research/kernels/data/sweep-request-grid.json` (sha256 da8885db24e3).

Server: engine commit b1388b1fbf5aaef47937fabe98931211684666a6, checkpoint /home/jovyan/models/Qwen3.8-27B-INT4, quantization compressed-tensors, max model length 16384, linear kernels {'CompressedTensorsWNA16': 'ConchLinearKernel'}, attention ['ROCM_ATTN'], KV cache 150528 tokens; ROCm None, torch 2.9.1+rocm7.13.0, Triton 3.5.1. Idle 20.5553 W with no server and 23.4457 W with it loaded; watts are amd-smi SOCKET_POWER (the whole APU package) over each decode window.

Baseline cell: `base-b1`, the pinned baseline of lab/refine/research/kernels/data/searchspace-inference-gfx1151.json (R35): server launched from its baseline mapping and fixed launch; request at batch 1, thinking off, temperature 0, the harness's 128 tokens with ignore_eos, a short text prompt.

| Cell | Batch | Thinking | Max tokens | Prompt (tokens in) | Decode tok/s per stream | Decode tok/s total | TTFT s | Watts avg | J/token | decode_tok_s_total vs baseline | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base-b1` (baseline) | 1 | off | 128 | short (126) | 2.74 (2.73-2.85) | 2.76 (2.75-2.87) | 0.44 (0.42-0.45) | 96.0 (96.0-110.0) | 34.95 (34.73-38.29) | +0.0% | 3 of 3 |
| `base-b4` | 4 | off | 128 | short (118) | 2.70 (2.69-2.81) | 10.73 (10.64-11.06) | 1.56 (1.23-1.62) | 96.0 (96.0-110.0) | 9.03 (8.95-9.94) | +288.2% | 3 of 3 |
| `base-b8` | 8 | off | 128 | short (118) | 2.61 (2.61-2.71) | 19.97 (19.96-20.76) | 3.18 (3.03-3.19) | 96.0 (96.0-110.0) | 4.81 (4.81-5.30) | +622.6% | 3 of 3 |
| `base-b16` (best) | 16 | off | 128 | short (120) | 2.51 (2.50-2.60) | 36.30 (36.01-37.64) | 6.64 (6.13-6.75) | 96.0 (96.0-109.9) | 2.67 (2.64-2.92) | +1213.3% | 3 of 3 |
| `tokens-32-b1` | 1 | off | 32 | short (131) | 2.75 (2.75-2.77) | 2.84 (2.84-2.86) | 0.72 (0.71-0.72) | 96.0 (96.0-102.0) | 33.82 (33.82-35.67) | +2.7% | 3 of 3 |
| `tokens-32-b4` | 4 | off | 32 | short (122) | 2.71 (2.67-2.71) | 10.24 (10.21-10.54) | 1.64 (1.24-1.90) | 96.0 (96.0-97.7) | 9.41 (9.11-9.54) | +270.5% | 3 of 3 |
| `tokens-32-b8` | 8 | off | 32 | short (120) | 2.62 (2.58-2.62) | 17.71 (17.51-17.71) | 3.21 (3.21-3.22) | 96.0 (96.0-97.8) | 5.42 (5.42-5.59) | +540.6% | 3 of 3 |
| `tokens-32-b16` | 16 | off | 32 | short (124) | 2.52 (2.52-2.52) | 27.70 (27.69-27.75) | 7.04 (6.78-7.08) | 96.1 (96.0-96.9) | 3.47 (3.47-3.49) | +902.1% | 3 of 3 |
| `thinking-on-32-b1` | 1 | on | 32 | short (168) | 2.73 (2.72-2.74) | 2.81 (2.81-2.83) | 0.74 (0.74-0.74) | 96.0 (96.0-96.0) | 34.12 (33.91-34.19) | +1.8% | 3 of 3 |
| `thinking-on-32-b4` | 4 | on | 32 | short (156) | 2.70 (2.69-2.70) | 9.89 (9.87-9.91) | 2.30 (2.29-2.30) | 96.0 (96.0-96.1) | 9.71 (9.69-9.73) | +257.6% | 3 of 3 |
| `thinking-on-32-b8` | 8 | on | 32 | short (157) | 2.61 (2.61-2.61) | 16.68 (16.65-16.73) | 4.37 (4.34-4.37) | 96.0 (95.9-96.0) | 5.76 (5.74-5.76) | +503.6% | 3 of 3 |
| `thinking-on-32-b16` | 16 | on | 32 | short (160) | 2.35 (2.34-2.36) | 25.07 (25.01-25.10) | 7.79 (7.79-7.83) | 96.1 (96.0-96.1) | 3.83 (3.83-3.84) | +806.9% | 3 of 3 |
| `prompt-long-b1` | 1 | off | 128 | long (1214) | 2.60 (2.59-2.60) | 2.62 (2.61-2.62) | 4.19 (4.19-4.20) | 96.0 (96.0-96.0) | 36.66 (36.63-36.72) | -5.3% | 3 of 3 |
| `prompt-long-b4` | 4 | off | 128 | long (1213) | 2.37 (2.35-2.45) | 8.89 (8.53-9.10) | 15.06 (12.91-15.11) | 96.0 (96.0-96.1) | 10.80 (10.55-11.27) | +221.5% | 3 of 3 |
| `prompt-long-b8` | 8 | off | 128 | long (1213) | 2.06 (1.97-2.11) | 13.38 (12.55-13.69) | 24.75 (23.32-25.78) | 96.0 (96.0-96.1) | 7.18 (7.01-7.65) | +384.2% | 3 of 3 |
| `prompt-long-b16` | 16 | off | 128 | long (1216) | 1.57 (1.56-1.57) | 18.03 (17.93-18.09) | 40.46 (40.12-40.68) | 96.0 (96.0-96.0) | 5.32 (5.31-5.35) | +552.4% | 3 of 3 |

Best cell by decode_tok_s_total: `base-b16` at 36.30 (36.01-37.64) against the baseline's 2.76 (2.75-2.87): +1213.3% over the baseline, +1153.7% to +1270.2% from the two cells' lowest and highest repeats.
Lowest joules per token: `base-b16` at 2.67 J/token (-92.4% against the baseline).
Held: sampling = greedy: the note marks no sampling setting as changing decode work on the pinned server: its only sampling line, KS2, is the MTP acceptance rate, and MTP (num_speculative_tokens) is a restart setting the pinned baseline keeps at 0; a profile under sampling and a slice that sets it adds the axis with no code change.
Held: prefix caching = on, with a per-request nonce at the front: a restart setting (the note's KS9); the harness's nonce keeps every measured request from hitting the cache.
Held: requests per stream = 1: batch B is B concurrent requests started together, so the cell measures B requests decoding at once.
