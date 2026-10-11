# Joules per token on the gfx1151 teacher: the energy model, the measurement protocol and the optimizer objective

What one generated token costs in socket energy on the SJC Strix Halo pods (Radeon 8060S, gfx1151, RDNA 3.5, 64 GB
shared, ROCm 7.13, torch 2.9.1+rocm7.13.0, Triton 3.5.1, Python 3.12), predicted from the measured chip sheet and
checked against the one measured batch curve on main. It covers the 27B INT4 teacher (`RedHatAI/Qwen3.8-27B-INT4`
served by vLLM 0.19.2 at b1388b1 plus `lab/vllm-gfx1201.diff`, "the April build") and the 4B bf16 student
(`Qwen/Qwen3.5-4B`). Every prediction below is printed by `python lab/energy_model.py` from the committed chip sheet
`data/chipsheet-gfx1151-2026-10-11.json`, so it can be recomputed; nothing here was measured for this note, and no
pod, GPU or Jetson was touched.

## Summary

Decode on this chip is weight-read bound at every batch the box uses, so the socket draws about the same watts at
batch 1 as at batch 16 while the tokens per second grow almost in proportion to the batch, and joules per token falls
nearly as 1 over the batch until a knee. The model predicts 41.7, 10.6, 5.4 and 2.8 J/token at batch 1, 4, 8 and 16
for the April build decoding alone at a 512-token context, and 8.1, 2.2, 1.2 and 0.7 J/token for a build whose weight
reads reach the chip sheet's 210.5 GB/s copy ceiling. For the teacher at that ceiling the curve flattens near batch 45
to 52 (the per-sequence state knee at 45, the int4 GEMMs turning compute-bound at 52); for the bf16 student near 72 to
201. Against R26's measured attach sweep on pod C (36.7 J/token at one stream down to 1.94 at 64) the throughput side
of the model holds within 7 % at every batch, and the watts side misses: the sweep read a flat 96.0 W where the model,
fed the April build's 120 W smoke reading, predicted 120 to 129 W. Anchored on the sweep's own batch-1 point, every
batch holds within 7 %. The optimizer's objective is gross `joules_per_token` at batch 1 with `decode_tok_s >= 2.88`
and `ttft_s <= 2.0` as outcome constraints, and a labeling lane at batch 8 that must not lose total throughput.

| Build | Workload | Batch 1 | Batch 4 | Batch 8 | Batch 16 | Knee (batch) |
| --- | --- | --- | --- | --- | --- | --- |
| Teacher, April build (2.88 tok/s, 120 W at batch 1) | decode only, 512-token context | 41.67 | 10.60 | 5.42 | 2.83 | 128 (the kernel's row tile) |
| Teacher, April build | harness bench-v1 (4 x 128 tokens a stream) | 42.80 | 12.03 | 6.90 | 4.36 | as above |
| Teacher, ceiling build (210.5 GB/s, 110 W) | decode only, 512-token context | 8.08 | 2.16 | 1.17 | 0.68 | 45 to 52 |
| Teacher, ceiling build | harness bench-v1 | 9.13 | 3.47 | 2.53 | 2.08 | as above |
| Student bf16, ceiling build (102.4 W) | decode only, 512-token context | 4.15 | 1.08 | 0.57 | 0.32 | 72 to 201 |

Values are gross J/token (socket energy over tokens, nothing subtracted). The harness rows are higher than decode
alone because `lab/bench.py`'s window holds the prefills of every request after the first.

## The model

At B concurrent streams, with P(B) the socket watts while decoding,

    J/token(B) = (P_dynamic(B) + P_idle) / tok_s(B),   tok_s(B) = B / t_step(B)

and one decode step is the sum of its kernels, each at its own roofline:

    t_step(B) = sum over weight classes c of max(tiles(B) * bytes_c / BW_w, B * flops_c / C) + B * seq_bytes / BW

Here bytes_c is a weight class read once per step (int4 values plus one bf16 scale per group of 128 for W4A16, two
bytes a parameter for bf16), flops_c is two times its parameters, seq_bytes is what each sequence adds per step (its
Gated DeltaNet recurrent state read and written in fp32, and the KV cache of the full-attention layers at the current
context), BW is the chip sheet's copy bandwidth beyond the 32 MiB last-level cache, C the sheet's dense matmul rate,
and BW_w the weight-read bandwidth of the build. tiles(B) is 1 for a kernel that serves the whole batch from one weight
read and ceil(B / m_tile) for one whose weight pass serves m_tile rows. The power side is
P(B) = (1 - u) * P_bw(1) * (1 + s * (B - 1)) + u * P_compute, with u the share of the step in compute-bound GEMMs,
s the chip sheet's median slope of loaded watts against M (0.12 % per added row), P_bw(1) the batch-1 watts and
P_compute the sheet's compute-bound GEMM watts (120.8 W). Below the knee P is nearly flat and J/token falls as 1 over
B; past it tok_s stops rising and J/token flattens. P_idle is the loaded idle, which the gross metric keeps and the
above-idle metric subtracts.

- **[measured]** The chip sheet measured on pod D gives the inputs: device copy beyond the cache at a median of
  210.5 GB/s (943 GB/s inside it), the best bandwidth-bound matmul weight read at 219.5 GB/s, a dense GEMM rate of
  42.3 TFLOP/s in bf16 and 38.3 in fp16, and median loaded socket watts of 75.0 W for copies, 102.4 W for the fp16 and
  bf16 matmuls, 110.0 W for the W4A16 dequant kernel and 120.8 W for the compute-bound LoRA GEMMs, with loaded watts
  nearly flat from M 1 to 16 (0.12 % per added row). Source: [L10] `data/chipsheet-gfx1151-2026-10-11.md` and its
  JSON, the copy, matmul, dequant and lora sections, read by [L14] `lab/energy_model.py` `chip_inputs`.
  - Implication: weights are the cost. The teacher reads 15.13 GB of weights a step and does 51.2 GFLOP a token, the
    student 8.41 GB and 8.4 GFLOP, so at batch 1 both sit far below the compute roof and the watts barely move with
    batch; every joule saved per token comes from reading the weights fewer times per token (batching) or faster
    (a better kernel).
  - Candidate experiment: EM1, a batch sweep at 1, 4, 8, 16, 32 and 64 on a cold spawn of the rebuilt teacher with
    the T6 sampler at 1 s, read with `python -m lab.energy_model check <rows.jsonl>` against the ceiling build.
    Expected movement: joules per token at batch 16 within 15 % of the model's prediction for the build's measured
    batch-1 rate, and per-stream decode within 15 % at every batch up to 32. Minimum sample size: 3 runs per batch
    size. Confidence: medium.

- **[measured]** The knees from the chip sheet directly: a measured shape turns compute-bound at the batch where C
  over its batch-1 rate is reached, which is 64 to 71 for the plain Triton W4A16 dequant kernel (block 64 by 64), 175
  to 202 for the fp16 and bf16 teacher-shaped matmuls, and 215 to 362 for the other shapes. Applied to the models, the
  teacher at the ceiling has its int4 GEMMs turn compute-bound at batch 52 and its per-sequence traffic (0.30 GB of
  recurrent state plus 64 KB of KV per context token) match the shared weight read at batch 45; the bf16 student's
  GEMMs turn at 201 and its state (0.10 GB) at 72; the April build is held by its kernel's 128-row tile before
  either. Source: [L10] the chip sheet's matmul and dequant rows; [L14] `sheet_knees` and `knees`.
  - Implication: for the teacher at the ceiling, batch 16 is still on the steep part (0.68 J/token decoding alone) and
    batching past about 48 buys almost nothing (0.35 J/token at 52 and at 64); for the student the curve keeps falling
    to about 72. The knee for an energy-optimal labeling job is therefore near 48 streams for the teacher, bounded in
    practice by the 120 s request timeout, not by the chip.
  - Candidate experiment: EM2, the same sweep as EM1 extended to 32, 48 and 64 streams, on the rebuilt teacher only.
    Expected movement: total tok/s rises less than 15 % from 48 to 64 and joules per token falls less than 10 %.
    Minimum sample size: 3 runs per batch size. Confidence: low.

- **[practitioner]** The ML.ENERGY Benchmark measures inference energy automatically with the Zeus library, and Zeus
  reads AMD GPUs through AMDSMI, which it supports only on ROCm 6.3 or later because older versions returned wrong
  power and energy values; where `current_socket_power` reads N/A it falls back to the average socket power.
  Source: [S62] the ML.ENERGY Benchmark paper (title and abstract: automated inference energy measurement); [S181] Zeus
  `docs/measure/index.md` at 42d76ac, section "ROCm and AMDSMI Versions" and the AMD GPU paragraphs.
  - Implication: our pods run ROCm 7.13, so the AMDSMI power path is inside the supported range; the gfx1151 socket
    reports SOCKET_POWER while energy and clocks read N/A, so energy has to be integrated from power samples rather
    than read from an energy counter, which is what `lab/pod_power.py` and `lab/bench.py` do.
  - Experiment: none beyond EM3.

## Checked against the measured frontier

- **[measured]** R26's attach sweep against the live teacher on pod C (April build, bench-v1, three repeats) read
  2.62, 9.32, 16.36, 26.95 and 49.53 tok/s total at 1, 4, 8, 16 and 64 concurrent streams with the socket at a flat
  96.0 W in every window, so 36.66, 10.31, 5.87, 3.56 and 1.94 J/token gross. The 82.4 W "loaded idle" in that
  session's rows was not the idle: it caught requests of an earlier attempt of the sweep that were still decoding; the
  real loaded idle, read after the sweep with the teacher up and nothing running, was 10.44 W. Source: [L15]
  `lab/README.md`, section "Batching sweep on the pod C teacher (R26, Strix Halo, gfx1151)"; [L16] the trimmed rows
  `fixtures/lab/energy_model/attach-sweep-pod-c-2026-10-10.jsonl`.
  - Implication: `python lab/energy_model.py check` on those rows shows where the model holds and where it misses.
    Stated as the April build (2.88 tok/s and 120 W at batch 1), the predicted total tok/s is within 7 % at every
    batch (ratios 0.93 to 0.97), the watts miss by 20 to 26 % (120.0 to 129.2 W predicted against 96.0 measured),
    and so J/token holds at batch 1 and 4 (ratio 0.86) and misses at 8, 16 and 64 (0.85, 0.82, 0.77). Anchored on the
    file's own batch-1 point (96 W, 2.62 tok/s), every metric holds within 7 % at every batch: tok/s 0.99 to 1.00,
    watts 0.93 to 1.00, J/token 0.94 to 1.01. The batch dependence is right; the miss is the watts input. The 120 W came
    from a single-stream smoke earlier the same day and the chip sheet's kernel classes drew 102 to 121 W, while the
    live server sat at 96 to 97 W through every window, which looks like the package held at a steady power limit.
    At 64 the anchored model still lets the watts climb toward the 120.8 W of compute-bound GEMMs (103.4 W
    predicted) where the meter stayed at 96, so past the knee the model overstates energy by about 6 %.
  - Candidate experiment: EM3, read the package power limit and the power state on the pod before and after a sweep
    (`amd-smi metric --power` and the readable `power1_cap` under the card's hwmon) and run batch 1 for 60 s both
    attached to the live teacher and on a cold spawn. Expected movement: if a limit near 96 W explains the flat
    reading, both runs read within 3 W of each other and of the cap. Minimum sample size: 2 runs per mode.
    Confidence: medium.

- **[measured]** The R52 request-side sweep on pod D, which the Done when names as the second check, has no result on
  main yet: `lab/refine/metrics.jsonl` holds no `sweep_request` line and no result file is committed. The checker reads
  its result format directly (schema `atlas.sweep_request/1`, the base slice at batch 1, 4, 8 and 16). Source: [L17]
  `lab/README.md`, section "Request-side settings sweep: `lab/sweep_request.py` (R52)"; [L14] `measured_points`.
  - Implication: the R52 reading is open, not skipped; when its result lands, one CPU command gives the table, and
    the note gains a finding with the ratios.
  - Candidate experiment: EM4, `python -m lab.energy_model check lab/refine/research/kernels/data/sweep-request-gfx1151-<date>.json`
    once R52's pod run is collected. Expected movement: anchored J/token within 15 % at 4, 8 and 16 streams. Minimum sample size:
    the grid's 3 repeats per cell. Confidence: medium.

## The measurement protocol

Which watts: the socket power amd-smi reports as SOCKET_POWER on the pod, sampled every second by the T6 sampler
(`lab/pod_power.py serve`, read through the hub proxy) for the harness, or every 0.25 s by the chip sheet runner. It
is the whole package (CPU, GPU and memory controller share it on this APU), so it is gross by construction. Which
window: `lab/bench.py` integrates the samples with the trapezoid rule from the first token of the first request to the
last token of the last, interpolating at both edges, and divides by the tokens generated in that window; the window
holds the prefills of every request after the first, which is why the harness rows sit above decode alone. How idle
is handled: the gross `joules_per_token` subtracts nothing and is the objective; `joules_per_token_above_idle`
subtracts a loaded idle read for 60 s with the model loaded and no request in flight, and that read is taken after the
sweep as well as before, because a read before can catch an earlier client's requests (R26's 82.4 W). The chip sheet's
per-section idles (24 to 55 W, read 3 s after the section before) are not settled and are not subtracted. What the
harness records on every row: batch size, requests per stream, max tokens, prompt set, per-request time to first
token and token counts, decode tok/s per stream and total, average and peak watts, loaded idle, both J/token fields,
and the conditions (engine commit, environment, node, hub and account). Comparisons hold one prompt set, max tokens,
requests per stream and thinking setting fixed across arms, and report medians over at least three runs.

## Objective and constraints for the optimizer

The night optimizer (`data/ax-config-night.json`, the R49 experiment) minimizes `joules_per_token.b1`, the harness's
gross field at batch 1, with two outcome constraints: `decode_tok_s.b1 >= 2.88`, so no arm wins by answering slower
per token than the April build, and `ttft_s.b1 <= 2.0`, so the first token of a live answer stays within 2 s (the April
build's batch-1 median is 1.04 to 1.37 s through the proxy). It tracks decode rate, TTFT and average watts beside the
objective. A second lane, for bulk teacher labeling, fixes batch 8 and asks that a lower J/token does not come with a
lower total tok/s than the incumbent, that TTFT stays under 60 s, and that every request finishes inside the
runtime's 120 s timeout. The same definition is `OBJECTIVE` in `lab/energy_model.py`, so the config and the code say
the same thing. Gross rather than above-idle is the objective because the meter's idle on a shared, live pod is the
least reliable number in the row, and gross is what the wall pays.

## Candidate experiments, ranked

1. EM1, the batch sweep on the rebuilt teacher, read against the ceiling build: it measures the after number.
2. EM4, the R52 result read with the checker: one CPU command once the pod run is collected.
3. EM3, the power limit read: settles whether the flat 96 W is a cap, which decides the watts input of every
   prediction.
4. EM2, the extended sweep to 64 streams: places the knee for the labeling lane.

## Open questions

- Whether the pod's package runs at a fixed power limit near 96 W under sustained decode (EM3), and if so whether a
  faster kernel would draw the same 96 W, which would make J/token fall exactly as 1 over tok/s.
- What the rebuilt teacher (R53, superseded by R60) actually reaches at batch 1; the ceiling build is an upper bound,
  not a forecast.

## Search log

- 2026-10-11, budget 3 h. Inputs were the committed chip sheet, R26's README section and its rows (trimmed into the
  fixture), `lab/bench.py`, `lab/pod_power.py`, and the in-repo model configs encoded in `lab/energy_model.py`. One
  external source opened: Zeus at 42d76ac through `git clone --depth 1` (its rendered site was not opened); the
  ML.ENERGY paper is cited under its existing id. A web search for Zeus AMDSMI ROCm 6.3 returned the project's GitHub
  and PyPI pages, which pointed to the docs file cited. No pod, GPU or Jetson was touched.
