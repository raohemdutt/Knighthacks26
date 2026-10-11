# Demo numbers sheet: every lab number the slides and the video use, with its source

One sheet for the demo team (O68). For each number the "under the hood" beat, the D3 before-and-after slide, D8 and
the submission video need, it gives the value, the unit, the node and conditions, the date (UTC) and the exact
citation: a row of `lab/refine/metrics.jsonl` (cited as `metrics.jsonl:<line>`), a fixture or ledger row file, a PR,
or a section of `lab/README.md`. The owner's priority is work per joule, so the second table turns the numbers into the
sentences the slides say, with the arithmetic shown, and the third section says what the lab has proven about energy
and what is still a prediction. Every value was read from main at b9ef77c or from a branch or PR head named in the
sources; a number with no committed source is written as pending with its owner. Nothing here was measured for this
sheet, and nothing is from memory.

## Scope

- In: the teacher on the Strix Halo at batch 1 (R29), the batching frontier (R26), the teacher pod's idle watts with
  the model loaded (R29, T6, T10, R26, T8), the Jetson's idle and loaded watts (T7) and its sampler cost (T13), energy
  per answer and attribution error (T8), link speed (R14), the A100 reference run, the R9700 numbers (another chip),
  the first fine-tune (L16) and the labeling cost (L13).
- Out: kernel experiments (R23 PR #292, R47 PR #274) and the routing closed loop (T11 PR #293). All three were read at
  their heads; none carries a measured number yet (fakes, synthetic fixtures and real-run steps only).
- Node names: the teacher is account C's pod on the SJC hub (Radeon 8060S, Strix Halo, gfx1151). The R29 rows say
  node `pod-c` and every later row says `pod-a` (the teacher's role); `lab/README.md` section "Teacher on the Strix
  Halo" reads them as the same pod. Rows on `pod-a` dated before 2026-10-10 11:00 UTC are the R9700, another chip.

## Summary (table 1, every number and its source)

All Strix Halo and Jetson values are low confidence where their row says so: one run per cell, and the sensors'
own error against an external meter is unmeasured (telemetry ruling 7, rubric TT1).

| # | Number | Value | Unit | Node and conditions | Date (UTC) | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Teacher decode, text, batch 1 | 2.88 | tok/s | teacher pod (account C, SJC, gfx1151), vLLM b1388b1 + `lab/vllm-gfx1201.diff`, Qwen3.8-27B-INT4, 16k context, temperature 0, thinking off, 31 in / 77 out, streamed on the pod | 2026-10-10 19:58 | `metrics.jsonl:24`, PR #194 (R29) |
| 2 | Teacher decode, 1280x720 caption, batch 1 | 2.82 | tok/s | as row 1, 902 in / 27 out | 2026-10-10 19:58 | `metrics.jsonl:26`, PR #194 |
| 3 | Teacher TTFT, text / caption | 0.39 / 3.51 | s | as rows 1 and 2 | 2026-10-10 19:58 | `metrics.jsonl:23`, `metrics.jsonl:25` |
| 4 | Teacher watts while decoding, text | 120.3 average (35 samples), 129 peak | W | amd-smi SOCKET_POWER (whole APU package), decode window | 2026-10-10 19:58 | `metrics.jsonl:29`, `metrics.jsonl:30` |
| 5 | Teacher watts while decoding, caption | 118.3 average (13 samples) | W | as row 4 | 2026-10-10 19:59 | `metrics.jsonl:31` |
| 6 | Teacher J/token at batch 1 (R29, derived) | 41.8 | J/token | 120.3 W / 2.88 tok/s, derived here | 2026-10-10 | rows 1 and 4 |
| 7 | Teacher engine ready after launch | 297 | s | warm torch.compile cache | 2026-10-10 19:58 | `metrics.jsonl:34` |
| 8 | Frontier, 1 stream | 2.62 total (2.70 per stream), 36.66 J/token gross, 32.67 above idle, 96.0 W, TTFT 1.18 s | tok/s, J/token, W, s | R26 attach mode through the hub proxy, bench-v1, 128 tokens ignore_eos, thinking off, median of 3 repeats, T6 sampler 1 s | 2026-10-11 00:23 | `metrics.jsonl:67` to `:72`, PR #269, `lab/README.md` R26 table |
| 9 | Frontier, 4 streams | 9.32 total, 10.31 J/token gross, 9.19 above idle, 96.0 W, TTFT 4.87 s | as row 8 | as row 8 | 2026-10-11 00:23 | `metrics.jsonl:73` to `:78`, PR #269 |
| 10 | Frontier, 8 streams | 16.36 total, 5.87 J/token gross, 5.23 above idle, 96.0 W, TTFT 7.78 s | as row 8 | as row 8 | 2026-10-11 00:23 | `metrics.jsonl:79` to `:84`, PR #269 |
| 11 | Frontier, 16 streams | 26.95 total, 3.56 J/token gross, 3.18 above idle, 96.0 W, TTFT 14.95 s | as row 8 | as row 8 | 2026-10-11 00:23 | `metrics.jsonl:85` to `:90`, PR #269 |
| 12 | Frontier, 64 streams | 49.53 total (0.89 per stream), 1.94 J/token gross, 1.73 above idle, 96.0 W (98 peak), TTFT 24.39 s | as row 8 | as row 8 | 2026-10-11 00:23 | `metrics.jsonl:91` to `:96`, PR #269 |
| 13 | Teacher pod idle, model loaded, deep idle | 10.27 (stdev 0.69, 2558 samples) | W | T10, sclk 600 to 650 MHz, sampler alone, no request | 2026-10-10 20:59 to 21:41 | `metrics.jsonl:42`, PR #228 |
| 14 | Same, other reads | 10.2 (T6, 60 s), 10.44 (after R26, 60 s), 10.33 (T8, 60 s), 9.2 (R29, 20 samples) | W | teacher loaded, no request | 2026-10-10 to 2026-10-11 | `metrics.jsonl:35`, `metrics.jsonl:97`, T8 ledger row, `metrics.jsonl:32` |
| 15 | Teacher pod, no server | 8 | W | before the server started | 2026-10-10 19:36 | `metrics.jsonl:33` |
| 16 | Settle after an answer | 4 (3 to 5), 46.2 J above idle | s, J | T10/T6, 15 requests | 2026-10-10 | `metrics.jsonl:45` |
| 17 | Jetson idle (gateway and student loaded, no request of ours) | 12.62 mean (median 15.07, 4.15 to 19.17) | W | T7, VDD_IN, MAXN_SUPER, 1 s period, 297 samples; bimodal from the gateway's own captioning | 2026-10-10 22:59 | `metrics.jsonl:64`, PR #244, `fixtures/telemetry/jetson_power_live/` |
| 18 | Jetson idle, earlier 100 ms read | 15.76 mean | W | T13, 3000 samples, three quarters over 15 W | 2026-10-10 21:58 | `metrics.jsonl:63`, PR #231 |
| 19 | Jetson loaded (student captioning) | 17.536 average, 19.127 peak | W | T7, 28 back-to-back captions to llama-server :8080, Qwen3.5-4B Q4_K_M, 939 prompt tokens, 80 max, 265 samples | 2026-10-10 23:03 | `metrics.jsonl:65`, `metrics.jsonl:66` |
| 20 | Student decode speed | 13.85 (median) | tok/s | T7 load stretch, 28 requests, mean 33.1 completion tokens | 2026-10-10 23:03 | `metrics.jsonl:65` conditions, `fixtures/telemetry/jetson_power_live/caption_requests.jsonl` |
| 21 | Jetson sampler CPU cost | 3.51 idle / 3.28 loaded at 100 ms; 1.83 / 1.70 at 200 ms | percent of one core | T13, tegrastats sampler, 5 min each | 2026-10-10 | `metrics.jsonl:59` to `:62`, PR #231 |
| 22 | Teacher joules per answer | 5,101.7 mean (sd 62.9) | J | T8, 40 sequential answers of 128 tokens, greedy, thinking off, 10 s gaps, gross | 2026-10-11 01:32 to 02:09 | T8 ledger row (`fixtures/telemetry/attribution_check_2026-10-11-r40_ledger.jsonl`), `lab/T8_T0_RESULT.md`, PR #270 |
| 23 | Attribution error | -1.93 | percent | windows miss 4,014.7 J of 204,067.6 J; conservation -0.0002 J; 46 samples a window | 2026-10-11 02:09 | as row 22 |
| 24 | Link speed to the pod, push and pull | pending | MB/s | R14: 1 GB and 3 GB through the contents API and the proxy | - | PR #273 merged the tool only; no `link.push_mb_s` or `link.pull_mb_s` row is committed. Owner: the R14 task holder |
| 25 | A100 reference run (Colab) | pending | - | `work/colab-a100` is not in git; no branch or PR on origin carries it | - | Owner: Ilan (Ilan3's agent lands it) |
| 26 | R9700 teacher (another chip) | 7.7 tok/s, 200 W average, 300 W peak, about 26 J/token, idle 39 W | as stated | DUB pod A, R9700 (gfx1201), same model and engine | 2026-10-10 10:22 | `metrics.jsonl:1` to `:12`, `metrics.jsonl:17`, PR #37 |
| 27 | R9700 smoke LoRA (another chip) | 0.51 samples/s, 8.7 GiB peak, loss 2.23 to 0.16 | as stated | DUB pod B, Qwen3.5-4B, 20 steps, batch 4, 256 tokens | 2026-10-10 10:23 | `metrics.jsonl:13` to `:16`, PR #37 |
| 28 | L16 first fine-tune: loss first and last, `trainer.samples_per_s`, peak memory, wall-clock | pending (placeholder) | loss, samples/s, GiB, s | gfx1151 pod D, Qwen3.5-4B, LoRA r 16, 84 rows, 3 epochs, 18 steps | - | Read the `train` ledger row and `fixtures/demo/l16-run.json` once they land (branch `ctx/learning/first-finetune`, `lab/L16_CONFIG.md`). Owner: Ilan |
| 29 | L13 labeling cost at 4 in flight | 55.75 s per example (request time), 14.19 s wall per example, 1,405.6 tokens per example (1,304.5 prompt, 101.0 completion), 2.20 requests | s, tokens | 142 of 142 accepted, teacher through the proxy, thinking off, greedy, 4 workers, 2,014.4 s wall | 2026-10-11 00:53 to 01:27 | `fixtures/learning/tool-calling-labels/label_manifest.json` (`per_example`), PR #283 |

## Takeaways (table 2, the slide sentences with the arithmetic)

| Slide sentence | Arithmetic | Inputs | Status |
| --- | --- | --- | --- |
| "Answering 64 questions at once gets 18.9 times more work out of every joule than answering one at a time." | 36.66 J/token / 1.94 J/token = 18.9. Same as tokens per joule: 1 / 36.66 = 0.0273 against 1 / 1.94 = 0.516, or 98 against 1,857 tokens per watt-hour (3,600 / J per token). Power stays at 96 W, so throughput carries it: 49.53 / 2.62 tok/s = 18.9. Above idle (10.44 W subtracted): 32.67 / 1.73 = 18.9 | rows 8 and 12 (`metrics.jsonl:70`, `:94`, `:68`, `:92`) | measured (median of 3, low confidence) |
| "One answer costs about 5,100 J on the teacher pod and about 160 J on the Jetson: about 31 times less." | Teacher: T8 mean 5,101.7 J for a 128-token answer. Student: 128 tokens / 13.85 tok/s = 9.24 s; 9.24 s x 17.536 W = 162.1 J. Ratio: 5,101.7 / 162.1 = 31.5 | rows 19, 20 and 22 | teacher measured; student derived (prediction) |
| "Batched, the big teacher's answer drops to about 250 J, close to the Jetson's." | 128 tokens x 1.9384 J/token = 248.1 J at 64 streams, against 128 x 36.66 = 4,692 J at 1 stream (T8 measured 5,101.7 J at 1 stream including prefill and a 111 W window) | rows 8, 12 and 22 | derived from measured rows |
| "Labeling all 142 training examples took 34 minutes and about 193 kJ (54 Wh) on one pod, about 1.4 kJ per example." | 2,014.4 s wall x 96.01 W (R26, 4 streams) = 193,408 J = 53.7 Wh; / 142 = 1,362 J per example. Decode-only cross-check: 14,343 completion tokens x 10.31 J/token = 147,835 J | rows 9 and 29 (`metrics.jsonl:77`, `:76`) | prediction: no power was sampled during the L13 run |
| "Waiting with the model loaded costs the teacher pod 10 W, a ninth of its working draw." | 96.0 W busy / 10.27 W deep idle = 9.3 | rows 8 and 13 | measured |

## Conclusions (what the lab has proven about energy tonight, and what is still a prediction)

Proven tonight, on our own nodes, with committed rows (each still low confidence because no external meter checked
the sensors):

- The teacher pod draws a flat 96 W from 1 to 64 concurrent streams, so joules per token fall 18.9 times as the
  batch grows from 1 to 64 (R26, PR #269).
- The answer windows hold 98 percent of the energy the pod spends: the attribution error is -1.93 percent and answers
  plus gaps equal the whole run to 0.0002 J (T8, PR #270). Joules per answer are trustworthy with that error stated.
- One 128-token teacher answer at batch 1 costs 5,101.7 J (T8).
- With the model loaded and nothing asked, the teacher pod sits at about 10 W; five separate reads agree (9.2 to
  10.44 W).
- The Jetson draws 17.5 W while the student captions and 12.6 W between requests, which is not a flat floor: the
  gateway's own captioning swings it between about 5 and 18 W (T7).

Still a prediction:

- The student's joules per answer (162 J) and the 31x teacher-to-student ratio: derived from the Jetson's loaded
  watts and decode speed, not measured by running T8's prompts on the Jetson; the two models also differ in size and
  the two sensors measure different things (APU socket against the board's input rail).
- The energy of labeling the fixture (193 kJ): the L13 run sampled no power, so the watts are R26's.
- Training energy: L16 has not run, and its config plans no power sampler.
- How far the sensors' joules sit from a wall meter (TT1).

## Teacher at batch 1 on the Strix Halo

- **[measured, n=1]** R29 measured the teacher at batch 1 on account C's Strix Halo pod (Radeon 8060S, gfx1151, vLLM b1388b1 with `lab/vllm-gfx1201.diff`, Qwen3.8-27B-INT4, 16k context, temperature 0, thinking off): 2.88 tok/s decode and 0.39 s TTFT for a 31-token text prompt with 77 out, 2.82 tok/s and 3.51 s for a 1280x720 caption (902 in, 27 out), at 120.3 W average (35 samples) and 129 W peak while decoding text and 118.3 W for the caption, read from amd-smi SOCKET_POWER on 2026-10-10 at 19:58 UTC. That is about 41.8 J per token (120.3 / 2.88, derived). Source: [S62] `lab/refine/metrics.jsonl:23` to `:31`; [S63] PR #194.
  - Implication: these are the "under the hood" teacher numbers; joules per token at batch 1 is set by a near-fixed 120 W over a weight-bound 2.9 tok/s, so batching (next section) and not a single request is where the slide's energy story lives.
  - Candidate experiment: repeat R29's two prompts five times each on the running teacher with the T6 sampler. Expected movement: decode within 5 percent of 2.88 tok/s and watts within 5 W of 120 W, turning the single run into a median with a spread. Minimum sample size: 5 runs per prompt.

## Batching frontier (R26)

- **[measured, n=3]** R26 swept the running teacher in attach mode through the hub proxy (bench-v1, 4 requests per stream, 128 tokens with ignore_eos, thinking off, T6 sampler at 1 s, medians of 3 repeats) on 2026-10-11 00:23 UTC: total throughput 2.62, 9.32, 16.36, 26.95 and 49.53 tok/s at 1, 4, 8, 16 and 64 concurrent streams, at a flat 96.0 W average (97 to 98 W peak), so gross joules per token were 36.66, 10.31, 5.87, 3.56 and 1.94 (32.67, 9.19, 5.23, 3.18 and 1.73 above the 10.44 W idle). Per-stream speed held from 2.70 to 2.07 tok/s up to 16 streams and fell to 0.89 at 64, where TTFT reached 24.39 s. Source: [S62] `lab/refine/metrics.jsonl:67` to `:96`; [S64] PR #269; [S69] `lab/README.md` section "Batching sweep on the pod C teacher".
  - Implication: work per joule at 64 streams is 18.9 times that at 1 stream (36.66 / 1.94), the headline of the work-per-joule slide; the rows are low confidence because 3 repeats are under the 5 that verdict() needs.
  - Candidate experiment: rerun the 1- and 64-stream cells with 5 repeats in the same attach mode. Expected movement: the 18.9x multiple holds within 10 percent and the rows lose their low-confidence flag. Minimum sample size: 5 repeats per cell.

## Idle watts of the teacher pod with the model loaded

- **[measured, n=2558]** T10 read the teacher pod at deep idle (teacher loaded, no request, sclk 600 to 650 MHz, sampler alone) at 10.27 W mean (stdev 0.69) over 2558 one-second samples, 2026-10-10 20:59 to 21:41 UTC; T6 read 10.2 W over 60 s, the read after R26 10.44 W over 60 s, T8's idle window 10.33 W over 60 s, and R29 9.2 W over 20 samples; with no server the pod read 8 W. Source: [S62] `lab/refine/metrics.jsonl:42`, `:35`, `:97`, `:32`, `:33`; [S57] PR #228; [S65] `fixtures/telemetry/attribution_check_2026-10-11-r40_ledger.jsonl`.
  - Implication: the idle stored beside every joules-per-answer value is 10.3 W (telemetry ruling 1 keeps joules gross); the busy draw of 96 W is 9.3 times it, so a pod that waits loaded costs little next to one that works.
  - Candidate experiment: read 10 minutes of deep idle before and after the next labeling run with the sampler alone. Expected movement: none beyond 0.5 W from 10.3 W; a larger drift flags a second reader or a clock-state change. Minimum sample size: 2 windows of 600 samples.

## Jetson watts and sampler cost (T7, T13)

- **[measured, n=297]** T7 read the Jetson's VDD_IN rail in MAXN_SUPER at one sample a second on 2026-10-10: 12.62 W mean (median 15.07, 4.15 to 19.17 W) over 297 samples with the gateway and student loaded and no request of ours, and 17.536 W mean (19.127 W peak) over 265 samples while 28 back-to-back caption requests ran (Qwen3.5-4B Q4_K_M, 939 prompt tokens, 80 tokens at most, median decode 13.85 tok/s, mean 33.1 completion tokens); the load stretch held 4,670.9 J, which is 166.8 J per caption. Source: [S62] `lab/refine/metrics.jsonl:64` to `:66`; [S59] PR #244; `fixtures/telemetry/jetson_power_live/caption_requests.jsonl`.
  - Implication: the student's watts and speed are the inputs of the per-answer comparison in table 2; the idle mean is not a floor, because the gateway's own captioning moves the rail between about 5 and 18 W.
  - Candidate experiment: send T8's two text prompts (128 tokens, ignore_eos, greedy) to the Jetson's llama-server one at a time with 10 s gaps while the sampler runs, and attribute joules per answer as T8 does. Expected movement: a measured student joules per answer near the derived 162 J (within 25 percent), replacing the prediction. Minimum sample size: 40 requests.
- **[measured, n=1]** T13 measured the Jetson sampler's own CPU at 3.51 percent of one core idle and 3.28 percent under caption load at a 100 ms period, and 1.83 and 1.70 percent at 200 ms, one 5-minute run per cell on 2026-10-10; an earlier 100 ms VDD_IN read gave 15.76 W mean idle over 3000 samples. Source: [S62] `lab/refine/metrics.jsonl:59` to `:63`; [S58] PR #231; `fixtures/telemetry/jetson_sampler_cost/`.
  - Implication: sampler overhead is over the 1 percent budget at both periods; the sheet cites it as the cost of watching the box, and the slide should not claim a free meter.
  - Candidate experiment: run the sampler at the ruled 350 ms period (E30) for 5 minutes on and 5 off. Expected movement: sampler CPU down to about 1 percent of one core. Minimum sample size: 3 runs per state.

## Energy per answer and attribution error (T8)

- **[measured, n=40]** T8 sent 40 sequential 128-token answers to the teacher (greedy, thinking off, ignore_eos, 10 s gaps, nothing else running per vLLM's counters) on 2026-10-11 from 01:32 to 02:09 UTC with the pod sampler at 1000 ms: 5,101.7 J mean gross per answer (sd 62.9; 39.9 J per output token; 45.8 s and about 111 W per answer window), 204,067.6 J over the answers, 8,269.4 J over the gaps and 212,337.0 J over the whole run, with conservation off by -0.0002 J; the windows missed 4,014.7 J, an attribution error of -1.93 percent, with a median 46 samples a window. Source: [S65] PR #270, `lab/T8_T0_RESULT.md`, `fixtures/telemetry/attribution_check_2026-10-11-r40_ledger.jsonl`.
  - Implication: joules per answer can go on a slide with "-1.9 percent, the windows read low" beside it; the miss sits in the 3 to 5 s settle tail, so widening each window by the settle time would recover most of it.
  - Candidate experiment: rerun T8 with each window widened by 5 s. Expected movement: attribution error from -1.93 percent to within plus or minus 0.5 percent. Minimum sample size: 40 requests.

## Labeling cost (L13)

- **[measured, n=142]** L13 labeled all 142 examples of the tool-calling fixture with the teacher through the hub proxy (thinking off, greedy, 4 in flight) on 2026-10-11 from 00:53 to 01:27 UTC, 142 accepted: 55.75 s of request time per example, 14.19 s of wall time per example (2,014.4 s in all), 1,304.5 prompt and 101.0 completion tokens per example (199,589 tokens in all), 2.20 requests and 1.20 tool calls per example. No power was sampled. Source: [S66] PR #283, `fixtures/learning/tool-calling-labels/label_manifest.json` (`per_example`).
  - Implication: at R26's 96 W the run would have cost about 193 kJ (2,014.4 s x 96.01 W), 1,362 J per example; that energy is a prediction until a labeling run is sampled.
  - Candidate experiment: run the label-agreement spot-check (30 examples, 4 in flight) with the T6 sampler on and attribute the run's energy. Expected movement: joules per example within 20 percent of 1,362 J. Minimum sample size: 30 examples.

## The other chip (R9700)

- **[measured, n=1]** On the R9700 (DUB pod A, gfx1201) on the morning of 2026-10-10 the same model and engine decoded at 7.7 tok/s with thinking on (7.8 off, 6.6 for a caption) at about 200 W average and 300 W peak, about 26 J per token (derived), with 39 W idle (a stated range, low confidence); the smoke LoRA on DUB pod B ran 0.51 samples/s with 8.7 GiB peak and loss 2.23 to 0.16. Source: [S41] PR #37; [S62] `lab/refine/metrics.jsonl:1` to `:17`.
  - Implication: these numbers are another chip's and must not sit on a Strix Halo slide without that label; the Strix Halo decodes about 2.7 times slower (7.7 / 2.88) at about 60 percent of the power (120.3 / 200 W).
  - Candidate experiment: none on this chip (the DUB pods are not the lab's any more); compare only if an R9700 returns. Expected movement: none for the metric set. Minimum sample size: 1 run per prompt if it does.

## Gaps (pending numbers and their owners)

- Link speed to the pod (R14): `link.push_mb_s` and `link.pull_mb_s` for 1 GB and 3 GB. PR #273 merged `python -m lab link-speed`; no run's ledger rows or `lab/R14_REPORT.md` are committed on main or on any branch on origin. Owner: the R14 task holder (plumbing pass 3 read the R14 page as handed to Hemdutt's agents, Sat 5:23 PM ET; entry S67 of `lab/refine/research/plumbing/SOURCES.md`, not this folder's S67).
- A100 reference run from Colab: `work/colab-a100` is not in git and no branch or PR on origin carries it. Owner: Ilan (Ilan3's agent lands it).
- L16 training numbers: the first and last loss, `trainer.samples_per_s`, peak memory and wall-clock come from the `train` ledger row and `fixtures/demo/l16-run.json` once the run lands from `ctx/learning/first-finetune`; until then the only figure is L16_CONFIG's prediction of 0.06 to 0.19 samples/s (22 to 70 minutes for 252 examples). Training watts stay null: the config runs no pod sampler. Owner: Ilan.
- A measured student joules per answer on the Jetson (the experiment under T7 above). Owner: the telemetry track holder (T7 and T8 author).
- A power-sampled labeling run (the experiment under L13 above). Owner: the L13 task holder.

## Open questions

- Does the teacher's flat 96 W under R26 mean a package power limit, and would the single-stream 120 W of R29 return outside attach mode? `lab/README.md` says it "looks like" a steady limit; no read of the limit exists.
- The open PRs R47 (#274), R23 (#292) and T11 (#293) may produce kernel and routing numbers later; the sheet should be refreshed when they land.

## Sources (IDs in this folder's SOURCES.md)

- S41: PR #37, the R9700 lab setup (existing entry).
- S57: PR #228, T10 idle baseline (existing entry).
- S58: PR #231, T13 Jetson sampler cost (existing entry).
- S59: PR #244, T7 Jetson live run (existing entry).
- S62: `lab/refine/metrics.jsonl` at b9ef77c.
- S63: PR #194, R29 teacher on the Strix Halo.
- S64: PR #269, R26 batching sweep.
- S65: PR #270, T8 attribution validation, with `lab/T8_T0_RESULT.md` and its fixtures.
- S66: PR #283, L13 labels, with `fixtures/learning/tool-calling-labels/label_manifest.json`.
- S67: PR #273, the R14 link-speed verb (no run's rows committed).
- S68: `lab/L16_CONFIG.md` on `ctx/learning/first-finetune` at c81a659.
- S69: `lab/README.md` at b9ef77c.
- S70 to S72: the open PRs #274, #292 and #293 (read, no numbers taken).

## Search log

- 2026-10-11, budget about 2 h, used about 1 h. Read `/home/user/atlas-tooling/cloud/tasks/O68.md`, `lab/refine/RESEARCH_FORMAT.md`, every row of `lab/refine/metrics.jsonl` grouped by metric name, `lab/README.md` (Jetson T7 and T13, teacher, Strix Halo teacher, idle baseline, T8, trainer, R14, labeling, R26, L13, L16 and L22 sections), `lab/T8_T0_RESULT.md`, the T8, T7 and T13 fixtures, the L13 label manifest, the telemetry notes and `SOURCES.md`, and the plumbing measurements record. Fetched origin: `ctx/telemetry/attribution-validation`, `ctx/runtime/link-speed` and `ctx/learning/desk-fixture-clips` no longer exist (T8, R14 and L22 merged as PRs #270, #273 and #199); read `ctx/learning/first-finetune` at c81a659 and `ctx/telemetry/routing-closed-loop` (PR #293) at 4ff33e0, and PRs #274 and #292 at their heads. Searched every ref on origin for an R14 report, a `*_RESULT.md` and any Colab or A100 file: only T8's result exists. Stopped when every number the task names had a row or a pending owner.
