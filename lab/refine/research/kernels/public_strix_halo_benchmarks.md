# What every published LLM benchmark on a Strix Halo says, and the decode prior fitted from it

This note describes `data/public-strix-halo-benchmarks.csv`: one row per published measurement of an LLM
running on an AMD Strix Halo (Ryzen AI Max+ 395 and its siblings, Radeon 8060S, gfx1151) that this search
could open. It then fits the prior in `lab/public_prior.py` from those rows. The prior gives decode tokens
per second from bytes read per token, backend, context depth and batch. Every predicted number below is a
prior, to be calibrated by our own trials on the pods, never a measurement. Nothing here ran on a pod, a
GPU or the Jetson. Kernel changes are covered in `w4a16_decode_gfx1201.md`, the runtime settings in
`settings_inference_gfx1151.md` and the chip in `strix_halo_hardware.md`.

## Summary

- **The data.** 13,291 rows from 86 sources (P1 to P86 in `SOURCES.md`), dated 2025-03-14 to 2026-10-09.
  - Quality flags: 12,972 reproducible-script, 314 forum-post, 5 vendor, 0 review.
  - Engines: llama.cpp 12,729 (Vulkan 4,538, ROCm or HIP 8,092 of which 2,998 with rocWMMA), vLLM 163,
    Ollama 122, lemonade 84, halogen-flash 73, SGLang 49, ExLlamaV3 11, LM Studio 10, MLX 5, and 45 others
    (CPU-only, NPU, distributed).
  - The columns follow the Done-when list exactly and stay empty where the source is silent. The share of
    rows that fill each column is printed by the commands in the search log.
- **The prior.** It is a bandwidth-bound decode model fitted by weighted least squares, with no boosting:
  log(tok/s) = a[backend] - beta log(GB per token) - g log(1 + depth in thousands) + m [MoE], plus a
  batch contention term.
  - Fit on 7,821 single-stream rows from 40 sources: R^2 0.877 in log space, a typical error of a factor
    of 1.38, and a 16.5% median error when each source is left out in turn.
  - beta is 0.84: decode is mostly bandwidth-bound, with a fixed cost that hurts small models.
- **What it says about us.** For the teacher (15.5 GB read per token) the prior gives 7.5 tok/s on vLLM and
  12.3 (ROCm) or 14.3 (Vulkan) on llama.cpp at batch 1.
  - We measure 2.88 tok/s on vLLM, 38% of the prior.
  - Public llama.cpp runs of dense 27B Qwen models in Q4_K_M decode at 11.2 (ROCm) to 12.1 (Vulkan)
    tok/s on the same chip.

## The dataset and how it was built

- **[measured, compiled]** The CSV holds 13,291 rows. Each row records:
  - the source URL and date, the machine and memory split, OS, kernel, driver and ROCm or Vulkan version;
  - the engine and its version, backend, model, parameters, quantization, context, batch or concurrency,
    and prompt and generation length;
  - every setting the source states, and prompt-processing and decode tokens per second;
  - TTFT, watts and joules per token where given, a quality flag, and notes that say how to read the row.

  Five search passes, one per source area, wrote rows from pages they opened. The merge then dropped
  2,771 duplicate measurements: the same repository, model, quantization, backend and rates seen twice,
  mostly where two passes parsed the same repository. It also dropped 36 rows that carried no numeric
  rate. Non-numeric values such as a stated range were moved into the notes column rather than averaged.
  - Source: `lab/refine/research/kernels/data/public-strix-halo-benchmarks.csv`; [P1, P2, P3] carry 84% of
    the rows.
  - Implication: almost every row is a llama-bench or batched-bench line with its command. The data is
    strongest on llama.cpp and weakest on vLLM, which is what the teacher runs. The prior's llama.cpp terms
    rest on thousands of rows, while its vLLM term rests on six (section "The prior fitted from the rows").
  - Candidate experiment: PB0, rerun the merge and the fit after every new source is added
    (`python lab/public_prior.py --fit`). Expected movement: none on our metrics; the vLLM effective
    bandwidth should move by more than 20% only when a new vLLM source lands. Minimum sample size: 1 run
    per added source. Confidence: high (0.9) that the llama.cpp terms stay within 5%.

- **[measured]** Two large sweeps dominate.
  - lhl's llm-bench sweeps [P1] contribute 5,257 rows over 15 models with HIP, rocWMMA, AMDVLK and RADV,
    hipBLASLt on and off, and flash attention 0 and 1. They also record per-run APU power, which supplies
    5,167 of the 5,501 rows that carry watts.
  - kyuz0's toolbox results [P2] contribute 5,222 rows across 35 git versions of `docs/results.json`, ROCm
    6.4.2 to 7.2.x and nightlies, at depths to 65,536.
  - Each row in the fit is weighted 1/sqrt(rows from its source), so these two count as about 72 rows each,
    not 5,000.
  - Source: [P1] lhl/strix-halo-testing; [P2] kyuz0/amd-strix-halo-toolboxes.
  - Implication: the settings effects below are mostly within-source A/B pairs from these two sweeps. They
    are reproducible, but they come from a few machines (Framework Desktop 128 GB, Fedora and CachyOS).
  - Candidate experiment: PB0 (above).

## The prior fitted from the rows

- **[measured]** On 7,821 single-stream decode rows from 40 sources, the prior's coefficients are:

  | Coefficient | Value |
  | --- | --- |
  | beta | 0.841 |
  | effective bandwidth at 1 GB per token, llama.cpp ROCm | 123.8 GB/s |
  | effective bandwidth at 1 GB per token, llama.cpp Vulkan | 143.0 GB/s |
  | effective bandwidth at 1 GB per token, vLLM | 75.1 GB/s |
  | effective bandwidth at 1 GB per token, SGLang | 71.4 GB/s |
  | MoE factor | 0.789 |
  | depth term g | 0.085 per unit of log(1 + depth in thousands) |
  | R^2 of log(tok/s) | 0.877 |
  | RMS error of log(tok/s) | 0.32 (a factor of 1.38) |
  | median absolute error, each source left out in turn | 16.5% |

  How the rows are read:
  - Bytes per token are the stated file size times the active parameter share. Failing a size, they are
    parameters times bits.
  - Rows kept out of the fit: multi-node RPC, speculative or MTP, offline total-throughput rows, and models
    whose architecture the row does not establish (GLM-4.5-Air, dots.llm1, MiniMax and Gemma's E-models).

  Rows per backend: llama.cpp ROCm 5,390, Vulkan 2,416, SGLang 9, vLLM 6.
  - Source: `lab/public_prior.py::fit` on `lab/refine/research/kernels/data/public-strix-halo-benchmarks.csv`;
    the rows from [P1, P2, P3, P4, P5, P6].
  - Implication: a beta below 1 means that halving the bytes read less than doubles decode, so a small
    student gains less from quantization than bandwidth alone would say.
  - The vLLM and SGLang terms are each fitted on fewer than ten rows from four sources: P3, P15, P67 and
    P1 for vLLM, and P14 for SGLang. They are the weakest numbers in the table.
  - Candidate experiment: PB1 (section "Prediction check against our own numbers") calibrates the vLLM
    term on our own teacher. Expected movement: the vLLM effective bandwidth for our checkpoint falls to
    about 45 GB/s at 15.5 GB per token unless the kernel changes. Minimum sample size: 5 runs of 256
    generated tokens. Confidence: medium (0.6).

- **[measured]** Effective bandwidth (decode tok/s times GB read per token) has these medians on dense
  single-stream rows above 10 GB:
  - llama.cpp ROCm: 204 GB/s (1,000 rows);
  - llama.cpp Vulkan: 211 GB/s (280 rows);
  - vLLM: 116 GB/s (3 rows).

  Single rows reach 311 GB/s on ROCm. A row's stated size can exceed the bytes actually read, for example
  when Ollama reports a size that includes buffers [P47].
  - Source: the CSV, computed by
    `lab/public_prior.py` row by row; [P1, P2].
  - Implication: about 200 GB/s is what a well-tuned decode path reaches on this chip in public. Our
    teacher moves about 44.6 GB/s ([strix_halo_hardware.md section 5](strix_halo_hardware.md#5-memory-and-the-cpu-gpu-split)),
    so the kernel path, not the memory, sets our decode speed. AMD's rated bandwidth for the 395 was not
    opened, so no fraction of peak is stated here.
  - Candidate experiment: none beyond PB1 and R33's SH7 bandwidth measurement.

## Backend and build settings the sources show matter

- **[measured]** Vulkan and ROCm compared on the same source, model, quantization and depth:
  - Decode: Vulkan is a median 1.04x ROCm (163 matched groups, range 0.32 to 1.54).
  - Prompt processing at 512 tokens: Vulkan is 0.86x ROCm (91 groups).

  - Source: the CSV, matched pairs computed over [P1, P2, P4, P6].
  - Implication: for a decode-bound teacher, the backend matters less than the kernel. ROCm wins on prompt
    processing, which is what TTFT and the vision prompt pay for.
  - Candidate experiment: PB5, the teacher's Q4_K_M GGUF under llama.cpp Vulkan and ROCm at batch 1, with a
    512-token and a 31-token prompt. Expected movement: Vulkan decode +4%, Vulkan prompt processing -14%.
    Minimum sample size: 5 runs per arm. Confidence: medium (0.6).

- **[measured]** Three build settings have no measurable effect on decode in matched pairs:
  - rocWMMA against plain ROCm: decode 1.00x (211 groups), prompt processing at 512 tokens 1.006x (209).
  - hipBLASLt off against on: decode 1.00x, prompt processing 0.998x (303 groups).
  - Flash attention on against off: decode 0.997x, prompt processing 1.04x (204 to 206 groups).

  RADV against AMDVLK decode is 1.027x (351 groups).
  - Source: the CSV, matched within source, model,
    quantization, depth and engine build, over [P1, P2].
  - Implication: none of these four belongs in the decode search space for llama.cpp. For vLLM they are
    untested in public, so the R35 shortlist stands as it is.
  - Candidate experiment: none for decode. Experiment: PB5 already covers the one backend contrast worth a
    slot.

## Context depth

- **[measured]** Decode at depth relative to depth 0, on the same source, model and backend:
  - 16k: 0.87 (398 rows)
  - 32k: 0.76 (1,354 rows)
  - 64k: 0.74 (381 rows)

  The fitted term is g = 0.085 per unit of log(1 + depth in thousands): -13% at 4k and -26% at 32k.
  - Source: the CSV, depth from the llama-bench `d` label; [P2, P4, P1].
  - Implication: the teacher's 16k-context answers should decode about 20% slower than a short prompt. The
    16k setting on pod C costs decode only when the context is actually filled.
  - Candidate experiment: PB3, teacher decode at depth 0, 4k, 16k and 32k on pod C. Expected movement:
    decode -13%, -21% and -26% against depth 0 if the public shape holds for vLLM. Minimum sample size:
    3 runs per depth. Confidence: medium (0.5).

## Batching and concurrency

- **[measured]** The batch term pairs 449 batch-or-concurrency rows with a batch-1 row of the same source,
  model, quantization and backend. The fitted contention k (0 is perfect scaling, 1 is no gain) is:
  - llama.cpp Vulkan: 0.31 (332 pairs);
  - llama.cpp ROCm: 0.19 (85 pairs);
  - vLLM: 0.10 (9 pairs);
  - SGLang: 0.08 (23 pairs).

  One SGLang source runs our own student, Qwen/Qwen3.5-4B in BF16: 23.1 tok/s at one stream, 85.8 tok/s
  aggregate at 4, and 159.9 at 8.
  - Source: [P14] JeremiahM37/strix-halo-sglang bench results; [P8]
    ggml-org/llama.cpp#25356 and [P16] ggml-org/llama.cpp discussion 15290 for llama.cpp batched-bench.
  - Implication: on this chip a batching server gains nearly linearly to 8 streams. The teacher serving
    label jobs at concurrency 8 should give several times its single-stream aggregate, which matters more
    for the labelling night than any single-stream setting.
  - Candidate experiment: PB2, teacher at concurrency 1, 4, 8 and 16 on pod C with the R11 harness.
    Expected movement: aggregate decode about 3.0x at 4 and 4.6x at 8 against concurrency 1 (k = 0.10).
    Minimum sample size: 3 spawns per level. Confidence: low (0.4), because the vLLM k rests on 9 pairs
    from two sources, 8 of them a 0.6B model.

- **[measured]** llama.cpp Vulkan has a batched-decode cliff on many-expert MoE models: aggregate decode
  falls at about 9 parallel sequences. One gpt-oss-120b sweep gains only 0.59x to 0.76x at 4 and 8
  sequences under Vulkan, against 1.99x and 2.85x under ROCm.
  - Source: [P8] ggml-org/llama.cpp#25356; [P16]
    ggml-org/llama.cpp discussion 15290.
  - Implication: the Vulkan k is pulled up by MoE rows and does not apply to the dense teacher. A dense
    model should be judged by the ROCm k until PB2 runs.
  - Candidate experiment: none beyond PB2. Experiment: PB2 covers the dense case.

## Power and energy per token

- **[measured]** 2,642 decode-only rows carry watts:
  - median 72.3 W during decode;
  - median 2.18 J per token (derived by the sources or from watts over tok/s);
  - 5,167 of the 5,501 rows with watts come from one source's APU telemetry, and hogeheer's density
    rows give package PPT, not wall power.

  - Source: [P1] lhl/strix-halo-testing llm-bench; [P3] hogeheer499-commits/strix-halo-guide.
  - Implication: our teacher draws 120 W while decoding at 2.88 tok/s, about 41.7 J per token
    ([L4](../../../README.md)). Public decode runs sit near 72 W with far more tokens per second, so our
    joules per token is high mainly because decode is slow. The metric moves with tok/s, not with a
    power cap.
  - Candidate experiment: PB6, read SOCKET_POWER during PB1's llama.cpp runs. Expected movement: about
    10 J per token at about 12 tok/s if the package stays near 120 W, a 75% cut. Minimum sample size: 5 runs
    per arm. Confidence: medium (0.5).

## Public numbers for our two model classes

- **[measured]** Dense 27B Qwen models (Qwen3.6-27B and Qwen3.8-27B) at batch 1, depth 0, decode at these
  medians:
  - llama.cpp Q4_K_M: 11.2 tok/s on ROCm (6 rows) and 12.1 on Vulkan (9 rows);
  - Q8_0 or Q8_K_XL: 6.4 to 7.8 tok/s;
  - Ollama and lemonade with Q4: 17.5 to 18.3 tok/s (one row each). One of those states a range and one
    states MTP neither way, so they are not used as a baseline.

  - Source: [P5] ihanesman/strix-halo-windows-llm-bench, [P4], [P6] slb350/strix-benchmarks, [P49]
    kinncj/halomarchy.
  - Implication: a 4-bit 27B on this chip decodes at about 11 to 12 tok/s in llama.cpp, four times our vLLM
    teacher. Switching the teacher's serving path is the largest single change the public data points to.
    It is a decision for Ilan, not for this note.
  - Candidate experiment: PB1 (next section).

- **[anecdote]** One vLLM source runs a dense 27B AWQ-INT4 (group 32) checkpoint, cyankiwi/Qwen3.6-27B-AWQ-INT4,
  with a custom HIP prefill kernel:
  - 5.64 tok/s per stream with speculative decoding off;
  - 4.3 tok/s for the BF16 checkpoint;
  - 18 to 25 tok/s with DFlash speculative decoding at 8 tokens, which is out of the fit.

  The source is one repository with patched kernels and a single machine, so it is an anecdote, not a
  baseline.
  - Source: [P15] hec-ovi/vllm-awq4-qwen.
  - Implication: vLLM W4A16 on gfx1151 is slow in public too. 5.64 tok/s is 2x our 2.88, which suggests our
    Conch path leaves about half on the table even among vLLM kernels. Speculative decoding is the biggest
    lever this source shows, which agrees with R35's MTP shortlist row.
  - Candidate experiment: none new; R35's KS2 (MTP depth) tests it. Expected movement: decode +30% to
    +120%, as that note states. Minimum sample size: 5 cold spawns per arm. Confidence: low (0.4).

- **[measured]** Public rows that run our student, Qwen3.5-4B:
  - llama.cpp Q8_0: 36.3 tok/s on ROCm and 37.9 to 40.0 on Vulkan [P6];
  - SGLang BF16: 23.1 tok/s at one stream [P14].

  - Source: [P6] slb350/strix-benchmarks; [P14] JeremiahM37/strix-halo-sglang.
  - Implication: the prior under-predicts the small model on the serving engines. For 8 GB BF16 it gives
    13.1 tok/s on vLLM against SGLang's measured 23.1, because beta and the vLLM term are fitted mostly on
    large models. For the student, the public rows beat the prior.
  - Candidate experiment: PB4, the student in BF16 on vLLM at batch 1, 4, 8 and 16 on pod D. Expected
    movement: 20 to 25 tok/s at batch 1 (SGLang's figure), above the prior's 13.1. Minimum sample size:
    3 spawns per level. Confidence: medium (0.5).

## Prior predictions for the teacher and the student

- **[opinion]** These are the prior's decode rates (per-request / aggregate tok/s) from
  `python lab/public_prior.py`. Teacher is 15.5 GB per token (`w4a16_decode_gfx1201.md` section 1.2).
  Student BF16 is 8.0 GB and student Q4_K_M is 2.4 GB, both from the nominal 4B parameters.
  - These are forecasts from public data, to be calibrated by our own trials, not measurements.
  - Source: `lab/public_prior.py::table`; byte budget from
    [w4a16_decode_gfx1201.md section 1](w4a16_decode_gfx1201.md#1-baseline-and-byte-budget).

  | Target | Backend | Batch 1 | Batch 4 | Batch 8 | Batch 16 |
  | --- | --- | --- | --- | --- | --- |
  | teacher 27B INT4 | vLLM | 7.5 / 7.5 | 5.7 / 22.9 | 4.3 / 34.7 | 2.9 / 46.9 |
  | teacher 27B INT4 | llama.cpp ROCm | 12.3 / 12.3 | 7.9 / 31.7 | 5.4 / 42.9 | 3.3 / 52.1 |
  | teacher 27B INT4 | llama.cpp Vulkan | 14.3 / 14.3 | 7.4 / 29.7 | 4.5 / 36.3 | 2.6 / 40.8 |
  | student 4B BF16 | vLLM | 13.1 / 13.1 | 10.0 / 39.9 | 7.6 / 60.6 | 5.1 / 81.8 |
  | student 4B BF16 | llama.cpp ROCm | 21.5 / 21.5 | 13.8 / 55.3 | 9.3 / 74.8 | 5.7 / 90.8 |
  | student 4B BF16 | llama.cpp Vulkan | 24.9 / 24.9 | 13.0 / 51.9 | 7.9 / 63.3 | 4.5 / 71.2 |
  | student 4B Q4_K_M | vLLM | 35.7 / 35.7 | 27.2 / 108.8 | 20.7 / 165.3 | 14.0 / 223.3 |
  | student 4B Q4_K_M | llama.cpp ROCm | 58.8 / 58.8 | 37.7 / 150.8 | 25.5 / 204.1 | 15.5 / 247.8 |
  | student 4B Q4_K_M | llama.cpp Vulkan | 67.9 / 67.9 | 35.4 / 141.6 | 21.6 / 172.9 | 12.2 / 194.3 |

  - Implication: these are the warm start for the settings optimizer (R49 names it). Each prediction
    carries the fit's typical factor of 1.38, and the vLLM rows carry more, because their term rests on
    six rows.
  - Candidate experiment: PB1, PB2 and PB4 calibrate the three cells the demo uses. Expected movement: the
    vLLM teacher cell falls to our measured 2.9 at batch 1. Minimum sample size: as in each experiment.
    Confidence: low (0.4) on any vLLM cell before calibration.

## Prediction check against our own numbers

- **[measured]** Our teacher on pod C decodes at 2.88 tok/s, against 7.49 from the prior: 38% of it.
  - Setup: vLLM 0.19.2 for gfx1151, RedHatAI/Qwen3.8-27B-INT4, batch 1, 16k context, 31 tokens in and 77
    out.
  - TTFT 0.39 s; 120 W average and 129 W peak while decoding; idle 9.2 W with the teacher loaded.
  - 2.88 tok/s is also 23% to 25% of the public llama.cpp 27B Q4_K_M rows.

  - Source: [L4] `lab/README.md`, section "Teacher on the Strix Halo (pod A, account C)", titled "Teacher on SJC today (pod C, Strix
    Halo)" when L4 was indexed; the prior from
    `lab/public_prior.py`.
  - Implication: the prior's vLLM term, fitted on other people's vLLM builds and kernels, is optimistic for
    our Conch W4A16 path by 2.6x. The calibration factor for our build is about 0.38 until a kernel or
    setting changes. The optimizer should treat a vLLM arm predicted above 7.5 tok/s as a kernel change,
    not a settings change.
  - Candidate experiment: PB1, the same checkpoint class served three ways at batch 1 on one pod: our vLLM
    build; llama.cpp ROCm and Vulkan with a Q4_K_M GGUF of Qwen3.8-27B; and vLLM with the HIP skinny W4A16
    path if R40 lands. Expected movement: llama.cpp 10 to 14 tok/s (prior 12.3 and 14.3, public 11.2 and
    12.1) and vLLM unchanged at 2.9. Minimum sample size: 5 runs of 256 generated tokens per arm.
    Confidence: medium (0.6).

## Gaps

Coverage gaps, largest first.

- **vLLM on gfx1151 is barely measured in public.**
  - Six single-stream vLLM rows survive the filters, and one is a dense 4-bit 27B.
  - The concurrency sweeps are a 0.6B model [P3] and kyuz0's offline throughput at 64 sequences [P7],
    whose numbers count prompt and output tokens together.
  - No TTFT or TPOT curve for a 4-bit dense model at concurrency was found.
- **Only GitHub could be opened from this session.** Every row comes from github.com, gist.github.com or
  raw.githubusercontent.com. The proxy refused or failed to resolve all of these, listed under "Searched
  but not opened" in `SOURCES.md`:
  - AMD's own pages (amd.com, rocm.blogs.amd.com, gpuopen.com);
  - Phoronix and OpenBenchmarking, ServeTheHome, Notebookcheck and Level1Techs;
  - Reddit, Hacker News and the Framework community forum;
  - LM Studio's and Ollama's sites, and YouTube.

  As a result there are 0 review rows and 5 vendor rows. The vendor rows are AMD's ROCm blog read from its
  GitHub source, the gaia release notes and a lemonade-sdk README.
- **Hardware and OS breadth is narrow.**
  - 1 row is on a Ryzen AI Max 385 or 390 (Radeon 8050S), and no 32 GB machine appears.
  - 508 rows are on 64 GB machines like our pods.
  - 298 rows are on Windows; the rest are Linux, mostly Fedora and CachyOS.
  - TTFT is given on 99 rows, and wall power on about ten.

## Open questions

- The public scoreboards in ggml-org/llama.cpp discussions 10879 and 15021 were read only in part: the
  pages truncated or timed out, so only their AI Max rows are in.
- The original Reddit, Framework-forum and Level1Techs threads behind the curated community rows in [P3]
  are linked in the row notes but were not opened.
- AMD's marketing endnotes (TTFT and tok/s claims for the 395 with their LM Studio settings) were not
  opened.
- Whether the Qwen3.8-27B Q4 rows on Ollama (18.3 tok/s) and lemonade (17.5) used MTP is not stated. If
  they did not, they set a higher llama.cpp-family baseline than 12 tok/s.

## Search log

- 2026-10-10, budget about 3 h (kernels track), used about 25 minutes of wall time across five parallel
  search passes (14 to 23 minutes each), then about an hour to merge, fit and write. The areas were llama.cpp on GitHub;
  AMD, lemonade, LM Studio and Ollama; review sites and machine reviews; Reddit, HN and forums; and
  serving engines and batch data.
- Queries included "strix halo llama.cpp benchmark", "ryzen ai max 395 llm benchmark", "8060S tokens per
  second", "gfx1151 llama-bench", "batched-bench strix halo", "gfx1151 vllm", "strix halo sglang" and
  "AI Max ollama", plus GitHub issue search per repository (vLLM, ROCm, TheRock, Ollama, lemonade,
  llama.cpp).
- The search stopped when new searches returned repositories already ingested and every non-GitHub host
  was refused.
- Merge and fit commands:
  - the per-pass CSVs are merged with duplicates dropped on (repository, model, quantization, backend,
    rates, concurrency, prompt and generation length);
  - `python lab/public_prior.py --fit` prints the fit and `python lab/public_prior.py` the table above;
  - `python -m pytest -q tests/test_lab_public_prior.py` checks the CSV's shape and the fit on it.
