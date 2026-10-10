# Refinement goal for the kernels and speed track

Created 2026-10-10. This file governs every refinement pass on the kernels and speed track: the
teacher's serving path on the AMD Radeon AI PRO R9700 (gfx1201), the Triton and HIP kernels inside
it, and the experiments that measure them. A pass that does not follow it is invalid and its edits
are reverted. Citations in square brackets point at the Sources section at the end; every source
there was opened by the author of this file on 2026-10-10.

## The goal

Refine the kernels and speed track until every board task and queued experiment in the walk list
below passes every dimension of the rubric, and the track's numbers move. The objective is the best
tokens per second and joules per token on gfx1201, told honestly [S4].

Target: a run on this track ends at the first of three things: a pass that converges, the tenth pass
of the run, or Ilan ending it [S9]. A pass converges when it has walked every board task and every
queued experiment of the track against every dimension of this rubric and changed nothing: no board
edit, no new queued experiment, and no new ruling [S9].

What this track refines: the vLLM build that serves the teacher (built from source on the pod with
the patches kept in the repo [S17]), the kernels inside it, and the board tasks and experiments that
plan and measure them. The pass rules give the kernels track "serving speed and kernel work: R11,
R12 and the ranked experiment queue on T0" [S9]; the walk list below is that set, with the subjects
those tasks act on. Telemetry samplers, the learning loop, and pod plumbing have their own rubrics
under `lab/refine/`, and the merge pass settles overlaps [S9].

## How passes are run

`lab/refine/PASS_RULES.md` is the process. Where it and this file disagree about process, it wins;
where they disagree about what the track is judged on, this file wins [S9]. In short: each pass is a
fresh Opus agent started through the open-ticket skill as its own Task Board task, never a fork;
passes on this track run strictly in sequence, the four tracks run in parallel in rounds, and a Fable
merge pass follows each round [S9]. A pass reads the pass rules, this file, the formats in
`lab/refine/TEMPLATES.md`, the counter and the log, the rulings, the metrics file
`lab/refine/metrics.jsonl` (later the ledger) [S11], the notes under `lab/refine/research/kernels/`
[S12], the board tasks of this track, and the experiment queue on T0 [S9][S10].

A pass edits plans only. It runs no GPU job, takes no GPU Queue slot, starts no pod or model server,
and writes no `data/events` [S9]. A measurement it needs becomes an experiment line on T0, which an
ordinary task later runs through the GPU Queue on pod B [S9][S16].

## Primary sources (the ONLY things that may justify a change)

In the order the pass rules give [S9], which every track's rubric uses [S14]. When sources disagree,
follow the higher one and record the disagreement; never average.

1. Ilan's rulings: a filled `Answer:` in `lab/refine/kernels/OWNER_RULINGS.md` [S10], and his
   decisions recorded in the Design Doc and on the Task Board: vLLM built from source is the serving
   path and the kernel work lives in the repo [S17]; pod A is a shared teacher service and pod B is
   reserved through the GPU Queue [S16].
2. The Design Doc and its context specs: sections 8 and 10 for the lab [S17], and Context: Telemetry
   for how energy is counted [S15].
3. Research notes: `lab/refine/research/kernels/` and its `SOURCES.md` [S12], and the outside
   sources they cite, each opened by the pass that cites it.
4. Our own measurements: `lab/refine/metrics.jsonl` until the ledger replaces it [S11], the measured
   table in `lab/README.md` [S1], and results written on T0 [S6].

This ranking decides what a pass may change. Whether a change worked on our card (a speed, a
wattage, whether a kernel fits in LDS) is decided by U1, the before and after measured here, never by
an outside report of what the same change did elsewhere: outside sources decide what to try, our
measurements decide what happened [S9].

A pass MUST open a source before changing anything that rests on it, and MUST cite the file and
heading, the URL, or the metrics-file line in the log. No citation, no change. Nothing from memory
[S4].

## The rubric

Ten dimensions: the two universal rules that every track's rubric carries [S4][S14], and the eight
kernel tests [S4]. Every pass checks all ten against every item it walks. A finding under any
dimension is logged whether or not it is fixed this pass.

### U1. Old versus new

Every task and experiment names the metrics it is meant to move, reads their current values from the
metrics file or ledger before starting, and ends by measuring them the same way. The pass entry shows
the values before and after and the conditions they were measured under. A change that moves nothing
is not an improvement. A regression is reverted, or it goes to a ruling with the trade-off stated.
The loop itself is judged by whether the track's numbers moved across passes [S9][S4].

Check: every value in the metrics-before block quotes its metrics-file line [S11]; the after values
were measured with the same harness, prompt set, checkpoint, quantization, batch sizes, node, and
power sampler as the before values; a missing before or a changed condition makes the comparison
invalid, never "roughly comparable".

### U2. Source-forced

Every change to a method or a plan item cites at least one source next to its before and after. Each
pass starts with a bounded research step on the track's open problems, written into
`lab/refine/research/kernels/` with citations. A source-backed idea that has not been tested becomes a
queued experiment with its citation and expected movement. A change that depends on a fact no source
in hand establishes is not made; the fact becomes a queued measurement or a ruling [S9][S4].

Check: each numbered change in the log has a SOURCE line naming a file and heading, a URL, or a
metrics-file line; research notes follow the research-note format (findings labeled measured,
practitioner, anecdote, or opinion, within the search bound in `lab/refine/RESEARCH_FORMAT.md`)
[S12]; each new T0 line carries its citation and expected movement (see K8). A pass entry with no
metrics-before block, or with any change that has no citation, is invalid and is reverted [S9].

### K1. Stock numbers first

Before any change is judged, the stock build is measured on a fixed prompt set, a fixed model and
quantization, and fixed decode settings, at batch sizes 1, 4, 8, and 16 [S4]. Stock means vLLM at
commit b1388b1 plus `lab/vllm-gfx1201.diff` as merged, serving `RedHatAI/Qwen3.8-27B-INT4` on ROCm
7.13 [S1][S2].

Check: (a) stock rows exist at all four batch sizes, measured with the R11 harness [S7]; (b) each
row's conditions name the engine commit and patch hash, checkpoint, quantization, maximum model
length, batch size, node, ROCm, torch and Triton versions, and the linear and attention kernels vLLM
reports selecting at startup, because one kernel-selection change alone cost about 10% of decode
on an R9700 with nothing failing [S35]; (c) a change to the engine commit, Triton, ROCm, the
checkpoint, or the prompt set starts a new stock record before anything is compared against it; (d)
the prompt set uses text and fixture images only, never camera frames, because raw frames never leave
the Jetson [S17].

### K2. One lever per experiment

An experiment changes one thing. A change that moves two things (a kernel and its launch config, a
kernel and the checkpoint, a kernel and the batch size) is split into two experiments, each measured
against its own before [S4].

Check: the experiment's diff against stock touches one lever. The build-enabling hunks of
`lab/vllm-gfx1201.diff` (platform detection, torch pins, the `compat.cuh` guards) are part of stock,
not levers [S2]. Batch size, KV-cache dtype, speculative decoding, the checkpoint, and each kernel's
tile and stage settings are separate levers [S6][S31]. The precedent: issue 57838 isolated a 10%
decode loss by removing one entry from vLLM's kernel list and changing nothing else [S35], and the
fix in PR 49264 was judged by swapping only the attention kernel [S34].

### K3. Repeats, spread, and a win larger than the spread

Every number is the summary of repeated runs and is reported with its spread; a win counts only when
the difference exceeds the run-to-run spread of both arms [S4].

Check: (a) each arm has at least five runs, the floor under which `verdict()` in
`lab/machine/verdict.py` labels a measurement too_early, and more when its `needed_per_arm` says the
observed gap needs them [S13]; (b) each metric is reported as median with min and max beside the mean
and standard deviation that `verdict()` reads, and normality is not assumed without checking [S42];
(c) two arms are compared with `verdict(old_runs, new_runs)`, Welch's t test, with
`higher_is_better=False` for joules per token, never by eye [S13][S42]; (d) warm-up runs are
discarded and said to be, and if prefix caching is on, repeats carry a per-run nonce so time to
first token does not measure a cache hit, as the RDNA4 port's harness does [S32]; (e) a single run
goes into the metrics file marked `low_confidence` [S11]. The PR 49264 authors called a +0.1% change
inside a 3.6% run-to-run spread no change [S34]; issue 57838 reported within-arm spread of 0.02 to
0.04 tok/s beside its 10% gap [S35]. The first recorded numbers are single runs with no spread [S1],
so they start the record; they are not yet a baseline a win can be judged against.

### K4. Outputs match the baseline

Outputs of a changed kernel match stock's on a fixed greedy prompt set, or the change is a bug until
proven otherwise [S4].

Check: (a) the greedy set (temperature 0, fixed `max_tokens`, thinking on or off stated) runs with
`--enforce-eager` on both builds [S25], because compiled graphs pick kernels by timing at first start
and an RDNA4 port measured about 80 to 90% of 300-token greedy completions diverging across compile
caches while eager outputs stayed bit-identical [S32]; (b) vLLM reproduces outputs only on the same
hardware and version [S28], and its batch-invariant mode is documented for NVIDIA and Intel only
[S29], so matching is judged one prompt at a time at batch 1 on the same node class and versions; (c)
stock is first run against itself, and its self-match rate is the floor a change is held to; (d) the
output match rate is the share of prompts whose token stream equals stock's token for token, and
each mismatch records the first divergent position and the logprob difference there, the acceptance
PR 49264 used for an attention-kernel swap [S34]; (e) a change below the floor records no speed
result until the mismatch is explained. PyTorch's GEMM tuner applies the same discipline: a solution
that misses the default's accuracy fails its profile [S30].

### K5. The 64 KB LDS limit, checked per kernel

On gfx1201 one work-group may allocate at most 64 KB (65,536 bytes) of LDS, in 1 KB blocks, out of
128 KB per work-group processor [S18][S19]. The ROCm specification table lists 128 KiB of LDS for the
R9700 without saying per what [S20]; the per-work-group limit is the one a kernel hits. Every Triton
or HIP kernel a change touches has its LDS use checked against that limit, and a resource failure
counts as a failure of the experiment [S4]. The first example is the `num_stages=1` fix in
`lab/vllm-gfx1201.diff`: the unified attention kernel's default pipelining needed 66,560 bytes and
Triton refused it, so double-buffering was removed [S2].

Check: (a) for a Triton kernel, record the compiled kernel's `metadata.shared`; Triton raises
`OutOfResources` before launch when it exceeds the device maximum [S27], and the same byte count
appears as the module's shared-memory attribute (`ttg.shared`, `triton_gpu.shared` in older Triton)
in the `MLIR_ENABLE_DUMP=1` output, while `AMDGCN_ENABLE_DUMP=1` prints the ISA with `.vgpr_count`
for occupancy [S22][S27]; (b) for a HIP kernel, record the code object's `.group_segment_fixed_size`,
the static group memory one work-group needs, which excludes LDS added dynamically at dispatch [S19];
(c) an autotuned kernel reports every config it tried, because Triton's autotuner turns
`OutOfResources` into an infinite time and moves on without a word unless verbose [S27]; (d) a kernel
near the limit states its occupancy, since LDS per work-group bounds the waves that fit [S22]; (e)
Instinct defaults are not assumed safe here: the ROCm tuning guide recommends `num_stages=2` for
single-GEMM kernels and is written for MI300X and MI350X only [S22], and default pipelining
overflowed on gfx1201 in our patch [S2] and in vLLM's AITER attention path at 65,792 bytes [S34].

### K6. Speed is always paired with joules per token

Every speed result is paired with joules per token from the same runs; the goal is the frontier of
both, not raw speed [S4]. A change that raises tokens per second and also raises joules per token is
a trade-off for a ruling, not a win.

Check: (a) joules per token = the node's watts integrated over the measured window, the same way
Telemetry defines joules per answer [S15], divided by the tokens generated in that window, at each
batch size; (b) the idle-baseline policy is the telemetry track's and is applied identically to old
and new [S14] (pod idle measured 38 to 40 W [S6]); (c) power comes from the pod sampler as reads
only, because the pods allow no power caps or clock locks [S15], so the power-cap and undervolt
levers that community ports report [S33] are out of reach and are not compared against; (d) each
result is placed on the frontier of tokens per second against joules per token per batch size, and a
point worse on both is a regression. Why: batch size significantly affects both generation time and
energy, and the time-energy frontier is convex, so the fastest configuration is not the most
efficient; in one measured case the configuration chosen for energy under a latency target used 44%
less energy than the one that minimizes latency [S41]. The tuners in this stack choose by time
alone: vLLM's MoE tuner keeps a config only when `kernel_time < best_time` [S24], PyTorch TunableOp
keeps the fastest GEMM [S30], and inductor keeps the fastest kernel at first start [S32]. Energy is
the gap this track fills.

### K7. Every change is a patch with a recipe that rebuilds on a fresh pod

Each change is a patch in the repo with a build recipe that reproduces it on a fresh pod from scripts
alone: engine commit, patch order, build flags, and the versions it was built against [S4].

Check: (a) the patch applies cleanly to the pinned commit with nothing rejected; (b) the recipe
rebuilds when the patch changes; (c) the build-enabling hunks and each kernel change are separate
patches (see K2) [S2]; (d) the rebuild time is recorded, since a vLLM build takes about an hour on the
pod [S1]; (e) the pod's own toolchain is the build target (ROCm 7.13.0, torch 2.9.1+rocm7.13.0, Triton
3.5.1+rocm7.13.0 [S1]), and a newer engine that pins another torch is a new stock record (K1). The
existing recipe is `lab/pod_a_teacher.sh build` with the patch at `lab/vllm-gfx1201.diff` [S3][S2];
R12 puts kernel diffs and their result rows under `gpu/` [S8]. Items (a) and (b) fail today; see Known
open items.

### K8. Experiments ordered by expected energy gain times confidence

Kernel and speed experiments are ordered by expected joules-per-token gain times confidence, with the
estimate written before the run and the result written after it, both on the queue on T0 [S4][S6].

Check: (a) each line is tagged `(kernels)` after its number and carries what the pass rules require
(the track, the lever, the arms, the metric to move, the citation, and the expected movement) [S9],
plus the expected change in joules per token at a named batch size, a confidence, and the minimum
runs from `runs_per_arm_relative` on the observed spread [S13], matching the ranking the kernels
research note gives its candidate experiments [S12]; (b)
after the run the measured change is written beside the estimate, slower results included, because a
slower result is still a result [S8]; (c) an estimate above the bandwidth ceiling in The baseline to
beat needs a lever that changes the bytes moved per token (a smaller weight format, speculative
decoding), named as such [S36][S21].

## The walk list

Every pass walks every item below against all ten dimensions [S9]. W1 to W6 are this track's board
tasks and queued experiments, and this list replaces the starting set in the pass rules [S9]; W7 to
W11 are the subjects those tasks and experiments act on, walked with them.

- W1. R11, the benchmark harness: the measuring stick both arms of every comparison depend on (U1,
  K3, K6) [S7].
- W2. R12, the first HIP kernel experiment: the unified attention kernel's stage and block settings
  for the 64 KB LDS (K2, K4, K5, K7) [S8].
- W3. T0 entry E2: vLLM against llama.cpp on the same card and prompts [S6].
- W4. T0 entry E3: the batch sweep and its joules-per-token curve [S6].
- W5. T0 entry E5: the checkpoint shootout (AMD Quark AWQ-INT4, RedHat INT4, official FP8); a new
  checkpoint is a new stock record (K1) [S6][S1].
- W6. Every other line in T0's experiment queue that names this track, as `E<n> (kernels)`, or that
  moves a metric in the metric set below. E1 (a trainer and a serving replica on one card) and E4
  (Jetson power modes) are left to the plumbing and telemetry tracks [S6][S9].
- W7. The stock record: vLLM b1388b1 plus `lab/vllm-gfx1201.diff`, `RedHatAI/Qwen3.8-27B-INT4`, at
  batch sizes 1, 4, 8, and 16, with repeats and greedy reference outputs (K1, K3, K4) [S1][S2].
- W8. The W4A16 linear path: `ConchLinearKernel`, the Triton fallback that `lab/README.md` names the
  first kernel target [S1][S23][S26].
- W9. The pod power sampler, a telemetry-track task, as the source of every joules-per-token number
  here (K6); a finding about it goes to the merge pass [S15][S9].
- W10. The build-enabling patch and its recipe: `lab/vllm-gfx1201.diff` and `lab/pod_a_teacher.sh
  build` (K2, K7) [S2][S3].
- W11. Any engine change a pass proposes (a newer vLLM commit, Triton, or ROCm), which resets the
  stock record (K1) and has its own documented regressions on this card [S35].
- W12. R20 to R27, the run tasks of this track: R20 the stock matrix on pod B, R21 the correctness
  harness, R22 thinking off for captions, R23 the Quark checkpoint, R24 the Conch config sweep, R25
  FP8 weights or KV cache, R26 the batching sweep and R27 the Jetson power modes (shared with Xander,
  so walked for findings only): added by merge 2 under kernels ruling 1 (owner, 2026-10-10: the walk
  list is every kernels task in Ilan's lane, R11, R12, R20 to R27, under the same rules).

## The metric set

Names follow the metrics-file schema (`metric`, `value`, `unit`, `node`, `conditions`, `source`,
`ts`, `low_confidence`) and its dotted metric names [S11]. Every row carries the K1 conditions.

| Metric | Name in `lab/refine/metrics.jsonl` | Unit | How it is measured |
|---|---|---|---|
| Tokens per second at each batch size | `teacher.decode_tok_s`, batch in `conditions` | tok/s | Per-stream decode rate and the aggregate across streams, R11 harness [S7] |
| Time to first token | `teacher.ttft_s` | s | From request sent to first token received, same harness [S7] |
| Joules per token | `teacher.joules_per_token` | J/token | As in K6, from the pod sampler over the same runs [S15] |
| VRAM | `teacher.vram_pct`, plus `teacher.max_concurrency`, the KV-cache capacity vLLM logs at startup | % | The serve step reserves 90% of the card (`--gpu-memory-utilization 0.90` [S3], also vLLM's default at this commit [S25]), so the percentage mostly restates a setting; a kernel change shows up as KV-cache capacity |
| Output match rate against baseline | `teacher.output_match_rate` (new name, first recorded by W1) | fraction | As in K4 |

## The baseline to beat

The first recorded numbers, from the R2 lab setup (PR 37, merged as `bb351e6` on 2026-10-10)
[S1][S5]. Stack: pod A, vLLM b1388b1 (2026-04-18) plus `lab/vllm-gfx1201.diff`, ROCm 7.13.0, torch
2.9.1+rocm7.13.0, Triton 3.5.1+rocm7.13.0 [S1][S2][S23]. Checkpoint: `RedHatAI/Qwen3.8-27B-INT4`,
4-bit integer weights in groups of 128 with bf16 activations (W4A16) [S36]; 16k context and one image
per prompt [S1]. All single runs at batch 1, measured on the pod on 2026-10-10:

| Prompt | Time to first token | Decode |
|---|---|---|
| Text, thinking off, 76 tokens out | 0.55 s | 7.8 tok/s |
| Text, thinking on, 160 tokens out | 0.28 s | 7.7 tok/s |
| 1280x720 JPEG caption, 28 tokens out | 1.09 s | 6.6 tok/s |

- Power: about 200 W average while generating (the O4 task page gives about 202 W) and 300 W peak
  [S1][S4][S5]. 300 W is the card's total board power [S21], so the peak reached the board limit.
- Energy: about 26 J per generated token at batch 1 (200 to 202 W over 7.7 tok/s is 26.0 to 26.2 J),
  whole-board power with nothing subtracted [S4]. With the 38 to 40 W pod idle removed [S6] it would
  be about 21 J; which one is reported is the telemetry track's idle policy [S14].
- Memory and start-up: the model holds 84% of VRAM with about 10x concurrency at 16k tokens; engine
  start takes about 270 s with graph capture [S1].
- Kernel: the W4A16 layers run on `ConchLinearKernel`, the Triton fallback, because at this commit
  ROCm's mixed-precision list is Conch then Exllama, while Marlin (inline PTX) and Machete (CUTLASS,
  Hopper) are CUDA-only [S23][S1]. `lab/README.md` names it the first kernel target [S1]. Read from
  the code, not measured: the Conch release this commit pins (1.2.1) launches one fixed tile for every
  shape and device (block 128 x 64 x 64, 8 warps, 2 stages) [S25][S26], so at decode, where the row
  count is the batch size (1 to 16), most of each 128-row tile is padding.
- What mature 4-bit kernels reach elsewhere: GPTQ and AWQ are the methods behind 4-bit group-wise
  weights like these [S40][S39]; Marlin runs such weights near the ideal 4x over FP16 up to batch 16
  to 32 on NVIDIA [S37], and Machete is its Hopper successor [S38]. Neither Marlin nor Machete exists
  for ROCm at this commit [S23].
- The ceiling (an estimate, not a measurement): plain decode at batch 1 reads every language-model
  weight once per token. The model file is 18.60 GB (the 0.85 GB MTP file is not read in plain
  decode), and its 248,320 x 5,120 bf16 embedding table (2.54 GB) is read one row at a time [S36], so
  at most about 16.1 GB moves per token; at 640 GB/s [S21] that allows roughly 40 tok/s or more before
  memory bandwidth stops it, about five times the measured 7.7 tok/s. Showing such a bound is a
  benchmarking rule [S42]. Speculative decoding with the MTP head changes what one step produces, so
  it is a separate lever this bound does not cover; with it, a port reports 62 to 66 tok/s
  single-stream on two R9700s with FP8 or MXFP4 weights [S31].
- What the record lacks: batch sizes 4, 8, and 16; repeats and spread; reference greedy outputs;
  per-kernel LDS use; the kernel selection lines from the startup log. Until those exist no kernel
  change can be judged against it (K1, K3, K4, K5).

## What a pass may NOT change

- Everything the pass rules list under "What a pass may never change" [S9], including this file: only
  the merge pass edits it, and only to carry out a filled `Answer:` in
  `lab/refine/kernels/OWNER_RULINGS.md` [S9][S10].
- Other people's tasks, the Demo Plan, and any task's Priority: proposals only, posted on the
  Dominick and Team page [S9].
- The teacher serving on pod A: nothing on this track restarts, rebuilds, or patches it. Kernel
  experiments run on pod B through the GPU Queue, as ordinary tasks [S8][S16].
- Starting a pod, a long job, a model publish, or anything that writes `data/events`: Ilan's explicit
  go only. The pods are metered at 4 credits per minute [S1][S9].
- The recorded rows and the first numbers above: appended to, never rewritten; a re-measurement is a
  new row [S11].
- The conditions of a comparison in progress (prompt set, checkpoint, engine commit, batch sizes):
  changing one starts a new stock record (K1).
- Shared files (`pyproject.toml`, `contracts/`, `runtime/`, `bus/`, `contexts/gateway/bootstrap.py`,
  `AGENTS.md`): additive changes only, with a heads-up, never as a side effect of an experiment [S43].
- Camera frames: never in a prompt set; raw frames never leave the Jetson [S17].

## The pass procedure (every pass, in this order)

Follow "What a pass does" in `lab/refine/PASS_RULES.md`, step by step [S9]. On this track the steps
carry the following.

1. Metrics before: for every metric in the metric set, per batch size, the current value with its
   unit, node, the K1 conditions, and the metrics-file line [S11]. A metric with no value yet is
   listed as having no baseline, and a measurement for it is queued on T0 [S9].
2. Research step: the note goes under `lab/refine/research/kernels/`, within the research-format
   bound [S12].
3. Walk: every item in the walk list against all ten dimensions, with every finding listed by
   dimension, fixed or not [S9].
4. Changes, highest impact first for this track: (a) a number bound for the slide or the stats page
   that breaks K3 or K6; (b) an experiment with no stock record, two levers, or no output match; (c) a
   change with no LDS check or no rebuild recipe; (d) an unsourced change or estimate; (e) queue order
   and wording. Each change records BEFORE, AFTER, SOURCE, WHY, and the metric it should move
   [S9][S10].
5. Queued experiments: each new T0 line carries what the pass rules require (the track, the lever,
   the arms, the metric to move, the citation, and the expected movement) and what K8 adds [S9].
6. The entry, in the format of `lab/refine/TEMPLATES.md` [S10], carries this track's blocks: the K1
   conditions, runs and spread per metric (K3), joules per token beside every speed number (K6), LDS
   bytes per changed kernel (K5), and the output match rate (K4). A pass that changed nothing ends
   with `CONVERGED at pass N`, and N goes into `lab/refine/kernels/refine_pass.txt` last [S9].
7. Whoever edits this file (the merge pass, carrying out a ruling [S9]) runs
   `python -m pytest -q tests/test_lab_kernels_rubric.py` from the clone root and pastes the real
   output; it checks this file's sections, rules, citations, baseline rows, and plain hyphens.

## Known open items for pass 1

Found on 2026-10-10 while writing this file.

- K1: only batch-1 numbers exist, and the fixed prompt set does not exist yet because R11 is not
  built [S7]. T0's E3 sweeps batch 1, 4, 16, and 64 while K1 names 1, 4, 8, and 16 [S6]; the stock
  record covers both sets until a ruling picks one.
- K3: the first numbers are single runs, and time to first token differs between the two text
  prompts (0.55 s and 0.28 s) with no stated cause [S1].
- K4: no greedy prompt set or reference outputs exist, stock's self-match rate is unknown, and vLLM's
  batch-invariant mode is not documented for ROCm [S29].
- K5: no per-kernel LDS inventory exists. Only the unified attention kernel is known to need
  `num_stages=1` [S2]; Conch's fixed tile runs, so it fits, but its margin under the limit is
  unmeasured [S26].
- K6: the 26 J figure comes from an average with no stated sampling rate, window, or idle policy
  [S1][S4]; the 300 W peak equals the board limit [S21], and whether the card throttled during the
  run is unknown (the pod power event carries a `throttled` field [S15]).
- K7: `lab/pod_a_teacher.sh` runs under `set -u` without `set -e` and applies the patch with
  `git apply --reject`, so a rejected hunk does not stop the build, and its build step returns early
  whenever vLLM already imports, so a changed patch is not rebuilt on a used pod [S3]. The patch file
  mixes build-enabling hunks with the attention `num_stages` change [S2].
- K8: T0's entries carry no track tag, expected gain, confidence, or minimum runs yet [S6][S9].
- W7: AMD's Quark checkpoints do not load at b1388b1 [S1], so E5 needs a backport of one mapper file
  or a newer vLLM, and a newer vLLM is a new stock record that has regressed decode on this card
  before [S35].

## Sources

Opened by the author of this file on 2026-10-10. Repo files are cited at the commit read.

- [S1] `lab/README.md` at `bb351e6` (PR 37): pod versions (line 9), the build and serve commands (lines 50 to 55), what the patch changes (lines 57 to 63), the measured teacher table, VRAM, power, and `ConchLinearKernel` as the first target (lines 65 to 76), the Quark checkpoints that do not load (lines 80 to 85). https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/README.md?plain=1#L65-L76
- [S2] `lab/vllm-gfx1201.diff` at `bb351e6`: `num_stages=1` in both unified attention launches with the 66,560-byte note (lines 92 to 113), and the platform-detection, torch-pin, and `compat.cuh` hunks. https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/vllm-gfx1201.diff
- [S3] `lab/pod_a_teacher.sh` at `bb351e6`: `set -u` (line 7), the pinned commit (line 11), the early return when vLLM imports (line 53), `git apply --reject` (line 58), the build flags (lines 61 and 62), the serve flags including `--gpu-memory-utilization 0.90` (line 79). https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/pod_a_teacher.sh#L52-L62
- [S4] O4 task page: the two universal rules, the eight kernel tests, the baseline to beat (about 202 W, about 26 J per generated token), the metric set, and the starting source list. https://app.notion.com/p/3f596ef5fec5816aae04d8fb488302c4
- [S5] R2 task page, progress line of Sat 6:25 AM: decode 7.7 tok/s on the Conch Triton fallback, about 200 W average and 300 W peak. https://app.notion.com/p/3f596ef5fec5816bac9ce34a28989784
- [S6] T0 standing ticket: the experiment queue E1 to E5 (E3 sweeps batch 1, 4, 16, 64) and the finding that pod idle power is 38 to 40 W. https://app.notion.com/p/3f596ef5fec581819588ec7a702a4bb1
- [S7] R11 task page: the benchmark harness (fixed prompt set, several batch sizes, amd-smi watts, one JSON row per run with decode tok/s, time to first token, average watts and joules per token, the same harness for vLLM and llama.cpp). https://app.notion.com/p/3f596ef5fec58181bef5d495fcea13e0
- [S8] R12 task page: the first HIP kernel experiment (unified attention stage and block settings for the 64 KB LDS), diff and rows under `gpu/`, never interrupts the teacher on pod A, a slower result is still recorded on T0. https://app.notion.com/p/3f596ef5fec58133b797da3de253aae4
- [S9] `lab/refine/PASS_RULES.md` at `a76af64` (PR 43, the O13 deliverable): the four tracks and the kernels track's starting set (R11, R12, the ranked queue on T0); passes as fresh Opus agents through the open-ticket skill, in sequence per track, tracks in parallel in rounds, a Fable merge pass; what a pass reads, does, may and may never change (no GPU job, no queue slot, no pod); runs of at most 10 passes and convergence; the universal rules; process questions go to the pass rules, judging questions to the rubric. https://github.com/raohemdutt/Knighthacks26/blob/a76af64/lab/refine/PASS_RULES.md
- [S10] O9 task page: `lab/refine/TEMPLATES.md` and the per-track `REFINE_LOG.md`, `OWNER_RULINGS.md`, `refine_pass.txt`, and `pass_reports/`, and the BEFORE, AFTER, SOURCE, WHY entry format. https://app.notion.com/p/3f596ef5fec581c38706ec07a84acf72
- [S11] `lab/refine/metrics.jsonl` at `3cd4b60` (PR 47, the O10 deliverable) and the O10 task page: one row per value with `metric`, `value`, `unit`, `node`, `conditions`, `source`, `ts`, `low_confidence`; seeded with `teacher.decode_tok_s`, `teacher.ttft_s`, `teacher.joules_per_token` (marked low confidence), `teacher.vram_pct`, and `teacher.max_concurrency`. https://github.com/raohemdutt/Knighthacks26/blob/3cd4b60/lab/refine/metrics.jsonl and https://app.notion.com/p/3f596ef5fec5812c9c58dd375d7e8d8d
- [S12] O8 and O36 task pages: kernels research notes under `lab/refine/research/kernels/` with `SOURCES.md`, the four labels, experiments ranked by expected joules-per-token gain times confidence, and the research-format rules. https://app.notion.com/p/3f596ef5fec5817fbd3bd22a0195ca9f and https://app.notion.com/p/3f596ef5fec58136b164e799928c7ec9
- [S13] `lab/machine/verdict.py` at `8a8a7d5` (PR 49, the O28 deliverable): measurements read with Welch's t test through `verdict(a, b)`, `higher_is_better=False` for joules per token, a floor of 5 runs per arm below which the label is too_early, `needed_per_arm` sized on the observed gap, and `runs_per_arm` and `runs_per_arm_relative` for planning. https://github.com/raohemdutt/Knighthacks26/blob/8a8a7d5/lab/machine/verdict.py
- [S14] O1 task page: the shared rubric format and authority order, the universal rules, and the telemetry test that fixes one idle-baseline policy everywhere. https://app.notion.com/p/3f596ef5fec581859648f9824af01daf
- [S15] Context: Telemetry: joules per answer is the answering node's watts integrated over the request window; pods allow power reads only, no caps or clock locks; `telemetry.gpu.power` carries `throttled`. https://app.notion.com/p/3f596ef5fec581ba9b28f19394fa042e
- [S16] GPU Queue (R9700 pods) database: GPU A is the "teacher shared service" and GPU B is "exclusive", 15-minute slots by default (60 for training), one extension with a progress note; and the FYI announcing it on the Dominick and Team page (Sat 7:20 AM): pod A is only called, never reserved, and pod B is reserved shortest job first. https://app.notion.com/p/c69762a396874e049a4af6819f762615 and https://app.notion.com/p/3f596ef5fec581828d8ee02ab7b912e3
- [S17] Atlas Design Doc, sections 8 and 10: vLLM built from source on the pod with the HIP kernel work kept in the repo; raw frames never leave the Jetson. https://app.notion.com/p/3f596ef5fec5819ea5cdcaa015e69917
- [S18] AMD, "RDNA4" Instruction Set Architecture Reference Guide, 7 April 2025: sections 1.2.2.1 and 12.1 (128 kB of LDS per work-group processor; one work-group may allocate up to 64 kB) and 3.3.5 (0 to 64 kbyte per wave, allocated in blocks of 1024 bytes). https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf
- [S19] LLVM AMDGPU backend: `FeatureISAVersion12` sets `AddressableLocalMemorySize65536` and `LDSAllocGranularity1024` (AMDGPU.td lines 2414 to 2422), gfx1200 and gfx1201 use it (GCNProcessors.td lines 297 to 303), and `.group_segment_fixed_size` is the group memory one work-group needs in bytes (AMDGPUUsage.rst line 5465). https://github.com/llvm/llvm-project/blob/3378a49e6e49/llvm/lib/Target/AMDGPU/AMDGPU.td#L2414-L2422 and https://github.com/llvm/llvm-project/blob/35a6d33ca2f3/llvm/lib/Target/AMDGPU/GCNProcessors.td#L297-L303 and https://github.com/llvm/llvm-project/blob/3378a49e6e49/llvm/docs/AMDGPUUsage.rst?plain=1#L5465
- [S20] ROCm documentation, AMD GPU specifications: Radeon AI PRO R9700 is gfx1201 with 64 compute units, wavefront size 32 or 64, and LDS 128 KiB (the unit is not stated). https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
- [S21] AMD Radeon AI PRO R9700 product page: 32 GB GDDR6, 640 GB/s, 300 W total board power, 191 TFLOPs FP16 matrix, 766 TOPs INT4 matrix. https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html
- [S22] ROCm documentation, AMD Instinct MI300X and MI350X workload optimization guide, Triton kernel performance optimization and occupancy: `num_stages=2` for single-GEMM kernels; `MLIR_ENABLE_DUMP=1` shows `triton_gpu.shared`, the LDS bytes allocated for the kernel; `AMDGCN_ENABLE_DUMP=1` prints the ISA, read for `.vgpr_count`; occupancy bounded by LDS per work-group; written for Instinct GPUs only. https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- [S23] vLLM at b1388b1 (2026-04-18), kernel selection: ROCm's mixed-precision kernels are `ConchLinearKernel` then `ExllamaLinearKernel` (lines 132 to 150); Machete needs CUDA and Hopper; Marlin needs CUDA; Conch takes group sizes -1 and 128. https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/kernels/linear/__init__.py#L132-L150 and https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/kernels/linear/mixed_precision/machete.py#L29-L36 and https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/kernels/linear/mixed_precision/marlin.py#L35-L39 and https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/kernels/linear/mixed_precision/conch.py#L14-L20
- [S24] vLLM at b1388b1, `benchmarks/kernels/benchmark_moe.py` lines 610 to 656: `tune()` keeps a config only when `kernel_time < best_time`. https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/benchmarks/kernels/benchmark_moe.py#L610-L656
- [S25] vLLM at b1388b1: `requirements/rocm.txt` pins `conch-triton-kernels==1.2.1` (line 19); `enforce_eager` disables graphs and runs eager (`vllm/config/model.py` lines 204 to 208); `gpu_memory_utilization` defaults to 0.9 (`vllm/config/cache.py` lines 41 to 44). https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/requirements/rocm.txt#L19 and https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/model.py#L204-L208 and https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/cache.py#L41-L44
- [S26] Conch 1.2.1, `conch/kernels/quantization/gemm.py` lines 136 to 145: one fixed set of tuning parameters for every device and shape (block M 128, N 64, K 64, group M 16, 8 warps, 2 stages). https://github.com/stackav-oss/conch/blob/v1.2.1/conch/kernels/quantization/gemm.py#L136-L145
- [S27] Triton 3.5.1: the autotuner returns an infinite time for a config that raises `OutOfResources` (runtime/autotuner.py lines 161 to 166); a compiled kernel whose `metadata.shared` exceeds the device maximum raises `OutOfResources` before launch (compiler/compiler.py lines 461 to 464); the AMD backend fills `metadata.shared` from `ttg.shared` (line 400). https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/runtime/autotuner.py#L161-L166 and https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/compiler/compiler.py#L461-L464 and https://github.com/triton-lang/triton/blob/v3.5.1/third_party/amd/backend/compiler.py#L400
- [S28] vLLM documentation, Reproducibility: results are not reproducible by default, for performance, and only on the same hardware and vLLM version. https://docs.vllm.ai/en/latest/usage/reproducibility.html
- [S29] vLLM documentation, Batch invariance: `VLLM_BATCH_INVARIANT=1`, in beta, for NVIDIA compute capability 8.0 and up and Intel XPUs; ROCm is not listed. https://docs.vllm.ai/en/latest/features/batch_invariance.html
- [S30] PyTorch 2.9 documentation, TunableOp: benchmarks every registered GEMM solution and keeps the fastest by average time, implemented for ROCm only (rocBLAS and hipBLASLt); a solution that misses the default's accuracy fails its profile. https://docs.pytorch.org/docs/2.9/cuda.tunable.html
- [S31] ROCm/vllm issue 1229 (2026-08-20), an RFC offering an RDNA4 serving port: tuned R9700 fp8 GEMM configs at +33% single-stream and about 1.9x at 32 concurrent over vLLM's defaults; Qwen3.8-27B at 62 to 66 tok/s single-stream on two R9700s with MTP-3 speculative decoding and FP8 or MXFP4 weights (practitioner, measured on their setup). https://github.com/ROCm/vllm/issues/1229
- [S32] Capicua25x/vllm-rocm-rdna4, `RDNA4-PORT.md`, section "Reproducibility - same config, different outputs, and why", and its benchmark-harness note: inductor autotuning picks kernels by timing at first start; about 80 to 90% of 300-token greedy completions diverge across compile caches; `--enforce-eager` outputs are bit-identical; the harness uses a per-invocation nonce. https://github.com/Capicua25x/vllm-rocm-rdna4/blob/19f71e90764c/RDNA4-PORT.md?plain=1#L154-L195
- [S33] drwolfen/radiance-vllm-r9700 README: a hand-written W4A8 fp8 WMMA GEMM reported at 325 TFLOP/s against 43 for stock Triton (the author's claim), with benchmarks run under a 235 W power cap and a -70 mV undervolt. https://github.com/drwolfen/radiance-vllm-r9700
- [S34] vLLM issue 48723 and PR 49264: on Wave32 architectures an AITER unified-attention config asks for 65,792 bytes of LDS against 65,536 and Triton aborts start-up; the fix was accepted on all greedy tokens matching, a first-token logprob difference of 0.0090, and a +0.1% throughput change inside a 3.6% run-to-run spread. https://github.com/vllm-project/vllm/issues/48723 and https://github.com/vllm-project/vllm/pull/49264
- [S35] vLLM issue 57838 (2026-09-20): on one R9700, decode fell 5 to 24% below an older baseline from v0.28 on with nothing failing; removing one kernel-list entry, and changing nothing else, recovered 10.0%; within-arm spread 0.02 to 0.04 tok/s. https://github.com/vllm-project/vllm/issues/57838
- [S36] Hugging Face, `RedHatAI/Qwen3.8-27B-INT4`: `model.safetensors` 18.60 GB and `model_mtp.safetensors` 0.85 GB; dtype bfloat16; compressed-tensors int4 symmetric weights in groups of 128 with unquantized activations; `lm_head` and the MTP head left in bf16; 48 linear-attention and 16 full-attention layers; vocabulary 248,320 and hidden size 5,120, embeddings not tied. https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4 and https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4/blob/main/config.json
- [S37] Frantar, Castro, Chen, Hoefler, Alistarh, "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models", arXiv 2408.11743, 2024: close to the ideal 4x speedup for 4-bit weights up to batch 16 to 32, and up to 2.8x end to end in vLLM. https://arxiv.org/abs/2408.11743
- [S38] Lucas Wilkinson, "Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs", Red Hat Developer, 14 October 2024 (no paper): w4a16 on Hopper through wgmma and TMA with weights pre-shuffled offline; the successor to Marlin. https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel
- [S39] Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", MLSys 2024 Best Paper, arXiv 2306.00978. https://arxiv.org/abs/2306.00978
- [S40] Frantar, Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023, arXiv 2210.17323. https://arxiv.org/abs/2210.17323
- [S41] Chung et al., "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization", NeurIPS 2025 Datasets and Benchmarks, arXiv 2505.06371 v2: batch size significantly affects generation time and energy (section 3.2); the time-energy frontier is convex, and one energy-chosen configuration used 44% less energy than the latency-minimizing one (section 4.3); measured on A100 and H100 only. https://arxiv.org/abs/2505.06371
- [S42] Hoefler and Belli, "Scientific Benchmarking of Parallel Computing Systems: Twelve ways to tell the masses when reporting performance results", SC '15: report confidence intervals for nondeterministic data (Rule 5), do not assume normality without checking (Rule 6), compare with non-overlapping intervals or ANOVA (Rule 7), document every varying factor (Rule 9), show upper performance bounds (Rule 11). https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
- [S43] `AGENTS.md` at `bb351e6`, section 8.3: shared files (`contracts/`, `runtime/`, `bus/`, `contexts/gateway/bootstrap.py`, `pyproject.toml`, `AGENTS.md`) take additive changes only, with an FYI to everyone affected. https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/AGENTS.md?plain=1#L232-L249
