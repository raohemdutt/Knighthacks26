# Telemetry track rubric

Created 2026-10-10 from the Task Board page of O1. This file says what the telemetry track is judged
on. The track covers energy and power measurement: the T tasks on the Notion Task Board (T0 to T5,
listed under "The walk list"), the experiments queued on T0 that belong to it, and the numbers that
reach the stats page and the slide [PASS_RULES, The four tracks]. A pass that does not follow it is
invalid and its edits are reverted. It shares its shape and its two universal rules with the
learning rubric [learning-rubric] and the plumbing and kernels rubrics, so its tests are named TT1
to TT7 as the learning tests are LT1 to LT9.

The track's objective: energy numbers trustworthy enough to drive routing and to put on the AMD
slide.

Shared files beside this one: `lab/refine/PASS_RULES.md` (how every pass runs, what it may change,
the cap, the merge pass), `lab/refine/TEMPLATES.md` (the log entry, ruling, counter and report
formats), `lab/refine/RESEARCH_FORMAT.md` (research notes and citations) and
`lab/refine/metrics.jsonl` (the numbers, until the ledger replaces it). This track's own files:
`REFINE_LOG.md`, `OWNER_RULINGS.md`, `refine_pass.txt` and `pass_reports/` in this folder, and
`lab/refine/research/telemetry/`. Where PASS_RULES.md and this file disagree about process,
PASS_RULES.md wins; where they disagree about what the track is judged on, this file wins; a filled
ruling beats both [PASS_RULES].

## The goal

Refine the telemetry track until every test below and both universal rules hold for every cell of
the node and workload matrix and every item in the walk list, and the metric set has moved toward
the state listed below. Target: **10 passes per run**, or convergence, whichever comes first (the
cap in PASS_RULES.md, "The cap"; a ruling can extend a run). Convergence = a pass that walks every
task in the walk list, every queued experiment of the track and every matrix cell against every
test and changes nothing: no board edit, no new queued experiment and no new ruling, though it may
still add research notes [PASS_RULES, Convergence]. The run is judged by the numbers, not by the
pass count (U1): a run that converges with the metric set unmoved has refined the plan but not the
measurements, and the closing section of the run says so.

The state the track is refined toward, one line per test:

- Joules per answer measured end to end for the Jetson student and for the teacher, from samples
  inside each request window, with a stated error and without the low confidence flag (TT1
  and TT2).
- Idle watts measured on the Jetson, pod A and pod B, each with its sampler, period, error and
  stated idle state (TT1 and TT4).
- Every cell of the matrix covered by a sampler that runs and writes rows (TT3).
- One definitions table that every number names (TT4).
- Sampler overhead measured on the Jetson and on a pod, under its bound (TT5).
- A routing decision that reads the numbers, with its effect on joules per answer and on quality
  measured (TT6).
- Every number on the slide and the stats page showing its ledger row and timestamp (TT7).

## How passes are run

PASS_RULES.md governs how passes run; this section summarizes it and adds what is particular to
this track. Each pass is a fresh agent at the Opus tier, started through the open-ticket skill with
the pass as its board task, and it runs the pass itself [PASS_RULES, How passes are run]. Passes on
this track run strictly in sequence, each from a fresh `origin/main` after the previous pass's pull
request has merged, because each pass edits the board rows and files the next one reads; the four
tracks (telemetry, learning, plumbing, kernels) run in parallel. Work happens in rounds, and a
Fable merge pass closes each round. Only the merge pass adds questions to `OWNER_RULINGS.md`, edits
a `RUBRIC.md` (to carry out a filled ruling), and changes anything that crosses a track boundary
[PASS_RULES, The merge pass].

A pass changes the plan, not the machines: the board properties of this track's tasks in Ilan's
lane, experiment lines on T0, and this track's log, report, counter and research notes. A pass runs
nothing on the Jetson or a pod; anything that needs a node becomes a task or a queued experiment and
runs later with Ilan's go, through the GPU Queue when a pod is involved (the open-ticket skill,
sections 4b and 7).

Besides what PASS_RULES.md lists under "What a pass reads", a telemetry pass reads the newest rows
each sampler has written (TT3) and runs the track checks below before it changes anything.

## Universal rules (every track, as the pass rules state them)

Copied from PASS_RULES.md ("The two universal rules") with the track's folder filled in, as every
track rubric carries them [PASS_RULES, Orchestration]; if this copy and PASS_RULES.md ever differ,
PASS_RULES.md wins. Elsewhere in this file, U1 is the first rule and U2 the second.

1. **Old versus new.** Every task and experiment names the metrics it is meant to move, reads their
   current values from the metrics file or ledger before starting, and ends by measuring them the
   same way. The pass entry shows the values before and after and the conditions they were measured
   under. A change that moves nothing is not an improvement. A regression is reverted, or it goes to
   a ruling with the trade-off stated. The loop itself is judged by whether the track's numbers
   moved across passes.
2. **Source-forced.** Every change to a method or a plan item cites at least one source next to its
   before and after. Each pass starts with a bounded research step on the track's open problems,
   written into `lab/refine/research/telemetry/` with citations. A source-backed idea that has not
   been tested becomes a queued experiment with its citation and expected movement. A change that
   depends on a fact no source in hand establishes is not made; the fact becomes a queued
   measurement or a ruling.

A pass entry with no metrics-before block, or with any change that has no citation, is invalid and
is reverted [PASS_RULES, Invalid passes].

In this track, "measured the same way" (U1) covers the sampler, its period, the window rule, the
idle policy, the node's power mode, and the GPU temperature at the start of each run, because
average energy correlates with the GPU temperature at the start of measurement [MLE-thermal]; a
before and an after that differ in any of these are not a comparison. Repeated runs follow the
published practice for built-in sensors: enough repetitions or runtime to cover the sensor's
update pattern, several trials with a pause between them, and the rise time discarded [Yang-2023,
section 5.1].

## Primary sources (the only things that may justify a change), in authority order

When sources disagree, follow the higher one and record the disagreement in the log; never
average. A number measured on our own node that disagrees with a general claim in a source is
recorded with both numbers and raised as a ruling question, because a vendor's statement about a
card family does not settle what our card did.

1. **Ilan's rulings.** A filled-in `Answer:` in `lab/refine/telemetry/OWNER_RULINGS.md` (a blank
   one is not a ruling; the `Now:` default stands until it is filled). Ilan's decisions on Notion:
   the answers on his Team Comms page [Ilan-page] (for this track, Q1: `amd-smi metric --power`
   and `rocm-smi --showpower` both read power on the pods, which have no root; Q4: telemetry stays
   open-ended, with no fixed metric list in the contract), the scope notes "from Ilan's split" on
   T0 to T5 and R11 [Tasks], and the GPU Queue rules. The binding process: `AGENTS.md` and the
   open-ticket skill (locks, the node gate, no pod or GPU work without Ilan's go). A pass may make
   these more precise, never weaker.
2. **The Design Doc's context spec.** Context: Telemetry [Telemetry-spec]: its language (node,
   sample, source, joules per answer), its events, its design (one sampler per node at the cadence
   its source supports, reads only on the pods, pod samplers read by the Jetson through the hub
   proxy, energy per answer and per day as projections rebuilt from the log, the routing hook) and
   its definition of done. With it: the routing hook section of Context: Inference Runtime
   [Runtime-spec], the hardware topology in Design Doc section 8 [Design-Doc], and the Demo Plan's
   "Under the hood" beat, which says which numbers reach the slide [Demo-Plan]. These are the
   people's plan: a pass cites them and never edits them.
3. **Research notes.** `lab/refine/research/telemetry/` and its `SOURCES.md` (format in
   RESEARCH_FORMAT.md), and the outside sources in the source list at the end of this file. A pass
   cites a note's finding by file and heading, and an outside source by URL and section, page or
   table.
4. **Our ledger.** `lab/refine/metrics.jsonl` until the ledger replaces it, then the ledger; a
   number is cited by its row (metric, node, ts, source). Measurements recorded elsewhere
   (`lab/README.md`, the findings on T0) are cited by file and line, or by page and line, until
   they are copied into the metrics file.

A pass MUST open the source before changing a line and MUST cite it in the log: file and heading,
URL and section, or ledger row. No citation, no change. Nothing from memory.

## The rubric: seven telemetry tests

Every pass checks all seven tests, and the two universal rules, against every task and queued
experiment in the walk list and every cell of the matrix in TT3. A finding under any test is logged
whether or not it is fixed this pass, with the test, the item or cell, and the evidence. Each test
is stated as the task page states it [Orchestration], then made checkable. TT1 to TT7 are the
tests; T1 to T5 on the board are tasks.

### TT1. Source, sampling rate and error on every power number

Every power number states its source, sampling rate, and error; an attributed window with too few
samples is marked low confidence, never reported as fact.

- Source means the tool and the field or rail. On the Jetson: `tegrastats`, rails VDD_IN ("Total
  Module Power"), VDD_CPU_GPU_CV and VDD_SOC from the Orin Nano's three-channel INA3221
  [Orin-power, Software-Based Power Consumption Modeling]; each rail prints its current and
  average power in mW [tegrastats]. On the pods: `amd-smi metric --power` [AMDSMI-cli], or
  `rocm-smi --showpower`, "Show current Average or Socket Graphics Package Power Consumption"
  [ROCmSMI].
- Sampling rate means the configured period and the period actually measured between samples.
  `tegrastats` defaults to 1000 ms [tegrastats]; the pod sampler's period is whatever its loop
  uses, and is stated.
- Error means a stated bound and where it came from. Only a bound measured on our own node against
  the best reference reachable there lets a number drop the low confidence flag: an external meter
  on the Jetson's DC input; on a pod, the energy counter (once validated against power) and a
  square-wave load of known timing, since the pods cannot be metered from outside and their
  absolute sensor accuracy is the vendor's, which every pod number says. Published bounds from
  other units are stated but do not lift the flag: Jetson internal sensors underestimated actual
  power by up to 50 percent in published calibration work, and calibration brought the error within
  3 percent [Shalavi-2023]; built-in GPU sensors can report an average of only part of the runtime
  (the past 25 ms every 100 ms on A100 and H100) [Yang-2023, section 4.4].
- The pod reading is an average. The amdsmi reference documents socket power as
  `average_socket_power` for Navi cards and `current_socket_power` only for MI300 and later
  [AMDSMI-power], and the R9700 is a Navi 48 (RDNA 4) card [R9700]. AMD does not document the
  averaging window. On MI250X a similar average took a few seconds to follow an idle-to-TDP step,
  while the energy counter refreshed every 1 ms [McDaniel-2026, sections II-B and V-A2]. So a pod
  window shorter than the measured averaging lag is low confidence whatever its sample count, and
  energy-counter differences (`amd-smi metric --energy` [AMDSMI-cli], `rocm-smi
  --showenergycounter` [ROCmSMI]) are preferred once they are validated against power, as Zeus does
  at start-up on ROCm 6.3 and later [Zeus, Hardware Support, AMD GPU]. Reading the energy counter
  rather than polling power is also the published best practice [MLE-practices].
- Low confidence. A window with fewer than 10 samples, a pod window shorter than the source's
  measured delay plus rise and fall time (the confidence window of [McDaniel-2026, section
  III-A1]), or a number whose error was not measured on our node, is marked low confidence and keeps
  the flag wherever it appears. Ten is this file's default, not a sourced number: each edge of a
  window can misplace up to one sample period, so at 10 samples up to a fifth of the window's
  duration sits on an edge. A pass may change it with a measured attribution error or a ruling.
- Checked mechanically (see "Track checks every pass runs"): every power or energy row in the
  metrics file or ledger carries `source` and `ts`, and inside `conditions` the keys `sampler`,
  `period_ms` and `error`, plus `samples` for a windowed number; `low_confidence` is true whenever
  the rule above says so.

### TT2. Joules per answer from the request window, under one idle policy

Joules per answer uses only samples inside the request window, with one stated idle-baseline policy
used everywhere (pod idle measured today at 38 to 40 W).

- The window is the request's `started` to `ended` on the answering node, from
  `telemetry.request`; Context: Telemetry defines joules per answer as "watts of the answering node
  integrated over the request's start and end timestamps" [Telemetry-spec]. Samples outside the
  window are excluded, and the edge rule (clip at the edges, or interpolate) is stated once and used
  everywhere.
- One clock. The request timestamps and the power samples come from one clock, or the offset
  between the two clocks is measured and applied. On the pods this is not automatic: their samplers
  are read by the Jetson through the hub proxy [Telemetry-spec, Design], so sample times come from
  the pod and request times from the Jetson.
- The idle policy, stated once (Now: gross). Joules per answer is gross: the node's measured power
  integrated over the window with nothing subtracted, which is the context spec's definition. Idle
  watts per node are reported beside it as their own metric. A number with idle subtracted is
  called marginal, says `marginal` in its metric name, and carries the idle value, its timestamp
  and its source in `conditions`. The choice matters: pod idle is 38 to 40 W [T0] against about
  200 W while the teacher generates [lab-README, lines 73-74], so idle is about a fifth of the
  teacher's gross draw. Which of the two the router compares is open (Known open items, 7).
- Overlap. When requests overlap on one node (vLLM sizes the teacher for about ten concurrent
  16k-token requests [lab-README, lines 73-74]), the energy of the overlap is split among the
  active requests by one stated rule, and per-answer joules over any interval add up to no more
  than the node's measured energy for that interval (the conservation check). Published practice
  measures energy while the batch is saturated at the server's maximum batch size and reports it
  per whole response [MLE-2025, sections 3.2 and 2.3]; a per-request split can follow a published
  decomposition, a one-time prefill cost plus a fixed setup cost plus a marginal cost per output
  token [Vellaisamy-2026].
- Per answer first, per token beside it. Where token counts are known, joules per output token is
  reported next to joules per answer, never instead of it: per-token energy can fall while
  per-request energy rises [Vellaisamy-2026], and responses differ in length, which is why the
  ML.ENERGY Benchmark reports energy per whole response [MLE-2025, section 2.3].

### TT3. A sampler for every node and workload class

Every node (Jetson, pod A, pod B) and every workload class (serving, training, idle) has a sampler,
or the gap is listed as a task.

The matrix, from Design Doc section 8 and the sources in Context: Telemetry [Design-Doc,
Telemetry-spec]. Each cell names its sampler and the task that delivers it. A cell with neither is
a finding, fixed by filing a task (in Ilan's lane, or as a proposal on Dominick ↔ Team when it
falls in someone else's). A cell is covered when its sampler runs and writes rows, not when its
task exists.

| Node | Serving | Training | Idle |
| --- | --- | --- | --- |
| Jetson (Orin Nano Developer Kit Super, JetPack 6.2.1 [R3]) | student answers and captions: `tegrastats` sampler (T2) plus request windows (T4) | does not occur: the Jetson never trains [Design-Doc] | T2, between requests |
| Pod A (teacher) | vLLM teacher: `amd-smi` sampler (T3) plus request windows (T4) | does not occur unless a job is moved there | T3 |
| Pod B (trainer) | a serving replica, if one is deployed (T0's E1 compares stopping it during training with running both): T3 | per-step watts on `telemetry.train.step` (L3) | T3 |

### TT4. One set of definitions and units

The same metric definitions and units hold across engines and nodes so numbers compare.

- One definitions table holds every metric's name, unit, formula, window rule, idle policy and
  power boundary. Its home is the language section of Context: Telemetry; until that page carries
  them all, the metric set below is the reference. Every number names the definition it was
  computed under. A definition change re-measures the before value under the new definition, or
  the before and after are not compared (U1).
- Power boundary. VDD_IN on the Jetson is "Total Module Power" [Orin-power], so the carrier board
  and the USB cameras are outside it. The R9700's reading is the graphics package (socket)
  [ROCmSMI, AMDSMI-power], so the host's CPU, memory and power supply are outside it. A Jetson to
  pod comparison is a module to graphics package comparison, and every number that compares them,
  the slide included, says so.
- Idle is a stated state. On the Jetson: no request in flight, with detection and capture running
  as in normal use, because the box is always on. On a pod: no request in flight, with the loaded
  model named. A reading without its state is not an idle number.
- Engines. vLLM and llama.cpp are compared by one harness with the same prompts, decode settings
  and token counting; R11's Done when already requires that "the same harness runs unchanged
  against both engines" [Tasks]. The kernels track pairs every speed result with joules per token
  [Orchestration]; that number uses this file's definition, so the two tracks' numbers compare.
- Units. Stored metrics use W, J, s, ms, tok/s and percent. Events may keep the source's native unit
  where their contract names it (the Jetson rails are mW in `telemetry.jetson.power`
  [Telemetry-spec]); the conversion happens once, in one place.
- One reading path per node. Every consumer of a node's power (the samplers, the benchmark harness,
  the trainer's per-step watts) reads it the same way, or the difference is a finding. Zeus measures
  both AMD GPUs and NVIDIA Jetson boards [Zeus-news], which makes it a candidate for one method on
  both node types: an untested idea, so a queued experiment (U2).

### TT5. The sampler's own cost, measured and bounded

The sampler's own overhead is measured and under a stated bound.

- The bounds are this file's defaults, not sourced numbers; a pass may tighten them with a
  measurement, and loosening them needs a ruling. On the Jetson, the sampler uses at most 1 percent
  of one CPU core, averaged over 10 minutes at the period in use. On a pod, decode tok/s and time to
  first token with the sampler running stay within the run-to-run spread of the same runs without
  it.
- Measured by running the same workload with the sampler on and off, interleaved, with enough
  repeats to give a spread: the kernels track's rule that a win must exceed the spread, applied in
  reverse, since interference must stay inside it.
- Why it is not free: one power or energy call typically takes less than 10 ms in Zeus [Zeus,
  Measuring Energy]; polling power consumes extra energy purely for monitoring [MLE-practices]; on
  AMD exascale nodes, instrumentation stretched reads to a few milliseconds [McDaniel-2026, section
  VII]. A shorter period buys samples for TT1 and costs overhead here; the period in use is the
  shortest one that stays under the bound, and the log records the trade.
- All samplers on one node count toward its bound together (on pod B during training: T3's
  sampler and L3's per-step watts).

### TT6. A routing decision that uses the numbers, with its effect measured

At least one routing decision consumes the numbers and that decision's effect is itself measured
(closed loop).

- The decision is the routing hook, `route(request, telemetry_snapshot) -> target`, declared by the
  Inference Runtime and implemented in `contexts/telemetry`. Its v1 answers on the Jetson student
  when it can and escalates to the teacher only when needed; later versions weigh joules per answer
  from `telemetry.request` and the latest power samples. It chooses only for work the Learning or
  Circle contexts already send to the lab; anything about the user's own space stays on the box
  [Runtime-spec].
- Consuming the numbers means the decision reads a value from the telemetry snapshot, not a
  constant or a capability rule alone.
- Measuring the effect means running the same set of routed requests under the policy and under a
  fixed baseline (all student, or all teacher), recording joules per answer, the work's quality
  score and latency for both, with enough repeats for a spread. The quality score is the scorer
  the owning track already uses for that work: the learning track's label agreement for labeling,
  the golden-question scorer `contexts/agent/eval.py` for answers. A policy is kept when joules per
  answer drops by more than its spread while quality and latency stay inside theirs (U1); otherwise
  it is reverted or ruled on. Routers between a strong and a weak model are judged the same way, by
  cost at unchanged quality [RouteLLM-2024]; the saving the hook chases is that task-specific
  systems cost orders of magnitude less energy per inference than multi-purpose generative ones
  [Luccioni-2024].

### TT7. A ledger row and timestamp behind every number shown

Every number that reaches the slide or the stats page carries a ledger source and timestamp.

- The slide's numbers come from `lab/slide_numbers.py`, which prints a source and a read-at time
  beside every number, and `[TBD: ...]` instead of a made-up number when a source is missing
  [slide-script]. The stats page's numbers come from the stats endpoint (R5), which reads the
  telemetry events rather than measuring on its own [Tasks]. Both must point at a ledger row (a
  metrics-file row until the ledger exists), show the row's timestamp, and show its low confidence
  flag when it is set.
- A finding: a number on either surface with no row, an old row shown as current, or a flagged row
  shown without its flag. R5 is shared with Xander and the slide script with Dominick, so for those
  a pass writes a proposal on Dominick ↔ Team and never edits their tasks.

## The metric set

Joules per answer, idle watts per node, sampler overhead (CPU percent and GPU interference),
samples per attributed window, attribution error against a reference reading. Names follow the
dotted, lowercase convention of `lab/refine/metrics.jsonl` (a row holds metric, value, unit, node,
conditions, source, ts, low_confidence) [metrics, Orchestration]. `pod.idle_watts` is already in
that file; the others are new, and the first pass that measures one adds it. If the file names a
metric differently, the file wins and the next pass corrects this table; a pass never renames a
metric that already has rows.

| Metric | Name | How it is measured | Value on 2026-10-10 | Better is |
| --- | --- | --- | --- | --- |
| Joules per answer | `telemetry.joules_per_answer` (J), with `target`, `tokens_out`, `samples`, `period_ms`, `sampler` and `error` in `conditions` | gross energy of the answering node over the request window (TT2) | not measured; the only related row is derived, `teacher.joules_per_token` = 25.97 J/token on pod-a (200 W over 7.7 tok/s), flagged low confidence because the watts and the decode rate come from different windows [metrics, line 12; lab-README, lines 69-74] | lower, but only while the routed work's quality stays inside its spread (TT6) |
| Idle watts per node | `pod.idle_watts` for the pods, `jetson.idle_watts` for the Jetson (W) | mean power over a stated window in the stated idle state (TT4) | `pod.idle_watts` = 39.0 W on pod-a, the midpoint of the 38 to 40 W stated on T0, flagged low confidence for having no sample count or sampling rate [metrics, line 17; T0]; no row for pod-b or the Jetson, and the loaded model is not recorded | not a target: tracked, and a move beyond its spread means a condition changed and is explained |
| Sampler overhead | `telemetry.sampler_cpu_pct` (Jetson, percent of one CPU core) and `telemetry.sampler_interference_pct` (pods, percent change in decode tok/s with the sampler on) | as in TT5 | not measured | lower, and under the bound |
| Samples per attributed window | `telemetry.samples_per_window` (count) | samples inside one request window, per answer and as the median per node | not measured | higher, up to the overhead bound; under 10 is low confidence (TT1) |
| Attribution error | `telemetry.attribution_error_pct` (percent) | measured energy minus reference energy, over reference energy, for the same window; the reference is an external meter on the Jetson's DC input, and on a pod the validated energy counter and a square-wave load of known timing [McDaniel-2026, section III-A1] | not measured | closer to zero; its size is the error TT1 states |

## The walk list

Every pass walks the whole matrix (TT3) and every task and queued experiment of the track against
every test [PASS_RULES, What a pass does], and takes at least two of them end to end through the
untidy cases below, rotating so that every task gets an end-to-end walk during the run (the log
says which). Walking a task means checking it against the two universal rules and the seven tests:
the metric it moves and its before value (U1), the source behind its method (U2), and each of TT1
to TT7 that applies.

The tasks this track covers (this list replaces the starting set in PASS_RULES.md, "The four
tracks"):

- T0, the standing question: what else should we measure, and what would it let us optimize.
- T1, the context skeleton and the per-source events (`contracts/telemetry.py`, a fake sampler, a
  projection rebuilt from the log).
- T2, the Jetson power sampler (`tegrastats`).
- T3, the pod power sampler (`amd-smi`), read through the hub proxy.
- T4, joules per answer.
- T5, the routing hook and its first policy.

The experiments on T0 that belong to this track are the lines marked `(telemetry)` and the lines
whose first metric is in this file's metric set [PASS_RULES, The four tracks]. Today there are none:
E1 to E5 name tok/s or joules per token first, which are kernels metrics, so they are the kernels
track's. The first telemetry experiment is the one in Known open items, 3.

Where this track's numbers are used outside it, a pass checks them (TT4 and TT7) and changes
nothing: it writes the finding in its entry for the merge pass, or posts a proposal when the task is
shared with another person [PASS_RULES, What a pass may never change]:

- R11, the benchmark harness, and T0's experiment lines (the kernels track): how they read pod power
  and define joules per token.
- L3's per-step watts on `telemetry.train.step` (the learning track): the training cell of the
  matrix.
- R5's stats endpoint (shared with Xander) and the slide script (shared with Dominick): TT7.

Each pass also checks the metrics-file or ledger rows this track writes (the provenance check
below) and `lab/refine/research/telemetry/` (U2).

Each end-to-end walk tries at least one untidy case, not only the tidy path:

- a request shorter than the sampler period, and one that spans a sampler restart;
- two overlapping requests on the teacher, and a burst of ten;
- a pod relaunched in the middle of a window (a pod starts in about 15 s [Design-Doc]), and a proxy
  timeout that leaves a gap in pod samples;
- a clock offset between the Jetson and a pod;
- a throttled card (`throttled` on `telemetry.gpu.power` [Telemetry-spec]), or a cold card at the
  start of one arm of an A/B;
- the Jetson in a power mode other than the one its rows record;
- a training step and a serving request on pod B in the same window (E1's coexist case).

## What a pass may NOT change

Everything PASS_RULES.md lists under "What a pass may never change" holds here: another person's
task, or one shared with someone or unassigned; the Demo Plan; any task's Priority; a Locked task; a
`RUBRIC.md`, this one included (only the merge pass edits it, to carry out a filled ruling);
`OWNER_RULINGS.md`; PASS_RULES.md and the universal rules; another track's rows, experiments or
folder; an earlier pass's change, without a newer source or metric; and anything outside plans
(code, contracts, config, fixtures, pods, model servers, GPU jobs, queue slots, `data/events`,
deleting a Notion page). For this track that means in particular:

- R5 (shared with Xander) and the slide script (shared with Dominick): proposals only, on
  Dominick ↔ Team.
- R11 and T0's E1 to E5 (the kernels track) and L3 (the learning track): findings for the merge
  pass, never edits.
- The Design Doc, including Context: Telemetry and Context: Inference Runtime: proposals only, on
  Dominick ↔ Team. A change to `contracts/telemetry.py` or `runtime/` is a task: additive, with the
  heads-up `AGENTS.md` requires.
- Metric names that already have rows, and the definition behind a number already on the slide: a
  rename or a redefinition needs a ruling, and a redefinition re-measures the before value (TT4).
- The Jetson's power mode: a mode change can require a reboot of the box [Orin-power, Power Mode
  Controls], which runs MAXN SUPER (nvpmodel mode 2) today [R3]; it runs only as a task with Ilan's
  go.
- The privacy line: the routing hook never sends anything about the user's own space to the lab
  [Runtime-spec], and raw frames never leave the Jetson.
- Another person's or agent's words on Notion, findings on T0 included: never edited or deleted
  (`AGENTS.md`, section 8.1).

## Track checks every pass runs

Run from the clone root with the venv's Python, and paste the real output into the log entry.

1. The provenance check (TT1 and TT7). It prints every metrics-file row with no source or
   timestamp, every power or energy row whose `conditions` lack `sampler`, `period_ms` or `error`,
   and every row with fewer than 10 samples or an `unknown` error that is not flagged
   `low_confidence`. Whether a stated error was measured on our own node is the pass's judgment,
   not the script's.

```
python - <<'PY'
import json
from pathlib import Path

ENERGY_UNITS = {"W", "mW", "J", "J/token", "J/answer"}
lines = Path("lab/refine/metrics.jsonl").read_text(encoding="utf-8").splitlines()
rows = [json.loads(line) for line in lines if line.strip()]
problems = []
for n, row in enumerate(rows, 1):
    name, cond = row.get("metric", "?"), row.get("conditions") or {}
    if not row.get("source") or not row.get("ts"):
        problems.append(f"row {n} {name}: no source or ts")
    if row.get("unit") not in ENERGY_UNITS:
        continue
    missing = [k for k in ("sampler", "period_ms", "error") if k not in cond]
    if missing:
        problems.append(f"row {n} {name}: conditions lack {', '.join(missing)}")
    few = isinstance(cond.get("samples"), int) and cond["samples"] < 10
    if (few or cond.get("error") == "unknown") and not row.get("low_confidence"):
        problems.append(f"row {n} {name}: should be low_confidence")
print(f"{len(rows)} rows, {len(problems)} problems")
print("\n".join(problems))
PY
```

2. The matrix (TT3): for each cell, the sampler, its task and that task's status, and whether
   the sampler has written a row since the last pass.

## Known open items for pass 1 (found 2026-10-10 from the board, the repo and the sources)

1. U1, every task: no Done when on T1 to T5 names a metric from the metric set, its before value or
   a target; the only number on any of them is "~40 W idle" on T3, with no period or error [Tasks].
2. TT1, the numbers we hold: the provenance check run on 2026-10-10 against the metrics file
   reported 22 rows and 4 problems, the four power and energy rows (`teacher.watts_avg`,
   `teacher.watts_peak`, `teacher.joules_per_token`, `pod.idle_watts`, lines 7, 8, 12 and 17), none
   of which records a sampler, period or error [metrics]. Two of them, the teacher's 200 W average
   and 300 W peak, are not flagged low confidence although TT1 says they are until a sampler
   re-measures them; the file is append-only and written by the lab's metrics writer [PASS_RULES],
   so the fix is a new measured row, not an edit. The idle row does not say whether the teacher was
   loaded (TT4). The averages also disagree slightly: `lab/README.md` says about 200 W
   [lab-README, line 74], while the Build Log's R2 entry says about 202 W [Build-Log].
3. TT1 and TT2, the pods: the averaging window and lag of the R9700's socket power are unknown.
   Candidate experiment, to queue on T0 marked `(telemetry)` and run on pod B in a GPU Queue slot:
   a square-wave load with on and off phases of 0.5, 1, 2 and 5 s, recording `amd-smi metric
   --power` and `amd-smi metric --energy` at the shortest period TT5 allows, plus Zeus's
   start-up check of the energy counter against power on ROCm 7.13 [Zeus, Hardware Support, AMD
   GPU]. Expected movement: pod attribution error goes from none to a measured value, and the
   shortest attributable pod window becomes a number [McDaniel-2026, sections III-A1 and V-A2].
4. TT1, the Jetson: `tegrastats` defaults to 1000 ms [tegrastats] and a warm caption takes about
   2.1 s [Build-Log], so at the default a caption holds about two samples, under the 10-sample
   default. T2's Done when says "a configurable interval" with no value; a pass sets the interval
   target from TT1 and holds it to TT5's bound.
5. TT1, the Jetson's error: VDD_IN is total module power [Orin-power], published work found
   Jetson internal sensors underestimating actual power by up to 50 percent [Shalavi-2023], and our
   box has no external meter on its DC input, so its attribution error cannot be measured. A meter
   is hardware and money: a question for Ilan, raised by the merge pass, not a board change.
6. TT2: no task states the window edge rule, the overlap rule, the idle policy, or how pod sample
   times are aligned with Jetson request times; T4's Done when says only that energy is computed "by
   integrating the node's power samples over the request window" [Tasks].
7. TT2, a ruling question: should the router compare gross or marginal joules? Now: gross, the
   context spec's definition. With pod idle near a fifth of the teacher's draw, the answer changes
   which node looks cheaper whenever a pod stays up anyway.
8. TT3: every cell's sampler (T2, T3, and L3's per-step watts) belongs to a task that has not
   started [Tasks], so no cell is covered yet.
9. TT4: pod B's power will be read by T3's sampler, by L3's per-step watts, and by R11, whose
   released lock line planned its own `lab/pod_power_sampler.sh` [Tasks]. Unless one reading path
   feeds all three, their numbers can drift apart. R11 is already blocked by T3; L3 is a
   learning-track task, so naming the shared path there is the merge pass's change.
10. TT5: no task measures sampler overhead on either node.
11. TT6: T5's Done when stops at "a unit test with a fake snapshot proves both branches; the
    choice is logged on telemetry.request" [Tasks]. Its v1 policy routes by capability and reads no
    telemetry value [Runtime-spec], and nothing measures the decision's effect on joules per answer
    or on quality.
12. TT7: R5's Done when lists the stats page's numbers with no source or timestamp, and the slide
    script prints a source and a read-at time, but its sources are endpoint URLs and log paths
    rather than ledger rows, and it has no low confidence column [slide-script]. Both are shared
    tasks, so proposals only.
13. U2, for the merge pass, since E3 is a kernels-track experiment: E3 on T0 says the ML.ENERGY data
    shows batching moves joules per token "6 to 8x" [T0]; the ML.ENERGY Leaderboard v3.0 post says
    increasing batch size "can unlock 3-5x energy per token reduction" [MLE-v3], and a 2026 study
    found the batch 16 over batch 1 token energy gain for 10-token outputs fell from 6.31x at 512
    tokens of context to 1.17x at 4K [Vellaisamy-2026]. E3's expected movement needs its source and
    the context length it assumes; the teacher runs at 16k [lab-README, lines 65 and 74].
14. U1, temperature: average energy correlates with the GPU temperature at the start of measurement
    [MLE-thermal], but `telemetry.gpu.power` has no temperature field [Telemetry-spec]. T1 has not
    started, so the field can join the first version of the event; a pass adds it to T1's Done when,
    and the spec page is Ilan's to change.
15. Live nodes, for the merge pass, since E4 (Jetson power modes) names tok/s first and so is a
    kernels-track experiment: changing the mode can require a reboot [Orin-power, Power Mode
    Controls], and the box runs mode 2 (MAXN SUPER) [R3]. E4's line on T0 needs Ilan's go, a time
    outside the demo window, and a step that restores mode 2.

## Sources (opened 2026-10-10)

Every source below was opened on 2026-10-10 by the author of this file; each line says what it
supports here.

Outside sources:

- [Zeus] Zeus documentation, "Measuring Energy", https://ml.energy/zeus/measure/ . AMD GPUs through
  AMDSMI with ROCm 6.3 or later ("Older versions returned wrong values from the AMDSMI power and
  energy APIs"), the energy counter checked against power at start-up, average power used when
  `current_socket_power` is N/A (Hardware Support, AMD GPU); a power or energy call typically takes
  less than 10 ms (Measuring Energy). TT1 and TT5, open item 3.
- [Zeus-news] Zeus home page, Project News, 2025/05, https://ml.energy/zeus/ . Zeus measures AMD
  GPUs and NVIDIA Jetson boards. TT4.
- [MLE-2025] J.-W. Chung, J. J. Ma, R. Wu, J. Liu, O. J. Kweon, Y. Xia, Z. Wu, M. Chowdhury, "The
  ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization", NeurIPS 2025
  Datasets and Benchmarks track (spotlight), arXiv:2505.06371, https://arxiv.org/abs/2505.06371 .
  Energy per whole response (section 2.3), steady state as a saturated batch (section 3.2), measured
  with Zeus (section 3.1). The leaderboard, https://ml.energy/leaderboard , renders in the browser;
  its method is this paper's. TT2.
- [MLE-v3] J.-W. Chung, "Diagnosing Inference Energy Consumption with the ML.ENERGY Leaderboard
  v3.0", ML.ENERGY Blog, 2026-01-29,
  https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/ .
  46 models and 1,858 configurations on NVIDIA H100 and B200 only; increasing batch size "can
  unlock 3-5x energy per token reduction"; energy per response is energy per token times output
  tokens. Open item 13.
- [MLE-practices] J.-W. Chung, "Measuring GPU Energy: Best Practices", ML.ENERGY Blog, 2023-07-24,
  https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/ . Prefer the energy
  counter to polling power; polling consumes extra energy purely for monitoring. TT1 and TT5.
- [MLE-thermal] R. Wu and J.-W. Chung, "Thermally Stable Profiling for Accurate GPU Energy
  Measurement", ML.ENERGY Blog, 2026-03-09,
  https://ml.energy/blog/energy/measurement/thermally-stable-profiling-for-accurate-gpu-energy-measurement/ .
  Average energy "strongly correlates with the GPU temperature at the start of measurement". U1,
  open item 14.
- [AMDSMI-cli] AMD SMI 27.1.0 documentation, "Using AMD SMI Command Line Interface tool",
  https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/using-AMD-SMI-CLI-tool.html .
  `amd-smi metric -p/--power` ("Current power usage"), `-E/--energy` ("Amount of energy consumed"),
  `-w/--watch INTERVAL`. TT1, open item 3.
- [AMDSMI-power] AMD SMI 7.0.2 reference, "amdsmi_power_info_t Struct Reference",
  https://rocm.docs.amd.com/projects/amdsmi/en/docs-7.0.2/doxygen/docBin/html/structamdsmi__power__info__t.html .
  `average_socket_power` for "Navi + Mi 200 and earlier Series cards", `current_socket_power` for
  "Mi 300+ Series cards". TT1 and TT4.
- [ROCmSMI] ROCm SMI LIB documentation for ROCm 7.2.2, "Using ROCm SMI with Python" (the
  `rocm-smi -h` listing),
  https://rocm.docs.amd.com/projects/rocm_smi_lib/en/docs-7.2.2/python_usage.html .
  `-P, --showpower`: "Show current Average or Socket Graphics Package Power Consumption";
  `--showenergycounter`: "Energy accumulator that stores amount of energy consumed". TT1 and TT4.
- [R9700] Geeks3D, "AMD Radeon AI PRO R9700 Announced (32GB VRAM, 300W)", 2025-05-21,
  https://geeks3d.com/20250521/amd-radeon-ai-pro-r9700-announced-32gb-vram-300w . Navi 48, RDNA 4,
  300 W total power. TT1.
- [tegrastats] NVIDIA Jetson Linux Developer Guide r36.4, "Tegrastats Utility",
  https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html .
  `--interval` in milliseconds, default 1000; each rail prints current and average mW. TT1, open
  item 4.
- [Orin-power] NVIDIA Jetson Linux Developer Guide r36.4, "Jetson Orin Nano Series, Jetson Orin NX
  Series and Jetson AGX Orin Series" (Platform Power and Performance),
  https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html .
  The INA3221 channels on Orin NX and Orin Nano: VDD_IN "Total Module Power", VDD_CPU_GPU_CV,
  VDD_SOC (Software-Based Power Consumption Modeling); `nvpmodel -q` and `-m`, and a mode change may
  require a reboot (Power Mode Controls). TT1 and TT4, open items 5 and 15.
- [Shalavi-2023] N. Shalavi, A. Khoshsirat, M. Stellini, A. Zanella, M. Rossi, "Accurate
  Calibration of Power Measurements from Internal Power Sensors on NVIDIA Jetson Devices", IEEE
  EDGE 2023, arXiv:2306.13107, https://arxiv.org/abs/2306.13107 . Internal sensors "underestimate
  the actual power by up to 50% in most cases"; calibration "reduces the error to within 3%". TT1,
  open item 5.
- [Yang-2023] Z. Yang, K. Adamek, W. Armour, "Part-time Power Measurements: nvidia-smi's Lack of
  Attention", arXiv:2312.02741, v3 2024-12-12 (the page lists DOI 10.1109/SC41406.2024.00028),
  https://arxiv.org/abs/2312.02741 . On A100 and H100 the tool "only reports the average of the
  past 25ms every 100ms" (section 4.4); good practices: 32 iterations or at least 5 s, controlled
  delays, four trials, the rise time discarded (section 5.1). U1, TT1.
- [McDaniel-2026] A. McDaniel, M. Jantz, A. Sharma, S. Abbott, S. Martin, S. Khandekar, B. Neth, B.
  Villasenor Alvarez, A. Kashi, W. Elwasif, O. Hernandez, "Fine-Grained Power and Energy
  Attribution on AMD GPU/APU-Based Exascale Nodes", arXiv:2604.06056, 2026,
  https://arxiv.org/abs/2604.06056 . Square-wave characterization of AMD sensors; energy counters
  refresh every 1 ms (section II-B); MI250X average power takes a few seconds to follow an
  idle-to-TDP step (section V-A2); the confidence window that trims delay, rise and fall (section
  III-A1); energy-counter differences bypass the filtering (section V-B2); reads can stretch to a
  few milliseconds (section VII). TT1 and TT5, the metric set, open item 3.
- [Vellaisamy-2026] P. Vellaisamy, V. Lam, S. Blanton, J. P. Shen, "Characterization of Request
  and Token Energy Costs for LLM Inference Workloads on GPU Platforms", accepted at IISWC 2026 per
  the arXiv comments, arXiv:2608.28044, https://arxiv.org/abs/2608.28044 . Token energy can fall
  while request energy rises; request energy as a prefill cost plus a setup cost plus a marginal
  cost per output token; for 10-token outputs, the batch 16 over batch 1 token energy gain falls
  from 6.31x at 512 context to 1.17x at 4K. TT2, open item 13.
- [Luccioni-2024] A. S. Luccioni, Y. Jernite, E. Strubell, "Power Hungry Processing: Watts Driving
  the Cost of AI Deployment?", ACM FAccT 2024, arXiv:2311.16863, https://arxiv.org/abs/2311.16863 .
  Multi-purpose generative architectures "are orders of magnitude more expensive than task-specific
  systems". TT6.
- [RouteLLM-2024] I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, J. E. Gonzalez, M. W. Kadous,
  I. Stoica, "RouteLLM: Learning to Route LLMs with Preference Data", arXiv:2406.18665, 2024,
  https://arxiv.org/abs/2406.18665 . Routing between a strong and a weak model cut cost by over 2
  times in some cases without lowering response quality. TT6.

Our own sources:

- [PASS_RULES] `lab/refine/PASS_RULES.md` (on `main` since a76af64): the four tracks and which tasks
  and experiments belong to each, how passes are run, what a pass reads and does, convergence, the
  cap, the merge pass, what a pass may and may never change, invalid passes, and the two universal
  rules this file copies.
- [learning-rubric] `lab/refine/learning/RUBRIC.md` (on `main` since 904d92c): the shape this file
  shares (the universal rules as their own section, tests named by track and number, the metric
  table, the walk list, what a pass may not change, the dated open items and sources).
- [Orchestration] The orchestration task pages on the Task Board: O1 (this rubric: the universal
  rules, the seven tests, the metric set, the format), O13 (pass rules: fresh Opus agents started
  through the open-ticket skill, 10 passes per run, the merge pass, an entry without its
  metrics-before block or citations is invalid), O9 (the four track folders, the file formats,
  passes never editing the rulings file), O10 (the metrics file's schema and naming), O5 (the
  research notes folder), O36 (the research-note format), and O2 and O4 (the learning and kernels
  tests this file shares wording with: the run-to-run spread, the `contexts/agent/eval.py` scorer,
  label agreement, speed paired with joules per token),
  https://app.notion.com/p/3f596ef5fec581bcb319e8687c1fdd8e .
- [Telemetry-spec] Context: Telemetry, Atlas Design Doc section 15,
  https://app.notion.com/p/3f596ef5fec581ba9b28f19394fa042e .
- [Runtime-spec] Context: Inference Runtime, section "Routing hook (energy-aware, implemented by
  Telemetry)", https://app.notion.com/p/3f596ef5fec5814097aecc119914b576 .
- [Design-Doc] Atlas Design Doc, section 8 (hardware topology and the GPU server lifecycle),
  https://app.notion.com/p/3f596ef5fec5819ea5cdcaa015e69917 .
- [Demo-Plan] Demo Plan (5 min), the "Under the hood" beat,
  https://app.notion.com/p/3f596ef5fec581a2bdc4f96c8da65252 .
- [Ilan-page] Ilan's Team Comms page (Ilan ↔ Dominick), the answers to Q1 and Q4,
  https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 .
- [Tasks] The Task Board pages for T1 to T5, R11, R5 and L3 (Done when, scope notes, lock lines),
  https://app.notion.com/p/3f596ef5fec581bcb319e8687c1fdd8e .
- [T0] T0, the first line under Findings (pod idle 38 to 40 W from amd-smi and rocm-smi) and the
  experiment queue, https://app.notion.com/p/3f596ef5fec581819588ec7a702a4bb1 .
- [R3] R3, the Jetson environment (Orin Nano Developer Kit Super, JetPack 6.2.1, L4T R36.4.4, MAXN
  SUPER as nvpmodel mode 2), https://app.notion.com/p/3f596ef5fec581c5887bf3490b947df5 .
- [Build-Log] Build Log, the entry "On-device model service, R8 (PR #36)" (student decode about
  15.7 tok/s, a warm caption about 2.1 s) and the R2 entry (the teacher at "~202 W average"),
  https://app.notion.com/p/3f596ef5fec5812fa37de5ae92086234 .
- [metrics] `lab/refine/metrics.jsonl` (on `main` since 3cd4b60), lines 7, 8, 12 and 17: the
  teacher's average and peak watts, the derived joules per token, and pod idle watts; read with
  `lab/machine/metrics.py`.
- [lab-README] `lab/README.md`, lines 65-76: the teacher measured at 16k context, its TTFT and
  decode table, about 200 W average and 300 W peak while generating, about ten concurrent
  16k-token requests.
- [slide-script] `lab/slide_numbers.py`, the module docstring.
