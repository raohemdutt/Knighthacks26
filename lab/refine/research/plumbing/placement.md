# Where each job runs on the two Strix Halo pods, and which jobs may share a chip

The rules `lab/placement.py` applies when it writes the placement plan (`python -m lab placement`) and when the GPU
Queue's request verb checks a slot request against it: which pod serves the teacher, where the power sampler, the
benchmark runs, the Ax trials and the LoRA jobs go, and which jobs may share one 64 GB pod and under what memory
margin. Opened 2026-10-10 by R46, which builds the plan offline from recorded facts. What a job running beside
another costs each of them (the interference) is left out: nothing measured gives it on gfx1151, and R43 measures
it on pod D and replaces the two judgement settings below (the margin and the host reserve) with measured numbers.

Labels as in `hub-api-lifecycle.md`. The metrics named are the plan's own (free memory per pod, the teacher's share,
the LoRA budget) and, where a rule protects a measurement, the kernels track's `teacher.decode_tok_s`, the learning
track's `trainer.samples_per_s` and this track's `link.failed_call_rate` (O3 rubric [R8]).

## What the teacher's vLLM holds

- **[practitioner]** vLLM's `gpu_memory_utilization` is the fraction of GPU memory the instance uses, 0.9 when not
  given, and a per-instance limit; at start-up the worker asks for the device's total memory times that fraction
  and refuses to start, naming the fraction, when the free memory on the device is less. Source: [S51] vLLM at
  b1388b1, `vllm/config/cache.py`, `CacheConfig.gpu_memory_utilization`; [S60] vLLM at b1388b1,
  `vllm/v1/worker/utils.py`, `request_memory`, L403-L423
  - Implication: the teacher holds its share of the memory torch reports, not its weights, from the moment it
    starts and whatever its traffic: at the serve script's default 0.90 (`lab/pod_a_teacher.sh:25`) on a pod whose
    torch reports 64.0 GiB that is 57.6 GiB. The plan counts that as held on the teacher's pod, and orders a shared
    pod's starts: the teacher first, because it checks free memory only then and a trainer's allocations grow later.
  - Candidate experiment: PL1 (R43's interference run): on pod D in one slot, start the teacher at a share of 0.60,
    read `torch.cuda.mem_get_info()` and MemAvailable, then run the 20-step LoRA beside a teacher-style decode load,
    and compare each with its solo run. Expected movement: both start with at least 4 GiB still free;
    `teacher.decode_tok_s` and `trainer.samples_per_s` each fall below their solo values by an amount nothing
    measured predicts. Minimum sample size: 3 runs per arm.
- **[measured]** Pod C's first teacher start on gfx1151 loaded 17.98 GiB of weights and then died with "HIP out of
  memory. Tried to allocate 256.00 GiB. GPU 0 has a total capacity of 64.00 GiB of which 42.49 GiB is free", while
  the pod's hour-0 facts gave 62 GB of RAM with 58 GB available and 64.0 GiB visible to torch; pod D's facts are the
  same (64.0 GiB visible to torch, 62 GB RAM, 58 GB available). Source: [S57] R29 task page, lines of Sat 3:23 PM and
  3:38 PM ET; [S61] R30 task page, line of Sat 5:40 PM ET
  - Implication: on the APU torch reports a 64 GiB device while the host that backs it has 62 GB, so the two numbers
    describe one pool from two sides; the plan takes the smaller of a pod's MemAvailable and its chip sheet's free
    memory as what it can place, 58 GiB on each pod today, and never adds them up.
  - Candidate experiment: PL1's first step reads `torch.cuda.mem_get_info()` beside MemAvailable with nothing
    running and again with the teacher up. Expected movement: the plan's free memory per pod changes from the
    hour-0 MemAvailable (58 GiB) to the smaller of the two live reads. Minimum sample size: 3 reads per state.

## Memory margin

- **[measured]** The 20-step smoke LoRA of Qwen/Qwen3.5-4B (bf16, batch 4, max_length 256, rank 16, 48 text examples)
  peaked at 8.7 GiB by `torch.cuda.max_memory_allocated` on one R9700. Source: [S53] `lab/refine/metrics.jsonl`
  line 14
  - Implication: it is the only measured LoRA peak, from the other chip and from settings smaller than L3's real
    runs (images, longer max_length), and max_memory_allocated leaves out what the caching allocator reserves; the
    plan therefore budgets a LoRA job at this peak plus a margin, 13.05 GiB at the 50 percent the plan uses.
  - Candidate experiment: PL2: run L3's trainer for 20 steps on the L11 fixture on pod D and record
    max_memory_allocated, max_memory_reserved and the process's resident host memory. Expected movement: the LoRA
    budget moves from 13.05 GiB to the measured reserved peak plus host memory, and the largest teacher share that
    still lets a LoRA job share its pod moves with it (0.63 today). Minimum sample size: 3 runs.
- **[opinion]** The plan's two judgement settings: a 50 percent margin on a measured peak, and a 4 GiB host reserve
  for the jobs' own processes (the vLLM API server, the trainer's Python and data loader), whose host memory on the
  APU comes from the same pool as the GPU's. With them the teacher plus a LoRA job need 74.65 GiB at today's 0.90
  share, more than a pod's 58 GiB, and fit at a share of 0.63 or less. Source: `lab/placement.py::Rules`
  - Implication: the plan says no to the teacher and a LoRA job on one pod today and yes on paper at a share of at
    most 0.63; both settings are command-line options (`--margin`, `--host-reserve`) so a measured value replaces
    them without a code change.
  - Candidate experiment: PL1 and PL2 replace both settings. Expected movement: the margin falls toward the gap
    between reserved and allocated memory on gfx1151 and the host reserve toward the measured resident memory.

## Measurements get a chip to themselves

- **[practitioner]** The GPU Queue's rule: anything that trains, benchmarks, restarts a process or needs the whole
  GPU asks for pod B's card and waits its turn, at most one Running row per card, and pod A is the shared teacher
  service, queued only to restart or change the server itself. Source: `lab/machine/slots.py`, module docstring
  and `SlotQueue.request`
  - Implication: benchmark runs and Ax trials (each trial a benchmark or training run, `lab/optimize_settings.py`)
    go to the exclusive card with only the sampler beside them, never beside the live teacher or a LoRA job; a
    request for one on pod A's card contradicts the plan and `python -m lab slot request` refuses it unless
    `--override` says why.
  - Candidate experiment: PL1's bench-beside-LoRA arm, which measures how far a co-running job moves a benchmark.
    Expected movement: `teacher.decode_tok_s` measured beside a LoRA job falls below the solo value, which is why
    measurements stay alone.
- **[measured]** On pod C the teacher decoded 2.88 tok/s at 120.3 W average over one text request, idled at 10.2 W
  with the model loaded over 60 s, and averaged 106.8 W inside the request windows of 50 requests sent one every
  12 s. Source: `lab/refine/metrics.jsonl:24`, `lab/refine/metrics.jsonl:29`, `lab/refine/metrics.jsonl:35` and
  `lab/refine/metrics.jsonl:36` [S62]
  - Implication: a benchmark on the teacher's pod would read the teacher's request bursts as its own watts; and
    between requests the teacher's chip idles at about 10 W while 0.90 of its memory stays reserved, which is the
    capacity the plan's lever names (serve the teacher at a share of at most 0.63 and pod A takes LoRA jobs while
    pod B runs benchmarks).
  - Candidate experiment: PL1. Expected movement: as above; the lever is worth taking only if the teacher's decode
    rate beside a LoRA job stays within the spread of its solo runs.

## Where the teacher serves

- **[measured]** The teacher's engine was ready 297 s after a start with a warm compile cache on pod C, and 523 s
  after a cold start on pod D, which holds the same vLLM build and checkpoint since R30. Source:
  `lab/refine/metrics.jsonl:34`; `lab/README.md`, section "The same build on pod D, from scripts alone (R30)"; [S61] R30 task page, lines of Sat 6:10 PM and 6:40 PM ET
  - Implication: moving the teacher costs 5 to 9 minutes with no teacher for the box (every call fails fast while
    nothing listens, `contention.md` section 2), so the plan keeps it on the pod that serves it and names the other
    pod as the fallback when that one goes down.
  - Candidate experiment: start the teacher on pod D once more inside a slot, now with a warm cache. Expected
    movement: the fallback's start time falls from 523 s toward pod C's 297 s. Minimum sample size: 2 starts.

## The sampler shares any chip

- **[practitioner]** `lab/pod_power.py` is one stdlib file that reads `amd-smi` once a period (default 1000 ms) and
  touches no GPU memory; on pod C one read takes 0.21 s. Source: `lab/pod_power.py`, module docstring;
  `lab/README.md`, section "Teacher on SJC today (pod C, Strix Halo)"
  - Implication: the sampler runs beside anything; every measured job needs one on its pod, so the plan places it
    on both pods and names the pod that has none recorded (pod D today).
  - Candidate experiment: PL3: on pod D, 5 benchmark runs with the sampler running and 5 without. Expected
    movement: decode tok/s differs by no more than the run-to-run spread. Minimum sample size: 5 runs per arm.

## Two training jobs on one chip

- **[opinion]** Two LoRA jobs fit one pod's memory on paper (2 x 13.05 GiB plus the reserve is 30.1 GiB) but both
  are compute-bound, so running them together finishes no sooner than one after the other and makes each run's
  samples per second meaningless; the plan keeps one training job per chip. Source: `lab/placement.py::build_plan`
  - Implication: `trainer.samples_per_s` stays a per-run number measured as `metrics.jsonl:13` defines it; a second
    LoRA job waits in the queue.
  - Candidate experiment: PL4: two 20-step LoRA runs on pod D at once, against the same two one after the other.
    Expected movement: the total wall-clock is equal or longer together, and each run's samples per second about
    halves. Minimum sample size: 3 pairs.

## Open questions

- How much of the 64 GiB torch reports does `torch.cuda.mem_get_info()` call free on these pods with nothing
  running, and does it fall with MemAvailable when a process allocates host memory? Nothing has read it yet.
- The smallest share at which the 27B INT4 teacher still starts at its 16,384-token context on gfx1151, and how
  many concurrent sequences it then holds; the plan's 0.63 is a memory bound, not a tested start.
- What a LoRA job beside the teacher costs each of them (R43).

## Search log

- 2026-10-10, R46: budget 2 h, used about 30 minutes. Opened: vLLM `vllm/config/cache.py` and
  `vllm/v1/worker/utils.py` at b1388b1 (S51, S60); the R29 and R30 task pages (S57, S61); `lab/refine/metrics.jsonl`,
  `lab/README.md`, `lab/pod_a_teacher.sh`, `lab/machine/slots.py`, `lab/pod_power.py` and `contention.md` on main
  at 7208c59. Stopped because the open questions need measurements on a pod, not more reading.
