# Sharing one R9700 between a serving replica and a fine-tune (experiment E1)

Research note for the plumbing track (O3 rubric tests T1, T3 and T8 through the controller's stop and start
policy), opened 2026-10-10 by plumbing pass 2 after merge 1 gave experiment E1 to this track. Question: can a vLLM
replica and a LoRA fine-tune run on one 32 GB card at all, and what does each of E1's arms cost the box-to-lab link?
The slowdown each job suffers from the other is left out: no source in this step's bound gives it for gfx1201, so
the experiment measures it.

Labels, keys and metric names as in `hub-api-lifecycle.md`: `measured` = observed on a running system with the
setup; `practitioner` = documented behavior from the people who build or run the system; `anecdote` = one report
without data; `opinion` = argument without evidence. Metrics (O3 rubric [R8]): CPH `pod.credits_per_hour`, TTS
`pod.launch_to_serving_s`, RT `pod.recovery_s`, FCR `link.failed_call_rate`; E1 also reads the learning track's
`trainer.samples_per_s` and the kernels track's `teacher.decode_tok_s`, which this track reads and does not move.

## 1. Memory: the replica's share decides whether coexisting can start

- **[practitioner]** vLLM's `gpu_memory_utilization` is "the fraction of GPU memory to be used for the model executor", 0.9 when unspecified, and "a per-instance limit" that "only applies to the current vLLM instance": another instance on the same GPU does not change it, and two instances on one GPU can take 0.5 each. Source: [S51] vLLM at b1388b1, `vllm/config/cache.py`, `CacheConfig.gpu_memory_utilization`, L41-L48, https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/cache.py#L41-L48
  - Implication: a replica started the way the teacher is started (`--gpu-memory-utilization 0.90`, `lab/pod_a_teacher.sh` L79) keeps nine tenths of the card for itself and does not give any back when a trainer starts beside it, so whether E1's coexist arm can start is decided by the share the replica was given (FCR, TTS).
  - Candidate experiment: E17 on T0, arms B and C. Expected movement: arm B fails when the trainer starts; arm C starts both jobs. Minimum sample size: 3 runs per arm.
- **[measured]** The teacher held 84 percent of the card's memory at `--gpu-memory-utilization 0.90` on pod A, the 20-step smoke LoRA peaked at 8.7 GiB allocated on pod B, and each pod's R9700 has 32 GB. Source: `lab/refine/metrics.jsonl` lines 10 and 14 [S53]; `lab/README.md` L3 and L104 and `lab/pod_a_teacher.sh` L79 on main at 97428e4 [S55]
  - Implication: at 0.90 about 3.2 GB of the 32 GB stay outside vLLM's share, about a third of the trainer's 8.7 GiB peak (about 9.3 GB, before the allocator's cache and the runtime's own context), so coexisting with a replica started like the teacher fails when the trainer starts; a share of about 0.6 or less leaves 12.8 GB, room for the trainer's peak with a margin, provided the replica's weights and the KV cache for its context still fit in its share, which nothing measured says (FCR during the training window; the trainer's and the teacher's throughput are read from the other tracks).
  - Candidate experiment: E17, arms B and C. Expected movement: arm B records an out-of-memory error at the trainer's first step and no samples per second; arm C starts both, each slower than alone by an amount nothing measured predicts. Minimum sample size: 3 runs per arm.

## 2. What preempting costs the link

- **[measured]** The teacher's vLLM engine took about 270 s to initialize on pod A, CUDA-graph capture included, and a proxied request to a pod port with nothing listening answered 500 in 152 ms. Source: `lab/refine/metrics.jsonl` line 9 [S53]; `measurements-2026-10-10.md` M9 [R1]
  - Implication: E1's preempt arm leaves the replica down for the whole fine-tune plus about 270 s of engine start, and every call the box sends it in that window fails fast with 500 unless the box holds its calls, so the replica's failed-call rate over the window is 1.0; the controller has no command that stops or starts a model server (`lab/controller.py` docstring: launch, stop, wait, status, quota, keep-alive and run-script), so a preempt policy runs through run-script until such a command exists, which the rubric's T1 lists as "start the model server" (FCR, TTS).
  - Candidate experiment: E17, arm A. Expected movement: link.failed_call_rate 1.0 for calls to the replica during the window, with the trainer at its solo rate (0.51 samples/s on the smoke run, `metrics.jsonl` line 13). Minimum sample size: 3 runs.

## 3. On a Strix Halo pod the memory question is open (plumbing pass 2, after the retarget)

- **[measured]** The lab's pods are now two SJC Strix Halo servers launched through the hub's web form (`gpu_selection` `strix-halo` since 19:07Z on accounts C and D), which the dispatcher describes as Radeon 8060S, gfx1151, RDNA 3.5, 64 GB of memory shared with the CPU, ROCm 7.13 and torch 2.9.1+rocm7.13, with no vLLM or transformers installed yet; Ilan has moved the lab onto them and wants this morning's R9700 numbers kept as another chip, never as a gfx1151 baseline. Source: `measurements-2026-10-10.md` M24 [S49]; Ilan1 and Ilan2 dispatcher page, lines of Sat 3:46 PM and 4:27 PM ET by that page's clock [S58]
  - Implication: section 1's arithmetic (a 32 GB card, the teacher's 0.90 share, the trainer's 8.7 GiB peak) belongs to the R9700 and predicts nothing for pod B now; how much of the 64 GB shared memory torch reports as the GPU's, and so what a vLLM share of it means, has not been read on these pods, so E1's coexist arms can only be sized after a measurement on pod B (FCR during the window; the trainer's and the teacher's throughput are read from the other tracks).
  - Candidate experiment: E21 on T0, which reads `torch.cuda.mem_get_info()` and the fine-tune's own peak on pod B before it sizes arm C. Expected movement: link.failed_call_rate 1.0 over the window in the preempt arm and near 0 in a coexist arm that starts; whether the unsized coexist arm starts is unknown. Minimum sample size: 3 runs per arm.

## 4. The placement plan and the owner's rule settle E21's question (plumbing pass 3, 2026-10-10)

### Takeaway
Since E21 was queued, pod D's memory has been read, the teacher has been started there from scripts, the placement
plan (R46) has been merged, and Ilan has ruled on R43 that nothing shares a chip with a measured job. On that plan
the teacher serves on pod C and pod D is the exclusive card, so E21's coexist arms test a placement the plan
refuses, and its preempt arm's cost is now a measured restart time.

### Cited findings
- **[measured]** Pod D reports 64.0 GiB visible to torch on a host with 62 GB of RAM and 58 GB available, the same as pod C; the R30 teacher (vLLM 0.19.2+gfx1151, RedHatAI/Qwen3.8-27B-INT4) was ready 523 s after a cold start on pod D and answered `GET /v1/models` through the proxy 294.1 s after a start with a warm compile cache. Source: R30 task page, lines of Sat 5:40 PM and 6:10 PM (Hemdutt's agent) [S61]; `lab/refine/metrics.jsonl:53` [S65]

  Implies: section 3's open memory question has its first answer (torch sees the 64 GiB pool, of which about 58 GB is free to place), and E21's preempt arm leaves the replica down for the fine-tune plus about 294 s warm or 523 s cold, in which every call fails fast with 500 (`proxy-path.md` section 5), so its link.failed_call_rate of 1.0 over that window is known before it runs (FCR, TTS).

  Experiment: none beyond HUB-22 in `hub-api-lifecycle.md` section 10, which times the start once more from a stopped account.

- **[practitioner]** `lab/placement.py` (R46) counts the teacher's vLLM share of the memory torch reports as held from its start (57.6 GiB at the serve script's 0.90 of 64 GiB), budgets a LoRA job at 13.05 GiB, and so says no to the teacher and a LoRA job on one pod at 0.90 (74.65 GiB against 58 GiB) and yes on paper only at a share of 0.63 or less; it keeps benchmarks and other measured jobs on the exclusive card alone, one training job per chip, and the teacher on the pod that serves it. Source: `placement.md` sections "What the teacher's vLLM holds", "Memory margin", "Measurements get a chip to themselves" and "Two training jobs on one chip" [S60] [S61] [S62]; `lab/placement.py`, main at fba248a [S64]

  Implies: E21's arm B (a replica at 0.90 beside a fine-tune on pod B) is predicted to fail at the trainer's start, and its arm C (a sized replica beside a fine-tune) is a placement the plan allows only on paper and the owner's rule below forbids for measured jobs (FCR).

  Experiment: none; see the next finding.

- **[practitioner]** On R43's page Ilan's rule for the placement plan and the GPU queue is recorded: requests are grouped by the card state they need (live teacher up; bare exclusive card; training stack loaded) and ordered so a card changes state as few times as possible, and the interference measurement is dropped from R43 because nothing shares a chip with a measured job. Source: R43 task page, line signed Sat 9:08 PM ET by the Ilan1 agent's clock [S67]

  Implies: the question E1, E17 and E21 asked (preempt the replica on pod B during a fine-tune, or coexist) is decided by the owner's rule and the plan rather than by a run on the pods, since a fine-tune never runs beside a serving replica, the teacher stays on pod C, and the cost that remains is a card's change of state on pod D (a teacher start of about 294 s warm), which the queue's ordering minimizes and HUB-22 measures (FCR, TTS).

  Experiment: none beyond HUB-22 in `hub-api-lifecycle.md` section 10; E21 has nothing left to decide.

## Open questions

- How much of a Strix Halo pod's 64 GB shared memory does torch on ROCm 7.13 report as the GPU's (`torch.cuda.mem_get_info()`), and does vLLM size its share from that number? Nothing was read on the pods in this pass.
- How much of the teacher's 84 percent is weights and how much is KV cache, and what is the smallest share at which the 27B INT4 teacher still starts at its 16,384-token context? Nothing in the metrics file or the R2 notes splits it.
- How much does a fine-tune beside a serving replica slow each of them on gfx1201? No source within this step's bound gives a figure; E17 measures it.
- Partly answered by section 4: torch reports 64.0 GiB on both pods with about 58 GB free to place (R30), and the owner's rule takes coexisting off the plan, so the slowdown questions above no longer decide a policy.

## Search log

- 2026-10-10, plumbing pass 2: part of the pass's research step (about 15 minutes in all; the search log of `hub-api-lifecycle.md` has the whole step). Starting point: E1 on T0 and its seed in `lab/machine/experiments.py` (track plumbing, metric trainer.samples_per_s, no expected movement, confidence or minimum sample). Opened: vLLM `vllm/config/cache.py` at b1388b1 (S51); `lab/refine/metrics.jsonl` lines 9, 10, 13 and 14 (S53); `lab/README.md`, `lab/pod_a_teacher.sh` and `lab/controller.py` on main at 97428e4 (S55). Stopped at the budget this pass gave the topic, with the slowdown question left open for the experiment.
- 2026-10-10, plumbing pass 3: part of the pass's research step (the whole step's search log is in `hub-api-lifecycle.md`). Starting point: E21 on T0 against what landed after it: the R30 and R43 task pages (S61, S67), `placement.md` and `lab/placement.py` (R46, S64), and `lab/refine/metrics.jsonl:53` (S65). Stopped when the owner's rule on R43 answered the policy question E21 was queued to decide.
