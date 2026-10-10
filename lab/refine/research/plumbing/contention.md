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

## Open questions

- How much of a Strix Halo pod's 64 GB shared memory does torch on ROCm 7.13 report as the GPU's (`torch.cuda.mem_get_info()`), and does vLLM size its share from that number? Nothing was read on the pods in this pass.
- How much of the teacher's 84 percent is weights and how much is KV cache, and what is the smallest share at which the 27B INT4 teacher still starts at its 16,384-token context? Nothing in the metrics file or the R2 notes splits it.
- How much does a fine-tune beside a serving replica slow each of them on gfx1201? No source within this step's bound gives a figure; E17 measures it.

## Search log

- 2026-10-10, plumbing pass 2: part of the pass's research step (about 15 minutes in all; the search log of `hub-api-lifecycle.md` has the whole step). Starting point: E1 on T0 and its seed in `lab/machine/experiments.py` (track plumbing, metric trainer.samples_per_s, no expected movement, confidence or minimum sample). Opened: vLLM `vllm/config/cache.py` at b1388b1 (S51); `lab/refine/metrics.jsonl` lines 9, 10, 13 and 14 (S53); `lab/README.md`, `lab/pod_a_teacher.sh` and `lab/controller.py` on main at 97428e4 (S55). Stopped at the budget this pass gave the topic, with the slowdown question left open for the experiment.
