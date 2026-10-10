# From samples to joules per answer: request windows, the idle baseline, concurrent requests and batching

Track: telemetry. Opened and written 2026-10-10. Scope: how published work turns power samples into energy per token and per answer, which window to integrate, what to do with idle power, how to split energy between requests that overlap on one GPU, and how much batching moves joules per token. Companion notes: `power-sampling.md` (what the samplers report and their error) and `power-and-clock-levers.md` (caps, clocks and power modes). Every source is listed in `SOURCES.md` with its URL or DOI and the date it was opened; bracketed IDs such as [S23] point there.

How to read a finding. Each finding is one bullet. It opens with its label and an ID (AT-01 and so on), states what was found with its numbers and setup, and ends with its citation. Under it, "Implies for our metrics" says what it means for our metric set, and "Candidate experiment" names the node, the procedure, the expected movement and the minimum sample size. A pass cites a finding by file and ID, for example `attribution-and-baselines.md AT-14`.

Labels. `measured`: a number someone measured, with the setup stated. `practitioner`: a documented statement or recommendation from the people who build or run the tool (vendor documentation, driver source, maintainer replies, benchmark methodology). `anecdote`: a single user's report. `opinion`: argument without measurement.

Our metric set (the telemetry rubric and the Telemetry context spec): joules per answer per node, idle watts per node, sampler overhead (CPU percent and GPU interference), samples per attributed window, and attribution error against a reference reading; in the metrics file these appear as `teacher.watts_avg`, `teacher.watts_peak`, `teacher.joules_per_token`, `pod.idle_watts` and so on. Joules per answer is defined in the Telemetry context as the watts of the answering node integrated over the request's `started` and `ended` timestamps from `telemetry.request`.

Minimum sample sizes. Unpaired comparisons use n per arm = 15.7 x (CV / d)^2, rounded up and never below 5, where CV is the run-to-run coefficient of variation of the metric, d is the relative change to detect, and 15.7 = 2 x (1.96 + 0.84)^2 (two-sided alpha 0.05, power 0.8). Paired comparisons (the same prompts in both arms) use 7.85 x (CV / d)^2 pairs. Until we have measured our own CVs, per-answer energy is assumed to have CV = 10% and steady-state window energy CV = 5%. This is an assumption, not a finding: 5% is the acceptance band Vellaisamy et al. use between repeated windows [S32] and close to the 4.89% residual error Yang et al. report after good practice [S28]. The first run of any experiment below replaces the assumption with our measured CV.

## 1. Which window to integrate

### Takeaway

Every careful benchmark integrates over a window it can defend: ML.ENERGY uses the steady state in which the batch is full and drops the ramps (AT-01), Vellaisamy et al. difference an energy counter at synchronized window boundaries after 10 warm-up runs (AT-03), and POLCA warms a server with at least three requests before measuring (AT-04). Measuring across a whole benchmark, ramps and gaps included, made ML.ENERGY's own earlier power figures low (AT-02). Our definition of joules per answer, the answering node's watts between `started` and `ended`, is right in principle; the risks sit at the edges (sensor lag, `power-sampling.md` PS-17, and timestamps taken on the wrong node, PS-24) and in the first answers after a restart.

### Cited findings

- **[practitioner]** AT-01. The ML.ENERGY Benchmark measures LLM energy with Zeus over a "steady state" window in which the batch is saturated at the server's configured maximum, excludes ramp-up and ramp-down because the batch is not full there, and computes energy per response as steady-state energy per token times the average output tokens per request; each configuration was run once ("due to the high monetary cost of even a single run"), on A100 and H100 with vLLM. Citation: J.-W. Chung et al., "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization", 2025, NeurIPS 2025 Datasets and Benchmarks, arXiv 2505.06371v2, Sec. 3.2 and 4, https://arxiv.org/abs/2505.06371 [S23]

  Implies for our metrics: `teacher.joules_per_token` for the slide should be a steady-state number from a saturated window, stated with its concurrency, while per-answer energy integrated over each request is a different and noisier metric; each stored number should record which method produced it (`window = steady_state` or `window = request`).

  Candidate experiment: pod B, a declared 60-minute slot (teacher engine start-up is about 270 s): the teacher at concurrency 8 for 5 minutes of captions, J/token computed from the steady-state window and from the whole run. Expected movement: whole-run J/token higher than steady-state by the share of time spent in ramps and gaps, several percent. Minimum sample size: 3 runs.

- **[measured]** AT-02. Comparing ML.ENERGY leaderboard v2.0 (September 2024) with v3.0 (December 2025) on the same H100s, the authors note that v2 measured power over the entire benchmark rather than only the steady-state window, which likely underestimated v2's power draw, while v3's steady-state power is substantially higher. Citation: ML.ENERGY blog, "LLM Inference Energy: A Longitudinal Analysis", 2026-02-09, https://ml.energy/blog/measurement/energy/llm-inference-energy-a-longitudinal-analysis/ [S26]

  Implies for our metrics: the window choice alone moves `teacher.watts_avg` materially, so a watts number without its window cannot be compared across passes (rubric test 4); every watts metric should name its window (steady decode, whole request, whole session).

  Candidate experiment: in the AT-01 run, report `teacher.watts_avg` over the steady state and over the whole run. Expected movement: the whole-run average is lower, pulled down by ramps and idle gaps. Minimum sample size: 3 runs.

- **[practitioner]** AT-03. Vellaisamy et al. take request energy as the difference of the NVML energy counter read at synchronized window boundaries; NVML power, which "reports power in mW averaged over a one-second interval", is "polled at a nominal 50 ms cadence as smoothed diagnostics" and not used for request energy; defaults are 10 warm-up runs and 10 profiled runs with energy taken over the last five, and a kernel replay is kept only if its net power varies by at most 5% across three 500 ms windows. Citation: P. Vellaisamy, V. Lam, S. Blanton and J. P. Shen, "Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms", 2026, IEEE International Symposium on Workload Characterization (IISWC 2026, accepted), arXiv 2608.28044, Sec. III and IV-B, https://arxiv.org/abs/2608.28044 [S32]

  Implies for our metrics: once the R9700 counter is confirmed (`power-sampling.md` PS-03), counter differencing at request boundaries is our reference method for pod joules per answer and the reference for the attribution-error metric; the 5% agreement rule between repeated windows is a ready-made test for marking a window high or low confidence.

  Candidate experiment: pod B: 10 warm-up and then 10 measured caption requests, energy by counter difference and by integrating `socket_w`. Expected movement: the two agree within 5% on the last five requests if the counter is live and the averaged field's lag is padded. Minimum sample size: 10 measured requests.

- **[practitioner]** AT-04. POLCA's authors found that serving frameworks allocate workspace on the first request after loading, so "the first inference executes much slower than in the steady state", and they warm up the serving framework "with at least three requests before taking measurements"; they also align counters that update on intervals (and lag) with instantaneous ones by matching peaks. Citation: P. Patel et al., "Characterizing Power Management Opportunities for LLMs in the Cloud", 2024, ASPLOS '24, DOI 10.1145/3620666.3651329, Sec. 3.4, https://foci.uw.edu/papers/asplos24-llmpower.pdf [S36]

  Implies for our metrics: after every teacher or student restart the first answers are slower and costlier, so the request log should mark the first three answers after a model load; they stay in per-day energy but are excluded from per-answer statistics and before/after comparisons.

  Candidate experiment: Jetson, at the next planned student restart (Ilan's go): time and energy of the first 5 answers to one golden question against answers 6 to 15. Expected movement: answer 1 clearly slower and costlier; from answer 4 on within the normal spread. Minimum sample size: 3 restart sequences.

## 2. The idle baseline

### Takeaway

Published practice splits two ways. MLPerf Power and the ML.ENERGY leaderboard report gross energy, idle share included (AT-07, AT-06), while Vellaisamy et al. measure a "resident floor" with the model loaded and nothing running and carry it as a named term inside the total (AT-05). ML.ENERGY's v3 analysis treats static power at low utilization as the waste that batching removes (AT-06), and GPU idle sits near 13 to 20% of board power in the published traces and in ours (AT-08, AT-09). Our pods idle at 38 to 40 W against about 200 W while generating, so the policy choice alone moves teacher joules per answer by about 20%. The rubric requires one stated idle policy everywhere. The evidence supports reporting gross joules per answer as the headline (it is what the node actually spent and what routing saves), net joules (gross minus idle watts x window length) beside it, and defining idle per node as the model-resident floor measured by the sampler, with its timestamp.

### Cited findings

- **[practitioner]** AT-05. Vellaisamy et al. measure the resident floor with the model loaded, the runtime initialized, the GPU synchronized and no kernels issued, using counter-derived power as the primary floor value; they model E_NVML = E_floor + E_active + E_residual with E_floor = P_floor x T_window, so the floor is an attribution term inside the measured total rather than something subtracted from reported request energy. Citation: P. Vellaisamy et al., "Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms", 2026, IISWC 2026 (accepted), arXiv 2608.28044, Sec. III-D, Eqs. 6 and 8, https://arxiv.org/abs/2608.28044 [S32]

  Implies for our metrics: `pod.idle_watts` should be the model-resident floor (teacher loaded, no requests), not the empty-pod reading, and both gross and net (gross minus floor x duration) should be stored per answer, so a pass can compare either without recomputing from raw samples.

  Candidate experiment: pod A through its sampler at a quiet time: 5 minutes with the teacher loaded and no requests, compared with the empty-pod idle from `power-sampling.md` PS-11. Expected movement: the resident floor differs from empty-pod idle by less than 10 W; the result fixes the one idle value used everywhere. Minimum sample size: 300 one-second readings per state.

- **[measured]** AT-06. The ML.ENERGY leaderboard v3.0 analysis (46 models, 7 tasks, 1,858 configurations on H100 and B200, vLLM 0.11.1) defines energy per response as energy per token times output tokens, finds Problem Solving responses averaging 4,625 J against 184 J for Text Conversation (25x, driven by output length), and explains low-batch inefficiency as static power drawn while the GPU makes little progress; increasing batch size "can unlock 3-5x energy per token reduction". Citation: ML.ENERGY blog, "Diagnosing Inference Energy Consumption with the ML.ENERGY Leaderboard v3.0", 2026-01-29, https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/ [S24]

  Implies for our metrics: at batch 1 our teacher works in the static-power-dominated regime, and a net-of-idle headline would hide exactly the waste batching removes, so the headline should stay gross; output length dominates per-answer energy, so per-answer comparisons across nodes (student against teacher) must be made at matched output lengths or reported per token as well.

  Candidate experiment: pod A through its sampler: the same 20 caption prompts at `max_tokens` 32 and 128. Expected movement: per-answer energy about 3x higher at 128 tokens (the fixed prefill cost keeps it below 4x). Minimum sample size: 20 paired prompts (CV 10%, d 20%: 7.85 x (0.1/0.2)^2 = 2, so 20 also traces the curve).

- **[practitioner]** AT-07. MLPerf Power measures full-system power over the execution phase only, excluding setup and teardown, with no separate idle baseline or subtraction; for the smallest systems the whole device, always-on components included, is measured. Citation: A. Tschand et al., "MLPerf Power: Benchmarking the Energy Efficiency of Machine Learning Systems from Microwatts to Megawatts for Sustainable AI", 2024 (revised 2025), arXiv preprint 2410.12032, Sec. IV-A and IV-B, https://arxiv.org/abs/2410.12032 [S27]

  Implies for our metrics: gross reporting matches the reference benchmark's convention; our setup and teardown (the teacher's ~270 s engine start, the student's ~40 s load) belong in per-day energy but not in per-answer energy, so the daily total must be computed from the node's whole timeline, not as a sum of answers.

  Candidate experiment: fixture plus logs on this machine, once one day of pod A samples and requests exist: per-day energy as the sum of per-answer gross energy against the integral of the node's whole timeline. Expected movement: the timeline integral is several times larger, because the node is idle most of the day. Minimum sample size: 1 full day of logs, repeated for 3 days.

- **[measured]** AT-08. In POLCA's server traces (DGX A100), GPU power during the communication phase of Flan-T5 training "goes down all the way to 20%" of TDP, "which corresponds to the idle power of the GPUs". Citation: P. Patel et al., "Characterizing Power Management Opportunities for LLMs in the Cloud", 2024, ASPLOS '24, DOI 10.1145/3620666.3651329, Sec. 4.1, https://foci.uw.edu/papers/asplos24-llmpower.pdf [S36]

  Implies for our metrics: an idle floor of roughly 13 to 20% of board power is normal (ours is 38 to 40 W of 300 W, 13%), so a measured pod idle far outside that range points at a sampler or field problem; the floor also changes with temperature (`power-sampling.md` PS-23), so idle should be re-measured, not set once.

  Candidate experiment: pod B: idle floor measured right after 5 minutes of load and again after 5 minutes of rest. Expected movement: hot idle above cool idle by a few watts. Minimum sample size: 3 hot and cool pairs of 60 one-second readings.

- **[measured]** AT-09. On our pods idle is about 38 to 40 W and the teacher draws about 200 W while generating (300 W peak), so idle is about a fifth of the teacher's gross power during an answer (39 / 200, our arithmetic). Citation: our measurements, T3 task page and O1 rubric task page, https://app.notion.com/p/3f596ef5fec5816da5e7c992c3f5b7e7 [S42]; PR #37 and lab/README.md on ctx/runtime/r2-gpu-lab, https://github.com/raohemdutt/Knighthacks26/pull/37 [S41]

  Implies for our metrics: gross against net moves teacher joules per answer by about 20%, more than most effects we will test, so the idle policy must be fixed before any before/after comparison (the rubric's old-versus-new rule), and both values must be stored from the first sampler run.

  Candidate experiment: pod A through its sampler: the first 20 logged answers computed both ways. Expected movement: net is about 0.80 x gross for decode-bound answers and closer to 0.85 for short prefill-heavy captions, where power runs above the 200 W average. Minimum sample size: 20 answers.

## 3. Concurrent requests

### Takeaway

When requests overlap on one GPU there is no single right split. Even splits (energy over batch size) and token-proportional splits are the common shortcuts, and against exact Shapley values from replayed subsets the token-proportional split was off by about 0.44 to 0.46 in normalized L1, which a small calibrated model cut to 0.12 to 0.18 (AT-10). Per-request energy models fit well: affine in output tokens with a fixed prefill cost (AT-11), or bilinear in input and output tokens with R^2 above 0.96 (AT-12), and prefill spikes make the fixed cost large for long inputs such as images (AT-13). For us: the teacher's request-window integral double counts whenever windows overlap, so attribution must split each sample interval's energy across the requests active in it, a conservation check (attributed energies sum to the node's integral) must hold on every replay, and the attribution method must be stored with each number.

### Cited findings

- **[measured]** AT-10. JouleShare replays subsets of requests under vLLM, integrates GPU power telemetry, and computes exact Shapley energy for each request as ground truth; over 16 model and workload runs on three data-center GPUs, token-proportional attribution deviated from exact Shapley by 0.440 normalized L1 under static batching and 0.458 under continuous batching, and the lightweight JCalib model reduced this to 0.116 and 0.177. Citation: Q. Luo, K. Li, Z. Wang, D. Wang and Y. Chen, "Request-Level Energy Attribution for Batched LLM Serving", 2026, arXiv preprint 2608.00026, abstract, https://arxiv.org/abs/2608.00026 [S34]

  Implies for our metrics: token-proportional attribution is a known-biased default; when we use it each number should record `attribution = token_proportional`, comparisons of joules per answer taken from overlapping windows are low confidence, and the attribution-error metric can be measured exactly this way (Shapley from replays) on small groups of requests.

  Candidate experiment: pod B, a declared 60-minute slot: 3 groups of 3 caption requests, each group replayed in all 8 subsets (24 runs, the empty subset giving the floor), Shapley shares against token-proportional shares. Expected movement: normalized L1 between the two around 0.4, as in the source. Minimum sample size: 3 groups (24 runs).

- **[measured]** AT-11. Vellaisamy et al. split a batched window evenly (E_req = E / B, E_token = E / (B x N)) and fit E_request(N) = E_fixed + N x E_decode,step locally over N in {10, 128, 512}; on an H200, Llama-3.2-1B at B = 16 and context 2048 gave E_fixed = 276.78 J and E_step / B = 351.06 mJ per token, while at B = 1 and context 512 the affine fit's residual reached +38.79% at N = 10; token energy fell from 7.46 to 0.72 J per token as output grew from 10 to 512 tokens (B = 16, context 4K). Citation: P. Vellaisamy et al., "Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms", 2026, IISWC 2026 (accepted), arXiv 2608.28044, Sec. III-A and V-A3, Table III, https://arxiv.org/abs/2608.28044 [S32]

  Implies for our metrics: for short answers (our captions are about 28 tokens) the fixed prefill cost is a large share of joules per answer, so a per-token number alone misstates the student-against-teacher comparison; `telemetry.request` already carries `tokens_in` and `tokens_out`, enough to fit E_fixed and E_step per node and model, and the fit's residual at short outputs says how far to trust it there.

  Candidate experiment: pod A through its sampler (requests only): 5 repeats of the same image prompt at `max_tokens` 16, 64 and 256, fitting E = E_fixed + N x E_step. Expected movement: E_fixed around 0.25 to 0.35 kJ (about 1.1 s of prefill at 250 to 300 W) and E_step around 30 J per token at batch 1 (200 W / 6.6 tok/s). Minimum sample size: 15 requests (3 lengths x 5).

- **[measured]** AT-12. Wilkins et al. fit per-model energy as e = a0 x t_in + a1 x t_out + a2 x t_in x t_out (and runtime the same way) by least squares on an 8x A100 40 GB node (GPU energy through PyJoules and NVML, batch 32, KV cache disabled), with energy R^2 from 0.964 (Falcon 7B) to 0.980 (Mixtral 8x7B); output tokens had the larger effect on energy (F = 126.63 against 15.86 for input tokens), and trials were repeated until runtime was within 0.5 s of the mean at 95% confidence, at most 25 per setting. Citation: G. Wilkins, S. Keshav and R. Mortier, "Offline Energy-Optimal LLM Serving: Workload-Based Energy Models for LLM Inference on Heterogeneous Systems", 2024, HotCarbon 2024, arXiv 2407.04014, Sec. 3 and 4, https://arxiv.org/abs/2407.04014 [S33]

  Implies for our metrics: a fitted energy model per node and model gives the routing hook a prediction of joules per answer before the request runs (what an energy-aware router needs), and gives attribution a principled split for overlapping requests (each request's predicted share); the stopping rule (within 0.5 s at 95% confidence, at most 25 trials) is a usable minimum-n rule for our sweeps.

  Candidate experiment: fixture plus logs on this machine, once 200 teacher and 200 student requests are logged with energy: fit the bilinear model per node and report R^2. Expected movement: R^2 above 0.9 for each node, lower for the teacher if overlapping requests are not yet attributed. Minimum sample size: 200 requests per node.

- **[measured]** AT-13. On A100-80GB servers with DCGM at 100 ms, every inference showed a power spike at its start, "often going beyond GPU TDP", during prompt processing, followed by a longer, stable, lower plateau during token generation; the peak and mean power depend mainly on the input prompt size and the batch size, while latency depends mainly on output size. Citation: P. Patel et al., "Characterizing Power Management Opportunities for LLMs in the Cloud", 2024, ASPLOS '24, DOI 10.1145/3620666.3651329, Sec. 4.2 (Insights 4 and 5), https://foci.uw.edu/papers/asplos24-llmpower.pdf [S36]

  Implies for our metrics: image prompts (large inputs) explain the 300 W peaks we saw on the teacher, so caption answers are prefill-heavy; splitting joules per answer by phase needs the time of the first token, which `telemetry.request` does not carry (only `started` and `ended`), so an optional `first_token` timestamp would be the smallest useful addition.

  Candidate experiment: pod A through its sampler: 20 caption requests (image) against 20 text requests with similar output length. Expected movement: captions show a spike near 300 W for about the first second and 20 to 30% more energy per answer. Minimum sample size: 20 per arm (CV 10%, d 20% needs 4; 20 also shows the shape).

## 4. Batching moves joules per token more than anything else we control

### Takeaway

Across every source, the batch is the dominant lever. ML.ENERGY's v3 data show J/token falling 6.2x for Qwen3-8B on one H100 and 10.1x for Llama 3.1 70B on four H100s between 8 and about 500 concurrent sequences, and the smallest batch in that data is 8 (AT-14), so our batch-1 teacher sits further up the curve than any published point. The gain shrinks when the KV cache limits the batch (AT-15) and with long contexts (6.31x at 512 tokens of context but 1.17x at 4K for short outputs, AT-17); software alone moved J/token 15 to 41% (AT-18). No published J/token measurement on an AMD GPU with vLLM turned up in this search, so our R9700 numbers would be new. Our teacher's ~26 J/token at batch 1 (AT-20) is the old value the batching experiment should move.

### Cited findings

- **[measured]** AT-14. In the ML.ENERGY Leaderboard v3.0 data (lm-arena-chat task, vLLM, steady state), Qwen3-8B on one H100 went from 0.444 J/token at `max_num_seqs` 8 (427 W) to 0.0721 J/token at 512 (616 W), a 6.2x drop, and on one B200 from 0.430 to 0.0531 (8.1x); Llama-3.1-70B-Instruct on four H100s went from 3.759 to 0.373 J/token (10.1x) and on two B200s from 3.569 to 0.189 (18.9x); the smallest batch in the data is 8. Citation: ML.ENERGY Leaderboard v3.0 data files `Qwen__Qwen3-8B__lm-arena-chat.json` and `meta-llama__Llama-3.1-70B-Instruct__lm-arena-chat.json` (index last updated 2026-02-16), github.com/ml-energy/leaderboard, public/data/models, https://github.com/ml-energy/leaderboard/tree/master/public/data/models [S25]

  Implies for our metrics: the "about 6x for Qwen3-8B, 10x for Llama 3.1 70B" figures in our design notes check out against the data (H100 rows); per-token energy falls while total power rises, so `teacher.watts_avg` goes up when batching works, and a watts-only stats page would show batching as a regression, which is why joules per token and per answer must sit next to watts.

  Candidate experiment: pod B in a declared 60-minute slot, or pod A through its sampler at a quiet time: the teacher at concurrency 1, 4, 8 and 16 on the same caption prompts, 5 steady windows of 30 s per arm. Expected movement: J/token falls from about 26 at concurrency 1 to between 4 and 9 at concurrency 8 (a 3x to 6x drop, the lower end because a 27B W4A16 model on a Triton fallback kernel may turn compute-bound sooner), while watts rise from about 200 toward the 300 W limit. Minimum sample size: 5 windows per arm (CV 5%, d 50%: 15.7 x (0.05/0.5)^2 = 0.2, floor 5).

- **[measured]** AT-15. The ML.ENERGY v3.0 analysis reports Qwen 3 32B on one B200 at 0.209 J/token with batch 128 and 0.151 J/token at batch 512, notes that DeepSeek R1 had not saturated the GPU even at the largest batch that fits in memory, and explains that longer outputs raise KV-cache use per response, which caps the batch and raises energy per token. Citation: ML.ENERGY blog, "Diagnosing Inference Energy Consumption with the ML.ENERGY Leaderboard v3.0", 2026-01-29, https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/ [S24]

  Implies for our metrics: the teacher's usable batch is capped by KV-cache memory (the model already holds 84% of VRAM with about 10x concurrency at 16k tokens, `power-sampling.md` PS-12), so our concurrency range is roughly 1 to 10 and the expected gain sits at the low end of the published ranges.

  Candidate experiment: the AT-14 sweep extended to concurrency 10 at 16k context, logging KV-cache usage from vLLM's metrics. Expected movement: J/token flattens before concurrency 10 as the KV cache fills. Minimum sample size: 5 windows per arm.

- **[measured]** AT-16. In the ML.ENERGY Benchmark's GPQA runs on H100 with reasoning models from 8B to 235B, energy per generation fell between 2.6x and 5.8x as the maximum batch size grew from 4 to 64 (for example 9,713.7 to 2,770.8 for the 8B model); separately, TDP-based estimates overestimated measured energy by up to 4.1x (CodeGemma 2B on H100). Citation: J.-W. Chung et al., "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization", 2025, NeurIPS 2025 Datasets and Benchmarks, arXiv 2505.06371v2, Table 1 and Sec. 4.1, https://arxiv.org/abs/2505.06371 [S23]

  Implies for our metrics: per-answer energy, not only per-token energy, falls with batch; and any number estimated from the R9700's 300 W board power instead of measured would overstate decode energy by about 1.5x (300 / 200 W) and idle energy by about 7x (300 / 40 W), so estimates must never stand in for samples on the slide.

  Candidate experiment: fixture plus logs on this machine, once the pod sampler has run: for 50 logged teacher answers, compare the 300 W x duration estimate with the measured integral. Expected movement: the estimate is 1.3x to 1.6x the measurement for decode-bound answers. Minimum sample size: 50 answers.

- **[measured]** AT-17. For Llama-3.2-1B on an H200 with 10 output tokens, the batch-16 against batch-1 token-energy gain fell from 6.31x at context 512 to 1.17x at context 4K, so the benefit of batching is bounded by context length. Citation: P. Vellaisamy et al., "Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms", 2026, IISWC 2026 (accepted), arXiv 2608.28044, abstract and Sec. V, https://arxiv.org/abs/2608.28044 [S32]

  Implies for our metrics: our captions carry an image and short outputs, so their batching gain may be far smaller than chat benchmarks suggest; every batching result must report its context length and image count or it does not transfer to the routing decision.

  Candidate experiment: within the AT-14 sweep, two prompt types: short text under 512 tokens and the 720p image caption. Expected movement: the captions' J/token gain at concurrency 8 is well below the text gain (for example 2x against 5x). Minimum sample size: 5 windows per arm per prompt type.

- **[measured]** AT-18. On the same H100s, moving from vLLM 0.5.4 (leaderboard v2) to 0.11.1 (v3) lowered energy per output token by 15 to 41% (Llama 3.1 8B at batch 64: 0.20 to 0.12 J/token), with the largest gains at small batch sizes, where CPU-side overheads leave the GPU underutilized and waste static power; vLLM's asynchronous scheduling is named as a mitigation. Citation: ML.ENERGY blog, "LLM Inference Energy: A Longitudinal Analysis", 2026-02-09, https://ml.energy/blog/measurement/energy/llm-inference-energy-a-longitudinal-analysis/ [S26]

  Implies for our metrics: our vLLM build (a pinned commit with a gfx1201 patch) is itself a lever on teacher J/token at batch 1, so the engine version must be part of every metrics row's conditions, and before/after comparisons across engine builds measure the engine, not anything else.

  Candidate experiment: pod B: the same batch-1 caption workload with and without vLLM's asynchronous scheduling option, if the pinned commit supports it. Expected movement: J/token 5 to 15% lower with asynchronous scheduling at batch 1. Minimum sample size: 5 windows per arm (CV 5%, d 10%: n = 3.9, floor 5).

- **[measured]** AT-19. Running LLaMA 65B sharded over 8 to 32 V100s (batch sizes 64 to 512, maximum generation length 512, GPU energy from DCGM divided by decoded tokens), "it takes about 3-4 Joules for a output token", and energy per output token rose as the number of shards grew. Citation: S. Samsi et al., "From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference", 2023, arXiv preprint 2310.03003, Sec. III-C and IV-C, https://arxiv.org/abs/2310.03003 [S35]

  Implies for our metrics: a batched 2023 baseline of 3 to 4 J per token for a 65B model puts our ~26 J per token for a 27B model at batch 1 in context: the gap is mostly batch size, not the card, so the slide must not set our batch-1 number beside batched published numbers.

  Candidate experiment: the AT-14 sweep supplies the comparison; report the concurrency-8 J/token next to this baseline. Expected movement: the concurrency-8 teacher lands within 2x of 3 to 4 J per token. Minimum sample size: as in AT-14.

- **[measured]** AT-20. From our own numbers: the teacher serving one request at a time (batch 1) draws about 200 W at 7.7 tok/s, which is about 26 J per output token gross and about 21 J net of a 39 W idle; a 28-token 720p caption (TTFT 1.09 s, decode 6.6 tok/s) takes about 5.3 s, about 1.1 kJ gross (our arithmetic from the measured values, no window or sampler recorded with the 200 W). Citation: our measurements, PR #37 and lab/README.md on ctx/runtime/r2-gpu-lab, https://github.com/raohemdutt/Knighthacks26/pull/37 [S41]; pod idle from the T3 task page, https://app.notion.com/p/3f596ef5fec5816da5e7c992c3f5b7e7 [S42]

  Implies for our metrics: these are the old values of `teacher.joules_per_token` and of the caption's joules per answer; both are low confidence (no sampler, window or field recorded) until the pod sampler re-measures them, and they are the baseline the batching and routing experiments must move.

  Candidate experiment: pod A through its sampler: 20 captions with energy integrated per request. Expected movement: 0.9 to 1.3 kJ gross per caption. Minimum sample size: 20 captions.
