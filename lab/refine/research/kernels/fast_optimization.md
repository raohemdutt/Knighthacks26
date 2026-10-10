# How the settings optimization fits one night on the Strix Halo pods: which optimizer, how many trials, which configuration

Decision: Ax 1.x (ax-platform 1.3.1, BoTorch, the default "fast" method with qLogNEI) run
asynchronously, one trial per free worker, warm-started from the baseline and four prior arms
instead of Sobol points, with repeats inside one server spawn and the trials grouped by the card
state they need, so each restart serves a whole block of measurements and reuses a warm compile
cache where it can; inference (R38) runs on pod D and training (R39) on pod C at the same time when
pod C passes its free-memory check, and every extra pod AMD grants tonight joins the inference lane.
Expected: a confirmed win after about 5 evaluations when one of the priors is right (about 0.69 by
R35's own odds) and about 17 when none is, plus 6 confirmation spawns; 17 inference trials fit in 3
hours on one pod and 35 on two, 15 training trials on one pod and 31 on two. The configuration is
`lab/refine/research/kernels/data/ax-config-night.json`, whose format is in "Approach" below.

Research note for the kernels track (Task Board R49), searched on 2026-10-10 with 2025 and 2026
sources first. Sources are indexed in `SOURCES.md` under "Added by R49" with ids `B1` to `B59`;
experiment ids `FO1` to `FO10` are defined once each and ranked at the end. What it covers: the
cost of one trial on our pods and harness, the trial counts that fit in three hours, grouping trials
by card state, the open-source optimizers at a budget of 30 to 60 evaluations, the literature on
tuning LLM inference servers, what Ax 1.x switches on today, parallel evaluation across pods,
whether pod C can host trials, public Strix Halo numbers as priors and as expected ranges, and a
separate recipe for the kernel lane, where a measurement costs milliseconds. It does not run
anything on a pod: every number of ours is from a committed file or a GPU Queue row.

## 1. The cost of one trial: restarts dominate

- **[measured]** The stock vLLM 0.19.2 teacher (`RedHatAI/Qwen3.8-27B-INT4`, vLLM at b1388b1 plus
  `lab/vllm-gfx1201.diff`) on pod D (account D, gfx1151) was ready 523 s after a cold start with
  no torch.compile cache yet, in the R30 slot of 2026-10-10 (21:47 to 21:57 UTC); pod C took 297 s
  with a warm cache. Source: [B49] GPU Queue row R30, Result; [B51] `lab/README.md`, "The same build
  on pod D, from scripts alone (R30)".

  Implication: every trial that changes a launch flag or a process environment variable pays 295 to
  523 s before its first token, more than the measurement itself at the screen fidelity of section
  2; the trial count per night is set by restarts, not by the optimizer's model.

  Experiment: FO2, the compile-cache classes: start the stock teacher on pod D twice, then once with
  `GPU_MAX_HW_QUEUES=1`, then once with `VLLM_ROCM_USE_SKINNY_GEMM=0`, reading `ready_s` from each
  harness row. Expected movement: about 295 s for the first (523 if R30's cache does not match the
  baseline's flags), then 295, 295 and 523 s (the last changes the compile key, the third does not).
  Minimum sample size: 4 spawns, two 15-minute slots. Confidence: medium (0.6).

- **[measured]** In the R18 slot (2026-10-10, 22:15 to 22:21 UTC) the hub proxy answered 500
  `model_server_down` for 294 s while the same R30 teacher started a second time on pod D, then 200.
  Source: [B50] GPU Queue row R18, Result.

  Implication: a warm restart on pod D is about 294 s, matching pod C's 297 s; the config file uses
  295 s for a warm restart and 523 s for a cold one.

  Experiment: FO2, as specified in the previous finding.

- **[measured]** On pod C at batch 1, 16k context, thinking off: time to first token 0.39 s and
  decode 2.88 tok/s on text (31 in, 77 out), 120 W average and 129 W peak while decoding, 8 W idle
  with no server and 9.2 W with the teacher loaded. Source: [B51] `lab/README.md`, "Teacher on SJC
  today (pod C, Strix Halo)".

  Implication: one request of 64 generated tokens takes 22.3 s and one of 128 tokens 44.5 s (derived:
  0.39 + 63 / 2.88 and 0.39 + 127 / 2.88). Power is nearly constant while decoding, so joules per
  token moves as 1 / tok/s, and the standard error of joules per token can be scaled from the
  standard error of the per-request decode rates inside one spawn.

  Experiment: FO3, the noise split: three spawns of the baseline on pod D, each with three
  64-token requests at batch 1. Expected movement: within-spawn sd of decode tok/s at most 2 percent
  and between-spawn sd at most 4 percent unless the spawns fall in bands as they did on gfx1201.
  Minimum sample size: 3 spawns of 3 requests. Confidence: medium (0.5).

- **[practitioner]** The benchmark harness counts a repeat as a server spawn; before each batch
  size it sends one warm-up request per prompt one after another (four prompts in
  `fixtures/lab/bench/prompts.json`) and then one round at the batch size, also inside a shared
  spawn; its defaults are 128 generated tokens, one request per prompt per stream, a 10 s idle
  reading taken twice per spawn and a 900 s readiness bound. Source: [B52] `lab/bench.py`, module
  docstring, `Session.measure` and the `DEFAULT_*` constants.

  Implication: a trial of three repeats at the defaults spends three restarts and about 220 s of
  warm-up per spawn; the night protocol keeps the harness but asks for one spawn per trial with
  three requests per stream inside it (`--repeats 1 --requests-per-stream 3`), 64 tokens for the
  screen and no probe.

  Experiment: FO3, as specified in the previous finding.

- **[practitioner]** At vLLM b1388b1 the compilation config's hash includes every declared field
  except a short opt-out list, so `cudagraph_mode` and the capture sizes change it; the
  torch.compile cache key also takes every vLLM environment variable except a few such as
  `VLLM_CACHE_ROOT` (default `~/.cache/vllm`), so `VLLM_ROCM_USE_SKINNY_GEMM`,
  `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` and `VLLM_DISABLED_KERNELS` change it too. Source: [B30]
  vllm `vllm/config/compilation.py::CompilationConfig.compute_hash` at b1388b1; [B31]
  `vllm/envs.py::compile_factors` at b1388b1.

  Implication: R35's ten settings fall into a warm-restart class (`gpu_max_hw_queues` and
  `amdgcn_use_buffer_ops`, process variables that are not vLLM's, so the compile cache is reused,
  though Triton recompiles its own kernels) and a cold-restart class (`linear_kernel` through
  `VLLM_DISABLED_KERNELS`, `cudagraph_mode`, `vllm_rocm_use_skinny_gemm`,
  `fla_packed_recurrent_decode`, and `num_speculative_tokens`, which adds a drafter to compile);
  `mamba_ssm_cache_dtype`, `attention_backend` and `enable_prefix_caching` are not verified and are
  treated as cold. The cache lives in the persistent home, so each compile class is paid cold once
  per pod and night.

  Experiment: FO2, as specified in the first finding of this section.

- **[practitioner]** At vLLM b1388b1, `max_cudagraph_capture_size` defaults to min(2 x
  `max_num_seqs`, 512) and the default capture list is [1, 2, 4] + range(8, 256, 8) + range(256,
  max + 1, 16); `skip_mm_profiling` "reduces engine startup time" by profiling the language
  backbone only. Source: [B30] `vllm/config/compilation.py` at b1388b1, `cudagraph_capture_sizes`
  and `max_cudagraph_capture_size` docstrings; [B32] `vllm/config/multimodal.py::skip_mm_profiling`
  at b1388b1.

  Implication: with R35's fixed `max_num_seqs` 256 every start captures 51 graph sizes (derived: 3 +
  31 + 17) although the night measures batch 1 and 8 only. Capturing [1, 2, 4, 8, 16] and passing
  `--skip-mm-profiling` for the baseline and every trial alike changes no measured number at those
  batch sizes and may shorten every restart; the size of the saving is unmeasured.

  Experiment: FO1, the restart diet: on pod D, a cold and a warm start of the stock teacher with
  and without `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}' --skip-mm-profiling`,
  reading the serve log's load, compile and capture times. Expected movement: warm `ready_s` from
  295 s down by 60 to 150 s. Minimum sample size: 4 spawns (cold and warm per arm). Confidence: low
  (0.35).

- **[practitioner]** R35's ten inference parameters are all launch flags or environment variables
  of the server process (`linear_kernel` through `--dtype` and `VLLM_DISABLED_KERNELS`,
  `num_speculative_tokens` through `--speculative-config`, `gpu_max_hw_queues` through
  `GPU_MAX_HW_QUEUES`, and so on); none is a field of a request. Source: [B56] PR #235,
  `lab/refine/research/kernels/data/searchspace-inference-gfx1151.json`, `mapping`.

  Implication: no R35 setting is restart-free. The restart-free axis is the request side only:
  concurrency (batch size), generated tokens, requests per stream and thinking on or off. Each spawn
  therefore measures the objective at batch 1 and, for the current top three, the labeling case at
  batch 8, without another restart.

  Experiment: FO7, the concurrency curve of the live baseline on pod C in attach mode (no restart):
  batches 1, 4, 8 and 16 at night, dropping windows in which other requests ran. Expected movement:
  joules per token at batch 8 at most one quarter of batch 1, per-stream tok/s down less than 15
  percent. Minimum sample size: 3 requests per stream per batch size. Confidence: low (0.4).

- **[measured]** The smoke LoRA run (`lab/smoke_lora.py`, `Qwen/Qwen3.5-4B`, bf16, LoRA r 16, batch
  4, accumulation 1, 20 steps, 256 tokens, 48 examples) trained at 0.51 samples/s with a peak of
  8.7 GiB on one R9700, the old chip. Source: [B53] `lab/refine/metrics.jsonl:13-14`.

  Implication: a 20-step trial at R36's fixed effective batch of 16 takes about 627 s of training
  (derived: 20 x 16 / 0.51) plus a model load that is not measured on gfx1151 (60 s assumed); the
  Strix Halo rate waits on R36's TS1.

  Experiment: FO5, the R39 night run: each trial 20 steps at effective batch 16 with the first 5
  excluded from samples/s. Expected movement: samples/s up 20 percent or more over the baseline if
  one of the training priors holds. Minimum sample size: 15 trials and 3 confirmation runs per arm.
  Confidence: low (0.4).

- **[practitioner]** R36's training space has ten parameters (`gdn_kernel`,
  `gradient_checkpointing`, `per_device_train_batch_size` with a derived accumulation that keeps the
  effective batch at 16, `packing`, `group_by_length`, `aotriton_experimental`, `torch_compile`,
  `tunableop`, `causal_conv1d_kernel`, `blas_backend`) and pins 50 steps per trial with 10 warm-up
  steps excluded. Source: [B57] PR #238,
  `lab/refine/research/kernels/data/searchspace-training-gfx1151.json`.

  Implication: training trials restart a Python process, not a server, so the split is in-process
  (batch size, checkpointing, length grouping, packing: changeable without reloading the model) and
  process-level (the kernel choices, `torch_compile` and the environment variables). The model load
  is small next to the 20 steps, so the split saves little; the trial budget is set by the steps.

  Experiment: FO5, as specified in the previous finding.

## 2. Trials that fit in three hours

The arithmetic below uses section 1's measured inputs: restart 523 s cold and 295 s warm, 2.88 tok/s
and 0.39 s to first token, two 10 s idle readings and about 5 s to stop per spawn. The batch-8
per-stream rate (2.6 tok/s) and the training model load (60 s) are assumptions marked as such.

| Inference trial, batch 1 | Cold | Warm | Trials in 3 h, one pod |
| --- | --- | --- | --- |
| L0, the driver as first written: 3 spawns per trial, 128 tokens, 4 requests, probe | 2,523 s for the trial (first spawn cold) | | 4.3 |
| L1, repeats inside one spawn (3 requests), 128 tokens | 904 s | 676 s | 11.9 cold, 16.0 warm |
| L2, plus the 64-token screen | 726 s | 498 s | 14.9 cold, 21.7 warm |
| L3, plus one warm-up request instead of one per prompt (a harness change) | 637 s | 409 s | 17.0 cold, 26.4 warm |
| L2 with each compile class run twice (one cold, one warm: 409 s mean restart) | 612 s | | 17.6; 35.3 on two pods; 70.6 on four |
| Training, 20 steps x 16 samples at 0.51 samples/s plus 60 s load | 687 s | | 15.7; 31.4 on two pods; 5.2 with 3 process repeats |
| Confirmation spawn, batch 1, 128 tokens, 3 requests, warm | 676 s | | |

- **[opinion]** The ladder above is derived from the measured restart, decode and harness numbers,
  not measured end to end: the driver as first written (three spawns per trial at the harness
  defaults) fits about 4 trials in three hours on one pod, and the night protocol (one spawn per
  trial with three requests, 64 tokens, compile classes run twice) fits about 17.6 on one pod and
  35.3 on two. Source: [B49] R30 row; [B50] R18 row; [B51] `lab/README.md`; [B52] `lab/bench.py`.

  Implication: of the speed-ups, repeats inside one spawn give the most (about 3 times), the warm
  compile cache next (523 to 295 s per restart), the 64-token screen and a one-request warm-up about
  20 percent each, and extra pods scale linearly; a multi-fidelity method can only save on the
  measurement part, which is 25 to 36 percent of a screen trial (derived: (726 - 548) / 726 and
  (498 - 320) / 498).

  Experiment: FO4, the R38 night run on pod D, as set out under "Decisions". Expected movement: joules
  per token at batch 1 down 20 percent or more within the first 5 trials with about 0.69 odds, and the
  run log's median trial wall clock within 15 percent of 612 s. Minimum sample size: 17 trials and 6
  confirmation spawns. Confidence: low (0.4).

- **[opinion]** The training lane at 20 steps and an effective batch of 16 fits about 15.7 trials in
  three hours on one pod at the old chip's 0.51 samples/s, and 5.2 if each trial ran three separate
  processes as the R39 task text first asked. Source: [B53] `lab/refine/metrics.jsonl:13`; [B57] PR
  #238.

  Implication: the training lane takes one process per trial and reads its noise from the per-step
  samples/s of steps 6 to 20; three repeats are for the confirmation of the best arm only. A faster
  arm shortens its own trial, so a run that finds a good region fits more trials.

  Experiment: FO5, as specified in section 1.

## 3. Grouping trials by card state

- **[opinion]** For 18 inference configurations in a night on pod D (the 17.6 that fit, rounded),
  the number of server restarts depends on the order: a spawn per repeat and per batch size (the
  driver as first written, 3 repeats at batches 1 and 8) needs 108 restarts, about 12 hours of
  restarts alone at a 409 s mean; one spawn per configuration in the optimizer's free order needs 18,
  nearly all cold because almost every suggestion is a new compile class (about 2.6 hours of
  restarts); one spawn per configuration with every restart-free measurement of that configuration
  run back to back in it, and the configurations ordered so each compile class is used twice, needs
  18 restarts of which 9 are cold (about 2.0 hours: 9 x 523 + 9 x 295 s). The confirmation adds 6
  warm restarts on purpose, because the spawns are its repeats. Source: [B49] R30 row; [B50] R18
  row; [B52] `lab/bench.py`; [B31] `vllm/envs.py::compile_factors` at b1388b1.

  Implication: grouping by card state cuts the restarts from 108 to 18 and the cold ones from about
  18 to 9, which is what makes 17 to 18 configurations fit in three hours; the config file's
  `group_by_state` rule says it: one server state per block, every restart-free measurement of that
  state in the block, blocks ordered by compile class, and on a card one lane at a time (the
  inference block, then the kernel block, then the training block when pod C cannot train).

  Experiment: FO4, as specified in section 2.

- **[opinion]** For the training lane the state is the loaded process: a trainer that can change
  batch size, checkpointing, length grouping and packing without reloading the model runs all
  trials that share the process-level settings (kernel choices, `torch_compile`, environment
  variables) back to back, one load per group. In a night of about 15 trials, about 8 to 10 distinct
  process-level combinations would be loaded instead of 15 processes, saving 5 to 7 loads of about a
  minute each (the load time is assumed, not measured). Source: [B57] PR #238; [B53]
  `lab/refine/metrics.jsonl:13`.

  Implication: grouping matters far less for training than for inference, because a training trial
  is about 11 minutes of steps against a minute of load; the rule is in the config for consistency,
  and the trainer change it needs is optional tonight.

  Experiment: FO5, as specified in section 1.

## 4. The open-source optimizers at 30 to 60 evaluations

- **[measured]** Ax's own authors benchmarked Ax against Vizier, Optuna, SMAC3 (GP and random
  forest facades), HEBO and random search on 47 problems with budgets of mostly 50 trials (10
  replications for the GP-based tools, 100 for the fast ones): on mixed or discrete spaces Ax scored
  69.4 against Vizier 55.9, Optuna 51.4, HEBO 50.7, random search 46.5, SMAC-BB 23.8 and SMAC-HPO
  15.2 (100 is the optimum, 0 the mean of 5 random trials); on noisy problems Ax 80.0 against SMAC-BB
  72.8, HEBO 61.5, Vizier 61.3, Optuna 47.2, SMAC-HPO 43.2 and random 35.0; in the asynchronous group
  SMAC-BB led with 99.1 (Ax 78.1). On mixed spaces Ax was level with the others until about 20 trials
  and pulled ahead after. Source: [B1] Olson et al., "Ax: A Platform for Adaptive Experimentation",
  AutoML 2025 (PMLR 293), Table 2 and Appendix A.3.

  Implication: at our budget the choice of library moves the result less than the trial count does;
  the tools that model categorical choices with a GP and a discrete acquisition optimizer (Ax) lead
  on our kind of space, and any GP tool needs about 20 trials before it beats a good random design.
  The numbers come from the tool's own authors.

  Experiment: FO10, an honest random control, run only if a third inference pod is granted: 12
  Sobol trials beside Ax's first 12 under the same protocol. Expected movement: Ax's best joules per
  token at least 5 percent below the random control's best at 12 trials each. Minimum sample size:
  12 trials per arm. Confidence: low (0.3).

- **[measured]** The carps benchmark (26 optimizer variants from 9 families, 2025) gives each
  blackbox task ceil(20 + 40 sqrt(d)) trials, 77 to 247 in its subsets; on the blackbox test subset,
  in the paper's three-optimizer comparison, SMAC3-BlackBoxFacade ranked first (mean rank 1.50)
  ahead of Nevergrad CMA-ES (1.82) and random search (2.67), and on multi-fidelity tasks SMAC3's
  multi-fidelity facade led DEHB and SMAC3 Hyperband; the authors find many optimizers "not
  statistically significantly different" per task type. Source: [B7] Benjamins et al., "carps: A Framework for Comparing N Hyperparameter Optimizers
  on M Benchmarks", arXiv:2506.06143 v2 (2025-09-18), Tables 5 and 6.

  Implication: carps's budget for a 10-dimensional task would be 147 trials (derived), four to eight
  times one lane's night; its rankings say little about 15 to 35 trials, where its authors note that
  ranks keep changing until 60 to 80 percent of the budget is spent.

  Experiment: FO10, as specified in the previous finding.

- **[measured]** In the NeurIPS 2020 black-box optimization challenge (16 rounds of 8 suggestions,
  128 evaluations per problem, 65 teams) 61 teams beat random search; the winner (Huawei Noah's Ark,
  HEBO) scored 93.519 against random search's 75.815, which random search would match only after
  about 15,512 evaluations (121 times more), and Optuna's developers placed fifth (91.806, about 60
  times); most top teams used GP-based ensembles. Source: [B10] Turner et al., "Bayesian Optimization
  is Superior to Random Search for Machine Learning Hyperparameter Tuning", arXiv:2104.10201, Table
  2.

  Implication: with a batch of parallel suggestions per round and about a hundred evaluations,
  model-based search is worth one to two orders of magnitude of random evaluations; it supports
  running a model at all on two to four pods, not the choice between GP tools.

  Experiment: FO10, as specified in the first finding of this section.

- **[measured]** HEBO (heteroscedastic, input and output warping, a multi-objective acquisition
  ensemble) placed first in that challenge and outperforms other optimizers on 108 Bayesmark tuning
  tasks per its authors; its library takes batches through `opt.suggest(n_suggestions=4)` and
  `opt.observe`. Source: [B11] Cowen-Rivers et al., "HEBO: Pushing The Limits of Sample-Efficient
  Hyperparameter Optimisation", arXiv:2012.03826 (JAIR); [B12] HEBO README.

  Implication: HEBO is the strongest synchronous-batch alternative, but its interface is
  suggest-a-batch then observe-the-batch, a barrier that idles a pod while the slower trial of a
  batch restarts; Vizier's report (next finding) also found it degrading with batch size.

  Experiment: none beyond FO4; HEBO is not run tonight.

- **[measured]** Google's report on the open-source Vizier GP-bandit algorithm finds it ahead of
  Ax, HEBO, Optuna, HyperOpt and others on high-dimensional categorical problems (20 to 25
  categorical or boolean parameters, 100 trials) and with batches of up to 25, where HEBO's
  performance degrades and HEBO halts with Cholesky errors on spaces with many categories; pending
  suggestions are handled by a constant-liar term on the standard deviation. Source: [B13] Song et
  al., "The Vizier Gaussian Process Bandit Algorithm", arXiv:2408.11527 v3 (2024-12-06), sections
  3.6, 4.3 and 4.4.

  Implication: Vizier and Ax each report themselves ahead on categorical spaces; self-reported
  benchmarks cannot settle the difference. Vizier needs a service and JAX on the driver host, while
  the lab's driver (R45) is written against Ax's Client, so a switch would cost the night.

  Experiment: none beyond FO4.

- **[practitioner]** Optuna's sampler table recommends TPE for 100 to 1,000 trials, CMA-ES for 1,000
  to 10,000, its GP sampler for up to 500 and the BoTorch sampler for 10 to 100; the GP sampler
  starts with 10 random trials and scores running trials with q-batch acquisition functions; the
  Wilcoxon pruner stops a trial once a signed-rank test over shared instance ids (prompts, folds)
  says it is worse than the best trial (default p 0.1, never before the third instance). Source:
  [B14] Optuna docs, samplers reference, comparison table; [B15] `optuna.samplers.GPSampler`; [B16]
  `optuna.pruners.WilcoxonPruner`.

  Implication: Optuna's own guidance puts TPE outside our budget; its GP sampler is a fair
  alternative to Ax but brings nothing Ax lacks here. The Wilcoxon idea fits our per-prompt requests
  and is copied as a rule (stop after the first request when it is far worse), not as a library.

  Experiment: FO4, as specified in section 2.

- **[practitioner]** SMAC3 uses a scikit-learn random forest by default for hyperparameter
  optimization since 2.4.0, supports multi-objective, multi-fidelity and multi-threading in every
  facade, and has an ask-and-tell interface with resumable runs. Source: [B17] SMAC3 README.

  Implication: SMAC3 is the strongest choice at carps's larger budgets and in Ax's asynchronous
  group, but its random-forest facade scored lowest on mixed spaces at 50 trials in [B1], and our
  space is all categorical and ordinal; it stays an alternative for a later, larger run.

  Experiment: none beyond FO4.

- **[practitioner]** Nevergrad's documentation suggests CMA for low-noise problems with a budget of
  about 1,000 times the dimension, TBPSA for noisy problems, a discrete one-plus-one portfolio for
  discrete hyperparameters, and `NgIohTuned` as the default meta-optimizer; with an executor and
  `batch_mode=False` it asks for a new point whenever a worker frees up. Source: [B18] Nevergrad
  docs, "How to perform optimization".

  Implication: Nevergrad's population methods parallelize without a barrier but are sized for
  thousands of evaluations; at 30 to 60 they spend most of the budget exploring.

  Experiment: none beyond FO4.

- **[measured]** OpenTuner runs an ensemble of search techniques and gives more of the test budget
  to the techniques that are doing well, through a bandit with area-under-the-curve credit; its
  authors built autotuners for 7 projects and 16 benchmarks with speed-ups of up to 2.8 times over
  prior techniques. Source: [B19] Ansel et al., "OpenTuner: An Extensible Framework for Program
  Autotuning", PACT 2014, abstract and section 3.2.1.

  Implication: OpenTuner is built for programs that run thousands of cheap tests; it fits the kernel
  lane (section 11), not a lane whose every test restarts a server.

  Experiment: FO8, as specified in section 11.

- **[measured]** Comparing black-box optimizers on the BBOB suite and OpenAI Gym at low budgets,
  Bayesian optimization performed well when budgets were limited, at a higher computational cost, and
  was often outperformed by other families when the budget grew. Source: [B23] Raponi et al.,
  "Optimizing with Low Budgets", arXiv:2310.00077 (IEEE TEVC, revised 2024-01-02).

  Implication: at 15 to 35 evaluations per lane a model-based method is the right family; the
  driver's seconds of model fitting per trial are negligible next to a 612 s trial.

  Experiment: none beyond FO4.

- **[measured]** Scaling a GP's lengthscale prior with the dimension makes standard Bayesian
  optimization perform "drastically better than previously thought in high dimensions" (Ax's
  default surrogate uses such dimension-scaled priors), while SAASBO puts sparse axis-aligned priors
  on the lengthscales and finds the few relevant dimensions with Hamiltonian Monte Carlo. Source:
  [B21] Hvarfner et al., "Vanilla Bayesian Optimization Performs Great in High Dimensions", ICML
  2024, arXiv:2402.02229; [B22] Eriksson and Jankowiak, "High-Dimensional Bayesian Optimization with
  Sparse Axis-Aligned Subspaces", UAI 2021, arXiv:2103.00349.

  Implication: ten settings are not high-dimensional for either model; Ax's "fast" default is
  enough, and the "quality" method's fully Bayesian SAAS model (section 6) is the fallback if the
  first 10 trials show only one or two settings matter and the model fits poorly.

  Experiment: none beyond FO4.

- **[practitioner]** A Plackett-Burman screening design estimates up to 11 two-level main effects in
  12 runs, but its main effects are heavily confounded with two-factor interactions. Source: [B28]
  NIST/SEMATECH e-Handbook of Statistical Methods, section 5.3.3.5.

  Implication: a 12-run screen of R35's settings at two levels would change nearly every
  compile-key setting between runs, so every run is a cold restart (12 x 726 s, about 2.4 hours on
  one pod), and two known interactions (speculative decoding with the packed decode path, hardware
  queues with graph mode) bias it; it is not used as the first stage. Prior arms plus the GP do the
  screening.

  Experiment: none beyond FO4.

## 5. Tuning LLM inference-server settings in the literature

- **[measured]** SCOOT (Ant Group, WWW 2025) tunes vLLM and TensorRT-LLM engine parameters with
  single- and multi-objective Bayesian optimization, a random forest that learns which
  configurations crash, and parallel suggestion; with 30 observations it improved request
  throughput by up to 68.3 percent and cut tail latency by up to 40.6 percent against the default
  configuration; one stress test takes 5 to 10 minutes, so 30 evaluations take 2.5 to 5 hours; two
  parallel suggestions gave the same result in nearly half the time, and more than 4 compromised the
  result unless the observation count grew. Source: [B8] Cheng et al., "SCOOT: SLO-Oriented
  Performance Tuning for LLM Inference Engines", WWW 2025, arXiv:2408.04323, sections 1, 4.6, 5.2
  and 5.3.

  Implication: the closest published setting to ours (engine flags, a restart per test, minutes per
  test) used 30 evaluations and found that two parallel workers halve the wall clock at no loss; it
  supports two to four inference pods tonight and warns against more than four.

  Experiment: FO4, as specified in section 2.

- **[measured]** SLO-Guard (2026-04) tuned eight vLLM 0.19 knobs for `Qwen2-1.5B` on one A100 with a
  15-trial budget over five seeds: it and uniform random search were both 75 of 75 feasible and
  statistically tied on best latency (Mann-Whitney p = 0.84); its gain was consistency (more trials
  in the fast regime, 10.20 against 7.40 of 15); one binary knob (`enforce_eager`) explained over 95
  percent of the latency variance, and the author calls tens of trials a realistic budget. Source:
  [B9] Lysenstoen, "SLO-Guard: Crash-Aware, Budget-Consistent Autotuning for SLO-Constrained LLM
  Serving", arXiv:2604.17627 (2026-04-19), abstract and section 5.5.

  Implication: at 15 trials a vLLM settings space dominated by one or two switches is found as fast
  by random search as by a model; the night's speed comes from trying the likely switches first
  (the prior arms) and from cheap trials, and random search is the honest baseline for any claim.

  Experiment: FO10, as specified in section 4.

## 6. What Ax 1.x switches on today

- **[practitioner]** Ax 1.3.1's `Client` offers `get_next_trials(max_trials, fixed_parameters)`
  (trials come back RUNNING, and `fixed_parameters` sets the given values on every returned
  parameterization), `attach_baseline`, `attach_trial`, `attach_data(trial_index, raw_data,
  progression)`, `complete_trial` (partial data is still used for modeling), `mark_trial_failed`
  (may be suggested again), `mark_trial_abandoned` (never suggested again),
  `mark_trial_early_stopped`, `should_stop_trial_early` (it falls back to a
  `PercentileEarlyStoppingStrategy`) and `run_trials(max_trials, parallelism)`, which passes
  `parallelism` to the orchestrator as its maximum of pending trials. Source: [B2] facebook/Ax
  `ax/api/client.py` at tag 1.3.1.

  Implication: everything the night needs is in the stable Client: asynchronous asks
  (`get_next_trials(1)` whenever a worker frees, other trials still RUNNING), grouping by state
  (`fixed_parameters` holding a worker's warm compile class), warm starts (`attach_baseline`,
  `attach_trial`), partial results from a stopped trial (`complete_trial` with what was measured),
  and abandoning crashed launches.

  Experiment: FO4, as specified in section 2.

- **[practitioner]** In Ax 1.3.1 the initialization budget defaults to 5 (one center point plus
  Sobol points, the center taking one slot), the transition to Bayesian optimization waits for max(1,
  budget // 2) completed trials, existing attached trials count toward it when
  `use_existing_trials_for_initialization` is true, and unless `allow_exceeding_initialization_budget`
  is set a pause criterion stops generation while as many trials as the budget await data; method
  "fast" uses the modular BoTorch model's defaults, "quality" a fully Bayesian SAAS GP with input
  warping, "random_search" Sobol points only. Source: [B3] facebook/Ax
  `ax/api/utils/generation_strategy_dispatch.py` at tag 1.3.1.

  Implication: the night sets the initialization budget to 0 with the center point off and attaches
  the baseline and the prior arms itself, so no worker ever waits on a Sobol pause; "fast" is the
  method, "quality" the fallback named in section 4.

  Experiment: FO4, as specified in section 2.

- **[practitioner]** Ax's paper describes the default as the center of the space, four Sobol trials,
  then a single-task GP with dimension-scaled priors and qLogNEI, with enumeration for small discrete
  spaces and an alternating optimizer for mixed ones; it supports suggesting designs "both
  synchronously and asynchronously", stops trials early with a percentile strategy that generalizes
  the median stopping rule, and lists cost-aware multi-fidelity optimization and transfer learning
  among capabilities "not yet API-stable". Source: [B1] Olson et al., AutoML 2025, sections 1.1,
  5.2, 5.3 and 7.

  Implication: multi-fidelity Bayesian optimization and multi-task transfer are not for tonight; the
  night uses a fixed low fidelity for the screen and full fidelity for confirmation, and warm starts
  by attaching arms.

  Experiment: FO9, as specified in section 9.

- **[practitioner]** Ax's trial-level early-stopping tutorial (1.3.1) reports intermediate values
  with `attach_data(..., progression=t)`, asks `should_stop_trial_early` after each, and marks a
  stopped trial with `mark_trial_early_stopped`; its sample logs stop trials "worse than 50.0-th
  percentile across comparable trials". Source: [B4] Ax docs, "Trial-level Early Stopping", version
  1.3.1.

  Implication: Ax's early stopping needs several completed curves at the same progression before it
  acts, which a night of 15 to 35 trials barely has; the training lane uses it after 5 completed
  trials (progression = step), and both lanes add a simple rule before that (stop after the first
  request or 5 steps when the trial is below half the baseline).

  Experiment: FO5, as specified in section 1.

- **[practitioner]** ax-platform 1.0.0 was released on 2025-05-08 and 1.3.1, the latest, on
  2026-06-09; it needs Python 3.11 or newer. Meta's announcement of Ax 1.0 (2025-11-18) describes a
  GP surrogate with an expected-improvement acquisition by default. Source: [B5] PyPI, ax-platform
  release history; [B6] Engineering at Meta, "Efficient Optimization With Ax".

  Implication: the config names the API as tested in 1.3.1; R45 pins the exact versions in its
  requirements file, and the driver host must run Python 3.11 or newer.

  Experiment: none beyond FO4.

## 7. Parallel and asynchronous evaluation across pods

- **[measured]** Asynchronous successive halving (ASHA) scales linearly with the number of workers
  and was shown on 500 workers. Source: [B20] Li et al., "A System for Massively Parallel
  Hyperparameter Tuning", MLSys 2020, arXiv:1810.05934.

  Implication: removing the barrier between batches is what lets added workers scale; with a fixed
  restart of 295 to 523 s per trial, rung-based early stopping saves at most the measurement share
  of a trial (25 to 36 percent, section 2), so the night takes ASHA's asynchrony and leaves its rungs
  out.

  Experiment: FO4, as specified in section 2.

- **[opinion]** With one Ax experiment per lane and every worker asking for its next trial as soon
  as its last one ends, two inference workers keep both chips busy without a barrier; staggering
  their start by about half a trial keeps one pod decoding while the other restarts, and a worker
  whose server is up runs every restart-free measurement on it (batch 8 and extra requests for a
  finalist) before it stops the server. Source: [B2] `ax/api/client.py` at 1.3.1,
  `get_next_trials`; [B8] SCOOT, section 5.3.

  Implication: the config carries a `workers` list, `asynchronous: true`, a 300 s stagger and two
  queues per worker: restart-needed trials from Ax and restart-free measurements tied to the running
  server. Every second trial on a worker is asked with `fixed_parameters` holding that worker's
  warm compile class, so each class is restarted cold once and reused warm once (section 2's mixed
  row and section 3).

  Experiment: FO4, as specified in section 2.

## 8. Pod C: a second teacher, training beside the teacher, or neither

- **[practitioner]** vLLM's `gpu_memory_utilization` (default 0.9) is "a per-instance limit, and
  only applies to the current vLLM instance"; two instances on one GPU can each be set to 0.5, and
  `kv_cache_memory_bytes` overrides it with an explicit KV size. Source: [B29] vllm
  `vllm/config/cache.py` at b1388b1, `gpu_memory_utilization` and `kv_cache_memory_bytes`.

  Implication: the live teacher on pod C, launched by `lab/pod_a_teacher.sh` with `GPU_UTIL`
  defaulting to 0.90 [B54], holds 90 percent of the memory the GPU sees whatever else runs; if the GPU
  sees all 64 GB that is 57.6 GB (19 GB of weights, the rest KV and activations), leaving at most
  6.4 GB for everything else on the APU, including the CPU side.

  Experiment: FO6, a read-only memory reading on pod C with the teacher up: `torch.cuda.mem_get_info()`
  total and free, `mem_info_vram_total` and `mem_info_gtt_total`. Expected movement: free at most 6.4
  GiB if the total is 64 GiB, less if the GTT limit caps the GPU at about half the RAM. Minimum
  sample size: 1 reading. Confidence: medium (0.6).

- **[measured]** The pods' GPU is a Radeon 8060S with 64 GB shared with the CPU; power is read as
  `amd-smi` `SOCKET_POWER`, the whole APU package; over 81 minutes beside the running teacher the
  chip was in deep idle 62 percent of the time and busy (about 110 W) 31 percent. Source: [B51]
  `lab/README.md`, "Teacher on SJC today" and "Idle baseline on the Strix Halo".

  Implication: a trial on pod C cannot measure joules per token, because the package power includes
  the live teacher's 100 W or so whenever it answers, and about a third of the time it does; its
  tok/s would also lose to the teacher's kernels on the same 40 CUs. Inference trials stay on pod D.

  Experiment: FO6, as specified in the previous finding.

- **[practitioner]** AMD's guidance cited by the R32 hardware note says the GTT pool used by PyTorch
  and other compute work is dynamic, "by default ... approximately 50 percent of total system RAM",
  and that when GTT is larger than VRAM the driver makes VRAM allocations from GTT. Source: [B58] PR
  #237, `lab/refine/research/kernels/strix_halo_hardware.md`, section 5.

  Implication: on a default host the GPU may see about 32 GB, of which the teacher holds 28.8 GB at
  0.90, and training beside it is impossible; on a host with a raised limit (the community sets
  about 115 GB on 128 GB machines [B34]) the 6.4 GB figure above holds. Only FO6 tells which.

  Experiment: FO6, as specified in the first finding of this section.

- **[opinion]** A second stock teacher on pod C is not worth it: at 0.90 there is no room; with
  both instances cut to about 0.35 (22.4 GB each, 19 GB of weights plus a small KV) plus the 9 GB
  LoRA job, the APU would hold about 54 GB plus the CPU side's 6 GB or so, near the 64 GB edge, the
  live teacher would need a restart (297 s of outage), its decode would share the CUs, and the second
  instance's joules per token could not be measured. Pod C takes the training lane only, and only if
  FO6 shows at least 12 GiB free (8.7 GiB measured peak plus 3 GiB) or Ilan approves restarting the
  teacher at `GPU_UTIL` 0.70 (44.8 GB) for the night; otherwise R39 runs on pod D after R38. Source:
  [B29] `vllm/config/cache.py` at b1388b1; [B51] `lab/README.md`; [B53]
  `lab/refine/metrics.jsonl:14`.

  Implication: both objectives run at the same time only when FO6 passes or Ilan gives that go; the
  config marks pod C's training role as conditional and gives pod C the restart-free attach
  measurements of the baseline (FO7) either way.

  Experiment: FO6, as specified in the first finding of this section.

## 9. Public Strix Halo numbers as priors

Every source below ran Linux; none of the opened sources ran Windows (AMD's own Ryzen AI Max+ 395
blog post, likely Windows and LM Studio, did not load and is listed as not opened). ROCm versions
differ from the pods' 7.13 in every case and are stated per finding. R51 is collecting these and
more into `lab/refine/research/kernels/data/public-strix-halo-benchmarks.csv` with a first prior
model (pending; not in the repository yet).

- **[measured]** On a Framework Desktop (Ryzen AI Max+ 395, Fedora 43, kernel 6.15 rc7, HIP 6.4 to
  6.5, May to August 2025): `rocm_bandwidth_test` measured 212.4 GB/s within the GPU (84 GB/s CPU to
  GPU); llama.cpp's HIP and Vulkan backends reached 70.8 to 73.3 percent of the 256 GB/s theoretical
  peak in decode; a 1024 x 1024 x 1024 half-precision GEMM ran at 6.1 TFLOPS through rocBLAS and 19.7
  TFLOPS with `ROCBLAS_USE_HIPBLASLT=1`; Llama-2-7B Q4_0 decoded at 52.2 tok/s on Vulkan and 48.7 on
  HIP (tg128). Source: [B33] llm-tracker, "Strix Halo" (2025-08-08 page, updates from 2025-05).

  Implication: a well-tuned 4-bit decode on this chip moves about 180 to 190 GB/s, while our teacher
  moves about 44.6 GB/s (2.88 tok/s x about 15.5 GB per token, R32's figure [B58]); the gap is the
  linear kernel, which makes `linear_kernel=exllama_fp16` the first prior arm. For training, the
  hipBLASLt factor of about 3 on GEMM makes `blas_backend=hipblaslt` a prior arm, with R36's caveat
  that torch 2.9.1 may override it back on gfx1151 [B57].

  Experiment: FO9, the prior calibration: the first five inference trials (baseline and four prior
  arms) compared with R51's predicted decode for each. Expected movement: our build at 20 to 30
  percent of the public model's bandwidth efficiency on the Conch path and 50 percent or more on the
  exllama path. Minimum sample size: 5 trials. Confidence: medium (0.5).

- **[measured]** A community spike (Skulk issue 144, May 2026; ROCm 6.4.4, Fedora 43 recommended)
  lists 256 GB/s theoretical and about 215 GB/s measured, about 116 GB GTT usable on 128 GB with
  `amdgpu.gttsize=117760 ttm.pages_limit=30408704` and the BIOS frame buffer at 512 MB, Vulkan
  ahead of HIP at short context (881 against 765 tok/s prompt, 52.8 against 45.0 decode) and Qwen3 8B
  BF16 prompt processing at 325 tok/s on ROCm 7.0.1 against 1,132 on ROCm 6.4.4. Source: [B34]
  Foxlight-Foundation/Skulk#144.

  Implication: the chip's measured bandwidth is 212 to 215 GB/s in two independent reports, the
  number R33's bandwidth row is checked against; ROCm versions move results by factors, so a public
  number is a prior on the level and needs our offset.

  Experiment: FO9, as specified in the previous finding.

- **[anecdote]** The same issue reports vLLM working on gfx1151 only with TheRock nightly and 12
  local patches, with a 4-minute cold start, and 4.3 tok/s against llama.cpp's 7.5 on a "27B BF16"
  model; a 27B BF16 model reads about 54 GB per token, which at 7.5 tok/s would need about 405 GB/s,
  above the chip's peak, so the model or its format is misstated. Source: [B34]
  Foxlight-Foundation/Skulk#144, "vLLM and Other Framework Notes".

  Implication: none for the metric set beyond a cold start of about 4 minutes elsewhere, near our 5
  to 9 minutes; the decode figures are not usable as priors.

  Experiment: none beyond FO9.

- **[measured]** On a GMKtec EVO-X3 (Ryzen AI Max+ 395, 128 GB, Ubuntu 24.04.4, kernel 6.17, ROCm
  7.14, 2026-08-11), vLLM 0.26 and 0.27 served `Qwen3.5-0.8B` at 86.3 tok/s for one client and 309 to
  317 tok/s aggregate for eight, and `Qwen3.5-122B-A10B` GPTQ-Int4 at 10.28 tok/s for one client and
  16.1 for two; a multi-token-prediction build raised a Qwen3.6-35B-A3B decode from 23.44 to
  27.38 tok/s (16.8 percent) but the 4,096-token output diverged from the deterministic baseline at
  byte 181, and a DFlash speculative build ran 3.64 times faster with 6 of 10 output hashes
  differing. Source: [B35] Soothill, "Strix Halo LLM upgrade: wins and regressions", 2026-08-11.

  Implication: vLLM on gfx1151 reaches about 138 GB/s on a small BF16 model (derived: 86.3 tok/s x
  1.6 GB), so vLLM itself is not the limit; speculative decoding speeds this chip up but changes
  outputs, so the MTP prior arm (`num_speculative_tokens=1`) must pass the R21 correctness gate
  before its speed counts.

  Experiment: FO4, as specified in section 2.

- **[measured]** A Japanese write-up (2026-06-17, ROCm 7.2.4, llama.cpp b9672) measured Qwen3.6-35B-A3B
  Q4_K_M at 52.7 tok/s decode on HIP and 63.1 on Vulkan (1,083.8 and 1,018.4 tok/s prompt), and
  relays a community report of `Qwen3.6-27B` AWQ-INT4 at 24.8 tok/s on vLLM 0.20.0 with ROCm 7.13
  nightly and DFlash speculative decoding, which the author did not reproduce. Source: [B36]
  zenn.dev/shuzan, "Local LLM Performance Improved on Ryzen AI Max+ 395".

  Implication: the relayed 24.8 tok/s for a 27B 4-bit model with speculative decoding is about 8.6
  times our 2.88 and is second-hand; it marks the top of the range, not a target.

  Experiment: FO9, as specified in the first finding of this section.

- **[measured]** Prior-guided Bayesian optimization: piBO multiplies the acquisition by a
  user-given distribution over the optimum's location, keeps convergence for expected improvement
  whatever the prior, and reported a 12.5 times time-to-accuracy speed-up on a deep-learning task
  (ICLR 2022); ColaBO (ICLR 2024) puts priors on the surrogate, including the optimum's location and
  value, accelerates with accurate priors and keeps about default performance with misleading ones;
  DynaBO (2025-11, revised 2026-07) accepts priors at any time with decaying weight and a safeguard
  that rejects misleading ones. Source: [B24] Hvarfner et al., arXiv:2204.11051; [B25] Hvarfner,
  Hutter and Nardi, arXiv:2311.14645; [B26] Fehring et al., arXiv:2511.02570.

  Implication: a prior from public data speeds the search only where it is informative and costs
  little where it is wrong; Ax's stable Client has no prior-mean or piBO switch, so tonight the
  public data enters as attached prior arms, and a prior-mean GP (R51's model plus a learned offset)
  waits for R45 to support a custom mean on the fake harness.

  Experiment: FO9, as specified in the first finding of this section.

- **[opinion]** A 2023 survey of transfer learning for Bayesian optimization sorts the methods into
  initial-point design, search-space design, surrogate models and acquisition functions, motivated
  by slow convergence when few trials exist. Source: [B27] Bai et al., "Transfer Learning for
  Bayesian Optimization: A Survey", arXiv:2302.05927.

  Implication: the public data can only transfer through the settings the sources vary. They vary
  the backend and kernel path, speculative decoding, batch, context and hipBLASLt; none of the opened
  sources varies the seven vLLM-only flags in R35's space (skinny GEMM, packed recurrent decode,
  attention backend, prefix caching, buffer ops, graph mode, Mamba cache dtype) on gfx1151. The
  transfer therefore replaces the trials that find the level, the ceiling and the two big switches
  (the 5 prior arms also calibrate the offset), not the 12 or so trials for those seven flags: about
  17 own trials per lane, 9 rounds on two inference pods (about 1.5 hours) or the full 3 hours on one.

  Experiment: FO9, as specified in the first finding of this section.

## 10. Expected ranges for the chip sheet (R33) and the decode profile (R47)

- **[opinion]** From the public numbers in section 9, R33's sustained GPU read bandwidth should land
  at 200 to 220 GB/s (212.4 and about 215 measured elsewhere, 256 theoretical); batch-1 decode of a
  model that reads about 15.5 GB per token has a ceiling of about 13.7 tok/s (212 / 15.5) and a
  llama.cpp-class kernel would reach about 11.7 to 12.1 (181 to 188 GB/s); a half-precision GEMM
  should run at about 6 TFLOPS on rocBLAS and about 20 with hipBLASLt at 1024 cubed. Source: [B33]
  llm-tracker, "Strix Halo"; [B34] Foxlight-Foundation/Skulk#144; [B58] PR #237, section 5.

  Implication: R47's decode profile should place our 2.88 tok/s at about 21 percent of the measured
  bandwidth; any batch-1 decode above 13.7 tok/s without speculative decoding is a measurement error,
  and a bandwidth reading under 150 GB/s points at the pod's memory or power state before the
  kernel.

  Experiment: FO9, as specified in section 9.

## 11. Kernel lane: measure cheaply, model, search big

- **[practitioner]** Triton's autotuner takes `prune_configs_by` with a `perf_model` that predicts
  each config's running time, a `top_k` of configs to benchmark and an `early_config_prune`
  function; it benchmarks the survivors with `do_bench` (median of runs; the deprecated defaults were
  25 ms warm-up and 100 ms repetition), records configs that fail to compile or run out of resources
  as infinitely slow, and can cache timings to disk. Source: [B37] triton-lang/triton
  `python/triton/runtime/autotuner.py` (main), `autotune` docstring and `Autotuner._bench`.

  Implication: the learned model plugs into Triton's own hook: it scores the whole space, the
  autotuner measures only the top k, and the winner is cached per shape, all on the pods' Triton
  3.5.1 with no new framework.

  Experiment: FO8, the cost-model pilot: on pod D, 500 random configs of the W4A16 decode Triton
  kernel at the teacher's GEMV shapes (batch 1), compiled ahead on the pod's CPU, timed with
  `do_bench`; fit gradient-boosted trees on config and compile features, score the full space,
  measure the top 50 and 100 held-out configs. Expected movement: held-out Spearman of at least 0.6
  and a top-50 config within 2 percent of the best measured, and a best config at least 10 percent
  faster than the shipped default. Minimum sample size: 650 measured configs per shape. Confidence:
  low (0.3).

- **[practitioner]** TVM's auto-scheduler tutorials state that about 1,000 measurement trials let the
  search converge for one operator and recommend about 900 times the number of tasks for a network,
  with at least 300 ms per measurement repeat. Source: [B38] apache/tvm
  `gallery/how_to/tune_with_autoscheduler/tune_conv2d_layer_cuda.py` at v0.15.0; [B39] apache/tvm
  `gallery/how_to/tune_with_autoscheduler/tune_network_cuda.py` at v0.15.0.

  Implication: a learned cost model trained from scratch needs on the order of 1,000 measurements per
  kernel shape before its top-k is reliable; at about 0.3 s each that is 5 minutes of GPU time per
  shape (derived), so one 15-minute slot holds about two shapes, if the compiling is done before the
  slot.

  Experiment: FO8, as specified in the previous finding.

- **[measured]** Ansor samples programs from a hierarchical space and refines them by evolutionary
  search guided by a learned cost model, reaching up to 3.8 times (Intel CPU), 2.6 times (ARM CPU)
  and 1.7 times (NVIDIA GPU) over the state of the art; a cost model pre-trained on TenSet's 52
  million program records from 6 platforms cut the search time of such a compiler by up to 10 times.
  Source: [B40] Zheng et al., "Ansor", OSDI 2020, arXiv:2006.06762; [B41] Zheng et al., "TenSet",
  NeurIPS 2021 Datasets and Benchmarks.

  Implication: transfer from earlier measurements cuts the measurements needed by up to 10 times, so
  once FO8 has measured one shape, the next shapes of the same kernel start from that model and need
  about 100 measurements each, not 1,000.

  Experiment: FO8, as specified in the first finding of this section.

- **[practitioner]** AMD's Triton guide (ROCm 6.1, written for the MI300X) computes occupancy from
  the compiled kernel: the VGPR count from the ISA, the LDS bytes from `triton_gpu.shared` (LDS-limited
  occupancy = floor(65536 / LDS)), and the waves per workgroup from `num-warps`; `waves_per_eu` hints
  the compiler to cut VGPRs to reach an occupancy, and `num_stages` is 0 for a single GEMM on those
  parts. Source: [B42] ROCm 6.1.0 docs, "Optimizing Triton kernels".

  Implication: the model's features are the config (block sizes, split-K, `num_warps`,
  `num_stages`, `waves_per_eu`), the compile metadata (VGPRs, spills, LDS bytes) and the occupancy
  they imply on gfx1151 by R32's numbers (1,536 VGPRs per SIMD in wave32 with a granule of 24, 16
  waves at most, so 96 VGPRs for full occupancy; 64 KiB of LDS per workgroup and 128 KiB per WGP; 20
  WGPs [B58]), plus the grid size against 20 WGPs and the bytes moved per call; the MFMA-specific
  knobs do not apply to RDNA's WMMA.

  Experiment: FO8, as specified in the first finding of this section.

- **[measured]** tritonBLAS (2025-12) picks Triton GEMM parameters from an analytical model of the
  architecture and reaches over 95 percent of autotuned performance with no autotuning time;
  WaveTune (2026-04) models latency with a wave-aware bilinear model, cutting the decision overhead by
  five orders of magnitude against exhaustive search, with kernel speed-ups of up to 1.83 times;
  Ringlein et al. (2025) find autotuning explores up to 15 times more configurations and beats
  vendor-optimized kernels by up to 230 percent. Source: [B43] Swann et al., arXiv:2512.04226; [B44]
  Zhang et al., arXiv:2604.10187; [B45] Ringlein et al., arXiv:2505.03780.

  Implication: an analytical model (bytes, waves, occupancy) is a free prior for the learned one,
  and the space should be wide; for training's half-precision GEMMs tritonBLAS is a candidate
  seed once it is checked on gfx1151.

  Experiment: FO8, as specified in the first finding of this section.

- **[measured]** LLM-driven kernel search on AMD: GEAK (2025-07) generated Triton kernels for MI300X
  and MI250 with up to 63 percent correctness and up to 2.59 times speed-up; KernelFoundry (ICML 2026)
  evolved kernels with an average 2.3 times speed-up on KernelBench SYCL kernels; AMDKernelVault
  (2026-09) built 62,153 execution-verified HIP kernels and 39,893 Triton kernels for CDNA GPUs and
  trained Qwen3-8B to 33 to 42 percent correctness. Source: [B46] Wang et al., arXiv:2507.23194;
  [B47] Wiedemann et al., arXiv:2603.12440; [B48] Liu et al., arXiv:2609.12471.

  Implication: generated kernels are correct a third to two thirds of the time and target CDNA; they
  are a later source of new kernel variants for the model to rank, not something to run unattended
  tonight on RDNA 3.5.

  Experiment: none beyond FO8.

- **[opinion]** The kernel-lane recipe for tonight: compile each candidate config ahead of the GPU
  slot on the pod's CPU, measure 500 random configs per shape at milliseconds each, fit the model,
  score the whole space (enumerate it below about a million configs, evolutionary search above), and
  measure the top 50 plus 100 held-out configs; stop the lane for a kernel when the best of the first
  500 is under 5 percent faster than the shipped default (a flat space), when the held-out Spearman is
  under 0.6 (then measure exhaustively if the space fits the slot), or when the best config fails
  the R21 per-shape correctness test. Source: [B37] Triton autotuner; [B38] TVM tutorial; [B42] ROCm
  Triton guide; [B58] PR #237.

  Implication: the kernel lane runs at milliseconds a measurement and is bounded by compiling, so it
  needs no Bayesian optimizer; the config file's `kernel` lane carries these numbers.

  Experiment: FO8, as specified in the first finding of this section.

## Decisions (and why the alternatives lost)

- Tool: Ax 1.x through its stable `Client`, method "fast" (dimension-scaled GP, qLogNEI), because it
  scored highest on mixed and noisy problems at 50 trials in the broadest published comparison [B1],
  has asynchronous asks, `fixed_parameters`, baseline and trial attachment in its stable API [B2],
  and is what the lab's driver (R45) is written against. Vizier reports itself ahead on categorical
  spaces [B13] but needs a service; SMAC3's strength is at larger budgets [B7]; HEBO's interface is a
  synchronous batch [B12]; Optuna recommends TPE above 100 trials [B14]; Nevergrad and CMA-ES are
  sized for thousands of evaluations [B18]; OpenTuner belongs to the kernel lane [B19].
- Strategy: no Sobol points; the baseline and four prior arms first, ordered to reuse the compile
  cache (baseline cold, `gpu_max_hw_queues=1` warm, then `exllama_fp16`, `exllama_fp16` with MTP 1,
  MTP 1); then one Ax trial per free worker, every second one held to the worker's warm compile
  class; trials grouped by card state (section 3), so each spawn runs every restart-free
  measurement of its configuration back to back; repeats are three requests inside one spawn; a
  trial stops after its first request if its decode is under half the baseline's and is completed
  with what it measured; a launch that is not ready within 1.5 times the expected restart is marked
  failed.
- Expected evaluations to beat the baseline: R35's own odds for its three big switches (0.3 for the
  linear kernel, 0.4 for MTP, 0.25 for hardware queues) give about 0.69 that one of the first five
  trials beats the baseline by about the margin R35 predicts (derived: 1 - 0.7 x 0.6 x 0.75);
  otherwise about 12 more Ax trials. A win counts after 3 interleaved spawns of the best arm and 3 of the baseline on the same
  pod (one-sided 95 percent, power 0.8 for a gain of about two spawn standard deviations); the lab's
  `verdict()` wants 5 per arm [B55], so tonight's verdict is low-confidence and the 5-spawn verdict
  is queued on T0.
- How the trial count falls with each technique: Ax's default (center plus 4 Sobol points, then the
  GP) is level with random search until about 20 trials on mixed spaces [B1]; replacing the Sobol
  points with the prior arms puts the likely win at trial 2 to 5 when a prior holds (about 0.69);
  the public data adds no further cut for the seven flags it does not vary (section 9), so about 17
  own trials per lane remain; early stopping, the 64-token screen, the warm compile cache and
  grouping by card state do not lower the count but make each trial cheaper (sections 2 and 3);
  parallel workers keep the count and divide the wall clock, up to four [B8].
- Expected wall clock to a confirmed inference win (612 s screen trials, 676 s confirmation
  spawns): one inference pod 2.0 hours if a prior hits and 4.0 if not (expected 2.6); two pods 1.1
  and 2.1 (expected 1.4); four pods 0.7 and 1.2 (expected 0.9).
- Pods: pod D takes inference (R38) and the kernel lane; pod C takes training (R39) only if FO6
  passes or Ilan approves the teacher restart at `GPU_UTIL` 0.70, and the restart-free attach
  measurements of the baseline (FO7) either way; an extra pod AMD grants joins the inference lane
  with the driver moved to this machine (hub runner) so one Ax experiment serves all inference
  workers.

## Approach (the night configuration file)

`lab/refine/research/kernels/data/ax-config-night.json` is read by the R45 driver
(`lab/optimize_settings.py`). The driver reads the keys below and ignores the ones it does not know;
`_comment` keys are notes for people.

| Key | Type | Meaning |
| --- | --- | --- |
| `schema` | string | `lab.optimize.night/1` |
| `decision` | string | the one-line decision at the top of this note |
| `tool` | object | `name` (`ax-platform`), `api_tested` (`1.3.1`), `api` (`ax.Client`), `mode` (`ask-tell`), `python_min` |
| `asynchronous` | bool | true: ask one trial whenever a worker frees; never wait for a batch |
| `window` | object | `start` and `result_by`, ISO 8601 with offset (23:00 to 08:00 ET) |
| `generation_strategy` | object | the arguments of `Client.configure_generation_strategy`, by their Ax names |
| `workers` | list of objects | `name`, `account`, `hub`, `gpu_queue` (the GPU Queue's GPU value), `lanes` (list of lane names), `exclusive` (bool), `enabled` (true, `"if_precondition"` or `"if_granted"`), `precondition` (string or null), `runner` (`"local"` on the pod, `"hub"` through the controller), `stagger_s` (int) |
| `driver_host` | object | where each lane's Ax experiment runs: `one_inference_worker` and `several_inference_workers` |
| `group_by_state` | object | the card-state rule: `principle`, then per lane the `state` that defines a block, `block` (what runs back to back in it), `order` (how blocks are ordered), and `card` (one lane at a time per card, in `order`) |
| `lanes.<lane>.task` | string | the Task Board task the lane serves |
| `lanes.<lane>.search_space` | string | repo path of the search-space file; `search_space_status` says where it is pending |
| `lanes.<lane>.objective` | object | `metric` (dotted name as R45 names metrics) and `goal` (`minimize` or `maximize`) |
| `lanes.<lane>.tracking` | list of strings | metrics recorded beside the objective on every trial |
| `lanes.<lane>.finalist_tracking` | list of strings | metrics measured on the running server only for the current top 3 |
| `lanes.<lane>.restart` | object | restart seconds (`cold_s`, `warm_s`, `process_s`), the parameter classes (`restart_free`, `warm_restart`, `cold_restart`, `cold_restart_unverified`, `in_process`, `process_restart`) and `ready_timeout_factor` |
| `lanes.<lane>.grouping` | object | `warm_class_every` (int: every n-th trial on a worker is asked with `fixed_parameters` = that worker's warm class) and `class_keys` (the parameters that define a class) |
| `lanes.<lane>.fidelity` | object | `screen` and `confirm`, each with the harness arguments (`bench_args` for inference; `steps`, `exclude_first_steps` for training) and the estimated trial seconds |
| `lanes.<lane>.repeat_policy` | object | requests or steps per trial, spawns or processes per trial, and how the SEM handed to Ax is computed |
| `lanes.<lane>.early_stop` | object | the first-request or first-steps rule, Ax's percentile strategy switch, `max_consecutive_failures` |
| `lanes.<lane>.initial_arms` | list of objects | in run order: `name`, `parameters` (changes from the baseline; `{}` is the baseline), `attach_as` (`baseline` or `trial`), `restart` (`cold` or `warm`), `priors` (ids into `priors`) |
| `lanes.<lane>.budget` | object | `minutes`, `slot_minutes`, `slots`, `trials_per_worker_est`, `restarts_est`, `reserve_for_confirmation_minutes` |
| `lanes.<lane>.confirmation` | object | `spawns_per_arm` (or `runs_per_arm`), `order`, `test` |
| `lanes.kernel` | object | the kernel lane: `targets`, `measure`, `initial_random_configs`, `model`, `features`, `search`, `top_k_measure`, `held_out`, `kill`, `slot_minutes`, `reuse` |
| `priors` | list of objects | settings to try first: `id`, `lane`, `parameters`, `expected`, `p` (number or null), `sources` (list of strings with dates), `evidence_os`, `evidence_rocm` |
| `expected_ranges` | object | the ranges R33 and R47 are checked against, each with `low`, `high`, `unit` and `sources` |
| `transfer` | object | how the public prior model (R51, pending) enters: `now`, `later`, `offset_calibration_trials`, `file` |
| `schedule` | list of objects | `start`, `end` (ET clock times), `worker`, `lane`, `what` |

## Candidate experiments, ranked

Ranked by expected gain in joules per token times confidence; FO4 and FO5 are the night runs the
others feed.

1. FO4, the R38 night run on pod D (joules per token at batch 1 down 20 percent or more with about
   0.69 odds within 5 trials; confidence low to medium).
2. FO1, the restart diet (every later trial up to 30 percent shorter; confidence low).
3. FO2, the compile-cache classes (decides the warm and cold classes the scheduler groups by;
   confidence medium).
4. FO6, pod C's free memory (decides whether both objectives run at once; confidence medium).
5. FO3, the noise split (sets the confirmation spawns; confidence medium).
6. FO5, the R39 night run (samples/s up 20 percent or more if a training prior holds; confidence
   low).
7. FO8, the kernel cost-model pilot (a decode kernel at least 10 percent faster; confidence low).
8. FO9, the prior calibration against R51's model (no movement itself; sets the offset).
9. FO7, the live baseline's concurrency curve on pod C (labeling-case joules per token).
10. FO10, the random control (only with a third inference pod).

## Open questions

- Pod C's GPU-visible memory and free memory with the teacher up (FO6, R32's SH8); the training
  lane's placement waits on it or on Ilan's go to restart the teacher at `GPU_UTIL` 0.70.
- Whether `num_speculative_tokens`, `mamba_ssm_cache_dtype`, `attention_backend` and
  `enable_prefix_caching` change the torch.compile key (the speculative and cache configs were not
  opened); FO2 answers it on the pod.
- Three search-space formats exist (R35's `atlas.searchspace.v1`, R36's Ax-config style, R45's
  `lab.optimize.searchspace/1` sample); R45 must read or convert both real files before R38 and R39
  start.
- The harness change for a one-request warm-up (L3 in section 2) is not made; it is worth about 90 s
  a trial.
- A trainer that changes in-process settings without reloading the model (section 3) does not exist
  yet; without it the training lane loads one process per trial.
- R51's public prior model (`lab/refine/research/kernels/data/public-strix-halo-benchmarks.csv`) is
  pending; FO9 compares the first five trials with it when it lands.
- Whether Ax 1.3.1's `ModelConfig` can pass a custom GP mean (the prior mean from R51) was not
  verified.
- AMD's Ryzen AI Max+ 395 blog post (probably Windows and LM Studio) timed out and was not opened;
  the Framework community threads and the Strix Halo setup guides were seen in search results only.

## Search log

- 2026-10-10, budget about 3 hours (kernels track), used about 2 hours 30 minutes. Started from the
  task page, R35's and R36's search-space files (PRs 235 and 238), R32's hardware note (PR 237),
  `lab/bench.py`, `lab/README.md` and the R30 and R18 GPU Queue rows. Queries: carps and the Ax
  paper's benchmark, SCOOT and LLM-serving autotuners (found SLO-Guard 2026 and FleetSieve 2026, the
  latter not opened), Ax 1.x early stopping and the Client API (read in the 1.3.1 source), Optuna
  pruners and samplers, SMAC3, HEBO, Vizier, Nevergrad, OpenTuner, the BBO 2020 analysis, low-budget
  comparisons, prior-guided and transfer Bayesian optimization, vLLM 0.19 restart and memory
  settings at b1388b1, Strix Halo llama.cpp and vLLM measurements (four community sources opened),
  learned kernel cost models and LLM-driven kernel search. PDFs that WebFetch could not read (the Ax
  paper, SCOOT, SLO-Guard, the BBO analysis, Vizier, OpenTuner) were saved and read with pdftotext.
  Stopped at the budget, with the last three Strix Halo sources repeating the bandwidth and backend
  figures already recorded.
