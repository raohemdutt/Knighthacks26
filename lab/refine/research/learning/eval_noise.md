# Eval noise: how many questions and runs before a before-versus-after difference is real

Research note for the learning track, opened 2026-10-10 (all sources opened that day; index in `SOURCES.md`).
It answers one question for the eval gate: when the LoRA student beats the current model on a small held-out set,
is the difference real or noise? Labels: `measured` = a number someone measured, with the setup; `practitioner` = a
documented recommendation from people who run evaluations; `anecdote` = a single report without data; `opinion` =
argument without evidence. Every finding carries its citation and a `[S..]` or `[R..]` key into `SOURCES.md`.
Under each finding, "Implies" maps it onto our metric set and "Experiment" proposes a test with the expected
movement and the minimum sample. Our metric set is the learning rubric's [R7] (`lab/refine/learning/RUBRIC.md`, "The metric set"):
held-out score before, after and teacher (`learning.heldout.current`, `learning.heldout.candidate`,
`learning.heldout.teacher`); label agreement rate (`learning.label_agreement`); loop wall-clock
(`learning.loop_minutes`); training samples per second (`trainer.samples_per_s`, 0.51 on pod B, PR #37);
package size (`learning.package_mb`).

Numbers marked "computed" are ours, not the sources': exact two-sided McNemar tests (binomial test on the
discordant pairs) and Miller's Eq. 9 and 10, worked in the appendix so anyone can re-run them.

## 1. What our eval can resolve today

### Takeaway
The agent's golden eval has 6 pass/fail questions and reports the fraction passed. With 6 paired questions an
exact McNemar test cannot call any improvement real unless every question flips from fail to pass and none flips
back (6 of 6 gives p = 0.031; 5 of 6 gives p = 0.063), so the golden set can show a demo beat but cannot gate a
deploy. A per-task held-out set of about 30 items detects a 30-point net gain about 71% of the time and a 20-point
gain about 40% of the time; 50 items raise those to 94% and 70% (computed, appendix).

### Cited findings
- **[measured]** Our golden eval holds 6 questions (5 that must cite evidence and 1 `no_evidence` question), each scored pass or fail, and the reported score is passes divided by questions. Source: Atlas repo at commit bf482f4, `fixtures/golden/questions.yaml` and `contexts/agent/eval.py` (`score()`), https://github.com/raohemdutt/Knighthacks26/blob/bf482f4/contexts/agent/eval.py [R1]

  Implies: the held-out score (before, after, teacher) moves in steps of 1/6 = 0.17 on this set, and its 95% Wilson interval at 3/6 is [0.19, 0.81] (computed), wider than any gain one LoRA run can be expected to deliver. The golden set stays a smoke check and a demo beat; the gate needs the per-task held-out sets.

  Experiment (EN-1): score base and candidate on the golden 6 and on a 30-item held-out split of the same task, and run the exact McNemar test on both. Expected: no golden-set comparison reaches p < 0.05 unless the base fails all 6; the 30-item set reaches p < 0.05 when the net gain is about 0.3 or more. Minimum sample: one paired run per model on each set.

### Gaps
No source gives a size for a tool-calling or desk-object held-out set specifically; the sizes here come from the
general power formulas in section 2.

## 2. Which test, and how many items

### Takeaway
Compare the two models item by item, not as two independent accuracies. For pass/fail items that is McNemar's
test (exact, because our n is small) or a paired Bayesian interval; the unpaired two-proportion test is the one
Dietterich says never to use. Size the held-out set before training with Miller's power formula, and report every
score with a small-sample interval (Wilson or Bayesian), not a CLT interval.

### Cited findings
- **[practitioner]** Dietterich compared five tests for two classifiers and found that "for algorithms that can be executed only once, McNemar's test is the only test with acceptable type I error", while "a test for the difference of two proportions" should never be used. Source: T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", 1998, Neural Computation 10(7):1895-1923, doi:10.1162/089976698300017197, https://mlanthology.org/neco/1998/dietterich1998neco-approximate/ (abstract) [S03]

  Implies: the gate's deploy rule for the held-out score should be "candidate beats current on the paired items", tested with an exact McNemar test on b (items that flip fail to pass) and c (pass to fail). At alpha 0.05 two-sided that needs b >= 6 with c = 0, b >= 8 with c = 1, b >= 10 with c = 2 and b >= 12 with c = 3 (computed). Any verdict math that compares two arms on the same items should use this paired form rather than a two-proportion test.

  Experiment (EN-2): replay a null comparison: score the current model twice on the same 30 items at production decode settings, 20 times over, and apply both tests to each pair of runs. Expected: both tests declare a difference in at most 1 of 20 replays (the unpaired test, if anything, less often, because it ignores the positive correlation between runs of one model); on the first real candidate the McNemar p-value is lower than the two-proportion p-value on the same data, which is the power the paired design buys. Minimum sample: 20 replays x 30 items, then one real paired run.

- **[practitioner]** Miller recommends inference "on the question-level paired differences" when two models answer the same questions, gives the paired standard error SE(A-B) = sqrt(SE_A^2 + SE_B^2 - 2 SE_A SE_B Corr(s_A, s_B)) (Eq. 7), and recommends "power analysis to determine whether an eval (or a random subsample) is capable of testing a hypothesis of interest". Source: E. Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations", 2024, arXiv preprint, https://arxiv.org/abs/2411.00640, sections 1 and 4.2 [S01]

  Implies: because base and candidate share most of their answers, the correlation term makes the paired comparison tighter than two separate error bars suggest; the slide's before and after held-out scores should come with the paired difference and its interval, not two overlapping intervals.

  Experiment (EN-3): on the first real run, compute the unpaired and the paired standard error of after minus before on the same items. Expected: the paired SE is smaller than the unpaired one by about 17% when 60% of items agree and the gain is 0.30, 26% at 70% agreement and a 0.20 gain, and 38% at 80% agreement and a 0.10 gain (computed from the paired and unpaired variances). Minimum sample: one paired run on a 30-item set.

- **[practitioner]** Miller's sample-size formula is n = (z_alpha/2 + z_beta)^2 (omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / delta^2 (Eq. 9), with omega^2 the variance of the per-question paired difference, K the answers sampled per question and delta the minimum detectable effect (Eq. 10 solves for delta). Source: Miller 2024, arXiv:2411.00640, section 5, https://arxiv.org/abs/2411.00640 [S01]

  Implies: with K = 1, alpha 0.05, 80% power and 5% of items regressing, the held-out score needs about 27 items to detect a 30-point net gain, 51 for 20 points and 149 for 10 points (computed). The minimum detectable effect at 40% discordance is 0.29 at n = 30, 0.23 at n = 50 and 0.17 at n = 100 (computed). The exact test is stricter than the normal formula (appendix), so round up: 50 items per task is the smallest set that certifies a 20-point gain most of the time.

  Experiment (EN-4): freeze each task's held-out set at 50 items before training (disjoint by frame_ref and by question, rubric test 3) instead of about 30. Expected: exact McNemar power for a 20-point net gain rises from 0.40 to 0.70 and for a 30-point gain from 0.71 to 0.94 (computed). Minimum sample: 50 held-out items per task, labeled once by the teacher and spot-checked.

- **[measured]** Card et al. found that "typical test sets of 2000 sentences have approximately 75% power to detect differences of 1 BLEU point" in machine translation, and that small test sets leave most comparisons to state of the art underpowered. Source: D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, D. Jurafsky, "With Little Power Comes Great Responsibility", 2020, EMNLP 2020, pp. 9263-9274, doi:10.18653/v1/2020.emnlp-main.745, https://aclanthology.org/2020.emnlp-main.745/ (abstract) [S04]

  Implies: a small held-out set can only certify large gains, so before each run the pass log should record the minimum detectable effect for that set size; a smaller observed gain in the held-out score is reported as "no measured movement", not as an improvement.

  Experiment (EN-5): add a pre-run line to each learning run's log with n, the previous run's discordance and Miller's Eq. 10 minimum detectable effect. Expected: about 0.29 at n = 30 and 0.23 at n = 50 for 40% discordance (computed); runs whose observed gain is below it are recorded as no movement. Minimum sample: every run; checked on the first 3 runs.

- **[measured]** Bowyer et al. show that CLT-based intervals on small benchmarks are "usually dramatically underestimating uncertainty"; in their simulations 95% CLT intervals cover only 92.5% at N = 100, and for paired comparisons "all non-Bayesian methods severely underperform when it comes to achieving nominal coverage for small N". Source: S. Bowyer, L. Aitchison, D. R. Ivanova, "Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints", 2025, ICML 2025 (PMLR 267:81143-81184), https://arxiv.org/abs/2503.01747, sections 3.1 and 3.4, Figures 2 and 5 [S02]

  Implies: the error bars on the held-out score and on the label agreement rate must not be "score plus or minus 1.96 SE" at our n of 30 to 50; those bars are too narrow and would let a noise-level gain pass the gate.

  Experiment (EN-6): compute the CLT and the Wilson 95% interval for every reported held-out score and label agreement rate on the first run. Expected: the CLT bar has the wrong shape near the ends of the scale: at 28/30 it runs past 1 ([0.84, 1.02]) and is narrower than Wilson's [0.79, 0.98]; at 21/30 the two are about as wide but the CLT bar sits higher ([0.54, 0.86] against [0.52, 0.83]) (computed). Minimum sample: every reported score on one run.

- **[practitioner]** For one model's accuracy at small n, Bowyer et al. "recommend using WS [Wilson score] or Bayesian intervals in practice"; for two models on the same questions they "recommend using the paired Bayes method as it can account for correlations and thus produce narrower intervals", and they ship a library for it (`bayes_evals`). Source: Bowyer et al. 2025, ICML 2025, https://arxiv.org/abs/2503.01747, sections 3.1 and 3.4; library https://github.com/sambowyer/bayes_evals [S02]

  Implies: the gate can report the before-versus-after held-out score as a paired Bayesian interval on the difference, which is the honest form for the slide; McNemar stays the yes-or-no deploy test.

  Experiment (EN-7): compute the paired-Bayes interval and the exact McNemar p-value for the first real candidate. Expected: both agree on deploy or reject when the net gain is 0.3 or more at n = 30; they may disagree near the minimum detectable effect, and a disagreement is recorded for a ruling. Minimum sample: one paired run per task.

### Gaps
None of these sources covers structured outputs scored with partial credit (for example a desk-object list that is
half right). If the desk-object scorer gives partial credit, the paired test becomes a paired t or Wilcoxon test on
per-item scores (Miller's Eq. 7 still applies); that choice is open for the eval gate.

## 3. Run-to-run noise: the same model can score differently twice

### Takeaway
Temperature 0 is not deterministic on a shared server: batch size changes the numerics, and llama.cpp's own docs
say prompt caching can make results nondeterministic. Our runtime does not even ask for greedy decoding (it sends
temperature 0.2 and no seed). The rubric's "margin larger than measured run-to-run noise" therefore needs that
noise measured on our servers, or removed by fixing the decode settings for eval runs.

### Cited findings
- **[measured]** Thinking Machines sampled "Tell me about Richard Feynman" 1,000 times at temperature 0 from Qwen3-235B-A22B-Instruct-2507 on vLLM and got 80 unique completions (the most common 78 times), identical for the first 102 tokens; with batch-invariant kernels all 1,000 were identical. Their stated cause is that the forward pass is not batch-invariant and "the load (and thus batch-size) nondeterministically varies". Source: H. He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025-09-10, Thinking Machines Lab: Connectionism (lab write-up with data), https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ [S05]

  Implies: the teacher column of the held-out score comes from vLLM on pod A, which also serves labeling jobs; its score can move between runs with server load even at temperature 0. The label agreement rate measured against a teacher re-run has the same noise floor.

  Experiment (EN-8): score the teacher on the same 30 items twice, once with the server otherwise idle and once while a labeling job runs concurrently. Expected: 0 items change pass/fail when idle; 1 to 3 of 30 change under load. Minimum sample: 2 runs x 30 items.

- **[measured]** Atil et al. ran five LLMs "configured to be deterministic" on eight tasks, 10 runs each, and saw "accuracy variations up to 15% across naturally occurring runs", with a gap between best and worst possible performance of up to 70%. Source: B. Atil et al., "Non-Determinism of 'Deterministic' LLM Settings", 2024 (revised 2025), arXiv preprint, https://arxiv.org/abs/2408.04667 (abstract) [S06]

  Implies: run-to-run spread on a single model can be as large as the gain a LoRA run is expected to deliver, so the gate margin on the held-out score cannot be set from a single run of the current model.

  Experiment (EN-9): run the current Jetson model 5 times on the 30-item held-out set at production decode settings and record the score spread. Expected: spread (max minus min) of 0.03 to 0.10, that is 1 to 3 items. Minimum sample: 5 runs x 30 items, one model.

- **[practitioner]** The llama.cpp server documents that with `cache_prompt` (default `true`) "the logits are not guaranteed to be bit-for-bit identical for different batch sizes" and "enabling this option can cause nondeterministic results"; `seed` defaults to `-1`, "which is a random seed", and `temperature` to 0.8. Source: ggml-org/llama.cpp, `tools/server/README.md` (vendor doc), sections on `cache_prompt`, `seed` and `temperature`, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md [S09]

  Implies: our `LlamaServerRuntime` sends `"temperature": 0.2` with no seed (`runtime/llama.py` line 161 at bf482f4 [R2]) and leaves `cache_prompt` on, so two eval runs of the same model sample different answers by design. Either the eval fixes temperature 0, a fixed seed and `cache_prompt: false` for every model it scores (same decode settings for before, after and teacher, rubric test 4), or it keeps production settings and samples K answers per item as in the next finding.

  Experiment (EN-10): score one model 3 times with production settings and 3 times with temperature 0, seed 0 and `cache_prompt` false. Expected: production settings flip 2 to 6 of 30 items at least once across the 3 runs; the fixed settings flip 0. Minimum sample: 3 runs x 30 items x 2 settings, one model on one node.

- **[practitioner]** Miller recommends "reducing variance by resampling answers"; scoring each question as the mean of K sampled answers cuts the within-question variance to sigma_i^2 / K, with little benefit once that term is small next to the between-question variance, and in his uniform-difficulty binary example the total variance can fall by at most 2/3. Source: Miller 2024, arXiv:2411.00640, sections 1 and 3.1, https://arxiv.org/abs/2411.00640 [S01]

  Implies: if the eval keeps sampling (temperature 0.2), scoring each held-out item K = 3 times shrinks the noise part of the held-out score; it cannot shrink the between-item part, which only more items fix.

  Experiment (EN-11): on one model, compare the run-to-run spread of the held-out score at K = 1 (5 runs) with K = 3 (5 runs of 3 samples each). Expected: the spread falls by about a quarter at K = 3 (in Miller's uniform-difficulty example the variance falls to (1 + 2/K)/3 of its K = 1 value, 56% at K = 3, so the standard deviation falls about 25%), and the paired before-versus-after p-value on the same items falls with it. Minimum sample: 30 items x 5 runs x 4 samples.

### Gaps
No source measures llama.cpp's own run-to-run variation on CUDA for a single slot (`-np 1`, as the Jetson runs it);
EN-10 measures it directly. Whether the Jetson's `--reasoning-budget 0` changes determinism is unknown.

## 4. Training-seed noise: one good run can be luck

### Takeaway
The before-versus-after comparison holds the candidate fixed; claims about which recipe is better (rank, learning
rate, data) need more than one training seed per arm, because fine-tuning on small data varies widely by seed.

### Cited findings
- **[measured]** Dodge et al. fine-tuned BERT "hundreds of times" on each of several GLUE tasks varying only random seeds and found that "distinct random seeds can lead to substantially different results"; weight initialization and data order contribute comparably, and on small datasets "many fine-tuning trials diverge part of the way through training". Source: J. Dodge, G. Ilharco, R. Schwartz, A. Farhadi, H. Hajishirzi, N. Smith, "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping", 2020, arXiv preprint, https://arxiv.org/abs/2002.06305 (abstract) [S08]

  Implies: a recipe change (for example rank 16 versus 8, or 300 versus 600 examples) moves the held-out score only if it beats the spread across training seeds; each run's ledger row must carry the seed (rubric test 5) so that spread can be computed.

  Experiment (EN-12): train the same recipe with 3 seeds on the same labeled set and score each on the same held-out items. Expected: held-out score spread across seeds of 0.03 to 0.10 at n = 30; any recipe comparison smaller than that spread is recorded as no movement. Minimum sample: 3 seeds x 1 recipe x 30 items (about 3 training slots on pod B).

### Gaps
The seed-variance evidence is for BERT-sized full fine-tuning on GLUE; none of the sources opened measures seed
variance for LoRA on a 4B model with a few hundred examples. EN-12 measures it for us.

## 5. Metrics that resolve differences with fewer items

### Takeaway
Pass/fail is a coarse signal. A continuous per-item score, such as the log-likelihood of the reference answer,
separates models with fewer items; it can sit beside the pass/fail held-out score as a dotted key in the
`learning.run.evaluated` scores dict (the Q4 answer on Ilan's Team Comms page already allows that [R8]).

### Cited findings
- **[measured]** Heineman et al. define signal ("a benchmark's ability to separate better models from worse models") and noise ("a benchmark's sensitivity to random variability between training steps"), and find that "switching to a metric that has better signal and noise (e.g., perplexity rather than accuracy)" leads to better reliability; averaging intermediate checkpoints also reduces noise. Source: D. Heineman, V. Hofmann, I. Magnusson, Y. Gu, N. A. Smith, H. Hajishirzi, K. Lo, J. Dodge, "Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation", 2025, arXiv preprint, https://arxiv.org/abs/2508.13144 (abstract) [S07]

  Implies: record a second held-out metric per model, the mean per-token negative log-likelihood of the teacher's reference answer computed on pod B with the bf16 weights; it moves before pass/fail does and shows whether a run is learning at all when the pass/fail gain is below the minimum detectable effect.

  Experiment (EN-13): for base and candidate, compute per-item reference-answer NLL on the 30 held-out items and test the paired difference (paired t on per-item NLL). Expected: the NLL difference is significant (p < 0.01) on runs whose pass/fail gain is only 0.1 to 0.2, below the pass/fail minimum detectable effect. Minimum sample: 30 items, one pass per model, a few minutes of pod B time.

### Gaps
The NLL is measured on the bf16 model, not on the Q4_K_M GGUF the Jetson runs; quantization shifts it (see
`packaging_model_card.md`). It is a training-progress signal, not a substitute for the deploy test.

## Appendix: how the computed numbers were made

Exact two-sided McNemar p-value: with b items flipping fail to pass and c flipping pass to fail, p = min(1, 2 x
BinomCDF(min(b, c); b + c, 0.5)). Power: each item independently flips fail to pass with probability p10 and pass
to fail with probability p01; power is the probability that b > c and p < 0.05, summed exactly over the
multinomial. The regression rate p01 is 0.05 throughout.

| Items n | Net gain +0.10 | +0.20 | +0.30 | +0.40 |
|---|---|---|---|---|
| 6 | 0.00 | 0.00 | 0.00 | 0.01 |
| 20 | 0.03 | 0.20 | 0.45 | 0.69 |
| 30 | 0.10 | 0.40 | 0.71 | 0.90 |
| 50 | 0.24 | 0.70 | 0.94 | 0.99 |
| 100 | 0.54 | 0.96 | 1.00 | 1.00 |

Miller's Eq. 9 with K = 1, alpha 0.05, power 0.8 and omega^2 = (p10 + p01) - (p10 - p01)^2 gives n = 149, 51, 27 and
17 for the same four gains; the exact test needs more (for +0.30, 30 items give 0.71 power, not 0.80), which is the
small-n optimism Bowyer et al. describe [S02]. Wilson 95% intervals: 3/6 = [0.19, 0.81], 5/6 = [0.44, 0.97], 12/30 =
[0.25, 0.58], 21/30 = [0.52, 0.83], 24/30 = [0.63, 0.90], 28/30 = [0.79, 0.98].

```python
from math import comb
from scipy.stats import binom


def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))


def power(n, p10, p01, alpha=0.05):
    p11 = 1 - p10 - p01
    return sum(
        comb(n, b) * comb(n - b, c) * p10**b * p01**c * p11 ** (n - b - c)
        for b in range(n + 1)
        for c in range(n + 1 - b)
        if b > c and mcnemar_p(b, c) < alpha
    )
```

## 6. Clustered items, leakage and the decode temperature (learning pass 1, 2026-10-10)

Added by the research step of learning pass 1. Each source was opened on 2026-10-10 before it was cited.

- **[practitioner]** Miller recommends clustered standard errors when "eval questions ... are drawn in groups, or clusters", reports in Table 4 that "clustered standard errors can be over 3X larger than naive standard errors" (DROP 1.34 against 0.44, MGSM 1.62% against 0.86%, RACE-H 0.51% against 0.46%), warns that ignoring the clustering "may lead an unsuspecting analyst to suppose that the measurement of the overall eval score is much more precise than it actually is", and suggests "including the cluster count alongside the question count". Source: E. Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations", 2024, arXiv preprint, https://arxiv.org/abs/2411.00640, section 2.2, Tables 3 and 4 [S01]
  - Implication: the export takes every keyframe of each matching episode (`contexts/learning/application/export.py`, keyframe selection), so a desk-object held-out set holds clusters of near-identical frames. Its held-out score (current, candidate and teacher) is reported with the episode count beside the frame count and a standard error clustered by episode, and the minimum detectable effect is computed from the episode count. A frame-level split can also put one episode on both sides of the split (rubric test LT3).
  - Candidate experiment (EN-14): on the first desk-object held-out set, compute the naive and the episode-clustered standard error of the current model's score. Expected movement: the clustered standard error of learning.heldout.current is 1.1 to 3 times the naive one (Miller's Table 4 ratios run from 1.10 to 3.05); a ratio near 1 means the episodes hold no near-duplicates and the frame count is a fair n. Minimum sample size: one held-out set with at least 10 episodes.

- **[practitioner]** Kapoor and Narayanan present "a fine-grained taxonomy of 8 types of leakage" and report errors in "17 fields ... collectively affecting 329 papers", "in some cases leading to wildly overoptimistic conclusions". Source: S. Kapoor and A. Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", arXiv:2207.07048 (July 2022), https://arxiv.org/abs/2207.07048 (abstract) [S65]
  - Implication: a held-out score is void when training and held-out share an episode, a frame or a question. The eval gate reads only the frozen split and checks both disjointness rules before it scores, so learning.heldout.candidate is not inflated by leakage. The fixture dataset planned on the L11 page is built from `fixtures/golden/questions.yaml`, the same questions the golden scorer asks [R1], so a gate that scores the golden set after training on it would report memorisation as learning.
  - Candidate experiment (EN-15): after the first tool-calling run, score the candidate on golden questions it trained on and on held-out questions it never saw. Expected movement: learning.heldout.candidate is higher on the trained questions than on the held-out ones; the gap is the leakage a golden-set gate would have shown as a gain. Minimum sample size: 6 golden and 30 held-out questions, one run.

- **[practitioner]** Miller's section 3.3 ("Don't touch the thermostat!") advises against lowering the sampling temperature to reduce score variance, "unless the purpose is to study the model at the new temperature", shows a case where lowering it "has inadvertently tripled the minimum variance in the score data from 1/12 to 1/4", and concludes that "In neither case should the sampling temperature be adjusted for the sake of reducing variance in the scores". Source: Miller 2024, arXiv:2411.00640, section 3.3, https://arxiv.org/abs/2411.00640 [S01]
  - Implication: this pulls against EN-10 (score at temperature 0 with a fixed seed to remove run-to-run noise). The gate scores each model at one recorded decode setting, the same for current, candidate and teacher, and shrinks noise by resampling K answers per item (section 3) rather than by changing the temperature. Which setting is pinned is a ruling raised by learning pass 1: the Qwen3.5-4B card recommends temperature 0.7 for non-thinking use [S24], while `runtime/llama.py` sends 0.2 with no seed [R2].
  - Candidate experiment (EN-16): score the current model on one held-out set 3 times at the served setting with K = 1, 3 times with K = 3, and once at temperature 0. Expected movement: the spread of learning.heldout.current across runs falls by about a quarter at K = 3 (as in EN-11), and the temperature-0 score sits outside the served-setting spread on some items, which is why it cannot stand in for the served model. Minimum sample size: 30 items x 7 runs, one model.

## 7. Scoring the desk-object task (learning pass 1, 2026-10-10)

- **[practitioner]** POPE, "a polling-based query method" (EMNLP 2023), is offered as a way to "evaluate the object hallucination in a more stable and flexible way" than methods that "might be affected by the input instructions and generation styles of LVLMs"; its authors report that LVLMs "mostly suffer from severe object hallucination issue". Source: Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, J.-R. Wen, "Evaluating Object Hallucination in Large Vision-Language Models", 2023, EMNLP 2023, https://arxiv.org/abs/2305.10355 (abstract) [S66]
  - Implication: the desk-object task has no scorer yet, because `contexts/agent/eval.py` scores the golden questions only [R1]. A scorer that asks per-object yes or no questions against a hand-checked object list gives pass or fail items that the paired McNemar test of section 2 reads directly, while a scorer that parses each model's free list depends on how each model phrases it, which the current, candidate and teacher do differently.
  - Candidate experiment (EN-17): build both scorers for the first desk-object held-out set, (A) set F1 of the parsed free list against the hand-checked list and (B) polling questions about listed and absent objects, and score the current model twice with each. Expected movement: the run-to-run spread of learning.heldout.current is smaller under B than under A. Minimum sample size: 30 keyframes, 2 runs per scorer.

## Open questions

- Does the AUP hub proxy accept `Authorization: Bearer <token>` as well as `Authorization: token <token>`? The JupyterHub REST API page opened on 2026-10-10 documents only the `token` form; if Bearer works, an OpenAI-style client with the hub token as its key could score the teacher through the proxy (rubric open item under LT4). Not tested.
- Which decode setting the three scores share, and how many held-out items each task needs: both are rulings raised by learning pass 1.
- A third-party digest of llama.cpp issues listed tool-call parsing errors on Linux CUDA in April 2026 (search snippet, not opened); issue #22422, which was opened, is MiniMax-only (see `distillation_tool_calls.md` section 5).

## Search log

- 2026-10-10, learning pass 1 research step: budget 2 h, used about 30 min (12:03 to 12:33 PM ET). Starting points: the rubric's known open items (held-out n, decode settings, the teacher behind the proxy, near-duplicate keyframes, the golden questions in L11's fixture, the desk-object scorer) and the Gaps sections of the four notes. Opened: Miller 2024 sections 2.2, 3.3 and 4.2; Dietterich 1998 abstract; Atil et al. abstract; the `bayes_evals` README; Kapoor and Narayanan abstract; POPE abstract; the JupyterHub REST API page; the TRL SFT Trainer page; the PyTorch 2.14 reproducibility notes; the Qwen3.5-4B and Qwen3.8-27B cards and the Qwen3.5-4B chat template; llama.cpp `docs/function-calling.md`, `common/chat.cpp` at 8b54361 and issue #22422; transformers `hub_kernels.py`; the fla README; the APIGen abstract. Query: "llama.cpp llama-server Qwen3.5 tool calling qwen3_coder parser issue 2026". Stopped when the last three sources (the search digest, issue #22422 and the issue list behind it) added nothing about Qwen3.5 beyond what `common/chat.cpp` already showed.
