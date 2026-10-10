# LoRA on Qwen3.5-4B on one R9700: model class, vision tower, rank, learning rate, speed and memory

Research note for the learning track, opened 2026-10-10 (all sources opened that day; index in `SOURCES.md`).
It covers the trainer recipe for the student (`Qwen/Qwen3.5-4B`, bf16, PEFT 0.21, TRL 1.15, transformers 5.19 on
pod B, torch 2.9.1+rocm7.13.0, gfx1201) on a few hundred teacher-labeled examples. Labels: `measured` = a number
someone measured, with the setup; `practitioner` = a documented recommendation or documented behaviour from the
people who build or run the thing (papers' recommendation sections, official docs, model cards, source code,
maintainers); `anecdote` = a single report without data; `opinion` = argument without evidence. Every finding
carries its citation and a `[S..]` or `[R..]` key into `SOURCES.md`. Under each finding, "Implies" maps it onto our
metric set and "Experiment" proposes a test with the expected movement and the minimum sample. Our metric set is the learning rubric's [R7] (`lab/refine/learning/RUBRIC.md`, "The metric set"):
held-out score before, after and teacher (`learning.heldout.current`, `learning.heldout.candidate`,
`learning.heldout.teacher`); label agreement rate (`learning.label_agreement`); loop wall-clock
(`learning.loop_minutes`); training samples per second (`trainer.samples_per_s`, 0.51 on pod B, PR #37);
package size (`learning.package_mb`).

Our own baseline: `lab/smoke_lora.py` loads the model with `AutoModelForCausalLM`, LoRA r = 16, alpha = 32, dropout
0, `target_modules="all-linear"`, lr 2e-4, batch 4, `max_length` 256, 48 synthetic text examples, 20 steps, TRL
defaults otherwise; measured loss 2.23 to 0.16, 0.51 samples/s, 8.7 GB peak [R3].

## 1. What the smoke run actually trained, and what the vision task needs

### Takeaway
Qwen3.5-4B is a hybrid model (24 Gated DeltaNet linear-attention layers, 8 gated-attention layers) with a 24-layer
vision encoder. `AutoModelForCausalLM` maps it to the text-only `Qwen3_5ForCausalLM`, so the smoke run's 0.51
samples/s and 8.7 GB say nothing about image examples. The desk-object task needs the image-text class and the
processor; once that class is loaded, `all-linear` also puts adapters on the vision encoder, which Qwen's own
fine-tuning recipe keeps frozen.

### Cited findings
- **[practitioner]** The Qwen3.5-4B model card describes a "Causal Language Model with Vision Encoder" with 32 layers, hidden size 2560 and "Hidden Layout: 8 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))"; its `config.json` lists a 24-layer vision encoder (hidden 1024, output 2560, spatial merge 2), an empty `deepstack_visual_indexes`, tied word embeddings and a 248,320-token vocabulary, and the card's quick start loads it with `AutoModelForMultimodalLM`. Source: Qwen, "Qwen/Qwen3.5-4B" model card and `config.json` (vendor docs), https://huggingface.co/Qwen/Qwen3.5-4B and https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json [S24] [S25]

  Implies: three quarters of the language layers are linear attention, so training samples per second depends on the Gated DeltaNet kernels (section 4), not only on attention kernels; any throughput or memory figure measured on a pure-transformer model of the same size does not transfer.

  Experiment (LR-1): profile one smoke step with `torch.profiler` and attribute time to the DeltaNet layers versus attention and MLP. Expected: the 24 DeltaNet layers take more than half of the step time while the fast kernels are missing. Minimum sample: 5 profiled steps after 5 warm-up steps.

- **[practitioner]** In transformers, `AutoModelForCausalLM` maps the `qwen3_5` config to `Qwen3_5ForCausalLM` (commented "# VLM compatibility"), while `AutoModelForImageTextToText` maps it to `Qwen3_5ForConditionalGeneration`, the class that holds the vision model (`Qwen3_5VisionModel`) and its `merger`. Source: huggingface/transformers, `src/transformers/models/auto/modeling_auto.py` and `src/transformers/models/qwen3_5/modeling_qwen3_5.py` (source code, main branch), https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_auto.py [S30] [S28]

  Implies: the smoke run's training samples per second (0.51) and peak memory (8.7 GB) are text-only numbers. The tool-calling task can keep the text-only class; the desk-object task cannot, and its samples per second and memory must be re-baselined with images before the loop wall-clock budget is set.

  Experiment (LR-2): run the 20-step smoke with `AutoModelForImageTextToText`, `AutoProcessor` and 48 fixture examples that each carry one blurred 1280x720 keyframe. Expected: samples/s falls below the text-only figure (each image adds hundreds of tokens) and peak memory rises from 8.7 GB to roughly 10 to 14 GB. Minimum sample: 2 runs of 20 steps.

- **[practitioner]** PEFT documents that with `target_modules="all-linear"` "all linear/Conv1D modules are chosen (if the model is a PreTrainedModel, the output layer excluded)". Source: Hugging Face PEFT, "LoRA" developer guide and `LoraConfig` reference (vendor doc), https://huggingface.co/docs/peft/main/en/developer_guides/lora, section "target_modules" [S16]

  Implies: on `Qwen3_5ForConditionalGeneration`, `all-linear` wraps the vision encoder and the merger as well as the language model. A vision-side adapter changes what the vision projector file must contain at packaging time (see `packaging_model_card.md`) and adds trainable parameters that a few hundred images may not justify; package size and the merge path both depend on this choice.

  Experiment (LR-3): print the wrapped module names and trainable-parameter counts for `all-linear` and for a language-only target list (a regex that excludes modules under the vision tower), then train both on the same desk-object set. Expected: the language-only adapter has noticeably fewer trainable parameters and a held-out score within the minimum detectable effect of the `all-linear` one. Minimum sample: 1 training run per arm, 50 held-out items.

- **[practitioner]** Qwen's official VLM fine-tuning code exposes `tune_mm_vision`, `tune_mm_mlp` and `tune_mm_llm` flags; its example script trains the language model only (`--tune_mm_llm True --tune_mm_vision False --tune_mm_mlp False`), uses separate learning rates for the projector (1e-5) and the vision encoder (1e-6), 3 epochs, batch 4 with 4 accumulation steps, and a LoRA example with r = 8, alpha = 16. Source: QwenLM, `Qwen3-VL/qwen-vl-finetune` README and example script (vendor doc), https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune [S34]

  Implies: the vendor's default is a language-only update with the vision side frozen; matching it keeps the base vision projector valid for packaging and keeps the comparison with the current Jetson model about language behaviour only.

  Experiment (LR-4): if the desk-object held-out score stays below the teacher's by more than the minimum detectable effect after a language-only run, train a second arm that also unfreezes the merger at lr 1e-5. Expected: 0.05 to 0.10 gain on desk objects only if the failures are perceptual (wrong objects) rather than format errors. Minimum sample: 1 run per arm, 50 held-out items, failures hand-tagged as perceptual or format.

- **[practitioner]** Unsloth's Qwen3.5 fine-tuning guide sets all of its vision fine-tuning flags on by default ("We set them all on by default!"), lists bf16 LoRA memory as 3 GB for 0.8B, 5 GB for 2B and 10 GB for 4B, and says the model "must use the same chat template you trained with" at inference. Source: Unsloth, "Qwen3.5 fine-tuning guide" (vendor doc), https://unsloth.ai/docs/models/qwen3.5/fine-tune [S35]

  Implies: two practitioner defaults disagree on the vision tower (Unsloth on, Qwen off), so LR-3 settles it with our data. The 10 GB figure matches our 8.7 GB text-only peak, which leaves most of the 32 GB card free for batch size or for turning gradient checkpointing off (section 4).

  Experiment (LR-3, same as above): the vision-on versus vision-off comparison, also recording peak memory for each arm. Expected: the language-only arm scores within the minimum detectable effect of the vision-on arm and uses less memory. Minimum sample: 1 training run per arm, 50 held-out items.

### Gaps
No source opened measures LoRA on Qwen3.5 with images on AMD hardware; LR-2 is the first such number. Whether
PEFT's `all-linear` wraps the Gated DeltaNet projections (they are linear layers, so it should) was not checked in
code; LR-3's module listing answers it.

## 2. Rank, alpha and learning rate for a few hundred examples

### Takeaway
With LoRA on all layers, rank barely matters for small datasets; the learning rate matters most. Adapters want
about 10 times the full fine-tuning rate, which puts the useful range at roughly 1e-4 to 5e-4. Our smoke setting
(r = 16, alpha = 32, lr 2e-4) sits inside the evidence; the open choice is the learning rate, picked on the held-out
task score rather than on training loss.

### Cited findings
- **[measured]** In the original LoRA paper, GPT-3 175B adapted on Wq, Wk, Wv and Wo scored 74.1, 73.7, 74.0, 74.0 and 73.9 on WikiSQL and 91.2, 91.7, 91.7, 91.5 and 91.4 on MultiNLI at r = 1, 2, 4, 8 and 64 (Table 6); "LoRA already performs competitively with a very small r", and "tuning alpha is roughly the same as tuning the learning rate". Source: E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021, arXiv preprint, https://arxiv.org/abs/2106.09685, section 4.1 and section 7.2, Table 6 [S10]

  Implies: rank is not the lever for the held-out score at our data size; keeping r fixed and sweeping the learning rate is the efficient use of pod B slots.

  Experiment (LR-5): train r = 8, 16 and 32 at the same learning rate on the same labeled set. Expected: held-out scores within the seed spread of each other (see `eval_noise.md`, EN-12). Minimum sample: 3 runs, 50 held-out items.

- **[measured]** QLoRA's authors found "LoRA r is unrelated to final performance if LoRA is used on all layers" (their Appendix A.1 sweep) and trained with r = 64, alpha = 16, dropout 0.1 for 7B and 13B, learning rate 2e-4 for 7B and 13B (1e-4 for 33B and 65B), a constant schedule and max gradient norm 0.3. Source: T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs", 2023, NeurIPS 2023, https://arxiv.org/abs/2305.14314, Appendix A.1, Appendix B.2, Table 9 [S14]

  Implies: covering all linear layers matters more than rank for the held-out score; 2e-4 is a measured-good rate for 7B-class adapters, close to our 4B.

  Experiment (LR-6): compare attention-only target modules with all-linear at r = 16 on the tool-calling set. Expected: all-linear scores at least 0.05 above attention-only. Minimum sample: 1 run per arm, 50 held-out items.

- **[measured]** Thinking Machines found that "the optimal LR for LoRA is consistently 10x the one used for FullFT in the same application", that "applying LoRA to the attention matrices shows no additional benefits beyond applying it to the MLPs only", that "for datasets that exceed LoRA capacity, LoRA underperforms FullFT", and that "LoRA is less tolerant of large batch sizes than full fine-tuning". Source: J. Schulman and Thinking Machines Lab, "LoRA Without Regret", 2025-09-29, Thinking Machines Lab: Connectionism (lab write-up with data), doi:10.64434/tml.20250929, https://thinkingmachines.ai/blog/lora/ [S12]

  Implies: a few hundred short answers are far inside LoRA's capacity at r = 16, so the held-out score is limited by labels and learning rate, not rank; MLP coverage is required; and small batches (4 to 16) are the right regime, which also suits the 32 GB card.

  Experiment (LR-7): sweep lr 1e-4, 2e-4 and 5e-4 at r = 16 with batch 4 for the same number of epochs. Expected: the best held-out score is at 2e-4 or 5e-4 and 1e-4 is at least 0.05 lower after the same steps. Minimum sample: 3 runs, 50 held-out items, one seed each (repeat the winner with a second seed).

- **[measured]** Biderman et al. instruction-tuned Llama-2-7B and at epoch 4 measured HumanEval 0.358, 0.417 and 0.498 for LoRA r = 16, 64 and 256 against 0.470 for full fine-tuning, and GSM8K 0.580, 0.624 and 0.625 against 0.642; LoRA's best learning rates (5e-4 code, 2e-4 math) were an order of magnitude above full fine-tuning's, and they recommend targeting all modules, alpha = 2r and sweeping the learning rate from 1e-5 to 5e-4 to pick the highest stable value. Source: D. Biderman et al., "LoRA Learns Less and Forgets Less", 2024, Transactions on Machine Learning Research (August 2024, Featured Certification), https://arxiv.org/abs/2405.09673, section 4.7, Table S5 [S13]

  Implies: rank starts to matter only when the dataset is large and far from the base model (a large code instruction-tuning set); our few hundred examples are the opposite case. Alpha = 2r, as in our smoke run, is the measured-good convention.

  Experiment (LR-7, same): the learning-rate sweep above covers this; run 5e-4 only if 2e-4 trains stably (no loss spikes above the first-step loss). Expected: the best held-out score at 2e-4 or 5e-4, with 1e-4 at least 0.05 lower after the same steps. Minimum sample: 3 runs, 50 held-out items.

- **[practitioner]** PEFT documents rank-stabilized LoRA: "The scalar is given by `lora_alpha/r` in the original implementation, but rsLoRA uses `lora_alpha/math.sqrt(r)` which stabilizes the adapters and increases the performance potential from using a higher `r`"; the rsLoRA paper says the original scaling "results in slowed learning and stunted performance for LoRA with higher-rank adapters". Source: Hugging Face PEFT LoRA developer guide (vendor doc), https://huggingface.co/docs/peft/main/en/developer_guides/lora, section on rsLoRA; D. Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA", 2023, arXiv preprint, https://arxiv.org/abs/2312.03732 (abstract) [S16] [S11]

  Implies: rsLoRA only matters if we raise the rank well above 16; at r = 8 to 32 it is a learning-rate rescale and should not be mixed into the learning-rate sweep.

  Experiment (LR-8, only once a run needs r of 64 or more): compare `use_rslora` on and off at r = 64 and the same learning rate. Expected: no held-out difference beyond the seed spread at our data size. Minimum sample: 1 run per arm, 50 held-out items.

- **[practitioner]** TRL's `SFTConfig` defaults to learning rate 2e-5, 3 epochs, per-device batch 8, `gradient_checkpointing=True`, `bf16=True` and `max_length=1024`, and the docs add that when training adapters "you typically use a higher learning rate (about 1e-4) since only new parameters are being learned"; for vision-language models they advise `max_length=None` because "truncating may remove image tokens". Source: Hugging Face TRL, "SFT Trainer" (vendor doc, main), https://huggingface.co/docs/trl/main/en/sft_trainer, sections "Train adapters with PEFT", "Training Vision Language Models" and `SFTConfig` [S17]

  Implies: every SFTConfig value our trainer does not set explicitly takes a full-fine-tuning default; the learning rate, `max_length` and gradient checkpointing must be explicit in the run config and in the ledger row (rubric test 5).

  Experiment (LR-9): before the first real desk-object run, tokenize the whole training set with the processor and record the longest example. Expected: image examples exceed 256 tokens, so `max_length` 256 would truncate them; set `max_length=None` or above the measured maximum. Minimum sample: every training example, once.

- **[practitioner]** Unsloth's LoRA guide recommends 2e-4 "as a starting point", "1-3 epochs" ("training for more than 3 epochs offers diminishing returns"), r of 16 or 32, `lora_alpha` equal to r or 2r, dropout 0 to 0.1, all major linear layers, an effective batch of 16, warmup of 5 to 10% of steps and weight decay 0.01. Source: Unsloth, "LoRA Hyperparameters Guide" (vendor doc), https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide [S18]

  Implies: our smoke values match this guide except warmup and weight decay; with about 300 examples and batch 4 one epoch is about 75 steps, so 1 to 3 epochs is 75 to 225 optimizer steps, which sets the loop wall-clock at a given samples per second.

  Experiment (LR-10): score the held-out set after epochs 1, 2 and 3 of one run. Expected: the held-out score peaks at epoch 2 or 3, and epoch 3 is not more than 0.05 below the peak. Minimum sample: 1 run with 3 checkpoints, 50 held-out items.

- **[practitioner]** Thinking Machines' Tinker docs default to rank 32, say supervised fine-tuning works well when "the number of LoRA parameters is at least as large as the number of completion tokens", and state that "the optimal learning rate does not depend on the LoRA rank". Source: Thinking Machines Lab, "LoRA Primer" (Tinker docs, vendor doc), https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/ [S19]

  Implies: a few hundred answers of a few dozen tokens are on the order of 10^4 completion tokens, while an r = 8 adapter on all linear layers has millions of parameters (one 2560-wide square projection alone adds 8 x (2560 + 2560) = 40,960 at r = 8, and each of the 32 layers has several); by this rule rank is nowhere near binding, so it can stay at 16 while the learning rate is tuned.

  Experiment (LR-5, same): the rank comparison above tests this directly. Expected: r = 8, 16 and 32 land within the seed spread of each other on the held-out score. Minimum sample: 3 runs, 50 held-out items.

- **[measured]** LIMA fine-tuned on 1,000 curated examples for 15 epochs (AdamW, learning rate 1e-5 decaying to 1e-6, batch 32), found that "perplexity does not correlate with generation quality", and so selected checkpoints by hand "between the 5th and the 10th epochs using the held-out 50-example development set"; "doubling the training set does not improve response quality". Source: C. Zhou et al., "LIMA: Less Is More for Alignment", 2023, arXiv preprint, https://arxiv.org/abs/2305.11206, section 3 and section 5 [S15]

  Implies: training loss (2.23 to 0.16 in the smoke run) is not evidence that the held-out score moved; the run should save checkpoints and pick one on a held-out task score, which needs a small dev split separate from the gate's held-out set.

  Experiment (LR-11): split 20 labeled items off as a dev set, save a checkpoint each epoch, and pick by dev score; then score only the chosen checkpoint on the held-out set. Expected: the dev-picked checkpoint scores at least as well on the held-out set as the last checkpoint. Minimum sample: 20 dev items, 3 checkpoints, 50 held-out items.

### Gaps
No source measures these hyperparameters on a 4B hybrid-attention VLM with a few hundred examples, on AMD, or on
tool-calling and object-listing tasks. The learning-rate multiplier is the best-supported transfer; LR-7 is the
experiment that turns it into our number.

## 3. Which tokens carry the loss, and the empty thinking block

### Takeaway
Our two tasks have long prompts (system text, tool schemas, memory snippets or an image) and short answers. The
measured evidence on loss masking points both ways: target-only loss won on MMLU for QLoRA, while two later studies
found that some prompt loss helps exactly when outputs are short and examples are few. TRL offers only on or off,
so treat it as an experiment. Whatever the masking, the training text must match what the Jetson serves with
thinking off.

### Cited findings
- **[measured]** QLoRA's ablation found "only training on the target is beneficial to MMLU performance": 38.6 mean 5-shot MMLU when training on the target only against 37.5 when training on source and target. Source: Dettmers et al. 2023, NeurIPS 2023, https://arxiv.org/abs/2305.14314, Appendix B.3, Table 10 [S14]

  Implies: completion-only loss is the safe default for the held-out score, and TRL already applies it to prompt-completion datasets.

  Experiment (LR-12): train the tool-calling set as a prompt-completion dataset (completion-only loss) and as a messages dataset with full-sequence loss, same seed and learning rate. Expected: completion-only scores within the noise of, or above, full-sequence loss. Minimum sample: 1 run per arm, 50 held-out items.

- **[measured]** Huerta-Enochian and Ko swept prompt-loss weight from 0 to 1 on LLaMA models and found a "statistically-significant negative quadratic relationship with PLW" on short-completion data, with an optimum near 0.242 and full masking "almost always" worst on long-generation benchmarks; on long-completion data "PLW can be safely ignored". Source: M. Huerta-Enochian and S. Y. Ko, "Instruction Fine-Tuning: Does Prompt Loss Matter?", 2024, arXiv preprint (v4), https://arxiv.org/abs/2401.13586, sections 5.1 and 5.2, Table 3 [S21]

  Implies: our outputs are short, which is the regime where full masking was worst in their full fine-tuning runs; that cuts against the TRL default for our held-out score.

  Experiment (LR-12, same): the masked versus full-sequence comparison; a fractional weight would need a custom loss and is not proposed. Expected: on these short-output tasks the full-sequence arm scores up to 0.05 higher, the opposite of the QLoRA result, with the difference likely inside the noise. Minimum sample: 1 run per arm, 50 held-out items.

- **[measured]** Shi et al. report that instruction modelling (loss over instructions as well as outputs) "is especially beneficial when trained on datasets with lengthy instructions paired with brief outputs" or with few training examples, and in the most favourable case "boosts model performance on AlpacaEval 1.0 by over 100%". Source: Z. Shi, A. X. Yang, B. Wu, L. Aitchison, E. Yilmaz, A. Lipani, "Instruction Tuning With Loss Over Instructions", 2024, NeurIPS 2024, https://arxiv.org/abs/2405.14394 (abstract) [S20]

  Implies: same as above; with few examples and short outputs, the masked-loss default may cost held-out score on free-text answers, though neither study used LoRA or tool-calling data.

  Experiment (LR-12, same): the masked versus full-sequence comparison, scored separately on the free-text answers and on the tool-call steps. Expected: any gain from prompt loss shows on free-text answers, not on tool-call format. Minimum sample: 1 run per arm, 50 held-out items.

- **[practitioner]** The ms-swift Qwen3.5 best-practice guide trains Qwen3.5-4B with `--add_non_thinking_prefix true` and `--loss_scale ignore_empty_think` and runs inference with `enable_thinking=False`; its example responses begin with an empty `<think></think>` block. Source: ModelScope ms-swift, "Qwen3.5 Best Practices" (vendor doc, v4.4), https://swift.readthedocs.io/en/v4.4/BestPractices/Qwen3_5-Best-Practice.html [S33]

  Implies: the Jetson serves the student with thinking off (`--reasoning-budget 0` [R4]), and Qwen3.5's template then emits an empty thinking block before the answer (see `distillation_tool_calls.md`); training targets must carry the same prefix, while the loss should not be spent on it. A mismatch shows up as format failures in the held-out score.

  Experiment (LR-13): render 20 training examples through the chat template with `enable_thinking=False` and diff them against what llama-server sends to the model for the same messages. Expected: identical token sequences, including the empty thinking block. Minimum sample: 20 examples, once per template change.

### Gaps
Neither loss-masking study used LoRA, a VLM, or tool-call outputs; LR-12 is the deciding measurement.

## 4. Speed and memory on one R9700

### Takeaway
The likely reason the smoke run reached only 0.51 samples/s is that the 24 Gated DeltaNet layers ran on
transformers' pure-PyTorch fallback: the fast kernels come from `flash-linear-attention` (`fla`), the pod setup does
not install it, and the only ROCm hub kernel for this layer targets Strix Halo (gfx1151), not gfx1201. `fla` claims
AMD support through Triton. The second lever is gradient checkpointing, on by default in TRL, which costs roughly
20 to 30% of throughput and buys memory we do not need at 8.7 GB of 32 GB.

### Cited findings
- **[measured]** Our smoke LoRA on pod B (one R9700, text-only class, r = 16, batch 4, 256 tokens, 20 steps, TRL defaults including gradient checkpointing) went from loss 2.23 to 0.16 at 0.51 samples/s with an 8.7 GB peak. Source: Atlas PR #37 (merged), "R2: GPU lab setup", verification section, and `lab/smoke_lora.py`, https://github.com/raohemdutt/Knighthacks26/pull/37 [R3]

  Implies: this is the baseline for training samples per second; at 0.51 samples/s, 300 examples for 3 epochs take about 29 minutes, right at the trainer's "under about 30 min" target [R9] before any image tokens are added.

  Experiment (LR-14): every speed change below is measured against this exact command (same 48 examples, 20 steps). Expected: run-to-run spread under 5% on an idle pod. Minimum sample: 2 runs of the unchanged baseline in the same slot.

- **[practitioner]** In transformers' Qwen3.5 code the Gated DeltaNet kernels are hooked with `use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")` (and `causal_conv1d` for the convolution), with `torch_chunk_gated_delta_rule` as the fallback; the helper tries hub kernels, then the "original package", then the "torch only path", and logs that the function "is falling back to its reference PyTorch implementation because" the package "is not installed. This is correct but much slower". Its hub mapping for `Qwen3_5GatedDeltaNet` has a CUDA entry for capability 12.1 and a ROCm entry only for "AMD Strix Halo (gfx1151, capability 11.5)". Source: huggingface/transformers, `src/transformers/models/qwen3_5/modeling_qwen3_5.py` and `src/transformers/integrations/hub_kernels.py` (source code, main branch), https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py [S28] [S29]

  Implies: on gfx1201 without `fla` installed (our `pod_b_trainer.sh deps` step does not install it [R3]) the smoke run took the slow path for 24 of 32 layers; training samples per second has a large, cheap lever before any kernel work.

  Experiment (LR-15): grep the existing `~/smoke_lora.log` on pod B for "falling back to its reference PyTorch implementation", then `pip install --user flash-linear-attention` (bare, so the pod's torch stays) and, if it builds, `causal-conv1d`, and re-run the smoke. Expected: the warning is in the old log and gone from the new one, and samples/s rises from 0.51 to at least 1.0 (the source says only "much slower", so the size is our guess). Minimum sample: 2 runs of 20 steps before and after, one 15-minute pod B slot.

- **[practitioner]** The `fla` README says "all implementations are platform-agnostic and verified on NVIDIA, AMD, and Intel hardware", installs on ROCm with `flash-linear-attention[rocm]` after a ROCm torch, and notes that a bare `pip install flash-linear-attention` does not install torch or Triton. Source: fla-org, "flash-linear-attention" README (vendor doc), https://github.com/fla-org/flash-linear-attention [S31]

  Implies: the bare install is the right one for pod B, where torch 2.9.1+rocm7.13.0 and Triton 3.5.1 come from the image and must not be replaced; gfx1201 is not named, so LR-15 is also the compatibility test.

  Experiment (LR-15, same): as above, plus a 2-step correctness check that the loss with `fla` matches the fallback's loss on the same batch to within 1%. Expected: matches. Minimum sample: 1 batch, both paths.

- **[practitioner]** The ms-swift Qwen3.5 guide installs `flash-linear-attention>=0.4.2` and `causal-conv1d` before training, recommends Python 3.12 because of an `fla` issue, and says "If you encounter slow training issues, please refer to" `fla` issue #758; its Qwen3.5-4B LoRA example uses `--lora_rank 8 --lora_alpha 32 --learning_rate 1e-4 --num_train_epochs 1 --target_modules all-linear --max_length 2048`. Source: ModelScope ms-swift, "Qwen3.5 Best Practices" (vendor doc, v4.4), https://swift.readthedocs.io/en/v4.4/BestPractices/Qwen3_5-Best-Practice.html [S33]

  Implies: a second practitioner treats the fast kernels as a prerequisite for Qwen3.5 training, not an optimisation; its rank 8 and lr 1e-4 sit at the low end of section 2's range.

  Experiment (LR-15 and LR-7, same): the kernel install as above, and 1e-4 included in the learning-rate sweep. Expected: samples per second rises from 0.51 to at least 1.0 with the kernels, and 1e-4 trails 2e-4 by at least 0.05 on the held-out score at equal steps. Minimum sample: 2 smoke runs per kernel setting; 1 run per learning rate, 50 held-out items.

- **[anecdote]** `fla` issue #758 reports that an unused kernel parameter "causes recompilation for different sequence lengths" in `l2norm_fwd_kernel`; it was closed on 2026-03-02 with a commenter's note that it is "Likely fixed by #745". Source: fla-org/flash-linear-attention, GitHub issue #758 (opened 2026-03-01), https://github.com/fla-org/flash-linear-attention/issues/758 [S32]

  Implies: with variable-length examples an older `fla` can recompile Triton kernels on every new length, which would erase the samples-per-second gain; install a release from after March 2026 and watch the per-step times.

  Experiment (LR-16): log per-step time for the first 50 steps of a variable-length run. Expected: step time settles after warm-up; if it keeps spiking on new lengths, set `pad_to_multiple_of=64` and the spikes stop. Minimum sample: 50 steps.

- **[measured]** In torchtune's single-device table (Llama 3.2 3B, bf16, A100, batch 2, sequence 4,096), turning on activation checkpointing cut peak memory from 42.9 to 24.9 GiB (-41.93%) and throughput from 9,174 to 7,210 tokens/s (-21.41%); the LoRA recipe in the same table peaked at 8.5 GiB. Source: PyTorch, "torchtune" README (vendor doc with measured tables), https://github.com/pytorch/torchtune, memory and throughput table [S22]

  Implies: TRL turns gradient checkpointing on by default [S17], so our 0.51 samples/s likely includes a 20%-class throughput tax that buys memory we are not short of (8.7 GB of 32 GB).

  Experiment (LR-17): re-run the smoke with `gradient_checkpointing=False`, after LR-15 so the two effects are separated. Expected: samples/s up 15 to 30% and peak memory up by about 1 to 4 GB (the bf16 weights are most of the 8.7 GB peak at 256 tokens, so activations are small), still far under 32 GB for text; for image batches, keep it on if peak memory exceeds about 28 GB. Minimum sample: 2 runs of 20 steps per setting.

- **[measured]** Chen et al.'s checkpointing algorithm costs "an extra forward pass per mini-batch" and cut a 1,000-layer residual network's memory "from 48G to 7G" with "only 30 percent additional running time cost on ImageNet problems". Source: T. Chen, B. Xu, C. Zhang, C. Guestrin, "Training Deep Nets with Sublinear Memory Cost", 2016, arXiv preprint, https://arxiv.org/abs/1604.06174 (abstract) [S23]

  Implies: the checkpointing cost is bounded by one extra forward pass, so LR-17's expected gain is at most about a quarter to a third of step time.

  Experiment (LR-17, same): the smoke run with `gradient_checkpointing=False`. Expected: step time falls by at most a quarter to a third, so samples per second rises 15 to 30%. Minimum sample: 2 runs of 20 steps per setting.

### Gaps
No source gives `fla` or `causal-conv1d` results on gfx1201 or ROCm 7.13, and none measures the PyTorch fallback's
slowdown in numbers. Whether SDPA uses a fused attention kernel on gfx1201 under torch 2.9.1 was not established
here (only 8 of 32 layers use attention, so it matters less than the DeltaNet path).

## 5. QLoRA on RDNA4

### Takeaway
QLoRA is not needed (bf16 LoRA on a 4B model uses about a third of the card) and Unsloth advises against it for
Qwen3.5 on any hardware. The design doc's claim [R10] of "a known rocBLAS bug" that stalls QLoRA on RDNA4 could not be
traced to a public source; the nearest report is a rocBLASLt library-lookup bug on the R9700 in an inference stack.

### Cited findings
- **[practitioner]** Unsloth: "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models", dense or MoE, because of "higher than normal quantization differences". Source: Unsloth, "Qwen3.5 fine-tuning guide" (vendor doc), https://unsloth.ai/docs/models/qwen3.5/fine-tune [S35]

  Implies: bf16 LoRA stays the only path; QLoRA would also change the base the adapter is trained against, which matters for the merge and the held-out score after quantization (see `packaging_model_card.md`).

  Experiment (LR-18, only if an image batch exceeds the card): keep bf16 and turn gradient checkpointing back on (LR-17 reversed) with batch 2 instead of trying QLoRA. Expected: peak memory falls (by up to about 40% in torchtune's measurement [S22]) at a 20 to 30% cost in samples per second, with the held-out score unchanged. Minimum sample: 20 steps per setting plus one 50-item held-out score.

- **[practitioner]** The bitsandbytes README lists AMD RDNA targets including "gfx120X" on Linux with LLM.int8(), QLoRA 4-bit and 8-bit optimizers all marked supported (development-branch table). Source: bitsandbytes-foundation, "bitsandbytes" README (vendor doc), https://github.com/bitsandbytes-foundation/bitsandbytes [S36]

  Implies: on paper the library supports our card; the reason not to use QLoRA is the model (finding above), not missing hardware support. 8-bit optimizers are available if optimizer state ever binds memory.

  Experiment (LR-19, only if LR-2 shows image batches near the 32 GB limit): switch to `optim="adamw_8bit"` for 20 steps. Expected: peak memory down by about 1 GB on a 4B adapter run (optimizer state is small for LoRA), samples per second and held-out score unchanged. Minimum sample: 20 steps plus one 50-item held-out score.

- **[anecdote]** ROCm issue #7192 reports that on an "AMD Radeon AI PRO R9700, gfx1201" rocBLASLt looks for `TensileLibrary_lazy_gfx1200.dat` and fails on every model load (Ollama with its bundled ROCm, host ROCm 7.2.1); it is open, with a contributor suggesting an Ollama fix. Source: ROCm/rocm-libraries, GitHub issue #7192, "[rocBLASLt] gfx1201 GPU causes lookup of wrong Tensile file (gfx1200.dat)", opened 2026-05-08, https://github.com/ROCm/rocm-libraries/issues/7192 [S37]

  Implies: this is the closest public match to the design doc's "known rocBLAS bug" and it is an inference-stack packaging issue, not a QLoRA failure; the claim should be marked unverified on the design doc rather than used as a reason. It does not touch our metric set.

  Experiment (LR-20, no GPU needed): grep the existing pod B smoke log (`~/smoke_lora.log`, PR #37) for "TensileLibrary" and "gfx1200". Expected: zero matches, which confirms the bf16 training path does not hit this lookup bug and training samples per second is unaffected by it. Minimum sample: the one existing log.

### Gaps
The source for "community runs of LoRA on a 7B vision model use about 17 GB on this card with ROCm 7.2" [R10] (Context:
Learning page) was not found; LR-2 replaces it with our own number for the 4B.

## 6. Loading the student for SFT, the Gated DeltaNet kernel on ROCm, and reproducibility (learning pass 1, 2026-10-10)

Added by the research step of learning pass 1. Each source was opened on 2026-10-10 before it was cited.

- **[practitioner]** TRL's SFT Trainer page says a model given as a string is loaded with the architecture its config names and that "If `dtype` is not specified in `args.model_init_kwargs`, it defaults to `float32`", while a model passed as an object must be a causal LM ("Only causal language models are supported"); for vision-language models it warns that "truncating may remove image tokens" and says to set `max_length=None` unless truncation is verified; and `SFTConfig` defaults to `seed=42`, with `data_seed` and `full_determinism` available. Source: Hugging Face TRL, "SFT Trainer" (vendor doc, main), https://huggingface.co/docs/trl/main/en/sft_trainer, sections "SFTTrainer" (the model parameter), "Training Vision Language Models" and `SFTConfig` [S17]
  - Implication: a trainer that hands TRL the model id to get the image-text class must also pass `model_init_kwargs={"dtype": torch.bfloat16}`, or it trains in float32, which roughly doubles the weights that make up most of the 8.7 GB peak and gives a trainer.samples_per_s that cannot be compared with metrics.jsonl:13. The seed and the data seed go in the run's ledger row (rubric test LT5).
  - Candidate experiment (LR-21): load Qwen3.5-4B through SFTTrainer by id with and without `model_init_kwargs` set to bf16 and print the parameter dtype and the peak memory after 2 steps. Expected movement: trainer.peak_mem_gb roughly doubles without the dtype argument; with it the text-only peak matches the 8.7 GB of metrics.jsonl:14. Minimum sample size: 2 steps per arm, one 15-minute pod B slot.

- **[practitioner]** In transformers main, `use_kernel_func_from_hub_with_fallback` tries "1. Hf kernels (if requested)", "2. Original package", "3. Torch only path", and on the last logs once that the function "is falling back to its reference PyTorch implementation because" the package "is not installed. This is correct but much slower"; the hub mapping for `chunk_gated_delta_rule` lists CUDA and XPU devices and no ROCm device. Source: huggingface/transformers, `src/transformers/integrations/hub_kernels.py::use_kernel_func_from_hub_with_fallback` and its kernel mapping (source code, main), https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py [S29]
  - Implication: on gfx1201 the hub has no Gated DeltaNet kernel to offer, so the only fast path is the `fla` package itself, which `lab/pod_b_trainer.sh deps` does not install (its line 19 installs peft, trl, transformers and the rest, not flash-linear-attention). The 0.51 samples/s of trainer.samples_per_s (metrics.jsonl:13) was most likely measured on the torch fallback for 24 of the 32 layers.
  - Experiment: LR-15, as specified in section 4, queued on T0 by learning pass 1.

- **[practitioner]** PyTorch's reproducibility notes say "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms" and that "results may not be reproducible between CPU and GPU executions, even when using identical seeds". Source: PyTorch 2.14 documentation, "Reproducibility" (vendor doc), https://docs.pytorch.org/docs/2.14/notes/randomness.html [S64]
  - Implication: a run is reproducible from its ledger row only within measured noise, and only if the row carries the torch, ROCm, PEFT, TRL and transformers versions and the seeds (rubric test LT5); a re-run on another torch build starts a new trainer.samples_per_s baseline rather than giving an after.
  - Experiment: EN-12, as specified in `eval_noise.md` section 4 (three seeds of one recipe), with every run's versions recorded.

## Search log

- 2026-10-10, learning pass 1: the search that added section 6 is logged in `eval_noise.md`, section "Search log".
