# L16: the config of the first real fine-tune on pod B, and where each number comes from

The run: LoRA SFT of the student, `Qwen/Qwen3.5-4B`, with `lab/train.py` (L3, PR #252) on the trainer pod, pod B's
role, held by the SJC Strix Halo pod on account D (Radeon 8060S, gfx1151, ROCm 7.13, torch 2.9.1+rocm7.13.0 from the
pod image, 64 GB shared). It is a text task: the tool-calling rows. `lab/l16_run.sh` holds the exact commands. The
results (loss start and end, samples per second, peak memory, wall-clock) are in the run's ledger row and in
`fixtures/demo/`; this page is what was chosen before the run and why.

## The data

| Item | Value | Why |
| --- | --- | --- |
| Labels | L13's teacher labels: `fixtures/learning/tool-calling-labels/` on main (PR 283, merge commit d8dbc68), Qwen3.8-27B-INT4 on pod C, 142 labeled, 0 rejects | the plan's labels (Context: Learning). The workstation copies only `manifest.json`, `examples.jsonl`, `split.json` and `images/` (`bash lab/l16_run.sh dataset`; `L16_LABELS` points it at another copy); the labeler's traces and ledger stay off the pod. Without L13's folder, `dataset` falls back to the reference traces of `expected.jsonl`, which learning ruling 12 allows; this run did not need it |
| Dataset | `tool-calling-fixture` (action `fixture:golden-questions`), 142 examples; `examples.jsonl` sha256 `6ae7fb9e0b19521b860f3503fb05a7c77a8946890905840091bfb9e47f855f0f` | the no-camera fixture (L11) with the teacher's targets |
| Split | the frozen `split.json`, sha256 `59227ead6c24d0b7b525ba99c6e2c06f1f7178d2c6f20bd8d556bc16d2c3de39`: 84 train in 12 groups, 58 held out in 4 groups | L15's split; the trainer reads membership only through `load_split()` and refuses any held-out row |
| Rows | 84, from `lab/format_traces.py --split --tokenizer` at template revision 851bf6e; sha256 `628bd220e2b805bb82b24662473a4c6c518dcdc6144752db6bff3a592f730c58` | every trace checked against the real tokenizer and template; 0 rejected, 0 failed, 58 held out skipped |
| Lengths | 577 to 911 tokens a row, 58,333 in all (mean 694); 8,775 tokens carry the loss (two assistant turns a row) | measured with the student's tokenizer at revision 851bf6e |
| `max_length` | none | nothing is truncated (the trainer refuses a cut); the longest row is 911 tokens. TRL's own default of 1024 would also hold it, but the learning note's LR-9 asks for the measured maximum or none (`lora_recipe.md` section 2, TRL SFT Trainer docs [S17]) |

## The model

| Item | Value | Why |
| --- | --- | --- |
| Base | `Qwen/Qwen3.5-4B` at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, in `~/models/Qwen3.5-4B` on the pod | the student the Jetson runs; the same commit as `TEMPLATE_REVISION` in `lab/format_traces.py`, so the rows were rendered with this checkpoint's own chat template. `hf download --local-dir` records the commit in the folder and the trainer puts it in its row (`refs.base_revision`) |
| Class and dtype | `Qwen3_5ForCausalLM` (text-only), bfloat16 | text rows; the trainer loads the weights in bf16 itself and checks it (`lora_recipe.md` section 6, [S17]: TRL would load float32 otherwise) |

## The training config

These are R36's pinned baseline for this step on gfx1151 (`lab/refine/research/kernels/settings_training_gfx1151.md`
section 13), which `lab/train.py` uses by default; `l16_run.sh` writes each one on the command line anyway.

| Setting | Value | Source |
| --- | --- | --- |
| LoRA rank, alpha, dropout | r 16, alpha 32, dropout 0 | Settings sheet section 13 (pinned). `lora_recipe.md` section 2: with LoRA on all layers rank barely matters at a few hundred examples (Hu et al. Table 6 [S10]; QLoRA Appendix A.1, "LoRA r is unrelated to final performance if LoRA is used on all layers" [S14]); alpha = 2r is Biderman et al.'s measured-good convention [S13]; the Tinker primer's rule, LoRA parameters at least the completion tokens [S19], holds by orders of magnitude (r 16 on every linear layer of 32 layers is tens of millions of parameters, against 8,775 loss tokens an epoch) |
| Target modules | `all-linear` on the language model (PEFT leaves out the LM head) | `lora_recipe.md` section 2: covering all linear layers, MLPs included, matters more than rank (QLoRA [S14], "LoRA Without Regret" [S12]) |
| Learning rate | 2e-4, TRL's default linear decay, no warmup | `lora_recipe.md` section 2: adapters want about 10 times the full fine-tuning rate (Thinking Machines [S12]), 2e-4 is QLoRA's rate for 7B and 13B [S14], Biderman et al.'s best for math [S13] and Unsloth's starting point [S18]; settings sheet section 13 pins it. LR-7's sweep (1e-4, 2e-4, 5e-4) is still to run; this run is its 2e-4 arm |
| Epochs and steps | 3 epochs = 18 optimizer steps (84 rows are 21 micro-batches of 4 an epoch; accumulation 4 gives 6 optimizer steps an epoch, the last holding one micro-batch); 252 examples trained, 288 by the trainer's examples-seen count | `lora_recipe.md` section 2: Unsloth's "1-3 epochs", "more than 3 epochs offers diminishing returns" [S18], TRL's default of 3 epochs [S17]; LR-10 expects the held-out score to peak at epoch 2 or 3. The trainer's `trainer.samples_per_s` counts steps x 4 x 4 (metrics.jsonl:13's measure), which is 288 here |
| Batch | 4 per device, gradient accumulation 4, effective batch 16 | Settings sheet sections 9 and 13: the search space holds the effective batch at 16 (Qwen's own example) and searches only the split; Unsloth recommends an effective 16 [S18]; LoRA is less tolerant of large batches [S12] |
| Seed | 42, data seed 42 | Settings sheet section 13 (pinned); TRL's default seed [S17]. Both go in the row (rubric LT5) |
| Gradient checkpointing | on, `use_reentrant` False | Settings sheet section 13 (pinned baseline; turning it off is TS12, a later experiment) |
| Attention, optimizer, loader | SDPA; fused AdamW; 2 forkserver workers, pinned memory, not persistent | Settings sheet sections 5, 10, 11 and 13 (pinned, outside the search space) |
| Packing, length grouping, torch.compile | off | Settings sheet sections 9 and 12: packing is incorrect on this stack (SDPA, torch Gated DeltaNet path); the others are later experiments (TS14, TS16) |
| Environment | `MIOPEN_FIND_MODE=NORMAL`, every other variable the sheet pins unset | Settings sheet section 13; the trainer exports it before torch is imported and records it as `refs.env` |
| Kernels | Gated DeltaNet and causal conv1d on transformers' torch fallback (no `flash-linear-attention` in the image) | Settings sheet section 13 (`gdn_kernel` torch, `causal_conv1d_kernel` torch); TS7 (fla) is the first speed experiment after this baseline |
| Loss head | TRL 1.15's Triton loss kernel; `--torch-loss-head` only if Triton cannot build it on gfx1151 | `lab/train.py`; the row's `refs.loss_head` says which ran |
| Power | no sampler: `socket_w` is null in the step events | `lab/pod_power.sh` on pod D waits for Ilan's go (its header); not part of L16's Done when |

## Predicted wall-clock

No training step has been measured on gfx1151 yet: the settings sheet ran nothing on a pod (its header and section
1; TS1, the pinned-baseline run, has not run), and the R9700's 0.51 samples/s at 256 tokens (metrics.jsonl:13) is
another chip, which section 1 uses only to bound memory. So the prediction is derived, and the run measures it:

- Scale the R9700's smoke step to these rows: 694 tokens a row on average against 256 is 2.7 times the tokens, and
  the torch Gated DeltaNet fallback (sheet section 6) walks the sequence in chunks, so time grows with length:
  about 0.19 samples/s on the R9700 (derived: 0.51 / 2.7).
- The sheet gives no gfx1151 to gfx1201 ratio, and the learning note reads the R9700's 0.51 as bound by the torch
  fallback's many small kernels rather than by matmul throughput (`lora_recipe.md` section 4), so a ratio of peak
  numbers would not carry over either. Assumed: the 8060S runs this step one to three times slower than the R9700,
  0.06 to 0.19 samples/s.
- 252 examples at that rate: 22 to 70 minutes of steps, plus about 2 minutes to load 9 GB of weights and build the
  first kernels. The low half fits the slot; the high half does not.

What the run does about it: every optimizer step's `step_ms` is in the log within two minutes of the start. If the
first steps put 18 steps past about 35 minutes, the run is stopped and started again with `--max-steps` set to what
fits (the dataset is pushed again, since the trainer deletes its copy whenever it ends), and the row records
`max_steps`. `l16_run.sh` also stops the trainer with SIGINT after 40 minutes (`TRAIN_TIMEOUT_S`), so the slot is
never overrun (prep, about 5 minutes for the venv and the 9 GB download, also runs inside the slot: nothing is
installed on pod D before the slot starts); an interrupted run saves no adapter and its row says `interrupted`.

Peak memory, from the same R9700 smoke run (8.7 GB at batch 4 and 256 tokens, sheet section 1) with the activations
of 2.7 times the tokens under gradient checkpointing: roughly 10 to 16 GB allocated, well inside the pod's share.

## Reproducing the run

On the workstation, from the repo root (Git Bash):

```
TOKENIZER=<folder with the student's tokenizer.json and chat_template.jinja at 851bf6e> bash lab/l16_run.sh dataset
bash lab/l16_run.sh bundle
bash lab/l16_run.sh dry-run            # optional: the same command line on the CPU, tiny fixture, tiny model
python -m lab.transfer push-file --account D ../L16-data/l16-bundle.tar.gz l16/l16-bundle.tar.gz
python -m lab slot request --task L16 --minutes 60 --kind lora --what "..." --why "..." --measured "..."
python -m lab slot start --row <row> --reply "..."
python -m lab.controller run-script --account D --timeout 1200 lab/l16_run.sh prep
python -m lab.transfer push-dataset --account D ../L16-data/tool-calling-fixture --to l16/datasets
python -m lab.transfer push-file --account D ../L16-data/rows.jsonl l16/datasets/rows.jsonl
python -m lab.controller run-script --account D --timeout 3300 lab/l16_run.sh train <run id> tool-calling-fixture
python -m lab.transfer pull-file --account D adapters/l16-<run id>/run.json fixtures/demo/l16-run.json
python -m lab slot done --row <row> --result "..."
```

The run is reproducible from its ledger row: the row holds the base repo and commit, the sha256 of
`examples.jsonl`, of `split.json` and of the rows, the seed and data seed, the library versions, and the full
LoraConfig and SFTConfig as JSON text. The same bundle commit, the same rows file (its sha256 checks it) and the same
command line give the same run up to the run-to-run noise PyTorch allows on one platform ([S64], `lora_recipe.md`
section 6).

## What the run measured (2026-10-10, 11:22 to 11:35 PM ET; 03:22 to 03:35 UTC on 2026-10-11)

Run `train-20261011T032149Z-7dd098`, bundle commit c81a659, slot row `3f696ef5fec5816e90d1eb5c9350c669` (started 11:19 PM
ET, done 11:36 PM ET). The trainer's row (verb `train`, outcome `trained`) and the files are in `fixtures/demo/`:
`l16-ledger.jsonl` (the pod's `train-install` and `train` rows and the workstation's `train-clip` row),
`l16-run.json`, `l16-telemetry.jsonl` (18 `telemetry.train.step` events), `l16-adapter_config.json`, `l16-train.log`,
`l16-loss.png`, `l16-train-clip.mp4` (the log scrolling, 2 fps, 807 s) and `l16-train-clip-10x.mp4` (the same, ten
times faster).

| Number | Value |
| --- | --- |
| Loss, first and last step | 0.1273 to 0.0095 (mean over the run 0.0544) |
| `trainer.samples_per_s` | 0.367 (288 by the trainer's count over 784.5 s; 252 examples actually trained, 0.321 a second) |
| Peak memory (`max_memory_allocated`) | 11.1 GiB of the 64 GiB the GPU reports |
| `trainer.train_s` | 784.5 s; the script's wall-clock 792 s; prep inside the slot 106 s (the pinned install 12 s, the base download 81 s) |
| Steps, tokens | 18 optimizer steps, 174,999 tokens seen, 8,775 loss tokens an epoch |
| Trainable parameters | 32,464,896 LoRA weights on a 4,205,751,296-parameter base |
| Base | `Qwen/Qwen3.5-4B` at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| Kernels | TRL's Triton loss kernel built and ran on gfx1151; Gated DeltaNet on the torch fallback |
| Power | none recorded: no sampler ran on pod D |
| Data | the dataset copy and the rows were deleted on the pod (`refs.data_deleted` yes) |

Against the prediction: 0.32 examples a second is above the predicted 0.06 to 0.19, so the
Strix Halo ran this step faster than the scaled R9700 figure, and 18 steps took 13 minutes instead of 22 to 70. The
first loss is already low (0.127, mean token accuracy 0.966): the base nearly writes the teacher's tool-call format
before training, so a lower training loss says little about the held-out score, which is L17's job (LIMA's warning,
`lora_recipe.md` section 2).
