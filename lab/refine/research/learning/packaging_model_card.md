# Merging the adapter, GGUF with the vision projector, Q4 quantization, runtime LoRA, and the model card

Research note for the learning track, opened 2026-10-10 (all sources opened that day; index in `SOURCES.md`).
It covers the path from a trained adapter to something the Jetson runs and the Hub hosts: merge into the base,
convert to GGUF, keep the vision projector, quantize to Q4_K_M, or ship the adapter itself; then the model card and
license. Labels: `measured` = a number someone measured, with the setup; `practitioner` = a documented
recommendation or documented behaviour from the people who build or run the thing (papers' recommendation sections,
official docs, model cards, source code, maintainers); `anecdote` = a single report without data; `opinion` =
argument without evidence. Every finding carries its citation and a `[S..]` or `[R..]` key into `SOURCES.md`. Under
each finding, "Implies" maps it onto our metric set and "Experiment" proposes a test with the expected movement and
the minimum sample. Our metric set is the learning rubric's [R7] (`lab/refine/learning/RUBRIC.md`, "The metric set"):
held-out score before, after and teacher (`learning.heldout.current`, `learning.heldout.candidate`,
`learning.heldout.teacher`); label agreement rate (`learning.label_agreement`); loop wall-clock
(`learning.loop_minutes`); training samples per second (`trainer.samples_per_s`, 0.51 on pod B, PR #37);
package size (`learning.package_mb`).

Our setup: the Jetson runs `Qwen/Qwen3.5-4B` as Q4_K_M plus `mmproj-F16` from `unsloth/Qwen3.5-4B-GGUF` under
llama-server (`--jinja -c 4096 -ngl 99 -np 1 --reasoning-budget 0`), with about 2.0 GB of RAM left under load [R4]. A
model package is `manifest.json` (`ModelManifest`: version, base, task, `gguf`, optional `mmproj`, sha256 per file,
scores) plus the files [R5].

## 1. Merging the adapter

### Takeaway
Merge into the bf16 base, never into a quantized base, and treat the merged model as a new artefact to be checked
against base plus adapter before conversion.

### Cited findings
- **[practitioner]** PEFT's guide: `merge_and_unload()` merges the adapter so the result is "a standalone model", is "not an in-place operation" (the returned model must be used), and `merge_adapter()` / `unmerge_adapter()` keep a reversible copy; "On a quantized model, unmerging is not exact: merging re-quantizes the base weight, and repeated merge and unmerge cycles can accumulate rounding error"; and if only some active adapters are merged, "the adapters that are not merged are silently not applied". Source: Hugging Face PEFT, "LoRA" developer guide (vendor doc), https://huggingface.co/docs/peft/main/en/developer_guides/lora, section "Merge LoRA weights into the base model" [S16]

  Implies: the package step loads the bf16 base, applies the one adapter, merges, and saves in bf16 (about 8.4 GB for this model [S56]) before conversion; any quantized merge would change the model the held-out score was measured on.

  Experiment (PK-1): before conversion, compare the merged bf16 model with base plus unmerged adapter on 20 held-out prompts at greedy decoding. Expected: identical outputs on all 20 and a maximum absolute logit difference under 1e-2. Minimum sample: 20 prompts, pod B, a few minutes.

### Gaps
No source opened reports merge problems specific to Qwen3.5's linear-attention layers; PK-1 checks it on our model.

## 2. llama.cpp support for Qwen3.5 and the vision projector

### Takeaway
llama.cpp has supported Qwen3.5 text and vision since February 2026, with the text model and the projector as
separate GGUF files. If the adapter only touches language layers, the vision weights are unchanged and the base
projector can ship untouched; only the text GGUF changes.

### Cited findings
- **[practitioner]** llama.cpp PR #19435, "[Model] Qwen3.5 dense and MoE support (no vision)", merged on 2026-02-08 and was reverted the next day (PR #19453); its author wrote "For now, only add text support, without the multimodal capabilities". Source: ggml-org/llama.cpp, GitHub PR #19435, https://github.com/ggml-org/llama.cpp/pull/19435 [S50]

  Implies: a llama.cpp build from that window can load Qwen3.5 text but not its vision path; the converter and server builds on pod B and on the Jetson must both come from after the follow-up below, or the desk-object held-out score cannot be measured on the shipped format.

  Experiment (PK-2): record the llama.cpp commit used on pod B (shallow clone of master [R3]) and on the Jetson [R4] in each run's ledger row. Expected: both at or after the 2026-02-10 merge of PR #19468. Minimum sample: one check per node, no GPU.

- **[practitioner]** llama.cpp PR #19468, "[MODEL] support qwen3.5 series", merged on 2026-02-10, was "verified with preview checkpoints from the Qwen Team in both vision and pure text modes" and "with or without mmproj file"; a contributor in the thread noted "deepstack was removed temporally", and a maintainer planned a follow-up to "deduplicate and optimize the delta net stuff". Source: ggml-org/llama.cpp, GitHub PR #19468 (merged), https://github.com/ggml-org/llama.cpp/pull/19468 [S51]

  Implies: Qwen3.5 vision runs in llama.cpp; the removed deepstack path does not affect our model, whose `deepstack_visual_indexes` is empty [S25]. The held-out score measured through llama-server is a fair proxy for the bf16 model only after PK-4 below confirms it.

  Experiment (PK-3): convert the unmodified base checkpoint on pod B (`convert_hf_to_gguf.py`, text and `--mmproj`) and load it with llama-server. Expected: it loads, and its tensor list matches the unsloth GGUF the Jetson runs (same names and shapes). Minimum sample: 1 conversion. This also answers the design doc's open item about the GGUF round trip with the vision projector [R10].

- **[practitioner]** llama.cpp's multimodal README says "you can use `convert_hf_to_gguf.py` with `--mmproj` flag to get the `mmproj` file", describes the file as the part that "handles the image encoding and projection", and warns that multimodal support is under "very heavy development" with "breaking changes are expected"; the multimodal doc adds "Use `-m model.gguf` option with `--mmproj file.gguf` to specify text and multimodal projector respectively". Source: ggml-org/llama.cpp, `tools/mtmd/README.md` and `docs/multimodal.md` (vendor docs), https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md and https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md [S52] [S53]

  Implies: with a language-only adapter (see `lora_recipe.md`, LR-3) the package needs a new text GGUF only, and `ModelManifest.mmproj` can point at the base projector with its existing sha256, so package size and transfer time over the hub proxy shrink by the projector's size. A vision-side adapter forces a new projector and a re-check of image behaviour on the Jetson.

  Experiment (PK-4): after a language-only run, regenerate the mmproj from the merged checkpoint and from the base checkpoint with the same converter commit and compare sha256. Expected: identical hashes (no vision weight changed); if they differ, compare tensor values and expect a maximum difference of 0. Minimum sample: 2 conversions, no GPU needed beyond pod B's CPU.

### Gaps
None of these pages documents Qwen3.5's mmproj conversion or its image-token defaults; the Jetson build warns that
Qwen-VL grounding wants about 1024 image tokens (`--image-min-tokens 1024`) [R4], which bears on the desk-object
held-out score and was not researched here.

## 3. What Q4_K_M costs, and how to measure it

### Takeaway
Q4_K_M is about 4.9 bits per weight. Merging a fine-tune and then quantizing it to 4 bits has cost 2 to 3 points of
a fine-tune's gain in a controlled comparison, while a quantization-aware fine-tune lost nothing. The deploy gate must
therefore score the Q4_K_M GGUF that ships, and keep the bf16 score beside it to see what quantization took away.

### Cited findings
- **[measured]** llama.cpp's quantize README, measured on Llama-3.1-8B: Q4_K_M at 4.8944 bits per weight, 4.58 GiB and 71.93 t/s generation, Q8_0 at 8.5008 bits, 7.95 GiB and 50.93 t/s, F16 at 16.0005 bits, 14.96 GiB and 29.17 t/s. Source: ggml-org/llama.cpp, `tools/quantize/README.md` (vendor doc with a measured table), https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md [S54]

  Implies: package size and decode speed scale with bits per weight; Q4_K_M keeps the student at about 30% of its bf16 size, which the Jetson's 2.0 GB of free RAM under load requires [R4].

  Experiment (PK-5): record package size (text GGUF, mmproj, manifest) for every packaged run. Expected: the text GGUF within 1% of the base Q4_K_M's 2.74 GB [S56] because the architecture does not change. Minimum sample: every run.

- **[practitioner]** The same README says quantization accuracy loss "can be minimized by using a suitable imatrix file" and offers per-tensor overrides: `--leave-output-tensor` ("may also increase quality, especially when requantizing"), `--output-tensor-type`, `--token-embedding-type` and `--tensor-type` with regex support. Source: ggml-org/llama.cpp, `tools/quantize/README.md` (vendor doc), options list, https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md [S54]

  Implies: an importance matrix computed on our own task prompts is the cheap lever for keeping the fine-tune's held-out gain through quantization, at no change in package size.

  Experiment (PK-6): quantize the merged model to Q4_K_M with and without an imatrix computed from 100 training prompts and score both GGUFs on the held-out set. Expected: the imatrix arm scores 0 to 0.05 higher, package size unchanged. Minimum sample: 2 quantizations, 50 held-out items each.

- **[measured]** QA-LoRA's Table 1 (LLaMA-7B, 4-bit, 5-shot MMLU) gives 38.4 (Alpaca) and 44.3 (FLAN v2) for QLoRA evaluated in FP16, but 36.0 and 41.4 for "QLoRA w/ GPTQ", the same fine-tune merged and then quantized to 4 bits after training; the authors attribute it to "the post-training quantization will not be compensated, resulting in unstable results". Source: Y. Xu et al., "QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models", 2023, arXiv preprint, https://arxiv.org/abs/2309.14717, section 4.2, Table 1 [S55]

  Implies: merge-then-quantize, which is our package path, can give back 2 to 3 points of the held-out gain; a gate that scores the bf16 merge would overstate the deployed model.

  Experiment (PK-7): score the candidate on the held-out set as bf16 (pod B) and as the Q4_K_M GGUF (llama-server), and record both as dotted keys (for example `candidate.bf16` and `candidate`). Expected: GGUF at or below bf16 by 0 to 0.05; the gate decides on the GGUF number. Minimum sample: 50 held-out items, both runtimes.

- **[measured]** TinyAgent quantized its fine-tuned models to 4 bits (group size 32) and accuracy did not drop: 80.06% to 80.35% for the 1.1B and 84.95% to 85.14% for the 7B, with about 30% lower latency and a 4x smaller model, which the authors credit to "the additional fine-tuning with simulated quantization". Source: L. E. Erdogan et al., "TinyAgent: Function Calling at the Edge", 2024, EMNLP 2024 (demo), https://arxiv.org/abs/2409.00608, section 3.5, Table 2 [S44]

  Implies: the quantization loss in PK-7 is avoidable with a quantization-aware step, but that step is not in our recipe; without it QA-LoRA's 2 to 3 points is the better prior for the held-out score.

  Experiment (PK-7, same): the bf16 versus GGUF comparison. Expected: a loss of 0 to 0.05; only a loss above the minimum detectable effect would justify adding quantization-aware training. Minimum sample: 50 held-out items.

- **[measured]** The `unsloth/Qwen3.5-4B-GGUF` repository lists Q4_K_M at 2.74 GB, Q4_K_S at 2.59 GB, Q5_K_M at 3.14 GB, Q8_0 at 4.48 GB and BF16 at 8.42 GB for this model. Source: Unsloth, "unsloth/Qwen3.5-4B-GGUF" model card (vendor doc), https://huggingface.co/unsloth/Qwen3.5-4B-GGUF [S56]

  Implies: these are the package-size baselines for the text GGUF; Q8_0 is 1.74 GB larger than Q4_K_M, which would use almost all of the 2.0 GB the Jetson has free under load [R4], so Q4_K_M (or Q4_K_S) is the only practical tier, and any quality loss must be recovered with PK-6, not with more bits.

  Experiment (PK-8): if PK-7 shows a quantization loss above the minimum detectable effect, keep the output and embedding tensors at Q8_0 with `--output-tensor-type` and `--token-embedding-type`. Expected: package size up by roughly 0.2 to 0.4 GB and part of the held-out loss recovered. Minimum sample: 1 quantization, 50 held-out items, plus a Jetson RAM check under load.

### Gaps
No source opened measures Q4_K_M's quality loss on Qwen3.5 or on a vision-language task; PK-7 measures it on ours.
The README's measured table has no perplexity or KL column, so bits per weight is the only quality proxy it gives.

## 4. Shipping the adapter instead of a merged model

### Takeaway
llama-server can load LoRA adapters on top of a quantized base and change their scale at runtime, and llama.cpp
ships a converter from a PEFT adapter to GGUF. That would shrink each package to the adapter and make rollback a
scale change, at the cost of serving a different numerical model from the merged-and-quantized one.

### Cited findings
- **[practitioner]** llama.cpp ships `convert_lora_to_gguf.py`, described as "Convert a Hugging Face PEFT LoRA adapter to a GGUF file", which reads `adapter_config.json`, takes the base model's config from `--base`, `--base-model-id` or the adapter config, offers `--outtype` f32, f16, bf16, q8_0 or auto, and reuses the converter's model definitions, exiting if the architecture is not supported. Source: ggml-org/llama.cpp, `convert_lora_to_gguf.py` (source code, master), https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py [S60]

  Implies: an adapter-only package is technically available for our model if its architecture is in the converter's model list; package size would drop from 2.74 GB to the adapter's size.

  Experiment (PK-9): convert the first trained adapter with `--outtype bf16` and record its size and whether the converter accepts the Qwen3.5 architecture. Expected: it converts, and the adapter GGUF is under 5% of the merged Q4_K_M's size. Minimum sample: 1 conversion, CPU only.

- **[practitioner]** The llama-server README documents `--lora FNAME`, `--lora-scaled FNAME:SCALE`, `--lora-init-without-apply` (load adapters and "apply later via POST /lora-adapters"), a `GET /lora-adapters` and `POST /lora-adapters` pair that sets each adapter's global scale, and a per-request `lora` field, and warns that "requests with different LoRA configurations will not be batched together, which may result in performance degradation". Source: ggml-org/llama.cpp, `tools/server/README.md` (vendor doc), sections on LoRA flags and `/lora-adapters`, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md [S09]

  Implies: on the Jetson (one slot, `-np 1`) the batching caveat does not bite, and a runtime adapter makes rollback a `POST` that sets the scale to 0 instead of a model swap; the held-out score must be measured in that serving form, because a higher-precision adapter on a Q4 base is not the same model as a merged Q4.

  Experiment (PK-10): serve base Q4_K_M plus the converted adapter with `--lora` and compare held-out score and decode tokens per second with the merged Q4_K_M. Expected: held-out score within 0.05 of the merged model, decode 5 to 15% slower, package size from 2.74 GB to tens of MB. Minimum sample: 50 held-out items and 10 timed 100-token decodes per arm.

### Gaps
No source opened says whether llama.cpp's LoRA path covers Qwen3.5's linear-attention projections or the vision
encoder; PK-9 and PK-10 test it. Our `ModelManifest` has no field for an adapter file [R5], so an adapter-only
package would need an additive contract field; that is a decision for the contract owners, not this note.

## 5. The model card and the license

### Takeaway
The card needs the task, the base model, the before, after and teacher scores with their sample size and interval,
the intended use and limits, the training setup, and the license, and it must not carry user data. The base is
Apache 2.0, which obliges a copy of the license and a notice of changes when the adapter or merged weights are
redistributed.

### Cited findings
- **[practitioner]** Model cards are "short documents accompanying trained machine learning models" that "provide benchmarked evaluation in a variety of conditions", "disclose the context in which models are intended to be used", and include "details of the performance evaluation procedures". Source: M. Mitchell et al., "Model Cards for Model Reporting", 2019, FAT* 2019, doi:10.1145/3287560.3287596, https://arxiv.org/abs/1810.03993 (abstract) [S57]

  Implies: the card reports the held-out score as the eval gate measured it (n, paired difference, interval; see `eval_noise.md`), split by condition where we can (for example per camera or lighting for desk objects), and states the intended use: one household's own table, served on the Jetson.

  Experiment (PK-11): draft the card from the first gated run and check it against the paper's sections (intended use, factors, metrics, evaluation data, training data, caveats). Expected: every section filled from the run's ledger row with no hand-typed numbers. Minimum sample: 1 card.

- **[practitioner]** The Hub's model-card docs say a card should describe the model, its intended uses and limitations, training parameters, datasets and evaluation results; its YAML carries `license`, `datasets` and `base_model`, the Hub infers the relation to the base ("adapter", "merge", "quantized", "finetune") or accepts an explicit `base_model_relation`, evaluation results go in a `model-index` block (or a newer eval-results format), and `library_name` should be set explicitly. Source: Hugging Face Hub docs, "Model Cards" (vendor doc), https://huggingface.co/docs/hub/model-cards, sections "Model card metadata", "Specifying a base model", "Evaluation Results" [S58]

  Implies: the adapter repo's card sets `base_model: Qwen/Qwen3.5-4B`, `library_name: peft`, `license: apache-2.0` and the held-out scores in `model-index`; a separate GGUF repo would declare `base_model_relation: quantized`. No `datasets` entry is published, because training data never leaves the box.

  Experiment (PK-12): lint the drafted card and `adapter_config.json` before upload: required YAML keys present, and no `frame_ref`, file path under `data/`, person name or camera id anywhere in the repo files. Expected: 0 missing keys and 0 matches. Minimum sample: every file in the upload, once per publish.

- **[practitioner]** Apache License 2.0, section 4: redistributors "must give any other recipients of the Work or Derivative Works a copy of this License", "must cause any modified files to carry prominent notices stating that You changed the files", must retain the original copyright, patent, trademark and attribution notices, and must carry forward a NOTICE file's attributions if the work includes one. Source: Apache Software Foundation, "Apache License, Version 2.0", section 4 (license text), https://www.apache.org/licenses/LICENSE-2.0 [S59]

  Implies: the published adapter (and any merged GGUF) ships with the license text and a line in the card stating what we changed (a LoRA fine-tune on our task); this does not move a metric but is a pass/fail condition of the card test in the rubric.

  Experiment (PK-13): check the upload for a LICENSE file, a "changes" statement and any NOTICE file carried from the base repository. Expected: all present before the first publish. Minimum sample: 1 upload.

- **[practitioner]** The `Qwen/Qwen3.5-4B` model card's metadata gives the license as `apache-2.0`, and so does the teacher's, `Qwen/Qwen3.8-27B`; the teacher's card says nothing either way about training other models on its outputs. Source: Qwen, "Qwen/Qwen3.5-4B" and "Qwen/Qwen3.8-27B" model cards (vendor docs), https://huggingface.co/Qwen/Qwen3.5-4B and https://huggingface.co/Qwen/Qwen3.8-27B, metadata [S24] [S61]

  Implies: publishing adapters on this base is permitted under Apache 2.0 terms, which clears the design doc's "verify license before publishing adapters" note [R10] for the 4B, and the teacher whose outputs the adapter learns is under the same license; the card should name both.

  Experiment (PK-14): record the base's and the teacher's license and Hub revision in each run's ledger row and print both in the card. Expected: Apache 2.0 for both on every run; a change in either card's license between runs blocks the publish. Minimum sample: 1 check per publish.

### Gaps
The Hub's newer eval-results format was linked but not opened. No source opened addresses whether training on a
model's outputs carries conditions beyond its license; the teacher's card is silent on it.
