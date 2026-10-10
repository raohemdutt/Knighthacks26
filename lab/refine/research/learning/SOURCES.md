# Sources for the learning research notes

Index of every source cited in the notes in this folder: `eval_noise.md`, `lora_recipe.md`,
`distillation_tool_calls.md` and `packaging_model_card.md`. Each source was opened on the date shown (the page
itself, not a search snippet) before it was cited. Keys are stable: a note cites `[S07]`, and a pass that relies on
a finding cites the note's file and heading. Venues are given only where the opened page states them.
The search ran on 2026-10-10 from about 6:42 to 7:12 AM ET (one research agent in parallel for the LoRA
hyperparameter sources, each of its findings re-opened before citing), well inside the two-hour bound, and stopped
when the last searches only returned sources already listed or nothing new.

## Papers, lab write-ups and vendor documentation

| Key | Source | Year | Type / venue | URL or DOI | Opened | Cited in |
|---|---|---|---|---|---|---|
| S01 | E. Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations" | 2024 | arXiv preprint | https://arxiv.org/abs/2411.00640 | 2026-10-10 | eval_noise |
| S02 | S. Bowyer, L. Aitchison, D. R. Ivanova, "Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints" | 2025 | ICML 2025, PMLR 267:81143-81184 | https://arxiv.org/abs/2503.01747 | 2026-10-10 | eval_noise |
| S03 | T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms" | 1998 | Neural Computation 10(7):1895-1923 | doi:10.1162/089976698300017197 (abstract at https://mlanthology.org/neco/1998/dietterich1998neco-approximate/) | 2026-10-10 | eval_noise |
| S04 | D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, D. Jurafsky, "With Little Power Comes Great Responsibility" | 2020 | EMNLP 2020, pp. 9263-9274 | doi:10.18653/v1/2020.emnlp-main.745 (https://aclanthology.org/2020.emnlp-main.745/) | 2026-10-10 | eval_noise |
| S05 | H. He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" | 2025 | Lab write-up with data (Connectionism) | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | 2026-10-10 | eval_noise |
| S06 | B. Atil et al., "Non-Determinism of 'Deterministic' LLM Settings" | 2024 | arXiv preprint (rev. 2025) | https://arxiv.org/abs/2408.04667 | 2026-10-10 | eval_noise |
| S07 | D. Heineman et al., "Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation" | 2025 | arXiv preprint | https://arxiv.org/abs/2508.13144 | 2026-10-10 | eval_noise |
| S08 | J. Dodge, G. Ilharco, R. Schwartz, A. Farhadi, H. Hajishirzi, N. Smith, "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping" | 2020 | arXiv preprint | https://arxiv.org/abs/2002.06305 | 2026-10-10 | eval_noise |
| S09 | ggml-org/llama.cpp, `tools/server/README.md` (`cache_prompt`, `seed`, `temperature`, LoRA flags, `/lora-adapters`) | 2026 | Vendor doc | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md | 2026-10-10 | eval_noise, packaging_model_card |
| S10 | E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" | 2021 | arXiv preprint | https://arxiv.org/abs/2106.09685 | 2026-10-10 | lora_recipe |
| S11 | D. Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA" | 2023 | arXiv preprint | https://arxiv.org/abs/2312.03732 | 2026-10-10 | lora_recipe |
| S12 | J. Schulman and Thinking Machines Lab, "LoRA Without Regret" | 2025 | Lab write-up with data (Connectionism) | doi:10.64434/tml.20250929 (https://thinkingmachines.ai/blog/lora/) | 2026-10-10 | lora_recipe |
| S13 | D. Biderman et al., "LoRA Learns Less and Forgets Less" | 2024 | TMLR, August 2024 (Featured Certification) | https://arxiv.org/abs/2405.09673 | 2026-10-10 | lora_recipe |
| S14 | T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs" | 2023 | NeurIPS 2023 | https://arxiv.org/abs/2305.14314 | 2026-10-10 | lora_recipe |
| S15 | C. Zhou et al., "LIMA: Less Is More for Alignment" | 2023 | arXiv preprint | https://arxiv.org/abs/2305.11206 | 2026-10-10 | lora_recipe, distillation_tool_calls |
| S16 | Hugging Face PEFT, "LoRA" developer guide and `LoraConfig` reference (merging, `all-linear`, rsLoRA) | 2026 | Vendor doc | https://huggingface.co/docs/peft/main/en/developer_guides/lora | 2026-10-10 | lora_recipe, packaging_model_card |
| S17 | Hugging Face TRL, "SFT Trainer" (`SFTConfig` defaults, adapters, VLMs, tool calling, loss masking) | 2026 | Vendor doc | https://huggingface.co/docs/trl/main/en/sft_trainer | 2026-10-10 | lora_recipe, distillation_tool_calls |
| S18 | Unsloth, "LoRA Hyperparameters Guide" | 2026 | Vendor doc | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide | 2026-10-10 | lora_recipe |
| S19 | Thinking Machines Lab, "LoRA Primer" (Tinker docs) | 2026 | Vendor doc | https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/ | 2026-10-10 | lora_recipe |
| S20 | Z. Shi, A. X. Yang, B. Wu, L. Aitchison, E. Yilmaz, A. Lipani, "Instruction Tuning With Loss Over Instructions" | 2024 | NeurIPS 2024 | https://arxiv.org/abs/2405.14394 | 2026-10-10 | lora_recipe |
| S21 | M. Huerta-Enochian, S. Y. Ko, "Instruction Fine-Tuning: Does Prompt Loss Matter?" | 2024 | arXiv preprint (v4) | https://arxiv.org/abs/2401.13586 | 2026-10-10 | lora_recipe |
| S22 | PyTorch, "torchtune" README (memory and throughput table) | 2026 | Vendor doc with measured tables | https://github.com/pytorch/torchtune | 2026-10-10 | lora_recipe |
| S23 | T. Chen, B. Xu, C. Zhang, C. Guestrin, "Training Deep Nets with Sublinear Memory Cost" | 2016 | arXiv preprint | https://arxiv.org/abs/1604.06174 | 2026-10-10 | lora_recipe |
| S24 | Qwen, "Qwen/Qwen3.5-4B" model card | 2026 | Vendor doc (model card) | https://huggingface.co/Qwen/Qwen3.5-4B | 2026-10-10 | lora_recipe, distillation_tool_calls, packaging_model_card |
| S25 | Qwen, `Qwen/Qwen3.5-4B` `config.json` | 2026 | Vendor doc (model config) | https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json | 2026-10-10 | lora_recipe, distillation_tool_calls, packaging_model_card |
| S26 | Qwen, `Qwen/Qwen3.5-4B` `chat_template.jinja` | 2026 | Vendor doc (chat template) | https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/chat_template.jinja | 2026-10-10 | distillation_tool_calls |
| S27 | Qwen, `Qwen/Qwen3.8-27B` `config.json` | 2026 | Vendor doc (model config) | https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json | 2026-10-10 | distillation_tool_calls |
| S28 | huggingface/transformers, `src/transformers/models/qwen3_5/modeling_qwen3_5.py` | 2026 | Source code (main) | https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py | 2026-10-10 | lora_recipe |
| S29 | huggingface/transformers, `src/transformers/integrations/hub_kernels.py` | 2026 | Source code (main) | https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py | 2026-10-10 | lora_recipe |
| S30 | huggingface/transformers, `src/transformers/models/auto/modeling_auto.py` | 2026 | Source code (main) | https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_auto.py | 2026-10-10 | lora_recipe |
| S31 | fla-org, "flash-linear-attention" README | 2026 | Vendor doc | https://github.com/fla-org/flash-linear-attention | 2026-10-10 | lora_recipe |
| S32 | fla-org/flash-linear-attention, issue #758 (Triton recompilation per sequence length) | 2026 | GitHub issue | https://github.com/fla-org/flash-linear-attention/issues/758 | 2026-10-10 | lora_recipe |
| S33 | ModelScope ms-swift, "Qwen3.5 Best Practices" (v4.4) | 2026 | Vendor doc | https://swift.readthedocs.io/en/v4.4/BestPractices/Qwen3_5-Best-Practice.html | 2026-10-10 | lora_recipe |
| S34 | QwenLM, `Qwen3-VL/qwen-vl-finetune` README and example script | 2026 | Vendor doc | https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune | 2026-10-10 | lora_recipe |
| S35 | Unsloth, "Qwen3.5 fine-tuning guide" | 2026 | Vendor doc | https://unsloth.ai/docs/models/qwen3.5/fine-tune | 2026-10-10 | lora_recipe |
| S36 | bitsandbytes-foundation, "bitsandbytes" README (accelerator support table) | 2026 | Vendor doc | https://github.com/bitsandbytes-foundation/bitsandbytes | 2026-10-10 | lora_recipe |
| S37 | ROCm/rocm-libraries, issue #7192, "[rocBLASLt] gfx1201 GPU causes lookup of wrong Tensile file (gfx1200.dat)" | 2026 | GitHub issue | https://github.com/ROCm/rocm-libraries/issues/7192 | 2026-10-10 | lora_recipe |
| S38 | Y. Kim, A. M. Rush, "Sequence-Level Knowledge Distillation" | 2016 | EMNLP 2016, pp. 1317-1327 | doi:10.18653/v1/D16-1139 (https://arxiv.org/abs/1606.07947) | 2026-10-10 | distillation_tool_calls |
| S39 | C.-Y. Hsieh et al., "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes" | 2023 | Findings of ACL 2023 | https://arxiv.org/abs/2305.02301 | 2026-10-10 | distillation_tool_calls |
| S40 | Hugging Face TRL, "Generalized Knowledge Distillation Trainer" (implements Agarwal et al., arXiv:2306.13649) | 2026 | Vendor doc | https://huggingface.co/docs/trl/main/en/gkd_trainer | 2026-10-10 | distillation_tool_calls |
| S41 | K. Lu and Thinking Machines Lab, "On-Policy Distillation" | 2025 | Lab write-up with data (Connectionism) | https://thinkingmachines.ai/blog/on-policy-distillation/ | 2026-10-10 | distillation_tool_calls |
| S42 | B. Zhang, Z. Liu, C. Cherry, O. Firat, "When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method" | 2024 | ICLR 2024 | https://arxiv.org/abs/2402.17193 | 2026-10-10 | distillation_tool_calls |
| S43 | J. Zhao et al., "LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report" | 2024 | arXiv preprint | https://arxiv.org/abs/2405.00732 | 2026-10-10 | distillation_tool_calls |
| S44 | L. E. Erdogan et al., "TinyAgent: Function Calling at the Edge" | 2024 | EMNLP 2024 (demo) | https://arxiv.org/abs/2409.00608 | 2026-10-10 | distillation_tool_calls, packaging_model_card |
| S45 | Z. Liu et al., "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets" | 2024 | arXiv preprint | https://arxiv.org/abs/2406.18518 | 2026-10-10 | distillation_tool_calls |
| S46 | S. G. Patil et al., "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models" | 2025 | ICML 2025, PMLR 267:48371-48392 | https://mlanthology.org/icml/2025/patil2025icml-berkeley/ | 2026-10-10 | distillation_tool_calls |
| S47 | Y. Cai et al., "LLaVA-KD: A Framework of Distilling Multimodal Large Language Models" | 2024 | arXiv preprint (v3, 2025) | https://arxiv.org/abs/2410.16236 | 2026-10-10 | distillation_tool_calls |
| S48 | ggml-org/llama.cpp, `docs/function-calling.md` | 2026 | Vendor doc | https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md | 2026-10-10 | distillation_tool_calls |
| S49 | ggml-org/llama.cpp, PR #16932, "Generalized XML-style tool-call parsing with streaming support" (merged 2025-11-18) | 2025 | GitHub PR | https://github.com/ggml-org/llama.cpp/pull/16932 | 2026-10-10 | distillation_tool_calls |
| S50 | ggml-org/llama.cpp, PR #19435, "[Model] Qwen3.5 dense and MoE support (no vision)" (merged 2026-02-08, reverted) | 2026 | GitHub PR | https://github.com/ggml-org/llama.cpp/pull/19435 | 2026-10-10 | packaging_model_card |
| S51 | ggml-org/llama.cpp, PR #19468, "[MODEL] support qwen3.5 series" (merged 2026-02-10) | 2026 | GitHub PR | https://github.com/ggml-org/llama.cpp/pull/19468 | 2026-10-10 | packaging_model_card |
| S52 | ggml-org/llama.cpp, `tools/mtmd/README.md` | 2026 | Vendor doc | https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md | 2026-10-10 | packaging_model_card |
| S53 | ggml-org/llama.cpp, `docs/multimodal.md` | 2026 | Vendor doc | https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md | 2026-10-10 | packaging_model_card |
| S54 | ggml-org/llama.cpp, `tools/quantize/README.md` | 2026 | Vendor doc with a measured table | https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md | 2026-10-10 | packaging_model_card |
| S55 | Y. Xu et al., "QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models" | 2023 | arXiv preprint | https://arxiv.org/abs/2309.14717 | 2026-10-10 | packaging_model_card |
| S56 | Unsloth, "unsloth/Qwen3.5-4B-GGUF" model card (file list and sizes) | 2026 | Vendor doc (model card) | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF | 2026-10-10 | packaging_model_card |
| S57 | M. Mitchell et al., "Model Cards for Model Reporting" | 2019 | FAT* 2019 | doi:10.1145/3287560.3287596 (https://arxiv.org/abs/1810.03993) | 2026-10-10 | packaging_model_card |
| S58 | Hugging Face Hub docs, "Model Cards" | 2026 | Vendor doc | https://huggingface.co/docs/hub/model-cards | 2026-10-10 | packaging_model_card |
| S59 | Apache Software Foundation, "Apache License, Version 2.0", section 4 | 2004 | License text | https://www.apache.org/licenses/LICENSE-2.0 | 2026-10-10 | packaging_model_card |
| S60 | ggml-org/llama.cpp, `convert_lora_to_gguf.py` | 2026 | Source code (master) | https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py | 2026-10-10 | packaging_model_card |
| S61 | Qwen, "Qwen/Qwen3.8-27B" model card | 2026 | Vendor doc (model card) | https://huggingface.co/Qwen/Qwen3.8-27B | 2026-10-10 | distillation_tool_calls, packaging_model_card |

## Our repository and team pages

These are the "our ledger line" kind of citation; the ledger itself does not exist yet, so repository files are cited
at a commit and Notion pages by URL.

| Key | Source | Type | Location | Opened | Cited in |
|---|---|---|---|---|---|
| R1 | Golden eval: `contexts/agent/eval.py` (`score()`) and `fixtures/golden/questions.yaml` (6 questions) | Repository, commit bf482f4 | https://github.com/raohemdutt/Knighthacks26/blob/bf482f4/contexts/agent/eval.py | 2026-10-10 | eval_noise |
| R2 | `runtime/llama.py` line 161: `LlamaServerRuntime._complete` sends `"temperature": 0.2`, no seed, and reads OpenAI `tool_calls` | Repository, commit bf482f4 | https://github.com/raohemdutt/Knighthacks26/blob/bf482f4/runtime/llama.py | 2026-10-10 | eval_noise, distillation_tool_calls |
| R3 | PR #37, "R2: GPU lab setup (teacher on GPU A, trainer on GPU B)" (merged): smoke LoRA 0.51 samples/s, 8.7 GB peak, loss 2.23 to 0.16; teacher decode 7.7 tok/s; `lab/smoke_lora.py`, `lab/pod_b_trainer.sh`, `lab/README.md` | GitHub PR and lab scripts | https://github.com/raohemdutt/Knighthacks26/pull/37 | 2026-10-10 | lora_recipe, distillation_tool_calls, packaging_model_card |
| R4 | Build Log entry "On-device model service, R8 (PR #36, bf482f4)": llama-server flags, unsloth Q4_K_M plus `mmproj-F16`, thinking off, RAM left under load, `--image-min-tokens 1024` warning | Notion (Build Log) | https://app.notion.com/p/3f596ef5fec5812fa37de5ae92086234 | 2026-10-10 | lora_recipe, distillation_tool_calls, packaging_model_card |
| R5 | `contracts/learning.py`: `ModelManifest` (version, base, task, gguf, mmproj, sha256, scores) | Repository, commit bf482f4 | https://github.com/raohemdutt/Knighthacks26/blob/bf482f4/contracts/learning.py | 2026-10-10 | packaging_model_card |
| R6 | `contexts/agent/application/qa.py`: `TOOLS = MEMORY_TOOLS` (`search_memory`, `last_seen`, `episodes_between`), `MAX_TOOL_CALLS = 3` | Repository, commit bf482f4 | https://github.com/raohemdutt/Knighthacks26/blob/bf482f4/contexts/agent/application/qa.py | 2026-10-10 | distillation_tool_calls |
| R7 | Learning track rubric: task O2 and its merged file `lab/refine/learning/RUBRIC.md` (PR #48, commit 904d92c), sections "The metric set" and "The rubric: nine learning tests" | Notion (Task Board) and repository | https://app.notion.com/p/3f596ef5fec5810db5ebe39aef2e0ec9 and https://github.com/raohemdutt/Knighthacks26/blob/904d92c/lab/refine/learning/RUBRIC.md | 2026-10-10 | all four notes |
| R8 | Ilan's Team Comms page, Q4 answer: `scores` keeps `current`, `candidate`, `teacher` and allows dotted per-metric keys | Notion (Team Comms) | https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 | 2026-10-10 | eval_noise |
| R9 | Task L3, "LoRA trainer for the Jetson model (GPU B)": a few hundred examples train in under about 30 min | Notion (Task Board) | https://app.notion.com/p/3f596ef5fec5813a88b4cd7f6ce2e8de | 2026-10-10 | lora_recipe |
| R10 | Atlas Design Doc (sections 2, 9, 13, 14) and its "Context: Learning" page | Notion (Design Doc) | https://app.notion.com/p/3f596ef5fec5819ea5cdcaa015e69917 and https://app.notion.com/p/3f596ef5fec5814bb5dfde0052175b1e | 2026-10-10 | lora_recipe, packaging_model_card |

## Searched for and not found

These claims were looked for and could not be traced to an opened source; the notes say so where they come up. The
design doc's "QLoRA on RDNA4 has a known rocBLAS bug that has stalled other people's runs" has no public source we
could find; the nearest report is S37, an inference-stack library lookup bug, not a QLoRA failure. The Context:
Learning page's "community runs of LoRA on a 7B vision model use about 17 GB on this card with ROCm 7.2" has no
measured source either. Nothing opened gives a measured slowdown for transformers' PyTorch fallback of the Gated
DeltaNet layers or any `fla` result on gfx1201, and nothing measures LoRA hyperparameters, seed variance or Q4_K_M
quality loss for Qwen3.5 on a vision or tool-calling task; the notes' experiments are how we get those numbers.
