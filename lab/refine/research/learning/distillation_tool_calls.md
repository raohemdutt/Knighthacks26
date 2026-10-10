# Distilling the 27B teacher into the 4B student: method, how many examples, and the tool-call format

Research note for the learning track, opened 2026-10-10 (all sources opened that day; index in `SOURCES.md`).
It covers what kind of distillation our loop is, how much labeled data moves a held-out score, how to keep the
teacher's labels honest, and how tool-call traces must be formatted so the learned skill fires on the Jetson.
Labels: `measured` = a number someone measured, with the setup; `practitioner` = a documented recommendation or
documented behaviour from the people who build or run the thing (papers' recommendation sections, official docs,
model cards, source code, maintainers); `anecdote` = a single report without data; `opinion` = argument without
evidence. Every finding carries its citation and a `[S..]` or `[R..]` key into `SOURCES.md`. Under each finding,
"Implies" maps it onto our metric set and "Experiment" proposes a test with the expected movement and the minimum
sample. Our metric set is the learning rubric's [R7] (`lab/refine/learning/RUBRIC.md`, "The metric set"):
held-out score before, after and teacher (`learning.heldout.current`, `learning.heldout.candidate`,
`learning.heldout.teacher`); label agreement rate (`learning.label_agreement`); loop wall-clock
(`learning.loop_minutes`); training samples per second (`trainer.samples_per_s`, 0.51 on pod B, PR #37);
package size (`learning.package_mb`).

Our setup: the teacher (Qwen3.8-27B, INT4 W4A16 on vLLM, pod A) answers each exported example; the trainer runs
plain SFT of the student (Qwen3.5-4B, pod B) on those answers. Two tasks: desk objects (one blurred keyframe in,
objects with positions out) and Atlas tool calling (the `search_memory`, `last_seen` and `episodes_between` tools,
at most 3 calls, then an answer [R6]). The Jetson serves the student with llama-server `--jinja` and thinking off
[R4].

## 1. What kind of distillation our loop is

### Takeaway
Training the student on the teacher's text is sequence-level knowledge distillation, a measured method. Token-level
and on-policy methods are stronger but need the teacher's log-probabilities; our two models share a tokenizer, so
that path stays open for later, but TRL's GKD trainer needs the teacher loaded inside the training process, which a
27B teacher cannot do beside the student on one 32 GB card.

### Cited findings
- **[measured]** Kim and Rush introduced sequence-level knowledge distillation (training the student on the teacher's decoded outputs); their best student "runs 10 times faster than its state-of-the-art teacher with little loss in performance", beats a baseline trained without distillation "by 4.2/1.7 BLEU with greedy decoding/beam search", and with pruning has "13 times fewer parameters" for "a decrease of 0.4 BLEU". Source: Y. Kim and A. M. Rush, "Sequence-Level Knowledge Distillation", 2016, EMNLP 2016, pp. 1317-1327, doi:10.18653/v1/D16-1139, https://arxiv.org/abs/1606.07947 (abstract) [S38]

  Implies: our loop (SFT on teacher answers) is the method with the measured greedy-decoding gain, which matters because the Jetson decodes near-greedily; the held-out score after training should sit between before and teacher, and the gap to teacher is the headroom.

  Experiment (DT-1): score base, sequence-distilled student and teacher on the same held-out set. Expected: after minus before of +0.2 to +0.4 on the tool-calling task, with after still below teacher. Minimum sample: 50 held-out items, one paired run per model.

- **[practitioner]** TRL's GKD trainer (now `trl.experimental.gkd.GKDTrainer`) wraps the SFT trainer with a `teacher_model` argument, computes losses from the teacher's token-level probabilities, and documents that `seq_kd=True` "can be viewed as supervised FT on teacher-generated output", that `lmbda` sets the share of on-policy student samples, and that "on-policy data (high `lmbda`) performs better". It implements Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (arXiv:2306.13649). Source: Hugging Face TRL, "Generalized Knowledge Distillation Trainer" (vendor doc, main), https://huggingface.co/docs/trl/main/en/gkd_trainer [S40]

  Implies: GKD as shipped needs the teacher in the trainer's process; a 27B teacher in bf16 is about 54 GB of weights alone, so it cannot sit next to the student on pod B, and our labels come from a separate vLLM server. Sequence-level KD is the TRL configuration our loop matches; the token-level upgrade would need the teacher's log-probs over the network.

  Experiment (DT-1, same): measure sequence-level KD first; only if the student's held-out score stalls below the teacher's by more than the minimum detectable effect, consider DT-3. Expected: as in DT-1. Minimum sample: 50 held-out items.

- **[measured]** Thinking Machines' on-policy distillation samples trajectories from the student and has the teacher grade every token through its log-probabilities (per-token reverse KL); they report "a baseline cost reduction of 9x when the SFT dataset is given" and about 30x including teacher sampling, cite Qwen3's report of 17,920 GPU hours for RL against 1,800 for on-policy distillation, and in a personalization test recovered IF-eval from 79% to 83% after mid-training on new documents. Source: K. Lu and Thinking Machines Lab, "On-Policy Distillation", 2025-10-27, Thinking Machines Lab: Connectionism (lab write-up with data), https://thinkingmachines.ai/blog/on-policy-distillation/ [S41]

  Implies: a short on-policy stage after SFT is a cheap way to recover behaviour SFT erodes (for example general instruction following while learning the tool format), and it only needs the teacher to score student tokens, which a vLLM server can do remotely.

  Experiment (DT-3): after one SFT run, sample 50 student answers to held-out-style prompts and ask the teacher server for log-probabilities of those exact tokens; compute the mean per-token reverse KL for base and SFT student. Expected: KL falls after SFT (a cheap progress signal); whether a vLLM prompt-log-prob option is available on our build was not checked here. Minimum sample: 50 prompts, about 5 minutes of pod A time.

- **[practitioner]** Both models' configs name the architecture `Qwen3_5ForConditionalGeneration` with `model_type` `qwen3_5` and a 248,320-entry vocabulary; the 27B has 64 language layers (16 full attention), hidden size 5120 and a 27-layer vision encoder (hidden 1152), against the 4B's 32 layers, 2560 and 24 vision layers (hidden 1024). Source: Qwen, `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.5-4B` `config.json` (vendor docs), https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json and https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json [S27] [S25]

  Implies: teacher and student token IDs line up, so token-level methods (DT-3) and the teacher-reference NLL metric (`eval_noise.md`, EN-13) are possible without vocabulary mapping; the vision encoders differ, so only outputs, not visual features, can be distilled.

  Experiment (DT-4): tokenize 100 teacher labels with both tokenizers. Expected: identical token sequences for all 100. Minimum sample: 100 labels, no GPU.

- **[measured]** LLaVA-KD distills a large multimodal model into 1B and 2B students with KL losses on response and visual-token distributions (1B: +4.0% Avg7 over SPHINX-Tiny) and reports that "larger teachers (Qwen1.5-7B and Qwen2.5-7B) exhibit lower distillation effectiveness": 57.4 against 57.9 Avg10 for a 7B versus a 4B Qwen1.5 teacher, and 58.3 against 58.9 for a 7B versus a 3B Qwen2.5 teacher (Table 8). Source: Y. Cai et al., "LLaVA-KD: A Framework of Distilling Multimodal Large Language Models", 2024 (v3 2025), arXiv preprint, https://arxiv.org/abs/2410.16236, sections 3.3.1, 4.2 and 4.3, Table 8 [S47]

  Implies: a bigger teacher is not automatically a better teacher for a small VLM; for our text-label distillation what limits the student is the teacher's label accuracy, so the label agreement rate on a hand-checked slice is the number that decides whether the 27B is worth its measured 7.7 tok/s decode on pod A [R3].

  Experiment (DT-5): label the same 50 desk-object keyframes with the 27B teacher and with the base 4B prompted the same way, and hand-check both. Expected: label agreement of at least 0.8 for the 27B and at most 0.6 for the 4B; if the gap is under 0.1 the teacher adds little for this task. Minimum sample: 50 hand-checked keyframes.

### Gaps
No source opened measures sequence-level distillation from a 27B into a 4B VLM on detection-style or tool-call
outputs; DT-1 is that measurement. LLaVA-KD's teacher-size result is for logit distillation and small gaps (3B to
7B teachers), not for our 27B-to-4B text labels.

## 2. Rationales as extra supervision

### Takeaway
Asking the teacher for a short rationale and training the student to produce it as a separate task bought large
data savings in one well-cited study. It has to stay out of the served answer, because the Jetson runs with thinking
off and tight token limits.

### Cited findings
- **[measured]** Distilling step-by-step trains small models on teacher labels plus teacher rationales as a multi-task objective; a "finetuned 770M T5 model outperforms the few-shot prompted 540B PaLM model using only 80% of available data", while standard fine-tuning of the same T5 "struggles to match even by using 100% of the dataset". Source: C.-Y. Hsieh et al., "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes", 2023, Findings of ACL 2023, https://arxiv.org/abs/2305.02301 (abstract) [S39]

  Implies: with only a few hundred examples per task, rationales are the cheapest known way to get more supervision per labeled frame; the held-out score could rise without more exports. The rationale must be a separate training target, not part of the served answer.

  Experiment (DT-6): for desk objects, train label-only against label plus rationale (a second task with its own prefix, never emitted at inference), same examples and seed. Expected: +0.05 to +0.10 held-out score at about 300 examples, no change in served answer length. Minimum sample: 1 run per arm, 50 held-out items.

### Gaps
The step-by-step results are for text tasks with T5 students; no source here tests rationales for object listing
from images.

## 3. How many examples move a held-out score

### Takeaway
Narrow tasks move far with little data: 1,000 curated examples aligned a 65B model, task-specific LoRAs beat their
bases by 34 points on average, and an 80K-example teacher-generated tool-calling set took a 1.1B model from 13% to
79%. Doubling data is not a reliable lever; quality is. For our tasks the expected shape is a steep rise in the first
few hundred examples, so the learning curve itself is the experiment.

### Cited findings
- **[measured]** LIMA fine-tuned a 65B LLaMA on "only 1,000 carefully curated prompts and responses" and its answers were "either equivalent or strictly preferred to GPT-4 in 43% of cases"; in its ablation "doubling the training set does not improve response quality". Source: C. Zhou et al., "LIMA: Less Is More for Alignment", 2023, arXiv preprint, https://arxiv.org/abs/2305.11206, abstract and section 5 [S15]

  Implies: a few hundred to a thousand well-labeled examples can move the held-out score a long way; label quality, measured as the label agreement rate, is the lever to protect before adding volume.

  Experiment (DT-7): train on 100, 200 and 400 examples of the same task (nested subsets, same seed) and score each on the same held-out set. Expected: most of the gain arrives by 200 examples; 400 adds less than 0.05 over 200. Minimum sample: 3 runs, 50 held-out items.

- **[measured]** Zhang et al. find that "LLM finetuning follows a powerbased multiplicative joint scaling law" between fine-tuning data size and each other factor, that fine-tuning "benefits more from LLM model scaling than pretraining data scaling", that scaling PEFT parameters is generally ineffective, and that "the optimal finetuning method is highly task- and finetuning data-dependent". Source: B. Zhang, Z. Liu, C. Cherry, O. Firat, "When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method", 2024, ICLR 2024, https://arxiv.org/abs/2402.17193 (abstract) [S42]

  Implies: the held-out score should follow a power law in example count, so three points (DT-7) are enough to fit it and predict what 800 examples would buy before spending a labeling slot; adding LoRA parameters will not substitute for data.

  Experiment (DT-7, same): fit a power law to the 100/200/400 points. Expected: a predicted gain from 400 to 800 examples below the minimum detectable effect, which would stop further labeling for that task. Minimum sample: 3 runs, 50 held-out items.

- **[measured]** LoRA Land fine-tuned 310 models (10 base models, 31 tasks) with "quantized low rank adapters" and found that "4-bit LoRA fine-tuned models outperform base models by 34 points and GPT-4 by 10 points on average". Source: J. Zhao et al., "LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report", 2024, arXiv preprint, https://arxiv.org/abs/2405.00732 (abstract) [S43]

  Implies: on narrow, well-defined tasks a LoRA student can exceed a much larger general model, so "after above teacher" on a single task is plausible rather than a sign of a scoring bug; the slide can show it if the paired test supports it.

  Experiment (DT-1, same): include the teacher in the same paired held-out run. Expected: after stays below teacher on desk objects but may reach it on the tool-calling format. Minimum sample: 50 held-out items.

- **[measured]** TinyAgent generated 80K training, 1K validation and 1K test examples with GPT-4-Turbo as the teacher and LoRA-tuned small models for 3 epochs at learning rate 7e-5; success (1 only if both the function set and the call graph are right) rose from 12.71% to 78.89% for TinyLlama-1.1B and from 41.25% to 83.09% for Wizard-2-7B, against 79.08% for GPT-4-Turbo. Source: L. E. Erdogan et al., "TinyAgent: Function Calling at the Edge", 2024, EMNLP 2024 (demo), https://arxiv.org/abs/2409.00608, section 3.3, Tables 1 and 2 [S44]

  Implies: teacher-generated tool-calling data can lift a small edge model to the teacher's level on a closed tool set like ours, and success should be judged on the call structure, not the answer text alone. It took 80K examples there for its own tool set and multi-step plans judged as call graphs; how many our 3 tools and short plans need is exactly what DT-7 measures.

  Experiment (DT-8): score tool-calling held-out items on two dotted metrics, call structure correct (function names and arguments) and final answer correct. Expected: call-structure accuracy rises first and further (+0.3 or more) than final-answer accuracy after the first run. Minimum sample: 50 held-out items.

- **[measured]** APIGen built 60,000 function-calling examples over 3,673 executable APIs in 21 categories, each "verified through three hierarchical stages: format checking, actual function executions, and semantic verification", and reports that 7B models trained on them reach state-of-the-art on the Berkeley Function-Calling Benchmark, ahead of multiple GPT-4 models, with a 1B model ahead of GPT-3.5-Turbo and Claude-3 Haiku. Source: Z. Liu et al., "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets", 2024, arXiv preprint, https://arxiv.org/abs/2406.18518 (abstract) [S45]

  Implies: the label agreement rate for tool-calling labels can be measured mostly by machine: our three tools run locally against fixture memory, so every teacher call can be format-checked, executed, and its answer checked against the evidence before it becomes training data.

  Experiment (DT-9): run the three-stage filter on the first teacher-labeled tool-calling set, then hand-check 50 kept labels. Expected: 5 to 20% of labels rejected at the format or execution stage, and a label agreement of at least 0.9 on the kept slice. Minimum sample: all labels through the filter, 50 hand-checked.

### Gaps
None of these sources gives a learning curve between 100 and 1,000 examples for a VLM or for tool calling with 3
tools; DT-7 measures ours.

## 4. Tool-call traces: train exactly the format llama-server will parse

### Takeaway
Qwen3.5 does not use JSON tool calls: its template writes `<tool_call>` blocks with `<function=...>` and
`<parameter=...>` tags (the Qwen3-Coder XML style), and with thinking off it emits an empty thinking block first.
The student must be trained on exactly that rendering. On the serving side, llama.cpp gained a parser for this XML
style in November 2025, but its docs do not list Qwen3.5 and users report rough edges, and our runtime only reads
parsed OpenAI `tool_calls`. Whether the Jetson's build parses Qwen3.5 tool calls is therefore the first thing to
check, before any training.

### Cited findings
- **[practitioner]** Qwen3.5-4B's chat template tells the model "If you choose to call a function ONLY reply in the following format with NO suffix", shows `<tool_call>`, `<function=example_function_name>` and `<parameter=example_parameter_1>` tags, says "Function calls MUST follow the specified format", returns tool results as `<tool_response>` blocks inside a user turn, emits `<think>\n\n</think>\n\n` when `enable_thinking` is false, and drops reasoning from assistant turns before the last user query. Source: Qwen, `Qwen/Qwen3.5-4B` `chat_template.jinja` (vendor doc), https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/chat_template.jinja [S26]

  Implies: the trace formatter must render each teacher trace through this template with `enable_thinking=False`, so the student learns the XML calls and the empty thinking prefix that the Jetson will produce; a trace rendered as Hermes-style JSON would train a format the model never emits at serving time, and the held-out tool-call score would fail on format.

  Experiment (DT-10): render 20 teacher traces through the template, tokenize, detokenize and parse the tool calls back. Expected: 20 of 20 round-trip with identical function names and arguments. Minimum sample: 20 traces, no GPU.

- **[practitioner]** The Qwen3.5-4B model card says the model runs "in thinking mode by default", is switched off with `"chat_template_kwargs": {"enable_thinking": False}` (the `/think` and `/nothink` soft switches are not supported), recommends instruct-mode sampling of temperature 0.7, top_p 0.8, top_k 20 and presence penalty 1.5, and serves tool calls in vLLM and SGLang with `--tool-call-parser qwen3_coder`; the teacher's card says the same of the teacher ("Qwen3.8 models operate in thinking mode by default", switched off the same way) and adds a `reasoning_effort` setting. Source: Qwen, "Qwen/Qwen3.5-4B" and "Qwen/Qwen3.8-27B" model cards (vendor docs), https://huggingface.co/Qwen/Qwen3.5-4B and https://huggingface.co/Qwen/Qwen3.8-27B, quick start and best practices [S24] [S61]

  Implies: teacher labels must be requested with thinking off, otherwise the 27B spends its budget reasoning (the lab README already passes `enable_thinking: false` for captions [R3]) and labeling wall-clock rises; and the teacher's calls come back parsed into OpenAI JSON by vLLM, so the formatter must turn them back into the template's XML, not copy the JSON.

  Experiment (DT-11): label 20 tool questions with the teacher with thinking on and off. Expected: thinking off uses at least 5 times fewer tokens per label (shorter loop wall-clock) with label agreement within 0.05 of thinking on. Minimum sample: 20 questions x 2 settings.

- **[practitioner]** llama.cpp's function-calling doc lists its native formats (Llama 3.x, Functionary, Hermes 2/3, Qwen 2.5, Qwen 2.5 Coder, Mistral Nemo, Firefunction v2, Command R7B, DeepSeek R1, GPT-OSS and others) and says "Generic tool call is supported when the template isn't recognized by native format handlers", which "may consume more tokens and be less efficient than a model's native format"; tool calling needs `--jinja`. Source: ggml-org/llama.cpp, `docs/function-calling.md` (vendor doc), https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md [S48]

  Implies: neither Qwen3.5 nor the Qwen3-Coder XML style appears in that list, so the doc alone does not show that the Jetson's llama-server turns the student's `<function=...>` output into the OpenAI `tool_calls` our runtime reads [R2]; if it falls back to the generic handler or leaves the XML as text, every tool question fails in the held-out score no matter how well the student learned.

  Experiment (DT-12): with the Jetson's llama.cpp commit (a CPU build is enough) and the base Q4_K_M GGUF, send one golden tool question with the `TOOLS` list 10 times and read the response and the server's chat-format log line. Expected: non-empty `tool_calls` in at least 9 of 10 responses and a native (not "Generic") chat format. Minimum sample: 10 requests, no GPU.

- **[practitioner]** llama.cpp PR #16932, "Generalized XML-style tool-call parsing with streaming support", merged on 2025-11-18, lists "Qwen3-Coder (Thinking and non-thinking)" among the formats it parses, with the author recommending its bundled chat template. Source: ggml-org/llama.cpp, GitHub PR #16932 (merged), https://github.com/ggml-org/llama.cpp/pull/16932 [S49]

  Implies: a parser for Qwen3.5's call syntax exists in llama.cpp builds after November 2025, so DT-12 is expected to pass on a 2026 build; if it fails, the fix is a newer build or the template override, not retraining.

  Experiment (DT-12, same): as above. Expected: non-empty `tool_calls` in at least 9 of 10 responses. Minimum sample: 10 requests.

- **[anecdote]** Commenters on PR #16932 report tool calling stopping mid-workflow in agentic clients across several models, one writing "it's very likely there is a bug in the current implementation", and a linked ik_llama.cpp issue is titled "Bad tool calling performance on Qwen3.5 35B A3B". Source: ggml-org/llama.cpp, GitHub PR #16932 discussion (user comments), https://github.com/ggml-org/llama.cpp/pull/16932 [S49]

  Implies: multi-step traces (call, tool response, second call, answer) are where parsing breaks; our golden questions can need up to 3 calls, so the held-out tool-call score should be broken down by step.

  Experiment (DT-13): run 10 multi-call golden questions end to end through llama-server and record at which step parsing fails. Expected: zero failures on single-call questions; any failures cluster after the first tool response. Minimum sample: 10 multi-call questions.

- **[practitioner]** TRL's SFT trainer "fully supports fine-tuning models with tool calling capabilities" from conversational examples with `tool_calls`, `tool`-role messages and a `tools` column of JSON schemas, and its assistant-only loss "requires the chat template to include `{% generation %}` and `{% endgeneration %}` keywords" (auto-patched "for known model families (e.g. Qwen3)"; with vision data it needs transformers 5.18 or newer). Source: Hugging Face TRL, "SFT Trainer" (vendor doc, main), https://huggingface.co/docs/trl/main/en/sft_trainer, sections "Tool Calling with SFT" and "Train on assistant messages only" [S17]

  Implies: the formatter can hand TRL structured messages plus the `TOOLS` schemas and let the model's own template render the XML, which removes a hand-written renderer; whether TRL's auto-patch covers the Qwen3.5 template is not stated, so the loss mask must be checked.

  Experiment (DT-14): build the TRL dataset for 20 traces and print the rendered text and the assistant mask. Expected: rendered text identical to the llama-server prompt plus completion for the same messages, and the mask covering only assistant spans. Minimum sample: 20 traces, no GPU.

- **[practitioner]** The Berkeley Function Calling Leaderboard evaluates "serial and parallel function calls" with "a novel Abstract Syntax Tree (AST) evaluation method that can easily scale to thousands of functions", and also tests whether models "abstain and reason in stateful multi-step agentic setting". Source: S. G. Patil et al., "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models", 2025, ICML 2025 (PMLR 267:48371-48392), https://mlanthology.org/icml/2025/patil2025icml-berkeley/ (abstract) [S46]

  Implies: score each tool call by parsing it and comparing the function name and argument values structurally, not by string match, and keep an abstention item (our golden set's "Where is my wallet?" with no evidence plays that role).

  Experiment (DT-8, same): the structural call metric as a dotted score key next to the final-answer score. Expected: structural accuracy moves by 0.3 or more after the first run, more than the final-answer score. Minimum sample: 50 held-out items.

### Gaps
No source opened confirms Qwen3.5 tool calling through llama.cpp end to end, and none covers how the empty thinking
block interacts with llama.cpp's tool-call parser when `--reasoning-budget 0` is set. DT-12 and DT-13 settle both
on our build before any training run.
