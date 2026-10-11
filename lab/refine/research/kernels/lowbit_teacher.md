# Lower-bit teacher weights: which 3-bit and 2-bit method vLLM can serve on gfx1151, and what it should buy

Decode of the 27B teacher on the Strix Halo is weight-read bound, so fewer bits per weight is the third way past the
bandwidth ceiling (after faster kernels and batching). This note decides how to make 3-bit and 2-bit variants of
`Qwen/Qwen3.8-27B` that vLLM on ROCm can actually serve: the checkpoint format, the method per width, the group size
and calibration set, the vLLM change the serving path needs, and the ceiling each width predicts. The quality cost
and the measured speed are not here yet: they need the Colab A100 run and the pod D serve test (Open questions).

## The checkpoint format: compressed-tensors W3A16 and W2A16

- **[practitioner]** compressed-tensors defines the presets W2A16 and W3A16 with the same helper as W4A16: integer weights, group strategy with group size 128, symmetric, static, activations left in 16 bits. Source: [S188] compressed-tensors `src/compressed_tensors/quantization/quant_scheme.py::_int_wnam` and the `W2A16`, `W3A16`, `W4A16` presets at 41103c0.
  - Implication: the low-bit variants keep the serving INT4 teacher's layout (symmetric, group 128) and change only the width, so the bytes per weight fall from 4.125 to 3.125 and 2.125 and every other part of the decode step stays the same.
  - Candidate experiment: none beyond KL1 (below), which serves both presets. Expected movement: none on its own. Minimum sample size: 1 load per width. Confidence: high.
- **[practitioner]** compressed-tensors packs sub-8-bit weights densely into int32 with no padding bits: element i of each 32-element block starts at bit i x B of the block's B words, and a value that crosses a word boundary is split across the two words. Source: [S189] compressed-tensors `src/compressed_tensors/compressors/pack_quantized/helpers.py::pack_to_int32` at 41103c0 (docstring and body).
  - Implication: a 3-bit checkpoint stores 32 weights in 3 words, so any loader or kernel that assumes 32 // 3 = 10 weights per word reads it wrong; the loader and kernel below both have to use the dense layout.
  - Candidate experiment: none beyond KL1. Expected movement: none on its own. Minimum sample size: 1 load per width. Confidence: high.

## Which vLLM can load 3 and 2 bits

- **[practitioner]** vLLM main at 276fbcf maps 2, 3, 4 and 8 bits to `uint2b2`, `uint3b4`, `uint4b8` and `uint8b128` in `CompressedTensorsWNA16`, sizes the packed weight with a `Fraction(32, num_bits)` pack factor and `ceil(input_size x num_bits / 32)` words, which matches the dense layout. Source: [S191] vLLM `compressed_tensors_wNa16.py`, `WNA16_SUPPORTED_TYPES_MAP` and `create_weights`, at 276fbcf.
  - Implication: the loader of a current vLLM already accepts our 3-bit and 2-bit checkpoints; only the kernel choice stands in the way (next topic).
  - Candidate experiment: KL1, as specified under "The ceiling each width predicts". Confidence: high.
- **[practitioner]** At b1388b1, the April commit the teacher runs on, the same scheme accepts only 4 and 8 bits (`{4: uint4b8, 8: uint8b128}`) with an integer pack factor `32 // num_bits`; 2 to 7 bits arrived with commit 56ca5997e, "Humming support for 2/3/5/6/7-bit pack-quantized weight-only inference (#46389)", which lies between b1388b1 and 276fbcf. Source: [S192] vLLM `compressed_tensors_wNa16.py` at b1388b1; [S200] vLLM commit 56ca5997e.
  - Implication: a low-bit checkpoint in compressed-tensors format needs the upgraded vLLM that R60 builds (`lab/vllm_upgrade.sh` pins main 5281e4990 of 2026-10-07, which contains 56ca5997e), never the b1388b1 build; `lab/vllm-lowbit.diff` does not apply at b1388b1 and would not help there. Source for the pin: [S210]. The GPTQ-format route in the next finding is the one that can serve on b1388b1.
  - Candidate experiment: none beyond KL1, which runs on the R60 build. Expected movement: none on its own. Minimum sample size: 1 load. Confidence: high.

- **[practitioner]** At b1388b1, vLLM's native GPTQ quantization method (checkpoints in GPTQ format, not compressed-tensors) accepts 2, 3, 4 and 8 bits, and its linear method calls `gptq_shuffle` and the same `gptq_gemm` ExLlama kernels. Source: [S211] vLLM `vllm/model_executor/layers/quantization/gptq.py`, `GPTQConfig.__init__` (the bits check) and `GPTQLinearMethod.process_weights_after_loading`, at b1388b1.
  - Implication: a 3-bit or 2-bit checkpoint repacked into GPTQ format can be served on the teacher's current b1388b1 build without `lab/vllm-lowbit.diff`, using the same kernels, so the pod D test does not have to wait for R60 if the checkpoint is in that format. Both routes end in `gptq_gemm`, so their speed should match; only the loader differs.
  - Candidate experiment: KL1 on b1388b1 with a GPTQ-format 3-bit checkpoint, beside the compressed-tensors one on the R60 build. Expected movement: decode tok/s within 5 percent between the two routes. Minimum sample size: 3 spawns per route at batch 1. Confidence: medium.

## The kernel that serves 3 and 2 bits on ROCm

- **[practitioner]** On ROCm, vLLM 276fbcf tries RDNA3W4A16, RDNAHybridW4A16, TritonW4A16, Conch and Exllama in that order; RDNA3 and RDNA hybrid accept `uint4b8` and `uint4` only, Triton W4A16 the same two, and Conch `uint4`, `uint8`, `uint4b8` and `uint8b128`. Source: [S195] vLLM `vllm/model_executor/kernels/linear/__init__.py`, `_POSSIBLE_KERNELS`, at 276fbcf; [S197] `mixed_precision/rdna_hybrid_w4a16.py`, `triton_w4a16.py` and `conch.py` supported-type lists at 276fbcf.
  - Implication: the RDNA hybrid kernel of PR 40977, the one R53 and R60 bring in for 4 bits, cannot serve 3 or 2 bits; ExLlama is the only candidate, so a low-bit variant is judged against the ExLlama path, not the hybrid kernel's 4-bit numbers.
  - Candidate experiment: none beyond KL1, which reads the chosen kernel class from the serve log. Expected movement: none on its own. Minimum sample size: 1 serve per width. Confidence: high.
- **[practitioner]** The ExLlama GPTQ kernels in vLLM's csrc already have 2-bit and 3-bit paths (`gemm_half_q_half_gptq_2bit_kernel`, `gemm_half_q_half_gptq_3bit_kernel`, `shuffle_3bit_kernel`, and the `MatrixView_q2_row` and `MatrixView_q3_row` readers, the latter reading 32 values from 3 words with value i at bit 3i), but `ExllamaLinearKernel.SUPPORTED_QUANT_TYPES` lists only `uint4b8` and `uint8b128`, with the comment that `uint2b2` and `uint3b4` are supported "in theory" but untested. Source: [S194] vLLM `csrc/libtorch_stable/quantization/gptq/q_gemm.cu` at 276fbcf; [S193] `csrc/libtorch_stable/quantization/gptq/matrix_view.cuh::MatrixView_q3_row` at 276fbcf; [S190] `vllm/model_executor/kernels/linear/mixed_precision/exllama.py` at 276fbcf.
  - Implication: serving 3 and 2 bits needs a Python-only change to `exllama.py`, no kernel build, which is `lab/vllm-lowbit.diff` (39 added lines): list the two types, and pack the synthetic zero points densely for 3 bits.
  - Candidate experiment: KL1. Confidence: medium (0.6) that the untested 2-bit and 3-bit HIP paths compile and run on gfx1151, since the 4-bit path of the same file does.
- **[practitioner]** For a symmetric checkpoint the ExLlama kernel synthesises zero points of value bias minus 1 and packs them with `pack_quantized_values_into_int32`, which packs `32 // size_bits` values per word (10 for 3 bits); and its `can_implement` requires the output features to be a multiple of that same `32 // size_bits`. Source: [S196] vLLM `vllm/model_executor/layers/quantization/utils/quant_utils.py::pack_quantized_values_into_int32` at 276fbcf; [S190] `exllama.py::process_weights_after_loading` and `can_implement` at 276fbcf.
  - Implication: for 3 bits that packing disagrees with the dense layout `MatrixView_q3_row` reads, so without the patch every 3-bit zero point would be read wrong even if the type were allowed. The patch packs them densely (32 values in 3 words) and requires output features to be a multiple of 32 for 3 bits; all our shapes (5120, 6144, 17408 and the attention projections) are. The 2-bit packing (16 per word) is already dense. `tests/test_lab_lowbit_serve.py` runs the patched packer against a transcription of `MatrixView_q3_row`.
  - Candidate experiment: KL1 plus the correctness check of KL2. Expected movement: a wrong zero-point layout shows as nonsense text at first token; correct packing gives coherent answers. Minimum sample size: 4 prompts per width. Confidence: high.
- **[practitioner]** `ExllamaLinearKernel.can_implement` refuses any activation type but float16. Source: [S190] `exllama.py::can_implement` at 276fbcf.
  - Implication: the low-bit serve runs with `--dtype float16`, as the existing `exllama_fp16` arm of `settings_inference_gfx1151.md` does; a bf16 run would find no kernel. `lab/serve_lowbit.sh` passes it.
  - Candidate experiment: none beyond KL1. Expected movement: none on its own. Minimum sample size: 1 serve. Confidence: high.
- **[practitioner]** Commit 2c9d68f80, "Remove GPTQ Group/Dynamic Activation Ordering (#54809)", removed the `g_idx` handling from `exllama.py` between b1388b1 and 276fbcf. Source: [S201] vLLM commit 2c9d68f80 and the diff of `exllama.py` from b1388b1 to 276fbcf.
  - Implication: the GPTQ recipe uses static activation order (columns reordered during quantization, groups kept contiguous, no `g_idx` at serve time), which is llm-compressor's default and what the serving INT4 teacher uses; group or dynamic order would produce a checkpoint this vLLM cannot load.
  - Candidate experiment: none beyond KL1. Expected movement: none on its own. Minimum sample size: 1 load. Confidence: high.

## Method per width

- **[practitioner]** llm-compressor's own Qwen3.8-27B W4A16 example loads the model with `Qwen3_5ForConditionalGeneration` inside `load_context`, quantizes `Linear` layers with GPTQ after AWQ, and keeps in bf16 the vision tower, `lm_head`, `embed_tokens` and the Gated DeltaNet `linear_attn.in_proj_a` and `in_proj_b`. Source: [S202] llm-compressor `examples/quantization_w4a16/qwen3_8_gptq_awq_example.py` at 2297585.
  - Implication: `lab/quantize_teacher.py` uses the same loader and the same ignore list, so the low-bit variants differ from the INT4 teacher only in the quantized layers' width, and the bytes of `lm_head` and the recurrent state stay as in the byte budget below.
  - Candidate experiment: KL2, as specified under "Quality cost". Confidence: medium.
- **[practitioner]** llm-compressor's `GPTQModifier` defaults to block size 128, dampening 0.01 and static activation order; its `AutoRoundModifier` defaults to 200 iterations and batch size 8; and a recipe may be a YAML or JSON string. Source: [S204] llm-compressor `src/llmcompressor/modifiers/gptq/base.py` and `modifiers/autoround/base.py` field defaults at 2297585; [S206] `src/llmcompressor/recipe/recipe.py::Recipe.create_instance` at 2297585.
  - Implication: 3 bits use GPTQ with those defaults (fast, one pass per layer); the recipe is committed as `fixtures/lab/quant/plan.json` and passed to `oneshot` as JSON text.
  - Candidate experiment: KL2. Confidence: medium.
- **[practitioner]** The AutoRound README lists W2A16 and W3A16 among its schemes and recommends "auto-round for W4A16 and auto-round-best with `enable_alg_ext` for W2A16"; its default is 200 iterations, and its light recipe warns of a "larger accuracy drop at W2". Source: [S208] intel/auto-round `README.md` at 7a660e1, sections on recipes and on the AutoRound API.
  - Implication: 2 bits use AutoRound rather than GPTQ, through llm-compressor's `AutoRoundModifier` at 200 iterations (the plain recipe), so the checkpoint stays compressed-tensors; the best recipe with `enable_alg_ext` is the next arm if the 2-bit eval drop is too large.
  - Candidate experiment: KL3, a second 2-bit run with 1000 iterations and `enable_alg_ext` if KL2 shows the 2-bit drop beyond the margin. Expected movement: L4 eval score of the 2-bit variant up by a few points at about 3 to 5 times the quantization time. Minimum sample size: 1 quantization and 1 eval. Confidence: low.
- **[opinion]** The AutoRound README states that it "delivers strong performance even at 2-3 bits", pointing at its own example models. Source: [S208] intel/auto-round `README.md` at 7a660e1, "Superior Accuracy".
  - Implication: none for the metric set yet: a project's claim about its own method, with no number for a hybrid Gated DeltaNet model such as ours; only KL2 can say what 2 bits cost the teacher.
  - Candidate experiment: none beyond KL2. Expected movement: none until KL2 runs. Minimum sample size: 1 eval per width. Confidence: low.

## Calibration set

- **[practitioner]** The calibration set is 184 rows of Atlas's own serve-time text from `fixtures/`: 118 labelled tool-calling turns, the 48 golden questions and phrasings, the 16 rendered chat-template cases and the 2 text bench prompts, in the Qwen3.5 chat format with thinking off, about 44,000 tokens at four characters per token; GPTQ reads them at up to 4096 tokens and AutoRound pads them to 2048. Source: `fixtures/lab/quant/make_calibration.py` and `fixtures/lab/quant/calibration.jsonl` in this repo.
  - Implication: the rounding is fitted to the traffic the teacher serves (short tool-calling turns) rather than a generic chat set, at the risk of overfitting a small set; it holds no image and no frame, so nothing private leaves the repository.
  - Candidate experiment: KL2 scores each variant on the L4 held-out set, whose questions are phrasings the calibration set does not contain verbatim. Expected movement: if the set overfits, the L4 drop is larger than the correctness drift suggests. Minimum sample size: 1 eval per width. Confidence: low.

## The ceiling each width predicts

- **[opinion]** From the serving checkpoint's byte budget (12.60 GB of int4 layers with bf16 group-128 scales, so about 24.4 billion quantized weights; 2.54 GB of bf16 `lm_head`; about 0.30 GB of float32 state per sequence) and the chip sheet's 210 GB/s DRAM copy rate, one decode step reads 15.44 GB at 4 bits, 12.39 GB at 3 bits and 9.33 GB at 2 bits at batch 1, a ceiling of 13.6, 17.0 and 22.6 tok/s, and 5.5 to 9.4, 4.4 to 7.5 and 3.3 to 5.7 J/token at the 75 to 128 W the pods draw loaded; at batch 4 the ceilings are 51.5, 63.4 and 82.3 tok/s in total. Source: arithmetic in `lab/lowbit_report.py::predict` on `lab/refine/research/kernels/w4a16_decode_gfx1201.md` (section 1.2, the byte budget) and [L10] the gfx1151 chip sheet.
  - Implication: 3 bits can raise the single-stream ceiling by about 25 percent and 2 bits by about 66 percent, but only if the ExLlama kernel streams weights as close to the bandwidth as the 4-bit kernel does; a slower kernel can eat the gain, so the serve test reports the fraction of the ceiling reached per width.
  - Candidate experiment: KL1, the pod D serve test: `lab/serve_lowbit.sh` per width, `lab/bench.py` at batch 1 and 4, three spawns each, then `lab/lowbit_report.py report` writes `data/lowbit-gfx1151-<date>.json`. Expected movement: decode tok/s up 10 to 25 percent at 3 bits and 20 to 60 percent at 2 bits over the 4-bit teacher on the same build, J/token down by about the same fraction. Minimum sample size: 3 spawns per width and batch size. Confidence: low (0.35), because the ExLlama path is untested on gfx1151.

## Quality cost

- **[practitioner]** The quality gate has two parts that already exist: the R21 correctness harness compares greedy token outputs with a recorded baseline and reports a match rate and normalised-text matches, and the L4 scorer (`python -m contexts.agent.eval --json`) reports a pass-rate `score` on the held-out questions. Source: `lab/correctness.py::summarize`; `contexts/agent/eval.py` (the `--json` output).
  - Implication: the correctness compare against the INT4 baseline will diverge for any other checkpoint (it needs `--allow-condition-drift`), so it is reported, not gated; the gate is the L4 score drop from the 4-bit teacher, and `lab/lowbit_report.py` names the lowest width within the margin, or none. The margin itself is not agreed yet (Open questions).
  - Candidate experiment: KL2, per width on the served variant: one correctness compare against `fixtures/lab/correctness/vllm-gfx1151-2026-10-10.json` and one L4 score, plus one L4 score of the 4-bit teacher on the same build. Expected movement: L4 score down by 0 to 3 points at 3 bits and by 5 to 20 points at 2 bits. Minimum sample size: 1 compare and 1 eval per width. Confidence: low.

## Running on the Colab A100

- **[practitioner]** The Colab CLI's `colab exec -f file.py` reads the file locally and sends its content to the runtime's kernel, with no argument list, while `colab run SCRIPT [ARGS...]` forwards arguments on a fresh VM it tears down after; `colab new --gpu A100 --high-mem` asks for a high-RAM machine; `colab upload` copies a local file to the VM. Source: [S209] googlecolab/google-colab-cli `README.md` at 930275e, command tables and "Transparent Code Execution".
  - Implication: inside the kernel the script reads its arguments from an uploaded `/content/quant/args.txt` (`fixtures/lab/quant/args-w3.txt`, `args-w2.txt`), installs `requirements-quantize.txt` before importing torch, and reads the Hub token from an uploaded file, since no environment reaches the kernel; `--high-mem` gives the CPU memory that holds the 54 GB bf16 model while llm-compressor onloads one layer at a time.
  - Candidate experiment: the 3-bit run first (about 75 minutes estimated, not measured), then the 2-bit run (about 180 minutes estimated). Expected movement: none on serving numbers; it produces the two checkpoints KL1 and KL2 need. Minimum sample size: 1 run per width. Confidence: medium.

## Open questions

- The agreed margin for the L4 eval drop is not set; `lab/lowbit_report.py --margin` takes it once Ilan decides it.
- Whether the ExLlama 2-bit and 3-bit HIP kernels run correctly on gfx1151, and how close to 210 GB/s they stream, is unmeasured until KL1.
- The quantization times (75 and 180 minutes) and peak memory are estimates; `atlas_quant.json` records the real ones.
- Whether the 2-bit variant needs a smaller group size (AutoRound offers W2A16G64 and G32) is a follow-up if KL2 shows a large drop; it adds about 0.125 or 0.375 bits per weight.

## Search log

- 2026-10-11, budget 3 h (kernels), about 1 h 30 min used across two sessions. Starting points: blobless clones of vllm-project/vllm (main 01abe975b, with 276fbcf, b1388b1 and 5281e4990 read by `git show`), vllm-project/llm-compressor at 2297585, vllm-project/compressed-tensors at 41103c0, intel/auto-round at 7a660e1 and googlecolab/google-colab-cli at 930275e. Stopped when the loader, kernel list and packing all agreed and the remaining questions (kernel speed and quality cost) need the GPU runs.
