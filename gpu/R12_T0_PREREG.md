# R12 pre-registration for T0 (post before the first real run)

This is the text the dispatcher posts on T0 (the experiment queue) before any R12 spawn on the pod. It is written
before the run on purpose: the measured change goes beside it afterwards, slower results included (kernels rubric
K8 (b): a slower result is still a result). The dispatcher adds the usual signature line when posting. The two
levers are two experiment lines, one lever each, measured in separate arms against the same stock arm.

## Lines to post

**E-next (kernels) R12a: unified attention decode launch, num_stages 1 -> 2 (gfx1151).**
Lever: `num_stages` of `kernel_unified_attention_3d` only (`gpu/patches/unified-attention-stages.patch`); the 2D
launch keeps the known fix (`num_stages=1`). Arms: stock (vLLM b1388b1 plus the whole `lab/vllm-gfx1201.diff`, the
teacher's build) against stages, both served with `--attention-backend TRITON_ATTN`. Metric to move:
`joules_per_token` at batch 1 (R11 harness, `lab/bench.py`), with `decode_tok_s` beside it.
Expected change in joules per token at batch 1: **0.0 %, range -0.5 % to +0.5 %** (no change the harness can
resolve). At batch 4: the same range. Confidence: medium (0.6) that the true change lies in that range.
Minimum runs: 5 cold spawns per arm (the verdict floor); 5 spawns resolve about 2 % to 3 % at a 1 % to 2 % spawn
spread (`runs_per_arm_relative`), so this line expects "no detectable change", and that is the result to record if
it comes. LDS: expected under 48 KiB for the 3D kernel at 2 stages (fits 65,536). Output match: expected equal to
stock's own rate (the change does not reorder any arithmetic). Citations: `lab/refine/research/kernels/
triton_gfx12.md` s.1 (66,560 bytes on gfx1201, the known fix), `strix_halo_hardware.md` s.3 (65,536 per workgroup on
gfx1151), `settings_inference_gfx1151.md` (attention on 16 of 64 layers; ROCM_ATTN is the default backend).

**E-next+1 (kernels) R12b: unified attention decode KV tile 16 -> 32 (gfx1151).**
Lever: `TILE_SIZE_DECODE` for 16-bit dtypes, 16 -> 32 (`gpu/patches/unified-attention-block.patch`), taken by
`kernel_unified_attention_3d` and `reduce_segments`; both launches keep `num_stages=1` as in stock. Arms: stock
against block, both with `--attention-backend TRITON_ATTN`. Metric: `joules_per_token` at batch 1.
Expected change in joules per token at batch 1: **0.0 %, range -0.5 % to +0.5 %**. At batch 4: the same range.
Confidence: medium (0.55). Minimum runs: 5 cold spawns per arm, as above. LDS: expected about 40 KiB for the 3D
kernel at tile 32 and one stage (fits). Output match: a wider tile changes the order of the softmax and P.V sums
inside each segment, so some near-tie divergences are possible; expected match rate no lower than stock's own,
confidence 0.6. Citations: as above, plus `triton_gfx12.md` s.3 (vLLM PR #58225: narrower KV tiles won 2.25x on
RDNA prefill attention, so wider is not expected to help).

## Reasoning (why both estimates sit at zero)

1. **Which kernel runs.** At b1388b1 the ROCm attention priority puts `ROCM_ATTN` first, so the teacher today never
   calls the Triton unified attention kernel for its text decoder (`settings_inference_gfx1151.md`, "Attention
   backend and block size"). Both R12 arms therefore pass `--attention-backend TRITON_ATTN`; the stock arm is the
   teacher's build with that one flag, not the teacher's served configuration. With TRITON_ATTN and 4 KV heads,
   `seq_threshold_3D` is 128 / 4 = 32 (rounded to the nearest graph capture size), so decode at batch 1 to about 32
   runs the 3D kernel with TILE_SIZE 16 and 16 softmax segments, and prefill runs the 2D kernel with TILE_SIZE 32.
2. **How much of a decode step is attention.** Decode reads about 15.5 GB of int4 weights per step and takes about
   350 ms at 2.88 tok/s (baseline: pod C, one spawn, about 41.7 J per token gross at 120 W). The bench prompts are
   short (tens to about 900 prompt tokens with the images, plus 128 generated), so one attention layer reads at
   most about 1,000 tokens x 4 KV heads x 256 x 2 (K and V) x 2 bytes = 4 MB of KV cache, 16 layers about 65 MB:
   about 0.4 % of the bytes of a step (derived). Each 3D launch has a grid of 1 x 4 x 16 = 64 workgroups doing one to
   four tile iterations, so its time is launch- and latency-bound, tens of microseconds, about 16 launches plus 16
   `reduce_segments` per step: of the order of 1 ms of a 350 ms step (about 0.3 %, derived, unmeasured).
3. **What each lever can do to that share.** Two stages let the next K and V tile load while the current one is
   used, but with one to four iterations per segment there is little to overlap, and the prologue adds work; a kernel
   change of -10 % to +5 % is plausible. A 32-wide tile halves the iterations per segment and doubles each load;
   PR #58225 found the opposite direction (narrower) faster on RDNA, so -5 % to +10 % on the kernel. Either way the
   kernel moves at most about 10 % of about 0.3 % of the step: under 0.05 % of joules per token, far inside the
   +-0.5 % range written above, which is there for spawn effects the lever could trigger indirectly (start-up
   compilation, memory placement), not for the kernel itself.
4. **Watts.** SOCKET_POWER is the whole APU; a kernel this small cannot move the package power measurably, so joules
   per token follows decode time, and decode time is weight-read bound.
5. **Why run it anyway.** It is the first change that goes through the whole kernel loop on gfx1151 (patch kept apart
   from the build diff, rebuild recipe, LDS bytes recorded per touched kernel, token match before speed, R11 before
   and after with bands) at almost no risk to the outputs; a result of "no change" fixes the noise floor that the
   larger levers (KS1 linear kernel, KS2 speculative decoding) will be judged against. Where the attention share is
   larger (8k-token contexts, batch 16), the same patches become worth a second line; that is not this run.

## What would change these estimates

- If the LDS check shows the 3D kernel at 2 stages over 65,536 bytes, R12a cannot run: the line records the bytes and
  the OutOfResources failure as its result, and the estimate is void, not "slower".
- If stock with `--attention-backend TRITON_ATTN` does not start or does not match itself across two cold spawns at
  a rate the prompt set can show, neither line runs; the stock record is the result.
- If the bench shows two decode bands (the gfx1201 band lottery), each arm's J/token is compared within a band, and
  the band shares are written beside it.

## After the run (the dispatcher fills this in on T0)

For each line: measured change in joules per token at batch 1 and batch 4 (median over spawns, per band when there
are bands, with the spawn count and spread), decode tok/s beside it, the LDS bytes of each touched kernel, the match
rate against stock's own rate, the rebuild wall-clock, and the commit of the rows in `gpu/results/`. A slower result
is written the same way as a faster one.
