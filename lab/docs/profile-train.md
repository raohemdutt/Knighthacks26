# Training step profile: `lab/profile_train.py` (R34)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab profile-train -h`.

Where each optimizer step of the student LoRA run goes on gfx1151: the run this explains is L16, the first real
fine-tune (pod D, 18 steps in 785 s, about 44 s a step, peak 11.1 GiB, no power sampler). Built and tested on the box;
the pod run is two GPU Queue slots on pod D, done by the dispatcher.

```
python -m lab profile-train self-check --quick               # the box: parser and writer on both recorded fixtures
python -m lab profile-train run --cpu --dataset D --rows R --model TINY --steps 3 --results DIR   # the box: rehearsal
python3 -m lab.profile_train run --dataset datasets/tool-calling-fixture --rows datasets/rows.jsonl \
    --model ~/models/Qwen3.5-4B --out ~/runs/r34-profile --results ~/runs/r34-profile/result --no-ledger   # pod D
python -m lab profile-train record <the pulled train-profile-gfx1151-<date>.json>   # the box: file it, ledger row
```

- **The run.** lab/train.py, unchanged but for an optional callbacks list, trains `--steps` optimizer steps (20) with
  its pinned baseline (batch 4 by gradient accumulation 4, r 16, gradient checkpointing, SDPA, fused AdamW) in a
  process of its own. A trainer callback puts forward hooks on every Gated DeltaNet layer (`*.linear_attn`), attention
  layer (`*.self_attn`), MLP, LoRA matmul (`*.lora_A.*`, `*.lora_B.*`) and the embedding and LM head, each opening a
  record_function range `atlas::<kind>::<module>`, and starts the torch profiler (record_shapes and profile_memory
  on, with_stack off) when training begins, so model loading stays out of the first step. The profiler's schedule
  is one active window of `--steps` steps, which is what makes it mark each step with a `ProfilerStep#<n>` range
  (without a schedule torch records no step ranges); `--active N` profiles only the last N steps when the trace must
  be smaller. With `rocprofv3` on PATH a second pass trains the same steps under `rocprofv3 --kernel-trace
  --memory-copy-trace` without the torch profiler. `--passes torch` or `--passes rocprof` runs one pass, and a second
  call with the same `--out` adds the other. Before training it installs `lab/requirements-train.txt`'s pins around
  the pod's torch exactly as `lab/train.py` does (`--no-install` skips it).
- **Its data.** On the pod lab/train.py deletes its dataset copy and rows however a run ends, so each pass trains on
  a copy of its own under `<out>/data-<pass>/`, and `run` deletes the originals after the passes (`data_deleted` in
  the answer). A `--cpu` rehearsal keeps its data.
- **How the trace is read.** Forward ops belong to the innermost `atlas::` range around them (LoRA sits inside its
  layer's range and is never counted there too). Backward ops run on the autograd thread outside every range: each
  belongs to the layer of the forward op it differentiates, through the profiler's `fwdbwd` flow events, and the ops
  a backward function calls inherit it. `Optimizer.step#...` and `Optimizer.zero_grad#...` are the optimizer's,
  `enumerate(DataLoader)#...` data loading's. A kernel or copy belongs to the op whose runtime call launched it
  (correlation id). On a GPU trace a share is kernel time over step wall time (data loading: its wall time; gpu_idle:
  no kernel or copy running); on a CPU trace it is op self time. The first two steps of six or more are warm-up and
  left out of the means.
- **The result**, `lab/refine/research/kernels/data/train-profile-gfx1151-<date>.json` and `.md`
  (`train-profile-cpu-<date>` for a rehearsal): time per step and its spread; the share and ms per step of the Gated
  DeltaNet layers, attention, the LoRA matmuls, the MLPs, the embedding and head, the optimizer, data loading,
  host-device copies, other work and GPU idle; the top 25 kernels by self time with launch counts and the bucket
  each mostly ran in (CPU ops by self time on a CPU trace), and rocprofv3's own table when that pass ran; bytes and
  FLOPs per step from the recorded shapes (leaf ops' inputs, outputs not counted, so a lower bound; 2MNK of every
  mm, addmm, bmm and baddbmm) over the step time and over kernel time, against the chip sheet's device-to-device copy
  over its largest buffer and its best bf16 matmul (R33's `chipsheet-gfx1151-2026-10-11.json`: about 208 GB/s);
  peak allocated and reserved memory from the allocator samples and the trainer's own max_memory_allocated; whether
  the flash-linear-attention kernels ran (kernel names, and lab/train.py's own `deltanet_kernel`), with the GDN share
  named the fallback share when they did not; and `trainer.samples_per_s` measured as `metrics.jsonl:13` measures
  it, under the profiler, so a lower bound. One `profile_train` ledger row on node pod-b (`record` on the box writes
  it for a pod result; `ingest RUN_DIR` does the same from a whole run folder).

**Pod runbook (the dispatcher's).** The rows first, on the box: `python lab/format_traces.py
fixtures/learning/tool-calling-labels --out rows.jsonl --split` (84 rows today; the unlabeled
`fixtures/learning/tool-calling-fixture` still answers "no example has a target yet": the teacher's labels are in
`tool-calling-labels`, L13, PR 283). Its dataset_id is `tool-calling-fixture`, which is where `push-dataset` puts it:

```
python -m lab.transfer push-dataset --account D fixtures/learning/tool-calling-labels --to datasets
python -m lab.transfer push-file --account D rows.jsonl datasets/rows.jsonl
# lab/ and contracts/ to the same Jupyter root, as for lab/train.py
# slot 1, 30 minutes, pod D (B exclusive): the torch pass
python3 -m lab.profile_train run --passes torch --dataset datasets/tool-calling-fixture --rows datasets/rows.jsonl --model ~/models/Qwen3.5-4B --out ~/runs/r34-profile --results ~/runs/r34-profile/result --no-ledger
# slot 2, 30 minutes, pod D (B exclusive): the rocprofv3 pass into the same folder (push the data again first)
python3 -m lab.profile_train run --passes rocprof --dataset datasets/tool-calling-fixture --rows datasets/rows.jsonl --model ~/models/Qwen3.5-4B --out ~/runs/r34-profile --results ~/runs/r34-profile/result --no-ledger
# back on the box
python -m lab.transfer pull-file --account D runs/r34-profile/result/train-profile-gfx1151-<date>.json lab/state/r34/train-profile-gfx1151-<date>.json
python -m lab profile-train record lab/state/r34/train-profile-gfx1151-<date>.json
```

Slot sizes from L16's 44 s a step: 20 steps are about 15 minutes, plus model load and the profiler's own cost
(record_shapes and the allocator samples), plus writing and reading the trace: about 25 minutes for the torch pass and
about 20 for the rocprofv3 pass, so 30 minutes each. The trace is large: the CPU rehearsal's two-layer model records
about 20,000 ops a step, so a 32-layer step is several hundred thousand, and 20 steps a JSON trace of a gigabyte or
more; if the first slot runs short, `--active 8` keeps the last 8 of the 20 steps. Each pass deletes its own data
copy, and the run deletes the pushed originals once its passes are done.

Tests: `tests/test_lab_profile_train.py` (the parser on `fixtures/lab/train_profile/rocm`, a synthetic GPU trace
whose totals per step can be checked by hand, written by `make_rocm_synthetic.py`: GDN 2.6 ms, attention 0.74, MLP
0.5, LoRA 0.045, optimizer 0.3, copies 0.02, data loading 0.2, steps 10.0, 10.5 and 9.5 ms; links broken one at a
time; the recorded CPU trace under `fixtures/lab/train_profile/cpu`, two steps of the tiny Qwen3.5 cut by
`trim_cpu_trace.py`; the result, its checks, the Markdown, the ledger row and the run's plumbing with the trainer
replaced) runs everywhere; `tests/test_lab_profile_train_cpu.py` runs the whole verb on the CPU and is skipped where
the training stack is not installed (CI). Not yet run on a pod.
