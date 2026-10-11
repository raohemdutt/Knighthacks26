# Circle chips: what a friend's chip receives, and what the ladder may assume

Research note for R44 (Stretch). It sits beside the code it explains, `lab/capacity.py` and `lab/ladder.py`, and
not under `lab/refine/`, which this task does not edit. Nothing here was run on a pod or a GPU: every number below
is the committed ladder's (measured on another chip, or predicted from that measurement) or a sample, and each says
which.

## 1. What the Circle shares: chips, never data

The Circle demo is friends lending their boxes' chips so the household's student trains faster, or a larger student
trains at all. The task page records the boundary (R44 page note, Sat 6:35 PM ET): `docs/spec1.md` lists
"fine-tunes on friends' pooled data" under superseded ideas and makes sharing between households stretch-only, so
R44 is filed as Stretch and pools chips, never data. `docs/spec1.md` is not in this checkout; the page note is the
source until it is.

## 2. What a friend's chip ever receives

Exactly what the AUP pods (C and D on the SJC hub) receive today, and nothing more:

- **Blurred keyframes.** An approved export from the box is `<dataset_id>/manifest.json`, `examples.jsonl` and
  `images/*.jpg`, and the images are "blurred keyframes only" (`contracts/learning.py`, the box and lab file
  formats; `lab/dataset.py`). The box blurs before it exports; the lab never sees an unblurred frame.
- **Text spans.** The training rows `lab/format_traces.py --split` writes: the rendered `text` and the `spans`,
  the character offsets of what the model generates in each reply turn. The loss sits on the spans and nowhere
  else (`lab/train.py`, README section "The LoRA trainer").
- **Stored nowhere.** On the pod, once the node check passes, `lab/train.py` deletes the dataset copy and the rows
  file however the run ends, and says in `refs.data_deleted` whether that worked; a run that cannot delete ends
  `data-left`, exit 1. What comes back is the adapter, the run's ledger row (hashes of the data, not the data) and
  its step telemetry.

A friend's box gets the same push, the same trainer and the same deletion. It receives no raw video, no unblurred
frame, no memory store, no events from the box's bus and no other household's rows; it keeps nothing after the run.
The approval rule holds as everywhere: the export to a friend's chip is an outside effect the phone approves.

## 3. What the capacity probe reads from a chip

`python -m lab capacity` reads the chip, never the data: the R33 chip sheet in its quick mode (copy bandwidth, bf16
matmul rate, versions, inside 110 s of GPU time) and the memory a new job can take (`bash lab/pod_a_teacher.sh facts`:
MemAvailable and the memory visible to torch). The record names the chip, its ROCm, its role and how long the probe
took against a two-minute budget. A friend's chip is probed the same way with `--node <name>` (role `friend`).

The measured gfx1151 sheet from pod D (`lab/refine/research/kernels/data/chipsheet-gfx1151-2026-10-11.json`, a full
run of 203 GPU seconds, so over the two-minute budget by design) reads as 42.7 TFLOPS best bf16 matmul (a 4B LoRA
step shape) and 943 GB/s best copy. The copy figure is the best over all sizes, as the placement plan defines it,
and its best size is 16 MiB, which likely sits in the chip's cache rather than its shared DRAM; the ladder does not
use bandwidth, but anyone reading the record as DRAM bandwidth should take the largest copy size instead.

## 4. What the ladder assumes, and what it does not

- **Fit.** A rung fits a chip when its LoRA step peak times 1.5 plus 4 GiB is within the chip's free memory: the
  same budget `lab/placement.py` gives a LoRA job (margin and host reserve, `placement.Rules`). On the sample's
  58 GiB free: Qwen3.5-4B needs 17.05 GiB (measured peak 8.7 GiB), Qwen3.5-9B 33.36 GiB (predicted 19.58 GiB),
  Qwen3.5-27B 92.09 GiB (predicted 58.73 GiB), so a 64 GB Strix Halo takes the 9B and not the 27B.
- **Rates.** The 4B's 0.51 samples/s was measured on one R9700 (gfx1201), another chip; the larger rungs divide it
  by their size ratio. Both are replaced by L3's runs on gfx1151 and R34's step profile. The ladder scales a rate
  by a chip's bf16 rate only when the chip's and the reference chip's rates are both measured.
- **More than one chip.** The placement plan decides which chips take a LoRA job now and never lets two LoRA jobs
  share a chip (both are compute-bound). So two chips run two different rungs at once, the larger one where it
  finishes soonest and the smaller one beside it as the comparison (the demo's smaller-versus-larger student).
- **Not assumed: one model across chips.** Data-parallel training of one model across pods would cut a run's time
  only if the pod-to-pod link carries each step's gradients fast enough. That is measured first, with a kill
  criterion, in `lab/R44_REAL_RUN.md` step 5; until it passes the ladder predicts no speedup from a second chip.

## 5. Open questions for the people

- Whether friends' boxes run the same trainer image (ROCm 7.13, gfx1151) or other chips: the ladder's rates assume
  the trainer's chip until a friend's chip has a capacity record and a measured run.
- Whether the Circle demo shows two pods as two friends (as planned) or one pod plus the box: that is Ilan's call
  and the Demo Plan's.
