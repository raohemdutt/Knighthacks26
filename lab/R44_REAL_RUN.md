# R44 real run: the chip probe on pods C and D, one chip against two, and the data-parallel experiment

The offline half of R44 (`lab/capacity.py`, `lab/ladder.py`, their tests on the fake chip client) is on this
branch. Everything below needs the pods, a GPU Queue slot per pod, the box's approved export and the team's
credentials, so it was not run from the build session. Run it after **L3** (the trainer runs on gfx1151) and
**L16**, as the R44 scope note says, in this order. Pod C is account C, role pod-a (the teacher's pod); pod D is
account D, role pod-b (the trainer). In the Circle demo they stand in for two friends' boxes.

Every step runs inside a GPU Queue slot on the pod it uses (`python -m lab slot request ...`, then `slot start`),
with the pod sampler running (`python -m lab.controller run-script --account <C|D> --with lab/pod_power.py
lab/pod_power.sh start <gpu-a|gpu-b>`). Nothing here sends data anywhere but the two pods, and the dataset push is
the approved export (see `lab/CIRCLE_CHIPS.md`).

## Step 1. Probe each chip (under two minutes each)

On each pod in turn (shown for D; for C use `--account C`, `--node pod-a`):

```
time (
  python -m lab.controller run-script --account D lab/pod_a_teacher.sh facts                          # save the printed facts lines as pod-b.facts.txt
  python -m lab.controller run-script --account D --timeout 300 --with lab/kernels/chipsheet/chipsheet.py --with lab/kernels/chipsheet/chipsheet_format.py --with lab/kernels/chipsheet/probe.hip --with lab/pod_power.py lab/kernels/chipsheet/pod_chipsheet.sh run --node pod-b --quick
  python -m lab.transfer pull-file --account D chipsheet/chipsheet-gfx1151-<date>.json data/chipsheet-gfx1151-<date>-quick-pod-b.json
)
python -m lab capacity --sheet data/chipsheet-gfx1151-<date>-quick-pod-b.json --facts pod-b.facts.txt --node pod-b --out data/capacity-pod-b.json
```

Done when: both records exist, `probe.within_budget` is true in each, and the `time` line plus the sheet's
`gpu_seconds` stay under 120 s per pod. Record the wall-clock in the evidence file (step 4). If the quick sheet
alone runs past 110 s the runner writes the late sections as skipped; the record then shows unknowns, not guesses.

## Step 2. The ladder's pick

```
python -m lab.ladder --capacity data/capacity-pod-b.json --samples <train rows> --epochs <L3 epochs> --json > data/ladder-one-chip.json
python -m lab.ladder --capacity data/capacity-pod-a.json --capacity data/capacity-pod-b.json --samples <train rows> --epochs <L3 epochs> --teacher-share 0 --json > data/ladder-two-chips.json
```

`--teacher-share 0` says no teacher runs on these chips, as on a friend's box; stop the teacher on pod C for the
step, or leave it serving and pass `--teacher-on pod-a` to see the plan keep LoRA off pod C. Today's prediction on
the sample (58 GiB free per pod): Qwen3.5-9B on one pod, Qwen3.5-4B beside it on the other.

## Step 3. One chip against two (the demo evidence)

The same fine-tune, the student on the L3 rows, then the larger rung the ladder picked:

```
# one chip: 4B then 9B, one after the other, on pod D
python3 lab/train.py --dataset datasets/<dataset_id> --rows datasets/rows.jsonl --model ~/models/Qwen3.5-4B --run-id r44-one-4b
python3 lab/train.py --dataset datasets/<dataset_id> --rows datasets/rows.jsonl --model Qwen/Qwen3.5-9B --base-repo Qwen/Qwen3.5-9B --run-id r44-one-9b
# two chips: the same two runs at once, 9B on pod D and 4B on pod C (push the dataset and rows to both pods first)
python3 lab/train.py ... --model Qwen/Qwen3.5-9B --base-repo Qwen/Qwen3.5-9B --run-id r44-two-9b                    # on pod D
python3 lab/train.py ... --model ~/models/Qwen3.5-4B --power-node gpu-a --run-id r44-two-4b                        # on pod C
```

Each run leaves its `train` ledger row with `trainer.samples_per_s`, its peak memory and its wall time; pull the
rows back with the runs. Check the 9B's peak against the ladder's predicted 19.58 GiB and its rate against
0.2267 samples/s, and replace the predicted rungs in `lab/ladder.py` with the measured ones (kind measured, source
the ledger row) in a follow-up commit.

## Step 4. Smaller against larger student, and the evidence file

Run the L-chain eval on the held-out split for the 4B and the 9B adapters (the same eval set, the same gate), then
commit one data file, `lab/r44_evidence/one-vs-two-<date>.json`, holding: both capacity records and the probe
wall-clocks; per run the model, chip, run id, run time, samples per second and peak memory with the ledger row
each came from; the one-chip total time (the two runs back to back) against the two-chip time (the longer of the
two at once); the eval answers and scores of the 4B against the 9B; and the ladder's predictions beside the
measurements.

## Step 5. Data-parallel training of one model across pods C and D (its own experiment, with a kill criterion)

The ladder assumes no speedup from a second chip for one model. This step decides whether that changes.

1. **Measure the link.** From pod D, find pod C's address (`hostname -I` on pod C) and time a 1 GiB transfer
   between them, five times each way, with a stdlib socket pair (a `python3 -c` listener on pod C, a sender on
   pod D that writes 1 GiB and prints the seconds). Record the median GB/s and the round trip of a 1-byte message.
   If the pods cannot reach each other directly and the only path is through the hub proxy (`lab/proxy.py`), the
   experiment is **killed** here: a public-internet hop per optimizer step is not a training link.
2. **Kill criterion.** Let G be the bytes of the LoRA gradients per optimizer step (the trainable parameter count
   PEFT reports at the run's start, times 4 bytes for fp32 gradients), T1 the one-chip time per optimizer step from
   step 3 (effective batch 16 / samples per second), B the measured link bandwidth and L the measured round trip.
   Two chips with data parallelism take at least T2 = T1 / 2 + G / B + L per step (a two-node all-reduce sends G
   each way). **Kill if T1 / T2 < 1.25**, that is if the link cannot give at least a 1.25x speedup on paper. A
   worked example, predicted only: the 4B at 0.51 samples/s has T1 = 31.4 s; with G of about 130 MB (about 33M
   trainable parameters at r 16 on all linear layers, to be replaced by PEFT's count), the criterion needs
   G / B + L <= 9.4 s, so a link below about 14 MB/s kills it.
3. **If it survives:** run 50 optimizer steps of the 4B with `torchrun --nnodes 2` over the measured link (gloo or
   RCCL, whichever the image supports between the pods), seed and rows as in step 3, and **kill if the measured
   two-node samples per second is under 1.25x the one-chip samples per second** or the loss after 50 steps is
   outside the one-chip run's loss plus three of its run-to-run standard deviations (R36's rule for a setting that
   breaks the math).
4. Record the link numbers, the criterion's arithmetic and the verdict (killed or kept, and at which check) in the
   evidence file, and only then let `lab/ladder.py` predict a two-chip speedup for one model.
