# The live learning loop: run of show (L29)

One command starts two group fine-tunes on pod D, one after the other: act 1 "alone" is one member's 100 examples,
act 2 "with a friend" is two members' 200 pooled. Each act labels at the pod C teacher (16 streams), trains a LoRA
(8 short steps, a loss point per step), packages a Q4_K_M model and offers it. The box's Learning lab shows both runs
as they happen, and a comparison card on this machine shows the three power traces (teacher, trainer, box) and what
act 2 cost against act 1. Rehearsed tonight in a pod D slot: **232.2 s** from the command to both runs on the box.

## Before the segment (10 minutes ahead)

1. A GPU Queue slot for pod D, Running (`LAB_ACTOR=ilan2-agent LAB_TASK=L29 python -m lab slot request --task L29
   --minutes 15 --kind lora --what "live loop demo"`, then `python -m lab slot start --row <row>`). Queue behind
   anyone holding pod D (R65 may); never pre-empt.
2. `bash tooling/demo/live_loop.sh --check` from the repo root (Git Bash). It wants: box address set, the box's
   Learning lab answering 200, the pod C teacher queue at 0 running and 0 waiting, pod D's `check ok`, `jobs:` empty.
3. Open the box's Learning lab (`http://<box>:8000/lab/`, Training run tab) on the screen now: the page takes up to
   two minutes to load the first time, and `GET /api/learning/lab` itself took 20 s once tonight.
4. Optional second screen: the Circle shared-brain card, or the standalone power page
   (`python tooling/demo/power_proxy.py`, branch `ctx/docs/amd-demo-setup`, http://127.0.0.1:8787).

## The command

```bash
bash tooling/demo/live_loop.sh --slot <GPU Queue row>                # starts now; the stopwatch starts at the command
bash tooling/demo/live_loop.sh --slot <GPU Queue row> --on-approval   # starts when the phone approves an export
```

The relay prints one stopwatch line per stage and serves the comparison card at http://127.0.0.1:8788/. It keeps a
copy of everything in `../L29-data/<stamp>/` and writes `live-loop.energy` rows to the lab's ledger. Run ids are
`live-<MMDD-HHMMSS>-solo` and `live-<MMDD-HHMMSS>-duo`; the first line printed names the stamp.

Options that matter: `--steps 8` (loss points per act), `--no-scores` (see "the stand-in gate" below), `--every 5`
(seconds between reads of pod D's run folder), `--hold-minutes 30` (how long the card stays up afterwards).

## The approval path (`--on-approval`)

The Teach screen's export card proposes `export_training_data`; when the phone approves it, the box's
`DatasetExporter` (contexts/learning/application/export.py) writes the export and publishes
`learning.dataset.exported`, and the gateway fans every event out on `/ws`. With `--on-approval` the relay listens on
that socket and starts the loop the moment the event lands, then times from there. Off by default, and nothing on the
box changes: no gateway restart, no edit to `contexts/gateway/bootstrap.py`. The run trains on the fixture pool
already staged on pod D (`~/l10/bigpool.tar.gz`: ilan and dom, 100 each), not on the export just approved; the
approval is the trigger. Not rehearsed live tonight: an approval is a data export and only a person approves one. If
the phone approval is not ready, approve on the phone first for the story, then run the command without the flag.

## Rehearsal, Sun 2026-10-11 4:03 AM ET (stamp 1011-040342, slot 3f696ef5...d444)

| From the command | Stage | Seconds |
| --- | --- | --- |
| 2.7 s | pod D has both acts started (run-script 1.2 s) | |
| 3.1 to 34.8 s | act 1 labeling, 100 examples at pod C, 16 streams | 32.1 |
| 34.8 to 88.7 s | act 1 training, 8 steps (model load and a 30 s first step, then about 0.6 s a step) | 53.9 |
| 35.0 to 96.8 s | act 2 labeling, 200 examples (pod C, while pod D trains act 1) | 61.8 |
| 88.7 to 159.8 s | act 1 packaging (merge, GGUF, Q4_K_M) and offer, on pod D's CPU at low priority | 71.1 |
| 96.8 to 165.6 s | act 2 training, 8 steps (on pod D's GPU while act 1 packages) | 68.8 |
| 165.6 to 224.8 s | act 2 packaging and offer | 59.2 |
| 232.2 s | the box's Learning lab lists both runs (manifest, steps, stand-in scores posted) | |

Act 1 took 157.0 s, act 2 189.8 s, the whole loop **232.2 s**. The first rehearsal (stamp 1011-035827) took 246.4 s;
two cuts brought it under 240: the relay no longer waits 3 s for the samplers before starting the pod (they read 90 s
of history anyway), and act 1's packaging runs at nice 15 so act 2's training is not slowed by it (act 2 training
went from 84.6 s to 68.8 s). After that rehearsal one more fix went in, not yet rehearsed: the box listing check
(`GET /api/learning/lab`, 20 s once) runs in the background for act 1, so it can no longer hold up reading pod D (it
delayed act 2's last loss points by 21 s tonight). The stand-in gate is used (no real gate: atlas-base is not
packaged on pod D), so the gate costs nothing; the offer is written (under a second) and kept.

Cuts already made against the plan: act 2 does not wait for act 1 to finish packaging (pod D alone, serial, is 240 s
of work), and act 1's trainer reads its split with a 25-example held-out floor (100 examples freeze 25; lab/train.py
otherwise asks for the 50 a deploying gate needs; act 1 never deploys).

## What the audience sees, second by second

| Time | Box's Learning lab | Comparison card / power |
| --- | --- | --- |
| 0 s | "Approve" on the phone, or the command | card clock starts |
| 3 to 35 s | nothing new yet | pod C jumps from 11 W to about 115 W: the teacher labels act 1 |
| 35 s | | pod C stays up (act 2 labeling); pod D rises from 10 W as act 1 loads |
| 80 to 90 s | act 1 appears (Training), 8 loss points land, loss 4.75 to 0.40, power trace under it | |
| 89 s | | pod D: act 1 packaging on the CPU, 70 to 100 W |
| 97 s | | pod C drops back to idle: all 300 labels are in; pod D carries both act 2 training and act 1 packaging |
| 160 s | act 1 shows its package (sha256, Q4_K_M) and the stand-in gate | |
| 160 to 186 s | act 2 appears, its 8 loss points (5.83 to 2.08) and power | |
| 166 to 225 s | | pod D: act 2 packaging, then back to 10 W |
| 232 s | both runs listed | the table fills in: act 2 against act 1 |

The box's own trace stays in its 5 to 18 W pattern the whole time: the work happens on the pods.

## The comparison (rehearsal 1011-040342)

Gross joules (T4's rule: integrated by `contexts/telemetry/domain/energy.integrate`, nothing subtracted), idle beside:
10.27 W calibrated per pod (measured quiet: pod C 11 W, pod D 10 W), the box 15.76 W calibrated, bimodal 5 and 18 W.
Pods are APU socket power (CPU and GPU together). Where pod D's windows overlap (act 2 training during act 1
packaging, 75 s), the stretch is split evenly between the two (T4's split, no token weights), so the stages add up.

| | alone (100) | with a friend (200) | ratio |
| --- | --- | --- | --- |
| act seconds | 157.0 | 189.8 | 1.21x |
| labeling J, pod C | 3723 | 6779 | 1.82x |
| labeling J per token (prompt and completion) | 0.330 | 0.305 | 0.92x |
| labeling J per example | 37.2 | 33.9 | 0.91x |
| training J, pod D | 2727 | 3421 | 1.25x |
| training J per step (8 steps each) | 341 | 428 | 1.25x |
| training J per training example | 36.4 | 22.8 | 0.63x |
| packaging J, pod D | 3334 | 4384 | 1.31x |
| both pods, J | 9784 | 14584 | 1.49x |
| J per example, every stage | 97.8 | 72.9 | 0.74x |
| the box, J during the act | 1151 | 1725 | 1.50x |

The line to say: with a friend the lesson is twice as big for one and a half times the energy, about a quarter less
per example. Labeling barely amortizes (both acts already run 16 streams at the teacher); training and packaging do,
because the fixed costs (loading the 4B model, the first step, merging and quantizing) are paid once per run whatever
the pool's size. Training J per step is higher in act 2 because act 1's packaging shared pod D at the same time.

## The stand-in gate

The run posts the stand-in gate's scores so the run reaches "gated" on the box. The Before/after section then lists
questions with the answers "fake miss" and "fake answer": keep the page on the Training section, or start with
`--no-scores` (not rehearsed: the run may then read "Training" on the box until a real eval arrives).

## Fallback: tonight's recorded runs

If pod D has no slot, pod C is busy, or the run fails, show the recorded rehearsal instead (no pod touched):

```bash
bash tooling/demo/live_loop.sh --fallback fixtures/learning/live-loop-2026-10-11
```

It posts both recorded runs to the box (steps, manifest, stand-in scores; a box that has them answers "0 new") and
serves the same card, with tonight's traces and table, at http://127.0.0.1:8788/. Screenshots of the box's Learning
lab for both runs and of the card are in the same folder (`learning-lab-solo.png`, `learning-lab-duo.png`,
`comparison-card.png`). The first rehearsal's numbers (246.4 s) are in `first-rehearsal-246s/`. Tonight's single
pooled L10 run (s3, 4 min 3 s) is in `fixtures/learning/group-run-2026-10-11/`.

## Three failure modes

1. **Pod C busy.** `--check` shows the teacher queue above 0, or act 1's labeling line runs past 40 s (or act 2's past
   70 s). The group brain shares that teacher and keeps priority. Do not restart the teacher. If the queue is busy
   before the segment, use the fallback. If it slows mid-run, let it finish; narrate the traces while the box shows
   act 1, and say the total.
2. **Pod D slot not free.** `python -m lab slot status` shows another Running row on GPU B. Queue behind it, never
   pre-empt; if it will not free up before the segment, run the fallback. The pod script refuses to start without
   a slot row (`--slot`), and a stamp is used once (`~/l29/<stamp>`).
3. **The box refuses a row.** The relay's step line reads `box 400` (or 401, 404, 413) instead of `box 200`. The run
   keeps going on pod D and the card keeps drawing (it reads the pod, not the box). Point the screen at the card.
   Afterwards the relay's copies post once with `python -m lab lab-push --box http://<box>:8000 --run-dir
   ../L29-data/<stamp>/<act>` (and `--manifest`, `--eval` from the same folder). A box that answers nothing (`box
   did not answer`) is retried every round.

## Files

- `tooling/demo/live_loop.sh`: the one command (calls `live_loop.py`).
- `tooling/demo/live_loop.py`: the relay, the stopwatch, the samplers' and the box's power traces, the energy table,
  the card server, `--check`, `--on-approval`, `--fallback`.
- `tooling/demo/live_loop_pod.sh`: pod D's side (`check`, `start`), run through `python -m lab.controller run-script`;
  it calls `lab/pod_b_group_finetune.sh run` once per act with `GATE=fake MIN_HELD_OUT=1 LABEL_WORKERS=16`,
  `--grad-accum 1`, `--max-steps 8` and a trainer wrapper that marks the stages.
- `tooling/demo/live_loop_card.html`: the card.
- `fixtures/learning/live-loop-2026-10-11/`: the rehearsal (stages, three power traces, both runs' steps, manifests,
  scores, comparison, relay log, screenshots).
- `tests/test_tooling_live_loop.py`: stage arithmetic and pod D's split on the recorded rehearsal.
