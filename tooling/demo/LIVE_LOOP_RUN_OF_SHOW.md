# The live learning loop: run of show (L29)

One command starts two group fine-tunes, one after the other: act 1 "alone" is one member's 12 examples, act 2 "with
a friend" is two members' 24 pooled (`--solo-examples`, `--duo-examples` per member; the budget below sets them, and
15 and 15 is the stretch).
Each act labels at the teacher (16 streams), trains a LoRA (8 short steps, a loss point per step), packages a Q4_K_M
model and offers it. The teacher is the 122B pair, served pipeline-parallel across pods C and D, its API on pod C's
port 8000 as "teacher" and "group-brain" (the same URL and token as the 27B before it). Act 1 trains on pod D alone;
act 2 trains on both pods at once (R65's two-pod LoRA: rank 0 on pod D, rank 1 on pod C, each on half the rows,
int8 deltas swapped through the hub proxy every 4 steps), beside the pair. On the box, act 2 reads "2 members" and
its training stage names both pods. The box's Learning lab shows both runs
as they happen (Training run tab: "Running", the loss points, the stage strip labeling, training, packaging, offered),
and its Compare tab ("Alone vs together", `/lab/?tab=compare`) shows the three power traces (teacher, trainer, box) on
one time axis and what act 2 cost against act 1. **The Compare tab on the box's /lab is where the audience looks.**
Everything on it is what this run posted to the box; no recording and no fixture is ever shown. The 27B rehearsal
(pod D alone, 100 and 200 examples) took **232.2 s**; the pair-and-two-pods version is not rehearsed yet, and its
budget is below.

## The 4-minute budget (pair teacher, act 2 on two pods)

The pair's labeling was measured exactly as the loop calls lab/label.py (16 workers, 32 fixture examples, all
accepted): **2.34 s a labeled example** of wall time (34.2 s an example within one stream, so 16 workers is the right
setting; max-num-seqs 32 was never reached), 1,170 tokens an example, both pods about 108 W on average and 127 W at
peak while labeling, about **505 J a labeled example** across the two pods. The 27B labeled at about 0.32 s and 37 J an
example (the rehearsal: 3,723 J for 100). So a label from the pair costs about 7x the time and 14x the energy: the
honest price of a brain four and a half times bigger (122B against 27B). The loop still finishes under four minutes,
because the acts are sized to it. The other stage times are the rehearsal's: act 1's training 54 s (model load and a
30 s first step), packaging 71 s and 59 s, 3 s to start the pods, about 10 s for the relay's last read and posts.

| From the command | What runs | Budget |
| --- | --- | --- |
| 0 to 3 s | pod C's rank 1 waiter and pod D's run start (two run-scripts at once) | 3 s |
| 3 to 31 s | act 1 labels 12 examples at the pair | 12 x 2.34 = 28 s |
| 31 to 98 s | act 1 trains on pod D, which also runs half the pair: the rehearsal's 54 s plus 25 percent | 67 s |
| 31 to 87 s | act 2 labels 24 examples at the pair | 24 x 2.34 = 56 s |
| 87 to 132 s | act 2 trains on both pods: handoff about 3 s, both ranks load in parallel (30 s first step), 8 steps on half the rows, 2 rounds of 0.5 to 0.9 s, a final sync; the 15 s relay timing is skipped (`--no-relay-timing`) | 45 s |
| 98 to 169 s | act 1 packages on pod D's CPU (nice 15) | 71 s |
| 132 to 202 s | act 2 packages, slowed while act 1 packages: the rehearsal's 59 s plus 11 | 70 s |
| 202 to 212 s | the relay's last read, the manifest, scores, power windows and comparison posted | 10 s |

So about **212 s**, 28 s under 4 minutes. In one line: the run ends at about 3 + r x (n1 + 2 x n2) + 125 s (45 s
two-pod training, 70 s packaging, 10 s relay), where r is the seconds a label, n1 act 1's examples and n2 act 2's per
member. At r = 2.34:

| Sizes | n1 + 2 x n2 | End | |
| --- | --- | --- | --- |
| 12 and 12 (the default) | 36 | 3 + 84 + 125 = about 212 s | 28 s spare |
| 15 and 15 (the stretch: `--solo-examples 15 --duo-examples 15`) | 45 | 3 + 105 + 125 = about 233 s | 7 s spare |
| 50 and 50 (the first budget, at the guessed 0.63 s) | 150 | 3 + 351 + 125 = about 479 s | far over |

With 12 and 12, act 2's labels are in at about 87 s while act 1 still trains on pod D until about 98 s, so rank 0 shares
pod D's GPU for about 11 s; if that slows act 2's training by as much, the run ends near 223 s, still under. Each act
stays trainable: 12 examples hold out 3 and train on 9 rows, act 2's 24 hold out 6 and give each rank 9 rows, which is 3
steps an epoch at batch 4, 9 over the 3 epochs, so the 8 steps are reached. `--serial` (act 2 trains only after act 1
has packaged, for memory) adds about 75 s: at 12 and 12 that is about 287 s, over, so use it only off stage.

Memory: the pair runs at gpu-memory-utilization 0.65, which leaves about 22 GB a pod; a rank of the two-pod LoRA
peaked at about 11 GB in R65, and its memory guard stops the rank (never the teacher) under 3 GiB free.

## Before the segment (10 minutes ahead)

1. A GPU Queue slot for each pod, Running: pod D for both acts and pod C for act 2's rank 1 (`LAB_ACTOR=ilan2-agent
   LAB_TASK=L29 python -m lab slot request --task L29 --minutes 15 --kind lora --what "live loop demo"`, then
   `python -m lab slot start --row <row>`, once per pod). Queue behind anyone holding a pod; never pre-empt.
2. Once per pod, before the night: both pods hold the L10 bundle from current main (`lab/pod_b_group_finetune.sh
   bundle` and `prep`, on account D and on account C), so `lab/r65_local_sgd.py --no-relay-timing` is there, and both
   are on the tailnet as pod-c and pod-d (`lab/pod_tailscale.sh up pod-c|pod-d`). The pair is up (the R66 seat's
   launch), answering on pod C's port 8000 with a Qwen3.5-122B checkpoint.
3. `bash tooling/demo/live_loop.sh --check` from the repo root (Git Bash). It wants: box address set, the box's
   Learning lab answering 200, the teacher queue at 0 running and 0 waiting, `check ok` on both pods with
   `two-pod LoRA ... yes` and both tailnet peers listed, `jobs:` with nothing but the pair's vLLM.
4. Open the box's Compare tab on the screen now: `http://<box>:8000/lab/?tab=compare`. Until this run's act 2 has
   finished it says the comparison is not there yet and keeps looking (the route answers 404 until then, by design).
   Keep a second tab on the Training run tab (`http://<box>:8000/lab/`) for the live loss and the stage strip. The
   page takes up to two minutes to load the first time, and `GET /api/learning/lab` itself took 20 s once.
5. Optional second screen: the Circle shared-brain card.

## The command

```bash
bash tooling/demo/live_loop.sh --slot <pod D row> --slot-c <pod C row>                # starts now
bash tooling/demo/live_loop.sh --slot <pod D row> --slot-c <pod C row> --on-approval  # when the phone approves
```

Fallbacks, each one flag: `--teacher-expect Qwen3.8-27B` when the 27B serves on pod C instead of the pair (its
energy is then pod C's alone; `--teacher-expect "Qwen3.5-122B|Qwen3.8-27B"` accepts either), and `--duo-pods 1`
(no `--slot-c` needed) when pod C cannot train: act 2 then trains on pod D alone, as rehearsed.

The relay prints one stopwatch line per stage and posts to the box as it goes: each act's stage lines as they appear
(`kind=stage`), its steps (as before), and when an act finishes its power windows (`kind=power`) and the comparison so
far (`kind=compare`); at the end both again with the last seconds of samples. It keeps a copy of everything in
`../L29-data/<stamp>/` and writes `live-loop.energy` rows to the lab's ledger. Run ids are `live-<MMDD-HHMMSS>-solo`
and `live-<MMDD-HHMMSS>-duo`; the first line printed names the stamp and the Compare tab's address.

Options that matter: `--steps 8` (loss points per act), `--solo-examples 12` and `--duo-examples 12` (per member;
see the budget), `--no-scores` (see "the stand-in gate" below), `--every 5` (seconds between reads of the pods'
run folders), `--serial` (act 2 trains after act 1 has packaged). `--card` also serves the old local card at http://127.0.0.1:8788/
(off by default; not part of the show).

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

| Time | Box's Training run tab | Box's Compare tab |
| --- | --- | --- |
| 0 s | "Approve" on the phone, or the command | "not there yet", looking again |
| 3 to 35 s | act 1 appears, Running, stage strip on labeling | (pod C jumps from 11 W to about 115 W: the teacher labels act 1) |
| 35 s | | pod C stays up (act 2 labeling); pod D rises from 10 W as act 1 loads |
| 80 to 90 s | act 1 appears (Training), 8 loss points land, loss 4.75 to 0.40, power trace under it | |
| 89 s | | pod D: act 1 packaging on the CPU, 70 to 100 W |
| 97 s | | pod C drops back to idle: all 300 labels are in; pod D carries both act 2 training and act 1 packaging |
| 160 s | act 1 shows its package (sha256, Q4_K_M) and the stand-in gate | |
| 160 to 186 s | act 2 appears, its 8 loss points (5.83 to 2.08) and power | |
| 166 to 225 s | | pod D: act 2 packaging, then back to 10 W |
| 232 s | both runs listed, act 2 offered | the three traces on one axis and the table: act 2 against act 1 |

The Compare tab fills in once act 2 is offered (the relay posts act 1's windows when act 1 finishes and the whole
comparison at the end); the trace column above says what the traces will show for each stretch. The box's own trace
stays in its 5 to 18 W pattern the whole time: the work happens on the pods.

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

## No recorded fallback

There is no replay in the show. The screens show only what the box records live; if the run cannot happen, say so and
show the box as it is. `live_loop.py --fallback` stays in the code for tests and for repairing a box after a failed
post, and the demo never uses it; the box would not show those runs anyway (the Learning lab never serves a run a
committed rehearsal names, nor anything under fixtures/). The rehearsal's files under
`fixtures/learning/live-loop-2026-10-11/` are test data only.

## Act 2 across two pods, step by step

1. Pod D's act 2 start line names `"members": 2`, `"pods": ["pod-d", "pod-c"]` and the part run
   `live-<stamp>-duo.pod-c`; the box turns it into a `learning.loop.act` and shows "2 members" and both pods.
2. When act 2's labels are in, pod D's trainer wrapper splits the training rows (even lines rank 0, odd lines rank 1)
   and publishes rank 1's half with the trainer's dataset copy on a handoff server on 127.0.0.1:29610, which only
   pod D's account token opens through the hub proxy.
3. Pod C's waiter (started with pod D's token in `l29_td`, deleted on every way out) fetches the handoff, checks its
   sha256, and trains rank 1; pod D trains rank 0. Both write telemetry; the relay posts pod C's steps and its two
   stage lines (`rank1_train`, `rank1_trained`) under the part run id.
4. The ranks agree over Tailscale and swap their int8 deltas through the hub proxy every 4 steps; a last round leaves
   both on the same weights, and pod D packages rank 0's adapter as before. lab/train.py deletes each pod's data copy
   when it ends, the waiter deletes what is left, and pod D deletes the handoff.
5. The comparison's act 2 row sums both pods' training joules and keeps each in `training_pods`; labeling's joules are
   both pods' (the pair), with `labeling.by_node` beside. Each pod's overlapping windows are split evenly.

## Four failure modes

1. **The teacher busy.** `--check` shows the teacher queue above 0, or act 1's labeling line runs past 40 s (or act 2's
   past 75 s). The group brain shares that teacher and keeps priority. Do not restart the teacher. If the queue is busy
   before the segment, wait for it or skip the segment; never replay a recording. If it slows mid-run, let it finish;
   narrate the Training run tab while the box shows act 1, and say the total.
2. **A slot not free.** `python -m lab slot status` shows another Running row on a pod. Queue behind it, never
   pre-empt; if pod C's will not free up, run with `--duo-pods 1` (act 2 on pod D alone); if pod D's will not, skip
   the segment. The pod script refuses to start without a slot row (`--slot`, `--slot-c`), and a stamp is used once
   (`~/l29/<stamp>`).
3. **Pod C's rank 1 does not start or stalls.** The relay says `pod C did not start rank 1` (then nothing ran), or act
   2 waits in training with no `pod C rank1_train` line. Rank 0 gives up after 180 s without its peer and act 2 fails;
   the next run uses `--duo-pods 1`. Pod C's waiter gives up after 600 s and deletes its copies and the token.
4. **The box refuses a row.** The relay's step line reads `box 400` (or 401, 404, 413) instead of `box 200`. The run
   keeps going on pod D and the relay keeps its copies. Afterwards they post once with `python -m lab lab-push --box
   http://<box>:8000 --run-dir ../L29-data/<stamp>/<act>` (and `--manifest`, `--eval` from the same folder), and the
   comparison with `curl -X POST --data-binary @../L29-data/<stamp>/comparison.json -H "Authorization: Bearer
   $ATLAS_TOKEN" "http://<box>:8000/api/learning/lab/import?kind=compare"`. A box that answers nothing (`box did not
   answer`) is retried every round for steps and stage lines.

## Files

- `tooling/demo/live_loop.sh`: the one command (calls `live_loop.py`).
- `tooling/demo/live_loop.py`: the relay, the stopwatch, the samplers' and the box's power traces, the posts to the
  box (stage lines, power windows, comparison), `--check`, `--on-approval`, `--card` (off by default), `--fallback`
  (tests and repairs only).
- `contexts/learning/application/loop_energy.py`: the energy arithmetic, shared by the relay and the box's compare
  route; `contexts/learning/application/lab_compare.py` and `lab_loops.py`: the box's side.
- `tooling/demo/live_loop_pod.sh`: the pods' side (`check` on either pod, `start` on pod D, `rank1` on pod C), run
  through `python -m lab.controller run-script`; it calls `lab/pod_b_group_finetune.sh run` once per act with
  `GATE=fake MIN_HELD_OUT=1 LABEL_WORKERS=16`, `--grad-accum 1`, `--max-steps 8` and a trainer wrapper that marks the
  stages and, for act 2, trains rank 0 of the two-pod LoRA.
- `tooling/demo/live_loop_card.html`: the old local card (with `--card` only).
- `fixtures/learning/live-loop-2026-10-11/`: the rehearsal, test data only (stages, three power traces, both runs'
  steps, manifests, scores, comparison, relay log, screenshots); never served.
- `tests/test_tooling_live_loop.py`: stage arithmetic, pod D's split, and the relay's posts against a fake box.
