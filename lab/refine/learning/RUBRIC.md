# Learning track rubric

Created 2026-10-10 from the Task Board page of O2. This file is the refinement goal for the learning
track: the board tasks and experiments that take a taught task from an approved export to a deployed,
published model (teacher labeling, fine-tune, eval gate, package, publish). It governs every refinement
pass on that track. A pass that does not follow it is invalid and its changes are reverted. The
telemetry, plumbing and kernels rubrics share its shape and its two universal rules.

## The goal

One honestly learned task by Saturday night (2026-10-10) and a publishable adapter. Honestly learned
means the run passes all nine tests below with evidence a reader can open: the person approved the data,
the teacher's labels were checked, the eval gate earned the deploy on a held-out set the model never
trained on, the run can be repeated from the ledger, the package installs on the box without hand edits,
and the published card carries no user data.

Target: convergence, or the cap of 10 passes per run, whichever comes first. Convergence = a pass that
walks every task and queued experiment in the walk list against all nine tests and changes nothing: no
board edit, no new queued experiment, no new ruling. The loop is judged by whether the metric set moved
across passes, not by how many edits the passes made.

## How passes are run

`lab/refine/PASS_RULES.md` governs the process: each pass is a fresh Opus agent started through the
open-ticket skill with the pass as its board task; passes on this track run strictly in sequence while
the four tracks run in parallel; a Fable merge pass reconciles each round and is the only pass that may
edit this file, to carry out a filled ruling; a run ends at convergence, at its tenth pass, or when Ilan
ends it. The entry, ruling and report formats are in `lab/refine/TEMPLATES.md`. Where this file and the
pass rules disagree about process, the pass rules win; where they disagree about what the learning track
is judged on, this file wins.

What the learning track adds to the pass rules:
- The metrics-before block lists the seven metrics in the metric set below. One with no value yet is
  listed as having no baseline, never estimated into one, and its measurement is queued on T0.
- The research step writes to `lab/refine/research/learning/`.
- The walk uses LT1 to LT9 as the rubric's dimensions and the walk list below as the tasks.
- Changes go highest impact first, in this order: (a) LT1 privacy; (b) a deploy, publish or demo claim
  without valid evidence (LT3, LT4, LT8, LT9); (c) a missing trace or reproducibility field (LT1, LT5);
  (d) the package (LT7); (e) the loop budget (LT6).
- A pass changes plans only. A test whose evidence needs a run (a score, a dry run, a timing) is judged
  from the records that run left. When they do not exist the finding says so, and the run that would
  produce them is a task or a queued experiment, never something the pass runs itself.

## Universal rules (every track, as the pass rules state them)

1. **Old versus new.** Every task and experiment names the metrics it is meant to move, reads their
   current values from the metrics file or ledger before starting, and ends by measuring them the same
   way. The pass entry shows the values before and after and the conditions they were measured under. A
   change that moves nothing is not an improvement. A regression is reverted, or it goes to a ruling with
   the trade-off stated. The loop itself is judged by whether the track's numbers moved across passes.
2. **Source-forced.** Every change to a method or a plan item cites at least one source next to its
   before and after. Each pass starts with a bounded research step on the track's open problems, written
   into `lab/refine/research/learning/` with citations. A source-backed idea that has not been tested
   becomes a queued experiment with its citation and expected movement. A change that depends on a fact
   no source in hand establishes is not made; the fact becomes a queued measurement or a ruling.

A pass entry with no metrics-before block, or with any change that has no citation, is invalid and is
reverted.

## Primary sources (the only things that may justify a change)

In the order the pass rules give. When two disagree, the higher one wins and the disagreement is
recorded, never averaged. A pass opens a source before it changes anything because of it and cites the
file and line, section, URL or ledger line. No citation, no change. Nothing from memory.

1. A filled ruling from Ilan: an `Answer:` in `lab/refine/learning/OWNER_RULINGS.md`.
2. The Design Doc spec for the context: the Learning context spec and, for the package and hot-swap, the
   Inference Runtime spec.
3. The track's research notes in `lab/refine/research/learning/`, each finding labeled measured,
   practitioner, anecdote or opinion, and the papers and docs they cite (the list under Sources is the
   starting set).
4. The metrics file or ledger (`lab/refine/metrics.jsonl`, later `lab/state/ledger.jsonl`), and PR
   bodies that carry the command and its output (PR #37 for the trainer and teacher numbers).

Two records sit beside that order rather than in it. The repo at the commit the pass reads is what
exists: `contracts/learning.py` is the ground truth for the learning events and for the two formats that
are the only coupling between Atlas and `lab/`, the dataset export and the model package, and a claim
about current behavior is checked against the code before a pass relies on it. When the repo and Notion
disagree on what should be built, Notion wins and the conflict is raised (AGENTS.md). The Demo Plan says
what the judges see, and no pass changes it.

## The rubric: nine learning tests

Every pass checks all nine against every task and queued experiment in the walk list. A finding under any
test is logged whether or not it is fixed this pass. Each test is stated as the task page states it, then
judged twice: on the board (does a task's Done when require it, with an owner and a check?) and on the
evidence (once a run exists, the files and ledger lines that prove it). File lines are at origin/main
a76af64.

### LT1. Trace and privacy

Every training example traces to an approved export (action id), a blurred image, and a teacher label;
no raw frames anywhere.
- Board: the tasks that read, label and train (L11, L2, L3) take input only from an approved export
  folder (`data/exports/<dataset_id>/`, export.py L1-7), and the run deletes the dataset from pod B when
  it ends ("Datasets are deleted from the GPUs after the run; only the adapter and eval scores are kept",
  Learning spec). A Done when that lets lab code read anything else is a finding.
- Evidence: per example, the export's `DatasetManifest.action_id` (contracts/learning.py L93-98) is an
  `export_training_data` action the phone approved (the exporter runs only after approval, export.py
  L63-64, proven by `test_nothing_is_written_until_approved`); `image` is a file in that export's
  `images/` folder of blurred keyframes (contracts/learning.py L56-59); `target` was filled by the
  teacher (L60); and a ledger row records the dataset's deletion from pod B, whose home directory
  otherwise persists across relaunches (lab/README.md L12).
- Moves: none; a run that fails LT1 has no valid score and is neither deployed nor published.
- Sources: `contracts/learning.py`; `contexts/learning/application/export.py`; the Learning spec; the
  Demo Plan's judge answer "the dataset is deleted after the run".

### LT2. Labels checked before training

Teacher labels are validated against the task's output schema and a slice is spot-checked for agreement
before training.
- Board: L2's Done when names the schema check, the size and draw of the spot-check slice and who checks
  it, and L3 does not start until both results are recorded.
- Evidence: every `target` parses and validates against `TaskSpec.output_schema` (contracts/learning.py
  L77; `{}` means free text, so a free-text task states its own check, such as length and required
  words); labels that fail are counted and dropped, never edited by hand; the slice is drawn at random
  with a recorded seed and compared with gold, either the thumbs verdicts the export carries
  (`DatasetExample.verdict`, contracts/learning.py L90; export.py L27-31, "so the lab can use them as gold
  labels") or a human check; the agreement rate and the slice size are in the ledger before the first
  training row.
- Moves: `learning.label_agreement`
- Sources: APIGen (format checking, execution and semantic verification before data is used to train);
  Northcutt et al. 2021 (at least 3.3% label errors in common test sets, enough to reorder models), which
  is why held-out labels get the same check.

### LT3. Held-out set fixed first and disjoint

The held-out set is fixed before training and disjoint from training data by frame_ref and by question.
- Board: L4 ("Held-out set per task") writes the split before L3 trains, and L11's fixture dataset draws
  no training example from the questions the scorer uses (see Known open items).
- Evidence: the held-out file's sha256 is in a ledger row older than the run's first training row; the
  `frame_ref` sets of the training and held-out files do not intersect; for data that carries questions
  (golden questions, tool-call traces) no normalized question text is on both sides; for a task with one
  fixed instruction (`TaskSpec.instruction`, contracts/learning.py L76) the question side holds by
  construction and the `frame_ref` check carries the test.
- Moves: none; a leak voids the held-out scores of that run.
- Sources: Kapoor and Narayanan 2023 ([L1.1] no test set, [L1.4] duplicates in datasets, [L3.2]
  nonindependence between train and test samples); Miller 2024 section 2.2 (related items need clustered
  standard errors).

### LT4. One set, one scorer, one decode setting, and a margin over noise

Before, after, and teacher scores come from the same held-out set, scorer (`contexts/agent/eval.py`), and
decode settings; a deploy needs a margin larger than measured run-to-run noise.
- Board: L4 scores the current model, the candidate and the teacher with the scorer's JSON mode
  (`python -m contexts.agent.eval --runtime llama --base-url <server> --json out.json`, eval.py L3-5 and
  the L4 page) and emits `learning.run.evaluated` with the `scores` keys current, candidate and teacher
  and a deploy or reject `decision` (contracts/learning.py L37-44; the gate reads only those three bare
  keys, Q4 on Ilan's Team Comms page). A failed gate keeps the current model on the box (Learning spec).
- Evidence: the three score rows carry the same held-out sha256, the scorer's commit and the same decode
  settings (temperature, top_p, top_k, max_tokens, thinking off, seed). Run-to-run noise is measured by
  scoring the same model on the same set at least twice: the runtime samples at temperature 0.2 with no
  seed (runtime/llama.py L161), so repeated runs can differ. The margin is computed on paired per-item
  differences (Miller section 4.2) and must exceed both the measured noise and the minimum detectable
  effect for the set's size (Miller section 5, equations 9 and 10); otherwise the decision is reject.
- Moves: `learning.heldout.current`, `learning.heldout.candidate`, `learning.heldout.teacher`
- Sources: Miller 2024; Card et al. 2020 (comparisons on small test sets are usually underpowered);
  Madaan et al. 2024 (seed variance decides whether a difference is meaningful); the Qwen3.5-4B card
  (sampling settings); `contexts/agent/eval.py`; `runtime/llama.py`.

### LT5. Reproducible from the ledger

Every run is reproducible from the ledger: run id, config, base model hash, dataset id, seed, package
checksums.
- Board: L3 and L5 write one ledger row per run with every field below (the ledger row carries `refs`
  with `run_id` and a `metrics` dict), and L7 cites that row.
- Evidence: the row has the `run_id` used by `learning.run.progress` and `learning.run.evaluated`
  (contracts/learning.py L26-44); the full trainer config (the LoraConfig and SFTConfig values: rank,
  alpha, target modules, learning rate, epochs, batch, max_length, as in lab/smoke_lora.py L55-70); the
  base model's Hub commit hash, not only the name `Qwen/Qwen3.5-4B`; the `dataset_id`
  (contracts/learning.py L94) and the sha256 of its `examples.jsonl`; the seed and data seed; the library
  versions (torch 2.9.1+rocm7.13.0, lab/README.md L9; PEFT, TRL and transformers as
  `lab/pod_b_trainer.sh deps` prints them, L20); and the package's `ModelManifest.sha256`
  (contracts/learning.py L107). A re-run from the row reproduces the score within the measured noise of
  LT4, not bit for bit: PyTorch does not guarantee identical results across releases, commits or
  platforms.
- Moves: none; it makes every number in the metric set checkable again.
- Sources: the PyTorch reproducibility notes; the TRL SFTConfig reference (`seed` defaults to 42, plus
  `data_seed` and `full_determinism`); `contracts/learning.py`; `lab/smoke_lora.py`.

### LT6. A loop short enough to run twice

The full loop (label, train, eval, package) fits a measured wall-clock budget small enough to run more
than once before Saturday night.
- Board: L7 ("Run it by Saturday night") states a budget per stage and in total, L3 keeps its own target
  ("a few hundred examples train in under ~30 min"), and each pod B stage fits one training slot in the
  GPU queue (60 minutes, one 15-minute extension with a progress note).
- Evidence: stage durations come from the ledger timestamps of one complete run, using the stages of
  `learning.run.progress` (labeling, training, eval, packaging; contracts/learning.py L31), and two loops
  plus slack fit in the time left when the run starts. Inputs measured so far, all text or single
  prompts: the trainer at 0.51 samples/s (smoke LoRA, 20 steps, batch 4, max_length 256, rank 16,
  synthetic text, 8.7 GB peak; PR #37; lab/smoke_lora.py L55-81); the teacher at 1.09 s to first token
  and 6.6 tok/s for a 720p caption (lab/README.md L71); at most 300 examples per export (export.py L21).
  Arithmetic from those inputs, not a measurement: 300 examples x 3 epochs (the SFTConfig default) /
  0.51 samples/s is about 29 minutes of training before any image tokens are added.
- Moves: `learning.loop_minutes`, `trainer.samples_per_s`
- Sources: PR #37; `lab/README.md`; `lab/smoke_lora.py`; the TRL SFTConfig reference; the GPU queue rules
  (FYI on the Dominick and Team page).

### LT7. The package installs through the manifest, proven by a dry run

The package installs on the box through the `ModelManifest` format without hand edits, proven by a dry
run.
- Board: L5 (merge the LoRA, convert to GGUF, quantize to Q4_K_M, keep the vision projector, write the
  manifest) and L6 (pull, verify checksums, hot-swap, keep the previous version) name the dry run as
  their acceptance check, and the Design Doc's open item "Confirm Qwen3.5-4B round-trips to GGUF +
  vision projector after LoRA" is closed by a recorded run before L7 depends on it.
- Evidence: the package folder holds exactly one model `.gguf` and at most one `mmproj*.gguf`
  (jetson/make_manifest.py L25-28); `manifest.json` validates as `ModelManifest` (contracts/learning.py
  L101-109) with `base` Qwen/Qwen3.5-4B, `task` set, a `sha256` for every file that matches it, and the
  eval `scores`; a recorded dry run installed the folder into a temporary `data/models/<version>/` off
  the box, and `model_version()` in runtime/registry.py (L24-30) read back the new version, base, task
  and scores, with no file edited by hand between the lab's output and the install.
- Moves: `learning.package_mb`
- Sources: `contracts/learning.py`; `jetson/make_manifest.py`; `runtime/registry.py`;
  `tests/test_make_manifest.py`; `jetson/README.md` section 5; the PEFT LoRA guide (`merge_and_unload`;
  the box runs a merged GGUF because `ModelManifest` has no adapter field).

### LT8. A model card with the numbers and without user data

The model card has task, base, eval numbers, license (Apache 2.0 base), and no user data.
- Board: L8's Done when ("LoRA adapter + model card (task, base model, eval before/after, no user data)")
  also names the license and asks for the held-out n and the measured noise beside the numbers.
- Evidence: the card's metadata has `license: apache-2.0`, `base_model: Qwen/Qwen3.5-4B` and
  `base_model_relation: adapter`; its text names the task and its instruction and gives the before, after
  and teacher scores with n and noise; it carries no image, caption, target or frame_ref from the
  household, and no dataset is uploaded ("only the adapter and eval scores are kept", Learning spec). The
  base's license was read on its card on 2026-10-10 (apache-2.0) and is read again at publish time.
- Moves: none; it gates the publish.
- Sources: the Hugging Face Hub model card docs; Mitchell et al. 2019; the Qwen3.5-4B card.

### LT9. A rehearsed demo beat with a fallback

The before-and-after is a beat in the five-minute demo with a rehearsed fallback.
- Board: the Demo Plan's 2:05 to 3:15 "It learns" beat (before versus after versus teacher scores, the
  model version badge, the adapter on Hugging Face) and the 1:05 to 1:35 keys beat ("That answer came
  from a skill Atlas learned yesterday") each keep their fallback (the finished run's numbers and the
  recorded training clip; replaying the keys fixture), and a task owns the rehearsal of both.
- Evidence: the numbers on the screen and the slide come from the eval JSON through
  `lab/slide_numbers.py --eval-before ... --eval-after ...` (L3-5), never typed by hand; a rehearsal of
  the beat and of its fallback is logged with the date, the duration and what broke.
- Moves: none; it is where the held-out scores are shown.
- Sources: the Demo Plan; `lab/slide_numbers.py`.

## The metric set

Names follow the dotted, lowercase convention of `lab/refine/metrics.jsonl` (a row holds metric, value,
unit, node, conditions, source, ts, low_confidence). `trainer.samples_per_s` is the name on that file's
task page (O10); the other six are new, and the first pass that measures one adds it. A score is
reported with its n and standard error (Miller section 2.1). The three held-out scores map one to one to
the `scores` keys of `learning.run.evaluated` (contracts/learning.py L42). A value taken under other
conditions (another batch size or max_length, or data with images) starts a new baseline; it is not an
after.

| Metric | Name | How it is measured | Value on 2026-10-10 |
| --- | --- | --- | --- |
| Held-out score, before | `learning.heldout.current` | LT4's scorer on the model the box runs now | not measured; no real-model score is recorded, and the fake runtime scores the 6 golden questions 1.0 on retrieval only |
| Held-out score, after | `learning.heldout.candidate` | same set, scorer and decode settings, on the candidate | not measured |
| Held-out score, teacher | `learning.heldout.teacher` | same, on the teacher on pod A | not measured |
| Label agreement rate | `learning.label_agreement` | share of the LT2 slice where the teacher matches gold, with the slice size | not measured |
| Loop wall-clock | `learning.loop_minutes` | first labeling row to the written package manifest, from one run's ledger stage times | not measured |
| Training samples per second | `trainer.samples_per_s` | examples seen / wall time of `trainer.train()` (lab/smoke_lora.py L72-81), with batch, max_length, rank and data recorded | 0.51 (pod B, 20 steps, batch 4, max_length 256, rank 16, synthetic text; PR #37) |
| Package size | `learning.package_mb` | total size of the package folder's model gguf and mmproj, in MB | not measured |

## The walk list

Every pass walks these tasks, which are the learning track's starting set in the pass rules (L2 to L9,
L11 and L12), and every queued experiment on T0 that belongs to the track (its line says `(learning)`, or
the first metric it names is in the metric set below), each against LT1 to LT9. The tests that bear
hardest on each task are named so a pass starts there; the walk still covers all nine.

- L2 teacher labeling on pod A: LT1, LT2, LT6.
- L3 LoRA trainer on pod B: LT1, LT5, LT6.
- L4 eval gate: LT3, LT4.
- L5 package (merge, GGUF, Q4_K_M, manifest): LT5, LT7.
- L6 deploy over the hub proxy, shared with Xander: LT7.
- L7 first learned task, end to end, shared with Dom: all nine.
- L8 publish to Hugging Face: LT8.
- L9 second taught task (P1, after L7): as L7.
- L11 dataset reader and fixture dataset: LT1, LT3.
- L12 tool-call trace formatting: LT2, LT3.
- Queued learning experiments: LT4 for how they are measured, plus the tests their lever touches.

L6 and L7 are shared, so a pass walks them and logs its findings but changes them only through a
proposal to their co-owner. Read for evidence, never edited, because other people own them: L1 and the
export code (Dom), the scorer `contexts/agent/eval.py` (Dom), and P6 face blur, R7 hot-swap, R8
on-device model and X10 Teach screen (Xander).

## What a pass may NOT change

Everything on the pass rules' list of what a pass may never change holds here: another person's task, a
shared or unassigned task, a Locked task, the Demo Plan, any Priority, this rubric, the rulings file,
another track's rows, and anything outside plans (code, contracts, config, fixtures, pods, GPU jobs,
queue slots, `data/events`). On the learning track that means in particular:

- L1, L6, L7 and the tasks and folders listed above as other people's: a finding there becomes a
  proposal on Dominick and Team, not an edit.
- This file. A pass that finds a test unclear or wrong raises a ruling; the merge pass edits the file
  once Ilan has answered it.
- A plan that swaps the held-out set, the scorer or the decode settings between the before and the after
  of one comparison. A new held-out set starts a new baseline, measured again for current, candidate and
  teacher.
- A Done when that lets a run read anything but an approved export, keep a dataset on a pod after the
  run, or install on the Jetson or publish to the Hub without Ilan's explicit go.
- A Done when that changes `contracts/learning.py` other than additively, with a version bump and a
  heads-up to its users (AGENTS.md section 8.3).
- Privacy: no raw frame, export image, caption, label or other household text goes into a pass log,
  report, ruling, PR or Notion post.

## Known open items for pass 1 (found 2026-10-10 by reading the code and the board; not run on a pod)

- LT3, LT4: `contexts/agent/eval.py` scores the six golden questions by citations and keywords (eval.py
  L64-84; `fixtures/golden/questions.yaml`) and has no path that scores a held-out set of exported
  images, so the desk-objects task has no scorer yet. One golden question moves the score by 0.167. By
  Miller's equation 10 with his illustrative item variance of 1/9, 80% power and a 5% level, the smallest
  detectable paired difference is about 0.38 on 6 items, 0.17 on 30, 0.13 on 50, 0.09 on 100 and 0.05 on
  300.
- LT4: three readings of the held-out size disagree: about 30 hand-checked examples (Design Doc section
  13), "too early" under 50 per arm (the verdict rule in O28, from the phone project), and Miller's
  estimate that about 1,000 items are needed to see a 3-point difference. L4 needs a ruling on n and on
  the deploy margin before it is built.
- LT3: L11 builds its fixture training data from `fixtures/golden/questions.yaml` and `memory.yaml`, the
  questions the scorer uses (L11 page). As written, training and held-out share every question.
- LT3: the export takes every keyframe of each matching episode (export.py L36-40), so neighbouring
  keyframes are near-duplicates and a split by `frame_ref` alone can put one episode on both sides ([L3.2]
  in Kapoor and Narayanan). Candidate experiment: split by episode id and cluster the standard errors by
  episode (Miller section 2.2); expected movement: lower held-out scores than a frame split gives, closer
  to what a new day of data would score.
- LT4: the decode settings differ by path. `runtime/llama.py` sends temperature 0.2 with no seed, top_p or
  top_k (L161); the box runs llama-server with thinking off (`--reasoning-budget 0`, jetson/README.md
  L73); the teacher on vLLM needs `"chat_template_kwargs": {"enable_thinking": false}` per request
  (lab/README.md L77); the Qwen3.5-4B card recommends temperature 0.7, top_p 0.8, top_k 20 and presence
  penalty 1.5 without thinking; Miller section 3.3 advises against lowering the temperature only to
  shrink variance. One setting must be pinned and recorded for all three scorers.
- LT4: the teacher is reached through the hub proxy, which needs an `Authorization: token` header
  (lab/README.md L35-45), and `LlamaServerRuntime` sends none (runtime/llama.py L61-78), so eval.py cannot
  score the teacher as written. The token stays in `.env` and never enters a log or a saved command line.
- LT1: blur is not enforced on main. Face blur on the Jetson is P6 (not started), and the Build Log entry
  for the Vault (PR #25) says nothing enforces the blur policy yet. Until P6 lands, an approved export can
  hold unblurred faces.
- LT6: `lab/smoke_lora.py` trains text at max_length 256 (L68). The TRL SFT docs say to set
  `max_length=None` for vision-language data, or to verify that truncation never removes image tokens.
  The 0.51 samples/s is a text-only number and is measured again with image examples before LT6 is
  judged.
- LT5: the ledger (`lab/state/ledger.jsonl`) is not on main yet, so a run's row has no durable home and
  LT5 cannot pass until it lands.
- LT7: the Design Doc (section 14) still lists "Confirm Qwen3.5-4B round-trips to GGUF + vision projector
  after LoRA" as open.
- Names: the plan's `learning.model.published` is not in `contracts/learning.py`, which has
  `learning.run.evaluated` and `learning.model.deployed` (L37-51); passes use the contract's names.
  Publishing to the Hub has no event.
- LT9: the "It learns" beat shows desk-objects (Demo Plan) and L7 asks for real table data, while the
  L11 page says the first learned task can start from the text fixtures (the tool-calling task). The beat
  needs a ruling on which task it shows.
- LT8: the Design Doc (section 9) says to verify the base's license before publishing adapters; the
  Qwen3.5-4B card read apache-2.0 on 2026-10-10, which settles it until the publish-time check.

## Sources (opened 2026-10-10)

Repo files, at origin/main a76af64:
- `lab/refine/PASS_RULES.md`: the process every pass follows: tracks, rounds, the cap, convergence, what a pass may and may never change, the two universal rules.
- `contracts/learning.py`: the learning events (L9-51), the export and package formats (L54-64), and `TaskSpec`, `DatasetExample`, `DatasetManifest` and `ModelManifest` (L74-109).
- `contexts/learning/application/export.py`: the approval-gated export, the 300-example cap, keyframe selection, thumbs verdicts kept as gold labels.
- `contexts/learning/tests/test_export.py`: nothing is written until the export is approved.
- `contexts/agent/eval.py`: the golden-question scorer and its `--json` output for before and after.
- `contexts/agent/tests/test_golden_eval.py`: the golden set scores 1.0 on the fake runtime, which checks retrieval only.
- `fixtures/golden/questions.yaml`: the six golden questions.
- `runtime/llama.py`: the request body (temperature 0.2, no seed) and a client that sends no auth header.
- `runtime/registry.py`: how an installed manifest becomes the model version.
- `config/models.yaml`: the manifest path the box reads (L12).
- `jetson/make_manifest.py`: the package folder rules and checksums.
- `tests/test_make_manifest.py`: the manifest matches the contract.
- `jetson/README.md`: the on-device model under `data/models/<version>/`, thinking off.
- `lab/README.md`: pod facts, the hub proxy, the teacher's numbers.
- `lab/smoke_lora.py`: the smoke trainer and how it computes samples per second.
- `lab/pod_b_trainer.sh`: the trainer setup and its version printout.
- `lab/slide_numbers.py`: before and after numbers on the slide.

Team pages and PRs:
- PR #37 (trainer 0.51 samples/s, 8.7 GB peak, loss 2.23 to 0.16 in 20 steps; teacher numbers): https://github.com/raohemdutt/Knighthacks26/pull/37
- O2 task page (the tests, the metric set, the starting sources): https://app.notion.com/p/3f596ef5fec5810db5ebe39aef2e0ec9
- O1 task page (the shared section list and the universal rules' wording): https://app.notion.com/p/3f596ef5fec581859648f9824af01daf
- O9 (pass log, rulings and counter formats): https://app.notion.com/p/3f596ef5fec581c38706ec07a84acf72
- O10 (metrics file schema and names): https://app.notion.com/p/3f596ef5fec5812c9c58dd375d7e8d8d
- O11 (the pass runner): https://app.notion.com/p/3f596ef5fec581c894e1e91d93290e33
- O12 (the merge pass): https://app.notion.com/p/3f596ef5fec581d38a47e66b288d65b1
- O13 (pass rules: triggers, cap, convergence): https://app.notion.com/p/3f596ef5fec581b395cdef8efccafd9e
- O15 (the ledger and its row): https://app.notion.com/p/3f596ef5fec581499019c49da7a9d322
- O27 (the experiment shape): https://app.notion.com/p/3f596ef5fec581c4a97cc3c1ac25bbe6
- O28 (verdict math and the 50-per-arm rule): https://app.notion.com/p/3f596ef5fec5811f9ed4d5b3352a9bc2
- O36 (the research-note format): https://app.notion.com/p/3f596ef5fec58136b164e799928c7ec9
- O38 (the first learning pass and its task list): https://app.notion.com/p/3f596ef5fec581629ea7f9c49efea08c
- Learning context spec: https://app.notion.com/p/3f596ef5fec5814bb5dfde0052175b1e
- Inference Runtime spec: https://app.notion.com/p/3f596ef5fec5814097aecc119914b576
- Atlas Design Doc (sections 9, 13 and 14): https://app.notion.com/p/3f596ef5fec5819ea5cdcaa015e69917
- Demo Plan (the keys beat, the "It learns" beat, their fallbacks): https://app.notion.com/p/3f596ef5fec581a2bdc4f96c8da65252
- Task Board (the L2 to L12 pages): https://app.notion.com/p/3f596ef5fec581bcb319e8687c1fdd8e
- Ilan's Team Comms page (Q4: the gate reads only current, candidate and teacher): https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7
- Dominick and Team page (the GPU queue rules FYI): https://app.notion.com/p/3f596ef5fec581828d8ee02ab7b912e3
- Build Log (A5 eval, R8 on-device model, PR #25 blur policy): https://app.notion.com/p/3f596ef5fec5812fa37de5ae92086234

Papers and docs:
- Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685. A frozen base plus trainable low-rank matrices: 10,000x fewer trainable parameters and 3x less GPU memory than fully fine-tuning GPT-3 175B, with no added inference latency: https://arxiv.org/abs/2106.09685
- Dettmers et al. 2023, "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314. LoRA through a frozen 4-bit base (NF4, double quantization, paged optimizers); a small high-quality dataset gave state-of-the-art results; chatbot benchmarks judged not trustworthy: https://arxiv.org/abs/2305.14314
- PEFT LoRA developer guide (docs at v0.21.0, the version listed for the pod): `target_modules="all-linear"`, `merge_and_unload()`: https://huggingface.co/docs/peft/en/developer_guides/lora
- TRL SFT Trainer (docs at v1.15.0, linking transformers v5.19.0): conversational and prompt-completion data, completion-only loss by default for prompt-completion, tool-calling data, vision data with `max_length=None`, `peft_config`, SFTConfig defaults (3 epochs, seed 42): https://huggingface.co/docs/trl/sft_trainer
- Qwen/Qwen3.5-4B model card: apache-2.0, image-text-to-text, thinking on by default and switched off with `enable_thinking: false`, sampling settings per mode: https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen/Qwen3.8-27B model card: apache-2.0, dense, vision input; tool calling with the open weights is not stated: https://huggingface.co/Qwen/Qwen3.8-27B
- Kim and Rush 2016, "Sequence-Level Knowledge Distillation", EMNLP 2016, pp. 1317-1327, arXiv:1606.07947. The student is trained on the teacher's highest-scoring beam output (figure 1), the shape of teacher labeling followed by SFT; the student ran 10 times faster with little loss: https://arxiv.org/abs/1606.07947
- Chen et al. 2023, "ShareGPT4V: Improving Large Multi-Modal Models with Better Captions", arXiv:2311.12793. 100K GPT4-Vision captions, grown to 1.2M by a caption model trained on them, improved LLaVA-7B, LLaVA-1.5-13B and Qwen-VL-Chat-7B when swapped into their SFT data: https://arxiv.org/abs/2311.12793
- Shu et al. 2024, "LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation", arXiv:2408.15881. A large multimodal teacher distilled into a student with 2B active parameters on 0.3% of the training data, ahead of Qwen-VL-Chat-7B by 8.8% on average: https://arxiv.org/abs/2408.15881
- Patil et al. 2023, "Gorilla: Large Language Model Connected with Massive APIs", arXiv:2305.15334. A fine-tuned LLaMA model writes API calls better than GPT-4 and hallucinates fewer of them: https://arxiv.org/abs/2305.15334
- Liu et al. 2024, "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets", arXiv:2406.18518. Every example passes format checking, execution and semantic verification; a 7B model trained on the data outperformed several GPT-4 models on the Berkeley Function-Calling Benchmark: https://arxiv.org/abs/2406.18518
- Miller 2024, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations", arXiv:2411.00640. Standard errors (2.1), clustered errors (2.2), resampling and temperature (3.1 to 3.3), paired differences (4.2), power and the minimum detectable effect (5, equations 9 and 10; about 969 items for a 3-point difference): https://arxiv.org/abs/2411.00640
- Card et al. 2020, "With Little Power Comes Great Responsibility", EMNLP 2020, arXiv:2010.06595. Underpowered comparisons are common, and small test sets cannot detect small differences: https://arxiv.org/abs/2010.06595
- Madaan et al. 2024, "Quantifying Variance in Evaluation Benchmarks", arXiv:2406.10229. Seed variance decides whether a difference in performance is meaningful: https://arxiv.org/abs/2406.10229
- Kapoor and Narayanan 2023, "Leakage and the Reproducibility Crisis in ML-based Science", Patterns, arXiv:2207.07048. The leakage taxonomy, including [L1.1] no test set, [L1.4] duplicates in datasets and [L3.2] nonindependence between train and test samples: https://arxiv.org/abs/2207.07048
- Northcutt, Athalye and Mueller 2021, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", NeurIPS 2021 Datasets and Benchmarks, arXiv:2103.14749. At least 3.3% label errors on average across ten test sets; ResNet-18 beats ResNet-50 if mislabeled test examples rise by 6%: https://arxiv.org/abs/2103.14749
- Mitchell et al. 2019, "Model Cards for Model Reporting", FAT* 2019, arXiv:1810.03993: https://arxiv.org/abs/1810.03993
- Hugging Face Hub, Model Cards (metadata: license, base_model, base_model_relation, eval results): https://huggingface.co/docs/hub/model-cards
- PyTorch 2.14, Reproducibility (results are not guaranteed identical across releases, commits or platforms, or between CPU and GPU): https://docs.pytorch.org/docs/2.14/notes/randomness.html
