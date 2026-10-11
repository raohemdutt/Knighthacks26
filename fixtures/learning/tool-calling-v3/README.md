# tool-calling-v3: the training set built from the box's own episodes

Built Sun 2026-10-11 by `lab/fixture_dataset_v3.py` (task L33). The rules are in `lab/L33_RULES.md`. The format is
tool-calling-v2's (`lab/fixture_dataset_v2.py`), so the trainer, the scorer and the pod D scripts read it the same
way; only the scenes, the clocks and the split changed.

## What came from the box

The Atlas box (the Jetson, `~/Knighthacks26/data`) was read without changing anything on Sun 2026-10-11 at about
5:10 AM ET. An anonymized copy was taken there and copied to the workstation. The copy held the event types the
memory projection replays (observations, zone events, scene captions, episode closes, hourly notes) and a snapshot of
`memory.sqlite` with connector items and vectors removed.

- The log covers Sat 2026-10-10 7:42 AM ET to Sun 2026-10-11 5:06 AM ET, from one camera, `cam-desk`. It holds
  85,669 replayed events and 95 episodes.
- Replaying the whole log through the memory context's own `MemoryProjection` lands exactly on the box's
  `memory.sqlite`: all 97 episodes with their ends and labels, 2,259 sightings, 45 tracks and 56 notes
  (`source.json`, `replay_matches_box_memory`).
- Two scripted test runs are left out: 190 events from Sat 6:36 PM and 7:13 PM ET. They have whole-second times, a
  `cam-door` the box does not have, and fake frame refs. Leaving them out drops the 2 door episodes, which is why 95
  episodes remain of the 97.
- Every scene is a real episode. Every object is a label the box's object lists or detector wrote. Every clock is
  the time of a logged event. Every tool result is what the agent's own tool code returned against the memory as it
  stood at that clock.
- Not taken: frames and keyframes (13,914 JPEGs, 1.7 GB, left on the box), face data, the Circle store, the vault,
  calendar items, and the questions people asked the box and its answers.
- Captions describe people only by what they wear. A scan for the team's names found none, and no image is in the
  set.

## The set

| part | groups | examples | rows | where its clocks fall (ET) |
| --- | --- | --- | --- | --- |
| train | 94 (13 objects, 67 scenes, 14 teacher) | 721 | 721 (`rows-train.jsonl`) | Sat 7:42 AM to Sun 1:09 AM |
| validation | 15 (3 objects, 12 scenes) | 126 | 126 (`rows-validation.jsonl`) | Sun 1:09 AM to 3:01 AM |
| held out | 20 (4 objects, 16 scenes) | 240 | none | Sun 3:01 AM to 5:06 AM |

- Labels: 1,032 rule-labeled and 55 teacher-labeled. The rule labels come from `reference_trace()` of
  `lab/fixture_dataset.py`, which is the agent's own pre-fetch. The teacher labels are train only, in the groups
  named `teacher-<thing>`, and listed in `source.json` (`teacher_ids`). Their raw answers are in `teacher.jsonl`.
- Objects:
  - train: book, bottle, bowl, cell phone, charger, keyboard, laptop, mouse, phone, potted plant, remote, tie, vase.
  - validation: can, toothbrush, umbrella.
  - held out: keys, backpack, cup, suitcase.
- Window lengths:
  - train: v2's train lengths, plus today and yesterday.
  - validation: v2's validation lengths.
  - held out: v2's held-out lengths and 60 minutes.
  - Train and validation hold nothing that tool-calling-fixture or tool-calling-v2 holds out, so those held-out sets
    stay clean tests of a model trained on this one.
- Tool results are long on the real box: a train window returns a median of about 22,000 characters. A row shows
  them shortened by rule (`clip()`: lines cut at 300 characters, kept up to 2,400, then "(N more lines)"), and the
  answer is written from what is shown. The call is never changed. The longest row is 1,899 tokens on the Qwen3.5-4B
  tokenizer (checked with the template's own rendering).
- `no_evidence` in `expected.jsonl` means "cites no episode or frame", as in v2. A detector-only sighting with no
  frame counts as no evidence even when its line names a time.

Files: manifest.json, examples.jsonl, expected.jsonl, split.json, validation.json, rows-train.jsonl,
rows-validation.jsonl, build.json (counts and the sha256 of every file), source.json (what came from the box),
teacher.jsonl, images/ (empty: no example has a picture).

Rebuild (the same bytes, apart from `source.json` when no snapshot is given):

    python lab/fixture_dataset_v3.py --events <box events.jsonl.gz> --snapshot <memory.sqlite> \
        --teacher fixtures/learning/tool-calling-v3/teacher.jsonl --out <folder>

The events copy is `work/L33-data/events.jsonl.gz` on Ilan's workstation (sha256 in `source.json`). It is not in the
repo because it is 10 MB.
