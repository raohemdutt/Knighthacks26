# L33: tool-calling v3, the rules, fixed before any model trains on it

Committed with the dataset, before any training run on it and before any model output on it. Nothing below changes
after a result is seen. If a rule turns out to be wrong, the result page says so and reports the number under the
rule as written. This page follows `lab/L16V2_RULES.md`. Where it says nothing, that page's rules hold.

## Why v3 exists

The v2 retrain (`lab/L16V2_RESULT.md`) took held-out tool-call agreement from 0.417 to 0.772 on 228 examples. Every
v2 scene was made up: the fixture night plus one invented sighting per object, on clocks spread over an imagined
evening. v3 asks the same kinds of questions about what the box really saw over its first two days, at the times
it saw them.

## The data

- Source. The box's event log and memory, copied without changing anything on Sun 2026-10-11 at about 5:10 AM ET
  (`fixtures/learning/tool-calling-v3/README.md` says what was taken and what was not).
- The replay. The events are replayed in the order the box logged them through `MemoryProjection` (the memory
  context's own code) into a new store. Each example's tool calls and results are computed at a checkpoint: the
  memory as it stood just before the first event stamped after that example's clock. With `--snapshot`, the build
  first checks that a replay of everything lands exactly on the box's `memory.sqlite`, and fails if it does not.
- Injected events are left out. An event is injected when its time has no fraction of a second and its frame_ref is
  missing or is not a real frame (`frames:<camera>:<13-digit ms>`). This rule drops the two scripted runs of Sat
  6:36 PM and 7:13 PM ET (190 events) and nothing else.
- Groups. A group is either one object the box saw (`last_seen-<object>`) or one real episode (`scene-<episode id>`),
  whatever the wording and the clock. An object is asked about only when it is in `ASKABLE`: things a person keeps
  on a desk. Detector misses such as surfboard, airplane or toilet are never asked about.
- Questions. These are v2's `LAST_SEEN_TEMPLATES` and `WINDOW_STEMS` with v2's minute and hour spellings, picked by
  v2's `_pick()`. A scene's window questions take each length of their part once before any length twice.
- Clocks. Each now is the time of a real logged event. Half are printed as the box prints its clock (UTC with
  microseconds). The other half are that time at -04:00, cut to the whole minute (the fixture's style). Which half
  an example falls in is set by its seed. A window example's now falls after its scene's start (by at least 5
  seconds) and close enough that the window contains that start.
- Object clocks. An object question's now falls after the object was first seen, except one row in five, which
  may fall anywhere in its part's stretch.
- Labels. The rule labels are `reference_trace()` of `lab/fixture_dataset.py` (the agent's pre-fetch): one call per
  question, last_seen or episodes_between, checked by v2's `_check_call()`. The answer is the evidence written out.
- Long tool results. A row shows each tool result shortened by `clip()`: lines cut at 300 characters, whole lines
  kept up to 2,400 characters, then "(N more lines)". The answer is written from the lines shown. The call is never
  touched. The reason is that a window over a busy stretch of the real log returns every caption of every hour (a
  median of 22,000 characters for a train window), far past the 4,096 tokens the box and pod D serve with. An
  example whose scene the box did not yet have at its clock is passed over for the next option of its group.
- Teacher labels. "When did I last see my X?" and "What happened to my X?" have no rule label: the agent answers
  them from evidence no single call returns. Those questions were asked of the teacher on pod C, about the 13
  train objects and the desk (56 questions; requests only, 16 at a time, nothing restarted). Before anything was
  sent, each question was confirmed to raise Unsupported in `reference_trace()` at its clock.
  - Each call the teacher made was run by the agent's tool code against the memory at that clock, and the result
    went back clipped as a row shows it.
  - Of the 56, 55 were kept; 1 used up its tool rounds.
  - The kept lines are train rows only, in groups named `teacher-<thing>`, and are listed in `source.json`. Validation
    and held-out examples are all rule-labeled, so agreement on them is measured against the rules.

## The split (by whole group and by time)

- Held out: the last 16 episodes as scenes and the objects keys, backpack, cup and suitcase, with clocks from the
  first held-out episode's start (Sun 3:01 AM ET) to the end of the log (5:06 AM). Lengths: v2's held-out lengths
  (25, 45, 75, 105, 420, 540, 840 and 2160 minutes) and 60 minutes.
- Validation: the 12 episodes before those and the objects can, toothbrush and umbrella, with clocks from Sun 1:09 AM
  to 3:01 AM ET. Lengths: v2's validation lengths (65, 85, 130, 660 and 1200 minutes). Validation rows are evaluated
  and never trained on.
- Train: every other group, with clocks before Sun 1:09 AM ET. Lengths: v2's train lengths, today and yesterday.
- So no train or validation tool result holds anything the box saw in a later part's stretch. Train and validation
  hold nothing that tool-calling-fixture or tool-calling-v2 holds out (keys, wallet, 60 minutes, overnight words,
  v2's held-out objects and lengths), and the build fails if one gets in. Overnight words appear nowhere.
- Counts: train 94 groups (721 rows), validation 15 groups (126 rows), held out 20 groups (240 examples).

## Scoring

- The rule is `lab/tool_call_agreement.py`, unchanged. Each v3 example's system prompt carries its own time now
  (`lab/fixture_dataset_v2.py` `system_for()` takes the v3 action id as well).
- Reported for v3's held-out side: before and after as counts with rates, flips both ways, McNemar, and the
  sign-flip test over the 20 groups (the one a claim of "better" rests on).
- v2's 228 held-out examples and the first fixture's 58 are scored as well, as the tests of questions this set never
  trained on.

## The recipe (Ilan1's v2 recipe, unchanged)

- Learning rate 1e-4, rank 16, alpha 32, dropout 0.05, warmup 0.10, batch 4 by accumulation 4, seed 42.
- At most 62 optimizer steps (992 example passes, 1.37 epochs of the 721 rows), validation loss every 8 steps and
  after the last, and the best checkpoint kept.
- No search and no other recipe is run on v3 before its held-out side is read.

## What is never done

- The held-out side chooses nothing: no recipe, no checkpoint, no early stop, no rerun.
- No retraining toward a held-out number after it is seen. If the model trained on v3 is not better, the result page
  says that.
- Nothing that identifies a person leaves the box: no frames, no faces, no names.
