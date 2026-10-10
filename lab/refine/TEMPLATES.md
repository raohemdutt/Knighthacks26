# Refinement loop file formats

Every refinement pass on a track writes the same shapes, and the pass runner, the merge pass and the
rulings flow read them back. This file defines those shapes and gives one filled example of each. The
code that reads them is `lab/machine/refine_formats.py`; its tests parse the examples below, so an
example that stops parsing fails the suite.

The formats are the phone project's refinement loop (pass log, owner rulings, pass counter, pass
reports) with its vocabulary swapped for the lab's: tasks and experiments instead of beats and
scenarios, and metrics instead of word budgets.

Every example is labelled EXAMPLE and shows the format, not a real pass. Where an example quotes a
task's text or a baseline figure, that text is real as of 2026-10-10 (the Notion Task Board and
`lab/README.md`); the pass around it is invented.

## Layout

```text
lab/refine/
  TEMPLATES.md          this file
  PASS_RULES.md         when a pass runs, the caps, what a pass may never change
  RESEARCH_FORMAT.md    research note format, citation forms and the search bound
  MERGE_LOG.md          merge pass entries, in the same log format
  metrics.jsonl         the metrics baseline file (the ledger takes over later)
  research/<track>/     the notes each pass starts from
  <track>/              telemetry, learning, plumbing, kernels
    RUBRIC.md           the rubric the passes judge the track against
    REFINE_LOG.md       one entry per pass, appended at the end
    OWNER_RULINGS.md    numbered questions; the owner fills in the Answer lines
    refine_pass.txt     the pass counter
    pass_reports/       pass_01.md, pass_02.md, ... one report per pass
```

A pass appends one entry to its track's `REFINE_LOG.md`, saves its report under `pass_reports/`, and
bumps `refine_pass.txt` as its very last step. Passes never edit `OWNER_RULINGS.md`: they list the
questions they could not decide under "Rulings raised", the merge pass numbers those into
`OWNER_RULINGS.md`, and the owner fills in the Answer lines. A filled Answer is an owner rule at the
top of the authority order: the next pass applies it and cites it. Owner corrections made outside a
pass go into the log too, so a pass never undoes one, but they do not bump the counter.

## Conventions

- Plain ASCII, plain hyphens, no em-dashes and no emoji, in every file here.
- Entries are appended, never rewritten. A log reads oldest first.
- A dimension id is the label the track's `RUBRIC.md` gives one of its tests: the Learning rubric's
  are LT1 to LT9 and the kernels rubric's are K1 to K8. If a rubric only numbers its tests, the id is
  D followed by the number. The two universal rules every rubric carries are U1 (old versus new) and
  U2 (source-forced).
- A metric name is the dotted lowercase name used in `lab/refine/metrics.jsonl`
  (`teacher.decode_tok_s`, `pod.idle_watts`).
- A reading is a metric's value with its unit, node and conditions, and the line it was read from:
  `metrics.jsonl:<line>` for the metrics file, `ledger:<seq>` for the ledger. A metric that has never
  been stored reads `none` in both the value and the source column. That states that no baseline
  exists, and the pass queues a measurement for it on T0; it is not a way to skip the block.
- A citation is a URL; a paper (authors, title, venue and year, with a URL or DOI); a vendor document
  with its section name; a repo file with its section or line range; a research note by file and
  heading (`lab/refine/research/<track>/<file>.md, heading "<heading>"`); a Notion page with its URL;
  or a metrics-file or ledger line. Say when you opened it. Cite only what you opened.
  `RESEARCH_FORMAT.md` defines the same forms in full for research notes.
- A pass entry with no metrics-before block, or with a change that has no citation, is invalid; see
  "Invalid passes" in `PASS_RULES.md`.

## 1. The track log (`REFINE_LOG.md`)

The file opens with a header (a title, a short paragraph, a rule). Entries follow, each starting with
a level-two heading and ending with a line of three hyphens. Four kinds exist: a pass entry (1.1), an
owner correction (1.2), the state at the end of a run (1.3) and, in `MERGE_LOG.md`, a merge entry (1.4).
Any other level-two heading in a log is a format error.

### 1.1 Pass entry

Heading: `## Pass <N>, <YYYY-MM-DD>`, where N is the counter after the pass (the counter before it,
plus one). Under the heading come a summary paragraph and then these sections. A section starts at the
beginning of a line with its label, an optional parenthetical (free text, it may contain colons) and a
colon.

- Summary (no label): one paragraph. What the pass walked, how many changes it made, and the verdict
  in capitals, CONVERGED or NOT CONVERGED.
- `Research step (sources opened):` one bullet per source, `- <citation> :: <what it says that matters
  here>`. At least one. Notes the step writes go to `lab/refine/research/<track>/`.
- `Metrics before (read before changing anything):` a table with the columns
  `metric | value | unit | node | conditions | source`, one row per metric the pass's tasks and
  experiments are meant to move, read from the metrics file or the ledger before any change is made.
- `Walked (task or experiment :: dimensions checked :: result):` one bullet per item,
  `- <id> :: <dimension ids, comma separated> :: held` or `... :: BROKE <ids>; FIXED (n)`.
- `Findings (by dimension):` one bullet per finding, `- <dimension id> <where> :: <what is wrong, or
  why it holds>. <disposition>`. The disposition is `FIXED (n)` (the change that fixes it),
  `RULING (x)` (a ruling raised below), `HELD` (checked, fine) or `OPEN` (found, not fixed this
  pass, the reason in the text). Write `none` when the walk found nothing.
- `Changes (<n> of at most <cap>):` the numbered changes below, or `none`. The cap is set in
  `PASS_RULES.md`.
- `Rulings raised (...):` lettered candidates `(a)`, `(b)`, each followed by `Now:`, `Recommend:`,
  `Source of tension:` and a blank `Answer: ____` line, or `none`. They are candidates; the merge pass
  numbers them into `OWNER_RULINGS.md`. A pass never fills in an Answer.
- `Metrics after (...):` the same table, read after the pass ran something, or `not run: <why>`.
- `Converged: YES|NO - <reason>`. YES means a full walk of the track's tasks and experiments against
  every dimension changed nothing, so a YES entry has no changes.
- `Left for the next pass: ...` what the next pass starts with and what waits on the owner.
- `Checks run (...):` optional. Commands run and their actual output, verbatim, such as the re-fetch
  of each board page a change touched.
- Closing line, last in the entry before the rule: `CONVERGED at pass <N>` when the verdict is YES,
  and `CAP REACHED at pass <N>, not converged` for the tenth pass of a run that does not converge.
  N is the entry's own number. A tenth pass that does not converge also raises the ruling that
  `PASS_RULES.md` describes. Every other entry has no closing line.

A change is a numbered block, numbered 1, 2, 3 without gaps:

```text
<n>. <task or experiment id> <what changed> (<dimension ids>)
   BEFORE: "<the exact text or value before>"
   AFTER:  "<the exact text or value after>"
   SOURCE: <one or more citations>
   WHY: <one or two sentences>
   METRIC: <metric name> <expected movement>
```

Quote BEFORE and AFTER exactly, without the signed prefix; Ilan undoes a board change by pasting
BEFORE back. METRIC names the metric the change is meant to move and in which direction. A change that
moves no metric, such as a Tier or a blocker fix, says `none - <reason>`. Every change traces to a
finding that says `FIXED (<n>)`, and every `FIXED (n)` names a change that exists.

A filled pass entry:

```text example:pass-entry
## Pass 3, 2026-10-10

EXAMPLE, not the record of a real pass. The quoted task text and the baseline figures are real (Notion board and lab/README.md, 2026-10-10); the pass around them is invented. Walked the kernels items on the experiment queue (E2, E3, E5) and R24 against kernel tests K1 to K8 and the two universal rules U1 and U2. Three changes, all to the plan, none measured yet. NOT CONVERGED; pass 4 repeats the walk.

Research step (sources opened):
- Notion task O4, sections "Kernel tests" and "Baseline to beat" (https://app.notion.com/p/3f596ef5fec5816aae04d8fb488302c4), opened 2026-10-10 :: the eight kernel tests and the baseline (decode 7.7 tok/s, about 26 J per token); test 1 asks for stock numbers at batch 1, 4, 8 and 16.
- lab/README.md, section "Teacher (pod A)", on main at bb351e6, opened 2026-10-10 :: the measured table, about 200 W average and 300 W peak, and the paragraph on the Quark checkpoints that vLLM b1388b1 rejects.
- Jae-Won Chung et al., "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization", NeurIPS Datasets and Benchmarks 2025, https://arxiv.org/abs/2505.06371, opened 2026-10-10 :: the abstract reports energy measured under realistic serving conditions and optimization savings above 40 percent in some cases; it gives no batch-size figures.

Metrics before (read before changing anything):
| metric | value | unit | node | conditions | source |
| --- | --- | --- | --- | --- | --- |
| teacher.decode_tok_s | 7.7 | tok/s | pod-a | RedHatAI/Qwen3.8-27B-INT4, vllm@b1388b1, batch 1, ctx 16384, text prompt, thinking on, 160 output tokens | metrics.jsonl:3 |
| teacher.watts_avg | 200 | W | pod-a | same model and engine, ctx 16384, while generating | metrics.jsonl:7 |
| teacher.joules_per_token | 25.97 | J/token | pod-a | derived: teacher.watts_avg / teacher.decode_tok_s, marked low confidence in the file | metrics.jsonl:12 |
| teacher.decode_tok_s | none | tok/s | pod-a | batch 4, 8 and 16, same model and engine; never measured, measurement queued as E3 | none |

Walked (task or experiment :: dimensions checked :: result):
- E2 :: K2, K6, K8, U1, U2 :: BROKE K8; FIXED (3)
- E3 :: K1, K3, K6, K8, U1, U2 :: BROKE K1, K3, K6, K8; FIXED (1, 3)
- E5 :: K2, K6, K8, U1, U2 :: BROKE K2, K8; FIXED (2, 3)
- R24 :: K2, K3, K4, K5, K6, K7 :: held

Findings (by dimension):
- K1 E3 :: sweeps batch 1, 4, 16 and 64, while kernel test 1 requires stock numbers at 1, 4, 8 and 16, so batch 8 is missing and the curve cannot be compared with the baseline. FIXED (1).
- K3 E3 :: names no repeat count, so a win cannot be told from the spread. FIXED (1). RULING (a).
- K2 E5 :: one item asks two questions: whether the Quark checkpoint loads on vLLM b1388b1 (lab/README.md says it does not) and how fast the checkpoints that load run. FIXED (2).
- K8 E2, E3, E5 :: no queue item carries an expected movement or a confidence, so the queue cannot be ordered by expected energy gain times confidence. FIXED (3).
- K6 E3 :: reports a J/token curve but names no tok/s, so the speed half of the pair is missing. FIXED (1).
- K6 E2, R24 :: each pairs speed with joules per token. HELD.

Changes (3 of at most 5):
1. E3 on T0, rewritten (K1, K3, K6)
   BEFORE: "E3 batching sweep on vLLM (batch 1, 4, 16, 64: J/token curve)"
   AFTER:  "E3 (kernels) batching sweep on vLLM (batch 1, 4, 8, 16, 64; five runs per point, report the mean and the spread of tok/s and J/token; a point counts as a win only if it clears the batch-1 spread)"
   SOURCE: Notion task O4, "Kernel tests" 1, 3 and 6 (https://app.notion.com/p/3f596ef5fec5816aae04d8fb488302c4); lab/refine/PASS_RULES.md, section "The four tracks" (a queued experiment line names its track)
   WHY: the sweep skipped batch 8, which the stock baseline requires, named no repeat count and reported J/token without tok/s, so no win could be told from noise and speed was not paired with energy; five runs is a starting count, to be raised if the spread is wide.
   METRIC: teacher.joules_per_token, stored at batch 1 only today (25.97 J/token, metrics.jsonl:12); after E3 it is stored at batch 1, 4, 8, 16 and 64.
2. E5 on T0, split into E5a and E5b (K2)
   BEFORE: "E5 Teacher checkpoint shootout: AMD Quark AWQ-INT4 vs RedHat INT4 vs official FP8 (loads? tok/s, J/token, caption quality on 20 fixture frames)"
   AFTER:  "E5a (kernels) Does the AMD Quark AWQ-INT4 checkpoint load on a vLLM commit from late August 2026 or later, or with the mapper backport: yes or no, with the error text. E5b (kernels) Shootout of the checkpoints that load (RedHat INT4, official FP8, and Quark if E5a says yes): tok/s and J/token at batch 1 and 16, caption quality on 20 fixture frames."
   SOURCE: lab/README.md, "Teacher (pod A)", paragraph "Checkpoints" (the Quark checkpoints are rejected by quark.py:apply_vllm_mapper on vLLM b1388b1; loading them needs a newer vLLM commit or a backport of that one file); Notion task O4, "Kernel tests" 2 (one lever per experiment)
   WHY: E5 mixed a load question that can fail with no speed data and a speed comparison, so a failed load would have hidden the other checkpoints' numbers.
   METRIC: teacher.joules_per_token for the Quark checkpoint, none stored today; E5b stores it if E5a passes.
3. Experiment queue on T0, expected movement added to E2, E3 and E5b (K8)
   BEFORE: "(no queue item states an expected movement or a confidence)"
   AFTER:  "E2 expected: J/token differs between the two engines, direction unknown, confidence low. E3 expected: J/token changes with batch size; T0 cites the ML.ENERGY data for a 6 to 8x movement, confidence low. E5b expected: unknown until E5a answers."
   SOURCE: Notion task O4, "Kernel tests" 8 (order by expected energy gain times confidence, with the estimate and the result both on the queue); Notion task T0, "Experiment queue", E3 (the 6 to 8x note, which cites the ML.ENERGY data second-hand); Chung et al., arXiv 2505.06371, abstract (optimization recommendations save more than 40 percent in some cases; no batch-size figure, so the 6 to 8x is not verified here)
   WHY: without an estimate the queue cannot be ordered by expected energy gain times confidence; the estimates say plainly how weak they are, and the first measured result replaces each one.
   METRIC: teacher.joules_per_token, expected to change with batch size (E3); the other items have no numeric expectation yet.

Rulings raised (candidates; the merge pass numbers them into OWNER_RULINGS.md):
(a) Must a kernel win beat the run-to-run spread, or a fixed percentage?
    Now: the spread, as kernel test 3 says; E3 uses five runs per point.
    Recommend: the spread. A fixed percentage ignores how noisy a pod is on a given day, and the spread costs only the repeat runs already added.
    Source of tension: Notion task O4, "Kernel tests" 3 (a win must exceed the spread) against the 15-minute slot the GPU queue gives a benchmark, which five runs at five batch sizes may not fit.
    Answer: ____

Metrics after (only if the pass ran anything):
not run: this pass changed the plan only, so no experiment executed and no metric was re-measured.

Converged: NO - three changes made; pass 4 repeats the walk of E2, E3, E5a, E5b and R24.

Left for the next pass: re-walk every item against K1 to K8 and U1 and U2; check that T0 reads as AFTER for E3, E5a, E5b and the expected-movement lines; E3 has not run, so every batch 4, 8 and 16 reading is still none. Waits on the owner: ruling (a).

Checks run (commands and their actual output):
Re-fetched T0 after changes 1 to 3: E3, E5a, E5b and the expected-movement lines read as AFTER.

---
```

### 1.2 Owner correction

Heading: `## Owner correction, <YYYY-MM-DD> (no increment)`; a short title may follow the words "Owner
correction". The owner changes the plan by hand, or answers a ruling, outside a pass. The entry records
it so a pass never undoes it, and it does not bump the counter. Under the heading: a paragraph, an
optional `Rulings applied: <n>, <n>` line naming the `OWNER_RULINGS.md` numbers it applies, and
numbered changes in the block format above (the METRIC line is optional here).

```text example:owner-correction
## Owner correction, 2026-10-10 (no increment)

Rulings applied: 4

EXAMPLE, not a record. The owner answered ruling 4 (a kernel win must beat the spread) and said five runs per point is too many for one 15-minute slot. The plan follows the answer. No pass ran, so the counter stays where it was.

1. E3 on T0, repeat count (K3)
   BEFORE: "E3 (kernels) batching sweep on vLLM (batch 1, 4, 8, 16, 64; five runs per point, report the mean and the spread of tok/s and J/token; a point counts as a win only if it clears the batch-1 spread)"
   AFTER:  "E3 (kernels) batching sweep on vLLM (batch 1, 4, 8, 16, 64; three runs per point, report the mean and the spread of tok/s and J/token; a point counts as a win only if it clears the batch-1 spread)"
   SOURCE: OWNER_RULINGS.md 4 (owner, 2026-10-10)
   WHY: three runs at five batch sizes fit one slot; the win still has to clear the spread.

---
```

### 1.3 State at the end of the run

Heading: `## State at the end of the run, passes <first> to <last>, <YYYY-MM-DD>`; the bare title
`## State at the end of the run` is accepted too. A run is the series of passes that follows one
trigger on one track. It ends when a pass converges, at the tenth pass of the run, or when the owner
ends it (`PASS_RULES.md`, "Runs, the cap and convergence"). The pass that ends a run writes this
section after its own entry; when the owner ends a run, the merge pass writes it. The next pass reads
the log from this section onward and counts the pass entries after it to find its place in its run.
Paragraphs, each starting with its label:

- `Run ended: converged at pass <N>`, `Run ended: cap reached at pass <N>` or
  `Run ended: ended by the owner after pass <N>`.
- `Metrics across the run:` each metric's value in the run's first metrics-before block against its
  value in the last, with the lines. The loop is judged by whether the track's numbers moved; if
  nothing moved, say so.
- `Open findings (by dimension):` what the rubric still flags, with the candidate change and its
  sources, or `none`.
- `Pending rulings:` the `OWNER_RULINGS.md` numbers still unanswered and the default each one
  applies, or `none`.
- `Untested ideas (with citations):` source-backed ideas that have not run, each with its citation
  and expected movement.
- `Dead ends:` optional. What was tried or proposed and dropped, and why, so the next run does not
  retry it.
- `Files:` optional. The log, the rulings file, the counter value and the last report.

```text example:run-state
## State at the end of the run, passes 1 to 5, 2026-10-10

EXAMPLE, not a record of a real run. Five passes on the kernels track. Pass 5 walked E2, E3, E5a, E5b and R24 against K1 to K8 and the two universal rules and changed nothing. Passes 1 to 4 made nine changes between them, all to the plan: no experiment has run, so no metric has moved.

Run ended: converged at pass 5.

Metrics across the run (first metrics-before block against the last):
| metric | first | last | unit | node | source |
| --- | --- | --- | --- | --- | --- |
| teacher.decode_tok_s | 7.7 | 7.7 | tok/s | pod-a | metrics.jsonl:3 |
| teacher.joules_per_token | 25.97 | 25.97 | J/token | pod-a | metrics.jsonl:12 |
No movement: the passes sharpened the queue; the numbers wait for the slots on pod B.

Open findings (by dimension): none.

Pending rulings: none. Ruling 4 (spread or fixed percentage) was answered on 2026-10-10 and applied.

Untested ideas (with citations): E3 (batching sweep, expected to move J/token, cited in pass 3 change 3), E5a, E5b, and R24 (Triton config sweep of the W4A16 fallback kernel).

Dead ends: none.

Files: lab/refine/kernels/REFINE_LOG.md, lab/refine/kernels/OWNER_RULINGS.md, refine_pass.txt reads 5, lab/refine/kernels/pass_reports/pass_05.md.

---
```

### 1.4 Merge entry

The merge pass runs after a round of track passes and writes to `lab/refine/MERGE_LOG.md` in the pass
format with these differences. The heading is `## Merge <N>, <YYYY-MM-DD>`, where N counts the merge
entries in that file, this one included (there is no counter file). The walk covers the cross-track
checks, so the dimension ids are the checks the merge skill names; the first set is M1 blockers across
tracks, M2 shared contract changes, M3 the demo flow, M4 duplicate tasks, M5 Tier consistency and M6
whether every entry of the round is valid. A merge entry needs no research step, no metrics-before
table and no METRIC line, and it may leave out `Converged`. Each task it moves between tracks, and
each `RUBRIC.md` edit it carries out (with the ruling's number in SOURCE), is a numbered change.
Questions it copies into a track's `OWNER_RULINGS.md` are listed under `Rulings raised`.

## 2. The rulings file (`OWNER_RULINGS.md`)

A header paragraph, then numbered rulings. Numbers run in order of arrival and are never reused or
renumbered. An entry:

```text
## <n>. <the question, one line> (<origin>)

Now: <what the plan does at this moment: the default the pass applied so the loop could continue>.
Recommend: <the recommendation, with the reason in two or three sentences>.
Source of tension: <the two sources or rules that disagree, by file and heading or by URL>.
Answer: ____
```

`<origin>` says where the question came from: `pass <N>` for one a pass raised, `merge <N>` for one the
merge pass raised itself, `owner, <date>` for one the owner added. The Answer line starts blank, as
`Answer: ____`. The owner fills it in, on the same line or below it, with the date. A ruling is
answered once anything other than underscores follows `Answer:`. An answered ruling is an owner rule at
the top of the authority order; the next pass applies it, and the log cites it in an owner correction
(1.2). Passes never edit the file, so the answer stays exactly as the owner wrote it.

A level-two heading that does not start with a digit is a free-form note for the owner and is skipped by
the parser. A heading that starts with a digit must be a complete ruling, so a malformed ruling raises
an error instead of vanishing.

```text example:ruling
## 4. Must a kernel win beat the run-to-run spread, or a fixed percentage? (pass 3)

Now: the spread, as kernel test 3 says; E3 uses five runs per point.
Recommend: the spread. A fixed percentage ignores how noisy a pod is on a given day, and the spread costs only the repeat runs already added.
Source of tension: Notion task O4, "Kernel tests" 3 (a win must exceed the spread) against the 15-minute slot the GPU queue gives a benchmark, which five runs at five batch sizes may not fit.
Answer: ____
```

The same ruling once the owner has answered it (only the Answer line changes):

```text example:ruling-answered
Answer: the spread, but three runs per point so the sweep fits one slot (owner, 2026-10-10)
```

## 3. The counter file (`refine_pass.txt`)

One non-negative integer and a newline: the number of valid passes ever run on the track. It is never
reset, and it reads `0` until the first pass completes. A pass is number N when the counter reads N-1
as it starts, and it writes N as its very last step, so a finished pass's number equals the counter. A
pass that fails before that step leaves the counter alone. Merge entries and owner corrections never
touch it.

```text example:counter
3
```

## 4. The pass report (`pass_reports/pass_<NN>.md`)

The pass's final message, saved as written, one file per pass, numbered with two digits (`pass_03.md`).
It puts the log entry, the board edits and the owner items in one file. Its second line states when the
pass ended, because the two-hour trigger in `PASS_RULES.md` reads that time: an ISO 8601 date and time
with a UTC offset. Layout:

```text
Verbatim report returned by <track> refinement pass <N> (<tier> sitting), <YYYY-MM-DD>. The log entry it appended is in lab/refine/<track>/REFINE_LOG.md.
Ended: <YYYY-MM-DDTHH:MM:SS+HH:MM>

---

<one paragraph: what ran, the verdict, the files touched>

**(a) Log entry appended, verbatim** (`REFINE_LOG.md`, "## Pass <N>, <date>", L<first>-L<last>):

<the entry, copied exactly>

**(b) Board edits and files.** One line per board edit, as posted on the Plan changes post but without
its signed prefix, which belongs on Notion only:
`<track> pass <N>, <task ID>: <what changed>. Was: <before>. Why: <reason>. Source: <citation>.`
Then the counter value and the metrics-file or ledger lines the pass added.

**(c) Converged or not.** The walk record, one clause per item (held, or BROKE and fixed), and the metrics before and after.

**(d) New owner items.** The rulings raised, each with Now, Recommend, Source of tension and a blank Answer, or none.

Out of scope, noted: <anything seen outside the track and left alone>
```

```text example:pass-report
Verbatim report returned by kernels refinement pass 3 (Opus sitting), 2026-10-10. The log entry it appended is in lab/refine/kernels/REFINE_LOG.md.
Ended: 2026-10-10T07:12:00-04:00

---

EXAMPLE, not a real report. Pass 3 on the kernels track walked E2, E3, E5 and R24 against K1 to K8 and U1 and U2 and made three changes to the experiment queue on T0. NOT CONVERGED. Files touched: lab/refine/kernels/REFINE_LOG.md, lab/refine/kernels/refine_pass.txt, T0 on the board.

**(a) Log entry appended, verbatim** (`REFINE_LOG.md`, "## Pass 3, 2026-10-10", L1-L58):

(the full text of the pass entry in section 1.1, copied exactly)

**(b) Board edits and files.**
- kernels pass 3, T0 E3: batching sweep rewritten to cover batch 1, 4, 8, 16 and 64 with five runs per point and to report tok/s with J/token. Was: "E3 batching sweep on vLLM (batch 1, 4, 16, 64: J/token curve)". Why: batch 8 is required by the stock baseline, and the line had no repeat count and no tok/s. Source: Notion task O4, "Kernel tests" 1, 3 and 6.
- kernels pass 3, T0 E5: split into E5a (does the Quark checkpoint load) and E5b (shootout of the checkpoints that load). Was: "E5 Teacher checkpoint shootout: AMD Quark AWQ-INT4 vs RedHat INT4 vs official FP8 (loads? tok/s, J/token, caption quality on 20 fixture frames)". Why: a failed load would hide the other checkpoints' numbers. Source: lab/README.md, "Teacher (pod A)", paragraph "Checkpoints".
- kernels pass 3, T0 queue: expected movement and confidence added to E2, E3 and E5b. Was: no estimates. Why: the queue could not be ordered by expected energy gain times confidence. Source: Notion task O4, "Kernel tests" 8.
- refine_pass.txt reads 3. No metrics-file lines added, because no experiment ran.

**(c) NOT CONVERGED.** E2 BROKE K8 (fixed); E3 BROKE K1, K3, K6, K8 (fixed); E5 BROKE K2, K8 (fixed); R24 held. Metrics before read from metrics.jsonl lines 3, 7 and 12; after: not run.

**(d) New owner items.** Ruling (a): must a kernel win beat the run-to-run spread, or a fixed percentage? Now: the spread. Recommend: the spread. Source of tension: kernel test 3 against the 15-minute GPU slot. Answer: ____

Out of scope, noted: none.
```

## 5. Reading these files from code

`lab/machine/refine_formats.py` reads everything above:

```python
from pathlib import Path

from lab.machine import refine_formats as rf

track = Path("lab/refine/kernels")
log = rf.parse_log((track / "REFINE_LOG.md").read_text(encoding="utf-8"))
for entry in log.passes:
    problems = rf.check_pass_entry(entry)  # [] when the entry is valid
rulings = rf.parse_rulings((track / "OWNER_RULINGS.md").read_text(encoding="utf-8"))
unanswered = [r for r in rulings if not r.answered]
completed = rf.parse_counter((track / "refine_pass.txt").read_text(encoding="utf-8"))
position = len(log.current_run) + 1  # the next pass is this number of its run
ended = rf.parse_report_end((track / "pass_reports" / "pass_03.md").read_text(encoding="utf-8"))  # a datetime with an offset
```

Structure errors (an unknown log heading, a duplicate number, a ruling with no Answer line, a counter
that is not an integer, a report with no `Ended:` line) raise `FormatError`. Content problems inside a
pass entry are returned by `check_pass_entry`, one sentence each, so a pass can be told exactly what to
fix. After editing a format in this file, run `python -m pytest -q tests/test_lab_refine_formats.py`:
the tests parse the examples above.
