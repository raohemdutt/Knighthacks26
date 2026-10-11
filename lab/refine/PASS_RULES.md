# Refinement pass rules

This file is the policy for every refinement pass on the lab's plan, which means the Task Board rows
and queued experiments of the four tracks below. It says when a pass starts, who runs it, what it
reads, what it may and may not change, and when a run of passes stops. The pass skill
(`refine-track`), the merge skill (`refine-merge`) and every agent that takes a pass as its task
read it before doing anything else.

The rules were copied from the phone project's refinement goal (its "How passes are run", "What a
pass may NOT change" and convergence paragraphs) and edited for the lab: passes are fresh agents
instead of forks, four tracks run side by side, and a merge pass reconciles them after each round.

Where this file and a track's `RUBRIC.md` disagree about process, this file wins. Where they
disagree about what the track is judged on, the rubric wins. A ruling beats both; a ruling is a
question that Ilan has answered on the `Answer:` line in a track's `OWNER_RULINGS.md`. No pass edits
this file. Changing it takes a ruling and an ordinary board task.

## The four tracks

| Track | Folder | Covers (starting set) |
| --- | --- | --- |
| telemetry | `lab/refine/telemetry/` | energy and power measurement: the T tasks, and the numbers that reach the stats page and the slide |
| learning | `lab/refine/learning/` | the chain from labeled dataset to published adapter: L2 to L9, L11, L12 |
| plumbing | `lab/refine/plumbing/` | pods, the hub proxy link, the controller, the package server and credits: R6, R9, R10 and the credit rules |
| kernels | `lab/refine/kernels/` | serving speed and kernel work: R11, R12 and the ranked experiment queue on T0 |

Each track folder holds `RUBRIC.md`, `REFINE_LOG.md`, `OWNER_RULINGS.md`, `refine_pass.txt` (the
counter) and `pass_reports/`. A task belongs to exactly one track. A track's rubric may list the
tasks it covers, and when it does that list replaces the starting set above. The merge pass records
any move of a task from one track to another in `lab/refine/MERGE_LOG.md`, and the latest record
wins. T0 is the standing ticket whose Experiment queue lists the experiments waiting to run. A
queued experiment belongs to the track of the metric it is meant to move, which its line names in
brackets after its number, for example `E6 (kernels)`; a line without one belongs to the track of
the first metric it names.

| File | Written by | Read by |
| --- | --- | --- |
| `lab/refine/PASS_RULES.md` | nobody (see above) | every pass |
| `lab/refine/TEMPLATES.md` | an ordinary task | every pass |
| `<track>/RUBRIC.md` | an ordinary task creates it; afterwards only the merge pass, to carry out a ruling | the track's passes |
| `<track>/REFINE_LOG.md`, `refine_pass.txt`, `pass_reports/` | the track's passes | the track's passes, the merge pass |
| `<track>/OWNER_RULINGS.md` | the merge pass adds questions; only Ilan writes an Answer | every pass |
| `lab/refine/research/<track>/` and its `SOURCES.md` | the track's passes and research tasks | the track's passes |
| `lab/refine/metrics.jsonl` (later the ledger) | the lab's metrics writer | every pass |
| `lab/refine/MERGE_LOG.md` | the merge pass | every pass |

## How passes are run

Each pass is a fresh agent at the Opus tier (the board's Tier select), never a fork of another
session. In the phone project a fork carried the whole conversation; here the files carry it. The
log, the rulings, the counter, the metrics, the research notes and the board hold everything a pass
needs, and a pass never reads another session's chat or notes.

A pass is started through the open-ticket skill with the pass as its task. Every pass is a Task
Board task (title `<ID>: Refinement pass <N>, <track> track`, Context Orchestration, Assignee Ilan,
Tier Opus), so it has a lock, a clone, a pull request and a Build Log entry like any other work. The
agent runs the pass itself and does not delegate it to another agent. It opens one pull request for
the files it changed and merges it under the open-ticket skill's rule for Ilan's folders. The pass
is finished when that pull request is merged.

Passes on one track run strictly in sequence, because each pass edits the files and board rows the
next one reads. The next pass starts from a fresh `origin/main` after the previous pass's pull
request has merged, so it reads the counter and log that pass left. The four tracks run in parallel
with each other.

Work happens in rounds. A round starts when at least one track has a pending trigger or a run in
progress and nothing is running. Every such track starts one pass; a track whose trigger arrives
later waits for the next round. When every pass of the round has finished, one merge pass runs, and
no track starts another pass until it has finished. If a pass of the round gets stuck, the
referee (the Fable seat that handles queue overruns and conflicts between sittings) releases its
lock under the open-ticket skill and the round goes on without it.

## What a pass reads

In this order: this file; the track's `RUBRIC.md`; `lab/refine/TEMPLATES.md` for the formats;
`refine_pass.txt` (this pass is number N, the value plus one); the track's `REFINE_LOG.md` from the
closing section of the latest run to the end (the whole log if no run has closed), and earlier
entries only where a change touches something their history explains; the track's `OWNER_RULINGS.md`;
`lab/refine/metrics.jsonl` (the ledger once it replaces it); the research notes under
`lab/refine/research/<track>/` with their `SOURCES.md`; the Task Board rows of the track; the
experiment queue on T0; and the latest entry of `lab/refine/MERGE_LOG.md`.

Sources rank in this order, and when two disagree the higher one wins and the disagreement is
recorded, never averaged: a filled ruling from Ilan, the Design Doc spec for the context, the
track's research notes, then the metrics file or ledger. A pass opens a source before it changes
anything because of it, and its entry lists every source it opened.

## What a pass does

1. Reads the counter and the inputs above.
2. Writes the metrics-before block: for every metric the track's tasks and queued experiments name,
   the current value from the metrics file or ledger, with its unit, node, conditions and the line it
   came from. A metric with no value yet is listed as having no baseline, and a measurement for it
   is queued on T0.
3. Does the bounded research step on the track's open problems and writes it under
   `lab/refine/research/<track>/` with citations, within the search bound in `RESEARCH_FORMAT.md`.
4. Walks every board task and every queued experiment of the track against every dimension of the
   rubric, and lists every finding by dimension whether or not it fixes it.
5. Makes the changes its findings justify, highest impact first, and at most 5 changes in one pass
   (the change cap; a change is one numbered block of the entry's Changes section, and telemetry
   ruling 6 in `lab/refine/telemetry/OWNER_RULINGS.md` set the number). Each change records BEFORE,
   AFTER, SOURCE, WHY and the metric it is meant to move, and its Plan changes line is posted at the
   moment the change is made, so a pass that dies midway still leaves what it did and what it
   replaced. A finding the change cap leaves unfixed stays OPEN, with the cap as its reason.
6. Raises a ruling for anything it cannot decide (see "Telling Ilan what changed").
7. If it ran anything, writes the metrics-after block, measured the same way as the before block.
   If it ran nothing, the entry says so.
8. Appends its entry to `REFINE_LOG.md` in the `TEMPLATES.md` format, writes its pass report (which
   states the time the pass ended), and as the very last step writes N to `refine_pass.txt`.

A pass that fails before the last step, including because Notion cannot be reached, does not
increment the counter, reverts its board changes and releases its task. A dry run (the pass skill's
rehearsal mode) touches no Notion page, does not move the counter and does not count toward the cap.

## The two universal rules

Both rules are binding on every pass, and every rubric repeats them.

### Old versus new

Every task and experiment names the metrics it is meant to move, reads their current values from the
metrics file or ledger before starting, and ends by measuring them the same way. The pass entry shows
the values before and after and the conditions they were measured under. A change that moves nothing
is not an improvement. A regression is reverted, or it goes to a ruling with the trade-off stated.
The loop itself is judged by whether the track's numbers moved across passes.

### Source-forced

Every change to a method or a plan item cites at least one source next to its before and after.
Each pass starts with a bounded research step on the track's open problems, written into
`lab/refine/research/<track>/` with citations. A source-backed idea that has not been tested becomes
a queued experiment with its citation and expected movement. A change that depends on a fact no
source in hand establishes is not made; the fact becomes a queued measurement or a ruling.

A pass entry with no metrics-before block, or with any change that has no citation, is invalid and
is reverted (see "Invalid passes").

## When a pass is triggered

A track has a pending trigger when one of three things happens, and a pending trigger starts a run
on the track at the start of the next round. Ilan can also start a run by hand at any time, under
the same rules.

1. A task in the track reaches Done: a row of the track has changed to Status Done since the
   previous pass on the track started.
2. A complaint on the GPU queue: a row in the GPU Queue (R9700 pods) database has a non-empty
   Complaint that no pass has walked yet. It goes to the track of the row's Task ID, or to plumbing
   if the row has none or the task is outside the four tracks. The pass can then split or reorder
   the work that is waiting.
3. Two hours without a pass: the track's last pass ended more than two hours ago, by the end time in
   its latest pass report.

Triggers do not stack. Every trigger that fired before a pass started is consumed by that pass, and
a trigger that fires while a pass is running waits for the next one. A capped track ignores all
three until its cap ruling is answered (see "The cap").

## Runs, the cap and convergence

A run is the series of passes that follows a trigger on one track, one pass per round, because a
pass that changed anything must be followed by a pass that walks the result. A run ends at the first
of three things: a pass that converges, the tenth pass of the run, or Ilan ending it. The counter
counts the valid passes ever run on the track and is never reset.

However a run ends, its last entry is followed by a closing section titled "State at the end of the
run": the open findings by dimension, the pending rulings, the untested ideas with their citations,
and how each metric moved from the run's first metrics-before block to its last. The pass that ends
a run writes the section; if Ilan ends a run, the merge pass writes it. The next pass reads the log
from that section onward, and finds its position in the run by counting the entries after it.

### Convergence

A pass converges when it has walked every board task and every queued experiment of the track
against every dimension of the rubric and changed nothing: no board edit, no new queued experiment
and no new ruling. It may still add research notes. A finding that waits on a blank ruling or on
someone else's task is listed, not fixed, and does not stop a pass from converging. The entry ends
with the line `CONVERGED at pass N`, the counter still increments, and the run is over. The track
starts a new run, at pass 1 of 10, on its next trigger.

### The cap

A run has at most 10 passes. The tenth pass is a full pass. If it does not converge, its entry ends
with the line `CAP REACHED at pass N, not converged`, and it raises one ruling that asks Ilan
whether to extend the run, open a new one, or leave the track as it is, with a recommendation based
on whether the track's numbers moved during the run. A capped track starts no pass, whatever the
trigger, until that ruling has an Answer. An Answer that extends the run raises its cap by the
number it names. An Answer that opens a new run makes the next pass pass 1 of 10. While a track is
capped, the merge pass still reconciles it with the others.

## The merge pass

After every round, one merge pass runs at the Fable tier, started through the open-ticket skill as
a board task like any pass. It reconciles what the track passes did: blockers that cross tracks,
shared contract changes, the demo flow (every demo beat Ilan owns still has a path), duplicate tasks
across tracks, Tier consistency, and whether every entry of the round is valid. It resolves what it
can on the board, writes its own entry in `lab/refine/MERGE_LOG.md` in the same format as a pass
entry, copies the rulings the round raised into the tracks' `OWNER_RULINGS.md` files, and posts the
round's summary on the Plan changes post.

The merge pass alone may touch:

- The `RUBRIC.md` of any track, and only to carry out a filled ruling that names the change. It
  cites the ruling's number in `MERGE_LOG.md`.
- The `OWNER_RULINGS.md` of any track, to append questions. It never writes an Answer.
- `lab/refine/MERGE_LOG.md`.
- Anything that crosses a track boundary: a Blocked by link between tasks of different tracks, the
  merging or closing of duplicates across tracks, moving a task from one track to another, and Tier
  values that have to agree across tracks.
- The revert of an invalid pass (see "Invalid passes").

The merge pass is bound by every rule below as well. The list above is the only thing it may do that
a track pass may not.

## What a pass may change

Inside Ilan's lane, which is the tasks whose only Assignee is Ilan, a pass changes the Task Board
properties of the tasks of its own track that are not Locked and not Done. It may split a task, merge
tasks of its track into one, reorder them by setting Wave and Blocked by so that the order follows
the dependencies (a Blocked by link into another track is the merge pass's to add), rewrite a Done
when, fix blockers, and set Tier (Fable for judgment, Opus for hard builds, Sonnet for volume work
such as tests, fixtures, benchmark slots and docs). It may queue experiments on T0 as signed
checklist lines that give the track, the lever, the arms, the metric to move, the citation and the
expected movement. These are the only edits a pass makes to Task Board properties.

New and retired tasks follow the open-ticket skill: a new task takes the next free ID in its context
and starts Not started and unlocked, and a task that is merged away or obsolete is moved under
Completed tasks (archive) with a signed line saying why, never deleted. A pass never edits text on a
task page that someone else wrote; it adds its own signed line.

Outside the board, a pass writes only the files of its own track: `REFINE_LOG.md`,
`refine_pass.txt`, `pass_reports/` and the notes under `lab/refine/research/<track>/`. It raises
rulings in its log entry and does not write `OWNER_RULINGS.md`.

## What a pass may never change

- Another person's task, a task shared with another person, or an unassigned one. A pass posts a
  proposal instead (see below).
- The Demo Plan. Proposals only, posted on Dominick ↔ Team.
- Any task's Priority. A pass does not set, clear or lower a P0, and does not move any task between
  P0, P1 and P2. A change takes a filled ruling, and the merge pass applies it. A pass may sharpen or
  split a P0 task's Done when but never weaken it.
- A Locked task. A finding about it goes into the entry, and the next pass acts on it once the lock
  is released.
- The rubric. Only the merge pass edits a `RUBRIC.md`, and only with a ruling.
- The rulings file. Only Ilan writes an Answer, and passes do not edit `OWNER_RULINGS.md` at all.
- This file and the two universal rules.
- Another track's rows, experiments or folder. A finding about another track goes into the entry for
  the merge pass.
- An earlier pass's change, unless the pass cites a source or a metric that is newer than that
  change. Otherwise it raises a ruling, so two passes do not undo each other.
- Anything outside plans. A pass edits no code, contract, config or fixture, starts no pod and no
  model server, runs no GPU job, takes no queue slot, writes no `data/events`, and deletes no Notion
  page.

## Telling Ilan what changed

Every board change is one line on the Plan changes post, the item headed "Plan changes" under FYI on
Ilan's Team Comms page (https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7). The first pass
that needs it creates the item if it is missing. The line carries the signed prefix that AGENTS.md
section 8.1 requires on every agent post, and then this shape:

`<track> pass <N>, <task ID>: <what changed>. Was: <before>. Why: <reason>. Source: <citation>.`

Ilan can skim the post and undo any line from its Was value.

A proposal about another person's task or the Demo Plan is a numbered question on Dominick ↔ Team
(https://app.notion.com/p/3f596ef5fec581828d8ee02ab7b912e3), in the format of AGENTS.md section 8.1:
one line naming the task and its owner, the evidence, what the pass would change and why, and a
blank Answer. The pass changes nothing until the owner answers, and its entry cites the proposal.

A ruling is for a question a pass cannot decide from its sources: two sources of equal authority
that disagree, a change that needs a priority or rubric edit, a regression that needs a judgment
about the trade-off, or a decision that is Ilan's. The pass writes it in its entry with Now (what
stands today), Recommend, Source of tension, and a blank `Answer:` line. A filled Answer is an owner
rule at the top of the authority order. A blank one is not, and the Now default stands until it is
filled.

## Invalid passes

An entry is invalid when it has no metrics-before block, when any change in it lacks a citation, or
when it changes something the previous sections forbid. A pass that notices this reverts its own
board changes from the Was values on its Plan changes lines and does not move the counter. Otherwise
the merge pass of the round does it, with a pull request that removes the entry and puts the counter
back, and records the revert on the Plan changes post. An invalid pass does not count toward the
cap. Two invalid passes in a row on one track stop that track until a ruling, which the merge pass
raises, says how to go on.

## Where the pieces live

- Formats of the log entry, the ruling and the pass report: `lab/refine/TEMPLATES.md`.
- What each track is judged on: `lab/refine/<track>/RUBRIC.md`.
- Research note format, citation forms and the search bound: `lab/refine/RESEARCH_FORMAT.md`.
- The metrics file and its schema: `lab/refine/metrics.jsonl`.
- The pass skill and the merge skill live in the workspace skills folder, outside this repo.
- The session prompt for each tier: `lab/machine/prompts/`.
