# Hand-over: optional events you can say yes to, and reminders when a planned block starts

Branch `ctx/day/reminders-and-optional` (from `origin/main` at `84184a1`). Not pushed, no PR. The branch holds the
pure groundwork with its tests; the read model, the API, the tick, the web card, the demo scene and the docs are
still to build. This document is meant to be enough for an agent with no other context to finish it.

---

## (a) The owner's request, and what "done" means

His words (condensed), about an optional event shown on the Calendar screen as
"Friends' dinner · until 6:15 PM · '63 South · with Riley, Sam, Alex · optional · via GroupMe":

> "for stuff like this it [should show] you to go to the dinner with the location and plan of adding it, nudging it,
> and Atlas speaks to you: 'hey, at this [time] we should do this'… and does Atlas do this when the time happens: it
> speaks the reminder, like 'hey, at this time you could do this', based on the plan?"

And later: the reminders must work "in tandem with other reminders and Atlas's functionalities".

Today Atlas narrates and speaks the morning plan, leave-now / late for FIXED events, re-plans, the check-in and the
evening review. It does NOT remind you when a planned block starts, and optional "Near you" events are ignored
everywhere (saying yes changes nothing on the box).

Done, for the person using it, is three things:

1. **The suggestion.** Between 90 and 30 minutes before an optional event that fits (or nearly fits) and names
   friends, Atlas says once, on the phone and by voice: "Riley, Sam and Alex are having dinner at '63 South at
   quarter past five. It's a seven-minute walk and it fits before your study block. Want me to add it?", with a card
   carrying "Add it" and "Not today".
2. **Yes → approval → a block with route and leave-by.** "Add it" (tap or voice) never changes the plan by itself:
   it raises a re-plan with the normal approval card; the same card offers "Review and approve". Once approved the
   dinner is an ordinary block of the day: walk, leave-by, "leave now", tracking and review all apply. "Not today"
   means it is never suggested again.
3. **The spoken reminder at block start.** When the time for the next planned block arrives (study, meal, rest, an
   accepted optional event), Atlas says so once: "It's two o'clock: your study block on Project Milestone 1,
   twenty-five minutes. You're at the desk already." / "Time for lunch at Student Union; it's a seven-minute walk."

All of it small and solid: the team is otherwise in a feature freeze.

---

## (b) What exists on this branch now

One commit of code (`Day: optional events and block reminders, the pure groundwork ...`) plus this document.
Everything below is pure logic or contract; **nothing is wired into the running app**, so the app behaves exactly as
on `main`. There is no endpoint, no config key read, no tick, no web change, and no part of the arbiter (item 1 of
section c).

### `contracts/day.py` (additive only)

- `PlanMove.origin: str | None = None`: `"optional"` for a block that is an accepted optional event.
- `DayReplanProposed.for_optional: str | None = None`: the optional event (`nearby:...`) a yes asked to add, or a no
  asked to take out.
- `Moment` gains `"suggest"` and `"start"` (so `DayNarrated.moment`, `GET /api/day/brief?moment=` and
  `POST /api/day/brief {moment}` accept them; the POST answers 404 "nothing to narrate" for them until built).
- New event `day.optional.answered` (`DayOptionalAnswered`), registered:

  | field | type | meaning |
  | --- | --- | --- |
  | `day` | date | the local day of the answer |
  | `id` | str | the optional event's `external_id` (`nearby:<slug>:<start ISO>`) |
  | `answer` | `"yes"` \| `"no"` | |
  | `at` | datetime | when it was answered |
  | `plan_id` | str \| None | the re-plan a yes raised, when it raised one |
  | `action_id` | str \| None | that re-plan's approval card |

- `BOX_ONLY_EVENTS` now also holds `day.optional.answered` (which invitations a person took or turned down is
  nobody else's business).

Sample and fake: `tests/test_contract_samples.py` has the sample payload; `fixtures/demo/fakes.yaml` has one step
appended at the END of the `day:` list (earlier steps keep their ids).

Tests touched because the set grew (list of every existing expectation changed):
- `contexts/circle/tests/test_health_privacy.py` line 96: `== BOX_ONLY_EVENTS` became `<= BOX_ONLY_EVENTS`.
- `contexts/learning/tests/test_export.py`: the log count `== 2` became `== len(BOX_ONLY_EVENTS) == 3`, plus one
  added assertion that neither `friends-dinner` nor `nearby:` is in the export.

### `contexts/day/domain/optional.py` (new, pure: no clock, memory or I/O)

- `PREFIX = "nearby:"`, `BLOCK_PREFIX = "optional:"`, `ORIGIN = "optional"`.
- `Offer` (id, title, start, end, place, place_name, tags, going, host, source; `.slug`, `.minutes`, `.people`).
- `parse(item, locate, name, tz) -> Offer | None`: reads a `nearby:` calendar item
  (summary `until <end> · <place> · <tag> · ...`; `with A, B`, `hosted by X`, `via Y`). Pass `Walks.locate` and
  `Walks.name` (`contexts/play/dayrun/routes.py`). All seven seeded events parse with a place (tested).
- `block_id(offer_id)` / `offer_id(block_id)`: `nearby:friends-dinner:...` ↔ `optional:friends-dinner:...`. The
  block id deliberately does not start with `nearby:`, so no guard skips it.
- `fit(offer, moves, now, walk, seen_place=None) -> Fit`: `fits` is `"yes"` (nothing in the way), `"nearly"` (only
  suggested blocks would move, or leaving at most 15 minutes early for the next fixed event) or `"no"` (it overlaps
  a fixed block, it is too late to walk there, or the next fixed event would need leaving more than
  `min(15, half the event)` minutes early). Also `walk_minutes`, `leave_by` (None under a 3-minute walk),
  `from_place`, `displaces`, `conflicts`, `leave_early_minutes`, `before` (the plan's next block, for "it fits
  before your study block") and `reason`. The walk is from the last place the plan has the person before the event,
  else from where the cameras saw them. `walk` is `lambda a, b: walks.walk(a, b)[0]`.
- `state_of(offer, now, answer, accepted) -> "suggested" | "accepted" | "declined" | "passed"`. A yes whose card
  was never approved is still `suggested`.
- `Gate` + `suggest_held(now, offer, fit, state, gate) -> str | None`: why nothing is suggested now, or None.
  Defaults: window 90 to 30 minutes before the start, 45 minutes between two suggestions, not during a fixed block,
  not within 30 minutes before a leave-by (until that event starts), not outside waking hours, not when nobody is
  named, not when it does not fit, once per event, never for declined / accepted / passed.
- `changes(old_moves, new_moves, now, tz) -> list[dict]`: what a re-plan does to the suggested blocks (`moved`,
  `shortened`, `lengthened`, `dropped`, each with `start`/`end`/`new_start`/`new_end` as local `HH:MM`). A fixed
  row here means the re-plan must be refused.

### `contexts/day/domain/narration.py` (additive)

- `MOMENTS` and `BOUNDS` gain `suggest` (4 sentences, 340 chars) and `start` (3, 260); `render` dispatches to the
  new `_suggest` and `_start`; `_replan` gains a branch for facts with `added` / `removed`.
- Facts shapes the templates take (times are local `"HH:MM"`, as everywhere in this module):
  - suggest: `{"moment": "suggest", "title", "who": [..], "host"?, "at", "until"?, "place"?, "walk_min"?,
    "leave_by"?, "fits": "yes"|"nearly", "before"?: {"title", "at"}, "moves"?: ["your study block"],
    "early_min"?, "next"?}`
  - start: `{"moment": "start", "block": "your study block", "kind", "at", "min"?, "place"?, "there"?: bool,
    "walk_min"?, "deadline"?, "count"?: n, "until"?}` (`count > 1` needs `until`)
  - a yes: the usual replan facts plus `"added": "<title>"` (or `"removed": "<title>"`).
- Lines the templates produce today (asserted in the tests):
  - "Riley, Sam and Alex are having dinner at '63 South at quarter past five. It's a seven-minute walk and it fits
    before your study block. Want me to add it?" / display "Friends' dinner with Riley, Sam and Alex: 5:15 PM to
    6:15 PM at '63 South. 7 min walk, leave by 5:08 PM. Fits before your study block at 6:30 PM. Add it?"
  - "Andria is hosting COP4533 review session at John C. Hitt Library at half past six, with Maya and Priya. It fits
    the rest of your day. Want me to add it?"
  - "Riley is going to Food truck night at Memory Mall at half past six. It's a twelve-minute walk, and your study
    block would have to move to make room. Want me to add it?"
  - "It's two o'clock: your study block on Project Milestone 1, twenty-five minutes. You're at the desk already."
  - "Time for lunch at Student Union; it's a seven-minute walk."
  - "It's two o'clock: time for your study block on Project Milestone 1, twenty-five minutes, at the desk."
  - "It's two o'clock: three short study blocks from here, through to quarter to four. You're at John C. Hitt
    Library already."
  - a yes: "Okay, I've worked Friends' dinner in: ... Leave by five to five for that; it's a sixteen-minute walk.
    Want me to switch to that?"
- The grounding check (`check`) needs no change: it accepts a faithful rewrite of both moments and rejects an
  invented time, name, number, event word or a dropped closing question (tested).

### `contexts/day/tests/test_optional.py` (new, 25 tests)

Parsing, fit and displacement (a fixed event is never listed as something to move), the four states, the window and
every quiet rule, once-only / declined / accepted / passed, `changes`, the contract additions, and the wording and
grounding of the two new moments.

### How to try what exists

```
/Users/hemduttrao/Knighthacks26/.venv/bin/python -m pytest -q contexts/day/tests/test_optional.py     # 25 passed
/Users/hemduttrao/Knighthacks26/.venv/bin/python -m contexts.demo fake day --data-dir <empty dir>     # the new event is the last line
```

There is nothing to curl yet. To see the raw material the read model will be built from:

```
python -m contexts.demo seed --data-dir <empty dir>
ATLAS_DATA_DIR=<dir> ATLAS_CAPTURE=0 ATLAS_CONNECTORS=0 ATLAS_MEMORY_FIXTURE= NEON_DATABASE_URL= NTFY_URL= ATLAS_PORT=<8930-8990> python -m contexts.gateway
curl -s 'http://127.0.0.1:<port>/memory/items' | grep nearby:      # the seven optional events
curl -s  http://127.0.0.1:<port>/api/day/plan                      # the plan they must be fitted against
curl -s  http://127.0.0.1:<port>/api/day/brief                     # latest narration per moment
```

(`ATLAS_PORT` is read in `contexts/gateway/__main__.py`. Never ports 8800, 8000, 18080. Stop the gateway by pid,
never `pkill`/`killall`. This walk was not run on this branch: only the seed and the tests were.)

### Test totals on this branch

- Full run (command in section e), before the rebase onto `84184a1`: **6457 passed, 19 skipped, 87 deselected**.
  The run after the rebase is in the commit message of this document.
- `python -m contexts.agent.eval`: **48/48**.
- `web/` untouched, so no `tsc` / build run.

---

## (c) What is NOT done (in order)

### 1. One arbiter for everything Atlas says unasked ("in tandem with other reminders")

**Current state, read from the code (nothing of the arbiter exists on this branch):**

| What speaks or pushes unasked | Trigger | Its own limit | Phone push (ntfy) | Web toast | Web voice (only with "speak up on its own") |
| --- | --- | --- | --- | --- | --- |
| `day.narrated` morning | `day.plan.proposed` (once per day at wake, or a remake) | never the same facts or the same display twice in a row (`Narrator.same_as_last`) | yes, "Your day" (`MOMENT_PUSH`, `contexts/day/application/loop.py`) | yes | kind `brief`, 120 s |
| `day.narrated` now | `day.status.changed` (at most one a minute, `tracker.MIN_GAP`) | as above | only when `urgent` (leave now, late): "Time to move" | only when urgent | only when urgent: kind `move`, 60 s |
| `day.narrated` replan | `day.replan.proposed` (drift: 15 min apart, at most 2 per fixed event; or asked) | as above | yes, "Re-plan" | yes | kind `replan`, 60 s |
| `day.narrated` coach | `day.coach.noted` (a block of the main activity ended) | as above | yes, "Coach" | yes | kind `coach`, 60 s |
| `day.narrated` evening | `day.reviewed` (bedtime or Habits closing the day) | as above | yes, "Your day, reviewed" | yes | kind `review`, 120 s |
| `day.narrated` approval | `workspace.approval.requested` for a plan card or a League card | once per action id | no (the card pushes itself, next row) | no | replaces the raw reason of the approval, kind `approval`, 30 s |
| `day.narrated` checkin | `day.checkin.raised` | rule: 90 min cooldown, awake, somebody at the desk, not in a fixed block, not within 30 min of a leave-by (`contexts/day/domain/checkin.py` `held`) | `day.checkin.raised` pushes "Check-in" with the nudges only (not the narration) | yes | kind `checkin`, 300 s |
| `day.narrated` suggest / start | **not emitted by anything yet** (templates only) | none yet | none yet | none yet | none yet (`NARRATED_KIND` in `web/lib/speechText.ts` does not list them) |
| Approval card | `workspace.approval.requested` | card expires after 5 minutes | yes, high priority, Approve/Deny buttons (`PhoneNotifier.on_approval`) | yes unless on /approvals | kind `approval`, 30 s |
| Habit nudge | `habits.nudge.raised` (`contexts/habits/application/nudges.py`: ticker, one per call) | 20 min between ticker nudges (`min_gap_minutes`), per-habit cooldown (default 180 min), none while "asleep" by Habits' own schedule | yes (`on_nudge`) | yes | kind `nudge`, 120 s |
| Alert (package at the door, a rule) | `agent.alert.raised` (`contexts/agent/domain/rules.py`) | per rule `cooldown_s` (default 60, `config/rules.yaml` 120) | yes, high priority, with photo (`on_alert`) | yes | kind `alert`, 20 s |
| Plan / leaving checklist | `agent.plan.created` (door: `Planner.leaving`; exam within 48 h) | the agent's own triggers | yes (`on_plan`) | no | kinds `leaving` 300 s, `plan` 120 s |
| Waste Watch milestone | `impact.milestone.reached` | per milestone | not checked | yes | kind `milestone`, 60 s |
| Sort It ("which bin", round result) | the Play screen and the room twin (`web/app/play/page.tsx`, `web/components/map/RoomTwin.tsx`) | `voice.say(..., {tag: "sort-it"|"sort-it-rule", replace: true})` | no | no | spoken directly, no cooldown table |
| Look guide's directions | `web/components/guide/GuideLayer.tsx` | `tag: "guide"`, only with the "speak" setting | no | no | spoken directly |
| Conversation answers | the Ask screen (`web/app/page.tsx`), "Brief me" on Now | `tag: "answer", replace: true` | no | no | spoken directly; on the Ask screen it goes to the box's speaker first (`api.voiceSay` → `POST /api/voice/say`) when the box voice is up, else the browser |
| Box-side voice (`contexts/voice`) | push-to-talk, "Hey Atlas", `POST /say` | speaks only while idle | no | no | n/a |

How they are coordinated today:

- **On the box: not at all across kinds.** Each producer has its own cooldown and knows nothing of the others. The
  check-in's rule looks at the plan (fixed block, leave-by) but not at alerts, nudges or other narrations. A
  check-in, a coach note and a habit nudge can all go out in the same minute.
- **In the web voice (`web/lib/voice.tsx` `speakEvent`):** one cooldown per `SpokenKind`
  (`SPEAK_COOLDOWN_MS` in `web/lib/speechText.ts`), an event id is never said twice, at most 3 unprompted sentences
  queue (the oldest is dropped), speech never overlaps the microphone, and with the narrator on the raw
  `day.coach.noted` and the raw approval reason are not said as well as their narration. There is **no global gap
  between kinds and no priority**: two different kinds due together are said back to back, in arrival order.
  Unprompted speech does not wait for the look guide or Sort It; it shares one queue with them, first come first
  spoken (an `answer` job is dropped when the screen changes, an `event` job is kept).
- **Box voice versus web voice:** the box voice does **not** subscribe to the bus and never says a `day.narrated`
  line, an alert or a nudge. The only line both could say is a conversation answer, and the Ask screen already
  picks one speaker (`speakOnBox` first, the browser only when that returns false). So there is no double speech
  today, and also no signal "the box spoke this" for unprompted lines, because the box never speaks them.
- **Pushes:** with the narrator on, `day_phone` pushes the narration and not the raw event, so a moment is pushed
  once. Two known doubles: a check-in pushes "Check-in" from `day.checkin.raised` while its narration is not pushed
  (fine, one push); a plan pushes "Your day" (narration) **and** "Atlas wants to act" (its approval card): two
  pushes for one moment by design today. There is no dedupe key; `POST /api/day/brief` publishing a fresh `morning`
  narration pushes "Your day" again.

**The design to implement (decided; do not re-decide):**

1. A single ordering, highest first: urgent now (leave now, late, infeasible) → approval asks → check-in → block
   start → optional-event suggestion → coach note → evening review → habit nudges. Alerts about the room (a package,
   a visitor) are a separate urgent class that may interrupt anything except another urgent line.
2. One global minimum gap between any two unasked speak-ups (default 90 s; config key, suggested
   `day: {speak_gap_s: 90}`), which urgent lines may break; per-kind cooldowns stay as they are underneath it.
3. Merge instead of queue when two are due within the gap and neither is urgent: one sentence covering both from
   the same facts ("Time for lunch at the Student Union; and Riley's dinner is at quarter past five if you want
   it"), built by the narrator's template with both facts in the grounding set; never two separate sentences back
   to back.
4. Suppression rules: no block-start reminder for an event that had a leave-now or late line in the last 10
   minutes; no optional suggestion while a check-in card or an approval card is pending or within 15 minutes after
   a check-in; no coach note during a fixed event; nothing non-urgent while a conversation answer or the look guide
   is speaking (queue it, drop it if it is stale after 2 minutes); nothing at all outside the wake/bed window
   except urgent.
5. One speaker: when the box-side voice is active for a line, the phone/web voice stays silent for that line. The
   box voice does not speak unprompted lines today, so there is no existing signal: propose the smallest additive
   one (for example `spoken_by: "box" | None` on `DayNarrated`, or a small event the web can see) and specify the
   hook for the owner of `contexts/voice`; do not edit `contexts/voice`. One push per moment: dedupe key =
   day + moment + ref.
6. Where it lives: on the box, in the day context, pure and tested (suggested
   `contexts/day/domain/arbiter.py` + a thin `application/` wrapper the narrator's emit path goes through), deciding
   which `day.narrated` moments are emitted and merged. The web layer only enforces "one speaker" and the
   queue-while-speaking rule. It **wraps** `Narrator.say` / `Narrator.on_event` (every `day.narrated` passes
   through it) and the check-in's and suggestion's own gates stay underneath; in the web it wraps `speakEvent` in
   `web/lib/voice.tsx` (the per-kind `SPEAK_COOLDOWN_MS` stays). Habit nudges and alerts are other contexts'
   events: the arbiter reads them from the bus (replay=True for the timestamps) to apply the gap and the ordering to
   the day's own lines; it does not suppress an alert.
7. Tests to write: each priority pair, the global gap, a merge, each suppression rule, urgent breaking through,
   replay not re-speaking, the dedupe key, and one end-to-end "busy quarter hour" scenario (a block start, a
   suggestion, a check-in and a leave-now all due within 15 minutes) asserting exactly what is said, in what order,
   and what is dropped.

- **Acceptance:** the "busy quarter hour" test passes and no two non-urgent `day.narrated` events are less than
  `speak_gap_s` apart in any test log.
- **Files:** new `contexts/day/domain/arbiter.py`, `contexts/day/application/` (narrator hook), `loop.py`
  (`day_phone` dedupe), `contracts/day.py` (additive `spoken_by` if chosen), `web/lib/voice.tsx`,
  `web/lib/speechText.ts`, tests.
- Build items 2 to 5 so that their emissions go through the same single call, or build this first and plug them in.

### 2. Optional events you can say yes or no to (box side)

Brief, verbatim:

> - A small read model in the day context listing today's and the next days' optional events from Memory/connector
>   items with the `nearby:` prefix (title, start, end, place id resolved in the place pack, tags parsed from the
>   summary, who is going, source), each with: whether it fits the current plan (what it would overlap or displace),
>   the walk and leave-by from where the person will be, and its state: `suggested | accepted | declined | passed`.
> - `GET /api/day/optional` (list with fit, walk, leave-by, state) and `POST /api/day/optional/{id} {answer: "yes" |
>   "no"}`. **Yes** never changes the plan by itself: it creates a re-plan proposal through the existing mechanism
>   (`day.replan.proposed` + approval card) that inserts the event as a block (origin `optional`) at its time with
>   the walk there, shifting or shortening only suggested (non-fixed) blocks, never a fixed event; the response says
>   what would move. On approval it is a normal block: leave-by, "leave now", tracking and review all apply. **No**
>   records the decline so it is not suggested again. State is carried by additive events (e.g.
>   `day.optional.answered {id, answer, at}`; plan changes use the existing events), replayable; contract samples and
>   a fake step appended at the END of the day list in `fixtures/demo/fakes.yaml`.
> - Once accepted and approved it IS a commitment: the `nearby:` guards must keep skipping only events that are not
>   accepted. Make acceptance flow through the plan (a plan block), not by un-guarding the raw calendar item, so
>   Circle free/busy and the leaving checklist see it through whatever they already read for planned blocks; state
>   plainly which consumers do and do not see an accepted optional event, and fix the leaving checklist/next-event
>   path if an accepted one would be invisible there (one-line changes in teammates' files listed line by line, or
>   reported instead).

Design already worked out (verified against the code, not yet written):

- New `contexts/day/application/optional.py`, class e.g. `OptionalEvents(planner, tracker, clock, memory, narrator,
  config)`. State rebuilt from the log (all `replay=True`): `answers` from `day.optional.answered`; which plan
  carries which optional block from `day.plan.proposed` / `day.replan.proposed` (moves whose id starts with
  `optional:`); `accepted` from `day.plan.committed` of such a plan; `suggested` from `day.narrated` moment
  `suggest` (`ref` = the offer id).
- **How the block gets into the plan without touching Day Run:** `DayRun.board()` already takes extra calendar
  items from a port, `dayrun.agreed.items(lo, hi, tz, known)` (`contexts/play/dayrun/agreed.py`; `known` is the
  list of Memory items in the window, the `nearby:` ones included). Give `OptionalEvents` the same `items` method
  and chain it in front of whatever `dayrun.agreed` already is (do it in `DayLoop.__init__`: bootstrap sets
  `dayrun.agreed` before it builds the `DayLoop`). For every accepted offer found in `known`, return an
  `Item(connector="day", kind="calendar_event", external_id=block_id(offer.id), title=offer.title, when=start,
  summary="until <end> · <place name>")`: **no tags, so no friends' names** ever reach a plan block. The board then
  makes it a fixed block (`'63 South` locates to `ucf.dining_63`; checked for all seven seeded places), every later
  plan and re-plan and the Day Run screen carry it, and the tracker gives it `next_event`, leave-by, "leave now" and
  "late" like any fixed event. Skip one that a real calendar item already covers (same title and start), as
  `AgreedPlans.items` does.
- **A yes:** refuse with 409 when there is no plan for today, when the event starts after the board's window
  (24 h), or when `fit(...).fits == "no"` (say why). Otherwise temporarily include the offer (a context manager on
  `OptionalEvents`, e.g. `with self.trying(id):`) around `planner.replan(now, where, why, asked=True, ...)`; add a
  `for_optional` parameter to `DayPlanner.replan` so the event carries it. In `plan_moves`
  (`contexts/day/application/planner.py`) set `origin="optional"` for blocks whose id starts with `optional:`.
  Refuse if `changes(...)` lists a fixed row. Publish `day.optional.answered` first, then the re-plan and its card.
  A yes for an event too far ahead to place records the answer and answers `proposed: false`; it stays
  `suggested`, and the nudge still comes before it starts.
- **A no:** publish the answer. If the event was already accepted, also raise a re-plan without it (same
  mechanism, `for_optional` set, narrated through the `removed` branch).
- **The card's calendar write:** `card_events` in `planner.py` lists only non-fixed blocks. Include the optional
  block (plain title, no "Day Run:" prefix) in the card of the re-plan that introduces it, and not in later ones,
  so a box with a calendar connector writes it once and it syncs back as an ordinary calendar item.
- **The narrated yes:** in `Narrator.replan_facts` add `f["added"] = short_title(title)` when `e.for_optional`
  (template exists).
- **Suggested response shapes** (not built; settle them and tell Xander):

  ```
  GET /api/day/optional
  {"now": "2026-10-12T16:00:00-04:00", "plan_id": "plan-2026-10-12-0805",
   "events": [{"id": "nearby:friends-dinner:2026-10-12T17:15:00-04:00", "slug": "friends-dinner",
               "title": "Friends' dinner", "start": "...17:15...", "end": "...18:15...",
               "place": "ucf.dining_63", "place_name": "'63 South", "tags": ["with Riley, Sam, Alex", "optional", "via GroupMe"],
               "going": ["Riley", "Sam", "Alex"], "host": null, "source": "GroupMe",
               "state": "suggested", "answer": null, "awaiting_approval": false, "action_id": null,
               "fit": {"fits": "yes", "reason": "", "walk_minutes": 16, "leave_by": "...16:59...", "from_place": "apartment.desk",
                       "displaces": [], "conflicts": [], "leave_early_minutes": 0}}]}

  POST /api/day/optional/{id}   {"answer": "yes"}        (id: the full external id, URL-encoded, or the slug)
  {"id": "...", "answer": "yes", "state": "suggested", "proposed": true, "plan_id": "plan-2026-10-12-1601r",
   "action_id": "<card>", "moves": [{"id": "m0", "title": "Study at Desk", "change": "moved", "start": "16:30",
   "end": "17:30", "new_start": "16:05", "new_end": "16:50"}], "block": {...PlanMove, "origin": "optional"}}

  POST ... {"answer": "no"}  ->  {"id": "...", "answer": "no", "state": "declined", "proposed": false}
  ```

- **Which consumers see an accepted optional event** (the answer to give in the final report):
  - See it, through the plan: the day's plan and every later plan/re-plan (`GET /api/day/plan`, `/api/day/now`),
    the tracker (next event, leave-by, leave now, late, drift re-plans), the narrator, the evening review, the Day
    Run board (through the `agreed` port).
  - See it only on a box whose calendar connector can write, after the card's write syncs back: the Circle's
    free/busy (`busy()` in `contexts/circle/application/protocol.py`), Habits' "leave ready"
    (`contexts/habits/domain/tracker.py`), the agent's leaving checklist. All three read Memory's calendar items and
    nothing about planned blocks.
  - The leaving checklist must not be blind to it on a box without that connector. The sanctioned fix is one port
    and one changed condition in `contexts/agent/application/planner.py`: an attribute
    `self.accepted_optional = lambda: ()` in `__init__`, and line 284's guard becoming
    `and (not i.external_id.startswith("nearby:") or i.external_id in self.accepted_optional())`, set in
    `bootstrap.py` with one additive line. List both lines in the PR. For the Circle's `busy()` report it instead of
    changing it (it is a module function with no state; a fix is more than one line).
- **Acceptance:** an API test walks GET (suggested, fit yes) → POST yes (re-plan + card, `moves` listed, plan
  unchanged) → approve → `GET /api/day/plan` shows the block with `origin: "optional"`, `fixed: true`, and
  `/api/day/now` shows it as `next_event` with `walk_minutes` and `leave_by`; a second box built on the same log
  (replay) shows the same state; POST no → `declined` and never suggested.
- **Files:** `contexts/day/application/optional.py` (new), `planner.py`, `loop.py`, `api.py`, `narrator.py`,
  `contexts/gateway/bootstrap.py` (additive), `contexts/agent/application/planner.py` (the two lines above),
  tests in `contexts/day/tests/`.
- **Privacy tests to extend** (brief: "an accepted optional event must not leak the friends' names from the
  summary into the Circle or the training export"): in `contexts/circle/tests/test_health_privacy.py` and
  `contexts/learning/tests/test_export.py`, accept an event and assert no "Riley"/"Sam"/"Alex" in the Circle outbox
  or the export, and that no plan block's title or id holds a name.

### 3. Atlas suggests it (the nudge)

Brief, verbatim:

> - A narrated, spoken suggestion at a sensible moment: when an optional event is coming up within a configurable
>   window (default: between 90 and 30 minutes before it starts), it fits or nearly fits, friends are named, the
>   person is not in a fixed block or within 30 minutes of a leave-by, and nothing was suggested in the last 45
>   minutes: one `day.narrated` moment (add `"suggest"` to the moment literal, additively) with spoken and display
>   text from real facts only [...]. Template engine first; the model rewrite path as for other moments, with the
>   same grounding check. At most once per event; never for declined or passed ones; off with
>   `day: {suggest_optional: false}`.
> - Answering: the toast / Now card carries "Add it" and "Not today" (web: you may add a small card component under
>   `web/components/day/` and mount it via the app-level toast/notification path in `web/lib/atlas.tsx`, NOT by
>   editing `web/app/now/`); by voice, "yes / add it / sure" and "no / not today" within 30 seconds of the suggestion
>   being spoken map to the POST above (web voice layer, additive; the yes still only creates the approval card:
>   voice cannot approve). If the approval flow makes "yes" a two-step for the user, make the second step one obvious
>   tap in the same card ("Review and approve").

Decided:

- Config keys under `day:` in `config/agent.yaml` (none is read yet): `suggest_optional: true`,
  `suggest_from_min: 90`, `suggest_until_min: 30`, `suggest_cooldown_min: 45`. The rule is `suggest_held`.
- Box: `OptionalEvents.tick(now)` (live only, looked at once a minute of event time at most) builds the `Gate` from
  the plan (fixed block running; the tracker status' `next_event.leave_by`), the clock (`wake_at`/`bed_at`) and its
  own replayed state, then hands the facts to the narrator. Add one narrator entry point, e.g.
  `Narrator.tell(moment, facts, ts, day, plan_id, ref)`, that builds the `Job` (event id
  `day-narrated-<moment>-<ref>`) and returns `self.say(job)`: template now, or the model's grounded rewrite from
  the background task. Keep a live-only "asked" set so an in-flight model job is not asked twice. `ref` = the
  offer id. For `before`, use the next block that is not a rest, labelled with `Narrator.label` ("your study
  block", "lunch"; say "bed" for the night's sleep).
- Push: do not put friends' names through ntfy. One line without names ("Atlas has a suggestion near you: Friends'
  dinner at 5:15 PM. Open Atlas to add it.").
- Web: a card component `web/components/day/OptionalSuggestion.tsx` mounted inside `AtlasProvider`
  (`web/lib/atlas.tsx`, next to `{children}`), fixed-position, listening on `/ws` for `day.narrated` (moment
  `suggest`), `day.optional.answered`, `day.replan.proposed` (`for_optional`) and the approvals list; buttons
  "Add it" / "Not today"; after a yes the same card shows what would move and one button "Review and approve"
  (link to `/approvals/`, or approve in place through the existing approvals API: a tap, never voice). Add
  `suggest` and `start` to `NARRATED_TOAST` only if the card does not already show them (do not show both).
  API client additions go in `web/lib/api.ts` (`DayMoment` needs the two new members).
- Voice: in `web/lib/voice.tsx`, when the spoken suggestion finishes (`onDone(true)`), remember
  `{id, until: now + 30 s}`; wrap `handlers.onFinal` inside `listen` so that within that window "yes / add it /
  sure" → POST yes and "no / not today" → POST no, and nothing else is swallowed. Add the matcher to
  `web/lib/speechText.ts` next to `voiceCommand`. Then say "Okay, there's a card to approve." **Voice cannot
  approve** (see `web/app/page.tsx` line 204 for the existing rule).
- Add both moments to the spoken set: `NARRATED_KIND` (`suggest: "suggest"`, `start: "start"`), `SpokenKind`,
  `SPEAK_COOLDOWN_MS` (suggested: `suggest` 300 s, `start` 60 s).
- **Acceptance:** tick tests: said at 16:00 for a 17:15 event, not at 15:44 or 16:46, not twice, not after a no,
  not after a restart on the same log, not with `suggest_optional: false`; the narration appears in
  `GET /api/day/brief` under `latest.suggest`; `npx tsc --noEmit` and `npm run build` pass.
- **Files:** `contexts/day/application/optional.py`, `narrator.py`, `loop.py`, `config/agent.yaml` (commented
  defaults), `web/lib/atlas.tsx`, `web/lib/api.ts`, `web/lib/voice.tsx`, `web/lib/speechText.ts`,
  `web/components/day/` (new).

### 4. Reminders when a planned block starts

Brief, verbatim:

> - The missing behaviour: when the time for the next PLANNED block arrives (any block of the committed plan: study,
>   meal, rest, an accepted optional event; fixed events already have leave-now), Atlas says so once: a
>   `day.narrated` moment `"start"` (additive) [...]. Built from the tracker's status (where the person is, whether
>   they are already doing it: if the camera already sees them doing it, say nothing or a short acknowledgement;
>   decide and test), with lead time for blocks that need a walk (use the leave-by), a cooldown (never two reminders
>   within 10 minutes; batch back-to-back short blocks: "three short study blocks from two"), quiet hours from the
>   configured wake/bed window, never during a fixed event, and never repeated after a restart (live-only emission
>   keyed by plan id + block id; replay rebuilds what was said). `day: {block_reminders: true}` to switch off.
> - The ntfy push for it follows the existing day push rules (one line, no health words); the web voice speaks it
>   when "speak up on its own" is on (same cooldown mechanism as other moments; add the two new moments to the
>   spoken set and to `speechText`).
> - `GET /api/day/brief` already lists the latest narration per moment: make sure the new moments appear there.

Decided (not written; a pure `contexts/day/domain/reminders.py` plus `application/reminders.py`, ticked from
`DayLoop.tick`):

- Which blocks: non-fixed blocks of the **committed** plan (`plan.plan_id in planner.committed`), plus blocks with
  `origin == "optional"`. For an accepted optional event that needs a walk of 3 minutes or more, the tracker's
  leave-now line is the reminder: say no `start` for it (this is also suppression rule 4 of item 1).
- When: the trigger is the block's start, or `start - walk` when the person is seen somewhere else (or the plan
  has them somewhere else) and the walk is 3 minutes or more. Said if `trigger <= now < trigger + 5 min`.
- Already there: seen at the block's place → the line ends "You're at the desk already." Continuing (the block
  before it ended within 5 minutes, same kind, same place, and the person is there) → say nothing.
- Batching: a run is consecutive blocks of one kind, each 30 minutes or less, with rests of 30 minutes or less
  between them (the chooser really plans 25 min study / 15 min rest); two or more → one line with `count` and
  `until`, and every block of the run (rests included) counts as said.
- Quiet: 10 minutes between two reminders; nothing outside `clock.wake_at(day) <= now < clock.bed_at(day)`;
  nothing while a fixed block is running.
- Once: `ref = "<plan_id>:<block ids joined by +>@<HH:MM>"`; state rebuilt from `day.narrated` moment `start`
  (`replay=True`): remember each `plan_id:block_id` and each `day@HH:MM`, so a re-plan landing mid-window does not
  re-announce the same start. Emission is from the tick only (live).
- Facts: `block` from `Narrator.label`, `deadline` from the move's deadline title (`short_title`), `place` from
  `short_place`, `there`, `walk_min`, `min`, `count`, `until`.
- Push: add `"start": ("Next up", "bell")` to `MOMENT_PUSH` (the display line; no health words in it by
  construction).
- Config: `day: {block_reminders: true}`.
- **Acceptance:** tests for lead time with a walk, the already-there line, silence when continuing, batching, the
  10-minute cooldown, quiet hours, not during a fixed event, not for an uncommitted plan, and a second loop built on
  the same log saying nothing; `latest.start` in `GET /api/day/brief`; a push test in the style of the existing
  `day_phone` tests in `contexts/day/tests/test_narration.py`.
- **Files:** `contexts/day/domain/reminders.py`, `contexts/day/application/reminders.py`, `loop.py`, `narrator.py`,
  `config/agent.yaml`, `web/lib/speechText.ts`, tests.

### 5. Demo and docs

Brief, verbatim:

> - The seeded night must show it: with the `nearby:` events already in the seed, `python -m contexts.demo seed
>   --now` (and `circle`) should leave one optional event in the suggestion window where the story allows, or add a
>   `python -m contexts.demo optional --data-dir <dir>` scene that plays suggest → yes → card → approve (locally) →
>   block reminder → leave-by, deterministic and labelled demo. Must not change today's seeded plan, next event or
>   leave-by for the existing story, the golden eval (48/48), or existing seed test expectations (list anything you
>   had to update).
> - Add the flow to `docs/demo-run-sheet.md` (steps, expected lines, fallback) and steps to
>   `scripts/demo_walkthrough.py` (read-only GETs, plus the POST "yes" only under the existing `--approve-local`
>   style guard: the card it creates is a local plan change; check its kind/target before approving in the script,
>   as the script already does).

Decided: add the scene (`contexts/demo/optional.py`, modelled on `contexts/demo/checkin.py`, wired in
`contexts/demo/__main__.py`). An invented day: at 08:05 the calendar sync of `checkin.calendar(end)` plus the
fixture's `friends-dinner` with `day` overridden to 0 (so it is today at 17:15); a person at the desk from 15:50;
then tick minute by minute from 16:00: the suggestion at 16:00, a yes at 16:01 (the same function the API calls),
the card approved at 16:02 by publishing `experience.approval.decided` (labelled "demo, approved locally"), the
block reminder at the next block's start, and the tracker's "Time to go" at the leave-by. Print each line.

- **Acceptance:** two runs into two empty directories print the same lines; `contexts/demo/tests` for the seed and
  `nearby` stay green unchanged; a gateway started on the scene's directory shows the block in `/api/day/plan`.
- **Files:** `contexts/demo/optional.py`, `contexts/demo/__main__.py`, `contexts/demo/tests/`,
  `docs/demo-run-sheet.md`, `scripts/demo_walkthrough.py`.

### 6. Final verification and report (section f)

---

## (d) Pitfalls found

- **The seeded story never commits a plan.** No `day.plan.committed` is in the seeded log (every card expires after
  five minutes of story time). Block reminders are for the committed plan, so the seeded night will show none; that
  is why the demo scene approves a card. If the owner wants reminders for an unapproved morning plan too, that is
  his decision to make, not ours.
- **Wake and bedtime are 08:00 and 20:00** in `config/agent.yaml` (`sleep:`), and `DayClock.bed_at` is when the
  loop's day ends. The seeded plan has study blocks until 22:07: with quiet hours from that window nothing is said
  after 20:00. Board game night (20:00) is suggested 18:30 to 19:30, inside the window.
- **Fixed blocks that overlap are silently reshaped.** `lay_out` in `contexts/play/dayrun/board.py` makes the later
  of two overlapping fixed blocks start when the first ends. An optional event must therefore never be put on the
  board when `fit` says `"no"`, or it would cut a real fixed event short.
- **`planner.replan` re-chooses every suggested block from `now`**, it does not edit the current plan. "What would
  move" must be the honest diff (`changes`), and it can list more than the blocks the event overlaps.
- **Approval cards live five minutes.** Never raise the yes card on a timer or ahead of time; raise it when the
  person says yes.
- **The narrator may answer later.** `Narrator.say` returns `[]` when it hands the job to the model in the
  background; "said once" state must come from the `day.narrated` event (replay) plus a live-only in-flight set.
  A draft under 120 characters (`MIN_DRAFT_CHARS`) is never sent to the model: most block-start lines are template
  only. With `day.narration: off` the narrator publishes nothing, so there are no suggestions or reminders either.
- **Grounding:** the check lets through any word that is in the facts' strings or in the template's draft. Meal
  words, weekdays and claims ("late", "behind") are guarded: put the title in the facts so "dinner" is grounded.
  Times in facts must be `"HH:MM"` strings, durations under keys ending `_min`.
- **Replay rules:** state handlers `replay=True` and pure; anything that publishes is live only; use `event.ts` /
  the tick's `now`, never the wall clock. The gateway's ticker calls `day.tick(datetime.now(UTC))`; the seed and
  the demo scenes call `app.day.tick(<event time>)` themselves, which is how tests and demos advance time.
- **The seed ends with one `day.tick(now)`.** Once the suggestion tick exists, `seed --now` at a moment 90 to 30
  minutes before a seeded optional event adds one `day.narrated` (suggest) to the log. With the script's own end
  (Sunday 09:30) no event is in the window, so the committed seed expectations do not move; check
  `contexts/demo/tests` anyway.
- **`BOX_ONLY_EVENTS` is asserted in three places** (`contexts/circle/tests/test_health_privacy.py`,
  `contexts/learning/tests/test_export.py`, `contexts/day/tests/test_checkin.py`), and the export test builds one
  event per member from `SAMPLES`: every new box-only type needs a sample.
- **Structural privacy tests** forbid `contexts/circle`, `contexts/play/league` and `contexts/learning` from
  naming `contracts.day`, `contexts.day` or any `"day.` string. Do not fix the Circle's `busy()` by importing
  anything of the day context there.
- **`fixtures/demo/fakes.yaml`:** a step's `at:` is the offset in seconds and is popped before the payload is
  built; the event's own `at` field comes from the sample. New steps go at the end of a context's list or earlier
  fake ids shift.
- **The `nearby:` guards** are four one-liners: `ignored_event_prefixes` (`contexts/play/dayrun/config.py`,
  `config/dayrun.yaml`), `Planner.leaving` (`contexts/agent/application/planner.py` line 284), `busy()`
  (`contexts/circle/application/protocol.py` line 79), Habits `on_item` (`contexts/habits/domain/tracker.py` line
  179). `contexts/demo/tests/test_nearby.py` asserts each of them keeps skipping unaccepted events.
- **Tooling in this worktree:** multi-line shell with heredocs feeding Python was refused by the sandbox here; use
  the file tools for edits and single plain commands. The full test run takes about ten minutes.

---

## (e) Rules of the repo that apply

- Read `AGENTS.md`. Contexts talk only through events in `contracts/`; pure projections `replay=True`, side effects
  live only; event time, not the wall clock; every plan change through an approval.
- Do not edit: `web/app/now/` (the designer's, Xander's: he is adding "Near you" yes/no there and will call the
  API), `web/app/dayrun/`, `web/components/dayrun/` (another session is redesigning that screen; it will show
  optional events as "+ Add" suggestions and can call the API), `contexts/voice`, `web/components/operator/`,
  `web/components/Shell.tsx`, the design tokens, `lab/`, `runtime/`.
- Additive only: `contracts/`, `contexts/gateway/bootstrap.py`, `contexts/gateway/app.py`.
- One-line fixes only, each listed in the report: `contexts/agent/`, the Circle protocol.
- No push, no PR, no Notion unless the owner says so. Small commits, each message ending
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never `pkill`/`killall`; not ports 8800, 8000, 18080; gateway on a port in 8930 to 8990 with
  `ATLAS_CAPTURE=0 ATLAS_CONNECTORS=0 ATLAS_MEMORY_FIXTURE= NEON_DATABASE_URL= NTFY_URL=`. No new dependencies, no
  network, do not contact the Jetson. Temporary files only in a new scratch subfolder.
- Tests: `/Users/hemduttrao/Knighthacks26/.venv/bin/python -m pytest -q --deselect tests/test_lab_register_pool.py
  --deselect tests/test_lab_placement.py --deselect tests/test_lab_settings_searchspace.py --deselect
  contexts/gateway/tests/test_app.py::test_answers_stream_word_by_word_over_the_websocket_before_the_final_answer`;
  `python -m contexts.agent.eval` must say 48/48; for `web/`: `npx tsc --noEmit` and `npm run build` with
  `web/node_modules` symlinked from `/Users/hemduttrao/Knighthacks26/web/node_modules` (remove the symlink before
  every commit; time-limit the build).
- The default theme is Night; Day is behind the theme switch. Check both.

---

## (f) How to verify at the end

1. Full test run, the golden eval, `tsc`, `npm run build`.
2. The API walk on a seeded directory (or the demo scene's): the suggestion appears in `GET /api/day/brief`
   (`latest.suggest`); `POST /api/day/optional/{id} {"answer": "yes"}`; the card in `GET /api/approvals`; approve
   it; `GET /api/day/plan` shows the block with its walk and `/api/day/now` its leave-by; the start reminder is
   narrated when event time reaches it (`latest.start`).
3. Screenshots of the suggestion card at 500×1000 and 1440×900 in both themes, and read them.
4. `docs/demo-run-sheet.md` (steps, expected lines, fallback) and `scripts/demo_walkthrough.py` (read-only GETs;
   the POST yes only under the existing `--approve-local` style guard, checking the card's kind/target before
   approving) are updated. Both are still owed.
5. Stop the gateway by pid; delete temp dirs and builds; `git status` clean.
6. The report the owner asked for: branch, commits, test totals; the API (routes, bodies, example responses); the
   event contract additions; the exact rules (windows, cooldowns, quiet rules) with defaults and config keys;
   example spoken lines for a suggestion, a yes, a block start with and without a walk, batched blocks; which
   consumers see an accepted optional event; what the seeded demo shows; lines changed in teammates' files; the note
   for Xander; what is not done or not verified (spoken output cannot be heard headlessly).

---

## (g) Note for Xander (the Calendar's "Near you" yes/no)

1. List with `GET /api/day/optional`: each event has `id`, `state` (`suggested | accepted | declined | passed`),
   `fit` (`fits`, `walk_minutes`, `leave_by`, what it would move) and `awaiting_approval`; show yes/no only for
   `suggested`.
2. Answer with `POST /api/day/optional/{id}` and `{"answer": "yes"}` or `{"answer": "no"}` (URL-encode the id).
   A yes changes nothing by itself: the answer carries `action_id` and `moves`; send the person to that approval
   card ("Review and approve"). Never approve for them.
3. Refetch on the `/ws` events `day.optional.answered`, `day.replan.proposed` and `day.plan.committed`: the event
   turns `accepted` only after the card is approved. (The endpoint does not exist yet: shapes are in section c,
   item 2, and may still change; whoever builds it must confirm them with you.)

---

## (h) Items 2 and 3 as built (added after sections a to g; where it differs from them, this section is what the code does)

Items 2 and 3 of section (c) are on this branch. Items 5 and 6 are not started. Everything below is in
`contexts/day/application/optional.py` unless another file is named; the tests are
`contexts/day/tests/test_optional_events.py` (the answers) and `contexts/day/tests/test_optional_suggest.py` (the
suggestion).

### The API (settled; section g's "may still change" no longer applies)

```
GET /api/day/optional
{"now": "...", "plan_id": "plan-2026-10-12-0810" | null,
 "events": [{"id": "nearby:friends-dinner:2026-10-12T17:15:00-04:00", "slug": "friends-dinner", "title": "Friends' dinner",
             "start": "...", "end": "...", "place": "ucf.dining_63", "place_name": "'63 South",
             "tags": ["with Riley, Sam, Alex", "optional", "via GroupMe"], "going": ["Riley", "Sam", "Alex"], "host": null,
             "source": "GroupMe", "state": "suggested" | "accepted" | "declined" | "passed", "answer": "yes" | "no" | null,
             "awaiting_approval": false, "action_id": null, "plan_id": null,
             "suggested_at": "..." | null, "suggest_held": "already suggested" | ... | null,
             "fit": {"fits": "yes" | "nearly" | "no", "reason": "", "walk_minutes": 16, "leave_by": "..." | null,
                     "from_place": "apartment.desk", "leave_early_minutes": 0,
                     "displaces": [{"id", "title", "kind", "start", "end", "overlap_minutes"}], "conflicts": [...]}}]}

POST /api/day/optional/{id}   {"answer": "yes" | "no"}     (id: the full id, URL-encoded, or the slug)
yes -> {"id", "answer": "yes", "state": "suggested", "proposed": true, "plan_id", "action_id",
        "moves": [{"id", "title", "kind", "fixed": false, "change": "moved" | "shortened" | "lengthened" | "dropped",
                   "start": "18:30", "end": "19:00", "new_start"?, "new_end"?}],
        "adds": [{"id", "title", "kind", "start": "18:20", "end": "19:40"}],
        "block": {...PlanMove, "fixed": true, "origin": "optional"}, "fit": {...}}
yes, more than a day ahead -> {"id", "answer": "yes", "state": "suggested", "proposed": false, "reason": "..."}
no  -> {"id", "answer": "no", "state": "declined", "proposed": false}
no to an accepted event -> {"id", "answer": "no", "state": "accepted", "proposed": true, "plan_id", "action_id", "moves", "adds"}
404 unknown event; 409 with the reason in `detail` (no plan for today yet; it does not fit; it has already started;
it is already in your plan); 422 for another answer.
```

- `awaiting_approval` with `action_id` and `plan_id` is true while the card about the event is open: the re-plan a
  yes raised, or the re-plan a no to an accepted event raised. A card that expires leaves the event `suggested`
  with `answer: "yes"`: the person can say yes again.
- `moves` is what happens to the blocks Atlas had chosen, as far ahead as the current plan reaches; `adds` is what
  the re-plan puts there instead. A re-plan chooses every suggested block again, so both can be longer than what
  the event itself overlaps. Times in them are local `HH:MM`.
- `suggest_held` says why Atlas is not bringing the event up right now (null when the next tick may).

### What differs from the design in section (c)

- A yes to an event that starts after the board's window (24 h) is not a 409: the answer is recorded and the
  response says `proposed: false` with a reason. It stays `suggested` and is still brought up before it starts.
- `OptionalEvents` takes the place in front of `dayrun.agreed` from the first yes on (the first time a re-plan is
  built with an event, or an accepted one is replayed), not in `DayLoop.__init__`:
  `contexts/play/tests/test_dayrun_agreed.py` asserts the port is `None` on a box without a Circle, and a box nobody
  said yes on builds its boards exactly as before.
- `day.optional.answered` gained `until: datetime | None`. When the fit is "nearly" because of leaving early for
  the next fixed event, the block ends that much earlier (the yes carries the time), so the next fixed event is
  reached on time and is never moved or cut.
- The "a fixed event never moves" check compares the day re-planned at the same moment with and without the
  event, and refuses when any fixed block differs. `changes()` against the current plan is only used for `moves`.
- Two yeses in a row: the second re-plan also carries the first event while its card is still open, so approving
  the newer card accepts both.
- A plan's card counts for acceptance only when that plan is what the day is now measured against: a card approved
  after a newer plan replaced it accepts nothing.
- The suggestion's screen line (`display`) no longer names who is going: "Friends' dinner: 5:15 PM to 6:15 PM at
  '63 South. 16 min walk, leave by 4:59 PM. Fits before your study block at 7:01 PM. Add it?". The push carries
  that line, and a block start it is merged into shows it, so no friend's name goes through the notification
  service. The names are in `spoken` and in `going` / `host` of the list.
- The push is `"Atlas has a suggestion near you: <display without 'Add it?'> Open Atlas to add it."`, title
  "Near you" (`suggest_push` in `contexts/day/application/loop.py`).
- "It fits before ..." names the plan's next block that is not a rest, and not a meal when the event is itself a
  meal ("dinner ... fits before dinner" says nothing); the night's sleep is "bed".

### The suggestion's rules (defaults; config keys under `day:` in `config/agent.yaml`)

Said once per event, on the Day loop's tick (looked at once a minute of event time), when all of these hold:
`suggest_optional` is true; the narration is not off; there is a plan for today; the event starts in
`suggest_until_min` (30) to `suggest_from_min` (90) minutes; it fits or nearly fits; it names somebody (`going` or
`host`); it is between wake and bedtime; no fixed block is running; it is not within 30 minutes before a time to
leave for a fixed event (until that event starts); the last suggestion was at least `suggest_cooldown_min` (45)
minutes ago; the event is still `suggested`. The arbiter adds: no approval card or check-in card open, not within
15 minutes after a check-in, the global gap, merging with a block start due in the same tick. "Said once" is read
from `day.narrated` moment `suggest` (`ref` is the event's id; a suggestion said inside another line, with
`merged_into`, counts), so a restart on the same log says nothing again.

### Web

- `web/components/day/OptionalSuggestion.tsx` (+ `optional.module.css`), mounted by `AtlasProvider`: the card, in
  the toasts' place and below any toast. "Add it" / "Not today"; after a yes, what would move and one link,
  "Review and approve", to `/approvals/`; "In your plan" once the card is approved. Hidden on `/approvals` and
  `/operator`. A `suggest` narration is not toasted as well.
- `web/lib/voice.tsx`: for 30 seconds after Atlas has said a suggestion (its own line, the line it was merged
  into, or a line the box's speaker says), a heard "yes / add it / sure" or "no / not today" answers it
  (`suggestionAnswer` in `web/lib/speechText.ts`) and Atlas says "Okay, there's a card to approve." or "Okay, not
  today.". A yes only ever raises the card. Anything else is heard as usual.
- `web/lib/api.ts`: `api.dayOptional()`, `api.dayOptionalAnswer(id, answer)` and their types;
  `web/scripts/check-api.mjs` checks `GET /api/day/optional`.

### Who sees an accepted optional event

- Through the plan: `GET /api/day/plan`, `/api/day/now`, every later plan and re-plan, the tracker (next event,
  walk, leave-by, leave now, late, drift re-plans), the narrator, block-start reminders, the evening review, the
  Day Run board.
- The agent's leaving checklist: yes, through `Planner.accepted_optional` (set in `bootstrap.py`).
- Only after a calendar connector wrote the card's event and synced it back: the Circle's free/busy (`busy()` in
  `contexts/circle/application/protocol.py`) and Habits' "leave ready" (`contexts/habits/domain/tracker.py`). On a
  box without a calendar write they do not see it. Not changed: each needs more than one line.

### Still open

- Items 5 and 6: the demo scene, the run sheet, `scripts/demo_walkthrough.py`, the final walk and screenshots.
- A no to an accepted event on a box whose calendar connector wrote it: the calendar item stays (there is no
  delete tool), so it comes back as an ordinary calendar block.
- A no said while the yes card is still open is recorded, and the card stays open: approving it afterwards still
  adds the event (the approval is the later decision).
- The spoken yes and no were checked as code and by phrase list, not by ear.

---

## (i) Items 5 and 6: the demo scene, and the report section (f) asks for

Written when items 1 to 5 were on the branch. Items 1 and 4 (the arbiter, block-start reminders) were built in
another session on this branch; what is said about them here is read from their code and tests and seen in the
runs below, not built in this one. Section (h) holds the API shapes and the differences from the design; this
section does not repeat them. Where section (h) says items 5 and 6 are not started, this section is the later
word: both are done with it.

### 1. Branch, commits, test totals

Branch `ctx/day/reminders-and-optional`, draft PR 311. Not rebased on `main` and `main` not merged in; it merges
cleanly with `main` as of `a6390c4` (a trial merge that was not pushed: the day, demo, gateway, walkthrough and
privacy tests passed on it, 577 passed and 1 skipped, `tsc` was clean and `check-guide.mjs` passed 392 checks).
Commits since the hand-over (`a6aa65d`), oldest first:

| Commit | What |
| --- | --- |
| `a17a1ad`, `ca81b54`, `6b90fee` | Item 1: the arbiter for everything said unasked, its rules and tests |
| `5c92e66`, `3590d72` | Item 4: block-start reminders through the arbiter, one push per moment, tests |
| `cd40673`, `3ddc580` | Web: one speaker, the two new moments; the box's speaker can say the unasked lines |
| `01e9ef9`, `5d4ee10`, `a169d80` | Item 2: the read model, the API, the yes behind the approval, the leaving checklist, privacy tests |
| `6698605`, `010bc2e`, `0826f5f` | Item 3: the suggestion tick through the arbiter, the web card, the spoken yes and no |
| `c6f6c49` | Section (h) of this document |
| `b273fed` to `db2d345` (eight checkpoints) and the commit that adds this section | Item 5: the scene, its tests, the walkthrough steps, the run sheet |

Totals:

- Full suite on Windows with the deselections of section (e), on `5614376`:
  `6 failed, 6568 passed, 36 skipped, 87 deselected, 1 warning in 682.54s (0:11:22)`. None of the six is in this
  work: the two file-mode tests in `contexts/circle/tests/test_face_share*.py` and the file-lock test in
  `contexts/memory/tests/test_read_side.py` fail the same way on a clean clone of `main` on that machine;
  `contexts/connectors/tests/test_health.py` fails between midnight and 7:26 AM on this branch's base and is
  fixed on `main` (PR 325); one test each in `contexts/perception/tests/test_face_profile.py` and
  `contexts/telemetry/tests/test_jetson_cpu.py` depends on timing and passed twice when run alone.
- After that run two commits followed: `26fe9d4` moved one row of the run sheet, and `db2d345` moved the
  optional event's calendar entry from `card_events` into `DayPlanner.card`, because `main` has since changed
  `card_events` and the two changes could not be merged. On `db2d345` the day, demo, gateway, play and agent
  contexts with the contract samples, the walkthrough's tests and the two privacy files:
  `1057 passed, 1 skipped, 1 deselected, 1 warning in 126.02s (0:02:06)`.
- CI on the pull request (Linux, the branch merged with `main`) for `db2d345`:
  `77 failed, 7372 passed, 38 skipped, 4 warnings, 11 errors in 315.03s (0:05:15)`. All 88 are in five lab test
  files (`tests/test_lab_metrics.py`, `tests/test_lab_routing_closed_loop.py`, `tests/test_lab_metrics_backfill.py`,
  `contexts/learning/tests/test_lab_bench.py`, `contexts/learning/tests/test_lab_experiments.py`) and `main` itself
  fails with the same 77 and 11 (`a6390c4`: `77 failed, 7249 passed, 38 skipped, 4 warnings, 11 errors`). The last
  green CI run of this branch was on `a169d80`, before `main` turned red: 7058 passed, 39 skipped.
- `python -m contexts.agent.eval`: `score 1.00 (48/48) model=fake-0`.
- `web/`: `npx tsc --noEmit` exit 0; `npm run build` exit 0; `node scripts/check-guide.mjs`: 223 checks passed;
  `node scripts/check-api.mjs` against a gateway on the scene's directory: all checks passed.
- `scripts/demo_walkthrough.py --approve-local` against a gateway on the seeded night (`seed --now`, today's plan
  made first): 100 PASS, 1 FAIL, 20 SKIP (121 steps). The FAIL is the run sheet's known defect 1 ("What happened
  while I was out?"). Against a gateway on the scene's directory, `--only day,coach`: 16 PASS, 0 FAIL, 5 SKIP.

### 2. The demo scene (item 5)

```
python -m contexts.demo optional --data-dir <an empty directory> [--day YYYY-MM-DD]
```

It publishes one invented day through the real app (`contexts/demo/optional.py`; every value is invented and the
output says so) and prints each line. Its lines for `--day 2026-10-12`, the same on every run and for every
directory (the command that runs the app on the directory goes to stderr, because it names the directory):

```
An invented Monday (2026-10-12), through the real app. Nothing here happened to anybody.

   8:05 AM  Atlas  Morning. Today looks like this: breakfast first, then a long study block before COP4533 lecture at twenty to eleven. Leave by half past ten for that; it's an eight-minute walk. Project Milestone 1 is due tomorrow, and this plan covers the five hours it still needs. Want me to keep this as today's plan?
   8:06 AM  You    Approve (demo, approved locally: no calendar connector on this box, nothing is written)
   1:01 PM  Atlas  It's one o'clock: your study block on Project Milestone 1, ninety minutes. You're at the desk already.
   4:00 PM  Atlas  Riley, Sam and Alex are having dinner at '63 South at quarter past five. It's a sixteen-minute walk and it fits before your study block. Want me to add it?
   4:02 PM  You    Add it (yes to Friends' dinner)
   4:02 PM  Plan   would change: Eat at Kitchen at 6:30 PM (dropped)
   4:02 PM  Plan   would change: Study at Desk at 7:01 PM (dropped)
   4:02 PM  Plan   would change: Rest at Desk at 8:31 PM (dropped)
   4:02 PM  Plan   would change: Study at Desk at 9:16 PM (dropped)
   4:02 PM  Atlas  Okay, I've worked Friends' dinner in: first up is Friends' dinner at quarter past five. Leave by five to five for that; it's a sixteen-minute walk. Want me to switch to that?
   4:03 PM  You    Approve (demo, approved locally: no calendar connector on this box, nothing is written)
   4:03 PM  Plan   Friends' dinner, 5:15 PM to 6:15 PM at '63 South, is a block of the day
   4:50 PM  Atlas  Time to go. Friends' dinner starts at quarter past five at '63 South, a sixteen-minute walk from here; leave in five minutes.

Suggested once, added only after its card was approved, and told when to leave. Every line Atlas says above is the app's own.
```

Against the acceptance lines of item 5:

- Two runs into two empty directories print the same lines: `contexts/demo/tests/test_optional_demo.py`
  (`test_two_runs_into_two_empty_directories_print_the_same_lines`), and by hand.
- `contexts/demo/tests` for the seed and `nearby` are green and unchanged. No existing expectation was updated.
- A gateway started on the scene's directory shows the block in `GET /api/day/plan` (`fixed: true`,
  `origin: "optional"`, committed): the same test file, and by hand on a test gateway, where `/api/day/now` also
  had it as the next event with a 16 minute walk and leave-by 4:59 PM, and `/api/day/brief` had `latest.suggest`
  and `latest.start`.
- The seeded story is untouched: nothing in `contexts/demo/seed.py`, `story.py`, `nearby.py` or the fixtures
  changed, and the seeded plan, next event and leave-by are what the unchanged seed tests assert.
- The golden eval is 48/48.

Where the scene differs from the sketch in section (c) item 5, and why:

- The sync is at 7:55 AM, before wake, and the plan is made at 8:05 AM when someone sits down. A sync at 8:05 AM
  makes the plan on the first item that arrives, before the assignment and the dinner are in Memory.
- The morning plan's card is approved too (locally, like the yes's card), and the scene watches the afternoon's
  study block start. After the yes is approved the next block of the day is the dinner itself, which is announced
  by its leave-now line, and the re-plan puts no chosen block before it: without the approved morning plan the
  day would have no block-start line at all.
- The yes is at 4:02 PM and the approval at 4:03 PM, not 4:01 and 4:02. A yes inside the arbiter's ninety seconds
  after the suggestion makes the spoken "Okay, I've worked it in" wait for the gap (see "found" below).
- The scene ends at the time to leave. The box's cameras are at home: past the leave-by the tracker can only see
  that nobody left, and says the event will be missed.

The run sheet has the flow as use case 3a (`docs/demo-run-sheet.md`), and `scripts/demo_walkthrough.py` reads the
optional events in every mode and, only with `--approve-local`, says yes to one and puts its card through the
existing guard (approved only when it stays on the box, else denied). It never says no.

### 3. The event contract additions (all additive)

- `PlanMove.origin` ("optional" for an accepted optional event); `DayReplanProposed.for_optional`.
- `Moment` gains `suggest` and `start`.
- `day.optional.answered` (`DayOptionalAnswered`: `day`, `id`, `answer`, `at`, `plan_id`, `action_id`, `until`),
  box-only.
- `DayNarrated.spoken_by` ("box" when the box's own speaker says the line) and `DayNarrated.merged_into` (said
  inside that narration: shown, never said or pushed again).

### 4. The rules, with defaults and config keys (`config/agent.yaml`, block `day:`)

- Everything said unasked goes through one arbiter (`contexts/day/domain/arbiter.py`). Order, highest first:
  leave now and late, approval asks and plans, check-in, block start, suggestion, coach note, evening review,
  habit nudges; room alerts are their own urgent class. At least `speak_gap_s` (90) seconds between two unasked
  lines, which urgent lines break. Two lines due in the same tick are said as one sentence when the second is a
  block start or a suggestion. A held line older than `speak_stale_s` (120) seconds is dropped, except plans,
  cards, check-ins and reviews. Nothing but urgent lines outside the wake and bedtime window. `speaker: box`
  makes the box's own speaker say the lines and the phone stay silent.
- Block-start reminders (`contexts/day/domain/reminders.py`, `block_reminders: true`): the chosen blocks of an
  approved plan, and accepted optional events; said at the block's start, or at start minus the walk when the walk
  is 3 minutes or more, within 5 minutes; nothing when carrying on with the same thing at the same place; short
  blocks back to back are one line; 10 minutes between two reminders; never while a fixed block runs; an accepted
  optional event with a walk gets its leave-now line instead; once per block and start time, also after a restart.
- The suggestion (`contexts/day/domain/optional.py`): `suggest_optional: true`, `suggest_from_min: 90`,
  `suggest_until_min: 30`, `suggest_cooldown_min: 45`; the full list is in section (h).

### 5. Example lines

- A suggestion: "Riley, Sam and Alex are having dinner at '63 South at quarter past five. It's a sixteen-minute
  walk and it fits before your study block. Want me to add it?"
- A suggestion that nearly fits: "Andria is hosting COP4533 review session at John C. Hitt Library at half past
  six, with Maya and Priya. It's a thirteen-minute walk, and dinner would have to move to make room. Want me to
  add it?"
- A yes: "Okay, I've worked Friends' dinner in: first up is Friends' dinner at quarter past five. Leave by five to
  five for that; it's a sixteen-minute walk. Want me to switch to that?" A later no to it: "Okay, Friends' dinner
  is out: ..."
- A block start without a walk: "It's one o'clock: your study block on Project Milestone 1, ninety minutes.
  You're at the desk already."
- A block start with a walk: "Time for lunch at Student Union; it's a seven-minute walk."
- Short blocks as one: "It's two o'clock: three short study blocks from here, through to quarter to four. You're
  at John C. Hitt Library already."
- A block start and a suggestion due together: "Time for lunch at Student Union; it's a seven-minute walk; and
  Riley's dinner is at '63 South at quarter past five if you want it."
- The time to leave for an accepted event: "Time to go. Friends' dinner starts at quarter past five at '63 South,
  a sixteen-minute walk from here; leave in five minutes."
- The push of a suggestion (no names): "Atlas has a suggestion near you: Friends' dinner: 5:15 PM to 6:15 PM at
  '63 South. 16 min walk, leave by 4:59 PM. Fits before your study block at 7:01 PM. Open Atlas to add it."

### 6. Who sees an accepted optional event

As in section (h): the plan and everything read from it (`/api/day/plan`, `/api/day/now`, the tracker, the
narrator, block reminders, the review, the Day Run board) and the agent's leaving checklist. The Circle's
free/busy and Habits' "leave ready" only after a calendar connector wrote the card's event and synced it back.

### 7. What the seeded demo shows

The seeded night (`seed`, `seed --now`, `circle`) is as it was: at its scripted end no optional event is inside
its suggestion window, no plan in it is approved, so it shows no suggestion and no block-start line. With the app
running on it, `GET /api/day/optional` lists the seven Near you events with their fit, and a yes to one works once
there is a plan for today (the walkthrough did this with Board game night). The suggestion, the yes behind its
card and the reminders are shown by the new scene (part 2).

### 8. Existing lines changed in files this feature does not own

- `contexts/agent/application/planner.py`: one attribute in `Planner.__init__` (`self.accepted_optional`), and the
  `nearby:` guard in `Planner.leaving` lets accepted ids through.
- `contexts/gateway/bootstrap.py`: one line that sets `planner.accepted_optional`.
- `contracts/day.py`: `DayOptionalAnswered.until`.
- `contexts/demo/__main__.py`: the import line `from datetime import UTC, datetime` became
  `from datetime import UTC, date, datetime`; the rest (the help text, the `optional` command) is added.
- `scripts/demo_walkthrough.py`, `tests/test_demo_walkthrough.py`, `docs/demo-run-sheet.md`,
  `web/scripts/check-api.mjs`, `contexts/circle/tests/test_health_privacy.py`,
  `contexts/learning/tests/test_export.py`: additions only.
- `web/lib/atlas.tsx`: the provider's return now also renders `<OptionalSuggestion />` (one line became a block),
  plus one import.
- `web/lib/voice.tsx`: the `speechText` import line lists two more names; in `listen` a spoken answer to a
  suggestion is looked for before the `if (active.discard)` branch; the `say(...)` call for a spoken event gained
  an `onDone`; `say()` closes the answer window. `web/lib/api.ts` and `web/lib/speechText.ts`: additions only.
- In the day context itself: `plan_moves` sets `origin`, `DayPlanner.card` and `DayPlanner.replan` gained
  optional arguments (callers unchanged; `card_events` is as it was), and the `_suggest` template's screen line
  lost " with <names>".
- One expectation changed: the display string of a suggestion in `contexts/day/tests/test_optional.py`, because
  the screen line no longer names who is going (section h). No hook, deep link or label was removed.
- Not touched: `web/app/now/`, `web/app/dayrun/`, `web/components/dayrun/`, `contexts/voice`,
  `web/components/operator/`, `web/components/Shell.tsx`, the design tokens, `lab/`, `runtime/`, `contexts/play`,
  the Circle protocol.

### 9. The note for Xander

Section (g) stands, with the settled shapes in section (h). Two things to know beyond it: with no plan for today
`fit` is worked out against an empty plan (it reads "yes") and `plan_id` is null, and a yes then answers 409;
`suggest_held` says in words why Atlas is not bringing an event up. The app-wide card
(`web/components/day/OptionalSuggestion.tsx`) shows only an event Atlas itself suggested, so it does not replace
yes and no on the Calendar's own list.

### 10. Not done, not verified, and found on the way

- Nothing was heard: the spoken lines, the spoken yes and no and the 30-second window were checked as code and
  by phrase list. Nothing was run on the Jetson.
- Four decisions wait for a confirmation from the author of this document: the 200 with `proposed: false` for a
  yes more than a day ahead, the port taken from the first yes on, the `until` field, and the screen line without
  names.
- Found, in the arbiter, not changed here: when a yes comes less than ninety seconds after the suggestion, the
  re-plan's line is held for the gap, and the card's own ask ("I'd like to make these eight blocks today's plan.
  Okay?") is then said two minutes later, also when the card was approved in between. `Arbiter.tick` does not
  treat a held ask as part of its plan's narration (only `offer` does), and an ask is not dropped when its card
  closes.
- Found, in the block reminders, not changed here: the walk is counted from where the cameras last saw the
  person, however long ago. When that lead time puts the reminder of the block after a fixed event inside the
  event, it is lost, because nothing is said while a fixed block runs (on the scene's day: the study block at
  6:20 PM, straight after the dinner).
- The limits listed at the end of section (h) stand.
