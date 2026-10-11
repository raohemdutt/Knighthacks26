# Atlas for inspection (NextEra SPOT challenge, Knight Hacks IX)

Atlas is a private AI box: a Jetson Orin Nano that runs everything on site, keeps an append-only
event log it can replay, answers with citations, and does nothing outside without a person's yes.
This is the same box and the same log pointed at a robot's inspection missions instead of a
student's desk. For a utility that means **inspection data never has to leave the plant**: the
operator's assistant works with the network cable out.

The code is one bounded context, `contexts/spot/`, a dashboard at `/spot/` (More → "Inspection
(SPOT)"), and the sponsor's kit under `fixtures/spot/kit/`. All data is the sponsor's synthetic,
fictional kit.

## NextEra brief check

The track brief: "Help plant operators understand what SPOT is seeing! Build a tool that turns Boston
Dynamics SPOT robot mission data into a clear, readable view of locations, timestamps, readings, and
anomaly flags, both in real time and after a mission. Make the information actionable and help
operators build confidence in SPOT without needing to become technical experts."

The screen at `/spot/` has three tabs: **Your shift** (what to do, the floor plan, a round step by
step), **Shift Run** (plan a round and see how it would go) and **Analysis** (the evidence, as
animated explainers with the tables behind them).

| The brief says | Where it is met | Honest gap |
| --- | --- | --- |
| locations | Your shift: the floor plan with one mark per piece of equipment, on the kit's plan image, both floors. Every sentence names the place ("Pipe Gallery (WP05)"). A round shows SPOT's route and where it is at each step. | The plan is the kit's fictional picture. Pin positions were read off it by eye; the kit's x/y do not match it (see below). Equipment is drawn at its home waypoint, although 61 of 270 actions were recorded somewhere else. |
| timestamps | Every step of a round starts with its time ("10:12 AM"). Rounds are labelled "Round 27 · today 10:00 AM". Equipment panels say "12 minutes ago", "2 days ago". The status line says when the data ends. | "Now" is the newest moment in the data, not the wall clock. The kit's times have no time zone and are shown as written. |
| readings | The equipment panel gives the last good reading in words with its number and unit ("Temperature 75.5 °C, in its usual range"), and how old it is. Each step of a round shows its reading. | The kit has no engineering limits. "Usual range" is our own band (site median ± 3 robust sd) and the panel says so. 50 actions have no reading at all. A reading does not explain the alarm raised with it. |
| anomaly flags | An anomaly is called an "alarm" and defined once on the card. Each one says whether it stands (the check that raised it worked) or is not confirmed (that check failed), and what to do: send a technician, or send SPOT back. | The kit does not say whether an alarm on a failed check is real. We call it not confirmed, never false. 32 of 40 alarms are of that kind. |
| in real time | A round, step by step: press Play and the box replays the recorded events in order; SPOT moves on the plan, each check arrives as it happened, the battery drains, and the label reads "Live · following SPOT". | It is a replay of recorded data. There is no live robot and no robot feed. The kit's event stream stops at event 500; the last 94 events are rebuilt from the tables and marked as rebuilt. |
| after a mission | The same section with the round picker: any of the 27 rounds as ten steps (tick, cross, alarm), one sentence per step, Back, Next and a slider. | One robot, 27 rounds, 270 actions. |
| clear, readable | One status sentence, three cards of at most three items, three levels that differ by shape as well as colour, 44 px targets, phone first. No statistics words on the shift screen; a test checks every sentence for raw enums, row ids and jargon. | The sentences are templates: correct but not graceful. Ask still answers some questions in the analyst's wording ("Exact wording" shows it). |
| actionable | "Do this first" gives one button per item: Send SPOT back or Send a technician. It raises an approval card; after "Yes, do it" a work order file is written on the box. Shift Run turns a planned round into a mission request the same way. The shift handoff is one button. **Take me to it:** ask "where is HX-202?" or "show me the pump room" (typed or spoken) and a red pointer lands on its mark, the plan switches floor if it has to, and the walk there is routed round the walls and the equipment, drawn on the plan, and spoken turn by turn while the red dot walks it ("Turn left, take the entrance steps and go along the west passage"). Start from a picked place, a tap on the plan, or where SPOT is. Say "next" to step through the do-first list without touching the screen. | Nothing reaches a robot or a maintenance system: the result is a JSON file on the box. That is deliberate, and it is all there is. The kit has no walls, doors or walkways, so the floor the walk is routed on was **traced by us, by eye, from the kit's plan picture**: every route is "about", never exact, and is said so on screen and aloud. The picture has no scale, so no distance is ever given. Six of the eleven doors and both pairings of stairs between the floors are our assumption (the picture shows a room but not where it is entered). A line is solid only where SPOT's rounds show it went between those two places; that vouches for the leg, not for the line. |
| build confidence in SPOT | The trust line at the top ("21 of 26 completed rounds still had a failed check inside. We show you the checks, not just the round"), "Looks fine" with how many were checked recently, every "Why?" playing a short explainer, and Shift Run, where a trainee sees why order and battery matter. | Confidence here means knowing when not to trust a result. The data does not let us say SPOT is reliable: 49 of 270 checks failed. Shift Run is a toy model from 27 rounds, a training aid, not a prediction. |
| without needing to become technical experts | Words come from a glossary (round, check, alarm, not confirmed, usual range). Six animated explainers show the analysis one idea at a time; "Show the numbers" opens the intervals and rows for whoever wants them. **Where do I look?** Nobody has to learn the screen: "where do I look for the battery?", "where can I plan the next round?", "is V2 better?" each open the right tab or fold and walk the person there in short sentences, with captions. With the opt-in "Use my eyes to guide me", the pointer starts from where the person is looking and notices when their eyes arrive. | The Analysis tab's tables are still analyst material. Not tested with a real operator. Webcam gaze is only as accurate as its own check measures on that screen (typically a few centimetres on a laptop at arm's length, worse with glasses glare or low light): it tells a region and which marked thing, never a pixel, and it switches itself off when the check is poor. The attention check ("seen" ticks, a prompt before Send SPOT back / Send a technician / Send this plan when the thing that matters was not looked at, and "Where you looked" with a scan path after a Shift Run) means the eyes rested there for 0.4 s, not that it was understood; it lives in memory for the visit only, never blocks, and without a webcam it can only be shown through the labelled simulation `?gazeSim=`. |
| Kit Part Two (Spot User Guide R1.1): how we used it | Vocabulary: a round is one Autowalk mission (p. 39), a check is an action (p. 40), and the section that plays a round back is no longer called "replay", because in the guide "replaying a mission" means SPOT walking it (p. 43). "The manual says…" notes on failed steps and two patterns: stuck at an obstacle (p. 44), lost (p. 44), connection lost and a weak battery (p. 11, p. 44), unevenly lit fiducials (p. 41), dusty floors (p. 11). Shift Run's hard mode is the guide's "skip part of the mission" (p. 43). | We were given pp. 1 to 12 and from p. 23 on (Autowalk is pp. 39 to 44); pp. 13 to 22, with battery life and charging, were not in the files. The pages we read state no battery life, runtime, walking speed or return-to-dock rule, so Shift Run uses only the kit's measured battery drop and minutes per step. The guide says not to use Autowalk on stairs (p. 40); the kit's rounds cross two floors and do not say how. The notes are the vendor's general guidance, never a diagnosis of this fictional plant. |

### What is new on the screen

- **Plain words from data** (`contexts/spot/application/plain.py`, `GET /api/spot/shift`,
  `GET /api/spot/shift/rounds/{id}`, `plain: true` on `POST /api/spot/ask`). Every triage item, finding and
  asset becomes `{headline, sentence, level: act | watch | ok, why_ref, action}` from templates and a
  glossary. No model call. A level is a stated rule: **needs you** when an alarm stands on a good check, a
  serious alarm has not been re-checked, or important equipment has had no good check for 24 hours;
  **keep an eye** for an alarm still waiting for a clean re-check, a serious alarm in the last 3 days, a
  failed last check, or no good check for 24 hours; **looks fine** otherwise.
- **Explainers** (`web/components/spot/Explainers.tsx`): completed hides failures, alarms not confirmed,
  battery, is the new setup better, weak spots on the plan, what the data is missing. SVG and CSS, drawn
  from the API's numbers, still with "reduce motion" on.
- **Shift Run** (`contexts/spot/domain/shiftrun.py`, `application/shiftrun.py`;
  `GET /api/spot/shiftrun/board`, `GET /api/spot/shiftrun/recommended`, `POST /api/spot/shiftrun/simulate`,
  `POST /api/spot/actions/plan`). A check's chance of failing is the site rate (49 of 270) with its odds
  multiplied by a capped ratio for the kind of check, one for the waypoint, and one for "late and low" (the
  larger of the battery-band and position ratios, never both: they are the same actions in the kit).
  Groups under 8 actions count for nothing. Battery drop (about 4.7 points) and time (about 2.7 minutes) per
  step are the kit's averages. The score is an expected value: what each asset is worth (triage score,
  importance, days since a good check, a little less for every stop it waits) times its chance of a good
  check, minus failed checks and checks on a low battery. Stars compare it with Atlas's plan, found by the
  same score. The seeded draw only drives the animation. The weights are judgement and are returned part
  by part.
- **Loading**: the first screen needs `GET /api/spot/shift` and the replay state. Findings, locations and
  the overview load when a "Why?" or the Analysis tab is first opened; the tables load when their fold is opened.
- **Links**: `?asset=HX-202`, `?round=M017&step=8`, `?why=battery` (`&at=end` or `&at=mid`),
  `?numbers=battery`, `?tab=analysis`, `?tab=run&route=last&turn=40&show=result`, `?floor=2`, `?all=1`.

### The look guide ("where do I look for X?")

- **Ask it.** In "Ask SPOT's data", type or say a where-question ("Where is Heat exchanger 2?", "Where do I look for the
  battery?", "Show me what to do first"), tap a "Guide me" chip, or say "next" to step through the do-first list.
  Anything that is not a where-question is answered as before.
- **What happens.** `GET /api/spot/guide?q=` (`contexts/spot/application/guide.py`) resolves the question by code: tags
  and ids first (HX-202, WP05, M017, round 27, floor 2), then names with a little spelling tolerance against about 120
  targets (20 pieces of equipment, 15 places, 27 rounds, 18 findings, the cards, tabs, explainers and controls). No
  model is called. It returns a guide id, short sentences, and the closest alternatives when it is not sure. The page
  switches tab, floor, round or fold as needed, a red pointer travels there around the cards (A* on a coarse grid of the
  screen, `web/lib/guidePath.ts`, the same algorithm as the room twin's router), a ring pulses on the target, and the
  sentences are shown as captions and spoken when "Speak answers" is on.
- **The walk.** For equipment and places, `GET /api/spot/route?from=&to=` routes on the walkable floor traced from the
  plan picture (next section): A* on a grid, 8 neighbours, no cutting a corner, then string pulling; between floors by
  whichever stairs make the walk shortest. The answer is the polyline per floor, one instruction per stretch
  (`contexts/spot/domain/directions.py`), `precision: "about"` with the reason "traced from the picture", and
  `walked: true | false` (did SPOT's rounds go between these two places). The old route over the legs SPOT walked
  (`contexts/spot/domain/route.py`) is still there under `via_walked_legs`, and is the whole answer on a box with no
  plan picture.
- **On screen.** The red dot goes to "You are here" and then walks the line at a walking pace, quicker on a long route.
  Each stretch lasts as long as its sentence when the speaker is on, and the sentence is captioned and said as the dot
  starts on it. Escape or a tap stops it. The line stays: solid where SPOT has walked that leg, dashed where it is from
  the trace only, with a legend. A route over two floors switches the plan to Both. Then the ring lands on the mark.
  "You are here" is the picker, **Tap the plan to set it** (the tap is snapped to walkable floor within 60 px, and
  refused when it is nowhere near any), or **From SPOT** (where SPOT is in the round on show). Under the plan:
  "Paths traced from the kit's floor-plan picture; the kit has no walkway data.", with **Show the traced floor**.
- **Rounds and plans.** "A round, step by step" and Shift Run's plan draw their legs along the same traced lines
  (`GET /api/spot/walkable/legs`, all 105 pairs of waypoints). Drawing only: Shift Run's score does not use them.
- **Eyes (opt-in, Settings).** MediaPipe Face Landmarker, served from `web/public/gaze/` (no CDN; see its README for
  versions, checksums and why the package is pinned to 0.10.35). Nine calibration dots, then four fresh dots that
  measure the miss; that measured miss sets the grid the estimate is rounded to and the snap radius, and a poor result
  turns gaze off with the reason. The camera picture never leaves the browser and nothing but the calibration fit is stored.
- **Attention check.** With eyes on, a fixation of 0.4 s that snaps to exactly one marked thing is noted (in memory
  only). The do-first items, the watch-out card, the trust line and the turn-back control get a "seen" tick, and the
  card says "You've looked at 1 of 3 things that need you". An action whose subject was not looked at gets a prompt
  and the pointer, with "Continue anyway"; nothing is blocked. Shift Run adds "Where you looked" (reasons and a scan
  path drawn from element rectangles) and an opt-in "Training: attention counts" that takes one star off. The handoff
  gains one sentence only if the operator ticks "Include what I reviewed". With eyes off there is one line inviting
  them and nothing else changes. `/spot/?gazeSim=do-first:1,trust` feeds scripted fixations through the same rules
  under a banner that says the gaze is simulated.
- **Links for a demo:** `/spot/?guide=where is HX-202&t=1` (the pointer held at the end of its way), `&t=0.5` (half
  way along the walk), `&here=WP06` | `&here=xy:0.30,0.57` | `&here=spot` (where the walk starts), `&walkable=1` (the
  traced floor), `?guide=explainer:battery`, `?guide=shiftrun:turn-back`, `?guide=round 17`; add `&gazeDebug=1` to see
  the gaze estimate.
- **Checks:** `pytest contexts/spot/tests/test_guide.py contexts/spot/tests/test_route.py contexts/spot/tests/test_walkable.py`;
  `node web/scripts/check-guide.mjs`.

### The walkable floor: how it was traced, how it is checked, and its limits

- **Traced.** `fixtures/spot/floor_walkable.json` was drawn by eye from `04_ASSETS/fictional_floor_plan.png`, read as
  six crops at 3.6x with a 0.02 grid laid over them, to about 6 px. Per floor: 22 and 8 rooms, corridors and steps
  (rectangles where one can stand), 6 and 5 doors, 30 and 17 pieces of equipment (tanks, turbines, pump and compressor
  skids, cabinets, crates, pipe racks, the stair footprints), and two stairs. Coordinates are normalised on the whole
  picture, the same frame as `waypoint_pixels.json`. Single thin pipes on the floor are not traced.
- **Rasterised.** `contexts/spot/domain/walkable.py` makes a grid of 4.6 px cells per floor (144 x 111 for floor 1,
  66 x 113 for floor 2). A cell is walkable when its centre is inside a room or door rectangle and at least 3 px from
  every piece of equipment. The A* and the string pulling are the room twin's (`contexts/play/room/grid.py`), used as
  pure functions; the straight-line test is stricter here, so a pulled line cannot clip the corner of a blocked cell.
- **Checked against the kit** (`contexts/spot/tests/test_walkable.py`; these fail if the trace is wrong):
  every one of the 15 pins stands on walkable floor or snaps to it within the stated 30 px (12 pins move less than a
  cell; WP06, WP08 and WP09 sit on the machine their area is named after and stand 20 to 25 px beside it); all 88
  waypoint-to-waypoint legs SPOT walked are reachable, on one floor without stairs and between floors through a
  traced stair; all 105 pairs of waypoints are joined; and every route, sampled each pixel, stays in walkable cells,
  inside a room or door rectangle, and out of every equipment rectangle. All 20 pieces of equipment are reached from
  the entrance.
- **Limits, plainly.** The picture is an illustration, not a survey. Five doors are drawn (seen: true); six are where a
  room had to be entered and the picture shows no door (seen: false, dashed in the overlay). Which stairs on floor 1
  meet which on floor 2 is our pairing. A waypoint's pin marks an area; "by the west pipe runs" is from where we
  traced that equipment, and the kit does not say which machine in a room is which asset. The equipment marks are
  drawn around a pin's head, so a route ends at the pin's tip and the ring lands on the mark. No scale was found
  anywhere in the kit (`locations.csv` x/y are "for visualization" and miss the picture by hundreds of pixels), so the
  words are relative ("about a third of the way along", "on your left") and never metres. The Spot user guide says not
  to use Autowalk on stairs (p. 40); the kit's rounds change floor anyway, and the route says "take the stairs" as a
  person would.

## Run it

```
python -m contexts.gateway            # then open http://<box>:8000/spot/
ATLAS_SPOT_KIT=/path/to/kit python -m contexts.gateway   # another copy of the kit (numbered folders, or flat)
ATLAS_PROFILE=inspection python -m contexts.gateway      # the shared screens say plant / inspection / mission / work order
```

- No kit folder: every `/api/spot/*` GET still answers 200 with `configured: false` and how to add one.
- Replay: press Play on the page, or `POST /api/spot/replay/start | pause | seek {index|mission_id|fraction} | speed {speed}`.
  Speed is data seconds per second (30, 60, 240, 1200); idle time between missions is skipped.
- Ask: `POST /api/spot/ask {"text": "what should I check first this shift?"}`.
- The scorer: `python -m contexts.spot.train` (`--check` compares with the committed weights).
- The fine-tune dataset: `python -m contexts.spot.dataset --out <dir>`.

## What the files turned out to hold (the loader reports all of it at `GET /api/spot/validation`)

- `replay.ndjson` stops at event 500, in the middle of M023; M024 to M027 never appear. The player
  rebuilds the missing 94 events from the tables and marks them as rebuilt.
- The stream has mission and action events only. Readings and anomalies are attached when their
  action ends (they share its timestamp).
- 61 of 270 actions name an asset at a waypoint other than the asset's `assets.csv` location.
- 50 actions have no reading row (7 of them failed).
- Reading `quality` repeats the action status exactly (220 of 220); `operator_review_required`
  repeats severity exactly (40 of 40); `scenario` and `configuration_version` are the same split;
  `operator_mode` is constant. These fields add nothing.
- `locations.csv` x/y do not describe the floor-plan image: the best affine fit to the image's pins
  misses by 314 px RMS on a 1095 px wide picture. Markers on the plan use pin positions read off the
  image (`fixtures/spot/waypoint_pixels.json`); the "CSV x/y" view draws the coordinates as given.
- Our counts and the five headline metrics agree with the kit's `validation_results.json` (12 of 12).

## Findings (every one is a pure function that returns number, n, interval, sentence, caveat, evidence ids)

| Question | Finding | Caveat |
| --- | --- | --- |
| Completed with failures inside? | 21 of 26 COMPLETED missions (80.8%) contain a failed action; 46 of 49 failed actions sit inside a COMPLETED mission. | M017 is FAILED with 3 failed actions; 9 COMPLETED missions had as many. The tables do not say what makes a mission FAILED. |
| Least reliable inspection type? | Acoustic: 13 of 43 failed (30.2%, 95% interval 19 to 45%) against 15.9% (36/227) for the rest, +14.4 points (+1.6 to +29.8, p=0.025). Visual is best, 6 of 68. | Intervals are not corrected for the number of comparisons. |
| Concentrated at locations? | WP09: 8 of 22 (36.4%, 20 to 57%) against 16.5% elsewhere, +19.8 points (+2.4 to +40.9, p=0.038). Floor 1 21.4% (41/192) against floor 2 10.3% (8/78). | All eight WP09 failures are V1 (8/17); 0 of 5 under V2 is too few to call it fixed. |
| Navigation at WP05? | 7 of 25 (28.0%) against 6 of 47 (12.8%) elsewhere: +15.2 points (-3.3 to +36.0, p=0.12). **Suggestive, not established.** | WP05 is the first stop of 25 of 27 missions, so "at WP05" and "first action" cannot be told apart. |
| Battery? | Below 40%: 25 of 71 failed (35.2%); at or above: 24 of 199 (12.1%); +23.2 points (+11.8 to +35.4, p<0.001). Under 20%: 8 of 11. | Low battery and late-in-mission are the same actions, and all of them are V1. Association, not cause. |
| V1 against V2? | Action success 91.1% (82/90) against 77.2% (139/180): +13.9 points (+4.4 to +21.8, p=0.005). Mission completion 9/9 against 17/18: no clear change. Thermal inspection, mission duration and battery (start, end, used) changed. Acoustic, gauge, navigation, visual, anomalies per mission, HIGH/CRITICAL anomalies per mission and action duration: no clear change. | **V2 never ran below 52% battery.** In the battery range both ran in, the gap is 96.2% (51/53) against 87.3% (62/71): +8.9 points (-1.8 to +19.0, p=0.11), no clear change. The data cannot separate "the configuration is better" from "V2 missions end with more battery". We do not claim a cause. |
| Improving or degrading? | Failed actions per mission fell from 2.3 (V1) to 0.9 (V2), a step at the change; inside V2 the rate is flat over 9 missions. | Nine missions is a short run. |
| Anomalies hidden in good missions? | 38 of 40 anomalies are inside COMPLETED missions, 14 of them HIGH or CRITICAL. 32 of 40 were raised by an action that itself failed (unverified); 8 by a successful one (confirmed). | The kit does not say whether an anomaly on a failed action is real. We call it unverified, not false. |
| Readings drifting? | 83 asset-and-measurement series; 8 have the 5 good readings a trend needs; none heads for the edge of the fleet band. | The kit has no engineering thresholds: the band is ours (fleet median ± 3 robust sd). Anomaly rows are not explained by values (a CRITICAL abnormal temperature sits on a 41.4 °C reading, well inside the fleet's usual range). |
| Time of day, durations? | No clear effect (p=0.37; p=0.58). There is no retry field. | |

**Triage** (`GET /api/spot/triage`) ranks what needs attention with a score an operator can read:
severity of the worst anomaly × recency + asset criticality + repeats + 15 if the latest inspection
attempt failed + staleness − "seen fine since". Top of the queue: HX-202 (CRITICAL asset, HIGH
vibration raised in the latest mission by an action that itself failed, no good inspection for 88 h)
→ re-inspect before sending anyone; HX-204 (CRITICAL anomaly from a successful inspection) → send a
technician. The uninspected-risk list is HX-202 (88 h), GAUGE-209 (48 h), HX-213 (32 h).

## The small model (honest version)

`python -m contexts.spot.train` fits a logistic regression (pure Python, deterministic) that scores
each action for "carries an anomaly row", from causal features (own status, reading against the
fleet and the asset's own history, jump from the previous reading, battery, position, duration,
type, waypoint and asset history, criticality). Split by mission in time (train M001 to M019, test
M020 to M027: 80 actions, 12 anomalies):

| | Precision | Recall | PR-AUC |
| --- | --- | --- | --- |
| Learned model | 7/8 | 7/12 | 0.65 |
| Rule: the action failed | 7/8 | 7/12 | 0.57 |
| Rule: reading outside the band | 3/3 | 3/12 | 0.36 |

Rolling cuts pooled (180 actions, 25 anomalies): model 17/30, 17/25, 0.57; status rule 19/30, 19/25,
0.51; band rule 5/9, 5/25, 0.22. **It beats the reading-threshold rule; it does not clearly beat
"the action failed"**, and among successful actions it is at chance (PR-AUC 0.10, base rate 0.07).
So the app uses it only to order ties in the queue and to explain a flag with its top features
(`GET /api/spot/model`); it never raises an alert.

## The fine-tune dataset

`python -m contexts.spot.dataset --out <dir>` writes `spot-qa-train/` (574 examples, anchors M006 to
M020) and `spot-qa-heldout/` (240, anchors M022 to M027) in the box's export layout
(`manifest.json`, `examples.jsonl` of `DatasetExample`, `expected.jsonl`, `images/`), readable by
`lab/dataset.py --text-only`. Each example is a question asked "as of" a mission, the evidence a
tool would return, and a label `{"answer", "citations"}` written by code from the tables. Nothing is
trained by this repo; the run is the lab's.

## Assumptions (the page's "Assumptions and method" drawer lists the same, from `GET /api/spot/assumptions`)

1. "Now" is the newest timestamp in the data, not the wall clock.
2. An action is failed when its status is not SUCCESS; mission status is read separately.
3. Scenario and configuration version are one comparison.
4. No cause is claimed for the V1 to V2 difference.
5. Rates carry 95% Wilson intervals; differences use Newcombe's interval with Fisher's exact test or a z-test; under 8 rows we say "not enough data"; under 5 good readings there is no trend.
6. Intervals are not corrected for multiple comparisons: single-waypoint findings are leads.
7. A waypoint's numbers use the action's location; an asset is drawn at its `assets.csv` location.
8. An asset counts as inspected when a non-navigation action on it succeeded; a failed action or QUESTIONABLE reading means it was effectively not inspected.
9. An anomaly raised by a failed action is "unverified"; by a successful one, "confirmed".
10. No engineering limits exist in the kit; the band is the fleet median ± 3 robust sd.
11. The triage weights are judgement, shown part by part.
12. `operator_review_required` is not used on its own.
13. The replay is played in event-time order; readings and anomalies attach when their action ends; missing missions are rebuilt from the tables; idle time is skipped.
14. The plan image is a backdrop with markers on its pins; the schematic draws CSV x right, y down.
15. One robot; timestamps have no zone and are used as written; everything is synthetic.

## How the judging themes are met

- **Operational usefulness:** the first row of the page is three things to do, each with why; one tap raises a re-inspection or work-order card.
- **Insight:** mission status hides 46 of 49 failures; most anomalies come from failed actions; V2's gain cannot be separated from battery; WP05 is suggestive only.
- **Usability:** one page, three columns on a display, one on a phone; every id opens its evidence.
- **Technical execution:** typed loaders with validation, pure analysis functions, an event-sourced replay that rebuilds after a restart, 82 tests.
- **Creativity:** the replay is a live control-room picture; the assistant works offline; a model was trained and then told it had not earned a vote.
- **Explainability:** every number has its n, interval, caveat and row ids; the assumptions are generated next to the analysis.

## What the hardware does here, honestly

This dataset is 270 actions in six small tables. The analytics, the replay and the scorer are plain
Python and need no GPU; they would run on anything. The Jetson's part is being the box that sits in
the plant with no uplink: it serves the page, keeps the log, and, when its on-device language model
is running, rephrases the grounded answers (every number and id is checked against the draft and the
draft is used on any mismatch). The lab's AMD Radeon GPUs are where a fine-tune on the dataset above
would run. We have not run that fine-tune, and the rephrasing path has only been exercised with a
stand-in model in tests, not on the Jetson.

## 75-second demo

On the operator screen the same beats are: (1) the status line and "Do this first"; (2) Play under "A round,
step by step"; (3) tap HX-202 on the plan, then "Why?"; (4) Analysis, the explainer "Is the new setup really
better?", then "Show the numbers"; (5) Ask; (6) "Send SPOT back", then "Yes, do it". The steps below are the
analyst wording of each, which is what the Analysis tab's tables still show.

1. (0:00) `/spot/` on the big display. "Three things to do, in order. Mission status says 96%; 81% of those missions hide a failed action."
2. (0:10) Press **Play** at 60×. The robot walks M001; two actions fail; the mission ends with the amber badge **completed, 2 actions failed**; alerts arrive as they happen.
3. (0:25) Triage #1, HX-202: read the reason and the score bar. "Critical asset, anomaly raised by a reading that itself failed, no good inspection for 88 hours: re-inspect, do not dispatch yet."
4. (0:40) V1 against V2: "+13.9 points, interval excludes zero. But V2 never ran below 52% battery; in the range both ran in, the interval includes zero. We do not claim a cause." Tap the line for the rows.
5. (0:55) Unplug the network. Ask **"What should I check first this shift?"**: a cited answer; tap an id.
6. (1:05) Tap **Re-inspect** on HX-202, **Approve** the card: "A work order was written on this box. Nothing left the building."
