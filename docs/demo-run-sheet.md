# Atlas demo run sheet

One sheet for every use case: what it shows, what must be true first, the exact steps, what you
should see, what to do when it goes wrong, and how far it has actually been verified.

The executable half is `scripts/demo_walkthrough.py`: it calls the same endpoints the screens call,
in the same order as this sheet, and prints `use case | step | PASS/FAIL/SKIP | seconds | note`.

```
python scripts/demo_walkthrough.py --gateway http://127.0.0.1:8000                     # everything a script may do
python scripts/demo_walkthrough.py --gateway http://<box>:8000 --read-only --ask       # someone else's box: read, and ask
python scripts/demo_walkthrough.py --gateway http://127.0.0.1:8000 --scene --approve-local --json out.json
python scripts/demo_walkthrough.py --gateway http://<box>:8000 --only profiles --expect-profile inspection
```

It never approves a card with an effect outside the box (messages, Circle shares, exports, a real
calendar). `--approve-local` approves only "keep this plan" on a box with no calendar connector and an
inspection work order (a file on the box); `tests/test_demo_walkthrough.py` pins that rule. The T-15
checklist stays `python -m contexts.demo preflight --gateway http://<box>:8000 --reset`; shapes of
every answer stay `node web/scripts/check-api.mjs`.

**Words used below.** *Simulated* = driven by a fake we ship (the seeded night, the fake camera scene,
the Circle of three on one box, the lab's labelled sample run, the sponsor's synthetic inspection
kit, the fake model). *Real* = the Jetson's own cameras, model, firewall. A simulated result is never
written in a real-hardware column.

---

## Results: run C, Sat 2026-10-10, 10:34 PM EDT (the newest; it adds the look guide, the bin rule and the check-in)

Seeded laptop: `python -m contexts.demo seed --data-dir <tmp> --now` (the demo night ending now, **without** the Circle of
three, so the Circle steps that need members skip), web built from branch `ctx/experience/look-guide-2`, gateway on
port 8947 with the fake model (`fake-0`), no cameras, no Neon, no ntfy, no token. Run with `--scene --approve-local`.
**100 PASS, 1 FAIL, 17 SKIP (118 steps).**

| Use case | PASS | FAIL | SKIP |
| --- | --- | --- | --- |
| 0 Fake camera scene | 1 | 0 | 0 |
| 1 Ask what the cameras saw | 6 | 1 | 1 |
| 2 Conversation | 2 | 0 | 3 |
| 3 The day (now with `GET /api/day/checkin`) | 15 | 0 | 0 |
| 4 Study coach | 5 | 0 | 1 |
| 5 Day Run | 9 | 0 | 0 |
| 6 Circle | 2 | 0 | 1 |
| 7 Sort It | 10 | 0 | 1 |
| 8 Waste Watch / Impact (now with the bin rule the look guide speaks) | 8 | 0 | 0 |
| 9 Identity | 6 | 0 | 2 |
| 10 Privacy | 3 | 0 | 2 |
| 11 Teach / learning | 8 | 0 | 0 |
| 12 Voice | 1 | 0 | 3 |
| 13 Inspection (now with `/api/spot/guide` twice and `/api/spot/route`) | 15 | 0 | 0 |
| 14 Profiles | 1 | 0 | 1 |
| 15 Operator, zones, Live | 8 | 0 | 2 |

- **The one FAIL** is the same one as in run A: "What happened while I was out?" answers "I didn't see that." on the
  seeded night with the fake model. Product code was not changed for it.
- **New steps, all PASS:** "where is HX-202?" resolves to `asset:HX-202`; "where is the coffee machine" is offered
  three alternatives and not guessed; the walk to HX-202 is "Main Entrance to Pipe Gallery: 1 leg, all walked by SPOT
  before"; cup → trash with its rule sentence; the check-in endpoint answers (none raised: "held: outside waking hours").
- **Not walked, on purpose:** `GET /api/learning/lab/bench` (TODO: it merged as #295 after this branch was cut; add a
  step once this branch is rebased), the "Near you" events (`/memory/items` answered 29 items, none with a `nearby:` id
  on this seed), profile suggestions and the calibration `source` field (neither was found on this build's API; TODO).
- The look guide's browser side (pointer, captions, gaze, attention check) is not something this script can see. It was
  checked in headless Chromium with screenshots; see section 16 for what that covered and what still needs a person.

## Results (Sat 2026-10-10, evening)

Both runs were made by Hemdutt's agent with `scripts/demo_walkthrough.py`. Run A's JSON
(`local.json`) and the screenshots (`final/*.png`, `after/*.png`) are in the session scratch folder
(`runsheet-k7q3x9/`), not in the repo. Run B's numbers are from its console output; the
`jetson.json` kept there is the second, cut-short run described below.

### Run A: seeded laptop, 7:57 PM EDT: 102 PASS, 2 FAIL, 17 SKIP (121 steps)

`python -m contexts.demo circle --data-dir <tmp>` (the demo night ending now, plus the Circle of
three), web built from `main` at `86f7b8d`, gateway on port 8953 with the fake model (`fake-0`), no
cameras, no Neon, no ntfy, no token. Run with `--scene --approve-local`.

| Use case | PASS | FAIL | SKIP | What the SKIPs and FAILs are |
| --- | --- | --- | --- | --- |
| 0 Fake camera scene | 1 | 0 | 0 | 135 events pushed in real time, 62 s |
| 1 Ask what the cameras saw | 6 | 1 | 1 | FAIL: "What happened while I was out?" answers "I didn't see that." (defect 1). SKIP: after the scene the keys answer cites no frame, so there is none to open |
| 2 Conversation | 2 | 0 | 3 | personal question, maths and the web question need the real model |
| 3 The day | 11 | 1 | 1 | FAIL: the coach loses its blocks after an approved re-plan (defect 2). SKIP: no closed day, so no evening review (PASS on the scripted-time seed, see use case 3) |
| 4 Study coach | 5 | 0 | 1 | the teach-task proposal needs 5 answers on one kind of break; the night has 2 breaks |
| 5 Day Run | 9 | 0 | 0 | |
| 6 Circle | 12 | 0 | 0 | |
| 7 Sort It | 10 | 0 | 1 | a face-started round needs a person |
| 8 Waste Watch / Impact | 7 | 0 | 0 | |
| 9 Identity | 6 | 0 | 2 | enrol and forget need a person; the recognition model is not installed here |
| 10 Privacy | 3 | 0 | 2 | looking at a blurred frame and the wipe need a person |
| 11 Teach / learning | 8 | 0 | 0 | the Learning lab shows its labelled sample run |
| 12 Voice | 1 | 0 | 3 | microphone and speaker steps need a person |
| 13 Inspection (SPOT) | 12 | 0 | 0 | |
| 14 Profiles | 1 | 0 | 1 | one box, one profile (three more gateways were run, see use case 14) |
| 15 Operator, zones, Live | 8 | 0 | 2 | no cameras on a laptop; drawing a zone needs a person |

Every step took under 0.15 s except the scene push (62 s, by design). These times say nothing about
the Jetson: the model here is a fake.

### Run B: the real Jetson, 7:46 PM EDT: 77 PASS, 1 FAIL, 24 SKIP (102 steps)

`--gateway http://100.124.34.36:8000 --proxy socks5h://127.0.0.1:1055 --read-only --ask`. Nothing was
approved, pushed, changed, enrolled, wiped or restarted; the only writes were questions and two day
briefs, which is normal use. Model `atlas-base` (the on-device Qwen behind llama-server).

| Use case | PASS | FAIL | SKIP | Notes (measured seconds) |
| --- | --- | --- | --- | --- |
| 1 Ask what the cameras saw | 7 | 1 | 0 | keys 3.1 s with a frame that opens (image/jpeg); package 2.4 s; "while I was out" 21.7 s, 34 citations; FAIL: "overnight" 30.7 s then "I didn't see that." (defect 3) |
| 2 Conversation | 6 | 0 | 0 | "How's your day?" 4.2 s; joke 4.5 s; favourite colour 6.2 s; 17 × 23 = 391 in 2.4 s; web question 27.8 s, 7 links cited |
| 3 The day | 8 | 0 | 2 | "Brief me" 2.7 s (model engine); plan Win 3 stars; no evening review yet; re-plan not run (read-only) |
| 4 Study coach | 3 | 0 | 3 | the box has no focus block today, so no note and no break to ask about |
| 5 Day Run | 9 | 0 | 0 | bad plan 0.35 s, good plan 0.53 s, hard mode 0.53 s, campus map 1.25 s |
| 6 Circle | 2 | 0 | 1 | the Circle is not configured on the box (no members) |
| 7 Sort It | 8 | 0 | 3 | room twin served; 0 rounds played; nobody in the room; no scene pushed |
| 8 Waste Watch / Impact | 5 | 0 | 2 | 1 of 3 recyclables recycled; no milestone yet |
| 9 Identity | 6 | 0 | 2 | **the face recognition model is not installed on the box**; 0 enrolled |
| 10 Privacy | 3 | 0 | 2 | blur on; **firewall enforced: 13 allowed destinations, 13 blocked** |
| 11 Teach / learning | 2 | 0 | 2 | 1 task, 2 deployments; no Learning lab in this build |
| 12 Voice | 1 | 0 | 3 | the voice service is running, state idle |
| 13 Inspection (SPOT) | 8 | 0 | 1 | kit loaded, 12 of 12 numbers agree; ask 20.7 s |
| 14 Profiles | 1 | 0 | 1 | student |
| 15 Operator, zones, Live | 8 | 0 | 2 | **only `cam-desk` reports, "degraded" at 10 fps; `cam-door` is not in `/capture/status`**; pre-flight 11/12 |

**What the box is running.** No endpoint names a commit. By which endpoints exist: the desktop
sidebar and splash are there (#226) and the face-profile fields are there (#205), but
`/api/circle/brain` (#229), `/runtime/stats` (#180) and `/lab/` (#242) answer 404, and the inspection
answer took 20.7 s, so "SPOT answers at once" (#227) is not there. So: `main` at about `07d0f3d`
(#226), before `004ea55` (#229). Polish round 2, the grounded web answers (#234, #241) and the
Learning lab are **not** on the box.

**A second run at 7:59 PM was cut short** (13 PASS, then every request failed): the box stopped
answering at about 8:01 PM for about a minute and came back with its camera counter reset
(`detected` 5,918 → 196), i.e. the gateway had restarted. Its own small talk also gave its uptime as
11 minutes at 7:47 PM and 5 minutes at 8:00 PM, so it restarted at least three times between about
7:36 and 8:01 PM. We did not restart it. The 8:01 PM stop came right after two long model questions
(23.9 s and 37.0 s) with 727 MB of memory free after the restart; whether the questions caused it is
not known. See defect 4. In that second run the same two questions answered differently ("overnight"
gave a cited answer that still began "I didn't see that."; "while I was out" gave the bare line after
37 s), so the model's long answers are not repeatable.

### Defects found (product code was not changed)

1. **"What happened while I was out?" gets "I didn't see that." on the fake model.** `NIGHT` in
   `contexts/agent/application/qa.py:211` reads `while (i|we) (was |were )?(asleep|sleeping|slept|were out)`,
   so "while I was out" never matches the overnight path (only "while I was were out" would). It falls
   to the tool loop, which the fake cannot do. On the Jetson the tool loop answered it once (21.7 s, 34
   citations) and failed once (37 s). Use "What happened at our table overnight?" on a laptop.
2. **Approving a re-plan empties the Coach.** Before: 1 focus block, 2 breaks, "Was that right?"
   offered. After approving "Re-plan from here": `/api/day/coach` has 0 sessions and the Coach screen
   says "No breaks to ask about" (the morning block is no longer in the plan). Show the coach
   **before** any re-plan.
3. **On the Jetson, "What happened at our table overnight?" took 30.7 s and said "I didn't see
   that."** The box's memory starts at 7:42 AM today, so only one 2-minute episode is inside the 8 PM
   to 8 AM window. For the demo the box needs a real night in memory, or the seeded night.
4. **The Jetson's gateway restarted at least three times in 25 minutes** (see above). Someone on the
   box should read `journalctl -u atlas` around 7:36, 7:55 and 8:01 PM before the demo.
5. **Inspection is clipped by the sidebar at 1440 px wide.** `web/components/spot/spot.module.css:22`
   sizes the page as `min(1480px, 100vw - 40px)` and ignores the 288 px sidebar, so the first card
   reads "ect HX-202 (CRITICAL asset)". It is correct at 1920 px. Show it full screen on a 1080p
   display, or zoom out once.
6. **The fake camera scene overwrites where the keys are.** After `--scene`, "Where did I leave my
   keys?" loses "left of the laptop, behind the coffee mug" and its frame. Ask the keys question
   before pushing the scene, and never push it into the demo box's real memory.
7. **A freshly seeded night starts with one pending card** ("make these nine blocks today's plan"),
   so `preflight` says "no stale approval cards: 1 pending" until it expires five minutes after the
   seed. Run `preflight --reset` after seeding, or use it as the approval beat.
8. **The suggested questions do not follow the profile.** Under `shift-worker` and `inspection` the
   Ask screen still offers "What is due this week?" and "I'm stressed about my exam".
9. **The Operator screen logs a React hydration error (#418)** on load (a text mismatch, most likely
   the clock). Nothing visible breaks.
10. **Small ones.** A log that ends after the wall clock greets "Good evening" at 9:30 AM (the
    scripted-time seed). The room twin reports "calibrated" from the placeholder points committed in
    `config/rooms/dorm.yaml`; that is not a calibration of the real table, and there is no screen for
    calibrating (it is one `PUT`, see the table-day checklist).

11. **One gateway test fails between 8 PM and midnight Eastern.**
    `contexts/gateway/tests/test_app.py::test_ask_answers_from_memory_and_streams_events` expects one
    event-log file and finds two (`2026-10-10.jsonl` and the next day's) once the UTC date has moved
    on. It is a date-boundary assumption in the test or the log, not a demo risk; the rest of the
    suite was green at 8:09 PM (5,366 passed).

---

## Before any run

| Where | Command |
| --- | --- |
| Laptop, seeded | `python -m contexts.demo circle --data-dir /tmp/atlas-demo`, then the command it prints, with `ATLAS_WEB_DIR=web/out` after `cd web && npm run build` |
| Laptop, with an evening review | `python -m contexts.demo seed --data-dir /tmp/atlas-night` (the scripted time, Sun 9:30 AM), then `ATLAS_DATA_DIR=/tmp/atlas-night ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 python -m contexts.gateway` |
| The box, T-15 | `python -m contexts.demo preflight --gateway http://<jetson>:8000 --reset` |
| Laptop, an optional event suggested and said yes to | `python -m contexts.demo optional --data-dir /tmp/atlas-optional` (an invented day; it prints each line Atlas says), then the command it prints (use case 3a) |
| Live features without a camera | `python -m contexts.demo fake perception --gateway http://<box>:8000 --realtime` (a throwaway box only) |

The phone app's tab bar is **Ask, Now, Today, Day Run, More**. **More** lists: Everyday (Now, Today,
Day Run), The room (Live, Room map, People), Plan (Approvals, Impact, Inspection), Other (Timeline,
Sort It, Circle, Teach, Learning lab, Zones, Privacy). The Operator screen has no link: open
`/operator/`.

---

## 1. Ask what the cameras saw

**Shows:** Atlas answers a question about the room in plain English and shows the frames it used.

**Setup:** memory with a night in it (the seeded night, or the box left running overnight), keys on
the desk in view of the desk camera, a package left in the door zone at some point.

**Steps**
1. Open `http://<box>:8000/` (the **Ask** tab, heading "Ask Atlas").
2. Tap **Where did I leave my keys?** under "Try one" (or type it in "Ask anything" and tap **Ask**).
3. Tap **What happened at our table overnight?**
4. Type **Did a package arrive?** and tap **Ask**.
5. On the box only: type **What happened while I was out?**

**Expected**
- Keys: one sentence with the zone and position, e.g. "You left your keys in the desk zone on
  cam-desk, left of the laptop." Under it "EVIDENCE · 1 SOURCE", a blurred frame, "Desk camera · Today
  7:31 PM", and "Answered by atlas-base at …". Seeded with the fake model the text is
  "(fake) Evidence: - keys last seen Sat 7:31 PM in the desk zone on cam-desk, left of the laptop,
  behind the coffee mug" and the tile says "Frame not stored on this box".
- Overnight: hour-by-hour lines ("Fri 10:00 PM hour: Two people carry backpacks…") and "Evidence · 22
  sources" on the seeded night.
- Package: "Yes, a package was seen in the door zone at …" with sources.
- Nothing seen: exactly "I didn't see that."

**What can go wrong**
- "I didn't see that." for "while I was out": defect 1 (laptop) or a slow tool loop (box). Ask the
  overnight question instead.
- A long wait: on the box the overnight and "while I was out" questions took 17 to 37 s. Ask keys and
  package live (2 to 6 s); show a long one only if it was asked once before in the session.
- No picture: the seeded log stores no frames. On stage use the box.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/agent/tests/test_qa.py`, `test_golden_eval.py`; 42/42 phrasings on the Jetson eval (README) | ⚠️ run A, 7:57 PM: keys, package, overnight PASS with citations; "while I was out" FAIL (defect 1). Simulated with: the seeded night + fake model. `final/01-ask-keys-500.png`, `01-ask-overnight-500.png` | ⚠️ run B, 7:46 PM: keys 3.1 s with a real frame, package 2.4 s, "while I was out" 21.7 s; overnight FAIL after 30.7 s (defect 3) | ❌ nobody has asked it at the table with the keys actually on the desk today |

---

## 2. Conversation with Atlas

**Shows:** Atlas can just talk, do simple sums, and look something up when the cameras cannot know.

**Setup:** the on-device model running (the fake only knows small talk); internet for the web question.

**Steps** (Ask tab)
1. Tap **How's your day?**
2. Tap **Tell me a joke**.
3. Type **What's your favorite color?**
4. Type **What is 17 times 23?**
5. Type **Who won the FIFA World Cup in 2022?**
6. Optional: tap **I'm stressed about my exam**.

**Expected**
- A short answer in Atlas's own voice that uses what it has seen, e.g. "My day is running smoothly…
  I've logged 35 moments so far." Under it "Answered by atlas-base at … · no sources attached".
- "17 times 23 is 391."
- The web answer names its sources: "Argentina won the 2022 FIFA World Cup, defeating France…" with
  link citations (7 on the box).
- Fake model: a joke and "Pretty good, my cameras logged 47 moments today…"; anything else gets "I'm
  Atlas, and on this laptop I'm running without my model, so small talk is all I've got."

**What can go wrong**
- The web question is slow (23.9 and 27.8 s measured). Keep talking, or skip it; the fallback line is
  the maths question (2.2 to 2.4 s).
- No internet (`make offline`): the web question gets the model's own answer or "I didn't see that."
  Say so; that is the offline story.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/agent/tests/test_talk.py` | ⚠️ run A: "How's your day?" and the joke PASS; personal, maths and web SKIP (fake model). `final/02-talk-day-500.png` | ✅ run B: all six PASS: 4.2 s, 4.5 s, 6.2 s, 2.4 s (391), web 27.8 s with 7 links | ⚠️ the same run is the real model on the real box; not tried by voice |

---

## 3. The day

**Shows:** Atlas plans the day from the calendar and what the cameras measured, says it out loud,
tells you when to leave with the walking route, re-plans when you fall behind (with one approval
card), and reviews the day in the evening.

**Setup:** the Day loop on (it needs the place packs), a calendar or the seeded night. For the
evening review: a closed day in the log (the scripted-time seed, or the box after its day closes).

**Steps**
1. Open **Now** (`/now/`). Read the top card: "Good evening · Today, as of 7:55 PM".
2. Tap **Brief me**.
3. Scroll "YOUR DAY" to the next block with "Leave by 8:41 PM · 10 min walk" and tap **Route**.
4. If the card is red ("Late: the mentor check-in started at 7:39 PM at Engineering II; last seen at
   the desk. It is a 10 min walk."), tap **Re-plan from here**. (Under **More**: **Plan again** and
   **Change it in Day Run**.)
5. Tap **Review the new plan** → **Approvals**. Read the card "Change something in a connected app":
   "I'd like to make these nine blocks today's plan. Okay?" → **Approve** (or **Deny**).
6. Evening: on Now scroll to **Yesterday** → "Yesterday's review" → **The whole review**.

**Expected**
- The brief: "You slept 1.5 h, so pace yourself. First up is the milestone 1 writeup at 8:44 AM
  (Engineering II). 9 min walk: leave by 8:35 AM. Later: mentor check-in 7:39 PM, submissions due
  9:19 PM…", with "Atlas, from the plan and the cameras" beside the button. On the box the words are
  the model's ("Free until dinner at 8:01 PM.", 2.7 s).
- Four tiles: meals seen, min at the desk, recycled, slept (desk camera).
- The route sheet "To Engineering II": "Leave by 7:29 PM: 10 minutes on foot to Engineering II,
  680 m", a campus map with the walk drawn and "© OpenStreetMap contributors", and the line "The part
  inside the building is counted in the minutes, not drawn."
- The re-plan: "The new plan, block by block (9 to add)", "Drift triggered it. Win, 2 stars…". The
  card ends "No calendar connector on this box: approving keeps this as today's plan, nothing is
  written" on a box without a calendar. After **Approve**, Approvals says "Nothing waiting for you".
- The review: "Yesterday's review 50% followed · no stars · 57 points", and in the whole review
  "…Waste Watch: all 2 things you threw away went in the right bin."

**What can go wrong**
- No red card: nothing is late. Tap **More** → **Re-plan from here** anyway; it always works when
  there is a plan.
- The card expired: every approval card lasts **5 minutes** (`APPROVAL_TTL`). Tap **Re-plan from
  here** again, and do not raise a card long before the beat that approves it.
- With a calendar connector the same card writes real events. Then it is the owner's tap, not a
  script's.
- The model is slow or off: the template's words appear instead (same facts). That is the fallback.
- No evening review: the day has not closed. Show it on the scripted-time seed.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/day/tests/test_day.py`, `test_narration.py`, `test_choose.py`; watch sleep `contexts/habits/tests/test_watch_sleep.py`, `contexts/connectors/tests/test_health.py` (synthetic `fixtures/health/freereps.yaml`) | ✅ run A: plan, narration, "Brief me", leave-by route (41 points, 680 m), re-plan, card approved by the guard, plan committed. Evening review + Waste Watch clause PASS on the scripted-time seed (port 8955). Simulated with: the seeded night; template engine. `final/03-now-500.png`, `03-now-route-500.png`, `16-approvals-500.png`, `03-evening-now-500.png` | ⚠️ run B: plan (Win, 3 stars), model-written brief 2.7 s, route PASS; no evening review yet; re-plan not run (read-only) | ❌ the walk has not been walked against "leave by"; sleep from a watch is not connected (Now says "watch: no night measured") |

### 3a. An optional event Atlas suggests, and the reminder when a planned block starts

**Shows:** Atlas brings up an optional "Near you" event once, when it fits, and says who is going.
"Add it" never changes the plan by itself: it raises a new plan with one approval card, and only
once that card is approved is the event a block of the day, with its walk, its leave-by and "Time to
go". On the same day Atlas says when a planned block starts.

**Setup:** `python -m contexts.demo optional --data-dir /tmp/atlas-optional` (an invented day,
everything in it labelled demo). It prints the lines under **Expected** and, on stderr, the command
that runs the app on that directory. For the card on a live box instead: a plan for today, and an
optional event 90 to 30 minutes ahead that fits and names somebody, between 8 AM and 8 PM, with no
approval card open.

**Steps** (the scene plays them in event time; on a live box they are these taps)
1. Morning, **Approvals**: **Approve** the day's plan. Block reminders are for a plan that was approved.
2. When a planned block starts Atlas says so (a "Next up" toast; spoken with "speak up on its own").
3. 90 to 30 minutes before the event the card "NEAR YOU · Atlas suggests" opens over whatever screen
   is up, and the same sentence is spoken. Tap **Add it**, or say "yes" / "add it" within 30 seconds
   of the spoken line.
4. The card now lists what would move and one button, **Review and approve** → **Approvals** →
   **Approve**. Voice never approves.
5. **Calendar** (`/now/`): the event is a block of the day. Ten minutes before the time to leave
   Atlas says "Time to go".

**Not today** at step 3 instead: the card goes away, the box records the no, and Atlas never
suggests that event again.

**Expected** (the scene's output; the same lines on every run for the same day, whatever the directory)

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

- With the app running on that directory: `GET /api/day/plan?day=2026-10-12` has the block
  `Friends' dinner` with `fixed: true` and `origin: "optional"`; `GET /api/day/brief` has
  `latest.suggest` and `latest.start`; `GET /api/day/optional` (on the scene's own day) says `accepted`.
- The block-start line is the afternoon study block's. After the yes is approved the next block of
  the day is the dinner itself, and an accepted event with a walk is announced by "Time to go", not by
  a block-start line.
- Nothing is written anywhere: both approvals are "keep this as today's plan" on a box with no
  calendar connector.

**What can go wrong**
- No suggestion on a live box: `GET /api/day/optional` says why in `suggest_held` for each event
  ("not within the suggestion window", "nobody is named", "does not fit: ...", "an approval card is
  open", "outside waking hours", "already suggested"). Fallback: play the scene and read its lines.
- The yes is refused (HTTP 409, the reason in words): there is no plan for today yet (make one on
  Now first), or the event no longer fits.
- The card expired: every approval card lasts 5 minutes. The event is still on offer: tap **Add it**
  again.
- A yes less than 90 seconds after the suggestion: the card on screen is there at once, but the
  spoken "Okay, I've worked it in" waits for the arbiter's gap between two unasked lines. The scene
  answers after two minutes for that reason.
- With a calendar connector the same card writes real events. Then it is the owner's tap: the scene
  stops and says so, and the walkthrough denies the card.
- The box's cameras are at home: after "Time to go" Atlas cannot see the walk, so the scene ends at
  the time to leave.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| Pass: `contexts/demo/tests/test_optional_demo.py` (the lines above, the same for two directories, a gateway on the directory, a restart saying nothing again), `contexts/day/tests/test_optional_events.py`, `test_optional_suggest.py`, `tests/test_demo_walkthrough.py` | Pass, on a test gateway on the scene's own directory: `GET /api/day/plan` has the block (`fixed`, `origin: "optional"`, committed), `/api/day/now` has it as the next event with a 16 min walk and leave-by 4:59 PM, the brief has `latest.start` and `latest.suggest`, `check-api.mjs` all passed, the walkthrough's day and coach steps 16 PASS, 0 FAIL, 5 SKIP. On the seeded night (`seed --now`, today's plan made first) the whole walkthrough with `--approve-local` said yes to Board game night, the guard approved its card and the block was in the plan: 100 PASS, 1 FAIL (the known "while I was out" defect), 20 SKIP. The card read in Night and Day at 500x1000 and 1440x900. Simulated with: the invented day; template engine; fake model | Not run | Not run: nobody has heard the suggestion or answered it by voice; the card was read on screen in Night and Day |

---

## 4. Study coach

**Shows:** the box measures a focus block at the desk, notices the breaks and guesses why, asks "Was
that right?", and after enough corrections offers to learn that kind of break from your own pictures.

**Setup:** a planned study block that has ended, with someone at the desk during it. **Do this
before any re-plan** (defect 2).

**Steps**
1. On **Now**, under the block "Study block: milestone 1 writeup", read the "COACH · 10 H AGO" note.
2. Open **Was that right? Tell the coach about the 2 breaks**.
3. For "8:52 AM, away 17 min · The coach's guess: right after a message arrived in the group chat",
   tap **Yes** or one of **No: phone**, **No: someone came in**, **No: food or drink**, **No:
   bathroom**, **No: I was done**, **Something else**. (A break with no guess asks "Was there a
   reason?": **No reason**, **Phone**, **Chat**, …)
4. Open **Coach** (`/coach/`, or the link "Every block and what Atlas is learning").

**Expected**
- The note: "Milestone 1 writeup: 15 of 50 minutes at the desk. 2 breaks; the longest was 17 min,
  right after the group chat lit up. Longest unbroken stretch: 10 min. Try this: hold chat
  notifications until the break."
- "The numbers and the evidence" opens min done / planned, breaks, longest stretch.
- On Coach after one answer: "How often the coach is right: The coach is right about chat breaks 0 of
  1 time." and "What Atlas is learning from this · phone in hand · 1 of 5 answers, 0 of 2 where the
  coach was wrong. Then Atlas asks to learn it."
- After 5 answers on one kind with 2 wrong: an export card on Approvals (a training export, so a
  person approves it).

**What can go wrong**
- "No breaks to ask about": nobody left the desk during a block, or a re-plan was approved (defect 2).
- No block at all (the box today): there is nothing to coach. Fallback: the seeded laptop.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/day/tests/test_coach.py`, `test_coach_feedback.py` (incl. the proposal after 5 answers), `test_coach_expression.py` | ⚠️ run A: 1 block, 2 breaks, the note, one break answered ("guess chat → answer phone"). The teach-task proposal SKIP: 2 breaks cannot reach 5 answers. Simulated with: the seeded night. `final/04-coach-500.png`, `after/coach-500.png` | ❌ run B: 0 focus blocks, 0 notes on the box today | ❌ no real study block has been measured at the table |

---

## 5. Day Run

**Shows:** try the next 24 hours as a game before living them: a bad plan fails with reasons, a good
plan earns three stars, hard mode throws problems at it and the box re-routes, and Play mode walks
the day on the real campus map.

**Setup:** none (the scripted day ships in `fixtures/play/demo.yaml`).

**Steps**
1. Open **Day Run** (`/dayrun/`). **Demo day** is selected ("A scripted Sunday at Knight Hacks.").
2. Tap **Bad plan**. (Link: `/dayrun/?plan=bad`.)
3. Tap **Good plan**. (`/dayrun/?plan=good`.)
4. Tap **Hard mode: throw problems at this plan**. (`/dayrun/?hard=1`.)
5. Tap **▶ Watch the day** (after hard mode: **▶ Watch the day go wrong**). (`/dayrun/?play=1`.)
   Speeds: **1 min/s**, **12 min/s**, **36 min/s**, **72 min/s**; **Close** ends it.

**Expected**
- Bad: "✗ Fail ☆☆☆ · The plan fails. COP4533 Project Milestone 1: at risk, 270 focused minutes left
  with 38 h to go (risk 0.94, needs 0.50 or less). Sleep: 5.3 h (6 h needed). Meals: 0 (2 needed)."
  and "1 move was rejected: you miss the 12:00 Knight Hacks: judging and demos by 11 minutes".
- Good: "✓ Win ★★★ · Win, 3 stars: every deadline on track, 8.5 h of sleep, 3 meals."
- Hard: "Hard mode: what went wrong · Knight Hacks: judging and demos moves from Engineering II to
  Classroom Building I and now starts at 11:45 instead of 12:00 · Student Union is closed today ·
  Before Win ★★★, 40 min walking… After Win ★★★, 47 min walking… The box made 4 changes to keep the
  plan standing."
- Play: a figure walks the campus map ("Campus map with 9 places and 8 walks", "© OpenStreetMap
  contributors"), with a list of walks ("1 09:30 to '63 South (eat) · 7 min · 570 m").

**What can go wrong**
- Nothing here needs a model or a camera. If the map is blank the place pack is missing
  (`config/places/ucf.map.json`).

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/play/tests/test_dayrun.py`, `test_dayrun_api.py`, `test_walknet.py`, `test_places.py` | ✅ run A: 9/9 (bad fails with reasons, good 3 stars, hard re-routes, map 254 buildings, a 290 m walk). Simulated with: the scripted demo day. `final/05-dayrun-bad-500.png`, `-good-`, `-hard-`, `-play-` | ✅ run B: 9/9, 0.35 to 1.25 s each | n/a (no hardware in this use case); "My day" from live sources not tried at the table |

---

## 6. Circle

**Shows:** friends' Atlases agree on a time by exchanging one-word answers; the agreed plan lands in
your day; a league with a stake; sharing a place name that expires.

**Setup:** a Circle configured (`circle.member_id`, `circle.circle_id`), or the seeded Circle of
three (Hemdutt on this box, Maya, Dominick). The Jetson has no Circle configured today.

**Steps**
1. **More → Circle** (`/circle/`). Tabs: **People**, **Plans (1)**, **League**, **Sharing**.
2. People: read the three cards; tap **Route to Maya** or **On the map**.
3. **Plans** → under "Being worked out", "Gym" → **Read the conversation ›**. Then **‹ All plans** →
   "Settled conversations (2)" → "COP4533 exam review" → **Read the conversation ›**
   (`/circle/?tab=plans&thread=demo-circle-study`).
4. **Find a time for all of us** starts a new plan: it becomes a card on Approvals; nothing is sent
   until **Approve**.
5. **League**: read the table; "Set the stake": **Pays for food**, **Picks the event**, **Custom** →
   **Set the stake**; "Start a dare" → **Start the dare**. Both become approval cards.
6. People → "Share where you are": tap **Library**, **1 hour**, **Share this place**.
7. **What leaves this box, and what never does ›** (or the **Sharing** tab).

**Expected**
- People: "3 in demo-circle"; "Maya · #2 this week · 46% · Library · Shared from their phone at … ·
  until …".
- The conversation: "Hemdutt's Atlas (yours) Can everyone do COP4533 exam review, tomorrow 2:00 to
  4:00 PM at the Library?" / "Dominick's Atlas Free." / "Maya's Atlas Busy then. Suggests 3:00 PM." /
  "Hemdutt's Atlas (yours) Can everyone do tomorrow 3:00 to 5:00 PM instead?" / "Free." / "Free." /
  "Agreed: tomorrow 3:00 PM, the Library, 3 of 3 in." / "Everyone is in. It is on the shared calendar."
- In the day: after the next plan, Now lists "on the calendar · Breakfast · Tomorrow 8:30 AM to 9:30
  AM" and the exam review as fixed blocks.
- League: "Maya had the worst week: 2 of 5 sleep targets. Dinner is on Maya. Hemdutt picks the place."
  and "Stake: The worst week buys the next group meal; the best week picks the place."
- Sharing without Neon: "Nothing is shared · This box is set up as Hemdutt in circle demo-circle, but
  it is not connected to the Circle's shared database, so nothing leaves it."

**What can go wrong**
- "This box is not connected to the Circle's shared database": no `NEON_DATABASE_URL`. The seeded
  Circle still shows every screen; say it is one box playing three.
- Two real boxes: replies take about 28 s and the agreement about 57 s after approval (measured
  earlier between two boxes on Neon). Fill the wait with the League tab.
- Never approve a Circle card from a script: it leaves the box.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/circle/tests/test_protocol.py`, `test_threads.py`, `test_presence.py`, `test_sharing.py`; `contexts/play/tests/test_dayrun_agreed.py`; `contexts/demo/tests/test_circle_demo.py` | ✅ run A: 12/12: members, three threads (one waiting, two agreed, one after a counter), the conversation, agreed plans in today's plan after the re-plan, League, a place shared and stopped (no shared store, so it stayed on the box), a new plan raised a `circle_share` card that the script denied. Simulated with: the Circle of three on one box. `final/06-circle-*.png` | ❌ run B: the Circle is not configured on the box | ❌ two boxes negotiating over Neon was not repeated today |

---

## 7. Sort It

**Shows:** hold an item up, the box says which bin and draws the way there through the room (around
people), then scores the throw against par.

**Setup:** the bin camera aimed at the bins with the `recycling` and `trash` zones drawn; a floor
plan (`config/rooms/dorm.yaml`) calibrated for each camera (four floor points); for a face-started
round, recognition on and the player enrolled.

**Steps**
1. **More → Sort It** (`/play/`): the room in 3D, **Room** / **Top** views.
2. Tap an item chip: **Can**, **Bottle**, **Paper**, **Cardboard**, **Cup**, **Napkin**, **Chip bag**,
   **Food container**, **Banana**, **Wine glass**.
3. Hold a can up to the bin camera for two seconds, walk to the bin, drop it in.
4. "Practise what you got wrong": tap **Practise: Can**.
5. **More → Room map** (`/map/`) → **3D**; under "Sort It: which bin?" tap **Can**
   (`/map/?view=3d&sort=can`).

**Expected**
- "Can goes in recycling: 3.1 m, about 5.1 s." with a pin "Can goes here" over the recycling bin and
  a line across the floor; with a person in the way the line bends around them.
- After the throw the result: right or wrong bin, seconds, points (the scene's round: "can right bin,
  6.0 s, 150 points").
- "The bin camera's own view · Can goes in recycling · 17 steps" (a 32 by 24 grid).
- "Are you getting better? · 1 round played in this sitting. The comparison fills in after five."
- Room map 3D: pins "Keys", "Laptop", "Wallet +2", "Dorm room · nobody in view", Render: **Native**,
  **FSR Quality**, **FSR Performance**, **Compare**.

**What can go wrong**
- "3D view is not available on this device": no WebGL. The same room is drawn from above; carry on.
- The round does not start: the camera does not see the item for two seconds, or the zones are off.
  Fallback: `python -m contexts.demo fake perception --gateway http://<throwaway box>:8000 --realtime`
  plays a scripted round (62 s) on `/play/`. Say it is the scripted scene.
- The dots are in the wrong place: the calibration is the placeholder (defect 10). Calibrate.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/play/tests/test_game.py`, `test_pathfinding.py`, `test_room_plan.py`, `test_room_grid.py`, `test_room_live.py` (the detour) | ✅ run A: 10/10: twin, live state with 2 people while the scene played, a round started and scored from the scene, route 3.06 m par 5.1 s, "which bin?" hint, practice set. Simulated with: the fake camera scene (`--realtime`). `final/07-map-3d-500.png`, `07-play-can-500.png` | ⚠️ run B: twin and routes served; 0 rounds, nobody in the room, placeholder calibration | ❌ no real throw, no face-started round, no real calibration |

---

## 8. Waste Watch / Impact

**Shows:** what the bin camera counted this week, what it adds up to, the weekly goal, and the rule
behind every wrong throw.

**Setup:** disposals in the log (the seeded night has 7; the scene adds 3).

**Steps**
1. **More → Impact** (`/impact/`). (From Today: "What this adds up to, and this week's goal".)
2. Read the top: "Kept out of landfill this week". Open **How these are estimated**.
3. "This week's goal".
4. "What went where": on a wrong one tap **Teach me**.
5. "Practise in Sort It".

**Expected**
- "49–70 g estimate · 4 of 6 recyclables the camera saw reached recycling (that count is measured)."
  and "Last milestone: 3 items kept out of landfill (4 so far this week)."
- "67% of recyclables recycled · goal 80% · The next 4 in the right bin, with none missed, reaches
  80%." and "Wrong items in recycling: 1 of at most 2 this week (1 to spare)."
- The feed: "Can → trash · Wrong bin · 6:04 PM · Empty drink cans go in recycling at UCF: aluminium is
  the most valuable thing in the bin."
- Estimates are labelled "estimate" and "Estimates from illustrative factors in config/impact.yaml,
  not measurements."

**What can go wrong**
- "nothing counted yet": no disposal this week. Throw one thing away in view of the bin camera, or
  use the seeded night.
- "no picture stored": the seeded log has no frames.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/impact/tests/test_story.py`, `test_waste_watch.py` | ✅ run A: 7/7: counts, goal, feed with its rule, milestone ("That's 5 items kept out of landfill this week." after the scene), "Teach me". Simulated with: the seeded night + the fake scene. `final/08-impact-500.png` | ⚠️ run B: 1 of 3 recyclables recycled, the feed with its rule; no milestone yet | ⚠️ the box's one counted disposal is real; nobody checked it against what was thrown |

---

## 9. Identity

**Shows:** recognition is off until you switch it on; you enrol your own face by a phone scan or
photos; expressions for the coach and an anonymous room mood are separate opt-ins; "forget" removes
you.

**Setup:** the face recognition model installed on the box. **It is not installed on the Jetson or
the laptop today**, so every step below is unverified on a live screen.

**Steps**
1. **More → People** (`/people/`).
2. "Recognise enrolled people" → **Switch on** → **I understand, switch it on**.
3. "Teach Atlas your face": type the **Name**, tick the consent box, then **Scan my face** (the
   phone's front camera; follow "Look straight", "Turn a little left", "A little right", "Chin slightly
   up") or under **Other ways**: **Choose or take photos**, or **Start at the box**.
4. "Read my expressions" (per person, for the coach).
5. "Room mood from everyone (anonymous)" → **Switch on…** → **Confirm the room mood**.
6. "Enrolled" → the person → **Forget for good**.

**Expected**
- "Enrolled. From now on Atlas counts their habits as theirs." and the name under "Enrolled".
- With it off: "Nobody can be enrolled, and nobody is recognised, while it is off."
- Without the model: "This box cannot recognise anyone yet: the face recognition model is not
  installed on this box." and **Switch on** is greyed out. That is what both boxes show today.

**What can go wrong**
- The model is missing: nothing can be shown live. Fallback: say what the screen says and show the
  test names; do not claim it.
- "The camera was not allowed": the phone needs https (`ATLAS_HTTPS=1`) for the scan; use photos.
- "This face is already enrolled as someone else here, so nothing was saved."

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `tests/test_owner_recognition.py`, `contexts/perception/tests/test_identity.py`, `test_identity_privacy.py`, `test_face_profile.py` | ⚠️ run A: the five read endpoints PASS; the screen shows the "not installed" state. `final/09-people-500.png` | ⚠️ run B: same: `available=False`, 0 enrolled | ❌ nobody has enrolled, been recognised or been forgotten on the box |

---

## 10. Privacy

**Shows:** faces are blurred before anything is stored, the box proves what left it, and one screen
deletes what it keeps.

**Setup:** the Jetson (the firewall only runs there).

**Steps**
1. **More → Privacy** (`/privacy/`). "Pictures": the **Blur faces** switch is on.
2. **More → Timeline** → open a frame with a person in it.
3. Privacy → "The internet": **Offline mode**. (Or `make offline` on the box.)
4. "What left the box".
5. "Delete": pick **Everything**, **Pictures only** or **Memory only** → **Delete…** → read the
   warning → **Cancel** (or **Yes, delete for good** on a throwaway box).

**Expected**
- "Faces are blurred on the box before a picture is written to disk."
- A frame where the face is a blur.
- On the Jetson, the list of what each group sent and what was blocked (the API says: enforced, 13
  allowed destinations, 13 blocked). On a laptop: "No firewall on this box".
- The wipe warning: "This will permanently delete all pictures, the memory, exports and the event
  log. The confirmation is valid for two minutes."

**What can go wrong**
- Do not press **Yes, delete for good** on the demo box. Show the warning and cancel.
- Offline mode pauses connectors, the Circle and learning; switch it back before the next beat.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/perception/tests/test_face_blur.py`, `contexts/vault/tests/test_policy.py`, `test_wipe.py`, `tests/test_egress.py`, `test_egress_api.py` | ⚠️ run A: policy (blur on) and egress endpoint PASS; the wipe dialog opened and was cancelled (`final/10-privacy-delete-500.png`); no frames, no firewall here | ⚠️ run B: blur on; firewall enforced, 13 allowed, 13 blocked; a cited frame opened as image/jpeg (not looked at by a person) | ❌ nobody has looked at a stored frame for the blur today, read the egress screen on the box, or run a wipe |

---

## 11. Teach / learning

**Shows:** you describe a task, approve the export of blurred examples, and the lab's teacher and
trainer turn it into a better model that is swapped onto the box. The Learning lab replays a run.

**Setup:** none for the card and the sample run; a real run needs the lab's GPUs.

**Steps**
1. **More → Teach** (`/teach/`). Fill "What should it be called?" (e.g. "Desk objects") and "What
   should Atlas do with each picture?"; pick **Any**, **Door camera** or **Desk camera**; thumbs up or
   down on a few examples; **Start teaching**.
2. **Approvals**: read the export card. **Deny** in rehearsal; **Approve** only when the lab is ready.
3. **More → Learning lab** (`/lab/`; `/lab/?sample=1` forces the sample).
4. Tap **Replay**; step through **1** to **5**; **How to read this**; **Sources**.

**Expected**
- Teach "Progress": the task appears after approval; before: "Nothing taught yet".
- Learning lab: "What Atlas learned · Deployed · The task (tool-calling): Pick the right memory tool,
  with the right arguments, for a question about what the cameras saw." Two runs: "2:05 PM ·
  Deployed" and "11:20 AM · Not deployed". "Start to finish: 17 min 55 s".
- The sample is labelled: "Sample run: not measured on our hardware." Say that sentence out loud.
- "1 Taught from · 112 examples left the box for this task. You approved this export at 11:11 AM."

**What can go wrong**
- No real run: the lab shows the sample. It says so itself; do not present its numbers as ours.
- The box has no `/lab/` yet (run B): open it on the laptop.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/learning/tests/test_export.py`, `test_lab_view.py`, `test_lab_sample.py` | ✅ run A: 8/8: teach raised an `export_training_data` card which the script denied (never approved); the lab's sample run and its story PASS. Simulated with: the lab's labelled sample run (`?sample=1`). `final/11-lab-sample-500.png`, `11-teach-500.png` | ⚠️ run B: 1 task and 2 deployments recorded; no Learning lab in the box's build | ❌ no export approved and trained end to end today; the first real fine-tune result is still "pending" in the README |

---

## 12. Voice

**Shows:** ask out loud, hear the answer, hear the day narrated, and be asked for an approval by
voice (which can deny but never approve).

**Setup:** a microphone and speaker. On a phone the page must be https (`ATLAS_HTTPS=1`) for the
microphone. For voice on the box: the voice service running and the speaker paired.

**Steps**
1. Ask tab: tap the microphone (**Talk to Atlas**) and say "Where did I leave my keys?"
2. Tap **Hands-free off** to turn it on; after the answer ask a second question; say "stop".
3. Settings (top right) → Voice: **Speak answers**, **Hands-free conversation**, **Speak up on its
   own**.
4. With "Speak up on its own" on: **Now** → **Brief me**.
5. **Now** → **Re-plan from here** and listen for the ask.

**Expected**
- The answer is read aloud ("Speak answers on").
- The brief in spoken form: "…first up is the milestone 1 writeup at twenty to nine…".
- The ask: "I'd like to make these nine blocks today's plan. Okay?" Saying "yes" only opens Approvals:
  "By voice you can deny a request but never approve one".
- Settings shows the voice in use ("Voice: Samantha (en-US), at 1.05× speed") and says that browser
  speech recognition can send audio to Google or Apple.

**What can go wrong**
- No microphone permission, or http on a phone: type the question.
- A noisy room: hands-free hears the crowd. Use tap-to-talk.
- Operator screen "Voice on the box · Off · The voice service is not running on this box.": start it
  on the Jetson, or use the phone's voice.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/voice/tests/test_session.py`, `test_service.py`, `test_engines.py`, `test_text.py`, `test_audio.py` | ⚠️ run A: the status endpoint only ("not running" on a laptop); the Settings sheet read (`final/12-settings-500.png`) | ⚠️ run B: the voice service is running, state idle | ❌ nothing was spoken or heard: no hands-free question, no narration, no spoken ask |

---

## 13. Inspection (NextEra SPOT)

**Shows:** the same box for a robot inspection operator: three things to do this shift, the missions
replayed, cited answers offline, a shift handoff, and a work order that stays on the box.

**Setup:** the kit in `fixtures/spot/kit` (it ships). A display at least 1700 px wide (defect 5).
Optional: `ATLAS_PROFILE=inspection`. Details and the 75-second script: `docs/spot.md`.

**Steps**
1. **More → Inspection** (`/spot/`) on the big display.
2. Read the three cards and the four numbers. **Assumptions and method**, **Data check 12/12**.
3. "Mission replay": **60 ×**, then **Play** (**Pause**, **Rewind**; **30 ×**, **240 ×**, **1200 ×**).
4. "Needs attention now", #1 HX-202: read the score bar and the reason.
5. "V1 against V2: what changed" → **Rows behind it**.
6. "Ask": tap **What should I check first this shift?** (also **Is V2 better than V1?**, **What
   happened at WP05?**, **Which inspection types are least reliable?**, **Show hidden failures**,
   **Status of HX-202**).
7. "Shift handoff" → **Copy**.
8. On HX-202 tap **Re-inspect** (or **Work order**), then **Approve** on the line "Waiting for your
   OK: re-inspection (HX-202). Stays on this box."

**Expected**
- Header: "27 missions, 270 actions, 40 anomalies · data to 2026-09-10 10:25 · synthetic challenge
  data" and "Runs on this box, offline".
- Cards: "1 Re-inspect HX-202 (CRITICAL asset): HIGH anomaly, unverified", "2 Low battery goes with
  failures: watch the next mission that drops below 40%", "3 Get eyes on GAUGE-209 (HIGH asset): no
  good inspection for 48 h". Numbers: 96% (26 of 27), 82% (221 of 270), 81% (21 of 26), 38% (15 of 40).
- Replay: "Sep 1, 18:03 · event 2 of 594 · M001", the counters climbing, amber blocks for missions
  "COMPLETED with failed actions inside".
- The answer: "Check these first, in this order. (1) Re-inspect HX-202…", "Evidence (32 rows)" as
  tappable ids, and "Built from the tables by code; no model wrote this."
- The card: "Re-inspect HX-202 Heat Exchanger 02 at WP05 on the next mission… Recorded as a local
  work order on this box; nothing is sent anywhere." After **Approve**: one file under
  `<data dir>/spot/work_orders/`.

**What can go wrong**
- The first card is cut off on the left: the window is narrower than about 1700 px (defect 5).
- The answer takes 20 s on the box (its build rephrases with the model). The laptop build answers at
  once. Tap the question before you start talking about it.
- "no kit on this box": set `ATLAS_SPOT_KIT`.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/spot/tests/` (7 files: analysis, replay, assistant/orders/API, scorer, dataset, kit) | ✅ run A: 12/12: triage (HX-202 first, 72.3), 18 findings, 12 of 12 numbers agree, handoff, the replay moved 0 → 22 of 594 in 3 s at 1200×, ask, Re-inspect HX-202 raised a card, the guard approved it, 1 work order on disk. Simulated with: the sponsor's synthetic kit and its replay. `final/13-spot-1920.png`, `13-spot-1440.png` (clipped), `13-spot-ask-1440.txt` | ⚠️ run B: 8/8 reads PASS; ask 20.7 s; replay and work order not run (read-only) | n/a: there is no robot; the data is the sponsor's synthetic kit. Not shown on the real big display yet |

---

## 14. Profiles

**Shows:** the same box with other words and rules: a night-shift worker, a family home, a plant
operator.

**Setup:** restart the gateway with `ATLAS_PROFILE=shift-worker`, `family-home` or `inspection`
(default: student). That is a restart, so do it on a laptop, not on the demo box.

**Steps**
1. `ATLAS_PROFILE=shift-worker python -m contexts.gateway`, open **Ask**, **Now**, **Today**, **Day
   Run**.
2. Repeat with `family-home` and `inspection`.
3. Check: `python scripts/demo_walkthrough.py --gateway http://127.0.0.1:<port> --only profiles --expect-profile shift-worker`.

**Expected**

| Profile | Ask screen says | Words (workplace, activity, event, deadline) | Day Run's demo day |
| --- | --- | --- | --- |
| student | "Ask about your desk, your classes or your evening." | desk, study, class, assignment | "A scripted Sunday at Knight Hacks." |
| shift-worker | "Ask about your station, your shifts or your evening." | station, work, shift, handover | "A scripted day for the shift-worker profile (night-shift)."; with nothing planned: "Meals: 0 (3 needed)" |
| family-home | "Ask about your table, your appointments or your evening." | table, homework, appointment, chore | "A scripted day for the family-home profile (school-night)."; "Leo: bins out: due at 20:00…", "Ava: science fair poster: at risk…" |
| inspection | "Ask about your plant, your missions or your evening." | plant, inspection, mission, work order | the student's scripted Sunday (this profile has no scenario or meters of its own) |

**What can go wrong**
- The suggested questions stay the student's (defect 8). Type a fitting question instead.
- The inspection profile changes words only; the inspection work itself is use case 13.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/profiles/tests/test_loader.py`, `test_boot.py` | ✅ three more gateways (ports 8957 to 8959) on the scripted-time night: `--expect-profile` PASS for shift-worker, family-home, inspection, and FAIL as it should against the student box. Screens read: `final/14-shift-*.png`, `14-family-*.png`, `14-inspection-*.png` | ⚠️ run B: the box runs `student`; switching needs a restart, which was not done | n/a |

---

## 15. Operator screen, zones editor, Live (teammates' screens)

**Shows:** what the box is doing right now, on a display next to it; where each camera's zones are;
the live pictures.

**Owner:** these three screens and the capture pipeline are the teammates' (the Jetson owner's). The
steps are listed so the sheet is complete; changes go through them.

**Steps**
1. **Operator**: open `http://<box>:8000/operator/` on the display. It has **Set zones** and **Live**
   at the top.
2. **More → Zones** (`/zones/`): pick the camera (**Door camera**, **Desk camera**), pick the zone
   (**Door**, **Recycling**, **Trash**; **Desk**), drag a box over that part of the picture, **Save
   zone**. **New picture**, **Whole picture**, **Use demo zones**.
3. **More → Live** (`/live/`).

**Expected**
- Operator: the camera tiles, the room in 3D with pins ("Keys", "Laptop", "Wallet +2"), "VOICE ON THE
  BOX", "LAB · Idle · Nothing is being taught right now.", "HAPPENING NOW" with events as they arrive
  (during the inspection replay: "Spot action started").
- Zones: "A thing counts as in a zone when the bottom middle of its box is inside. Sort It reads the
  bin boxes when the app starts, so restart the gateway after moving Recycling or Trash."
- Without cameras: "No cameras on this box" (Operator, Live) and "No picture from this camera right
  now. Is it plugged in?" (Zones).

**What can go wrong**
- A camera is missing or "degraded": check the USB cable and `ls /dev/video*` (the pre-flight says
  the same). Today the box reports only `cam-desk`, degraded at 10 fps.
- Moving the bin zones needs a gateway restart to reach Sort It. Do it before the demo, not during.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/perception/tests/test_zones_api.py`, `test_zones.py`, `contexts/capture/tests/` | ⚠️ run A: the three screens load; zones, rules, alerts, runtime numbers and pre-flight endpoints PASS (pre-flight 5/11 ready, as expected on a laptop); no cameras. `final/15-operator-1440.png` (defect 9) | ⚠️ run B: `cam-desk` degraded at 10 fps and no `cam-door`; zones for both cameras; pre-flight 11/12 ("not ready: camera cam-desk") | ❌ nobody looked at Operator or Live on the box's display, or drew a zone, in this pass |

---

## 16. The look guide ("where is…?", "where do I find…?", "which bin?")

**Shows:** ask where something is and Atlas shows you: a red dot travels there, the thing is marked, and the same
sentences are captions and (with Speak answers on) speech. One layer for every screen.

**Setup:** none for the pointer. For eyes: Settings → **Use my eyes to guide me** on `https://` or `localhost`.

**Steps and what you should see**
1. **Ask → "Where did I leave my keys?"** The answer arrives as before; then the dot travels to the picture under it
   and the picture is outlined with a dashed "About here" (the box keeps a zone and words, not a box in the picture).
   The caption carries the answer's first sentence and **Show me in the room**, which opens the room map on that
   thing's pin. On the seeded laptop the picture is the "Frame not stored on this box" placeholder and the caption says so.
2. **Ask → "Where's my League?"** No question goes to the box. Circle opens, the League tab is pressed, the tab is ringed.
3. **Ask → "Which bin for the cup?"** Sort It opens, the room lights the trash bin ("Cup goes here"), the route is
   drawn, the dot fades, and the caption reads "Throw it here: Cup → Trash" with the rule sentence from the rules file.
4. **Inspection → "Where is Heat exchanger 2?"** (or the Guide me chip). The plan switches floor if needed, green
   dashes draw the walk from "You are here", the dot follows it, the mark is ringed "Here it is". Then type **next**:
   the dot steps to the next do-first item. **"Where do I look for the battery?"** opens the Analysis tab first.
5. **Attention (simulated):** open `/spot/?gazeSim=do-first:1,trust,card:watch`. A banner says the gaze is simulated;
   "✓ seen" appears on those three; "You've looked at 1 of 3 things that need you". Tap **Send a technician** on item
   2: "You haven't looked at Heat exchanger 4's alarm yet", the dot goes there, **Continue anyway** carries on.
   `/spot/?tab=run&route=last&show=result&gazeSim=shiftrun:turn-back,shiftrun:run` shows "Where you looked" and the scan path.

**Links that hold the pointer for a screenshot:** `/?guide=keys&t=1`, `/?guide=where's my League?&t=1`,
`/?guide=which bin for the cup?&t=1`, `/spot/?guide=where is HX-202&t=0.5`, `/spot/?guide=explainer:battery&t=1`.

**What can go wrong**
- Nothing moves: Escape or a tap anywhere ends a run by design. Ask again.
- "I am not sure what to point at. Did you mean…": the name matched nothing well; tap one of the offers.
- The eyes switch is greyed out: the page is on plain http. Use `ATLAS_HTTPS=1` or `localhost`. The pointer still works.

**Verification**

| tests | seeded app | Jetson | real hardware |
| --- | --- | --- | --- |
| ✅ `contexts/spot/tests/test_guide.py`, `test_route.py` (76 tests); `node web/scripts/check-guide.mjs` (223 checks: intent, resolver, A*, calibration maths, snap, attention rules, exact-or-about) | ✅ run C: the three API steps. Headless Chromium at 1440×900 and 500×1600: the links above, the cross-screen League run, the bin run, the attention run by `?gazeSim`, the camera opt-in with Chromium's fake camera (model and runtime loaded from `/gaze/` only, zero cross-origin requests, no-face fallback). Simulated: `?gazeSim` is a script, and the fake camera has no face in it. | ⬜ not tried | ⬜ real gaze accuracy, dwell arrival and snap on a face; spoken output. See table-day rows 13 and 14. |

---

## The 5-minute judge path

Order chosen so that everything that needs the model is asked early (it warms up), the coach comes
before the re-plan (defect 2), and nothing needs a restart.

| Time | Beat | Do | Fallback |
| --- | --- | --- | --- |
| 0:00 | It sees and remembers, and shows you | Ask → **Where did I leave my keys?** Let the red dot travel to the picture ("About here"), then **Show me in the room**. | Laptop with the seeded night: same answer and pointer, "Frame not stored on this box" |
| 0:30 | It can talk and look things up | **How's your day?**, then **What is 17 times 23?** | Skip the web question unless there is time to wait 25 s |
| 1:00 | It runs the day | Now → **Brief me** → **Route** on the next block | The template brief appears if the model is slow |
| 1:40 | It coaches | The coach note → **Was that right?** → **No: phone** | Laptop, seeded night |
| 2:10 | It asks before acting | **Re-plan from here** → **Review the new plan** → **Approve** | If no card: **More** → **Re-plan from here** |
| 2:40 | A game with real stakes | Day Run → **Bad plan** → **Good plan** → **Hard mode…** → **▶ Watch the day go wrong** | None needed: no model, no camera |
| 3:20 | The room | Sort It: hold up a can, walk it to the bin | Tap **Can**; or the scripted scene on a throwaway box |
| 3:50 | Friends | Circle → Plans → "COP4533 exam review" → **Read the conversation ›** → League | Laptop with the seeded Circle of three |
| 4:20 | Private by construction | Privacy → "What left the box" on the Jetson; unplug the network and ask again | Laptop: "No firewall on this box", so use the box |
| 4:40 | Another world, same box | Inspection on the big display → type **Where is Heat exchanger 2?** (the dot walks the route to its mark) → **next** → **Send SPOT back** → **Approve** | Laptop build answers at once; `/spot/?guide=where is HX-202&t=1` holds the pointer |

**The three questions most likely to impress, with what was measured on the box**
1. "Where did I leave my keys?" 3.1 and 6.4 s, with the real frame. Fallback: the seeded night.
2. "What is 17 times 23?" → "391", 2.2 to 2.4 s (then "Who won the FIFA World Cup in 2022?" if the
   room will wait about 25 s for seven cited links). Fallback: "Tell me a joke".
3. Inspection: "What should I check first this shift?" Cited to row ids, works with the cable out.
   20.7 s on the box's current build, instant on a current one. Fallback: tap the first card
   (HX-202) and read its reason.

Avoid live: "What happened while I was out?" and "overnight" on the box (17 to 37 s, and not
repeatable today).

**What is simulated and what is real, per track**

- **Main (Atlas for a student).** Real on the Jetson: the cameras, the on-device model's answers and frames, the
  firewall. Simulated on the laptop: the seeded night, the fake model (`fake-0`, which echoes its evidence), the fake
  camera scene. The look guide's pointer is real UI on both; its "About here" on a picture is honest about having no
  box for the thing. Gaze is real only on a person's own webcam after its own check; everywhere else say "gaze off".
- **NextEra (Inspection).** Everything runs on the sponsor's synthetic kit: 27 recorded rounds of one robot, replayed.
  No robot is connected. The resolver and the route are real code over that data; the walk is drawn only over legs
  SPOT walked (mostly one straight leg in this kit: not a corridor map, and no metres). The attention check shown
  with `?gazeSim` is a scripted stand-in and says so on screen; with a webcam it means "looked at", not "understood".
- **OneEthos (Waste Watch / Sort It).** The rules file and the bin answer are real. The room, its bins and the route
  are the box's room plan; on the laptop the round is the scripted scene and the feed is the seeded week. A real
  throw needs the bin camera (table-day row 5).

---

## Table-day checklist (needs a person and hardware)

None of these has been done in this pass. Each has the simulated stand-in to fall back on; say on
stage that it is the stand-in.

| # | Do this, exactly | You should see | If the hardware fails on stage |
| --- | --- | --- | --- |
| 1 | **Aim and light the desk camera.** Desk, keys and laptop in frame; a lamp on the desk; no window behind. Open `/live/` and `/operator/`. | Both camera tiles with a picture; `/capture/status` says `ok` near 15 fps for `cam-desk` **and** `cam-door` (today: only `cam-desk`, "degraded", 10 fps). | The seeded laptop: `python -m contexts.demo circle --data-dir /tmp/atlas-demo` |
| 2 | **Draw the zones.** `/zones/` → Door camera: **Door**, **Recycling**, **Trash**; Desk camera: **Desk**; **Save zone** each. Restart the gateway once after the bins (ask the box's owner). | A ✓ on each zone; a person walking in raises a zone event on Operator. | **Use demo zones** (big forgiving boxes) |
| 3 | **Calibrate the room twin, four floor points per camera.** Tape a square on the floor; measure its corners in metres from the room's corner; read the same corners off the picture as fractions (0 to 1). Then `curl -X PUT http://<box>:8000/api/room/calibration/cam-door -H 'Content-Type: application/json' -d '{"image": [[x,y]×4], "floor": [[m,m]×4]}'` (same order; add `-H "Authorization: Bearer $ATLAS_TOKEN"` if the box has a token). Repeat for `cam-desk`. There is no screen for this. | `GET /api/room` lists both cameras under `calibration`; on `/map/?view=3d` a person standing on the tape is drawn on the tape. | The placeholder calibration: dots move, roughly. Use **Top** view and do not point at exact positions |
| 4 | **Install the face model and enrol a face.** People → **Switch on** → **I understand, switch it on** → name, consent → **Scan my face**. | "Enrolled. From now on Atlas counts their habits as theirs." | Not installed today on either box. No stand-in exists: describe it, cite `tests/test_owner_recognition.py`, do not claim it |
| 5 | **Run a Sort It throw.** Stand in view, hold a can up to the bin camera for two seconds, walk to the recycling bin, drop it. | "Can goes in recycling", the route, then right bin, seconds and points; Impact's feed gains "Can → recycling · Right bin". | `python -m contexts.demo fake perception --gateway http://<throwaway box>:8000 --realtime` (a scripted round, 62 s). Verified in run A |
| 6 | **Pair the speaker and check voice.** Pair it with the box; Operator shows "Voice on the box". On the phone: Settings → **Speak answers**; tap the microphone and ask. | The answer out loud; `preflight` says "voice on the box: ready". | The phone's own voice (Settings → Speak answers), or type |
| 7 | **Leave a package, pick up the keys.** Put a box in the door zone for 5 s. | The alert "Package left at the door" on the phone, with its blurred photo. | `preflight --reset` clears the cooldown; the seeded night already has two such alerts |
| 8 | **Look at one stored frame of a face.** Timeline → a frame with a person. | The face is blurred. | `contexts/perception/tests/test_face_blur.py` |
| 9 | **Prove the firewall.** Privacy → "What left the box"; then `make offline`, ask the keys question, `make online`. | The list of allowed and blocked destinations; the answer still comes. | The API read from run B: enforced, 13 allowed, 13 blocked |
| 10 | **Two boxes agree on a plan.** Circle → **Find a time for all of us** → **Approve** on each owner's phone. | "Free." in about 28 s, "Agreed" in about 57 s. | The seeded Circle of three |
| 11 | **Show Inspection on the real display,** full screen at 1920 × 1080. | Three cards fully visible. | Zoom out once (defect 5) |
| 13 | **Eyes: the two-minute test.** On the demo laptop (https or localhost): Settings → **Use my eyes to guide me** → allow the camera → **Start**; look at each of the 9 dots, then the 4 check dots, until the ring fills. Open `/spot/?gazeDebug=1` and look at the four corners. Then look at the far left and type **Where is Heat exchanger 2?** | The result line with a measured number ("Accurate to about 1/8 of the screen width (… px …)"); expect roughly 1/8 to 1/15 of the width on a laptop camera at arm's length, worse with glasses glare or low light. The debug dot and dashed cell follow your eyes; the caption starts "From where you are, look …"; the ring turns green with "That's it." after about 0.7 s on the mark. | Gaze off: the pointer starts from the middle and everything else is the same. If the check says "not good enough to use", that is the feature working: say so. |
| 14 | **Attention, with real eyes.** With row 13 done, open Inspection, read item 1 of "Do this first", then tap **Send a technician** on item 2 without looking at it. | "✓ seen" on item 1; "You haven't looked at Heat exchanger 4's alarm yet" with the dot on it and **Continue anyway**. | `/spot/?gazeSim=do-first:1,trust,card:watch`, and say it is the simulated path (the banner says it too). |
| 12 | **Run the pre-flight at T-15.** `python -m contexts.demo preflight --gateway http://<jetson>:8000 --reset` | "N/N ready - go." | Fix what it names; each line says how |

---

## Known gaps

**Look guide, not built (as of run C):** outlining an object inside a camera picture (a tracked object's box on the
refreshed snapshot, a bin's zone polygon over a snapshot, a boxed or "about here" position for keys-like things: Memory
does not keep a box today); routing on the plant floor around equipment from a traced walkable map (the route is the
leg graph only); the pointer following the campus route on Day Run. There is therefore no live-camera spotlight step in
the table-day checklist yet: add it when that work lands.

- **Identity is unverified everywhere outside tests:** the recognition model is on neither box, so
  enrol, recognise, expressions, room mood and forget have not been seen working.
- **Voice is unverified beyond "the service is running":** nothing was spoken or heard in this pass.
- **No real Sort It round, no real calibration, no face-started round.** The live round was only the
  scripted scene.
- **The Jetson is behind `main`** (about #226): no Polish round 2, no grounded web answers, no
  Learning lab, no `/runtime/stats`. Its long answers take 17 to 37 s and its gateway restarted
  several times this evening (defect 4).
- **The Jetson has one camera reporting, degraded,** no Circle, no focus block, no night in memory.
  The coach, Circle, overnight and evening-review beats cannot be shown on it as it stands.
- **The evening review** exists only on a log with a closed day (verified on the scripted-time seed).
- **The teach loop end to end** (approve export → label → train → gate → swap) was not run; the
  Learning lab shows a labelled sample, and the README's first learned task is still "pending".
- **Sleep from a watch** is covered by tests on a synthetic fixture only; the app runs on the
  empty-desk estimate.
- **Not exercised by anything here:** a real calendar write, a real GroupMe or ntfy message, a Neon
  round trip, the wipe itself, the group brain (#229), and the phone as an installed PWA.
- **The walkthrough checks what the endpoints return, not what the screen draws.** The screens were
  read once, from headless Chrome at 500 × 1800 and 1440 × 900, on the seeded laptop only.
