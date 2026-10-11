# Atlas film v4.7: plan and tracking document

Status: **PLAN, waiting for Dom's approval. Nothing is rendered until it is approved.**
Branch `ctx/demo/video-hyperframes`. Owner: Dom (Notion task D12).
v4.1 (Sun Oct 11): Dom's answers to v4 folded in; NextEra scene added; full order with every spec beat.
v4.2: the friend is Ilan (not Maya); the meals goal set in the morning and called back at dinner; "where are my keys?" folded
into the morning; the stake is driving; under 5:00; Atlas's voice as proposed; NextEra written as a story.
v4.3: NextEra uses our Inspection page (`/spot`) and its simulations; a new NextEra spec (the replayable event log,
answers computed in code); the operator is Xander. The full script is in `video/hf/SCRIPT.md`; **its wording wins** where this plan's
beat notes are longer (the script was tightened to fit 5:00).
v4.4: voice casting, time budget per beat, build order with milestones, the person-fix status, and the Notion
handoff (sections 9 to 13).
v4.5: two repo updates folded in (#381 outbound blur gate, #382 the real L16 run).
v4.6: **Hemdutt's Notion notes (Sun 3:21 AM, agreed by him) folded in**; section 14 maps each one. The Encompass
data-capture story is named in beat 3. Section 10's timing is corrected: **the film runs over 5:00, and Dom picks the
cuts.** **Waiting on Hemdutt's approval** (Hemdutt ↔ Dominick, Q20).
v4.7: Hemdutt approved v4.6 in substance (Q20, Sun 4:04 AM) with a cut rule: cut from the demo action and the
narrator's wording, never a spec scene or a sponsor-track clause; beat 13 whole; the AMD numbers on screen. Cuts a, b,
e, g taken, plus wording trims in beats 0 to 3, 5, 11, 12, 14 and 16 (`SCRIPT.md` is the wording). **Measured with
the real voices** (edge-tts, every line synthesised, 1.2x; gaps 0.3 s, tail 0.7 s, pad 0.3 s; action mostly under
speech): **5:28**. Speech alone is 4:26. The last ~35 s needs Dom: see section 10a.

Every fact on screen or in a line comes from the repo, with the source next to it, or Dom has said so explicitly
(section 6 lists those). Tick a box only when it has been built *and* checked in a frame.

---

## 1. How the film works

1. **Demo, then spec, then demo, then spec.** Demo beats show the person and Atlas living a day together. Spec beats
   stay **as they are in the current film** (intro, the model, the learning lab, on the device). Only small new spec
   beats are added where a new demo needs one.
2. **In demo beats the product talks, not the narrator.** Atlas speaks to the person, the person answers, no
   voiceover.
3. **Atlas speaks the way Hemdutt's natural-speech rule says** (see 1a).
4. **Atlas's apps on screen.** A window in the top-right corner shows what Atlas is doing as it happens, built from
   the website's real components (`web/`). It works like a screen recording of the app, as part of the video.
5. **Four voices, all different** (Dom: the human must not sound like the narrator):
   - **Narrator:** edge-tts Andrew (the current one).
   - **Atlas:** the box's own voice if I can run it here (Piper `en_US-ryan-medium`, pitched down as on the box,
     Xander's #348). Otherwise a distinct edge-tts voice.
   - **Hemdutt (the person):** a third voice.
   - **Ilan and Dominick:** two more (and the operator and trainee in NextEra).
   - Captions name the speaker.
6. **The person is the star.** Colour, hair, a face with expressions, real movement (section 4).
7. **No plain mistakes.** Every set is checked from two angles before a scene is done.

### 1a. Natural speech: Hemdutt's rule (#368, `contexts/day/domain/ear.py`, and his Notion notes, Sun 3:21 AM)

The owner's rule at the top of `ear.py`:

> "Atlas talks like a person about its work for you. It answers what was asked in one or two short plain sentences,
> then at most a brief offer. It never reads records out: no lists, no '10:40 AM to 11:30 AM' ranges, no ids, source
> tags, brackets, camera or zone ids."

- **Times said as people say them:** "9:20", "noon", "2", "Tuesday night".
- **At most three things, then "Want the rest?", only for casual questions** ("what's on today?"). When the person
  asks Atlas to walk through the day, as in the morning brief, Atlas walks through the **whole** day in the human
  voice and ends with an offer about the plan (Hemdutt; the product is being corrected to match on
  `ctx/agent/natural-speech-2`). So beat 2's full brief is right as written.
- **No ranges.** So "judging runs noon to two" becomes "judging's at noon".
- **Course codes said as a course:** "your algorithms writeup", not "COP4533".

Checked on Notion (v4.6): there is no separate natural-speech page; his notes above are the addition. **[x]**

---

## 2. The full order (demo and spec)

**Hard cap 5:00. As written, v4.6 runs about 6:30 (section 10), so Dom picks the cuts.** The seconds below are
speech plus action. **D** = demo, **S** = spec.

| # | Beat | Type | ≈ s | Spec unchanged? | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | The problem | narrator | 14 | new | [ ] |
| 1 | This is Atlas, and what's inside | S | 16 | **as now** (intro: assemble, explode, x-ray) | [ ] |
| 2 | **Wake up, good morning, the brief, the meals goal, lock the plan, where are my keys** | **D** | 70 | new | [ ] |
| 3 | What fed the brief, and what the camera captures (Encompass) | S | 16 | new, short | [ ] |
| 4 | The model on the box (48 of 48) | S | 8 | **as now** ("It learns"), wording fix (6.E), 48 of 48 | [ ] |
| 5 | **At the desk: you look tired, the check-in** | **D** | 45 | new | [ ] |
| 6 | Faces: expressions on the device; clear on your own screen, blurred when saved or sent | S | 14 | new, short | [ ] |
| 7 | **Kitchen: the can, the phone, the count, the nudge** | **D** | 30 | rebuild | [ ] |
| 8 | **Sort It round** | **D** | 15 | new | [ ] |
| 9 | What Sort It runs on, the bin rules and sources, the look guide | S | 18 | new, short | [ ] |
| 10 | The learning lab, then Hugging Face and the live swap | S | 21 | as now plus one line | [ ] |
| 11 | On the device: cable pull, voice offline, your question first, ledger | S | 14 | as now plus one clause | [ ] |
| 12 | **Friends: the league, three calendars agree, dinner with Ilan (meals goal callback)** | **D** | 55 | rebuild | [ ] |
| 13 | How the boxes talk (Neon), boxes recognise each other, many groups | S | 18 | rebuild (6.C) | [ ] |
| 14 | Same box, other lives (more) | S | 16 | rebuild; the Day Run line is gone | [ ] |
| 15 | Same engine, different data (into the plant) | S | 22 | new (Hemdutt's version) | [ ] |
| 16 | **NextEra: an operator's shift with SPOT** | **D** | 45 | new | [ ] |
| 17 | End: "It sees your day, learns you, and keeps you to itself." | narrator | 6 | new line (Hemdutt) | [ ] |

---

## 3. Beat by beat

### 0. The problem (narrator, about 14 s)
- **Visual:** the person's day as scattered app windows (calendar, chat, notes, a sleep app, a map) floating in the
  dorm, not talking to each other. They pull together into the box on its shelf.
- **N:** "Your day lives in a dozen apps: classes, deadlines, sleep, friends. None of them sees how your day actually
  goes, and the assistants that could help send your life to someone else's cloud."
- **N:** "This is Atlas: a personal AI computer that sits on your desk, sees your day, and keeps it on the device."

### 1. This is Atlas, and what's inside (spec, as now)
- **Visual:** the current intro: assemble, logo splash, explode, x-ray, labelled parts.
- **N:** the current lines, ending "...an agent that acts on your behalf, like an assistant."

### 2. Wake up, good morning, the brief, lock the plan (demo, about 55 s)
**Set:** the dorm. Atlas on the shelf at eye height, aimed at the chair, the bed in its view.

**Action and lines**
1. The person lies in bed, then sits up, sleepy-faced.
2. **P** (stretching, to the room): "Good morning."
3. Atlas wakes: the lens glows and the LED goes blue.
   **A:** "Good morning. You slept about seven and a half hours, right around your usual, and over your six-hour
   goal. Nice work."
   - Sources: synthetic watch 7 h 28 (`fixtures/demo/fakes.yaml:43`); your usual is the median of your own nights,
     7.4 h (`contexts/demo/history.py:66-69`, `checkin.py:44-46`); goal ≥ 6 h (`config/habits.yaml:32`).
4. The person stands, picks up the phone from the nightstand, walks to the desk and sits.
   **A** (the brief, in Hemdutt's style):
   - "Big day. First is your mentor check-in at 9:20 in Engineering II. It's a ten-minute walk, so leave by ten
     past."
   - "Submissions close at 11, and judging's at noon."
   - "Maya's study session is at the library at 2."
   - "This week, your algorithms writeup is due Tuesday night." (the greedy quiz is cut, Hemdutt)
   - "If you want a break, Sam and Riley are going to board game night at the Union at 8." (Andria's quiz review is
     cut, Hemdutt; board game night stays because it pays off in beat 12)
   - "Does this look good? I can lock this plan, or would you like something changed?"
5. **P:** "Looks good. Lock it. And today, help me actually eat two real meals."
   **A:** "Done. Two meals is your goal today, and I'll keep track. Tap to approve the plan." The person taps
   Approve on the phone. **A:** "Locked."
   (Meals is a real habit, target 2, `config/habits.yaml:39-48`. It is called back at dinner in beat 12.)
6. **P** (patting pockets, getting up): "Atlas, where are my keys?"
   **A:** "On your desk, behind the mug. I saw you put them there last night." In the room a soft red ring lands on
   the keys, and the person picks them up.
   - App window: the Ask card with the blurred-free desk frame and the keys ringed, then the room view with the pin
     "Keys here" (the room twin's look, `web/components/map/RoomTwin.tsx`).
   - Source: the keys question and the red dot ("Show me in the room") are the first beat of the judges' demo
     (`docs/demo-run-sheet.md:911-922`). Answered in 3.1 s on the box (`:109`), and the time isn't said.
- Sources:
  - Calendar: `fixtures/demo/night.yaml:52-59`, milestone `:34-37`, quiz `:38-41`.
  - Optional: `fixtures/demo/nearby.yaml:31-48`, friends' chat `fixtures/groupme/friends_chat.yaml`.
  - Walk: 10 min, 680 m (`docs/demo-run-sheet.md:304-305`).
  - Approval: `narration.py:665-666`.

**App window, top right** (tabs switch as Atlas talks)
1. **Today:** the day ribbon. Calendar blocks are solid; Atlas's suggestions are dashed.
2. **Route:** the real UCF map, dorm → Engineering II, "Leave by 9:10", 680 m.
3. **Today's Plan** (was Day Run, #338; its line is "Try your day before you live it."): the day on fast-forward on
   the campus map, the figure walking it.
4. **This week:** writeup Tue 11:59 PM.
5. **Optional:** board game night, with friends' names.
6. **Approve:** the approval card, plus "Goal today: 2 meals", then "Locked".
7. **Ask:** "Where are my keys?", the frame with the keys ringed, then the room view with the pin.

### 3. What fed the brief (spec, about 10 s)
- **Visual:** labelled particles fly into the box: Apple Watch, Outlook and Canvas calendars, GroupMe, UCF Events,
  OpenStreetMap. One line goes out to the phone: "Approve?".
- **N:** the script's first line (the brief's sources, under three seconds, "nothing is booked until you say yes"),
  then **Encompass:** "And what the camera sees becomes events, then moments it remembers, then answers, and, with
  your OK, the examples it learns from."
- **Visual (Encompass):** a counter inside the box, frames, then events, then episodes, then answers, from the box's
  own totals (`python -m contexts.demo totals`; Saturday's run: about 10,700 frames, 9,738 observations, 19 episodes,
  55 answers; refresh on Sunday before the render).
- Sources: `config/connectors.yaml`; "Brief me" 2.7 s (`docs/demo-run-sheet.md:109-111`); the totals as above.

### 4. The model on the box (spec, as now)
- **Visual and lines:** the current ones (the Qwen x-ray, Fast / Accurate / Private, **48 / 48** per Hemdutt,
  `python -m contexts.agent.eval`, "Atlas learns the tool").
- **Wording fix:** "A four-billion-parameter Qwen runs right here" instead of "It is not small" (6.E).

### 5. At the desk: you look tired, the check-in (demo, about 45 s)
**Action**
- Afternoon. The person has been at the desk a long time, typing, three cans by the keyboard, the face strained.
- Atlas's lens wakes. The order follows Dom: the face first, then sitting, then the deadline and the texts, then
  "but first".

**Lines**
1. **A:** "You look tired. That's the same look you had Tuesday afternoon, when you'd gone two hours without moving."
   This draws on expressions: opt-in, read on the device, compared with your own past looks.
2. **A:** "You've been sitting for an hour and forty-five minutes."
3. **A:** "I know your algorithms writeup is due Tuesday. Maya just texted the class chat that the professor said half
   the grade is the runtime analysis, and Jordan says weeks five and six cover it."
4. **A:** "I've pulled together notes from your group chat and r/ucf, plus a couple of articles on knapsack and
   interval scheduling. They're on your screen when you're ready."
5. **A:** "But first: that's your third energy drink today, and your limit is two. I haven't seen you drink water in
   over an hour and a half, and your watch says your resting heart rate is up. Go fill your bottle and walk for a few
   minutes. I'll hold the notes."
6. **P:** "Yeah, okay." They stand, stretch, and pick up the bottle and the empty can. This leads into the kitchen.

- Sources:
  - The check-in demo: 105 min sitting, 100 min without a drink, 21 min strained (`contexts/demo/checkin.py:53`).
  - Expressions compared with your own (`contexts/day/domain/coach.py:29`, `checkin.py:49-51`).
  - Energy drink limit 2 (`config/habits.yaml:54-60`); resting HR (`checkin.py:13-14`).
  - Notes with citations (`contexts/agent/application/planner.py:1-10`).
  - GroupMe: Jordan's line (`fixtures/groupme/class_chat.yaml`). Maya's "professor said" message is new (6.A).
  - "Tuesday afternoon, two hours without moving" is the **target** line (Hemdutt: keep it). It lands when his stage 1
    is merged (Atlas remembers an expression with what was going on around it, and calls back to it). Until then the
    script speaks the fallback "the look you get after a long stretch without moving" (6.I).

**App window, top right**
1. **Check-in card:** heart-rate line, chips "Looked tired · like Tue", "Sat 1 h 45", "No water 100 min",
   "3rd can · limit 2".
2. **GroupMe:** Maya's professor message and Jordan's reply, as chat bubbles.
3. **Notes:** a compiled outline with source tags (GroupMe, r/ucf thread, two articles).

### 6. Faces: expressions on the device; clear on your own screen, blurred when saved or sent (spec, about 14 s)
- **Visual:**
  - The person's face, clear, on his own phone's live view, with a small "looked tired · 30 s window" tag that stays
    inside the privacy shell.
  - Beside it, the saved frame of the same moment, face blurred.
  - A frame crossing the shell's edge, with the face blurring as it crosses.
- **N:** "If you turn it on, Atlas reads your expressions on the device, to coach you and to train your own model.
  You see yourself clear on your own screen. Saved pictures are blurred, and so is anything that leaves." (Hemdutt's
  exact wording.)
- Sources: Hemdutt's locked design (Dominick ↔ Team, Sun 2:47 AM), step 1 merged as #381.
- "Train your own model" lands with Hemdutt's stage 2 (training on expressions); keep the line (Hemdutt).

### 7. Kitchen: the can, the phone, the count, the nudge (demo, about 30 s)
**Action**
- The person walks in holding the phone, the empty can and the bottle, fills the bottle, and heads for the grey trash
  bin. The wall camera sees it.

**Lines**
1. **A** (the kitchen Atlas): "Hold on: cans go in recycling at UCF. The blue one."
   - The phone screen and the top-right window show Sort It: "Can goes in recycling", the blue bin highlighted, the
     route from "You". The person turns and drops it in the blue bin.
2. **A:** "Right bin. That's five of seven this week, seventy-one percent, and your goal's eighty. Four more and
   you're there."
- Sources:
  - The rule (`config/sorting.yaml:65`).
  - 4 of 6 = 67%, goal 80%, "The next 4…" (`docs/demo-run-sheet.md:560-564`); 5 of 7 is the same arithmetic after
    this throw.

**App window, top right:** the Sort It route card (`web/app/play/page.tsx`), then the Impact ring 67 → 71% with the
goal tick.

### 8. Sort It round (demo, about 15 s)
- **A:** "Want a quick round of Sort It? Hold up the next thing."
- The person holds up a cup for two seconds. The phone shows "Cup goes in trash" and the route. They walk it there.
- "Right bin · 6.0 s · 150 points" (`docs/demo-run-sheet.md:521-525`).
- **App window:** the live round with route, timer and score.

### 9. What Sort It runs on (spec, about 10 s)
- **Visual:** the bin camera view with boxes, track ids, bin zones, and a grid route bending round the person.
- **Visual, added:** the look guide: a dot follows his eyes across the room view, tagged "frames stay in the browser".
- **N:** the script's two lines: the detector, the tracking and the walk; **OneEthos:** "The bin rules are UCF's own,
  and every impact number cites its source."; **eye tracker:** "The look guide follows your eyes to point the way, and
  those frames never leave the browser."
- Sources: `config/models.yaml:22-25`, `README.md:84`, `contexts/play/room/grid.py`, `homography.py`;
  `config/sorting.yaml`, `config/impact.yaml`; the look guide #290, #297, #327.

### 10. The learning lab (spec, as now)
- Two AMD Strix Halo pods: your trainer, and the network's trainer with consent. The test gate. Hugging Face.
- Dom confirms the group and network learning is built or being built (6.C), so the line stands.
- **Added line (Hugging Face):** "A test decides if the new model ships. When one passes, it goes up on Hugging Face,
  and the box checks it, swaps it in live, and rolls back if anything fails." (L6, #367, `python -m lab deploy`.) It
  is worded as the mechanism, not as "our model shipped", because the real L16 gate **rejected** its candidate
  (candidate 1.0 = current 1.0 on 24 held-out items; `fixtures/demo/l16-gate-scores.json`).
- **AMD numbers for the overlay** (Hemdutt's candidates, from `lab/refine/metrics.jsonl`; **shown only after Ilan
  confirms**, Ilan Q10):
  - The teacher on a Strix Halo pod: about 11 tok/s for one request to about 51 tok/s batched, at about 96 W; energy
    per token from 8.8 J to 1.9 J (session `bench-20261011T044643Z-6240`, lines 188-209).
  - Idle: the pod about 10 W (line 97), the Jetson 12.6 W (line 64).
  - L16 (#310): 18 steps in about 12 minutes on pod B, loss 0.127 to 0.010.
- 6.E (fine-tune numbers, the Hugging Face repo name) still needs Ilan's confirmation before numbers are shown.

### 11. On the device: cable pull, ledger (spec, as now)
- Everything live on the Jetson. Pull the cable: it keeps seeing, remembering, alerting and answering. Ledger: 13
  allowed, 13 blocked.
- **Face line:** none here; beat 6 carries it (Hemdutt's wording).
- **Added (main track):** the voice runs on the box, offline (#275, #348), and a question goes ahead of the cameras
  (GPU scheduler, #361). See the script's wording.

### 12. Friends: the league, three calendars agree, dinner with Ilan (demo, about 55 s)
**Cast:** Hemdutt in the dorm with his Atlas; **Ilan** at '63 South on his phone; Dominick at his desk on his
computer. Three panes side by side, then two. Maya stays only as the classmate in the class GroupMe.

**The story, simply**
1. **Sunday evening, the league result lands.** All three Atlases say it at once, each to its own person, over each
   other:
   - "This week's league is in."
   - "Dominick hit the most goals, so he picks what you do tonight."
   - "Hemdutt, you had the roughest week, so you're driving."
   - Window: the league table, with each person's percent and which goals they hit.
2. **Dominick picks.** He tells his Atlas, "Board game night at the Union, eight o'clock." His Atlas asks the other two
   boxes.
3. **The calendars sync, shown in the top-right window.**
   - Three calendars side by side, one per person. Each box looks only at its own: the others see grey "busy" blocks,
     never what they are.
   - One-word answers travel between the boxes as small bubbles. Hemdutt's box: "Free." Ilan's box: "Busy at 8. How
     about 8:30?" Dominick's box: "8:30 works."
   - "Agreed, three of three." The same 8:30 block drops onto all three calendars at the same moment.
4. **Before that, dinner.** Ilan's phone shares his place: "Ilan is at '63 South."
   **A** (to Hemdutt): "Ilan's at '63 South right now, and you're free until 8:30. Dinner with him? That would be
   your second real meal today, the goal you asked me for this morning."
   **P:** "Yeah, tell him I'm coming."
   **A:** "Told him. Route's open: a few minutes' walk, then you're driving everyone to the Union."
5. **The callback.** A short beat at '63 South, Hemdutt and Ilan at a table. Hemdutt's phone buzzes; Atlas asks
   "Did you eat with Ilan?" **P:** "Yep." **A:** "Logged. Two meals today: goal met."
   - Window: the meals goal ticks from 1 of 2 to 2 of 2. The league row updates.

- Sources:
  - Negotiation format and one-word replies (`docs/demo-run-sheet.md:476-479`, `contexts/circle/README.md:3-8`).
  - Shared place with an expiry (`contexts/demo/circle.py:141`).
  - League scoring (`contexts/play/league/domain/scoring.py`, `config/league.yaml`).
  - Meals target 2 (`config/habits.yaml:39-48`).
  - Board game night (`fixtures/demo/nearby.yaml`).
  - Walk time to '63 South to be computed on the real map.
  - **Depends on** the optional-events PR (#311, Ilan1's side) for the dinner suggestion (Hemdutt).
  - Stake: driving (Dom); the app says the worst week buys the next group meal.

### 13. How the boxes talk, the trusted group, many groups (spec, about 18 s)
- **Visual:**
  - Three shells; only one-word replies cross between them.
  - Inside the group the boxes **recognise each other** (a friend's name appears over them), and a soft link of
    learning grows between their models. No faces unblur: between friends' boxes only text and face numbers travel
    (the sealed, opt-in face share, #258), never pictures (Hemdutt).
  - A small Neon database in the middle that holds only what each person chose to share.
  - Zoom out to many small groups across campus, each with its own network.
- **N:**
  - "Each box checks its own calendar and answers in one word: free, busy, or another time."
  - **Neon:** "They meet in a Neon database that holds only what each person chose to share." (Presence, shared habit
    counts, league, RSVPs, sealed face-share messages; it also works over 443 when the database port is blocked, #355.)
  - "In a group you trust, friends can let their boxes recognise each other, and learn from each other's examples,
    and every group runs its own network." (Hemdutt's wording.)
- Dom: this is built or being built (6.C).

### 14. Same box, other lives (spec, about 22 s; the Day Run line is gone)
- **Visual:** three short vignettes, each with its own app window.
  - **Night-shift worker:** shift 10 PM-6 AM, sleep target 6.5 h, energy-drink limit 1, the leave checklist (keys,
    badge, phone, charger).
  - **Family home:** each member's own habits; "Leo: bins out", "Ava: science fair poster".
  - **Plant operator:** hands over to NextEra.
- **N:**
  - "It isn't just for students. The same box runs a night-shift worker's upside-down day, a family's homework and
    chores, a plant floor."
  - "It's twelve watts at idle, it answers offline, and your data never has to leave."
- Sources: `config/profiles/*`, `habits.yaml`; 12.62 W idle (`demo_numbers_2026-10-11.md:47`).

### 15. Same engine, different data (spec, about 22 s; Hemdutt's version, replaces the v4.5 version kept below)
- **Why:** the old version repeated "data never leaves" right after beat 14. Hemdutt: make it "same engine, different
  data", calling back to the film.
- **Visual:**
  - The morning brief's cards become a shift triage.
  - The red pointer that found the keys walks the plant floor plan over paths SPOT has actually walked
    (`contexts/spot/domain/walkable.py`, `fixtures/spot/floor_walkable.json`).
  - Numbers pop up, each with its citation.
  - The Learning lab tab shows the SPOT test: answers must cite the right rows and contain every expected number (the
    SPOT scorer, L26 #315; the gate with its own output length, L28 #363).
  - The Inspection screen and Shift Run.
- **N:** "Point the same box at a plant and the same pieces line up. The brief becomes a shift triage. The pointer that
  found your keys walks a floor plan, over paths SPOT has actually walked. Every number is computed from the inspection
  rows and cites them. And the lab re-trains the model on the plant's own data and tests it before it's allowed on
  the box."

**Superseded v4.5 version, kept for reference:**
The privacy points (on the box, cited, approved) are already made in beats 6, 11 and 13, so this spec shows what is
new and true and leads into the plant.
- **Visual:** the dorm's day as a ribbon of events (Hemdutt waking, the brief, the can) running into the box as an
  append-only log. The camera rewinds along it. The ribbon swaps to SPOT's mission log (missions, checks, readings,
  alarms), the room dissolves into the plant's floor plan, and the same box sits on a control-room console.
- **N:** "Under everything Atlas does is one log of events that it can replay, minute by minute. Swap the profile and
  the same log holds a robot's inspection rounds instead of a student's day. The findings are computed in code, with
  their error bars, and the model only puts them into words."
- Sources:
  - Append-only event log it can replay (`docs/spot.md:3-4`, AGENTS.md "the event log is the source of truth").
  - The inspection profile changes the words only (`config/profiles/inspection/profile.yaml`, `docs/spot.md:146`).
  - Findings are pure functions that return number, n, interval, sentence, caveat and evidence ids; no model call for
    the plain words or the guide (`docs/spot.md:43, 172`, `contexts/spot/application/plain.py`, `guide.py`).

### 16. NextEra: an operator's shift with SPOT (demo, about 45 s)
**Set:** a small control room at a plant. Atlas on the console. The big display and the top-right window show **our
Inspection page** (`/spot`, nav label "Inspection", `web/app/spot/page.tsx`), rebuilt from its components: "Your
shift" (`Shift.tsx`, `Triage.tsx`), the floor plan with the walking pointer (`FloorMap.tsx`, look guide), "A round,
step by step" (`Round.tsx`, `ReplayPanel.tsx`), the animated explainers (`Explainers.tsx`), and Shift Run
(`ShiftRun.tsx`). Through the window, a glimpse of the plant floor with a SPOT robot at its dock.
**Cast:** **Xander**, the operator (hi-vis vest, coffee). A trainee appears at the end (name TBD).
**Simulations shown:** the pointer walking the traced route, the round replay with SPOT moving and the battery
draining, the battery explainer, and Shift Run's fast-forward round with stars. Optionally "Use my eyes to guide
me": the pointer starts from where Xander is looking (the page's opt-in gaze guide). It echoes the expressions
thread from the morning, if there's time.

**The story, simply**
1. **Start of shift.** Xander sits down with a coffee. "Morning, Atlas. What should I check first?"
2. **Atlas gives a short list, not a dashboard.** "Three things. First, heat exchanger 2 in the pipe gallery. It
   raised an alarm, but the check that raised it failed, and it hasn't had a good check in 88 hours. Send SPOT back
   before you send anyone."
   - Window: "Your shift", three cards: Do this first / Keep an eye / Looks fine. HX-202 is at the top, with a
     "Why?" button.
3. **"Where is it?"** A red pointer drops onto HX-202 on the floor plan and walks there from the control room while
   Atlas says the way: "Turn left, take the entrance steps, and go along the west passage." Xander now knows where
   without reading a map.
4. **Atlas says what the numbers hide.** "Heads up: mission status says ninety-six percent complete, but most of
   those rounds still had a failed check inside. And checks below forty percent battery fail about three times as
   often."
   - Window: the trust line, then the short battery animation (35% of checks fail on low battery, 12% otherwise).
5. **Xander checks it directly.** "Show me round 17." The round plays back like a replay: SPOT's dot moves along the
   plan, each check pops in (tick, cross, alarm), the battery bar drains, and the failed check is marked where it
   happened.
6. **Xander acts, with one yes.** "Send SPOT back to it." An approval card: "Send SPOT back to heat exchanger 2?" Yes.
   **A:** "Done. A work order was written on this box. Nothing left the building."
7. **Anyone can learn this job.** The trainee steps in. **A:** "Want to plan the next round?" In **Shift Run**
   he drags the stops into an order and presses Run. The round plays out fast, the battery matters, and he gets two
   stars out of three against Atlas's plan, with one line on why ("check the far exchangers before the battery
   drops").
8. **Handoff.** End of shift: Xander taps one button. "Shift handed off." Atlas's note for the next operator is ready.

**What the judges should take away:** it answers the track's questions (what happened, where, why it matters, what
needs attention, is it getting better), in plain words, offline, with every number traceable to the rows.

- Sources:
  - `docs/spot.md`: the brief table; "Mission status says 96%; 81% of those missions hide a failed action"
    (`:176, :267`).
  - HX-202, no good check for 88 h, top of triage (`:189-192`).
  - Battery 35.2% vs 12.1% (`:180`).
  - The look guide's walking directions; work order on the box (`:270-272`).
  - Shift Run with stars against Atlas's plan (`contexts/spot/domain/shiftrun.py`); the shift handoff is one button.
- Honest wording kept: the round is a replay of the kit's recorded synthetic data, not a live robot; the route is
  "about" (traced from the kit's picture); "we do not claim a cause".
- Names: Xander is the operator (Dom). Trainee name TBD.

### 17. End (as now)
- The website's logo and splash, back on the shelf in the dorm.
- **N:** "Atlas. It sees your day, learns you, and keeps you to itself." (Hemdutt: it echoes the opening. Dom's
  earlier line, "Atlas. The personal AI computer. It learns you, and keeps you to itself.", is the fallback if Dom
  prefers it.)

---

## 4. Fixes to the last round

- [ ] **4.1 Chair backward.** Rotation `-π/2` → `+π/2`. Check from overhead and from the side: the backrest is
  behind the person.
- [ ] **4.2 Colour.** Shirt, trousers, shoes and skin, coloured per bone (shader), a different palette per person
  (Hemdutt, Ilan, Dominick, the operator with hi-vis, the trainee).
- [ ] **4.3 Hair.** Quaternius hairstyles (CC0, already downloaded), rigged to the head bone. Fallback: a cap mesh.
- [ ] **4.4 Face with expressions.** Eyes, brows and mouth on the head: sleepy, neutral, smile, tired or strained,
  talking (the mouth moves with the person's voice line).
- [ ] **4.5 More movement.**
  - Sitting_Talking_Loop when speaking.
  - Breathing and glances.
  - Typing (forearm override).
  - Holding a phone (arm override plus the phone prop).
  - Picking things up.
- [ ] **4.6 Lying in bed and sitting up.** Built from the sitting pose pivoted at the hip, then Sitting_Exit. Check
  for no clipping.
- [ ] **4.7 Atlas sees the bed and the chair.** A camera's-eye still.
- [ ] 4.8 Keep: shelf at eye height, no monitor webcam, the real UCF map, the website look, faster renders.

## 5. New things to build

- [ ] **5.1 App window.** Top right, website style, tabs and a cursor, reused in every demo. Panes: Today, Route, Today's
  Plan, Week, Optional, Approve, Check-in, GroupMe, Notes, Sort It, Impact ring, League, Circle conversation,
  `/spot/` (Your shift, floor plan pointer, round replay, Shift Run).
- [ ] **5.2 Dialogue.** `film.yaml` gets `speaker:` per line and one voice per speaker; captions name the speaker; the
  lens pulses while Atlas talks; overlapping lines for the league.
- [ ] **5.3 Props:** phone (its screen mirrors the window), cans, water bottle, cup, nightstand, coffee.
- [ ] **5.4 Sets:** '63 South (small dining hall); Dominick's desk; the control room (console, big display); split
  screen framing.
- [ ] **5.5** Problem-statement visual (scattered windows into the box).
- [ ] **5.6** Particles for "what fed the brief".
- [ ] **5.7** Sort It camera overlay (boxes, ids, zones, grid route).
- [ ] **5.8** Trusted-group visual (boxes recognise each other with a name appearing, the Neon hub, model links,
  many groups zoom-out). No faces unblur.
- [ ] **5.10** Encompass counter for beat 3 (frames, events, episodes, answers), from the box's own totals.
- [ ] **5.9** A small SPOT figure for the plant spec (a stylised quadruped, or the icon from `/spot/`).

## 6. Decisions (Dom, Sun Oct 11) and what they mean

- **6.A Professor reminder: a friend relays it in GroupMe.**
  - New chat line from Maya: "prof said in class half the milestone grade is the runtime analysis, we should go over
    weeks 5 and 6 together".
  - I'll also add it to `fixtures/groupme/class_chat.yaml`, so the product shows it too. That's an additive fixture
    line, with an FYI to the fixture's owner.
- **6.B Expressions: coaching and training.** The film says both. The training part isn't in the code yet (expression
  history is excluded from face share today); the product team needs to land it.
- **6.C Friends' models learning from each other, many groups, unblurring faces between friends (with consent).**
  Dom says this is built or being built, so the film shows it as working. **v4.6 (Hemdutt):** nothing unblurs between
  friends; boxes recognise each other through the sealed, opt-in face share (#258, Hemdutt); group learning is the
  network pool (L10, #366, Ilan). Who owns the many-groups part is still asked (Hemdutt Q18).
- **6.D "The route helps meet a goal"** (Dom asked what I meant).
  - Your earlier note had Atlas say the route "can also help meet this goal". There's no walking or steps goal in the
    app, so that line would claim something that doesn't exist.
  - Fix: tie it to a goal that does exist. Lunch at '63 South is "your second real meal today, which keeps your meals
    goal on track". Meals is a real habit and a league target.
- **6.E Wording and numbers to confirm.**
  - "A four-billion-parameter Qwen" replaces "It is not small".
  - The fine-tune numbers and the public Hugging Face repo name need Ilan's confirmation before they are shown.
    **Update (#382, Ilan, on main):** the real L16 run's gate scores and package manifest are now demo fixtures
    (`fixtures/demo/l16-gate-scores.json`, `l16-manifest.json`, `lab/docs/lab-push.md`). Ilan still to say which
    numbers the film may show.
- **6.F Exam.** There's no exam in the demo day; the film uses the writeup and the greedy quiz.
- **6.G Watch data.** Synthetic numbers only (`config/connectors.yaml:82-85`).
- **6.H Faces (v4.6, Hemdutt's locked design): blur on everything saved and everything that leaves; clear only on
  your own live screen.** The film uses his wording (beat 6). The notes below are history. **Update (#381, Hemdutt, on main):** an outbound gate now blurs
  every face in any picture that leaves the box *whatever the local switch says*, and sends nothing if it fails. So
  "blurred the moment a picture leaves" is true in the product. What remains is the default of the local
  "Blur faces before saving" switch (still on). The film's "your face stays clear on the box" needs that default
  off, or the line becomes "you choose whether faces are blurred on the box; anything that leaves is always
  blurred".
  - The film says "blurred only when a picture leaves the box".
  - Today's code also blurs keyframes before storing them (`contexts/perception/infrastructure/face_blur.py`,
    `contexts/vault/application/policy.py`). Making the product match is a policy change for the perception owner.
    **[ ] raise on Notion.**
- **6.I The past-look line** ("the same look you had Tuesday, after two hours without moving").
  - The coach compares expressions with your own history.
  - **v4.6 (Hemdutt): keep the Tuesday line as the target.** His team builds it in two stages: (1) Atlas remembers an
    expression with what was going on around it and calls back to it; (2) training on it. Until stage 1 is merged,
    the script speaks the reworded fallback ("the look you get after a long stretch without moving").

## 7. Questions still open

- ~~Q1 Length~~ Dom: under 5:00 (this plan: about 4:50).
- ~~Q2 Atlas's voice~~ Dom: as proposed.
- ~~Q3 "It remembers"~~ Dom: fold "where are my keys?" into the morning, after the meals goal (done in beat 2). The
  17 × 23 and the UCF score are cut.
- ~~Q4 Stake~~ Dom: driving.
- ~~Q5 Names~~ Dom: Hemdutt, **Ilan** (instead of Maya) and Dominick. Maya stays only as a classmate in the GroupMe.
- ~~Q6~~ Operator: Xander (Dom). **Trainee name still open.**

## 8. QA before any render

- [ ] `npx hyperframes check` clean.
- [ ] Two-angle stills of every set (dorm chair, bed, shelf, desk, kitchen, '63 South, Dominick's desk, control room).
- [ ] A camera's-eye still from Atlas: bed and chair in view.
- [ ] Contact sheet of every beat; captions name the speaker; no card covers a face.
- [ ] Every number ticked against its source.
- [ ] One-scene drafts for Dom first (`python video/scripts/film_render.py <scene>`); full render only after approval.
- [ ] `pytest -q contexts/demo/tests/test_video.py` green, commit, push, Build Log and D12 on Notion.

---

## 9. Voice casting (all checked to exist in edge-tts on this machine)

| Speaker | Voice | Notes |
| --- | --- | --- |
| Narrator | `en-US-AndrewMultilingualNeural` | unchanged; rate +20% (1.2x) |
| Atlas | Piper `en_US-ryan-medium`, pitch 0.88, speed 0.95 | the box's own voice (Xander, #348). `piper-tts` is not installed here yet; fallback `en-US-GuyNeural` |
| Hemdutt | `en-US-BrianMultilingualNeural` | younger, warmer; clearly not the narrator |
| Ilan | `en-US-ChristopherNeural` | |
| Dominick | `en-US-EricNeural` | |
| Xander (operator) | `en-US-RogerNeural` | |
| Trainee | `en-US-SteffanNeural` | no spoken line in the current script, so optional |

`film.yaml` gets `speaker:` on every line; `film_build.py` picks the voice per speaker and caches by
speaker+text. The league beat has three Atlas lines that overlap by about half a second each.

## 10. Time budget per beat (v4.6, recounted from `SCRIPT.md`)

**Correction:** the v4.4 table said "≈ 4:48", but its own rows added up to 348 s (**5:48**). v4.6 recounts the speech
from the script's words at about 3.2 words a second (edge-tts at +20%). The action-only seconds are kept from v4.4.

| # | Beat | Speech | Action-only | Total |
| --- | --- | --- | --- | --- |
| 0 | Problem | 14 | 3 | 17 |
| 1 | What's inside | 9 | 5 | 14 |
| 2 | Good morning, brief, meals, keys | 42 | 12 (sit up, walk, approve, find keys) | 54 |
| 3 | What fed the brief + Encompass | 16 | 2 | 18 |
| 4 | The model (48 of 48) | 6 | 3 | 9 |
| 5 | At the desk | 31 | 4 | 35 |
| 6 | Faces | 12 | 3 | 15 |
| 7 | Kitchen | 8 | 6 (turn, drop) | 14 |
| 8 | Sort It round | 4 | 7 (hold up, walk, score) | 11 |
| 9 | Sort It tech + bin rules + look guide | 18 | 2 | 20 |
| 10 | Lab + Hugging Face swap | 20 | 3 | 23 |
| 11 | On the device + voice + your question first | 12 | 3 | 15 |
| 12 | Friends | 31 | 8 (calendars drop, dinner cut) | 39 |
| 13 | How the boxes talk + Neon | 17 | 3 | 20 |
| 14 | Other lives | 9 | 3 | 12 |
| 15 | Same engine, different data | 21 | 2 | 23 |
| 16 | Xander's shift | 33 | 12 (pointer walk, replay, Shift Run) | 45 |
| 17 | End | 4 | 2 | 6 |
| | **Total** | **≈ 5:07** | **≈ 1:23** | **≈ 6:30** |

On top of this, `film_build.py` adds 0.4 s between lines and 1.7 s per scene (pad plus tail), up to about a minute
more where it doesn't overlap the action-only time. **As written, v4.6 is about 6:30 to 7:30. The cap is 5:00.**

### Cuts to reach 5:00 (Dom chooses; none applied)

| Option | What goes | Saves about |
| --- | --- | --- |
| a | Run action under speech: the brief while he walks to the desk, the calendars dropping under the bubbles, the pointer walking under the directions. No words lost. | 20 s |
| b | The old trim order: dinner callback becomes "Logged: goal met"; the trainee is one shot; the problem loses its second sentence | 15 s |
| c | Beat 14 folds into 15 (15 already moves to the plant) | 12 s |
| d | Beat 8 (the Sort It round) goes; beat 7 keeps Sort It's route and count | 11 s |
| e | Beat 16 drops the round 17 replay (triage, pointer, approval and Shift Run stay) | 10 s |
| f | Beat 15 keeps only Hemdutt's first and last sentences | 8 s |
| g | Beat 12 drops the three overlapping league lines (the window shows the table) | 8 s |
| h | Beat 4 folds into beat 1 ("a four-billion-parameter Qwen, forty-eight of forty-eight") | 6 s |

All of a to h together save about 1:30, which brings 6:30 to about 5:00. That's before the build's gaps, so reaching
the cap means taking nearly all of them, or Dom raising the cap. The sponsor-track clauses (sections 3 and 14) are
not on this list: Hemdutt asked that they not be lost.

### 10a. Measured, v4.7 (seconds; speech is the synthesised lines, scene adds gaps, action and tail)

| # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| speech | 9.2 | 6.7 | 40.6 | 13.9 | 7.2 | 19.6 | 10.0 | 10.7 | 3.9 | 17.8 | 18.5 | 12.9 | 21.6 | 15.3 | 6.6 | 18.1 | 28.8 | 4.2 |
| scene | 11.2 | 9.4 | 48.0 | 15.9 | 8.9 | 23.2 | 11.7 | 14.7 | 9.6 | 19.8 | 20.5 | 14.9 | 27.1 | 17.3 | 8.3 | 20.1 | 35.3 | 6.9 |

**Total 5:28.** Under Hemdutt's rule and Dom's 1.2x voice, about 35 s more has to go; the open choices are in the
D12 Notion page. The new `film.yaml` must use gap 0.3, pad 0.3, tail 0.7 (the old film used 0.4, 0.5, 1.2).

## 11. Build order and milestones (each milestone ends with stills and one-scene drafts for Dom)

1. **M1 · People and the dorm** (section 4): facing direction, chair, colours, hair, face, expressions, typing and
   phone poses, lying in bed and sitting up; a camera's-eye still from Atlas. **Dom reviews stills.**
2. **M2 · Dialogue and voices** (5.2, section 9): speakers in `film.yaml`, per-speaker voices, speaker captions, the
   lens pulse, overlaps.
3. **M3 · The app window** (5.1): one component with every pane; the morning panes first.
4. **M4 · Demo beats 2 and 5** (morning, desk) with their spec beats 3, 4 and 6. **One-scene drafts.**
5. **M5 · Kitchen and Sort It** (beats 7 to 9). **Drafts.**
6. **M6 · Friends** (beats 12 and 13): '63 South and Dominick's desk sets, split screen, calendar sync, league,
   dinner callback. **Drafts.**
7. **M7 · NextEra** (beats 15 and 16): the control room, a SPOT figure, the Inspection page panes (Your shift, the
   pointer walk, round replay, the battery explainer, Shift Run). **Drafts.**
8. **M8 · Bookends and specs** (beats 0, 1, 10, 11, 14, 17), then the timed full draft (15 fps), the QA checklist,
   and the delivery render (30 fps) **only after Dom approves**.

## 12. Status of the person fixes (M1, in progress, not committed)

- Done in code, not yet verified in a still:
  - The person was facing away from the desk the whole time: the UE-rigged mannequin faces -z in its own space. It
    is now turned to face +z, so every act's yaw is right. This was also why the chair looked backward.
  - The chair is turned to match (backrest toward the bed).
  - Clothes coloured per body region from the skin weights, with a palette per person (`LOOKS` in
    `world/sets/human.js`).
  - A soft fitted hair cap in three styles (Quaternius' textured hair cards render as patches without their
    textures).
  - A face with eyes, brows and mouth and expressions: smile, sleepy, strain, talk, plus blinking.
  - Expression state in the world (`personSmile`, `personSleepy`, `personStrain`, `personTalk`).
- Next: stills from two angles, a face close-up, fix what they show, commit.

## 13. Notion handoff (for the agent that posts it)

**Done for v4.4 (Sun 3:20 AM ET); v4.6 posted as a D12 subpage for Hemdutt's approval.**

What to post, where, and the text. This session could not reach Notion; the posting agent needs the Notion MCP.

1. **D12 task page** (find "D12" in the Task Board data source `collection://68af27b3-01c7-487d-807e-71af7d3fc536`):
   add a section "Film v4.4 plan" with a short summary (the full order table from section 2) and links to
   `video/hf/FILM_PLAN.md` and `video/hf/SCRIPT.md` on branch `ctx/demo/video-hyperframes`. Keep the task Locked
   and In progress; note "M1 (people) in progress".
2. **Hemdutt's Team Comms page** (https://app.notion.com/p/3f596ef5fec5819f8099d245fa78799d):
   - Under FYI: the plan is up for review, plus which beats involve him (he's the person in the film; the brief
     follows his natural-speech rule).
   - Under Open questions, a new Q numbered after his highest: whether his Notion page on natural speech adds anything
     beyond `contexts/day/domain/ear.py` (#368) that Atlas's lines should follow (script beat 2, the morning brief;
     beat 5, the desk check-in). Add it to "Open questions at a glance" on the Team Comms hub.
3. **Build Log** (https://app.notion.com/p/3f596ef5fec5812fa37de5ae92086234): an entry at the top, "D12: film
   v4.4 plan and script (review)", with the commit hash and the two file links. Nothing for others to change.
4. **Read and report back to Dom:** Hemdutt's natural-voice page and the NextEra track page. If either changes a line
   or a beat, say which.
5. **Questions for other teammates** (each on their own page, numbered after their highest Q):
   - **Ilan:** confirm the L16 fine-tune numbers and the public Hugging Face adapter repo name before the film shows
     them (6.E).
   - **Whoever owns perception (Xander or Hemdutt):** the film says faces are blurred only on egress, but the code
     blurs stored keyframes too (6.H). Change the product to match?
   - **Who owns the friends' group learning, multi-group and unblur-with-consent work** (6.C), so the film and the
     Build Log can point to it.
   - **The fixture owner:** Maya's new "professor said" GroupMe line (6.A) and an optional seeded Tuesday history
     entry (6.I).
6. Sign every post `🤖 Dominick's agent · <Day h:mm AM/PM ET>:` (AGENTS.md 8.1).

## 14. Hemdutt's notes (Notion, Sun 3:21 AM, on v4.2) and where each went (v4.6)

| Note | What changed | Beat |
| --- | --- | --- |
| A1 "42 of 42" is now 48 of 48 | lines, card and table | 4 |
| A2 Day Run is now "Today's Plan" | window tab 3, pane list 5.1 | 2 |
| A3 cut Andria's review and the greedy quiz; the full brief is right; 1a corrected | brief, windows 4 and 5, section 1a | 2 |
| A4 face wording | his exact line; 6.H rewritten; beat 11's face line removed (beat 6 carries it) | 6, 11 |
| A5 keep the Tuesday line and "train your own model" | Tuesday line is the target after his stage 1, fallback until then; training after stage 2 | 5, 6 |
| A6 dinner depends on #311 | dependency noted | 12 |
| A7 "recognise each other", not unblur | line, visual 5.8, 6.C | 13 |
| B beat 15 as "same engine, different data" | his narration and visual | 15 |
| C AMD numbers | overlay candidates, shown only after Ilan confirms (Ilan Q10) | 10 |
| C Hugging Face | the live swap line, worded as the mechanism (the L16 gate rejected its candidate) | 10 |
| C Neon | "they meet in a Neon database that holds only what each person chose to share" | 13 |
| C OneEthos | "the bin rules are UCF's own, and every impact number cites its source" | 9 |
| C main track | voice on the box offline; your question goes first | 11 |
| C eye tracker | the look guide line and visual | 9 |
| D last line | "Atlas. It sees your day, learns you, and keeps you to itself." (Dom's old line is the fallback) | 17 |
| E closed items | the public repo and how the demos get into the film: nothing to change here | n/a |

### Sponsor tracks the film speaks to

| Track | Where |
| --- | --- |
| Main, AI, Embedded | 1, 4, 11 (all on the Jetson; voice offline; your question first) |
| **Encompass: Data Capturing Hardware** | **3**: the camera and the watch capture the day; frames become events, then episodes, then answers, and (with your OK) training examples; the counter uses the box's own totals. Also 9 (the bin camera to counts) and 15 (inspection rows to cited findings) |
| NVIDIA | 4, 9, 11 |
| AMD | 10 |
| Hugging Face | 10 |
| Neon | 13 |
| OneEthos | 7, 9 |
| Full Sail | 2 (Today's Plan), 8, 16 (Shift Run) |
| NextEra | 15, 16 |

