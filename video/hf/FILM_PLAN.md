# Atlas film v4.1: plan and tracking document

Status: **PLAN, waiting for Dom's approval. Nothing is rendered until it is approved.**
Branch `ctx/demo/video-hyperframes`. Owner: Dom (Notion task D12).
v4.1 (Sun Oct 11): Dom's answers to v4 folded in; NextEra scene added; full order with every spec beat.

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
   - **Maya and Dominick:** two more.
   - Captions name the speaker.
6. **The person is the star.** Colour, hair, a face with expressions, real movement (section 4).
7. **No plain mistakes.** Every set is checked from two angles before a scene is done.

### 1a. Natural speech: what Hemdutt's work says, checked against the repo (#368, `contexts/day/domain/ear.py`)

I could not open Notion from this session (no Notion connector here), so I checked his merged code instead. The
owner's rule at the top of `ear.py`, with the brief written to fit it:

> "Atlas talks like a person about its work for you. It answers what was asked in one or two short plain sentences,
> then at most a brief offer. It never reads records out: no lists, no '10:40 AM to 11:30 AM' ranges, no ids, source
> tags, brackets, camera or zone ids."

- **Times said as people say them:** "9:20", "noon", "2", "Tuesday night".
- **At most three things named, then an offer:** "You've got four things today. First is ..., then .... Want the
  rest?" (#368).
- **No ranges.** So "judging runs noon to two" becomes "judging's at noon".
- **Course codes said as a course:** "your algorithms writeup", not "COP4533".

If Hemdutt's Notion page adds anything beyond `ear.py`, the brief changes to match. That check is
**[ ] still to do on Notion**.

---

## 2. The full order (demo and spec)

About 5:10 in total (the length question is still open; see Q1). **D** = demo, **S** = spec.

| # | Beat | Type | ≈ s | Spec unchanged? | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | The problem | narrator | 14 | new | [ ] |
| 1 | This is Atlas, and what's inside | S | 20 | **as now** (intro: assemble, explode, x-ray) | [ ] |
| 2 | **Wake up, good morning, the brief, lock the plan** | **D** | 55 | new | [ ] |
| 3 | What fed the brief | S | 10 | new, short | [ ] |
| 4 | The model on the box (42 of 42) | S | 18 | **as now** ("It learns"), one wording fix (6.E) | [ ] |
| 5 | **At the desk: you look tired, the check-in** | **D** | 45 | new | [ ] |
| 6 | Faces: expressions on the device, blur only on egress | S | 14 | new, short | [ ] |
| 7 | **Kitchen: the can, the phone, the count, the nudge** | **D** | 30 | rebuild | [ ] |
| 8 | **Sort It round** | **D** | 15 | new | [ ] |
| 9 | What Sort It runs on | S | 10 | new, short | [ ] |
| 10 | The learning lab | S | 22 | **as now** | [ ] |
| 11 | On the device: cable pull, ledger | S | 16 | **as now**, blur wording changed (6.H) | [ ] |
| 12 | **Friends: three people, three boxes, the league** | **D** | 55 | rebuild | [ ] |
| 13 | How the boxes talk, the trusted group, many groups | S | 18 | rebuild (6.C) | [ ] |
| 14 | Same box, other lives (more) | S | 22 | rebuild; the Day Run line is gone | [ ] |
| 15 | NextEra: why it works on a plant floor | S | 12 | new | [ ] |
| 16 | **NextEra: an operator's shift with SPOT** | **D** | 45 | new | [ ] |
| 17 | End | narrator | 6 | as now | [ ] |

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
   - "This week, your algorithms writeup is due Tuesday night, and the greedy quiz is Thursday."
   - "If you want a break, Sam and Riley are going to board game night at the Union at 8, and Andria's running a
     quiz review. Both optional."
   - "Does this look good? I can lock this plan, or would you like something changed?"
5. **P:** "Looks good. Lock it."
   **A:** "Locked. Tap to approve." The person taps Approve on the phone. **A:** "Done."
- Sources:
  - Calendar: `fixtures/demo/night.yaml:52-59`, milestone `:34-37`, quiz `:38-41`.
  - Optional: `fixtures/demo/nearby.yaml:31-48`, friends' chat `fixtures/groupme/friends_chat.yaml`.
  - Walk: 10 min, 680 m (`docs/demo-run-sheet.md:304-305`).
  - Approval: `narration.py:665-666`.

**App window, top right** (tabs switch as Atlas talks)
1. **Today:** the day ribbon. Calendar blocks are solid; Atlas's suggestions are dashed.
2. **Route:** the real UCF map, dorm → Engineering II, "Leave by 9:10", 680 m.
3. **Day Run:** the day on fast-forward on the campus map, the figure walking it.
4. **This week:** writeup Tue 11:59 PM, Quiz 5 Thu 5 PM.
5. **Optional:** board game night, quiz review, with friends' names.
6. **Approve:** the approval card, then "Locked".

### 3. What fed the brief (spec, about 10 s)
- **Visual:** labelled particles fly into the box: Apple Watch, Outlook and Canvas calendars, GroupMe, UCF Events,
  OpenStreetMap. One line goes out to the phone: "Approve?".
- **N:** "That brief came from your watch, your calendars, your group chats and the campus map. The model on the
  Jetson wrote it in under three seconds, and nothing touched your calendar until you said yes."
- Sources: `config/connectors.yaml`; "Brief me" 2.7 s (`docs/demo-run-sheet.md:109-111`).

### 4. The model on the box (spec, as now)
- **Visual and lines:** the current ones (the Qwen x-ray, Fast / Accurate / Private, 42 / 42, "Atlas learns the
  tool").
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
  - "Tuesday afternoon, two hours without moving" is the past-look comparison. It needs a matching seeded history
    entry, see 6.I.

**App window, top right**
1. **Check-in card:** heart-rate line, chips "Looked tired · like Tue", "Sat 1 h 45", "No water 100 min",
   "3rd can · limit 2".
2. **GroupMe:** Maya's professor message and Jordan's reply, as chat bubbles.
3. **Notes:** a compiled outline with source tags (GroupMe, r/ucf thread, two articles).

### 6. Faces: expressions on the device, blur only on egress (spec, about 14 s)
- **Visual:**
  - The person's face inside the privacy shell with a small "looked tired · 30 s window" tag that stays inside.
  - A frame crossing the shell's edge, with the face blurring exactly as it crosses.
- **N:** "Atlas reads your expressions only if you turn it on, on the device, in thirty-second windows. It uses them
  to coach you and to train your own model. Your face stays clear on the box, and it's blurred the moment a picture
  would leave it."
- Dom's decisions: coaching and training both from expressions (6.B); no blur by default, only on egress (6.H).

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
- **N:** "Sort It runs a YOLO11 detector at forty-one frames a second on the Jetson, tracks every item, maps the
  camera onto the floor, and plans the walk around you on a ten-centimetre grid. Only counts leave the box."
- Sources: `config/models.yaml:22-25`, `README.md:84`, `contexts/play/room/grid.py`, `homography.py`.

### 10. The learning lab (spec, as now)
- Two AMD Strix Halo pods: your trainer, and the network's trainer with consent. The test gate. Hugging Face.
- Dom confirms the group and network learning is built or being built (6.C), so the line stands.
- 6.E (fine-tune numbers, the Hugging Face repo name) still needs Ilan's confirmation before numbers are shown.

### 11. On the device: cable pull, ledger (spec, as now)
- Everything live on the Jetson. Pull the cable: it keeps seeing, remembering, alerting and answering. Ledger: 13
  allowed, 13 blocked.
- **Changed line:** "A face is blurred only when a picture leaves the box." (6.H)

### 12. Friends: three people, three boxes, the league (demo, about 55 s)
**Cast, cut in a split screen (three panes, then two)**
- **Hemdutt:** in the dorm, talking to his Atlas.
- **Maya:** at '63 South (a new small dining-hall set, Ferrell Commons), standing, on her phone with her Atlas app.
- **Dominick:** at his own desk, on his computer.

**Flow (the league comes first, as Dom asked)**
1. **The league verdict.** All three Atlases speak at once, overlapping, each to its own person:
   - "This week's league is in."
   - "Dominick hit the most goals, so he picks what you do tonight."
   - "Hemdutt, you had the roughest week, so you're driving."
   - The window shows the league table: each person's percent and which targets they hit.
2. **Dominick picks**, on his computer: "Board game night at the Union."
   The boxes negotiate: "Free." / "Busy at 8, how about 8:30?" / "Agreed, three of three in."
3. **Maya first.** Her phone: "Maya shared she's at '63 South right now."
   **A** (to Hemdutt): "Maya's at '63 South. You're free until two. Want to grab lunch with her? It's your second
   real meal today, which keeps your meals goal on track."
   **P:** "Yeah, tell her I'm coming."
4. **A:** "I've opened the route, six minutes on foot. Then tonight, you three are at the Union at 8:30, and you're
   driving."
5. The window shows the shared plan on the calendar, the route, the league.

- Sources:
  - Negotiation format (`docs/demo-run-sheet.md:476-479`).
  - Shared places (`contexts/demo/circle.py:141`).
  - League scoring and stake (`contexts/play/league/domain/scoring.py:190, 210`; `config/league.yaml`).
  - Meals habit (`config/habits.yaml`).
  - Board game night (`fixtures/demo/nearby.yaml`).
  - The walk time to '63 South gets computed on the real map.
  - "Driving" is Dom's wording; the app's stake is "the worst week buys the next group meal" (Q4).

### 13. How the boxes talk, the trusted group, many groups (spec, about 18 s)
- **Visual:**
  - Three shells; only one-word replies cross between them.
  - The faces of the three friends unblur inside the group (with consent), and a soft link of learning grows between
    their models.
  - Zoom out to many small groups across campus, each with its own network.
- **N:**
  - "Each box checks its own calendar and answers in one word: free, busy, or another time."
  - "In a group you trust, friends can choose to unblur their faces to each other and learn from each other, each
    model getting better from the others' examples."
  - "And there can be as many groups as there are friendships, each with its own network."
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

### 15. NextEra: why it works on a plant floor (spec, about 12 s)
- **Visual:** the box on a control-room console. Its shell widens to cover a plant floor plan. A small SPOT robot icon
  walks a round; readings flow into the box, and nothing flows out.
- **N:** "Point the same box at a plant. Atlas reads a SPOT robot's inspection missions, the locations, times,
  readings and alarms, right on site. Inspection data never leaves the plant. Every answer cites the rows behind it,
  and nothing reaches a robot or a work system without a person's yes."
- Sources: `docs/spot.md:3-7, 14-20`; the track brief quoted there.

### 16. NextEra: an operator's shift with SPOT (demo, about 45 s)
**Set:** a small control room. Atlas on the console, a big display showing `/spot/`, the operator (a mannequin in a
hi-vis vest and hard hat colours) with a coffee. The app window mirrors the display.

1. **Operator:** "Morning, Atlas. What should I check first this shift?"
2. **A:** "Three things. First, heat exchanger 2 in the pipe gallery. It raised an alarm on a check that itself
   failed, and it hasn't had a good check in 88 hours. Send SPOT back before you send anyone."
   - Window: **Your shift**, three cards (Do this first / Keep an eye / Looks fine), HX-202 at the top.
3. **Operator:** "Where is it?"
   - Window: a red pointer lands on HX-202 on the floor plan, then walks the traced route.
   - **A:** "Turn left, take the entrance steps, and go along the west passage." (the guide's turn-by-turn style).
4. **Insight: why trust needs checking.**
   - **A:** "Heads up: mission status says ninety-six percent, but most of those rounds still hide a failed check.
     And checks under forty percent battery fail about three times as often."
   - Window: the trust line, then the battery explainer animating (35% vs 12%).
5. **Replay:** "Show me round 17." The window plays the round step by step: SPOT moves on the plan, each check arrives
   (tick, cross, alarm), the battery drains, and the step where it failed is marked.
6. **Act:** **Operator:** "Send SPOT back to it."
   - Approval card "Send SPOT back to HX-202?" → Yes.
   - **A:** "Done. A work order was written on this box. Nothing left the building."
7. **"If you wanted to do this work":** a trainee version in **Shift Run**. Plan the next round by dragging stops; the
   run plays out; stars against Atlas's plan; "battery and order matter". **A:** "Want to try planning the next
   round?"
8. **Handoff:** one tap, "Shift handed off."

- Sources:
  - `docs/spot.md`: "Mission status says 96%; 81% of those missions hide a failed action" (`:176, :267`).
  - HX-202, 88 h, top of triage (`:189-192`); battery 35.2% vs 12.1% (`:180`).
  - The guide's walking directions (`docs/spot.md`, look guide); work order on the box (`:270-272`).
  - Shift Run (`contexts/spot/domain/shiftrun.py`); the three tabs (Your shift, Shift Run, Analysis).
- Honest wording kept from `docs/spot.md`:
  - It's a replay of recorded synthetic data, not a live robot.
  - Routes are "about" (traced from the kit's picture).
  - "We do not claim a cause."

### 17. End (as now)
- The website's logo and splash, back on the shelf in the dorm.
- **N:** "Atlas. The personal AI computer. It learns you, and keeps you to itself."

---

## 4. Fixes to the last round

- [ ] **4.1 Chair backward.** Rotation `-π/2` → `+π/2`. Check from overhead and from the side: the backrest is
  behind the person.
- [ ] **4.2 Colour.** Shirt, trousers, shoes and skin, coloured per bone (shader), a different palette per person
  (Hemdutt, Maya, Dominick, the operator with hi-vis).
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

- [ ] **5.1 App window.** Top right, website style, tabs and a cursor, reused in every demo. Panes: Today, Route, Day
  Run, Week, Optional, Approve, Check-in, GroupMe, Notes, Sort It, Impact ring, League, Circle conversation,
  `/spot/` (Your shift, floor plan pointer, round replay, Shift Run).
- [ ] **5.2 Dialogue.** `film.yaml` gets `speaker:` per line and one voice per speaker; captions name the speaker; the
  lens pulses while Atlas talks; overlapping lines for the league.
- [ ] **5.3 Props:** phone (its screen mirrors the window), cans, water bottle, cup, nightstand, coffee.
- [ ] **5.4 Sets:** '63 South (small dining hall); Dominick's desk; the control room (console, big display); split
  screen framing.
- [ ] **5.5** Problem-statement visual (scattered windows into the box).
- [ ] **5.6** Particles for "what fed the brief".
- [ ] **5.7** Sort It camera overlay (boxes, ids, zones, grid route).
- [ ] **5.8** Trusted-group visual (unblur inside the group, model links, many groups zoom-out).
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
  Dom says this is built or being built, so the film shows it as working. Who owns it, so the Build Log can point to
  it? Not known to me: **[ ] ask on Notion**.
- **6.D "The route helps meet a goal"** (Dom asked what I meant).
  - Your earlier note had Atlas say the route "can also help meet this goal". There's no walking or steps goal in the
    app, so that line would claim something that doesn't exist.
  - Fix: tie it to a goal that does exist. Lunch at '63 South is "your second real meal today, which keeps your meals
    goal on track". Meals is a real habit and a league target.
- **6.E Wording and numbers to confirm.**
  - "A four-billion-parameter Qwen" replaces "It is not small".
  - The fine-tune numbers and the public Hugging Face repo name need Ilan's confirmation before they are shown.
- **6.F Exam.** There's no exam in the demo day; the film uses the writeup and the greedy quiz.
- **6.G Watch data.** Synthetic numbers only (`config/connectors.yaml:82-85`).
- **6.H No blur by default; blur only on egress.**
  - The film says "blurred only when a picture leaves the box".
  - Today's code also blurs keyframes before storing them (`contexts/perception/infrastructure/face_blur.py`,
    `contexts/vault/application/policy.py`). Making the product match is a policy change for the perception owner.
    **[ ] raise on Notion.**
- **6.I The past-look line** ("the same look you had Tuesday, after two hours without moving").
  - The coach compares expressions with your own history.
  - The seeded history needs one Tuesday entry with a long sit and strained looks for the line to be true in the
    demo data. That's a small fixture addition. **[ ] add, or reword to "the look you get after a long stretch
    without moving"** (that version needs no new data). I recommend the reworded line.

## 7. Questions still open

- **Q1 Length:** this plan runs about 5:10. Is there a hard limit (Devpost, judges)? If it's 3 minutes, I'd cut beats
  3, 6, 9 and 11 into the neighbouring beats and shorten 14.
- **Q2 Atlas's voice:** the box's real Piper voice (recommended, it's the product), or a polished edge-tts voice?
- **Q3 "It remembers"** (keys, 17 × 23, the UCF score): it isn't in this plan. Cut, or fold "where are my keys?" into
  the morning?
- **Q4 League stake:** "you're driving" (your wording) or the app's "buys the next group meal"?
- **Q5 Names:** Hemdutt, Maya and Dominick, as in the Circle demo?

## 8. QA before any render

- [ ] `npx hyperframes check` clean.
- [ ] Two-angle stills of every set (dorm chair, bed, shelf, desk, kitchen, '63 South, Dominick's desk, control room).
- [ ] A camera's-eye still from Atlas: bed and chair in view.
- [ ] Contact sheet of every beat; captions name the speaker; no card covers a face.
- [ ] Every number ticked against its source.
- [ ] One-scene drafts for Dom first (`python video/scripts/film_render.py <scene>`); full render only after approval.
- [ ] `pytest -q contexts/demo/tests/test_video.py` green, commit, push, Build Log and D12 on Notion.
