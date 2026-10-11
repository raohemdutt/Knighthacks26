# Atlas film v4: plan and tracking document

Status: **PLAN, waiting for Dom's approval. Nothing is rendered until it is approved.**
Branch `ctx/demo/video-hyperframes`, last commit `ac77b55`. Owner: Dom (Notion task D12).

Every fact on screen or in a line comes from the repo, with the source next to it, or it is listed under
"Claims that need a decision" (section 6). Tick a box only when it has been built *and* checked in a frame.

---

## 1. How the film works (principles from Dom, Sun Oct 11)

1. **Demo, then spec, then demo, then spec.** Demo beats show the person and Atlas living a day together. Spec beats are
   short (about 10 s) and say what made that moment possible.
2. **In demo beats the product talks, not the narrator.** Atlas speaks to the person in its own voice. The person
   answers. No voiceover over a demo. The narrator only speaks in the problem statement, the spec beats and the close.
3. **Atlas speaks naturally and never reads a screen out word for word.** The screen shows the detail and the voice
   gives the gist, for example "you're free until two" rather than reading out a list of blocks.
4. **Atlas's apps are on screen.** A window in the top-right corner shows what Atlas is doing as it happens, built
   from the website's real components (`web/`): calendar, route map, Day Run simulation, Sort It bin highlight,
   Circle conversation, league and check-in. Tabs switch as Atlas talks.
5. **The person is the star.** Colour, hair, a face that changes expression, real movement (sections 4 and 5).
6. **No plain mistakes.** Every set is checked from two angles before a scene is called done. The chair facing
   backward was one such mistake (fix: 4.1).

## 2. My understanding of the flow up to the kitchen (Dom asked me to confirm)

1. **Wake up.** The person wakes in bed. Atlas says good morning, then:
   - how they slept;
   - how that compares with their usual nights;
   - whether it met the goal, with a "nice work" or a "let's do better".
2. **Morning brief.** In a natural voice, Atlas walks through today:
   - the calendar, what's first and when to leave;
   - what's due this week;
   - optional events friends are going to.

   Then it asks: "Does this look good? I can lock this plan, or would you like something changed?" The person says
   lock it and approves on the phone. While it talks, the app window walks through the calendar, the route on the
   campus map, the Day Run simulation of the day, and the approval card.
3. **(spec)** What made the brief: watch, calendar, Canvas, GroupMe, UCF events, campus map, the model on the Jetson,
   approval before anything is written.
4. **At the desk, later.** The person has been sitting a long time. Atlas checks in:
   - "You've been sitting for 1 h 45."
   - What's coming up (the real deadline from the calendar).
   - What classmates said in the class GroupMe.
   - "I've pulled together notes from the group chat and r/ucf and a couple of articles; go through them, but first…"
   - Third energy drink against a limit of two, no water in 100 minutes, resting heart rate up on the watch.
   - "Take a walk, stretch your legs."

   The app window shows the check-in card with the watch data, then the notes with their sources.
5. **The face segue.** Atlas also says "and you've looked strained for the last 21 minutes, more than your usual
   afternoons". That leads into a spec beat on facial expressions. Faces are blurred before any picture leaves the
   box. If the user opts in, their expressions are read on the device to coach them, and possibly to train their
   model (see 6.B). Nothing else changes in that scene.
6. **Kitchen.** The person walks to the kitchen holding their phone, about to throw a can in the trash. Atlas says
   "cans go in recycling at UCF, the blue bin". The blue bin lights up on their phone, and we see that same phone
   screen in the top-right window. They put it in recycling. Atlas gives the weekly count and the nudge (goal,
   67% → 71%). Then **Sort It** as its own short interactive bit ("want a quick round?"), and a 10 s spec on what
   Sort It runs on.

I understand this and the plan below follows it. Places where the repo doesn't back a line are in section 6.

---

## 3. The sequence

Durations are estimates. The total is about 4:45 to 5:00 (see question Q1).
Voices: narrator (N), Atlas (A), the person (P), friends (Maya M, Dominick D). See Q2.

| # | Beat | Type | ≈ s | Status |
| --- | --- | --- | --- | --- |
| 0 | The problem | narrator | 14 | [ ] new |
| 1 | This is Atlas, and what's inside | spec | 16 | [ ] trim the existing intro |
| 2 | Wake up and the morning brief | **demo** | 50 | [ ] new |
| 3 | What made the brief | spec | 10 | [ ] new |
| 4 | The model on the box (42 of 42) | spec | 12 | [ ] trim the existing "learn" |
| 5 | Desk check-in, notes and the watch | **demo** | 40 | [ ] new (replaces "remember" and the old desk beats) |
| 6 | Faces and expressions, privacy, cable pull, ledger | spec | 22 | [ ] new plus trimmed "device" |
| 7 | Kitchen: the can, the phone, the count, the nudge | **demo** | 30 | [ ] rebuild |
| 8 | Sort It round | **demo** | 15 | [ ] new |
| 9 | What Sort It runs on | spec | 10 | [ ] new |
| 10 | The learning lab | spec | 15 | [ ] trim the existing "lab" |
| 11 | Friends: three people, three boxes | **demo** | 50 | [ ] rebuild |
| 12 | How the boxes talk (and the trusted group) | spec | 14 | [ ] new (needs 6.C) |
| 13 | Same box, other lives (expanded) | spec | 22 | [ ] rebuild, Day Run line removed |
| 14 | NextEra: the plant operator and SPOT | demo plus spec | 30 | [ ] **waiting on Dom's idea** |
| 15 | End | narrator | 6 | [ ] keep |

### 0. The problem (narrator, about 14 s)
- **Visual:** the person's day as scattered app windows (calendar, chat, notes, a sleep app, a map) floating around
  the dorm, not talking to each other. They pull together into the box on its shelf.
- **N:** "Your day lives in a dozen apps: classes, deadlines, sleep, friends. None of them sees how your day actually
  goes. And the assistants that could help send your life to someone else's cloud."
- **N:** "This is Atlas. A personal AI computer that sits on your desk, sees your day, and keeps it on the device."

### 1. This is Atlas: what's inside (spec, about 16 s)
- **Visual:** the existing assembly, explode and x-ray, shortened.
- **N:** "A camera on top, a speaker in front, and an NVIDIA Jetson inside that runs the vision, the memory and an
  agent that acts on your behalf, like an assistant."
- **Labels:** Webcam · sees / Speaker · talks / Jetson Orin Nano · vision, memory, agent.

### 2. Wake up and the morning brief (demo, about 50 s)
**Set:** the dorm (fixed per section 4). Atlas on the shelf, aimed at the chair and able to see the bed.

**Action**
1. The person is lying in bed and sits up (built per 4.4), with a sleepy face.
2. They stand, pick up their phone from the nightstand (a new prop), walk to the desk and sit.
3. They look at Atlas while it talks.

**Lines**
- **A:** "Good morning. You slept seven and a half hours, right around your usual, and over your six-hour goal. Nice
  work."
  - Sources: synthetic watch night 7 h 28 (`fixtures/demo/fakes.yaml:43`); your usual is the median of your own nights,
    7.4 h (`contexts/demo/history.py:66-69`, `contexts/day/domain/checkin.py:44-46`); goal ≥ 6 h
    (`config/habits.yaml:32`).
- **A** (natural, never reading):
  - "Today's a big one. Your mentor check-in is at 9:20 in Engineering II. It's a ten-minute walk, so leave by ten
    past."
  - "Submissions close at eleven, and judging runs noon to two."
  - "Maya set up a COP4533 study session at the library at two."
  - "Your milestone 1 writeup is due Tuesday night, and quiz 5 is Thursday."
- **A:** "If you want a break, Sam and Riley are doing board game night at the Union at eight, and Andria's running a
  quiz review at the library. Both optional."
- **A:** "Does this look good? I can lock this plan, or would you like something changed?"
- **P:** "Looks good. Lock it."
- **A:** "Locked. It's on your phone to approve."
- The person taps Approve on the phone.
- Sources:
  - Calendar: `fixtures/demo/night.yaml:52-59`, milestone `:34-37`, quiz `:38-41`.
  - Board game night: `fixtures/demo/nearby.yaml:31-44`. Review session: `nearby.yaml:47-48`.
  - Walk: "10 minutes on foot to Engineering II, 680 m" (`docs/demo-run-sheet.md:304-305`).
  - Approval card: "I'd like to make these nine blocks today's plan. Okay?" (`narration.py:665-666`).

**App window, top right** (tabs switch as Atlas talks)
1. **Today:** the day ribbon. Calendar blocks are solid; Atlas's suggestions are dashed (`web/app/now`).
2. **Route:** the real UCF map, dorm → Engineering II, 680 m, "Leave by 9:10" (the RouteSheet look).
3. **Day Run:** the day played on fast-forward on the campus map, with the figure walking it
   (`web/components/dayrun/DayPlayer.tsx`, `Figure.tsx`). This replaces the old "Play tomorrow" closing line.
4. **Week:** milestone 1 Tue 11:59 PM, Quiz 5 Thu 5 PM.
5. **Optional:** board game night, quiz review, with friends' names.
6. **Approve:** the approval card on the phone, then "Locked".

### 3. What made the brief (spec, about 10 s)
- **Visual:** the brief's sources fly into the box as labelled particles: Apple Watch (via FreeReps), Outlook and
  Canvas calendars, GroupMe, UCF Events, OpenStreetMap. Then one line out to the phone, "Approve?".
- **N:** "The brief comes from your watch, your calendars, your group chats and the campus map. A model on the Jetson
  writes it in under three seconds. Nothing goes on your calendar until you approve it."
- Sources: connectors `config/connectors.yaml`; "Brief me" 2.7 s (`docs/demo-run-sheet.md:109-111`); approval rule
  (`README.md:49`).

### 4. The model on the box (spec, about 12 s)
- **Visual:** the existing Qwen-on-Jetson x-ray, plus "42 / 42".
- **N:** "A four-billion-parameter Qwen runs right here: fast, accurate and private. Forty-two test questions,
  forty-two right."
- The current film line "It is not small" contradicts the repo (Qwen3.5-4B Q4). See 6.E.

### 5. Desk check-in, notes and the watch (demo, about 40 s)
**Action**
1. Afternoon light.
2. The person is slumped at the desk, typing (pose per 5.3), three cans beside the keyboard, a tired or strained face.
3. Atlas's lens wakes.

**Lines**
- **A:** "Hey, you've been sitting for an hour and forty-five minutes."
- **A:** "I know the milestone 1 writeup is due Tuesday. Priya posted in the class chat that half the points are the
  runtime analysis, and Jordan says weeks five and six cover it."
- **A:** "I've pulled together notes from your group chat and r/ucf, plus a couple of articles on knapsack and
  interval scheduling. Take a look when you're ready."
- **A:** "But first: that's your third energy drink today, and your limit is two. I haven't seen you pick up water in
  over an hour and a half. Your watch says your resting heart rate is up, and you've looked strained for the last
  twenty minutes, more than your usual afternoons."
- **A:** "Go fill your water bottle and walk around for a few minutes. I'll hold the notes."
- **P:** "Yeah, okay." They stand and stretch, and pick up the bottle and the can, which leads into the kitchen.
- Sources:
  - The check-in demo: sat 105 min, no drink for 100 min, strained for 21 min (`contexts/demo/checkin.py:53`).
  - Energy drink limit 2 (`config/habits.yaml:54-60`); three cans (`fixtures/demo/night.yaml:167-169`).
  - Resting heart rate rule (`checkin.py:13-14`).
  - GroupMe lines (`fixtures/groupme/class_chat.yaml:7-15`).
  - The exam-prep planner pulls the web and the subreddit with citations (`contexts/agent/application/planner.py:1-10`).
  - It speaks only when two signals agree (`checkin.py:23-26`).

**App window, top right**
1. **Check-in card:** the heart-rate line shape, plus chips "Sat 1 h 45", "No drink 100 min", "3rd can · limit 2",
   "Looked strained 21 min" (`web/components/checkin/CheckinCard.tsx`).
2. **Watch:** resting HR against your own median, last night's sleep.
3. **Notes:** quoted GroupMe messages with names, an r/ucf thread, two articles, each with a source tag.

### 6. Faces and expressions, privacy (spec, about 22 s)
- **Visual:**
  - A frame from the desk camera with the face blurred as it crosses the box's edge (the existing device scene).
  - The person's face on the device, with a small "looked strained · 30 s window" tag that stays inside the shell.
  - The cable pull, then the ledger: 13 allowed, 13 blocked.
- **N:**
  - "Every face is blurred before a picture can leave the box."
  - "If you choose, Atlas reads your expressions on the device, in thirty-second windows, never frame by frame, so it
    can notice a hard afternoon and help." (Whether to add "and learn from them": see 6.B.)
  - "Pull the network cable, and it keeps seeing, remembering and answering. It keeps a ledger of what left:
    thirteen destinations allowed, thirteen blocked."
- Sources:
  - YuNet blur on by default (`contexts/perception/infrastructure/face_blur.py:1-8`).
  - The expression model, opt-in, 30 s windows (`contexts/perception/domain/expression.py`, `identity.py:38-43`).
  - Ledger (`docs/demo-run-sheet.md:118`).

### 7. Kitchen: the can, the phone, the count, the nudge (demo, about 30 s)
**Action**
1. The person walks into the kitchen holding the phone and a can.
2. They step toward the grey trash bin. The wall camera sees it.

**Lines**
- **A** (from the kitchen Atlas): "Hold on: cans go in recycling at UCF. The blue one."
- The phone and the top-right window show the Sort It screen: "Can goes in recycling", the blue bin highlighted, the
  route from "You", "N steps". The person turns and drops it in the blue bin.
- **A:** "Right bin. That's five of seven this week, seventy-one percent. Your goal's eighty."
- **A:** "Four more in the right bin and you're there. Aluminum's the most valuable thing in that bin."
- Sources:
  - The rule (`config/sorting.yaml:65`).
  - 4 of 6 = 67%, goal 80%, "The next 4 in the right bin…" (`docs/demo-run-sheet.md:560-564`).
  - 5 of 7 after this throw is our arithmetic.

**App window, top right:** the Sort It route card (`web/app/play/page.tsx:212-300`), then the Impact ring moving
67 → 71% with the goal tick (`WeekStory.tsx`).

### 8. Sort It round (demo, about 15 s)
- **A:** "Want a quick round of Sort It? Hold up the next thing."
- The person holds up a cup for 2 seconds (a round starts on a held face). The phone shows "Cup goes in trash" and the
  route. They walk it there.
- Points pop up: "Right bin · 6.0 s · 150 points" (`docs/demo-run-sheet.md:521-525`).
- **App window:** the live round with route, timer and score; then "right bin, round 1 / round 5 onward"
  (`web/app/play/page.tsx:181-185`).

### 9. What Sort It runs on (spec, about 10 s)
- **Visual:** the bin camera view with boxes, track IDs, bin zones and a grid route bending round the person.
- **N:** "Sort It runs a YOLO11 detector at 41 frames a second on the Jetson, tracks every item, maps the camera onto
  the floor, and plans the walk around you on a ten-centimetre grid. Only counts leave the box."
- Sources: `config/models.yaml:22-25`; `README.md:84`; `contexts/play/room/grid.py:1-8`; `homography.py`.

### 10. The learning lab (spec, about 15 s)
- Keep the current visuals: two Strix Halo pods, consent, the gate, Hugging Face.
- **Fix:** the "network's trainer learns from everyone's examples" line has no code behind it. See 6.C.
- Numbers that exist: LoRA r16, loss 0.127 → 0.0095, 784.5 s (`fixtures/demo/l16-run.json`). The repo disagrees with
  itself on whether this run is "pending" (see 6.E).

### 11. Friends: three people, three boxes (demo, about 50 s)
**Three people, three places, three Atlases, cut in a split screen**
- **Hemdutt (our person):** in the dorm, talking to his Atlas.
- **Maya:** at '63 South (a new small dining-hall set, inside Ferrell Commons), standing, on her phone.
- **Dominick:** at his own desk, on his computer.

**Flow (league first, as Dom asked)**
1. **A** (to Hemdutt): "Maya just shared that she's at '63 South. You're free until two. Want to head over?"
   **P:** "Sure."
2. **A:** "Dominick's free tonight too. There's board game night at the Union at eight."
   - Each box answers free or busy, one word. The bubbles animate between the three boxes: "Free." / "Busy then,
     how about eight-thirty?" / "Agreed, three of three in."
3. **League:** "Based on this week's goals, Dominick hit the most, so he picks where you go. Hemdutt, you had the
   roughest week, so you're driving." See Q4: the repo's rule is that the worst week buys the food.
   - All three Atlases say it at once, overlapping, with the league table in the window.
4. **A:** "I've opened the route for you. It's also the walk your afternoon was missing." See 6.D: there is no steps
   goal.
5. The window shows the shared plan on the calendar, the route, and the league table.

- Sources:
  - The conversation format (`docs/demo-run-sheet.md:476-479`).
  - Shared places (`contexts/demo/circle.py:141`).
  - League scoring and stake (`contexts/play/league/domain/scoring.py:190, 210`; `config/league.yaml:2`).
  - Board game night (`fixtures/demo/nearby.yaml`).
  - The circle members in the demo are Hemdutt, Maya and Dominick (`contexts/demo/circle.py:39-41`).

### 12. How the boxes talk, and the trusted group (spec, about 14 s)
- **Visual:**
  - Three shells. Only one-word replies cross between them.
  - Zoom out to many small groups, each with its own lines.
- **N, based on what's built:**
  - "Each box checks its own calendar and answers in one word: free, busy, or another time. Calendars, pictures and
    raw signals never leave."
  - "With a tap, friends' boxes can learn to recognise each other, and the group can ask the lab's bigger model
    together."
- Dom asked for more: group model learning between friends, many groups each with its own network, and unblurring
  faces between friends to learn from expressions. **None of that is built.** See 6.C before we write the line.

### 13. Same box, other lives (spec, about 22 s; the Day Run line is removed)
- **Visual:** three short vignettes in the same world, each with its own app window. Drop the campus-route ending.
  - Night shift: shift 10 PM-6 AM, sleep target 6.5 h, energy drinks limit 1, leave checklist keys / badge / phone /
    charger.
  - Family home: each member's own habits; "Leo: bins out", "Ava: science fair poster".
  - Plant operator: hands off to the NextEra beat.
- **N** (more on why it isn't only for students; specs kept simple):
  - "It isn't just for students. The same box runs a night-shift worker's upside-down day, a family's homework and
    chores, a plant floor."
  - "It all runs on twelve watts at idle, it answers offline, and it never needs your data to leave."
- Sources:
  - Profiles: `config/profiles/*`; `habits.yaml`.
  - Power: 12.62 W idle, 17.5 W working (`lab/refine/research/telemetry/demo_numbers_2026-10-11.md:47-49`).

### 14. NextEra: the plant operator and SPOT (about 30 s)
**Waiting on Dom's idea.** Facts ready to use:
- "Mission status says 96%; 81% of those missions hide a failed action."
- HX-202, no good inspection for 88 h, top of triage at 72.3.
- "A work order was written on this box. Nothing left the building."

Sources: `docs/spot.md:176, 189-192, 267-272`.

### 15. End (about 6 s)
- The same website logo and splash, on the shelf in the dorm (not on the campus map, since Day Run left the ending).
- **N:** "Atlas. The personal AI computer. It learns you, and keeps you to itself."

---

## 4. Fixes to the last round (Dom's notes)

- [ ] **4.1 Chair backward.** The back faces the desk because the rotation is `-π/2`; it should be `+π/2`.
  - Check: from overhead and from the side, the backrest is behind the person, between them and the bed.
- [ ] **4.2 Person too plain: colour.** Shirt, trousers, shoes and skin on hands and head, coloured per bone (shader,
  no new mesh), with a different palette for each person (Hemdutt, Maya, Dominick).
- [ ] **4.3 Hair.** Quaternius hairstyles, rigged to the head bone. They're CC0 and in the free "Universal Base
  Characters" pack we already downloaded: `Hair_SimpleParted`, `Hair_Long`, `Hair_Buns`, plus eyebrows.
  - Check that they fit the mannequin's head. If not, fall back to a simple cap mesh.
- [ ] **4.4 Face with expressions.** A soft stylised face on the head bone (eyes, brows, mouth) with blendable
  expressions: sleepy, neutral, smile, strained, talking (mouth moving while they speak).
  - Driven by the film clock, like everything else.
  - This also gives the expression segue (beat 5 → 6) something to show.
- [ ] **4.5 More movement.**
  - "Sitting_Talking_Loop" when the person speaks.
  - Idle sway, breathing, glances.
  - Typing (forearm override on top of sitting).
  - Holding a phone (arm pose override plus a phone prop in the right hand).
  - Picking up an object.
- [ ] **4.6 Lying in bed and sitting up.** There's no clip for it in the free pack. Build it from the sitting pose on
  a pivot at the hip (lying back → sitting up), then the existing Sitting_Exit.
  - Check: no clipping through the mattress.
- [ ] **4.7 Atlas can see the bed and the chair.** The bed is already in its view cone (check in a camera's-eye
  frame).
- [ ] 4.8 Keep: shelf at eye height, no monitor webcam, the real UCF map, website look, faster renders.

## 5. New things to build

- [ ] **5.1 App window component.** Top right, about 760 × 470, website style, with tabs and a cursor. It's a
  sub-composition reused in every demo beat. Content is static HTML ported from `web/` with real fixture values.
- [ ] **5.2 Dialogue system.** Per-speaker voices and captions labelled by speaker ("Atlas", "You", "Maya"…).
  - `film.yaml` gets `speaker:` on each line.
  - Atlas's voice also plays a small lens pulse while it talks.
- [ ] **5.3 Props:** phone (with its screen mirroring the app window), cans by the keyboard, water bottle, cup,
  nightstand.
- [ ] **5.4 '63 South set** (small dining hall) and **Dominick's desk** set for the split screen; split-screen
  framing.
- [ ] **5.5 Problem-statement visual:** scattered app windows pulling into the box.
- [ ] **5.6 Data-flow particles** for beat 3 (labelled sources into the box).
- [ ] **5.7 Sort It camera overlay** for beat 9: boxes, track IDs, zones, grid route.

## 6. Claims that need a decision (the repo doesn't back them as asked)

- **6.A Professor reminder.** No professor message exists in the fixtures.
  - Option 1: use the TA comment Priya relays ("the TA said half the points are the runtime analysis").
  - Option 2: add a professor message fixture and make it true in the product too.
  - I recommend option 1.
- **6.B Expressions used to train the model.** Expressions are used for coaching (built). There is no consent or code
  to use them for training; face share explicitly excludes expression history.
  - Option 1: say "to coach you".
  - Option 2: say "and, if you allow it, to personalise your model" as a stated future feature.
- **6.C Friends.** Three things Dom asked for aren't built:
  - The group learning from each other's models, or a shared network trainer. L10 group fine-tune is P1 only; the lab
    says the Circle "pools chips, never data".
  - Many groups each with their own network: a box has one circle.
  - Unblurring faces between friends to learn from expressions: face share shares a recognition template only and
    never unblurs or shares expressions.

  Options:
  - (a) Present them as "what's next", clearly worded as future.
  - (b) Cut them.
  - (c) Build a thin version first.

  I recommend (a) for the group network and (b) for unblurring, because unblurring cuts against our privacy story.
  This also affects the existing lab line in 10.
- **6.D "The route helps meet this goal."** There's no steps or walking goal. Option: tie it to the check-in ("the walk
  your afternoon was missing") or to the focus and break coaching. A new steps habit would be a product change.
- **6.E Wrong or stale numbers in the current film.**
  - "Qwen is not small": it is 4B Q4. Reword to "a four-billion-parameter Qwen".
  - L16 fine-tune: the run JSON has numbers, but the README and demo numbers say "pending". Ilan should confirm before
    we show them.
  - The public Hugging Face adapter has no published repo name in the repo. Confirm it exists before we say "public on
    Hugging Face".
- **6.F Exam.** There's no exam in the demo day. The nearest are Quiz 5 (Thu) and "COP4533 exam review". The plan uses
  the milestone and Quiz 5.
- **6.G Watch data.** We use the synthetic file (7 h 28 night); the real Apple Watch export must not be shown
  (`config/connectors.yaml:82-85`).

## 7. Questions for Dom

- **Q1 Length.** This plan runs about 4:45-5:00, against the current 3:07. Is there a hard limit (Devpost or the
  judges)? If it's 3 minutes, I'd shorten beats 1, 4 and 10 and merge 9 into 8.
- **Q2 Voices.** Narrator: keep Andrew. Atlas: a distinct calm voice (suggest edge-tts "Ava").
  Hemdutt/Maya/Dominick: three other edge-tts voices. OK, or do you want real voices recorded by the team?
- **Q3 "It remembers" (keys, 17 × 23, the UCF score).** You didn't mention it. Cut it, or fold "where are my keys?"
  into the morning as the person heads out?
- **Q4 League stake.** The repo rule is that the worst week buys the food and the best week picks the place. You said
  the worst week drives. Keep "drives" (film only), switch to "buys the food" (matches the app), or both?
- **Q5** Decisions on 6.A-6.D (my recommendations are in each).
- **Q6 NextEra beat.** Waiting for your idea, as you said.
- **Q7 Names.** Use Hemdutt, Maya and Dominick as in the Circle demo, or unnamed "you, Maya, Dominick"?

## 8. QA checklist before any render

- [ ] `npx hyperframes check` clean.
- [ ] Two-angle stills of every set (chair, bed, shelf, desk, kitchen, '63 South).
- [ ] A camera's-eye still from Atlas showing the bed and the chair.
- [ ] Contact sheet of every beat at three points; captions show speakers; no card covers the person.
- [ ] Every number on screen ticked against section 3's sources.
- [ ] Single-scene draft renders (`python video/scripts/film_render.py <scene>`) for Dom first; full render only after
  approval.
- [ ] `pytest -q contexts/demo/tests/test_video.py` green; commit; push; Build Log and D12 on Notion (needs Notion
  access in this session).
