# Atlas film: the whole script (v4.6)

This is the read-through. `FILM_PLAN.md` has the sources, the build list and the decisions behind every line.
**N** = narrator · **A** = Atlas · **H** = Hemdutt (the person) · **I** = Ilan · **D** = Dominick · **X** = Xander
(the operator) · **T** = the trainee. Each speaker has their own voice. [Brackets] are what we see. "Window" is the app
window in the top-right corner.

Demo beats: the product talks, no narrator. Spec beats: the narrator, short.

v4.6 folds in Hemdutt's notes from Notion (Sun 3:21 AM, agreed by him): 48 of 48, Today's Plan, the face wording,
beat 13's "recognise each other", a new beat 15, one clause per sponsor track, and a new last line. It also names the
Encompass data-capture story in beat 3. **It runs over the 5:00 cap; see `FILM_PLAN.md` section 10 for the honest
timing and the cuts Dom chooses from.**

---

### 0 · The problem (N, 0:00)
[Hemdutt's dorm, morning light. App windows float around the room, none touching: a calendar, a group chat, notes, a
sleep app, a map. They drift together into a small white box on a shelf.]
**N:** Your day lives in a dozen apps, and none of them sees how it actually goes. The assistants that could help
send your life to someone else's cloud.
**N:** This is Atlas: a personal AI computer that sees your day, and keeps it on the device.

### 1 · What's inside (spec, as now)
[The Atlas logo draws in. The box assembles, opens, and goes see-through: webcam, speaker, Jetson.]
**N:** Inside: a camera on top, a speaker in front, and a Jetson that runs the vision, the memory, and an agent
that acts for you, like an assistant.

### 2 · Good morning (demo)
[Hemdutt lies in bed. He sits up, sleepy, and stretches.]
**H:** Good morning.
[Atlas's lens glows blue on the shelf, aimed at him.]
**A:** Good morning. You slept about seven and a half hours, right around your usual, and over your six-hour goal.
Nice work.
[He gets up, takes his phone from the nightstand, walks to the desk and sits. Window: Today.]
**A:** Big day. Your mentor check-in is at 9:20 in Engineering II, so leave by ten past.
[Window: Route, the real UCF map, dorm to Engineering II.]
**A:** Submissions close at 11, judging's at noon, and Maya's study session is at 2. Your algorithms writeup is due
Tuesday night.
[Window: Today's Plan ("Try your day before you live it."), the day playing on fast-forward on the campus map; then
This week.]
**A:** And if you want a break, Sam and Riley are at board game night at 8.
[Window: Optional, with friends' names.]
**A:** Does this look good? I can lock this plan, or would you like something changed?
**H:** Looks good, lock it. And help me actually eat two real meals today.
**A:** Done. Two meals, I'll keep track. Tap to approve.
[He taps Approve on his phone. Window: the approval card, "Goal today: 2 meals", then "Locked".]
**A:** Locked.
[He pats his pockets.]
**H:** Atlas, where are my keys?
**A:** Behind the mug on your desk. You left them there last night.
[A soft red ring lands on the keys. Window: the answer card with the frame, keys circled, then the room map with a
"Keys here" pin. He picks them up.]

### 3 · What fed the brief (spec)
[Labelled sparks fly into the box: the camera, Apple Watch, Outlook, Canvas, GroupMe, UCF Events, OpenStreetMap. Inside,
a counter runs: frames, then events, then episodes, then answers. One line goes out to the phone: "Approve?"]
**N:** That brief came from your watch, your calendars, your chats and the campus map, written on the Jetson in
under three seconds. Nothing is booked until you say yes.
**N:** And what the camera sees becomes events, then moments it remembers, then answers, and, with your OK, the
examples it learns from.

### 4 · The model on the box (spec, as now)
[The box goes see-through around the Jetson. Cards: "Qwen · on the Jetson", Fast, Accurate, Private, "48 / 48".]
**N:** A four-billion-parameter Qwen runs right here: fast, accurate, and private. Forty-eight demo questions,
forty-eight right.

### 5 · At the desk (demo)
[Afternoon. Hemdutt hunched at the desk, typing. Three cans by the keyboard. His face is tired. Atlas's lens wakes.]
**A:** You look tired: the look you get after a long stretch without moving. You've been sitting an hour and
forty-five minutes.
<!-- Target line, once Hemdutt's stage 1 (an expression remembered with its context) is merged: "You look tired.
That's the same look you had Tuesday afternoon, when you'd gone two hours without moving." -->
[Window: the check-in card. Chips: "Looked tired", "Sat 1 h 45", "No water 100 min", "3rd can · limit 2".]
**A:** I know the writeup's due Tuesday. Maya just texted the class chat that the professor said half the grade is
the runtime analysis.
[Window: the class GroupMe, Maya's message and Jordan's reply.]
**A:** I've pulled notes from the chat and r/ucf, plus two articles. They're on your screen.
[Window: Notes, an outline with source tags.]
**A:** But first: that's your third energy drink, and your limit is two. No water in an hour and a half, and your
watch says your heart rate's up. Go fill your bottle and take a walk.
**H:** Yeah, okay.
[He stands, stretches, and picks up his water bottle and the empty can.]

### 6 · Faces (spec)
[His face, clear, on his own phone's live view, inside the privacy shell, with a small "looked tired · 30 s window" tag
that stays inside. Beside it, a saved frame of the same moment with the face blurred. A frame slides to the shell's
edge, and the face blurs as it crosses.]
**N:** If you turn it on, Atlas reads your expressions on the device, to coach you and to train your own model.
You see yourself clear on your own screen. Saved pictures are blurred, and so is anything that leaves.

### 7 · The kitchen (demo)
[Hemdutt walks into the kitchen with his phone, the can and the bottle. He fills the bottle and steps toward the grey
trash bin. The wall camera watches.]
**A:** Hold on: cans go in recycling at UCF. The blue one.
[His phone and the window show Sort It: "Can goes in recycling", the blue bin lit, the route from "You". He turns and
drops it in the blue bin.]
**A:** Right bin. Five of seven this week, seventy-one percent. Your goal's eighty.
[Window: the ring moves from 67 to 71, the goal tick at 80, a small "sources" chip under the weekly numbers.]

### 8 · Sort It (demo)
**A:** Want a quick round of Sort It? Hold up the next thing.
[He holds up a cup. The phone shows "Cup goes in trash" and the route. He walks it there. "Right bin · 6.0 s · 150
points".]

### 9 · What Sort It runs on (spec)
[The bin camera's view: boxes on the can and the cup, track numbers, the bin zones, a grid route bending round him.
Then the look guide: a dot follows his eyes across the room view, with a "frames stay in the browser" tag.]
**N:** Sort It runs a YOLO detector at forty-one frames a second, tracks every item, and plans the walk around you.
The bin rules are UCF's own, and every impact number cites its source. Only counts leave the box.
**N:** The look guide follows your eyes to point the way, and those frames never leave the browser.

### 10 · The learning lab (spec, as now)
[Two AMD Strix Halo pods. Streams from Atlas boxes; a consent check; the test gate; Hugging Face; a box swapping in a
new model, with a rollback arrow. Measured numbers only after Ilan confirms them (plan 6.E).]
**N:** The lab is two AMD Strix Halo pods. One fine-tunes your own model on your own examples. The other, with
everyone's consent, learns from all of them, so every Atlas gets better.
**N:** A test decides if the new model ships. When one passes, it goes up on Hugging Face, and the box checks it,
swaps it in live, and rolls back if anything fails.

### 11 · On the device (spec, as now)
[The box see-through, the network cable pulled, chips: seeing, remembering, answering, speaking, learning (paused). A
question cuts ahead of the camera queue. The ledger: 13 allowed, 13 blocked.]
**N:** Pull the network cable: it keeps seeing, remembering and answering, out loud, on the box. Only learning
pauses. When the cameras are busy, your question still goes first.
**N:** And a ledger shows what left: thirteen destinations allowed, thirteen blocked.

### 12 · Friends (demo)
[Sunday evening. Three panes: Hemdutt in his dorm; Ilan at '63 South on his phone; Dominick at his desk on his
computer. Each has an Atlas.]
**A ×3** (over each other): This week's league is in. · Dominick hit the most goals, so he picks what you do tonight. ·
Hemdutt, you had the roughest week, so you're driving.
[Window: the league table.]
**D:** Board game night at the Union. Eight o'clock.
[Window: three calendars side by side. Each box reads only its own; the others are grey "busy" blocks. Bubbles
between the boxes:]
**A (Hemdutt's):** Free. · **A (Ilan's):** Busy at 8. How about 8:30? · **A (Dominick's):** 8:30 works.
[The same 8:30 block drops onto all three calendars at once. "Agreed, three of three."]
[Ilan's phone: "Ilan is at '63 South."]
**A** (to Hemdutt): Ilan's at '63 South, and you're free till 8:30. Dinner with him? That's your second real meal,
the goal you set this morning.
**H:** Yeah, tell him I'm coming.
**A:** Told him. Route's open.
[At '63 South: Hemdutt and Ilan at a table. Hemdutt's phone buzzes.]
**A:** Did you eat with Ilan?
**H:** Yep.
**A:** Logged. Two meals today: goal met.
[Window: meals goal 2 of 2; the league row updates.]

### 13 · How the boxes talk (spec)
[Three shells; only single words cross between them, through a small Neon database in the middle that holds only what
each one chose to share. Inside the group, the boxes recognise each other: a friend's name appears over them. Soft
links of learning grow between their models. Zoom out: many small groups across campus, each with its own network.]
**N:** Each box reads only its own calendar and answers in one word. They meet in a Neon database that holds only
what each person chose to share.
**N:** In a group you trust, friends can let their boxes recognise each other, and learn from each other's
examples, and every group runs its own network.

### 14 · Same box, other lives (spec)
[Three quick vignettes, each with its own window: a night-shift worker's afternoon "morning" (shift 10 PM-6 AM,
checklist keys / badge / phone / charger); a family kitchen ("Leo: bins out", "Ava: science fair poster"); a plant
control room.]
**N:** It isn't just for students: a night-shift worker's upside-down day, a family's chores, a plant floor. Twelve
watts at idle, offline, and your data never has to leave.

### 15 · Same engine, different data (spec, into NextEra)
[The morning brief's cards turn into a shift triage (Do this first / Keep an eye / Looks fine). The red pointer that
found the keys drops onto a plant floor plan and walks paths SPOT has actually walked. Numbers pop up with a citation
under each. The Learning lab tab flashes a test: "cites the right rows · every expected number". The same box sits on
a control-room console.]
**N:** Point the same box at a plant and the same pieces line up. The brief becomes a shift triage. The pointer
that found your keys walks a floor plan, over paths SPOT has actually walked.
**N:** Every number is computed from the inspection rows and cites them. And the lab re-trains the model on the
plant's own data and tests it before it's allowed on the box.

### 16 · Xander's shift (demo)
[Control room. Xander, hi-vis vest and coffee, sits down. On the big display and in the window: our Inspection page,
"Your shift". Through the window, SPOT at its dock.]
**X:** Morning, Atlas. What should I check first?
**A:** Three things. First, heat exchanger 2. Its alarm came from a check that failed, and it hasn't had a good
check in 88 hours. Send SPOT back before you send anyone.
[Window: three cards, Do this first / Keep an eye / Looks fine. HX-202 at the top.]
**X:** Where is it?
[A red pointer drops onto HX-202 on the floor plan and walks there along the traced corridors.]
**A:** Turn left, take the entrance steps, and go along the west passage.
**A:** Heads up: missions show ninety-six percent complete, but most hide a failed check. And under forty percent
battery, checks fail three times as often.
[Window: the trust line, then the battery explainer animating.]
**X:** Show me round 17.
[Window: the round replays. SPOT's dot moves along the plan, checks pop in (tick, cross, alarm), the battery bar
drains, the failed check is marked.]
**X:** Send SPOT back to it.
[Window: approval card, "Send SPOT back to heat exchanger 2?" Xander taps Yes.]
**A:** Done. The work order is on this box. Nothing left the building.
[The trainee leans in.]
**A:** Want to plan the next round?
[Window: Shift Run. The trainee drags the stops into an order and presses Run; the round plays out fast, the battery
drains, two stars of three, one line on why.]
[End of shift. Xander taps one button: "Shift handed off."]

### 17 · End (N)
[The Atlas logo draws in again above the box on its shelf in the dorm.]
**N:** Atlas. It sees your day, learns you, and keeps you to itself.

---

**Timing:** about 980 spoken words, roughly 5:07 of speech alone at the film's 1.2x voice rate, before the
action-only moments (walking, the Sort It round, the replay, the split screen) and the gaps `film_build.py` puts
between lines and scenes. **This is over the 5:00 cap.** `FILM_PLAN.md` section 10 has the numbers per beat and the
cuts for Dom to choose from; none are applied yet.

**Open:** the trainee's name; which lab numbers Ilan clears for beat 10 (Hemdutt's notes list candidates); Hemdutt's
stage 1 (beat 5's Tuesday line) and stage 2 (beat 6's "train your own model"); #311 for beat 12's dinner.
