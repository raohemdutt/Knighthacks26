# Atlas film: the whole script (v4.3, about 4:45, cap 5:00)

This is the read-through. `FILM_PLAN.md` has the sources, the build list and the decisions behind every line.
**N** = narrator · **A** = Atlas · **H** = Hemdutt (the person) · **I** = Ilan · **D** = Dominick · **X** = Xander
(the operator) · **T** = the trainee. Each speaker has their own voice. [Brackets] are what we see. "Window" is the app
window in the top-right corner.

Demo beats: the product talks, no narrator. Spec beats: the narrator, short.

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
[Window: Day Run, the day playing on fast-forward on the campus map; then This week.]
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
[Labelled sparks fly into the box: Apple Watch, Outlook, Canvas, GroupMe, UCF Events, OpenStreetMap. One line goes out
to the phone: "Approve?"]
**N:** That brief came from your watch, your calendars, your chats and the campus map, written on the Jetson in
under three seconds. Nothing is booked until you say yes.

### 4 · The model on the box (spec, as now)
[The box goes see-through around the Jetson. Cards: "Qwen · on the Jetson", Fast, Accurate, Private, "42 / 42".]
**N:** A four-billion-parameter Qwen runs right here: fast, accurate, and private. Forty-two demo questions,
forty-two right.

### 5 · At the desk (demo)
[Afternoon. Hemdutt hunched at the desk, typing. Three cans by the keyboard. His face is tired. Atlas's lens wakes.]
**A:** You look tired: the look you get after a long stretch without moving. You've been sitting an hour and
forty-five minutes.
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
[His face inside the privacy shell, a small tag "looked tired · 30 s window" staying inside. A frame slides to the
shell's edge, and the face blurs exactly as it crosses.]
**N:** If you turn it on, Atlas reads your expressions on the device, to coach you and to train your own model.
Your face stays clear on the box, and it's blurred the moment a picture leaves it.

### 7 · The kitchen (demo)
[Hemdutt walks into the kitchen with his phone, the can and the bottle. He fills the bottle and steps toward the grey
trash bin. The wall camera watches.]
**A:** Hold on: cans go in recycling at UCF. The blue one.
[His phone and the window show Sort It: "Can goes in recycling", the blue bin lit, the route from "You". He turns and
drops it in the blue bin.]
**A:** Right bin. Five of seven this week, seventy-one percent. Your goal's eighty.
[Window: the ring moves from 67 to 71, the goal tick at 80.]

### 8 · Sort It (demo)
**A:** Want a quick round of Sort It? Hold up the next thing.
[He holds up a cup. The phone shows "Cup goes in trash" and the route. He walks it there. "Right bin · 6.0 s · 150
points".]

### 9 · What Sort It runs on (spec)
[The bin camera's view: boxes on the can and the cup, track numbers, the bin zones, a grid route bending round him.]
**N:** Sort It runs a YOLO detector at forty-one frames a second, tracks every item, maps the camera onto the
floor, and plans the walk around you. Only counts leave the box.

### 10 · The learning lab (spec, as now)
[Two AMD Strix Halo pods. Streams from Atlas boxes; a consent check; the test gate; Hugging Face.]
**N:** The lab is two AMD Strix Halo pods. One fine-tunes your own model on your own examples. The other, with
everyone's consent, learns from all of them, so every Atlas gets better. A test decides if the new model ships.

### 11 · On the device (spec, as now)
[The box see-through, the network cable pulled, chips: seeing, remembering, alerting, answering, learning (paused).
The ledger: 13 allowed, 13 blocked.]
**N:** Pull the network cable: it keeps seeing, remembering and answering. Only learning pauses. And a ledger shows
what left: thirteen destinations allowed, thirteen blocked.

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
[Three shells; only single words cross between them. Inside the group, the three friends' faces unblur to each other,
and soft links of learning grow between their models. Zoom out: many small groups across campus, each with its own
network.]
**N:** Each box reads only its own calendar and answers in one word. In a group you trust, friends can unblur
their faces to each other and learn from each other's examples, and every group runs its own network.

### 14 · Same box, other lives (spec)
[Three quick vignettes, each with its own window: a night-shift worker's afternoon "morning" (shift 10 PM-6 AM,
checklist keys / badge / phone / charger); a family kitchen ("Leo: bins out", "Ava: science fair poster"); a plant
control room.]
**N:** It isn't just for students: a night-shift worker's upside-down day, a family's chores, a plant floor. Twelve
watts at idle, offline, and your data never has to leave.

### 15 · One log, replayable (spec, into NextEra)
[The day we watched runs into the box as a ribbon of events, then rewinds. The ribbon turns into a robot's mission log
(rounds, checks, readings, alarms). The dorm dissolves into a plant's floor plan, and the same box sits on a
control-room console.]
**N:** Under all of it is one log of events Atlas can replay, minute by minute. Swap the profile, and that log holds
a robot's inspection rounds. The findings are computed in code, with error bars; the model only puts them into
words.

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
[Window: approval card, "Send SPOT back to heat exchanger 2?" He taps Yes.]
**A:** Done. The work order is on this box. Nothing left the building.
[The trainee leans in.]
**A:** Want to plan the next round?
[Window: Shift Run. The trainee drags the stops into an order and presses Run; the round plays out fast, the battery
drains, two stars of three, one line on why.]
[End of shift. Xander taps one button: "Shift handed off."]

### 17 · End (N)
[The Atlas logo draws in again above the box on its shelf in the dorm.]
**N:** Atlas. The personal AI computer. It learns you, and keeps you to itself.

---

**Timing:** about 830 spoken words, roughly 4:15 of speech at the film's 1.2x voice rate, plus the action-only
moments (walking, the Sort It round, the replay, the split screen). The first full build is timed. If it comes out
over 5:00, these go first, in this order: the dinner callback shortens to "Logged: goal met"; the trainee moment in
NextEra drops to one shot; the problem statement loses its second sentence.

**Open:** the trainee's name; whether Hemdutt's Notion page on natural speech adds anything beyond his merged code
(`contexts/day/domain/ear.py`).
