# Ask Atlas: what the box did with nine questions

Sent to the box's `POST /api/ask` (the route the Ask Atlas tab and voice use) on Sun 2026-10-11 at 8:09 AM ET,
box model `atlas-tool-calling-v2-a100-f2r4` (Qwen3.5-4B LoRA, llama-server, 4096 context, one slot). The path each
question took is read from the code on main at 0e33160 (contexts/agent/application/qa.py, talk.py, planner.py) and the
gateway journal; `GET /api/approvals` was empty afterwards.

## Before (main at 0e33160)

The tools the model could pick from: search_memory, last_seen, episodes_between, item_history, upcoming, get_keyframe,
prepare_for, web_search, reddit_search, read_page. Nothing for calendar writes, reminders, the camera, the room map,
the box's model, energy, the lab or the Circle. And most questions never reached the model's tool choice at all:
the code-first pre-answer (planner, pre-fetch, conversation) took them first.

| Question | Seconds | What came back | Path taken | Failure |
|---|---|---|---|---|
| What is on my calendar today? | 0.8 | "There are four things on campus today. First is Theatre UCF Open House at 11, then Dungeons & Dynos at noon. Want the rest?" | pre-fetch calendar path in code, no model | works (read only; only the campus feed has items on the box) |
| Add a meeting with Sam at 3 pm tomorrow called demo prep | 21.5 | A markdown study plan for today ("Sun 9:00 AM to 11:00 AM: Study demo prep materials ...") | the planner's regex took it ("prep" matches PLAN_QUESTION) | no tool exists, the gateway swallowed it, and nothing was proposed (approvals empty) |
| What did I do this morning? | 42.6 | 64 hourly camera captions read out from midnight ("Sun 12:00 AM hour: Multiple young men in white hoodies ...") | pre-fetch time window, then the model on the dump | chosen in code, wrong window (morning taken from midnight) and a record read out; slow |
| Where is my phone? | 6.5 | "Phone last seen Sun 7:53 AM in the desk zone on cam-desk, right of the bottle." | pre-fetch last_seen | works |
| Set a reminder to call mom at 6 | 3.2 | "I can't set reminders on this box, but you can set one on your phone." | the conversation (small talk) | no tool exists |
| Turn off the camera | 11.7 | "To turn off the camera on a Mac, close the program that's using the camera ..." | tool loop, the model chose web_search | no tool exists; the model chose the web |
| Show me the room map | 10.6 | "Roomstyler provides a 3D room planning tool ..." | tool loop, web_search | no tool exists; the model chose the web |
| What is the box's model right now? | 3.3 | "I don't know the box's model, but I'm glad to chat with you on it." | the conversation | no tool exists |
| How much energy did my last answer cost? | 2.4 | "I don't know how much energy that cost, but I'm glad we're chatting." | the conversation | no tool exists |

In short: two of nine work, and both are answered in code before any model looks at them. The calendar add is the
worst one: the planner's keyword match turns it into a study plan, and no approval card is ever made.

Correction to row 3 after reading the journal: "What did I do this morning?" went to the tool loop, where the 4B chose
`episodes_between` from midnight; the answer is that tool's raw hourly record (three model calls, 42.6 s).

## After (branch ctx/agent/ask-tools)

Sent at 8:33 AM ET to this branch's gateway running on the workstation, with its model calls going to the box's own
llama-server over an SSH tunnel (the same 4B, the same weights, sharing the box's one slot with its caption traffic).
The memory is the demo fixture (fixtures/memory/memory.yaml, moved to now) and connectors are off, so the calendar is
the fixture's and there is no calendar connector and no camera capture on the workstation. Every row was routed by the
box's model: no rule and no network brain was needed.

| Question | Seconds | Tool the 4B chose (arguments) | What came back |
|---|---|---|---|
| What is on my calendar today? | 6.3 | calendar.list (day 2026-10-11) | "You've got the submissions due at 9:59, the judging and demos at 10:59, then your COP 4533 study session at 12:59 PM." |
| Add a meeting with Sam at 3 pm tomorrow called demo prep | 8.8 | calendar.add (title demo prep, start 3 pm tomorrow) | "I have proposed demo prep for your calendar, tomorrow at 3. Approve it on your phone." The card is on GET /api/approvals. |
| What did I do this morning? | 8.0 | memory.recap (part morning) | "The cameras saw 15 moments in 4 hours. The busiest was around 6 AM: the person opens the laptop and starts typing." |
| Where is my phone? | 8.3 | memory.last_seen (label phone) | "I haven't seen your phone." (the fixture has no phone; on the box this reads its sightings) |
| Set a reminder to call mom at 6 | 12.2 | reminders.add (text call mom, at 6 pm) | "I have proposed a reminder to call mom at 6 tonight. Approve it on your phone." The card is on GET /api/approvals. |
| Turn off the camera | 8.1 | camera.pause (action pause) | "I can't pause the cameras from here. You can stop them from the Privacy screen." (no capture on the workstation; on the box this proposes the pause) |
| Show me the room map | 7.5 | room.map | "Here is the map of the dorm room: 2 cameras and 4 zones, like door, recycling and trash." The Ask tab opens /map/. |
| What is the box's model right now? | 1.6 | box.model | "The box is serving atlas-base, built on Qwen/Qwen3.5-4B." (the workstation's name for the model; on the box it is the deployed package's) |
| How much energy did my last answer cost? | 7.4 | box.energy_last_answer | "I don't have a power reading for the last answer yet." (no power sampler on the workstation; the box samples the Jetson's rails) |

A direct probe of the box's llama-server with the same prompt and tools (8:20 AM ET) chose the same tool for all nine,
in 1.2 to 5.5 s each, and answered NONE for "How are you doing today?" (which then goes to the conversation as before).
The extra seconds above are the wait for the box's one llama-server slot while it captions frames.

Still to do on the box itself, after the dispatcher stages this branch and restarts the gateway: rerun the nine
(`python tooling/demo/ask_atlas_nine.py http://100.124.34.36:8000 after-box.jsonl`) and add that table here.
