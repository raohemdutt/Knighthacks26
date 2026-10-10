# The Circle

Friends who also run Atlas. Each person's Atlas plans with the others: one proposes a time, each of the
others checks **its own person's calendar on its own box** and answers one word (free, busy, or another
time), and when everyone is free the event is on the shared calendar and in each attendee's day.

What leaves a box is only: a proposal after its approval card, a one-word reply, a named place with an
expiry (opt-in), and daily summaries / counts / scores (opt-in). Never pictures, raw signals or the calendar.

| Piece | Where |
| --- | --- |
| Message shapes, the store port, propose / reply / agree | `application/protocol.py` |
| Named places with an expiry | `application/presence.py`, `presence_sync.py` |
| Invite and group message after an agreement (approval cards) | `application/delivery.py` |
| The shared calendar the phone reads | `application/calendar.py` |
| **The conversations the phone reads** | `application/threads.py` |
| Endpoints | `api.py` (`/api/circle/events`, `/sharing`, `/proposals`, `/threads`, `/threads/{id}`, `/members`) |
| Neon tables | `infrastructure/schema.sql`, `infrastructure/neon.py` |
| The phone screen | `web/app/circle/page.tsx`, `web/components/circle/` |
| An agreed plan as a fixed block of the day | `contexts/play/dayrun/agreed.py` |
| The group brain: opt-in planning questions to the teacher, text only | `application/group_brain.py`, `infrastructure/teacher.py`, `brain_api.py`, `brain.py` |

The phone's Circle screen (`/circle/`) has four parts: **People**, **Plans**, **League**, **Sharing**.
Links open a part or a conversation directly: `/circle/?tab=plans&thread=<thread id>`.

## The group brain (G4)

Hard planning across members ("where should we meet tonight?") can be asked of the big model, the Qwen3.8-27B
teacher on the lab pod in the `pod-a` role (`LAB_POD_ROLES`; account C on SJC today), through the hub proxy. It is
the one place a lab GPU is in a request path, so it is never needed: a failure raises `BrainUnavailable` and the
caller answers with the box's own model.

- **Opt-in only.** Nothing is sent until the member switches it on: `POST /api/circle/brain/consent {"on": true}`
  writes `circle.brain.consent.changed` to the log, and every ask names that event's `consent_id` as its
  `opt_in_ref`. Off again is the same call with `false`. The switch outlives a restart (replayed from the log).
- **Text only.** A question and its facts are strings; anything else, or a data URL inside the text, is refused
  before anything is sent or logged, and the transport refuses the same again at the wire. No picture, frame or file.
- **Every send is logged first.** `circle.brain.asked` (exactly the messages sent, the pod role, the hub, the model
  name, the priority and the token cap) is in the log before the request leaves; `circle.brain.answered` follows with
  the answer or the failure's kind. If the log cannot be written, nothing is sent.
- **Small and one at a time.** One question in flight per box (another is refused, not queued), at most 6000
  characters and 40 facts, answers capped at 256 tokens (about 90 s at the Strix Halo teacher's 2.9 tok/s, streamed),
  thinking off, sent at priority -10 so the teacher serves it before labeling (`lab/README.md`, Teacher).
- **Shown.** Every answer carries the badge "Asked the group brain".

Use it from the planner (G5's "planned by the group brain"):

```python
from contexts.circle.application.group_brain import BrainRefused, BrainUnavailable
try:
    answer = await group_brain.ask(question, evidence_lines, purpose="where_to_go")
    text, badge = answer.text, answer.badge          # show the badge with the answer
except (BrainRefused, BrainUnavailable):
    ...                                              # not opted in, busy or unreachable: the box's own model answers
```

Wiring (not in `bootstrap.py` yet; one block when the Circle is configured):

```python
from contexts.circle.application.group_brain import GroupBrain
from contexts.circle.infrastructure.teacher import TeacherLink
group_brain = GroupBrain(circle_cfg["circle_id"], circle_cfg["member_id"], TeacherLink.from_env(), bus.publish)
bus.subscribe("circle.brain.consent", group_brain.on_consent, replay=True)   # state only
# app.py: app.include_router(group_brain_router(atlas.group_brain, guard=authorized))   (contexts/circle/brain_api.py)
```

From a terminal (keys from `.env`; events to a folder of its own, never the box's `data/events`):

```bash
python -m contexts.circle.brain status       # which name answers, "group-brain" or "teacher"; sends no member text
python -m contexts.circle.brain ask --log-dir /tmp/brain-log --circle team --member ilan --opt-in \
  --fact "Maya is at the Library until 9:00 PM" "Where should we meet tonight?"
```

## Conversations (`GET /api/circle/threads`, `/api/circle/threads/{id}`)

Rebuilt from this box's event log, then filled in from the Circle's store when it can be read:

| In the log | Event |
| --- | --- |
| This box's own proposal: the card, then the send | `agent.action.proposed` (kind `circle_share`, `params.kind: event_proposal`), `workspace.action.completed` |
| Every other member's proposals and replies | `circle.message.received` |
| The agreement | `circle.event.agreed` |

Two things the log does not hold, and how the screen deals with them:

- **This box's own replies** are posted straight to the store and never logged. They appear when the store
  can be read (`"source": "log+store"`). From the log alone, an agreed plan still shows this member as free,
  because an agreement means every attendee answered free; that line is marked `"source": "agreement"`.
- **A counter-offer the proposer accepts is a new thread** in the protocol (a new `thread_id`). It is shown as
  the same conversation: a proposal by the same member, with the same title, at exactly the time a reply in
  an earlier, unagreed thread suggested. `legs` lists every thread id; any of them opens the conversation.

Each message has a `line` in plain language, written from its structured fields only: "Maya's Atlas: Busy
then. Suggests 3:00 PM." A reply carries one word and at most a time, so a line never gives a reason.

States: `proposed` (your card is waiting; nothing sent), `waiting`, `countered`, `agreed`, `declined`.

## One box, no network: the seeded demo

A believable Circle of three on one machine, from the log alone. Nothing here touches Neon or a second box.

```bash
python -m contexts.demo circle --data-dir /tmp/atlas-circle
# it prints the command to run the app on it:
ATLAS_DATA_DIR=/tmp/atlas-circle ATLAS_CONFIG_DIR=/tmp/atlas-circle/config ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 \
  python -m contexts.gateway
# then open http://127.0.0.1:8000/circle/   (set ATLAS_PORT for another port)
```

It seeds the demo night (the same as `python -m contexts.demo seed`, ending now) plus the Circle `demo-circle`:
Hemdutt (this box), Maya and Dominick; Maya at the Library and Dominick at the Student Union; "COP4533 exam
review" agreed after Maya's Atlas countered an hour later; a meal Maya proposed, agreed directly; and "Gym",
asked ten minutes ago, with Dominick free and Maya not answered. The League has Maya's week from the fixture.

The committed `config/agent.yaml` never names a member. The demo writes its own copy of the config directory
(`<data-dir>/config`, or `--config-dir`) with `circle.member_id: hemdutt` and `circle.circle_id: demo-circle`.
`--end 2026-10-11T09:30:00-04:00` pins the story's end instead of using the current time; seeding the same
directory again needs `--force` and adds nothing (every event has a fixed id).

On one box nobody answers a **new** proposal: the friends here are events in the log, not running boxes. A plan
you start shows its card, then "…'s Atlas is checking the calendar", then "no answer yet". For real answers,
use two boxes (below). To throw the demo away: `rm -rf /tmp/atlas-circle`.

## Two boxes, live: the real demo

Box **A** is the Jetson (Hemdutt), box **B** a laptop (Dominick). Both need the same Neon database and the
same circle id. **Not run as part of this change**: the timings are the ones measured between two boxes earlier
(a friend's place visible within about 20 s; a free reply about 28 s after the approval; the agreement about
57 s after it). Each box talks to Neon on its 30 s timer, which is where those numbers come from.

### Config

On both boxes: `pip install -e ".[neon]"`, and `NEON_DATABASE_URL=` in `.env` (the same pooled connection
string on both; see `config/.env.example`). The tables are `contexts/circle/infrastructure/schema.sql`; apply
it once to a new database.

Keep the committed config clean by running each box on a copy:

```bash
cp -R config /tmp/circle-config        # on each box, then edit /tmp/circle-config/agent.yaml
```

Box A (`/tmp/circle-config/agent.yaml`, the `circle:` section):

```yaml
circle:
  member_id: hemdutt
  circle_id: kh-demo-1          # the same on both boxes; a new name is a clean slate
  share_presence: true
  share_summaries: true          # so the League has both weeks
  names: {hemdutt: Hemdutt, dom: Dominick}   # only the two of you: a plan needs every named member to say free
```

Box B: the same, with `member_id: dom`.

Run each box:

```bash
ATLAS_CONFIG_DIR=/tmp/circle-config python -m contexts.gateway
```

Phone A opens `http://<box A>:8000/circle/`, phone B `http://<box B>:8000/circle/`. The top of the screen must
not say "This box has not joined a Circle", and Sharing must say the database is connected.

### What to tap, and what appears

| Step | Tap | Appears | Within |
| --- | --- | --- | --- |
| 1. A place | Phone B: People, "Share where you are", Library, 2 hours | Phone A, People: Dominick's card shows "Library, until …"; "Route to Dominick" gives the walk | about 20 s |
| 2. A plan | Phone A: Plans, "Find a time for all of us"; Dominick, "Study session", "Tomorrow afternoon", a place; "Ask their Atlases" | Phone A: "… is waiting for your OK. Nothing has been sent"; the plan under "Being worked out" as **Needs your OK** | at once |
| 3. Approve | Phone A: Approvals, approve the card. Then Plans, open the conversation | "Hemdutt's Atlas: Can everyone do Study session, tomorrow 2:00 to 4:00 PM at …?" and "Dominick's Atlas is checking Dominick's calendar…" | at once |
| 4. The answer | nothing | Phone A: "Dominick's Atlas: Free." Phone B, Plans: the same conversation, proposed by Hemdutt, with its own "Free." | about 28 s after the approval |
| 5. Agreed | nothing | Both phones: "Agreed: tomorrow 2:00 PM, …, 2 of 2 in.", an "Everyone's in" alert, the plan under "Coming up", and a fixed block in the day (Now / Day Run) when it starts within 24 hours. Box A also raises the calendar-invite card if its calendar connector can write | about 57 s after the approval |

The counter-offer, which is the part worth showing:

1. Put something on Dominick's calendar at the time you will propose. With his calendar connector on, a real
   event does it. Without one, push an item into box B (it is input the box accepts, like a camera frame):

   ```bash
   curl -X POST http://<box B>:8000/api/events -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $ATLAS_TOKEN" \
     -d '[{"type":"connector.item.synced","connector":"calendar","kind":"calendar_event","external_id":"demo-lecture","title":"Lecture","when":"2026-10-11T14:00:00-04:00","summary":"until 3:00 PM"}]'
   ```

   (Use the day and hour you will propose for `when`; leave the `Authorization` header out when the box has no
   `ATLAS_TOKEN`.) A calendar item counts as one hour.
2. Phone A proposes 2:00 to 4:00 PM that day and approves the card.
3. About 28 s later the conversation on phone A shows "Dominick's Atlas: Busy then. Suggests 3:00 PM." and the
   chip turns to **Another time suggested**. Box A raises a second card at the same moment: "dom is busy for
   "Study session" and suggested Sun 3:00 PM: ask the circle about the new time?"
4. Phone A approves it on Approvals (a card lasts 5 minutes; if it ran out, the conversation offers "Ask their
   Atlases about 3:00 PM", which raises it again).
5. The same conversation continues: "Hemdutt's Atlas: Can everyone do tomorrow 3:00 to 5:00 PM instead?",
   "Dominick's Atlas: Free.", "Agreed: tomorrow 3:00 PM, …, 2 of 2 in." About 57 s after the second approval.

The League part needs no tapping: with `share_summaries: true` each box's closed habit days reach the other's
standings (League tab) on the next timer tick after a day closes.

### Reset

- **Cleanest:** choose a new `circle_id` on both boxes (`kh-demo-2`) and restart them. Nothing is deleted;
  the old circle's rows stay in Neon and no screen reads them.
- Each box's own log still holds the old conversations. For an empty screen start each box on a fresh data
  directory too: `ATLAS_DATA_DIR=/tmp/circle-data-2 ATLAS_MEMORY_FIXTURE= …`.
- To remove a throwaway circle from Neon for good, `DELETE FROM circles WHERE id = 'kh-demo-1';` (everything in
  it cascades). Only ever with a demo circle id, never the team's.

## What the protocol could give the screen (not changed here)

`application/protocol.py` is Dominick's. The conversation view works around three things there; each would be
a small additive change:

1. `CircleAgent.poll` posts this box's own reply to the store without an event, so the log never holds what
   this box answered. An event for it (or returning the posted `AgentMessage`) would make a conversation
   complete from the log alone and let the screen update the moment the reply is posted.
2. `CircleAgent._follow_up` starts a new `thread_id` for the counter time and nothing links it to the thread it
   answers. A `parent_thread_id` in the card's `params` would replace the title-and-time match in `threads.py`.
3. `Reply` has no reason field, so a line can only say "Busy then". If a friend's Atlas may ever say why
   ("has a lecture until 3:00"), that is a contract and privacy decision for the team, not a screen change.
