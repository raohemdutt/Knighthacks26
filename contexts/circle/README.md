# The Circle

Friends who also run Atlas. Each person's Atlas plans with the others: one proposes a time, each of the
others checks **its own person's calendar on its own box** and answers one word (free, busy, or another
time), and when everyone is free the event is on the shared calendar and in each attendee's day.

What leaves a box is only: a proposal after its approval card, a one-word reply, a named place with an
expiry (opt-in), and daily summaries / counts / scores (opt-in). Never pictures, raw signals or the calendar.
One more thing can leave, and only when its own switch is on (it is off by default): a person's own face
template, as ciphertext only one named friend's box can read, after a card on both boxes. See "Face share" below.

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
| **Face share** (off by default): a friend's box recognises you | `application/sealed.py` (the envelope), `application/face_share.py`, `face_share_api.py`, `infrastructure/neon_sealed.py` |

The phone's Circle screen (`/circle/`) has four parts: **People**, **Plans**, **League**, **Sharing**.
Links open a part or a conversation directly: `/circle/?tab=plans&thread=<thread id>`.

## What is in the database

One shared Postgres database (Neon) is the only place two boxes meet. Only Circle data is in it. Cameras,
pictures, Memory, the day plan, health, approvals and the event log stay on each box and never go there.

| Table | What a row is | Who writes it | How long it stays |
| --- | --- | --- | --- |
| `circles` | a circle id | the first box that writes anything for that circle | kept |
| `members` | a member id, the display name from that box's `circle.names`, "has a box" | a member's own box, on its first write (never overwritten after) | kept |
| `agent_messages` | one plan message: a proposal (title, start, end, place, invitees) or a reply (free / busy / another time) | the proposer's box after its approval card; each invitee's box for its reply | the writer's box removes its own after `circle.keep_days` (30) |
| `events`, `rsvps` | a plan everyone agreed, and who is in | the proposer's box, once, when the last "free" arrives | kept |
| `shares` | the receipt of one opt-in summary a box sent (kind and the same few fields) | that box's outbox worker | the writer's box removes its own after `circle.keep_days` |
| `habit_summaries` | member, habit, day: met or not, the day's value, the streak | the member's box when a habit day closes (`share_summaries`) | kept (the League reads 21 days back) |
| `impact_counters` | a member's latest Waste Watch totals | the member's box (`share_summaries`) | one row per member, overwritten |
| `game_scores` | one scored Sort It round | the member's box (`share_summaries`) | kept |
| `presence` | the named place a member shares now, and when it ends | the member's box (`share_presence`); "stop sharing" ends the row | one row per member, overwritten; removed by its own box once expired for `circle.keep_days` |
| `league_stakes`, `league_weeks`, `league_challenges` | the week's stake, a declared week's standings, a dare | the box whose person set, closed or started it | kept |
| `sealed_messages` | face share only (off by default): a box's public keys, a face template as ciphertext for one friend's box, a revoke | the sharing box | the boxes delete a share when it is used, taken back or revoked |

Two views, `habit_leaderboard` and `impact_leaderboard`, are read by the Circle screen.

**Set up.** `pip install -e ".[neon]"`, then `NEON_DATABASE_URL=` in `.env` (the pooled connection string, the
same on every box in the circle). `schema.sql` is the intended state. Check the database against it, and create
what is missing, once for a new database and again whenever `schema.sql` gains a table or an index:

```bash
python -m contexts.circle.schema --check    # read-only: what differs between the database and schema.sql
python -m contexts.circle.schema --apply    # creates the missing tables, indexes and views; nothing else
python -m contexts.circle.schema --counts   # read-only: --check plus columns, indexes, sizes, rows per circle
```

None of the three prints the connection string or a row's contents (ids are shortened to two letters).
`--apply` runs `schema.sql`'s own `CREATE ... IF NOT EXISTS` statements in one transaction and only for objects
that do not exist; it cannot alter, replace or drop anything, and it leaves an existing view as it is. A column
that `schema.sql` gained on an existing table is reported by `--check` and needs a written migration.
(`psql "$NEON_DATABASE_URL" -1 -f contexts/circle/infrastructure/schema.sql` does the same and also re-creates
the two views.) A table that is missing (for example `sealed_messages` on a database set up before face share)
makes that feature's timer job fail on every tick with an error in the log; the rest of the Circle keeps working.

**Which box is in which circle.** `circle.member_id` and `circle.circle_id` in `config/agent.yaml`, or without
editing the committed file: `ATLAS_CIRCLE_MEMBER`, `ATLAS_CIRCLE_ID`, `ATLAS_CIRCLE_SHARE=1` (opt in to places
and summaries) and `ATLAS_CIRCLE_FACE_SHARE=1` in `.env`. Without `NEON_DATABASE_URL` a box keeps its Circle in
memory: nothing is shared with another box.

The repository's `neon.ts`, `package.json` and the `.neon` / `.env.local` files (PR #287) are the Neon command
line's project link (`neon link`, `neon deploy`) and the S3 client for Neon Object Storage. The Python app reads
none of them: it uses `NEON_DATABASE_URL` only, and stores no objects. `neon.ts` declares nothing, so `neon
deploy` does not create or change tables; `schema.sql` is still how they get there.

**When the database cannot be reached** the box keeps working and says so on the Sharing part of the screen:

| What | While offline | When it is back |
| --- | --- | --- |
| Summaries, counts, scores | wait in `data/circle.sqlite` (`circle_outbox`), across restarts | sent oldest first on the next tick |
| A place you share, or "stop sharing" | the latest one waits in `data/circle.sqlite` (`presence_outbox`), across restarts | sent on the next tick; a place that ran out meanwhile is never sent, a stop always is |
| Friends' places | the ones already shown stay until they run out | a friend's stop takes effect on the next tick |
| A plan you approve | the card ends as failed with "Neon is unreachable"; nothing is queued | propose it again |
| Replies and agreements | not read, not answered | read and answered on the next tick (a poll that failed half-way starts over) |
| League declarations | kept in the log | pushed on the next tick |

The box talks to the database only from its 30 s timer and from a few screen reads, always off the event loop.
A connect attempt gives up after 10 s; after a failure every call fails at once for 15 s, so a tick pays for one
attempt instead of one per job; a job that gets no answer in 20 s is dropped and tried again on the next tick.

**When the network blocks port 5432.** Some networks refuse Postgres's port and leave 443 open. The box can
reach the same database over Neon's HTTPS SQL endpoint instead (`infrastructure/neon_https.py`; standard library
only, so it also works without the `neon` extra). `ATLAS_NEON_TRANSPORT` in `.env` chooses:

| Value | What it does |
| --- | --- |
| `auto` (default) | psycopg on 5432. If that cannot connect (refused, timed out, driver not installed), HTTPS on 443 for the rest of the process; 5432 is tried again only after a restart. One log line says which is in use. |
| `tcp` | psycopg on 5432 only, as before. |
| `https` | HTTPS on 443 only. Set this before a demo on a network known to block 5432: `auto` spends up to 10 s finding out. |

The schema tool follows the same setting and prints the transport on its first line:
`ATLAS_NEON_TRANSPORT=https python -m contexts.circle.schema --check`.

What is different over HTTPS:

- **Transactions.** There is no open transaction over HTTP. Statements wait until the code needs an answer or
  finishes, and go as one request that commits as a whole. A write that only inserts (a share, a place, a plan
  message, a sealed message, a League row) is as atomic as over TCP. Two calls read an answer in between and so
  commit in two steps: recording an agreed event (the event first, its RSVP rows second: if the second request is
  lost the calendar shows that event with nobody listed, and a repeat does not add them) and the daily cleanup
  (three independent deletes; a half-finished sweep is finished by the next one).
- **Offline is the same.** A refused connection, a timeout, an HTTP error or a sign-in page instead of an answer
  all count as "the database cannot be reached": the outbox and the place queue wait and the tick carries on. A
  request gives up after 10 s.
- **Egress.** The request goes to the host named in `NEON_DATABASE_URL`, on 443 instead of 5432. The Jetson's
  egress allowlist (`jetson/egress.py`, group `neon`) allows that host by address on any port, so it needs no change.
- **Secrets.** The connection string travels in a request header over TLS. It is never logged, and errors say
  only what kind of failure it was. A redirect is never followed.

The request and reply shapes are written down in `infrastructure/neon_wire.py`, the one file to correct if the
service differs from what it assumes.

**Clocks.** A place's expiry is stored and read against the database's clock, so a box whose own clock is wrong
neither hides a friend's place early nor shows it late. Plan messages carry the writer's own time; each poll
re-reads the last 10 minutes to allow for a slow clock.

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

Wiring: `bootstrap.py` builds it with `contexts/circle/brain_power.py`'s `build_shared_brain(circle_cfg, bus)` when the
Circle is configured and the teacher pod's account (the pod-a role of `LAB_POD_ROLES`) is in `.env`; `app.py` serves
`group_brain_router` and `brain_power_router`. The Circle screen's "Shared brain" card (People tab) reads
`GET /api/circle/brain/power`: the teacher pod's power sampler (`lab/pod_power.py`, port 8790) relayed by the box with
the hub token kept server side, its 10.27 W idle floor, and the joules of the last group-brain answer (the pod's power
integrated over `circle.brain.asked` to `circle.brain.answered`, moved onto the pod's clock by the offset measured from
the sampler's own timestamps; gross, with the idle figure beside it). What `build_shared_brain` does:

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

## Face share: a friend's box recognises you (off by default)

Until now a friend was recognised by your box only if they enrolled on your box. With face share, a person who
is enrolled on their **own** box can let the boxes of **named** friends in the same Circle recognise them. It was
decided by Hemdutt and built without waiting for the protocol's owner, so it is additive, off by default and has
one switch. Nothing in `protocol.py`, `neon.py`, `api.py` or `agent_messages` was changed for it.

**Switch.** `circle.face_share: true` in `config/agent.yaml`, or `ATLAS_CIRCLE_FACE_SHARE=1` in `.env`, then restart.
It is independent of the recognition switch and of `ATLAS_CIRCLE_SHARE`. `circle.face_share_days` (30, at most 90)
is how long a share lasts.

- **Off, and never switched on:** no key is made, nothing is read from or written to the Circle for it, no file is
  created, and every `/api/circle/face-share` endpoint answers 409 with the reason.
- **Switched off after it was on:** at the next poll this member's shares are revoked, the templates friends shared
  with this box are deleted, the box's public key is taken out of the Circle, and a share that still arrives is
  ignored and written to the log as ignored (`circle.face_share.changed`, `state: ignored`, `reason: switched_off`).

**The flow** (each step has a test in `tests/test_face_share.py`):

| Step | On whose box | What happens |
| --- | --- | --- |
| 0 | both | Each box makes a key pair once and publishes the public half. Each shows its own fingerprint and the friend's. |
| 1 | the sharer's | People, "Friends who can recognise you", **Share**: an approval card. Nothing has left the box. |
| 2 | the sharer's | **Approve**: the template is sealed to that one friend's box and one row is written to the Circle. |
| 3 | the friend's | Next poll (30 s): a card, "<name> wants your Atlas to recognise them. Accept?". Nothing is decrypted yet. |
| 4 | the friend's | **Approve**: decrypted, stored in that box's own vault as a member with `source: "circle"`, `shared_by`, `received_at`, `expires_at`. The row is removed from the Circle. It is used only while recognition is on there; the card says so when it is off. |
| later | the sharer's | **Revoke**: a signed revoke; the friend's box deletes the template at its next poll. |
| later | the friend's | **Forget**, a wipe of everything, or the expiry: deleted. The expiry needs no network. |

**Only the template, only one's own.** What is sealed is the mean vector, the exemplar vectors, the display name and
the Circle member id (`IdentityService.own_template`): about 37 kB at the most, refused above 256 kB. Never a photo,
a frame, a crop or expression history. A box shares only the one person enrolled on it, with consent, who is linked
to that box's own Circle member ("Which Circle member is this?" on the People screen). A visitor who enrolled on your
box is not yours to share, and a template your box received cannot be shared on, edited or added to.

**The envelope** (`application/sealed.py`; every primitive is from the `cryptography` package, which the vault
already uses; nothing was added to the dependencies and no cipher was written here):

- Each box has an Ed25519 signing key and an X25519 encryption key. The private halves are in the box's vault
  (`data/vault/circle/face_share.bin`, AES-256-GCM under the vault's key) and never leave it.
- A share: a fresh X25519 key pair per message; X25519(ephemeral, friend's public key); HKDF-SHA256 over the result
  with the protocol name, the ephemeral key, the friend's key and the sender's signing key as `info`;
  ChaCha20-Poly1305 with a random 96-bit nonce, and the header as associated data; then an Ed25519 signature over
  the header, the ephemeral key, the nonce and the ciphertext.
- The header is `{v, kind, circle_id, from, to, share_id, seq, issued_at, expires_at, model}`. Changing who it is
  from or to, its counter or its dates breaks both the signature and the decryption.
- A revoke is `{v, kind, circle_id, from, to, seq, issued_at}` with a signature and no ciphertext.
- `seq` only goes up per sender. A friend's box remembers the highest counter it has seen and the highest it was
  told to revoke: an old share put back into the database after a revoke (or after a forget) is refused.

**Rows** (`contracts.circle.SealedMessage`, version 1, the `sealed_messages` table; apply `schema.sql` again to add
it): `box_key` (to everyone: two public keys, self-signed), `face_share` (to one member: header, `eph_pub`, `nonce`,
`ciphertext`, `sig`), `face_revoke` (to one member: header, `sig`). The table has no CHECK on `kind`, and a box
skips rows whose kind or version it does not know. It is a separate channel on purpose: `CircleAgent.poll` turns
every `agent_messages` row into a `circle.message.received` whose `kind` is a closed list, so a new kind in that
table would break the poll of a box that does not know it. A box without this code never reads `sealed_messages`.

**Limits, said plainly.**

- **Keys are trusted on first use.** A box keeps the first key it sees for a member. Someone who controls the
  shared database could put their own key under a friend's name before your box has seen the real one, and then
  read what you share with that friend. The defence is to **compare fingerprints**: your screen shows the friend's
  box key, theirs shows their own, and they must match (and the other way round). A key that changes later is
  never used silently: sharing with that friend stops until you compare and trust the new one.
- **Whoever runs the database can hold messages back.** That includes a revoke. The expiry is the backstop: the
  friend's box deletes the template after 30 days on its own clock, with or without the Circle.
- **The box cannot tell whose face a template is.** "This box's own person" is the enrolled person linked to
  this box's Circle member, with the consent that was ticked at enrolment. Whoever holds the box's token sets
  that link, so the rule keeps a box from sharing a visitor by accident, not a box owner who lies to it.
- **A friend's box that accepted has the template.** Revoke and expiry are carried out by that box's software; a
  box that was modified, or a copy of its disk, keeps what it was given. Share with friends you trust.
- **No forward secrecy.** A share is removed from the database once it is accepted, revoked, replaced or expired.
  If one is copied from the database before that and the friend's box key is stolen later, it can be read.
- **Who shares with whom is visible** to anyone who can read the database: member ids, times, the expiry, the
  face model's name and the size. The template and the name kept with it are not.
- A share works only between boxes with the same face model (the header names it; the card says when it differs).
- A wipe of everything deletes the templates friends shared and takes this member's own shares back (when the
  Circle can be reached; otherwise they run out). It keeps the box key, as it keeps the vault's master key.
- **Not run against the real Neon database or with real faces.** The store is tested in memory and on the fake
  connection; the faces in the tests are bytes the fake recogniser turns into vectors.

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
