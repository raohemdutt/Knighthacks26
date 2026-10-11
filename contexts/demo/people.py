"""People for the demo: who the People screen lists on a laptop that has no camera and no face model.

    python -m contexts.demo circle --data-dir <dir>      # always includes it
    python -m contexts.demo seed --people                # the plain night, plus the two people who need no Circle

DEMO DATA. Nobody was enrolled and no face was ever seen: every "template" here is a made-up list of numbers
(a random unit vector from a fixed seed, see `vector`), stored under the model name `demo-synthetic`, which no
recognizer has, so the box can never match it to a real face. Every member written here carries `demo: true`
(contexts/perception/infrastructure/identity_store.py), which the People screen shows as a "Demo data" tag, the
service refuses to share (`IdentityService.own_template`), to match (`_templates`) and to add pictures to.

What is written, through the identity service and its store (never a hand-made vault blob):

- the owner (`m-demo`, the member the check-in story reads, contexts/demo/checkin.py): enrolled by scan, "read my
  expressions" on, linked to this box's own Circle member when there is a Circle;
- one person saved from photos only: kept, not recognised, "scan to finish" on the People screen;
- with a Circle only, one friend whose own box shared their template (`IdentityService.receive_shared`, so it is
  a member with `source: "circle"` that runs out in thirty days like any other).

And a week of `perception.expression.observed` for the owner, on the desk camera:

- inside the story (its last two days), only while the story itself has a person at the desk: one stretch every
  three minutes, mostly neutral, more tired late at night. Never strained for half of any twenty minutes, so the
  Day loop's check-in rule (contexts/day/domain/checkin.py) has nothing to raise from it;
- before the story, a morning stretch each day, and an afternoon one that reads as strained on the days the
  check-in story calls "like this" (6, 4 and 2 days back, at the hours in `checkin.FELL_AT`, with its shares).
  Days the story already covers are left to the story.

Nothing else is published: no `perception.person.identified` (Habits and the Impact story count from those, and
the story's numbers must stay what they are) and no `perception.identity.changed` (the app says that itself when
it starts on the directory). The looks are published after the story and after its closing jobs, with the
check-in's live look held off, so no plan, review, nudge or alert of the story can change. Nothing here goes to
the Circle, to Neon or into an export: the templates stay in the data directory's vault, the looks in its log.

Deterministic and safe to run twice: event ids are fixed (`demo-people-...`), and a member already there is left alone.
"""

import hashlib
import math
import random
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta

from contexts.demo.checkin import DESK_CAM, FELL_AT, LIKE_THIS, MEMBER
from contexts.demo.story import LOCAL
from contexts.perception.application.identity import SHARED, shared_member_id
from contexts.perception.infrastructure.identity_store import Member
from contracts.events import Event
from contracts.perception import ExpressionObserved, ObservationRecorded

MODEL = "demo-synthetic"  # not a recognizer: a template under this name is never compared with a face
DIMENSIONS = 128  # what SFace gives, so the blob has the shape of a real one
OWNER, PHOTOS_ONLY, FRIEND = MEMBER, "m-demo-photos", "maya"  # FRIEND is a Circle member id (contexts/demo/circle.py)
NAMES = {OWNER: "Hemdutt", PHOTOS_ONLY: "Dominick", FRIEND: "Maya"}
OWNER_CIRCLE, PHOTOS_CIRCLE = "hemdutt", "dominick"
ENROLLED_AGO = {OWNER: timedelta(days=9), PHOTOS_ONLY: timedelta(days=3, hours=2), FRIEND: timedelta(days=5, hours=6)}
SHARE_DAYS = 30
WEEK = 7  # days of looks
WINDOW = timedelta(minutes=3)
GAP_S = 10  # the story sees a seated person every 2 s: a longer silence ends the stretch
MORNING = (time(9, 30), 52)  # what the check-in story's earlier days had: 52 of 60 minutes at the desk
AFTERNOON = (time(15, 0), 45)
CALM = {"neutral": 0.7, "happy": 0.2, "sad": 0.1}
UPBEAT = {"happy": 0.55, "neutral": 0.4, "surprised": 0.05}
TIRED = {"neutral": 0.55, "sad": 0.35, "angry": 0.1}  # late at night; 45% strained, under the check-in rule's half
STRAINED = {"sad": 0.36, "angry": 0.28, "neutral": 0.36}  # the check-in story's own shares


def vector(name: str, n: int = 0) -> list[float]:
    """A made-up unit vector, the same every time for the same name. It was computed from no picture."""
    rng = random.Random(f"atlas-demo-people:{name}:{n}")
    raw = [rng.gauss(0.0, 1.0) for _ in range(DIMENSIONS)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _mean(vectors: list[list[float]]) -> list[float]:
    return [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]


def members(now: datetime, circle: bool) -> list[Member]:
    """The locally "enrolled" demo people (the friend comes in through `receive_shared`, see `write`)."""

    def at(who: str) -> str:
        return (now - ENROLLED_AGO[who]).isoformat()

    eight = [vector(OWNER, n) for n in range(8)]
    three = [vector(PHOTOS_ONLY, n) for n in range(3)]
    return [
        Member(member_id=OWNER, name=NAMES[OWNER], enrolled_at=at(OWNER), model=MODEL, frames=8, vector=_mean(eight), vectors=eight,
               expressions=True, updated_at=at(OWNER), source="scan", consent_at=at(OWNER),
               circle_member_id=OWNER_CIRCLE if circle else None, scanned=True, demo=True),
        Member(member_id=PHOTOS_ONLY, name=NAMES[PHOTOS_ONLY], enrolled_at=at(PHOTOS_ONLY), model=MODEL, frames=3, vector=_mean(three),
               vectors=three, expressions=False, updated_at=at(PHOTOS_ONLY), source="photos", consent_at=at(PHOTOS_ONLY),
               circle_member_id=PHOTOS_CIRCLE if circle else None, scanned=False, demo=True),
    ]


def write(identity, now: datetime, circle: bool) -> list[str]:
    """Store the demo people through the identity service's store and its own way in for a shared template.
    A member that is already there is left as it is. Returns the ids written now."""
    written = []
    for m in members(now, circle):
        if identity.store.get(m.member_id) is None:
            identity.store.save(m)
            written.append(m.member_id)
    friend_id = shared_member_id(FRIEND)
    if circle and identity.store.get(friend_id) is None:
        four = [vector(FRIEND, n) for n in range(4)]
        received = (now - ENROLLED_AGO[FRIEND]).astimezone(UTC)  # UTC: thirty days later is thirty days later across a clock change
        # The way a friend's template really arrives; the events it returns are not published (the app announces
        # who is enrolled when it starts). Then the times are set to the story's and the demo mark is put on.
        identity.receive_shared(FRIEND, NAMES[FRIEND], MODEL, _mean(four), four, expires_at=received + timedelta(days=SHARE_DAYS),
                                share_id="demo-people-share-maya")
        got = identity.store.get(friend_id)
        assert got is not None and got.source == SHARED
        identity.store.save(replace(got, enrolled_at=received.isoformat(), updated_at=received.isoformat(),
                                    received_at=received.isoformat(), demo=True))
        written.append(friend_id)
    return written


def _shares(start: datetime, n: int) -> dict[str, float]:
    hour = start.astimezone(LOCAL).hour
    if hour >= 22 or hour < 6:
        return TIRED
    pick = int(hashlib.sha256(f"demo-people:{n}".encode()).hexdigest(), 16) % 4
    return UPBEAT if pick == 0 else CALM


def _look(start: datetime, shares: dict[str, float]) -> ExpressionObserved:
    end = start + WINDOW
    return ExpressionObserved(event_id=f"demo-people-looks-{int(end.timestamp())}", ts=end, camera_id=DESK_CAM, member_id=OWNER,
                              window_start=start, window_end=end, dominant=max(shares, key=shares.get), shares=dict(shares),
                              samples=40, confidence=0.8, looks=52, head_down=0.1, turned_away=0.08)


def _windows(start: datetime, end: datetime) -> list[datetime]:
    """Whole three-minute windows inside [start, end)."""
    out = []
    while start + WINDOW <= end:
        out.append(start)
        start += WINDOW
    return out


def desk_stretches(story: list[Event]) -> list[tuple[datetime, datetime]]:
    """When the story has a person in front of the desk camera, as (from, to)."""
    seen = sorted(e.ts for e in story if isinstance(e, ObservationRecorded) and e.camera_id == DESK_CAM
                  and any(t.label == "person" for t in e.tracks))
    out: list[tuple[datetime, datetime]] = []
    for ts in seen:
        if out and (ts - out[-1][1]).total_seconds() <= GAP_S:
            out[-1] = (out[-1][0], ts)
        else:
            out.append((ts, ts))
    return out


def looks(now: datetime, story: list[Event]) -> list[ExpressionObserved]:
    """The owner's week at the desk camera, oldest first: the days before the story, then the story's own stretches."""
    begins = min((e.ts for e in story), default=now)
    today = now.astimezone(LOCAL).date()
    starts: dict[datetime, dict[str, float] | None] = {}  # window start -> shares (None: chosen by `_shares`)
    for back in range(WEEK - 1, -1, -1):
        day = today - timedelta(days=back)
        sessions = [(MORNING, None), ((FELL_AT[back], 14 + 2 * back), STRAINED) if back in LIKE_THIS else (AFTERNOON, None)]
        for (at, minutes), shares in sessions:
            start = datetime.combine(day, at, LOCAL)
            for w in _windows(start, start + timedelta(minutes=minutes)):
                if w + WINDOW <= begins:  # the story's own days are told by the story
                    starts[w] = shares
    for a, b in desk_stretches(story):
        for w in _windows(a, min(b, now)):
            starts.setdefault(w, None)
    ordered = sorted(starts)
    kept: list[datetime] = []
    for w in ordered:  # no two windows overlap
        if not kept or w >= kept[-1] + WINDOW:
            kept.append(w)
    return [_look(w, starts[w] or _shares(w, n)) for n, w in enumerate(kept)]


async def publish(app, now: datetime, story: list[Event], settle) -> tuple[list[str], int]:
    """Write the people and publish the owner's looks through `app`. Returns (member ids written now, looks handed
    to the bus). Call it after the story and its closing jobs: see the module docstring for why."""
    if app.identity is None:
        return [], 0
    written = write(app.identity, now, circle=app.circle is not None)
    events = looks(now, story)
    checkin = app.day.checkin if app.day is not None else None
    was = checkin.enabled if checkin is not None else False
    if checkin is not None:
        checkin.enabled = False  # these are a week of looks arriving at once, not an afternoon to speak up about
    try:
        for e in events:
            await app.bus.publish(e)
            await settle()
    finally:
        if checkin is not None:
            checkin.enabled = was
    return written, len(events)
