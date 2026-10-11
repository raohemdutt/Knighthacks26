"""One arbiter for everything Atlas says unasked: which line goes out now, which waits, which is merged, which is dropped.

Pure: no clock, no bus, no I/O. The caller (contexts/day/application/arbiter.py) feeds it what was said (from the
log, so a restart knows) and a `Scene` of the moment, and asks `decide` about each line that is due.

The rules, in order:

1. One ordering, highest first: urgent now (leave now, late) -> approval asks (a plan, a re-plan, a card) ->
   check-in -> block start -> optional-event suggestion -> coach note -> evening review -> habit nudges. Alerts
   about the room are a separate urgent class: they may interrupt anything except another urgent line.
2. One global gap between two unasked speak-ups (`Rules.gap_s`, default 90 s), which urgent lines break. Each
   producer's own cooldown stays underneath.
3. Two lines due within the gap, neither urgent: merged into one sentence when the second can be said as a clause
   of the first (a block start and a suggestion), else the lower one waits for the next free slot. A plan's own
   approval card, due with the plan's narration, is said inside it (the narration already asks).
4. Suppression: no block start for an event that had a leave-now or late line in the last 10 minutes; no
   suggestion while a check-in or an approval card is open or within 15 minutes after a check-in; no coach note
   during a fixed event; nothing non-urgent while an answer is being said (it waits, and is dropped once it is
   older than `stale_s`); nothing outside the wake/bed window except urgent lines.
5. One speaker per line: the caller decides who says it (`spoken_by`); this module never speaks.
6. One push per moment: `key(day, moment, ref)` is the dedupe key of a push.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

# Lower is said first. Morning and re-plan narrations carry their own approval ask, so they rank with approvals.
RANK = {"urgent": 0, "alert": 0, "approval": 1, "morning": 1, "replan": 1, "checkin": 2, "start": 3, "suggest": 4,
        "coach": 5, "evening": 6, "nudge": 7}
# What may be said as the tail clause of another line ("...; and Riley's dinner is at quarter past five if you want it").
TAILS = ("suggest", "start")
# A held line of these moments waits for its slot however long it takes; the others go stale (rule 4).
PATIENT = ("approval", "morning", "replan", "checkin", "evening")
CHECKIN_CARD_TTL = timedelta(minutes=60)  # an unanswered check-in card stops counting as open after this long


@dataclass(frozen=True)
class Rules:
    gap_s: float = 90.0  # day: {speak_gap_s: 90}
    stale_s: float = 120.0  # a held line older than this is dropped (unless PATIENT)
    after_urgent_min: int = 10  # no block start for an event that had a leave-now / late line this recently
    after_checkin_min: int = 15  # no suggestion this soon after a check-in

    @classmethod
    def of(cls, raw: dict | None) -> "Rules":
        """`raw` is agent.yaml's `day:` block."""
        raw = raw or {}
        return cls(gap_s=float(raw.get("speak_gap_s", 90)), stale_s=float(raw.get("speak_stale_s", 120)))


@dataclass(frozen=True)
class Line:
    """One unasked line that is due: its moment, when it became due, and what it is about.

    `about` holds the plan block / event ids the line concerns (a block start's blocks, a leave-now's event);
    `plan_id` lets a plan's approval card be recognised as part of the plan's own narration."""

    moment: str
    ts: datetime
    ref: str | None = None
    urgent: bool = False
    plan_id: str | None = None
    about: frozenset[str] = frozenset()

    @property
    def rank(self) -> int:
        return RANK["urgent"] if self.urgent else RANK.get(self.moment, len(RANK))


@dataclass(frozen=True)
class Scene:
    """The moment as the caller sees it: whether the person is awake (the wake/bed window), whether a fixed block
    is running, whether an answer is being said, and whether a card is open."""

    awake: bool = True
    fixed_running: bool = False
    answering: bool = False


@dataclass
class Said:
    moment: str
    ts: datetime
    urgent: bool
    plan_id: str | None
    event_id: str | None


@dataclass
class Ledger:
    """What has been said and what is open, rebuilt from the log by the caller (all `note_*` are pure updates)."""

    last: Said | None = None  # the newest unasked speak-up (any kind, the day's or another context's)
    urgent_about: dict[str, datetime] = field(default_factory=dict)  # event/block id -> last leave-now / late line
    checkin_at: datetime | None = None
    checkin_open: dict[str, datetime] = field(default_factory=dict)  # check-in id -> raised at, until answered
    cards: dict[str, datetime] = field(default_factory=dict)  # open approval cards: action id -> expires at

    def note_said(self, moment: str, ts: datetime, urgent: bool = False, plan_id: str | None = None,
                  event_id: str | None = None) -> None:
        if self.last is None or ts >= self.last.ts:
            self.last = Said(moment, ts, urgent, plan_id, event_id)

    def note_urgent(self, ids, ts: datetime) -> None:
        for i in ids:
            if i and (i not in self.urgent_about or ts >= self.urgent_about[i]):
                self.urgent_about[i] = ts

    def note_checkin(self, checkin_id: str, ts: datetime) -> None:
        if self.checkin_at is None or ts >= self.checkin_at:
            self.checkin_at = ts
        self.checkin_open[checkin_id] = ts

    def note_checkin_answered(self, checkin_id: str) -> None:
        self.checkin_open.pop(checkin_id, None)

    def note_card(self, action_id: str, expires_at: datetime) -> None:
        self.cards[action_id] = expires_at

    def note_card_closed(self, action_id: str) -> None:
        self.cards.pop(action_id, None)

    def cards_open(self, now: datetime) -> bool:
        return any(now < exp for exp in self.cards.values())

    def checkin_card_open(self, now: datetime) -> bool:
        return any(now - at < CHECKIN_CARD_TTL for at in self.checkin_open.values())


@dataclass(frozen=True)
class Verdict:
    """`say`: out now. `companion`: said inside the last line (`into` is its event id): published silent.
    `hold`: wait for the next free slot. `drop`: not said at all. `reason` says why, for logs and tests."""

    action: str
    reason: str = ""
    into: str | None = None


def key(day: date, moment: str, ref: str | None) -> str:
    """The dedupe key of a push: one push per moment."""
    return f"{day.isoformat()}:{moment}:{ref or ''}"


def suppressed(line: Line, scene: Scene, ledger: Ledger, now: datetime, rules: Rules) -> str | None:
    """Why this line must not be said at all now (rule 4), or None. Urgent lines are never suppressed."""
    if line.urgent or line.moment == "alert":
        return None
    if not scene.awake:
        return "outside the wake/bed window"
    if line.moment == "start":
        recent = timedelta(minutes=rules.after_urgent_min)
        if any(i in ledger.urgent_about and now - ledger.urgent_about[i] < recent for i in line.about):
            return "a leave-now or late line about it was said in the last 10 minutes"
        if scene.fixed_running:
            return "a fixed event is running"
    if line.moment == "suggest":
        if ledger.cards_open(now):
            return "an approval card is open"
        if ledger.checkin_card_open(now):
            return "a check-in card is open"
        if ledger.checkin_at is not None and now - ledger.checkin_at < timedelta(minutes=rules.after_checkin_min):
            return "a check-in was said in the last 15 minutes"
    if line.moment == "coach" and scene.fixed_running:
        return "a fixed event is running"
    return None


def decide(line: Line, scene: Scene, ledger: Ledger, now: datetime, rules: Rules = Rules()) -> Verdict:
    """What to do with one due line now. The caller says the line (`say`), publishes it silent as part of the last
    one (`companion`), keeps it for its next tick (`hold`) or forgets it (`drop`)."""
    why = suppressed(line, scene, ledger, now, rules)
    if why:
        return Verdict("drop", why)
    if line.urgent:
        return Verdict("say", "urgent")
    if line.moment == "alert":
        last = ledger.last
        return Verdict("say", "alert") if last is None or not last.urgent or (now - last.ts).total_seconds() >= rules.gap_s \
            else Verdict("hold", "another urgent line was just said")
    if (now - line.ts).total_seconds() > rules.stale_s and line.moment not in PATIENT:
        return Verdict("drop", "stale")
    if scene.answering:
        return Verdict("hold", "an answer is being said")
    last = ledger.last
    if last is None or (now - last.ts).total_seconds() >= rules.gap_s:
        return Verdict("say", "the slot is free")
    if line.moment == "approval" and line.plan_id and last.plan_id == line.plan_id and last.moment in ("morning", "replan"):
        return Verdict("companion", "the plan's narration already asks", into=last.event_id)
    return Verdict("hold", "within the global gap")


def order(lines: list[Line]) -> list[Line]:
    """Highest priority first; the earlier due first within one rank."""
    return sorted(lines, key=lambda x: (x.rank, x.ts))


def mergeable(first: Line, second: Line) -> bool:
    """Whether `second` can be said as a clause at the end of `first` (rule 3): neither urgent, the second a tail."""
    return not first.urgent and not second.urgent and second.moment in TAILS and first.moment != second.moment \
        and first.moment not in ("approval", "alert", "nudge")
