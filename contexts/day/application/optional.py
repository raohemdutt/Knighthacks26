"""Optional events ("Near you") you can say yes or no to (the application side of contexts/day/domain/optional.py).

    optional = OptionalEvents(planner, tracker, clock, memory=memory)
    optional.subscribe(bus)                          # DayLoop does both
    optional.view(now)                               # GET /api/day/optional: each event with its fit, walk, leave-by and state
    events, body = optional.answer(now, id, "yes")   # POST /api/day/optional/{id}: what to publish, and what to answer

An optional event is a calendar item in Memory whose `external_id` starts with `nearby:`. Nothing treats it as a
commitment until the person says yes AND approves the card that yes raises:

- a yes never changes the plan. It builds a re-plan of the rest of the day with the event on the board as a fixed
  block (`day.replan.proposed`, `for_optional`) and the same approval card every re-plan gets; the answer
  (`day.optional.answered`) is published first. Only suggested blocks may move to make room: a re-plan that would
  touch a fixed event is refused, and so is an event that does not fit.
- once that card is approved (`day.plan.committed` of the plan that carries it) the event is accepted: from then on
  this object hands it to Day Run's board as a calendar item (the `agreed` port: `items`), so every later plan, the
  tracker (next event, leave-by, leave now, late), the narrator, the review and the Day Run screen carry it as an
  ordinary fixed block whose id starts with `optional:`. The item it hands over has the title, the time and the
  place and nothing else: the friends' names in the calendar item's tags never reach a plan block.
- a no is recorded and the event is never suggested again. A no to an accepted event also raises a re-plan without
  it, and it stays accepted until that card is approved.

State rebuilds from the log (every handler below is replay=True and pure): the answers, which plan carries which
optional block, which are accepted, and what was already suggested. Building a re-plan happens only when the phone
asks (the API), in event time.

This object sits in front of Day Run's `agreed` port only from the first yes on: a box nobody said yes on builds its
boards exactly as it did before.
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, tzinfo

from contexts.day.domain import optional as rule
from contexts.day.domain.clock import DayClock
from contexts.day.domain.optional import Fit, Offer, Touched
from contexts.play.dayrun.board import to_tick
from contracts.day import DayNarrated, DayOptionalAnswered, DayPlanCommitted, DayPlanProposed, DayReplanProposed, PlanMove
from contracts.events import Event
from contracts.memory import Item

log = logging.getLogger("atlas.day.optional")
CONNECTOR = "day"  # the calendar item an accepted event is handed to the board as
DAYS = 3  # the list covers today and this many days after it
KEEP_PLANS = 200  # plans whose optional blocks are remembered


class Refused(Exception):
    """An answer that cannot be taken, with the reason in plain words (the API answers 409 with it)."""

    def __init__(self, reason: str, status: int = 409) -> None:
        super().__init__(reason)
        self.reason, self.status = reason, status


def clock_time(t: datetime, tz: tzinfo) -> str:
    t = t.astimezone(tz)
    return f"{t.hour % 12 or 12}:{t:%M} {t:%p}"


class OptionalEvents:
    def __init__(self, planner, tracker, clock: DayClock, memory=None, days: int = DAYS) -> None:
        self.planner, self.tracker, self.clock, self.memory, self.days = planner, tracker, clock, memory, days
        self.inner = None  # what Day Run's `agreed` port held before this object took its place (the Circle's agreed plans)
        # --- state, rebuilt from the log ---
        self.answers: dict[str, DayOptionalAnswered] = {}  # offer id -> the latest answer
        self.carried: dict[str, frozenset[str]] = {}  # plan id -> the optional events its blocks carry
        self.dropped: dict[str, str] = {}  # plan id -> the optional event it was made to take out
        self.accepted: set[str] = set()  # said yes to, and the plan that carries it was approved
        self.until: dict[str, datetime] = {}  # offer id -> when its block ends, when a yes meant leaving early
        self.suggested: dict[str, datetime] = {}  # offer id -> when Atlas brought it up
        # --- live only ---
        self._try: dict[str, datetime | None] = {}  # offers on the board for the re-plan being built (id -> its end, or None)
        self._drop: set[str] = set()  # ... and accepted ones left off it

    def subscribe(self, bus) -> None:
        bus.subscribe("day.optional.answered", self.on_answered, replay=True)
        bus.subscribe("day.plan.proposed", self.on_plan, replay=True)
        bus.subscribe("day.replan.proposed", self.on_plan, replay=True)
        bus.subscribe("day.plan.committed", self.on_committed, replay=True)  # after the planner's own: it reads `current`
        bus.subscribe("day.narrated", self.on_narrated, replay=True)

    # --- state handlers (replay=True) ---

    def on_answered(self, e: DayOptionalAnswered) -> None:
        last = self.answers.get(e.id)
        if last is not None and e.ts < last.ts:
            return
        self.answers[e.id] = e
        if e.answer == "yes" and e.until is not None:
            self.until[e.id] = e.until
        elif e.answer == "yes":
            self.until.pop(e.id, None)

    def on_plan(self, e: DayPlanProposed | DayReplanProposed) -> None:
        self.carried[e.plan_id] = frozenset(o for m in e.moves if (o := rule.offer_id(m.id)))
        gone = getattr(e, "for_optional", None)
        if gone and gone not in self.carried[e.plan_id]:
            self.dropped[e.plan_id] = gone
        while len(self.carried) > KEEP_PLANS:
            old = next(iter(self.carried))
            del self.carried[old]
            self.dropped.pop(old, None)

    def on_committed(self, e: DayPlanCommitted) -> None:
        """The card of a plan was approved. It counts only when that plan is what the day is now measured against: a
        card approved after a newer plan replaced it changes nothing here either."""
        current = self.planner.current.get(e.day)
        if current is None or current.plan_id != e.plan_id:
            return
        self.accepted |= self.carried.get(e.plan_id, frozenset())
        self.accepted.discard(self.dropped.get(e.plan_id, ""))
        if self.accepted:
            self.hook()

    def on_narrated(self, e: DayNarrated) -> None:
        if e.moment == "suggest" and e.ref:
            self.suggested.setdefault(e.ref, e.ts)

    # --- Day Run's board (the `agreed` port) ---

    def hook(self) -> None:
        """Take the place in front of Day Run's `agreed` port; whatever was there (the Circle's agreed plans) is asked first."""
        dayrun = self.planner.dayrun
        if dayrun.agreed is not self:
            self.inner, dayrun.agreed = dayrun.agreed, self

    def items(self, lo: datetime, hi: datetime, tz: tzinfo, known: list[Item] | None = None) -> list[Item]:
        """What the port before this one hands over, plus every accepted optional event in [lo, hi) as a calendar item
        of its own: title, time and place only. One a real calendar item already covers (same title, same start) is
        left out, as the Circle's agreed plans do."""
        out = list(self.inner.items(lo, hi, tz, known)) if self.inner is not None else []
        want = (self.accepted | set(self._try)) - self._drop
        if not want:
            return out

        def key(title: str, when: datetime) -> tuple:
            return title.strip().lower(), when.astimezone(tz).replace(second=0, microsecond=0)

        taken = {key(i.title, i.when) for i in [*(known or []), *out]
                 if i.kind == "calendar_event" and i.when and not i.external_id.startswith(rule.PREFIX)}
        walks, done = self.tracker.walks, set()
        for i in known or []:
            if i.external_id not in want or i.external_id in done:
                continue
            offer = rule.parse(i, walks.locate, walks.name, tz)
            if offer is None:
                continue
            done.add(offer.id)
            end = self._try.get(offer.id) or self.until.get(offer.id) or offer.end
            if end <= lo or offer.start >= hi or end <= offer.start or key(offer.title, offer.start) in taken:
                continue
            until = "until " + end.astimezone(tz).strftime("%I:%M %p").lstrip("0")
            out.append(Item(connector=CONNECTOR, kind="calendar_event", external_id=rule.block_id(offer.id), title=offer.title,
                            when=offer.start, summary=f"{until} · {offer.place_name}" if offer.place_name else until))
        return sorted(out, key=lambda i: (i.when, i.external_id))

    @contextmanager
    def board_with(self, add: dict[str, datetime | None] | None = None, drop: set[str] | None = None):
        """While this is open the board carries `add` (offer id -> its end, None for the event's own) and leaves `drop`
        off: for building one re-plan, never for longer."""
        self.hook()
        self._try, self._drop = dict(add or {}), set(drop or ())
        try:
            yield
        finally:
            self._try, self._drop = {}, set()

    # --- reading ---

    def offers(self, now: datetime) -> list[Offer]:
        """Today's and the next days' optional events from Memory's calendar items, soonest first."""
        if self.memory is None:
            return []
        lo = self.clock.day_start(self.clock.day_of(now))
        try:
            items = self.memory.items(lo, lo + timedelta(days=self.days + 1), kind="calendar_event")
        except Exception:
            log.exception("could not read the optional events")
            return []
        walks, out = self.tracker.walks, {}
        for i in items:
            offer = rule.parse(i, walks.locate, walks.name, self.clock.tz)
            if offer is not None:
                out[offer.id] = offer
        return sorted(out.values(), key=lambda o: (o.start, o.id))

    def find(self, key: str, now: datetime) -> Offer | None:
        """An offer by its full id, or by its slug (the soonest one of that name that has not ended)."""
        offers = self.offers(now)
        exact = next((o for o in offers if o.id == key), None)
        if exact is not None:
            return exact
        same = [o for o in offers if o.slug == key]
        return next((o for o in same if o.end > now), same[0] if same else None)

    def walk(self, a: str | None, b: str | None) -> int:
        try:
            return self.tracker.walks.walk(a, b)[0]
        except Exception:  # a place the packs do not have: no walk to say
            return 0

    def plan(self, now: datetime):
        return self.planner.current.get(self.clock.day_of(now))

    def fit(self, offer: Offer, now: datetime, also: list[PlanMove] | None = None) -> Fit:
        """`offer` against today's plan as it stands (not a re-plan still waiting for the phone); `also` are blocks to
        count as well (other optional events said yes to whose card is still open)."""
        plan = self.plan(now)
        seen = self.tracker.where(now)
        return rule.fit(offer, [*(plan.moves if plan else []), *(also or [])], now, self.walk, seen[0] if seen else None)

    def state(self, offer: Offer, now: datetime) -> str:
        answer = self.answers.get(offer.id)
        return rule.state_of(offer, now, answer.answer if answer else None, offer.id in self.accepted)

    def waiting(self, offer_id: str, now: datetime) -> DayReplanProposed | None:
        """The re-plan about this event that is still waiting for the phone: the one a yes raised (it carries the
        block) or the one a no to an accepted event raised (it was made to take it out)."""
        for pending in self.planner.pending.values():
            adds = offer_id in self.carried.get(pending.plan_id, ()) and offer_id not in self.accepted
            removes = self.dropped.get(pending.plan_id) == offer_id and offer_id in self.accepted
            if (adds or removes) and self.card_open(pending, now):
                return pending
        return None

    def also_waiting(self, now: datetime, but: str) -> dict[str, PlanMove]:
        """The other optional events said yes to whose card is still open, with their blocks: the next re-plan carries
        them too, so a yes to two in a row loses neither (approving the newer card takes both)."""
        out: dict[str, PlanMove] = {}
        for pending in self.planner.pending.values():
            if not self.card_open(pending, now):
                continue
            for m in pending.moves:
                other = rule.offer_id(m.id)
                if other and other != but and other not in self.accepted:
                    out[other] = m
        return out

    def card_open(self, plan: DayReplanProposed, now: datetime) -> bool:
        p = self.planner
        if not plan.action_id or plan.plan_id in p.committed or p.card_status.get(plan.action_id):
            return False
        expires = p.expires.get(plan.action_id)
        return expires is None or now < expires

    def row(self, offer: Offer, now: datetime) -> dict:
        answer, pending = self.answers.get(offer.id), self.waiting(offer.id, now)
        said = self.suggested.get(offer.id)
        return {"id": offer.id, "slug": offer.slug, "title": offer.title, "start": offer.start.isoformat(), "end": offer.end.isoformat(),
                "place": offer.place, "place_name": offer.place_name, "tags": list(offer.tags), "going": list(offer.going),
                "host": offer.host, "source": offer.source, "state": self.state(offer, now),
                "answer": answer.answer if answer else None, "awaiting_approval": pending is not None,
                "action_id": pending.action_id if pending else None, "plan_id": pending.plan_id if pending else None,
                "suggested_at": said.isoformat() if said else None, "fit": fit_out(self.fit(offer, now))}

    def view(self, now: datetime) -> dict:
        plan = self.plan(now)
        return {"now": now.isoformat(), "plan_id": plan.plan_id if plan else None, "events": [self.row(o, now) for o in self.offers(now)]}

    # --- answering: a re-plan behind the usual card, never a change by itself ---

    def answer(self, now: datetime, key: str, answer: str) -> tuple[list[Event], dict]:
        """(the events to publish, the answer's body) for a yes or a no to the offer `key` (its id or its slug).
        LookupError when there is no such offer; Refused when the answer cannot be taken."""
        offer = self.find(key, now)
        if offer is None:
            raise LookupError(key)
        return self.yes(now, offer) if answer == "yes" else self.no(now, offer)

    def answered(self, now: datetime, offer: Offer, answer: str, replan: DayReplanProposed | None = None,
                 until: datetime | None = None) -> DayOptionalAnswered:
        return DayOptionalAnswered(event_id=f"day-optional-{offer.id}-{answer}-{now.timestamp():.0f}", ts=now, day=self.clock.day_of(now),
                                   id=offer.id, answer=answer, at=now, plan_id=replan.plan_id if replan else None,
                                   action_id=replan.action_id if replan else None, until=until)

    def replan(self, now: datetime, why: str, for_optional: str, draft: bool = False) -> DayReplanProposed | list[Event] | None:
        seen = self.tracker.where(now)
        events = self.planner.replan(now, seen[0] if seen else None, why, asked=True, draft=draft, for_optional=for_optional)
        if not events or not isinstance(events[0], DayReplanProposed):
            return None
        return events[0] if draft else events

    def yes(self, now: datetime, offer: Offer) -> tuple[list[Event], dict]:
        state = self.state(offer, now)
        if state == "accepted":
            raise Refused(f"{offer.title} is already in your plan")
        if state == "passed":
            raise Refused(f"{offer.title} has already started")
        body = {"id": offer.id, "answer": "yes", "state": "suggested", "proposed": False}
        cfg = self.planner.dayrun.cfg
        if offer.start >= to_tick(cfg, now) + timedelta(hours=cfg.window_hours):
            # Further ahead than a plan reaches: the answer is kept, nothing is proposed, and it is still on offer.
            reason = "It is more than a day ahead, so there is no plan to put it in yet. Atlas will bring it up before it starts."
            return [self.answered(now, offer, "yes")], {**body, "reason": reason}
        current = self.plan(now)
        if current is None:
            raise Refused("there is no plan for today yet to work it into; make one first")
        others = self.also_waiting(now, offer.id)
        fit = self.fit(offer, now, list(others.values()))
        if not fit.ok:
            raise Refused(f"{offer.title} does not fit: {fit.reason}")
        # Leaving early for the next fixed event is part of the yes: the block ends when you would have to go.
        until = offer.end - timedelta(minutes=fit.leave_early_minutes) if fit.leave_early_minutes else None
        mine = rule.block_id(offer.id)
        carry = {other: self.until.get(other) for other in others}
        try:
            with self.board_with(add=carry, drop={offer.id}):
                without = self.replan(now, "", offer.id, draft=True)  # the rest of the day re-planned without it, to compare
            with self.board_with(add={**carry, offer.id: until}):
                events = self.replan(now, f"You said yes to {offer.title} at {clock_time(offer.start, self.clock.tz)}.", offer.id)
        except Exception as exc:
            log.exception("the re-plan for an optional event failed")
            raise Refused(f"could not re-plan the day around {offer.title}") from exc
        replan = events[0] if events else None
        block = next((m for m in replan.moves if m.id == mine), None) if replan else None
        if block is None or without is None:
            raise Refused(f"could not work {offer.title} into the rest of the day")
        touched = fixed_changed(without.moves, replan.moves, mine)
        if touched or block.end != (until or offer.end) or block.start > max(offer.start, replan.start):
            what = touched[0].title if touched else offer.title
            raise Refused(f"{offer.title} does not fit: it would cut into {what}, which is fixed")
        moves, adds = self.diff(current, replan, now)
        body |= {"proposed": True, "plan_id": replan.plan_id, "action_id": replan.action_id, "moves": moves, "adds": adds,
                 "block": block.model_dump(mode="json"), "fit": fit_out(fit)}
        return [self.answered(now, offer, "yes", replan, until), *events], body

    def no(self, now: datetime, offer: Offer) -> tuple[list[Event], dict]:
        body = {"id": offer.id, "answer": "no", "state": "declined", "proposed": False}
        if offer.id not in self.accepted or now >= offer.end:
            return [self.answered(now, offer, "no")], body
        # Already a block of the day: taking it out is a change to the plan, so it goes through a card as well.
        current = self.plan(now)
        try:
            with self.board_with(drop={offer.id}):
                events = self.replan(now, f"You took {offer.title} out of today.", offer.id) if current else None
        except Exception:
            log.exception("the re-plan without an optional event failed")
            events = None
        if not events or events[0].action_id is None:
            raise Refused(f"could not re-plan the day without {offer.title}; it stays in your plan")
        replan = events[0]
        moves, adds = self.diff(current, replan, now)
        body |= {"state": "accepted", "proposed": True, "plan_id": replan.plan_id, "action_id": replan.action_id, "moves": moves, "adds": adds}
        return [self.answered(now, offer, "no", replan), *events], body

    def diff(self, current, replan: DayReplanProposed, now: datetime) -> tuple[list[dict], list[dict]]:
        """What the re-plan does to the plan it would replace, as far ahead as that plan reaches (a re-plan covers a day
        from now, so it always has later blocks the old plan never had): `moves` are the old plan's suggested blocks
        that are moved, shortened, lengthened or dropped; `adds` are the suggested blocks that are new. A re-plan
        chooses every suggested block again, so both can list more than what the event itself overlaps."""
        old = list(current.moves)
        reach = max((m.end for m in old), default=now)
        new = [m for m in replan.moves if m.start < reach]
        moves = [c for c in rule.changes(old, new, now, self.clock.tz) if not c["fixed"]]
        adds = [{k: c[k] for k in ("id", "title", "kind", "start", "end")}
                for c in rule.changes(new, old, now, self.clock.tz) if c["change"] == "dropped" and not c["fixed"]]
        return moves, adds


def fixed_changed(without: list[PlanMove], with_it: list[PlanMove], mine: str) -> list[PlanMove]:
    """The fixed blocks that adding the event changed: in the day re-planned without it but gone, moved, or cut in the
    day re-planned with it. A yes must leave this empty."""
    after = {m.id: m for m in with_it if m.fixed}
    return [m for m in without if m.fixed and m.id != mine
            and (m.id not in after or after[m.id].start != m.start or after[m.id].end != m.end)]


def touched_out(t: Touched) -> dict:
    return {"id": t.id, "title": t.title, "kind": t.kind, "start": t.start.isoformat(), "end": t.end.isoformat(),
            "overlap_minutes": t.overlap_minutes}


def fit_out(f: Fit) -> dict:
    return {"fits": f.fits, "reason": f.reason, "walk_minutes": f.walk_minutes, "leave_by": f.leave_by.isoformat() if f.leave_by else None,
            "from_place": f.from_place, "displaces": [touched_out(t) for t in f.displaces],
            "conflicts": [touched_out(t) for t in f.conflicts], "leave_early_minutes": f.leave_early_minutes}
