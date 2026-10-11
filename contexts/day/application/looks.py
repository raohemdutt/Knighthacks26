"""The look memory: Atlas remembers a sustained look with what was going on around it, calls back to it (stage one),
and learns from all of them, per person, what tends to ease a look and what the looks share (stage two).

    looks = LookMemory(planner, coach, checkin, clock, habits=habits)
    looks.subscribe_looks(bus); ...; looks.subscribe(bus)   # DayLoop does both, around the check-in
    back = looks.callback(now)         # the check-in asks when a look is one of its signals

"You've looked strained for the last twenty minutes. That's the same look you had Tuesday afternoon, when you'd gone
two hours without moving. Ten minutes away from the desk helped then." The rule for what is remembered, what is
similar and what is said is contexts/day/domain/looks.py; nothing here calls a model. The kind of look kept with a
moment is "tired" or "tense" (the People screen's own two words for a strained look); a callback is only ever to
the same kind.

A pure projection of the log (every handler below is replay=True and uses event time), so it rebuilds on a restart
and replays identically. No new event is published for a moment or its outcome: no other context needs one. What it
reads:

- how the person looked: `perception.expression.observed` on the desk camera, for a member (never the anonymous room);
- whose looks may be remembered: `perception.identity.changed`. Its `expressions` list is the members who did the face
  scan AND switched "read my expressions" on. Once the log has one, only those members are remembered, and a member who
  leaves the list (switched it off, or was forgotten) has every remembered look CLEARED here: turning it off stops the
  remembering and takes the memories with it. Before the log has one, Perception's own rule is what holds (it publishes
  a member's looks only for a scanned member who switched it on: contexts/perception/application/identity.py);
- the desk: `perception.zone.event` (kept here too, so an outcome can be read after the coach has let go of the day).
  Where the desk zone has no record that reaches the moment (no zone drawn, or the zone engine was not running), the
  member's own windows of looks stand in: an unbroken run of them (no gap of three minutes) is that member in front
  of the desk camera, and a gap of three minutes or more is them gone from it;
- a drink: a pick-up the check-in's own `DrinkWatch` completed on a desk observation, or a desk caption that says
  someone is drinking (its DRINK_VERBS); a container in view from the same tracks and the desk's object list. Only
  times are kept, never the frame. Kept here by time, so an event that reaches the log late still lands on its day;
- the watch: the check-in's nights, as two booleans (short night, resting heart rate above the person's own usual);
- what they were doing: the Day plan's block at that time; a deadline within a day from `connector.item.synced`;
- energy drinks so far against the limit: Habits' digest, through the port the Day loop already has;
- what Atlas suggested and how it was answered: `day.checkin.raised` / `day.checkin.answered`;
- what was forgotten: `day.look.forgotten` (box-only), which is what keeps a forgotten moment gone on replay.

Stage two, what it has learned (the rule and the words are contexts/day/domain/look_model.py; nothing here calls a
model and nothing reads a picture). `model(member)` fits a small table of counts from `examples(member)` and from
nothing else, every time it is asked: it is never stored, so it is exactly as old as the remembered looks, and
forgetting them (one, all, a member, expressions off, a full wipe) forgets it. It is a pure function of the moments,
so a replay of the log gives the same model and the same sentences. With enough known outcomes a check-in's advice
order and its "what followed" sentence come from it ("Getting up has eased this four of the last five times; water
alone hasn't either time."), the card may carry one quiet line when the person has been at the desk about as long as
it usually takes (`ahead`; agent.yaml `day: {look_ahead: false}` turns it off; never said, never pushed), and three
more questions are answered from it (`said`: "helps", "causes", "how"). With too few, everything is stage one's.

Box-only, all of it: the moments live in this object and are served to the owner's own screen by GET /api/day/looks.
They are in no event, so no Circle handler, shared table, export, push or Learning lab reader can carry them; the one
place their words go is the box-only `day.checkin.raised` (its `callback`). Nothing is logged at info level.
"""

import logging
from bisect import bisect_left
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta

from contexts.day.domain import checkin as rule
from contexts.day.domain import look_model
from contexts.day.domain import looks as memory
from contexts.day.domain.checkin import Look
from contexts.day.domain.clock import DayClock
from contexts.day.domain.coach import DeskPresence, Span
from contexts.day.domain.looks import Context, Moment
from contexts.day.domain.narration import Words, short_title, the
from contracts.connectors import ItemSynced
from contracts.day import CheckinCallback, DayCheckinAnswered, DayCheckinRaised, DayLookForgotten
from contracts.events import Event
from contracts.perception import ExpressionObserved, ObservationRecorded, SceneDescribed, ZoneEvent

log = logging.getLogger("atlas.day.looks")
KEEP_LOOKS = timedelta(hours=2)  # raw windows: enough for the rule and for an outcome
KEEP_DESK = timedelta(days=1)
SAME_VIEW = timedelta(seconds=60)  # a container in view is noted about once a minute, not once a second
KEEP_RAISED = 50
ENERGY = "energy_drinks"  # the habit's id in config/habits.yaml
SHOWN = 5  # "Looks Atlas remembers": the most recent five


class LookMemory:
    def __init__(self, planner, coach, checkin, clock: DayClock, habits=None, words: Words = Words(), look_ahead: bool = True) -> None:
        self.planner, self.coach, self.checkin, self.clock, self.habits, self.words = planner, coach, checkin, clock, habits, words
        self.look_ahead = look_ahead  # the card's quiet "you usually start to look strained around now" (never said or pushed)
        # --- state, rebuilt from the log ---
        self.allowed: set[str] | None = None  # who scanned and switched expressions on; None: the log has not said yet
        self.looks: dict[str, list[Look]] = {}
        self.labels: dict[str, dict[datetime, dict[str, float]]] = {}  # member -> window end -> the strained labels' counted looks
        self.faces: dict[str, list[Span]] = {}  # member -> the windows in which the desk camera had their face, counted or not
        self.presence = DeskPresence()
        self.drinks: list[datetime] = []  # when a drink was seen picked up (times only), in time order
        self.containers: list[datetime] = []  # when a drink container was in view, about once a minute, in time order
        self.dues: dict[str, tuple[str, datetime]] = {}  # assignment -> (short title, when it is due)
        self.moments: dict[str, Moment] = {}
        self.open: dict[str, str] = {}  # member -> the moment whose stretch is still holding
        self.pending: set[str] = set()  # moments whose outcome is not known yet
        self.raised: list[tuple[datetime, str, tuple[str, ...]]] = []  # check-ins that had a look among their signals
        self.answers: dict[str, str] = {}
        self.seen: datetime | None = None  # the latest event time this object has taken in

    def subscribe_looks(self, bus) -> None:
        """Before the check-in subscribes: a check-in raised by a window of looks then finds that window already here."""
        bus.subscribe("perception.identity", self.on_identity, replay=True)
        bus.subscribe("perception.expression", self.on_expression, replay=True)

    def subscribe(self, bus) -> None:
        """After the check-in: its drink watch has taken the same desk event in by the time `on_desk` reads it."""
        bus.subscribe("perception.zone", self.on_zone, replay=True)
        bus.subscribe("perception.observation", self.on_desk, replay=True)
        bus.subscribe("perception.scene", self.on_desk, replay=True)
        bus.subscribe("connector.item.synced", self.on_item, replay=True)
        bus.subscribe("day.checkin.raised", self.on_raised, replay=True)
        bus.subscribe("day.checkin.answered", self.on_answered, replay=True)
        bus.subscribe("day.look.forgotten", self.on_forgotten, replay=True)

    # --- state handlers (replay=True) ---

    def on_identity(self, e: Event) -> None:
        """Who may be remembered. Anyone no longer on the list is cleared: their windows and every remembered look."""
        self.allowed = set(getattr(e, "expressions", ()) or ())
        self.looks = {m: rows for m, rows in self.looks.items() if m in self.allowed}
        self.labels = {m: rows for m, rows in self.labels.items() if m in self.allowed}
        self.faces = {m: rows for m, rows in self.faces.items() if m in self.allowed}
        self.moments = {i: m for i, m in self.moments.items() if m.member_id in self.allowed}
        self.open = {m: i for m, i in self.open.items() if m in self.allowed}

    def on_zone(self, e: ZoneEvent) -> None:
        if e.label != "person" or e.zone_id != self.coach.desk_zone:
            return
        self.presence.add(e.ts, (e.camera_id, e.track_id), e.kind, e.dwell_s)
        self.settle(e.ts)
        if e.kind == "exit":
            self.presence.prune(e.ts - KEEP_DESK)

    def on_desk(self, e: ObservationRecorded | SceneDescribed) -> None:
        """After the check-in took the same event in: did its DrinkWatch complete a pick-up on it, does a caption say
        someone is drinking, is a container in view."""
        if e.camera_id != self.coach.desk_camera:
            return
        if isinstance(e, SceneDescribed):
            drank = e.zone_id is None and bool(rule.DRINK_VERBS.search(e.caption or ""))
            in_view = e.zone_id == self.coach.desk_zone and any(o.label.lower() in rule.DRINK_LABELS for o in e.objects)
        else:
            in_view = any(t.label.lower() in rule.DRINK_LABELS for t in e.tracks)
            watch = self.checkin.drinks
            drank = in_view and bool(watch.moments) and watch.moments[-1][0] == e.ts
        if drank and all(abs(e.ts - d) >= rule.SAME_DRINK for d in self.drinks):
            self.drinks = sorted([d for d in self.drinks if d >= e.ts - KEEP_DESK] + [e.ts])
        if in_view and all(abs(e.ts - c) >= SAME_VIEW for c in self.containers[-3:] + self.near(self.containers, e.ts)):
            self.containers = sorted([c for c in self.containers if c >= e.ts - KEEP_DESK] + [e.ts])
        self.settle(e.ts)

    @staticmethod
    def near(times: list[datetime], ts: datetime) -> list[datetime]:
        i = bisect_left(times, ts)
        return times[max(0, i - 1):i + 1]

    def in_view(self, at: datetime, either_side: bool = False) -> bool:
        """A drink container was in view within the check-in's CONTAINER_WITHIN before `at` (or after it, for an outcome)."""
        lo = bisect_left(self.containers, at - rule.CONTAINER_WITHIN)
        return any(c <= at + (rule.CONTAINER_WITHIN if either_side else timedelta(0)) for c in self.containers[lo:lo + 1])

    def on_item(self, e: ItemSynced) -> None:
        if e.kind != "assignment" or e.when is None:
            return
        self.dues[e.external_id] = (short_title(e.title), e.when)
        if len(self.dues) > 200:
            self.dues = {k: v for k, v in self.dues.items() if v[1] >= e.ts - timedelta(days=2)}

    def on_raised(self, e: DayCheckinRaised) -> None:
        if not any(s.kind == "expression" for s in e.signals) or any(i == e.event_id for _, i, _ in self.raised):
            return
        self.raised = sorted([*self.raised, (e.at, e.event_id, tuple(s.kind for s in e.suggestions))])[-KEEP_RAISED:]
        self.attach()

    def on_answered(self, e: DayCheckinAnswered) -> None:
        self.answers[e.checkin_id] = e.answer
        self.moments = {i: (m if m.checkin_id != e.checkin_id else memory.with_outcome(m, memory.replace(m.outcome, answer=e.answer)))
                        for i, m in self.moments.items()}

    def on_expression(self, e: ExpressionObserved) -> None:
        """One window of looks. When the check-in's own rule holds over the last twenty minutes, a stretch is under
        way: its first window makes ONE moment, the later ones only move its end."""
        if e.member_id is None or e.camera_id != self.coach.desk_camera:
            return  # the anonymous room count is nobody's
        if self.allowed is not None and e.member_id not in self.allowed:
            return  # not scanned, or expressions are off: nothing is remembered
        now = e.window_end
        if e.samples > 0 or (e.looks or 0) > 0:  # their face was in front of the desk camera, counted or turned away
            seen = self.faces.setdefault(e.member_id, [])
            seen.append((e.window_start, e.window_end))
            if len(seen) > 6000:
                self.faces[e.member_id] = [w for w in seen if w[1] >= now - KEEP_DESK]
        if e.samples <= 0:
            self.settle(now)
            return
        rows = self.looks.setdefault(e.member_id, [])
        rows.append(Look(e.window_start, e.window_end, e.samples, rule.strained_looks(e.shares, e.samples)))
        labels = self.labels.setdefault(e.member_id, {})
        labels[e.window_end] = {label: share * e.samples for label, share in e.shares.items() if label in memory.LOOK_WORDS}
        self.settle(now)
        if rows[0].end < now - KEEP_LOOKS - timedelta(hours=1):
            self.looks[e.member_id] = rows = [x for x in rows if x.end >= now - KEEP_LOOKS]
            self.labels[e.member_id] = {end: counts for end, counts in labels.items() if end >= now - KEEP_LOOKS}
        if rule.looked_strained(rows, now) is None:
            self.open.pop(e.member_id, None)
            return
        current = self.moments.get(self.open.get(e.member_id, ""))
        if e.member_id in self.open:  # the stretch goes on (a moment that was forgotten stays forgotten)
            if current is not None:
                self.moments[current.id] = memory.replace(current, until=now)
            return
        try:
            context = self.context(now, e.member_id)
        except Exception:  # remembering must never stop the day
            log.debug("no context for a look", exc_info=True)
            return
        day = self.clock.day_of(now)
        m = Moment(id=memory.moment_id(e.member_id, now, self.clock.tz), member_id=e.member_id, at=now, until=now, day=day, context=context,
                   look=self.look(e.member_id, now))
        self.open[e.member_id] = m.id
        self.moments[m.id] = m
        self.pending.add(m.id)
        self.attach()
        old = day - timedelta(days=memory.KEEP_DAYS)
        if min(x.day for x in self.moments.values()) < old:
            self.moments = {i: x for i, x in self.moments.items() if x.day >= old}

    def on_forgotten(self, e: DayLookForgotten) -> None:
        if e.moment_id:
            self.moments.pop(e.moment_id, None)
        else:
            self.moments = {i: m for i, m in self.moments.items() if e.member_id is not None and m.member_id != e.member_id}

    def on_wipe(self, e: Event) -> None:
        """Live only. A wipe of everything removes the log this is a projection of: empty it now, not at the next restart."""
        if getattr(e, "scope", "") == "all":
            self.looks, self.labels, self.faces, self.moments, self.open, self.raised, self.answers = {}, {}, {}, {}, {}, [], {}
            self.drinks, self.containers, self.dues, self.presence = [], [], {}, DeskPresence()

    # --- what a moment holds ---

    def spans(self, start: datetime, as_of: datetime) -> list[Span]:
        """When a person was at the desk in [start, as_of]. A stay with no exit yet is taken to go on until `as_of`."""
        stays = [*self.presence.closed, *((a, max(b, as_of)) for a, b in self.presence.open.values() if a <= as_of)]
        out: list[Span] = []
        for a, b in sorted((max(a, start), min(b, as_of)) for a, b in stays if a <= as_of and b >= start):
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out

    def desk(self, member_id: str, at: datetime, as_of: datetime) -> list[Span]:
        """When `member_id` was at the desk around `at`, as far as `as_of` knows: the desk zone's own record when it
        reaches `at`; else the windows in which the desk camera had this member's face."""
        zone = self.spans(at - timedelta(hours=12), as_of)
        if rule.sitting([s for s in zone if s[0] <= at], at) is not None:
            return zone
        return sorted(w for w in self.faces.get(member_id, ()) if w[1] >= at - timedelta(hours=12) and w[0] <= as_of)

    def context(self, now: datetime, member_id: str) -> Context:
        """What is going on at `now`, from what the log held by then. Labels and whole numbers; never a heart value."""
        tz, day = self.clock.tz, self.clock.day_of(now)
        local = now.astimezone(tz)
        run = rule.sitting(self.desk(member_id, now, now), now)
        sitting = int((now - run[0]).total_seconds() // 60) if run else None
        since_drink = None
        if run and self.in_view(now):
            last = max((d for d in self.drinks if run[0] <= d <= now), default=run[0])
            since_drink = int((now - last).total_seconds() // 60)
        nights = self.checkin.nights
        night = nights.get(day)
        rhr_up = rule.resting_up(nights, day) if night and night.resting_hr is not None and rule.own(nights, day, "resting_hr") else None
        short = rule.short_night(nights, day) if night and night.asleep_min is not None and rule.own(nights, day, "asleep_min") else None
        activity, doing = self.doing(day, now)
        due = min(((when, title) for title, when in self.dues.values() if now < when <= now + timedelta(days=1)), default=None)
        drinks, limit = self.energy(now)
        return Context(minute_of_day=local.hour * 60 + local.minute, weekday=local.weekday(), sitting_min=sitting, since_drink_min=since_drink,
                       energy_drinks=drinks, energy_limit=limit, rhr_up=rhr_up, short_night=short, activity=activity, doing=doing,
                       deadline=due[1] if due else None, deadline_hours=round((due[0] - now).total_seconds() / 3600) if due else None)

    def doing(self, day: date, now: datetime) -> tuple[str | None, str | None]:
        """(the kind of block, in words) the plan had at `now`: a fixed event first, else the chosen block."""
        plan = self.planner.current.get(day)
        moves = sorted((m for m in plan.moves if m.start <= now < m.end), key=lambda m: not m.fixed) if plan else []
        if not moves or moves[0].kind == "sleep":
            return None, None
        m = moves[0]
        if m.fixed:
            word = next((w for w in memory.CLASS_WORDS if w in m.title.lower()), None)
            return ("class", f"in a {word}") if word else ("event", f"at {the(short_title(m.title))}")
        if m.kind == "study":
            return "study", "studying" if self.words.activity == "study" else f"in a {self.words.activity} block"
        return m.kind, memory.DOING.get(m.kind)

    def energy(self, now: datetime) -> tuple[int | None, int | None]:
        if self.habits is None:
            return None, None
        try:
            row = next((r for r in self.habits.digest(now=now)["habits"] if r["id"] == ENERGY), None)
            return (int(row["value"]), int(row["target"])) if row and row.get("value") is not None else (None, None)
        except Exception:
            log.debug("no energy drink count for a look", exc_info=True)
            return None, None

    def attach(self) -> None:
        """A check-in raised inside a moment's stretch is what Atlas suggested then."""
        for at, checkin_id, kinds in self.raised:
            for i, m in self.moments.items():
                if m.checkin_id is None and m.at - timedelta(minutes=1) <= at <= m.until + timedelta(minutes=rule.EXPR_WINDOW_MIN):
                    self.moments[i] = memory.replace(m, checkin_id=checkin_id, suggested=kinds,
                                                     outcome=memory.replace(m.outcome, answer=self.answers.get(checkin_id)))

    # --- the outcome: filled in when it is known ---

    def as_of(self, m: Moment, now: datetime) -> Moment:
        """`m` with its outcome as far as `now` knows it. Pure: the same from a live read as from a replay."""
        if m.outcome.known:
            return m
        rows = self.looks.get(m.member_id, [])
        got = memory.outcome(m.at, now, self.desk(m.member_id, m.at, now), self.drinks, self.in_view(m.at, either_side=True), rows,
                             m.outcome.answer)
        return memory.with_outcome(m, got) if got.known else m

    def settle(self, now: datetime) -> None:
        """Keep every outcome that is known by `now`, while the windows and the desk it is read from are still held."""
        self.seen = now if self.seen is None or now > self.seen else self.seen
        for i in [i for i in self.pending if i not in self.moments or now >= self.moments[i].at + timedelta(minutes=memory.EASE_MIN)]:
            if i in self.moments:
                self.moments[i] = self.as_of(self.moments[i], now)
            self.pending.discard(i)

    def tick(self, now: datetime) -> list[Event]:
        self.settle(now)
        return []

    # --- reads ---

    def clock_now(self, now: datetime) -> datetime:
        """The wall clock, or the last moment the log knows about when it ends later (a seeded or replayed log)."""
        return now if self.seen is None or now >= self.seen else self.seen

    def remembered(self, now: datetime) -> list[Moment]:
        """Every remembered moment, most recent first, each with its outcome as far as it is known."""
        return sorted((self.as_of(m, now) for m in self.moments.values()), key=lambda m: m.at, reverse=True)

    def look(self, member_id: str, now: datetime) -> str:
        """The kind of look in the twenty minutes up to `now`: "tired" or "tense", from the labels behind the strained looks."""
        got = rule.looked_strained(self.looks.get(member_id, []), now)
        added: dict[str, float] = {}
        for window in got[1] if got else []:
            for label, count in self.labels.get(member_id, {}).get(window.end, {}).items():
                added[label] = added.get(label, 0.0) + count
        return memory.look_word(added)

    def strained_member(self, now: datetime) -> str | None:
        hits = [(sum(x.samples for x in got[1]), member) for member, rows in self.looks.items() if (got := rule.looked_strained(rows, now))]
        return max(hits)[1] if hits else None

    def callback(self, now: datetime) -> CheckinCallback | None:
        """The earlier moment this look is like, in words, for a check-in raised at `now`; None when nothing is similar
        enough (never a forced callback) or nobody's look is being remembered."""
        member = self.strained_member(now)
        if member is None or (self.allowed is not None and member not in self.allowed):
            return None
        today, tz = self.clock.day_of(now), self.clock.tz
        context, rows, look = self.context(now, member), self.remembered(now), self.look(member, now)
        then = memory.similar(context, today, member, rows, look)
        shown = memory.view(then, today, tz, self.words) if then is not None else None
        lines = [f"{shown['when']}, around {shown['time']}. {shown['going_on']}", shown["helped"]] if shown else []
        # Stage two: with enough known outcomes, what followed and the advice to lead with are what the counts over
        # every remembered look of this person say, in a context like this one.
        model, features = self.model(member, now), asdict(context)
        counted = look_model.learned_line(model, features, self.words)
        rests_on = then or next((m for m in rows if m.member_id == member and m.outcome.known and m.outcome.eased is not None), None)
        if counted and rests_on is not None:
            evidence = look_model.evidence_line(model)
            return CheckinCallback(moment_id=rests_on.id, day=rests_on.day, look=then.look if then else look,
                                   said=memory.callback(then, today, tz) if then else "", learned=counted,
                                   lead=look_model.lead(model, features), chip=memory.chip(then, today, tz) if then else "What has eased it",
                                   lines=[*(lines or [counted]), evidence], pattern=look_model.pattern_line(model, self.words), evidence=evidence)
        if then is None:
            return None
        helped, lead = memory.learned(then, self.words)
        return CheckinCallback(moment_id=then.id, day=then.day, look=then.look, said=memory.callback(then, today, tz),
                               learned=helped, lead=lead, chip=memory.chip(then, today, tz), lines=lines)

    # --- stage two: what it has learned (contexts/day/domain/look_model.py) ---

    def model(self, member_id: str, now: datetime | None = None) -> look_model.Model:
        """What Atlas has learned about one member's looks: fitted here and now from `examples` and nothing else.
        Never kept: it is a function of the remembered moments, so it is gone when they are."""
        return look_model.fit(self.examples(member_id, now))

    def learner(self, member_id: str | None = None) -> str | None:
        """Whose looks a question or a screen is about: `member_id`, else the member of the newest remembered look,
        else the one member being remembered. None when that is nobody."""
        if member_id is not None:
            return member_id
        newest = max(self.moments.values(), key=lambda m: m.at, default=None)
        if newest is not None:
            return newest.member_id
        return next(iter(self.allowed)) if self.allowed and len(self.allowed) == 1 else None

    def ahead(self, now: datetime) -> str:
        """For the check-in card only: "You usually start to look strained around now. A short break tends to help.",
        when the member at the desk has been there about as long as it usually takes for a look to start. "" when that
        is not a finding, when nobody is there, or when the setting is off. Nothing is said and nothing is pushed."""
        if not self.look_ahead:
            return ""
        members = sorted({m.member_id for m in self.moments.values() if self.allowed is None or m.member_id in self.allowed})
        if len(members) > 1:  # the desk zone cannot tell two people apart: only the one the desk camera has in front of it
            members = [m for m in members if rule.sitting(sorted(self.faces.get(m, ())), now) is not None]
        for member in members:
            run = rule.sitting(self.desk(member, now, now), now)
            line = look_model.ahead_line(self.model(member, now), int((now - run[0]).total_seconds() // 60), self.words) if run else ""
            if line:
                return line
        return ""

    def ahead_holds(self, e) -> bool:
        """Whether a check-in's quiet line may still be shown: the setting is on and some remembered looks still say so."""
        return self.look_ahead and any(self.model(m).onset_min is not None for m in {x.member_id for x in self.moments.values()})

    def view(self, now: datetime, limit: int = SHOWN, member_id: str | None = None) -> dict:
        """For the owner's own screen (GET /api/day/looks): the most recent moments in plain lines, one member's when
        `member_id` is given. Box-only."""
        now = self.clock_now(now)
        today, rows = self.clock.day_of(now), [m for m in self.remembered(now) if member_id in (None, m.member_id)]
        who = self.learner(member_id)
        return {"on": bool(self.allowed) or (self.allowed is None and bool(self.looks)), "members": sorted(self.allowed or ()),
                "count": len(rows), "kept_days": memory.KEEP_DAYS, "callback_days": memory.CALLBACK_DAYS,
                "moments": [memory.view(m, today, self.clock.tz, self.words) for m in rows[:limit]],
                # what Atlas has learned from them (stage two): sentences with their counts, or "not enough yet"
                "learned": look_model.view(self.model(who, now), self.words) if who is not None and rows else None,
                "note": "Labels and numbers only, kept on this box. No picture is stored. Switching expressions off clears them."}

    def forget(self, now: datetime, moment_id: str | None = None, member_id: str | None = None) -> DayLookForgotten | None:
        """The event that forgets one moment, or all of them (one member's with `member_id`). None when there is no such moment."""
        if moment_id is not None and moment_id not in self.moments:
            return None
        return DayLookForgotten(event_id=f"day-look-forgotten-{moment_id or member_id or 'all'}-{now.timestamp():.0f}", ts=now,
                                moment_id=moment_id, member_id=None if moment_id else member_id)

    def said(self, kind: str, now: datetime) -> str | None:
        """An answer for the ear to "when did I last look this tired?" (`last`), "why do you think I'm tired?" (`why`)
        or "what helped last time?" (`helped`); and, from what it has learned (stage two), to "what usually helps when
        I look tired?" (`helps`), "what makes me look tired?" (`causes`) and "how do you know?" (`how`). None when
        nobody's looks are remembered on this box, so the question goes wherever it went before."""
        if not (self.allowed or (self.allowed is None and self.moments)):
            return None
        now = self.clock_now(now)
        today, tz, rows = self.clock.day_of(now), self.clock.tz, self.remembered(now)
        current = next((m for m in rows if m.day == today and now - m.until <= memory.RECENT), None)
        earlier = [m for m in rows if m is not current]
        if kind == "last":
            return memory.said_last(earlier[0] if earlier else None, today, tz)
        if kind == "helped":
            known = next((m for m in earlier if m.outcome.known and m.outcome.eased is not None), None)
            return memory.said_helped(known or (earlier[0] if earlier else None), today, tz, self.words)
        if kind == "why":
            then = memory.similar(current.context, today, current.member_id, earlier, current.look) if current else None
            return memory.said_why(current, then, today, tz, self.words)
        if kind in ("helps", "causes", "how"):
            who = self.learner()
            if not rows or who is None:
                return look_model.said_how(look_model.Model(), self.words) if kind == "how" else memory.NOTHING
            model = self.model(who, now)
            return {"helps": look_model.said_helps, "causes": look_model.said_causes, "how": look_model.said_how}[kind](model, self.words)
        return None

    def examples(self, member_id: str | None = None, now: datetime | None = None) -> list[dict]:
        """READ-ONLY, the only thing stage two's learning reads: the remembered moments as labelled examples (look,
        context features, advice, outcome), no pictures and no ids. `member_id`: one person's (the rows still carry
        no id). `now`: outcomes as far as that time knows them; the last event time by default."""
        now = now or self.seen or datetime(1970, 1, 1, tzinfo=UTC)
        return memory.labelled_examples([m for m in self.remembered(now) if member_id in (None, m.member_id)])

