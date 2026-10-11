"""Day Run's application service: gathers a day from the objects it is given and runs the engine.

    dayrun = DayRun(places, memory=memory, habits=habits, presence=presence, hub=hub, data_dir=s.data_dir)

It reads other contexts only through those objects: connector items through a
MemoryReader (`items`), last night's sleep and today's meals through the habit
tracker's `digest`, friends' shared places through the presence board, and it
proposes calendar writes through the connector hub (never creating events
itself). The only state it keeps is the committed plan of each day, as JSON under
`<data_dir>/dayrun/plans/`, so the review can compare it with what happened.

The engine takes time as an argument. The clock here is used only when a request
names no start, and to stamp a commit.
"""

import json
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel

from contexts.play.dayrun import fixes, modes, talk, words
from contexts.play.dayrun.board import build_board, initial_state, local
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.history import DayBook, DayRecord
from contexts.play.dayrun.models import (
    Board,
    CampaignResult,
    CommitResult,
    Disruption,
    EditResult,
    Estimate,
    ForecastResult,
    FriendAt,
    HardResult,
    InitialOverride,
    Measured,
    Move,
    PlannedBlock,
    ReplayResult,
    Review,
    SimResult,
    Totals,
)
from contexts.play.dayrun.review import forecast, reality
from contexts.play.dayrun.routes import Walks
from contexts.play.domain.places import PlaceSet
from contexts.profiles import Profile
from contracts.agent import ActionProposed
from contracts.memory import Item

CALENDAR_WRITE = "calendar.create_event"
SCENARIO_NAME = re.compile(r"^[a-z0-9_-]{1,40}$")


class PlanNotWinning(ValueError):
    """Only a winning plan is committed to the calendar."""

    def __init__(self, result: SimResult) -> None:
        self.result = result
        super().__init__(result.outcome.summary)


class ScenarioFriend(BaseModel):
    member: str
    place: str  # a shared place name ("Library") or a pack.place
    start: datetime
    end: datetime


class Scenario(BaseModel):
    """A scripted day (fixtures/play/*.yaml): items, habit numbers and friends, in place of live sources."""

    now: datetime
    start_place: str | None = None
    items: list[Item] = []
    habits: dict[date, dict[str, float]] = {}  # day -> {sleep (hours), meals, focus_hours}
    friends: list[ScenarioFriend] = []
    plans: dict[str, list[Move]] = {}
    disruptions: list[Disruption] = []
    campaign: list[list[Move]] = []
    actual: dict[date, Measured] = {}

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    # the two read interfaces Day Run uses, served from the script
    def items_between(self, start: datetime, end: datetime, kind: str | None = None) -> list[Item]:
        return sorted((i for i in self.items if i.when and start <= i.when < end and (kind is None or i.kind == kind)), key=lambda i: i.when)

    def digest(self, day: date | None = None, now: datetime | None = None) -> dict:
        day = day or (now or self.now).date()
        row = self.habits.get(day, {})
        names = {"sleep": "sleep", "meals": "meals", "focus_hours": "focus"}
        return {"day": day.isoformat(), "habits": [{"id": names[k], "value": v, "applies": True} for k, v in row.items() if k in names]}


class DayRun:
    def __init__(
        self,
        places: PlaceSet,
        memory=None,  # MemoryReader: .items(start, end, kind)
        habits=None,  # HabitTracker: .digest(day=, now=)
        presence=None,  # PresenceBoard: .friends(now)
        hub=None,  # ConnectorHub: .is_write(tool), .propose_write(tool, arguments, reason, now)
        config_dir: str | Path = "config",
        data_dir: str | Path = "data",
        fixtures_dir: str | Path = "fixtures/play",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        cfg: DayRunConfig | None = None,
        profile: Profile | None = None,  # layers its dayrun.yaml over the base one; its scenarios/ are searched first
        agreed=None,  # AgreedPlans (contexts/play/dayrun/agreed.py): .items(lo, hi, tz, known)
    ) -> None:
        self.cfg = cfg or DayRunConfig.load(Path(config_dir) / "dayrun.yaml", Path(config_dir) / "agent.yaml", profile)
        self.walks = Walks(places)
        self.memory, self.habits, self.presence, self.hub = memory, habits, presence, hub
        self.agreed = agreed  # plans the Circle agreed: fixed blocks of the real day, never of a scripted one
        self.plans_dir = Path(data_dir) / "dayrun" / "plans"
        self.fixtures_dir = Path(fixtures_dir)
        self.scenarios_dir = profile.scenarios_dir() if profile is not None else None
        self.clock = clock
        self.book = DayBook(Path(data_dir) / "events")  # the days the log has a plan for (read-only)
        self._labels: dict[str, list[str]] = {}  # plan id -> what needed fixing in it, for a past day's one line

    # --- sources ---

    def scenario(self, name: str | None) -> Scenario | None:
        if not name:
            return None
        if not SCENARIO_NAME.match(name):
            raise LookupError(f"no scenario {name!r}")
        folders = [self.scenarios_dir] if self.scenarios_dir else []
        path = next((p for d in [*folders, self.fixtures_dir] if (p := d / f"{name}.yaml").exists()), None)
        if path is None:
            raise LookupError(f"no scenario {name!r}")
        return Scenario.load(path)

    def script(self, name: str) -> dict:
        """A scenario's own presets for the phone: its plans, disruptions and campaign (never a result)."""
        sc = self.scenario(name)
        if sc is None:
            raise LookupError("no scenario ''")
        return {"name": name, "now": sc.now, "start_place": sc.start_place,
                "plans": {k: [m.model_dump(mode="json", exclude_none=True) for m in v] for k, v in sc.plans.items()},
                "disruptions": [d.model_dump(mode="json", exclude_none=True) for d in sc.disruptions],
                "campaign": [[m.model_dump(mode="json", exclude_none=True) for m in day] for day in sc.campaign]}

    def _habit_rows(self, sc: Scenario | None, day: date, now: datetime) -> list[dict] | None:
        source = sc or self.habits
        if source is None:
            return None
        try:
            return source.digest(day=day, now=now).get("habits", [])
        except Exception:  # habits are a starting point, never a reason to refuse a board
            return None

    def _friends(self, sc: Scenario | None, start: datetime) -> list[FriendAt]:
        if sc:
            shared = [(f.member, f.place, f.start, f.end) for f in sc.friends]
        elif self.presence:
            shared = [(p.member_id, p.place, max(local(self.cfg, p.ts), start), p.expires_at) for p in self.presence.friends(start)]
        else:
            shared = []
        out = []
        for member, name, a, b in shared:
            place = self.walks.shared(name) or (name if "." in name else None)
            if place:  # a shared place the packs do not have cannot be routed to, so it is not offered
                place = self.walks.ref(place)
                out.append(FriendAt(member=member, place=place, place_name=self.walks.name(place),
                                    start=local(self.cfg, a), end=local(self.cfg, b)))
        return out

    def board(self, start: datetime | None = None, initial: InitialOverride | None = None, scenario: str | None = None) -> Board:
        sc = self.scenario(scenario)
        asked = start
        start = local(self.cfg, start or (sc.now if sc else self.clock()))
        lo = start - timedelta(days=1)  # an event that began earlier may still be running
        hi = start + timedelta(hours=self.cfg.window_hours + self.cfg.deadline_horizon_hours)
        items = list(sc.items_between(lo, hi) if sc else self.memory.items(lo, hi) if self.memory else [])
        if self.agreed is not None and not sc:
            items += self.agreed.items(lo, hi, self.cfg.tz, items)
        if asked is not None and not sc:
            items = self._logged_items(start, items)
        over = initial or InitialOverride()
        if sc and sc.start_place and not over.start_place:
            over = over.model_copy(update={"start_place": sc.start_place})
        state = initial_state(self.cfg, start, self._habit_rows(sc, start.date(), start), over, origin="scenario" if sc else "habits")
        return build_board(self.cfg, self.walks, start, list(items), state, self._friends(sc, start), over)

    def _logged_items(self, start: datetime, items: list[Item]) -> list[Item]:
        """A board asked for at the very start of a plan the log holds for an earlier day is that plan's own day: the
        fixed blocks the plan was made around are added back from the plan event (they may no longer be in Memory,
        or never were: a day lived before the connectors synced). Any other start is untouched."""
        try:
            rec = self.book.days().get(start.date())
            if rec is None or start.date() >= self.today():
                return items
            plan = next((p for p in [rec.lived, rec.plan, *rec.replans] if p and local(self.cfg, datetime.fromisoformat(p["start"])) == start), None)
        except Exception:
            return items
        if plan is None:
            return items
        have = {i.external_id for i in items}
        out = list(items)
        for m in plan.get("moves", []):
            if not m.get("fixed") or m.get("kind") == "sleep" or m["id"] in have:
                continue
            a, b = local(self.cfg, datetime.fromisoformat(m["start"])), local(self.cfg, datetime.fromisoformat(m["end"]))
            out.append(Item(connector="calendar", kind="calendar_event", external_id=m["id"], title=m["title"], when=a,
                            summary=f"until {b.hour % 12 or 12}:{b:%M %p}" + (f" · {m['place_name']}" if m.get("place_name") else "")))
        return out

    # --- play ---

    def simulate(self, moves: list[Move], start: datetime | None = None, initial: InitialOverride | None = None,
                 scenario: str | None = None) -> SimResult:
        return simulate(self.cfg, self.walks, self.board(start, initial, scenario), moves)

    def hard(self, moves: list[Move], start: datetime | None = None, initial: InitialOverride | None = None,
             scenario: str | None = None, seed: int = 0, count: int = 1, disruptions: list[Disruption] | None = None) -> HardResult:
        return modes.hard(self.cfg, self.walks, self.board(start, initial, scenario), moves, seed, count, disruptions)

    def campaign(self, days: list[list[Move]], start: datetime | None = None, initial: InitialOverride | None = None,
                 scenario: str | None = None) -> CampaignResult:
        first = self.board(start, initial, scenario).start
        return modes.campaign(self.cfg, self.walks, lambda t, over: self.board(t, over, scenario), first, days, initial)

    # --- commit: calendar blocks, through an approval ---

    def commit(self, moves: list[Move], start: datetime | None = None, initial: InitialOverride | None = None,
               scenario: str | None = None, now: datetime | None = None) -> tuple[ActionProposed, CommitResult]:
        """Propose the plan's blocks as calendar events. Nothing is created here: the approval card decides."""
        result = self.simulate(moves, start, initial, scenario)
        if not result.outcome.won:
            raise PlanNotWinning(result)
        if self.hub is None or not self.hub.is_write(CALENDAR_WRITE):
            raise RuntimeError("the calendar connector cannot create events on this box")
        blocks = [b for b in result.blocks if not b.fixed]
        events = [{"title": f"Day Run: {b.title}", "start": b.start.isoformat(), "end": b.end.isoformat(), "location": b.place_name}
                  for b in blocks]
        day = result.start.date()
        reason = f"Add {len(events)} Day Run block(s) for {day:%a %b} {day.day} ({result.stars}-star plan)"
        proposal = self.hub.propose_write(CALENDAR_WRITE, {"events": events}, reason, now or self.clock())
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        (self.plans_dir / f"{day}.json").write_text(json.dumps({
            "day": day.isoformat(), "start": result.start.isoformat(), "action_id": proposal.action_id,
            "moves": [m.model_dump(mode="json") for m in moves],
            "blocks": [b.model_dump(mode="json") for b in result.blocks],
            "totals": result.totals.model_dump(), "stars": result.stars,
        }, indent=1))
        return proposal, CommitResult(action_id=proposal.action_id, day=day, events=events, stars=result.stars, reason=reason)

    def committed(self, day: date) -> dict:
        path = self.plans_dir / f"{day}.json"
        if not path.exists():
            raise LookupError(f"no plan was committed for {day}")
        try:
            plan = json.loads(path.read_text())
        except ValueError:
            plan = None
        if not isinstance(plan, dict) or not isinstance(plan.get("blocks"), list) or not isinstance(plan.get("totals"), dict):
            raise ValueError(f"the plan committed for {day} is damaged ({path.name}); commit it again")
        return plan

    # --- afterwards ---

    def measured(self, day: date, scenario: str | None = None) -> Measured:
        """What Habits measured for the day. Sleep is the night that followed it, read from the next day's row.

        That night is unknown (None) until Habits has seen the whole of it: a night still ahead, or half
        way through, is not an empty desk. A scripted day's numbers are taken as written."""
        sc = self.scenario(scenario)
        if sc and day in sc.actual:
            return sc.actual[day]
        end = datetime.combine(day + timedelta(days=1), datetime.min.time(), self.cfg.tz)
        seen = None if sc else getattr(self.habits, "now", None)  # the latest event time the tracker has seen
        sched = getattr(self.habits, "sched", None)  # the night is Habits' sleep window, not Day Run's usual one
        night_end = (sched.sleep_window(day + timedelta(days=1))[1] if sched
                     else datetime.combine(day + timedelta(days=1), self.cfg.sleep.wake, self.cfg.tz))
        night_over = seen is None or seen >= night_end
        today = {r["id"]: r for r in self._habit_rows(sc, day, end) or []}
        tomorrow = {r["id"]: r for r in self._habit_rows(sc, day + timedelta(days=1), end + timedelta(days=1)) or []} if night_over else {}
        return Measured(
            focus_minutes=round(today["focus"]["value"] * 60, 1) if "focus" in today else None,
            meals=int(today["meals"]["value"]) if "meals" in today else None,
            sleep_hours=tomorrow["sleep"]["value"] if "sleep" in tomorrow and tomorrow["sleep"].get("applies", True) else None,
        )

    def review(self, day: date, measured: Measured | None = None, scenario: str | None = None) -> Review:
        plan = self.committed(day)
        blocks = [PlannedBlock.model_validate(b) for b in plan["blocks"]]
        return reality(self.cfg, self.walks.ref(self.cfg.desk), day, blocks, Totals.model_validate(plan["totals"]),
                       measured or self.measured(day, scenario))

    def replay(self, day: date, moves: list[Move], baseline: list[Move] | None = None, actual: Measured | None = None,
               scenario: str | None = None) -> ReplayResult:
        """A past day again: what happened becomes the baseline plan, and a new plan is scored against it."""
        start = datetime.combine(day, self.cfg.sleep.wake, self.cfg.tz)
        board = self.board(start, InitialOverride(meals=0), scenario)  # the day's meals had not happened yet at wake
        actual = actual or self.measured(day, scenario)
        plan = simulate(self.cfg, self.walks, board, moves)
        # A night nobody has measured is not zero hours and not a full night: it is left out of the comparison
        # by giving the baseline the night the new plan schedules, so sleep cannot win or lose a point.
        night_unknown = baseline is None and actual.sleep_hours is None
        night = [(moves[b.move].slot, b) for b in plan.blocks if b.kind == "sleep" and b.move is not None] if night_unknown else []
        base_moves = baseline if baseline is not None else modes.baseline_moves(self.cfg, self.walks, board, actual, night)
        base = simulate(self.cfg, self.walks, board, base_moves)
        points, stars = plan.totals.points - base.totals.points, plan.stars - base.stars
        verdict = "beats" if points > 0 else "matches" if points == 0 else "does worse than"
        if night_unknown:
            seen = [f"{actual.meals} meal{'' if actual.meals == 1 else 's'}" if actual.meals is not None else "meals not measured",
                    f"{actual.focus_minutes:g} minutes at the desk" if actual.focus_minutes is not None else "desk time not measured"]
            happened = (f"{day:%a %b} {day.day} so far: {', '.join(seen)}. That night's sleep has not been measured yet, so both "
                        f"sides are given the night in your new plan ({plan.totals.sleep_hours:g} h) and sleep counts for neither. "
                        f"On everything else your new plan {verdict} what happened")
        else:
            happened = (f"What happened on {day:%a %b} {day.day}: {base.outcome.summary} ({base.totals.points} points). "
                        f"Your new plan {verdict} it")
        text = f"{happened} by {abs(points)} points: {plan.outcome.summary}"
        return ReplayResult(day=day, board=board, actual=actual, baseline_moves=base_moves, baseline=base, plan=plan,
                            points_gained=points, stars_gained=stars, text=text, unknown=["sleep"] if night_unknown else [])

    def forecast(self, estimates: list[Estimate], measured: dict[str, float] | None = None, day: date | None = None,
                 scenario: str | None = None) -> ForecastResult:
        if measured is None:
            if day is None:
                raise ValueError("give `measured` minutes per block, or the `day` of a committed plan")
            measured = {b.id: b.measured_minutes for b in self.review(day, scenario=scenario).blocks if b.measured_minutes is not None}
        elif any(not (0 <= m < float("inf")) for m in measured.values()):  # a negative or infinite length scores as nonsense
            raise ValueError("measured minutes must be 0 or more")
        return forecast(estimates, measured)

    # --- today's plan: fixes, sources in words, edits by saying them, and the days around today (all additive) ---

    def explain(self, moves: list[Move], start: datetime | None = None, initial: InitialOverride | None = None,
                scenario: str | None = None) -> dict:
        """What needs fixing (each with a suggested edit) and one true sentence per number a person might question."""
        board = self.board(start, initial, scenario)
        return fixes.explain(self.cfg, self.walks, board, moves, simulate(self.cfg, self.walks, board, moves))

    def edit(self, text: str, moves: list[Move], start: datetime | None = None, initial: InitialOverride | None = None,
             scenario: str | None = None) -> EditResult:
        """A change asked for in plain words, as the new list of moves (contexts/play/dayrun/talk.py). Nothing is stored."""
        return talk.edit(self.cfg, self.walks, self.board(start, initial, scenario), moves, text)

    def today(self) -> date:
        """The local day the strip calls today: the clock's when the log has a plan for it (or has no plans at all),
        else the newest planned day, which is what the Calendar screen shows on a log that ends on another day
        (contexts/day/api.py `plan_for`)."""
        now = local(self.cfg, self.clock()).date()
        planned = [d for d, r in self.book.days().items() if r.plan]
        return now if now in planned or not planned else max(planned)

    def _wake(self, day: date) -> datetime:
        return datetime.combine(day, self.cfg.sleep.wake, self.cfg.tz)

    def _problems(self, plan: dict) -> list[str]:
        """What needed fixing in a logged plan, by running it again on that day's board (the same engine, the same items)."""
        key = plan.get("plan_id", "")
        if key not in self._labels:
            try:
                moves = [Move(**m) for m in plan.get("dayrun_moves", [])]
                found = self.explain(moves, datetime.fromisoformat(plan["start"]))["fixes"]
                self._labels[key] = [f.problem for f in found]
            except Exception:
                self._labels[key] = []
        return self._labels[key]

    def _past(self, rec: DayRecord) -> dict:
        plan, lived, review = rec.plan or {}, rec.lived or {}, rec.review
        outcome = lived.get("outcome") or {}
        score = review.get("score") if review else None
        problems = [] if outcome.get("won", True) else self._problems(lived)
        replanned = lived is not plan and bool(rec.replans)
        if replanned:
            label = "re-planned, still made it" if outcome.get("won") else "re-planned"
            summary = lived.get("why") or "The day changed and Atlas re-planned it."
        elif problems:  # the plan itself did not work; the line under it says whose it was and how much of it happened
            label = words.past(problems[0])
            whose = "Your own plan" if lived.get("by") == "person" else "Atlas's plan"
            summary = f"{whose}. Nothing in it could be measured." if score is None else f"{whose}; {score}% of it happened."
        elif score is None:
            label = "planned, not measured"
            summary = "Atlas planned this day, but nothing in it could be measured."
        else:
            label = "on plan" if score >= 80 else f"{score}% of the plan happened"
            slipped = review.get("slipped") or []
            summary = f"You followed {score}% of this plan." + (f" {slipped[0]} slipped." if slipped else "")
        return {"has_data": True, "label": label, "summary": summary, "won": bool(outcome.get("won")),
                "stars": review.get("stars") if review else None, "score": score}

    def _ahead(self, day: date) -> dict:
        """A day not lived yet, from what the box already has for it: fixed things, what is due, free time, and how
        hard the open deadlines press if no work is done before it. Nothing is predicted."""
        start = self._wake(day)
        board = self.board(start)
        bed = datetime.combine(day, self.cfg.sleep.bedtime, self.cfg.tz)
        if bed <= start:
            bed += timedelta(days=1)
        fixed = [b for b in board.blocks if b.activity != "sleep" and b.start < bed]
        free = sum(max(0.0, (min(s.end, bed) - max(s.start, start)).total_seconds() / 60) for s in board.slots)
        due = [d for d in board.deadlines if local(self.cfg, d.due).date() == day]
        open_ones = [d for d in board.deadlines if d.work_left > 0]
        optional = [e for e in [*board.events, *board.suggestions] if e.start < bed]
        parts = [words.count(len(fixed), "fixed thing") if fixed else "Nothing fixed", f"{words.dur(free)} free"]
        summary = ", ".join(parts) + "." + "".join(f" {words.assignment(d.title)} is due {words.clock12(local(self.cfg, d.due))}." for d in due[:1])
        return {"has_data": bool(fixed or due or optional), "label": summary.split(".")[0].lower(), "summary": summary,
                "fixed": len(fixed), "agreed": sum(1 for b in fixed if b.id.startswith("circle:")), "optional": len(optional),
                "free_minutes": round(free), "busy_minutes": round(sum((b.end - b.start).total_seconds() / 60 for b in fixed)),
                "due": [{"id": d.id, "title": words.assignment(d.title), "due": d.due, "work_left": d.work_left} for d in due],
                "pressure": round(max((d.risk for d in open_ones), default=0.0), 3)}

    def days(self, back: int = 14, ahead: int = 5) -> dict:
        """The date strip: `back` days behind today, today, `ahead` in front; a day the box has nothing for is not selectable."""
        today, book = self.today(), self.book.days()
        out = []
        for n in range(-back, ahead + 1):
            day = today + timedelta(days=n)
            row: dict = {"day": day, "when": "past" if n < 0 else "today" if n == 0 else "future"}
            if n < 0:
                row |= self._past(book[day]) if day in book and book[day].plan else {"has_data": False, "label": "nothing recorded", "summary": ""}
            elif n == 0:
                row |= {"has_data": True, "label": "today", "summary": ""}
            else:
                try:
                    row |= self._ahead(day)
                except Exception:
                    row |= {"has_data": False, "label": "nothing yet", "summary": ""}
            out.append(row)
        return {"today": today, "days": out}

    def day(self, day: date) -> dict:
        """One day of the strip. Past: the plan that was lived, block by block, with what the review measured. Today:
        the Day loop's plan, to edit. Future: an empty draft that starts at wake time."""
        today, rec = self.today(), self.book.days().get(day)
        when = "past" if day < today else "today" if day == today else "future"
        lived = (rec.lived if rec else None) or {}
        out: dict = {"day": day, "when": when, "start": lived.get("start") or (self._wake(day) if when != "today" else None),
                     "moves": lived.get("dayrun_moves", []), "blocks": [], "review": None, "changes": [], "first": None,
                     "outcome": lived.get("outcome")}
        if when == "future":
            return out | self._ahead(day)
        if rec is None or rec.plan is None:
            if when == "past":
                raise LookupError(f"nothing was recorded for {day}")
            return out | {"has_data": True, "label": "today", "summary": ""}
        review = rec.review or {}
        followed, slipped = set(review.get("followed") or []), set(review.get("slipped") or [])
        morning = {(m["title"], m["start"], m.get("place")) for m in rec.plan.get("moves", [])}
        replanned = lived is not rec.plan
        # a re-planned day is told whole: the morning as first planned, then the new plan from where it took over
        before = [m for m in rec.plan.get("moves", []) if replanned and m["end"] <= lived.get("start", "")]
        for m in [*before, *lived.get("moves", [])]:
            mark = "done" if m["title"] in followed else "missed" if m["title"] in slipped else None
            out["blocks"].append({**m, "mark": mark if when == "past" else None,
                                  "moved": replanned and (m["title"], m["start"], m.get("place")) not in morning})
        if replanned:
            out["changes"] = [{"at": lived.get("ts"), "why": lived.get("why", "")}]
            out["first"] = {"start": rec.plan.get("start"), "moves": rec.plan.get("dayrun_moves", [])}
        if rec.review:
            out["review"] = {k: review.get(k) for k in ("score", "stars", "points", "plan_stars", "followed", "slipped", "change_tomorrow")}
        return out | (self._past(rec) if when == "past" else {"has_data": True, "label": "today", "summary": ""})
