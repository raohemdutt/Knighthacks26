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

from contexts.play.dayrun import modes
from contexts.play.dayrun.board import build_board, initial_state, local
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import (
    Board,
    CampaignResult,
    CommitResult,
    Disruption,
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
    ) -> None:
        self.cfg = cfg or DayRunConfig.load(Path(config_dir) / "dayrun.yaml", Path(config_dir) / "agent.yaml")
        self.walks = Walks(places)
        self.memory, self.habits, self.presence, self.hub = memory, habits, presence, hub
        self.plans_dir = Path(data_dir) / "dayrun" / "plans"
        self.fixtures_dir = Path(fixtures_dir)
        self.clock = clock

    # --- sources ---

    def scenario(self, name: str | None) -> Scenario | None:
        if not name:
            return None
        path = self.fixtures_dir / f"{name}.yaml"
        if not SCENARIO_NAME.match(name) or not path.exists():
            raise LookupError(f"no scenario {name!r}")
        return Scenario.load(path)

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
        start = local(self.cfg, start or (sc.now if sc else self.clock()))
        lo = start - timedelta(days=1)  # an event that began earlier may still be running
        hi = start + timedelta(hours=self.cfg.window_hours + self.cfg.deadline_horizon_hours)
        items = sc.items_between(lo, hi) if sc else self.memory.items(lo, hi) if self.memory else []
        over = initial or InitialOverride()
        if sc and sc.start_place and not over.start_place:
            over = over.model_copy(update={"start_place": sc.start_place})
        state = initial_state(self.cfg, start, self._habit_rows(sc, start.date(), start), over, origin="scenario" if sc else "habits")
        return build_board(self.cfg, self.walks, start, list(items), state, self._friends(sc, start), over)

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
        return json.loads(path.read_text())

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
        return forecast(estimates, measured)
