"""Scoring a week: pure functions over daily summaries. No clock, no I/O, deterministic.

The rule in plain words: for every target that counts (config/league.yaml), a member's share is
the days they hit it over the days that were judged (a day with no summary is not judged). Those
shares are averaged with the targets' weights and shown as a percent. The lowest percent is the
worst week, the highest the best; a member with nothing judged has no score and cannot be either.
Ties break on more target-days hit, then on the member id, so a replay always picks the same person.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from contexts.play.league.config import LeagueConfig
from contracts.play import Stake, Standing

RECYCLING = "recycling"
DAYRUN = "dayrun_stars"
YOU = "You"  # this box's display name when it plays alone (no Circle)


@dataclass(frozen=True)
class DaySummary:
    member_id: str
    habit_id: str
    day: date
    met: bool
    value: float


@dataclass(frozen=True)
class Count:
    """One Waste Watch running total (contracts.impact.CountUpdated), at `ts`."""

    member_id: str
    ts: datetime
    items_seen: int
    recyclable_seen: int
    recycled: int
    contamination: int


@dataclass(frozen=True)
class PlanStars:
    member_id: str
    day: date
    stars: int


@dataclass
class TargetScore:
    target: str
    label: str
    unit: str
    weight: float
    met: int
    judged: int
    detail: str | None = None  # "71% diverted this week"

    @property
    def share(self) -> float | None:
        return self.met / self.judged if self.judged else None


@dataclass
class MemberWeek:
    member_id: str
    display_name: str
    targets: dict[str, TargetScore] = field(default_factory=dict)

    @property
    def judged(self) -> int:
        return sum(t.judged for t in self.targets.values())

    @property
    def met(self) -> int:
        return sum(t.met for t in self.targets.values())

    @property
    def score(self) -> float | None:
        scored = [t for t in self.targets.values() if t.judged and t.weight > 0]
        if not scored:
            return None
        total = sum(t.weight for t in scored)
        return round(100 * sum(t.weight * t.share for t in scored) / total, 1)

    def reasons(self) -> list[str]:
        """Misses first, biggest first; then what was hit. "missed sleep 4 of 7 nights"."""
        out = []
        judged = [t for t in self.targets.values() if t.judged]
        for t in sorted(judged, key=lambda t: (-(t.judged - t.met), t.target)):
            if t.detail:
                out.append(t.detail)
            elif t.met < t.judged:
                out.append(f"missed {t.label} {t.judged - t.met} of {t.judged} {t.unit}")
            else:
                out.append(f"hit {t.label} {t.judged} of {t.judged} {t.unit}")
        return out

    def headline(self) -> str:
        """The one reason that sums the week up: the target missed the most. "2 of 7 sleep targets"."""
        judged = [t for t in self.targets.values() if t.judged]
        if not judged:
            return "nothing judged"
        # a run of days says more than a single number: prefer targets judged on more than one day
        pool = [t for t in judged if t.judged > 1] or judged
        t = min(pool, key=lambda t: (t.share, t.target))
        if t.detail:
            return t.detail
        return f"{t.met} of {t.judged} {t.label} targets"


def week_of(day: date) -> date:
    """The local Monday the week starts on."""
    return day - timedelta(days=day.weekday())


def week_end(week_start: date) -> date:
    return week_start + timedelta(days=6)


def weekly_delta(counts: list[Count], start: date, end: date, local_day) -> tuple[int, int, int] | None:
    """(recycled, recyclable, contamination) added during the week: the last total in the week minus the
    last total before it. None when nothing was counted in the week."""
    before = [c for c in counts if local_day(c.ts) < start]
    during = [c for c in counts if start <= local_day(c.ts) <= end]
    if not during:
        return None
    last = max(during, key=lambda c: c.ts)
    base = max(before, key=lambda c: c.ts) if before else None
    sub = (base.recycled, base.recyclable_seen, base.contamination) if base else (0, 0, 0)
    return (max(0, last.recycled - sub[0]), max(0, last.recyclable_seen - sub[1]), max(0, last.contamination - sub[2]))


def score_week(cfg: LeagueConfig, week_start: date, names: dict[str, str], habits: list[DaySummary],
               counts: list[Count], plans: list[PlanStars], local_day) -> list[MemberWeek]:
    """Every named member's week, best first. Members with data but no name are included under their id."""
    end = week_end(week_start)
    members = dict(names)
    for s in habits:
        members.setdefault(s.member_id, s.member_id)
    for c in counts:
        members.setdefault(c.member_id, c.member_id)
    weeks: dict[str, MemberWeek] = {m: MemberWeek(m, n) for m, n in members.items()}

    for target, rule in cfg.targets.items():
        label, unit = cfg.label(target), rule.unit
        for m, week in weeks.items():
            if rule.source == "habit":
                days = {s.day: s for s in habits if s.member_id == m and s.habit_id == target and week_start <= s.day <= end}
                if days:
                    week.targets[target] = TargetScore(target, label, unit, rule.weight,
                                                       sum(1 for s in days.values() if s.met), len(days))
            elif rule.source == "dayrun":
                days = {p.day: p for p in plans if p.member_id == m and week_start <= p.day <= end}
                if days:
                    week.targets[target] = TargetScore(target, label, unit, rule.weight,
                                                       sum(1 for p in days.values() if p.stars >= rule.min_stars), len(days))
            elif rule.source == "impact":
                delta = weekly_delta([c for c in counts if c.member_id == m], week_start, end, local_day)
                if delta and delta[1] > 0:
                    rate = delta[0] / delta[1]
                    met = rate >= rule.min_rate
                    week.targets[target] = TargetScore(target, label, unit, rule.weight, int(met), 1,
                                                       detail=f"recycled {rate:.0%} of what could be (target {rule.min_rate:.0%})")
    return sorted(weeks.values(), key=_rank_key)


def _rank_key(w: MemberWeek):
    score = w.score
    return (score is None, -(score or 0), -w.met, w.member_id)


def worst_and_best(weeks: list[MemberWeek]) -> tuple[MemberWeek | None, MemberWeek | None]:
    """Only members with something judged can win or lose. One such member is the best and nobody is the worst."""
    judged = [w for w in weeks if w.score is not None]
    if not judged:
        return None, None
    ranked = sorted(judged, key=_rank_key)
    best = ranked[0]
    worst = ranked[-1] if len(ranked) > 1 else None
    return worst, best


def standings(weeks: list[MemberWeek]) -> list[Standing]:
    return [Standing(member_id=w.member_id, display_name=w.display_name, rank=i, score=w.score, met=w.met,
                     judged=w.judged, reasons=w.reasons()) for i, w in enumerate(weeks, start=1)]


def summary_text(worst: MemberWeek | None, best: MemberWeek | None, stake: Stake) -> str:
    """"Maya had the worst week: 2 of 7 sleep targets. Dinner is on Maya. Hemdutt picks the place.\""""
    if best is None:
        return "Nothing was judged this week, so nobody owes anything."
    if worst is None:
        return f"Only {best.display_name} had a judged week ({best.headline()}). Nobody owes anything."
    you_worst, you_best = worst.display_name == YOU, best.display_name == YOU
    lead = f"{worst.display_name} had the worst week: {worst.headline()}."
    if stake.kind == "pays_for_food":
        on = "you" if you_worst else worst.display_name
        picks = "You pick" if you_best else f"{best.display_name} picks"
        return f"{lead} Dinner is on {on}. {picks} the place."
    if stake.kind == "picks_the_event":
        picks = "pick" if you_best else "picks"
        return f"{lead} {best.display_name} had the best week ({best.headline()}) and {picks} the next outing."
    text = (stake.text or "").strip().rstrip(".")
    return f"{lead} {text}." if text else lead


def stake_text(stake: Stake) -> str:
    if stake.kind == "pays_for_food":
        return "The worst week buys the next group meal; the best week picks the place."
    if stake.kind == "picks_the_event":
        return "The best week picks the next outing."
    return stake.text or "A custom stake."


@dataclass
class TeamWaste:
    members: list[dict]
    recycled: int
    recyclable: int
    rate: float | None
    target: float
    line: str
    contamination_of_the_week: str | None


def team_waste(cfg: LeagueConfig, names: dict[str, str], counts: list[Count], week_start: date, local_day) -> TeamWaste:
    """Table vs table (OneEthos): everyone's bins together this week against one target, said kindly."""
    end = week_end(week_start)
    rows = []
    for m in sorted({c.member_id for c in counts} | set(names), key=lambda m: (m not in names, m)):
        delta = weekly_delta([c for c in counts if c.member_id == m], week_start, end, local_day)
        if delta is None:
            continue
        recycled, recyclable, contamination = delta
        rows.append({"member_id": m, "display_name": names.get(m, m), "recycled": recycled, "recyclable": recyclable,
                     "contamination": contamination, "diversion_rate": round(recycled / recyclable, 3) if recyclable else None})
    recycled = sum(r["recycled"] for r in rows)
    recyclable = sum(r["recyclable"] for r in rows)
    rate = round(recycled / recyclable, 3) if recyclable else None
    target = cfg.team_diversion_target
    if rate is None:
        line = f"team: nothing counted yet this week, target {target:.0%}"
    else:
        line = f"team: {rate:.0%} diverted, target {target:.0%}" + (" — made it" if rate >= target else "")
    worst = max((r for r in rows if r["contamination"] > 0), key=lambda r: (r["contamination"], r["member_id"]), default=None)
    kindly = None
    if worst is not None:
        n = worst["contamination"]
        kindly = (f"{worst['display_name']}'s bin had the most mix-ups this week ({n} {'item' if n == 1 else 'items'} "
                  "in recycling that didn't belong). Happens to everyone; check the label before you toss.")
    return TeamWaste(rows, recycled, recyclable, rate, target, line, kindly)
