"""The Impact screen's story: what Waste Watch's counts mean, and what the table can do about them.

    story = ImpactStory(waste, RuleBook.load(sorting_file, config_dir), Factors.load(config_dir / "impact.yaml"), tz=tz)
    story.subscribe(bus)
    story.view(now)            # GET /api/impact/story
    story.practice(now)        # GET /api/impact/practice: the week's mistakes, for Sort It
    story.day_clause(day, now) # one clause for the evening review

Waste Watch (contexts/impact/domain/waste_watch.py) stays the counter and the only judge of right and
wrong; nothing here counts an item a second time. This module reads its `seen` and `disposed` and adds:

- real units: grams, kWh, phone charges and CO2e as RANGES, from the editable factors in config/impact.yaml.
  They are estimates and every figure says so;
- the moment of disposal: a feed of what went where, with the rule's sentence for a wrong one;
- goals and momentum: the weekly goal, clean days in a row, the best day, a projection, before and after
  nudges with n on each side (and "not enough data yet" under a minimum n);
- milestones, as `impact.milestone.reached` (live only; once per week, kind and threshold);
- the practice set: the items the table got wrong this week, plus the ones someone tapped "teach me" on
  (`impact.practice.added`).

State that rebuilds from the log (those handlers replay): who was tagged on which camera (owner
recognition), which disposal belongs to whom, the milestones already said, the practice taps.
Publishing a milestone is live only. Every duration is event time; nothing here reads the wall clock.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from contexts.impact.domain.waste_watch import WasteWatch
from contexts.play.domain.pathfinding import SortingRules
from contracts.events import Event
from contracts.impact import ItemDisposed, MilestoneReached, PracticeAdded

FRAME_REF = re.compile(r"frames:([\w.-]+):([\w-]+)")
FEED_LIMIT = 20
EMPTY_TEXT = ("Nothing counted yet. When a camera sees a can, a bottle or a cup at the table and which bin it ends in, "
              "this screen fills in: each disposal with its picture and whether the bin was right, what the week adds up to "
              "in grams and energy, and the table's goal for the week.")


def week_of(day: date) -> date:
    """The local Monday the week starts on (the League's week)."""
    return day - timedelta(days=day.weekday())


def _a(word: str) -> str:
    return f"{'An' if word[:1].lower() in 'aeiou' else 'A'} {word}"


# --- the rules file, with its sentences ---


@dataclass
class RuleBook:
    """One sorting rules file (config/sorting.yaml or a profile's copy): the bins, and one sentence per item."""

    rules: SortingRules
    file: str = "config/sorting.yaml"  # as shown on screen
    name: str = "Sorting rules"
    place: str | None = None
    note: str | None = None
    reasons: dict[str, str] = field(default_factory=dict)
    default_reason: str | None = None
    overridden: bool = False  # True when a profile's file replaced the base one

    @classmethod
    def load(cls, path: str | Path = "config/sorting.yaml", config_dir: str | Path | None = None) -> "RuleBook":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config_dir = Path(config_dir) if config_dir else None
        shown, overridden = str(path), False
        if config_dir is not None:
            try:
                rel = path.resolve().relative_to(config_dir.resolve())
                shown, overridden = f"config/{rel.as_posix()}", rel.as_posix() != "sorting.yaml"
            except ValueError:
                pass
        return cls(SortingRules(raw["bins"], raw["items"], raw.get("aliases") or {}), file=shown,
                   name=raw.get("name") or path.stem, place=raw.get("place"), note=raw.get("note"),
                   reasons=dict(raw.get("reasons") or {}), default_reason=raw.get("default_reason"), overridden=overridden)

    def item(self, label: str) -> str:
        """The listed item a detector label names ("soda can" -> "can"); the label itself when unlisted."""
        return self.rules.canonical(label) or label

    def reason(self, label: str) -> str:
        """Where this item goes here, in one sentence."""
        item = self.item(label)
        if item in self.reasons:
            return self.reasons[item]
        where = f" at {self.place}" if self.place else ""
        if item not in self.rules.items and self.default_reason:
            return self.default_reason
        return f"{_a(item)} goes in {self.rules.bin_for(label)}{where}."

    def verdict(self, label: str, bin: str) -> dict:
        """Right or wrong for this file, with the sentence."""
        belongs = self.rules.bin_for(label)
        return {"correct": belongs == bin, "belongs_in": belongs, "rule": self.reason(label)}

    def summary(self) -> dict:
        return {"file": self.file, "name": self.name, "place": self.place, "note": self.note, "overridden": self.overridden,
                "bins": list(self.rules.bins),
                "items": [{"label": i, "bin": b, "rule": self.reason(i)} for i, b in self.rules.items.items()],
                "default_rule": self.default_reason,
                "swap": "Another campus reuses Waste Watch by replacing this one file (or adding config/profiles/<name>/sorting.yaml)."}


# --- the factors file ---


@dataclass
class Factors:
    """config/impact.yaml: estimated masses per item, energy and CO2e per material, the goal, the milestones."""

    raw: dict = field(default_factory=dict)
    file: str = "config/impact.yaml"
    loaded: bool = False

    @classmethod
    def load(cls, path: str | Path = "config/impact.yaml") -> "Factors":
        path = Path(path)
        if not path.is_file():
            return cls()
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, f"config/{path.name}", True)

    @property
    def goals(self) -> dict:
        g = self.raw.get("goals") or {}
        return {"diversion_target": float(g.get("diversion_target", 0.8)), "contamination_ceiling": int(g.get("contamination_ceiling", 2)),
                "min_n": int(g.get("min_n", 5)), "min_day_items": int(g.get("min_day_items", 2))}

    def thresholds(self, kind: str) -> list[int]:
        default = {"kept_out": [1, 3, 5, 10, 25, 50, 100, 250], "clean_streak": [3, 5, 7]}[kind]
        return sorted(int(n) for n in (self.raw.get("milestones") or {}).get(kind, default))

    def equivalents(self, items: list[str]) -> dict | None:
        """Grams, kWh, phone charges and CO2e for these recycled items, low to high. None without a factors file."""
        if not self.loaded:
            return None
        table, materials = self.raw.get("items") or {}, self.raw.get("materials") or {}
        counts: dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        rows, unweighed = [], []
        g = [0.0, 0.0]
        kwh = [0.0, 0.0]
        co2 = [0.0, 0.0]
        for item, n in sorted(counts.items()):
            f = table.get(item)
            if not f or not f.get("mass_g"):
                unweighed.append({"item": item, "count": n})
                continue
            lo, hi = n * float(f["mass_g"]["low"]), n * float(f["mass_g"]["high"])
            g[0] += lo
            g[1] += hi
            m = materials.get(f.get("material")) or {}
            row = {"item": item, "count": n, "material": m.get("name") or f.get("material"),
                   "mass_g": {"low": round(lo, 1), "high": round(hi, 1)}, "each_g": f["mass_g"],
                   "mass_source": f.get("source"), "mass_confidence": f.get("confidence", "low"),
                   "energy_kwh": None, "co2e_kg": None, "material_source": m.get("source"), "material_confidence": m.get("confidence")}
            if m.get("energy_kwh_per_kg"):
                e = (lo / 1000 * float(m["energy_kwh_per_kg"]["low"]), hi / 1000 * float(m["energy_kwh_per_kg"]["high"]))
                kwh[0] += e[0]
                kwh[1] += e[1]
                row["energy_kwh"] = {"low": round(e[0], 3), "high": round(e[1], 3)}
            if m.get("co2e_kg_per_kg"):
                c = (lo / 1000 * float(m["co2e_kg_per_kg"]["low"]), hi / 1000 * float(m["co2e_kg_per_kg"]["high"]))
                co2[0] += c[0]
                co2[1] += c[1]
                row["co2e_kg"] = {"low": round(c[0], 3), "high": round(c[1], 3)}
            rows.append(row)
        eq = self.raw.get("equivalents") or {}
        charge = eq.get("phone_charge_wh") or {"low": 12, "high": 20}
        # fewest charges: the least energy into the hungriest phone; most: the reverse
        charges = {"low": int(kwh[0] * 1000 // float(charge["high"])), "high": int(kwh[1] * 1000 // float(charge["low"]))}
        return {
            "estimate": True,
            "disclaimer": self.raw.get("disclaimer") or "Estimates from illustrative factors, not measurements.",
            "file": self.file,
            "items": len(items),
            "kept_out_g": {"low": round(g[0]), "high": round(g[1])},
            "energy_kwh": {"low": round(kwh[0], 2), "high": round(kwh[1], 2)},
            "phone_charges": charges,
            "phone_charge_source": eq.get("phone_charge_source"),
            "phone_charge_confidence": eq.get("phone_charge_confidence", "medium"),
            "co2e_kg": {"low": round(co2[0], 2), "high": round(co2[1], 2)},
            "by_item": rows,
            "unweighed": unweighed,  # counted, but no mass in the factors file: nothing is claimed for them
        }

    def contamination(self, wrong_in_recycling: int, good_in_same_bin: int) -> dict:
        c = self.raw.get("contamination") or {}
        at = int(c.get("spoils_bag_at", 2))
        text = (c.get("text") or "{n} wrong items can spoil a bag.").format(n=at)
        return {"wrong_in_recycling": wrong_in_recycling, "spoils_bag_at": at, "at_risk": wrong_in_recycling >= at,
                "good_items_sharing_the_bin": good_in_same_bin, "text": text, "source": c.get("source"),
                "confidence": c.get("confidence", "low"), "rule_of_thumb": bool(c.get("rule_of_thumb", True))}


# --- the story ---


class ImpactStory:
    def __init__(self, waste: WasteWatch, book: RuleBook, factors: Factors | None = None, tz: tzinfo | None = None,
                 frames_dir: str | Path | None = None, profile: str | None = None) -> None:
        self.waste, self.book = waste, book
        self.factors = factors or Factors()
        self.tz = tz or ZoneInfo("America/New_York")
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.profile = profile
        # --- state, rebuilt from the log ---
        self.recognition = False  # owner recognition on, with someone enrolled
        self.members: list[str] = []
        self.tags: dict[tuple[str, int], str] = {}  # (camera, track) -> member
        self.who: dict[str, str] = {}  # item id -> the one tagged person at that camera when it was thrown away
        self.said: dict[tuple[date, str, str | None], MilestoneReached] = {}  # (week, kind, member) -> the highest said
        self.milestones: list[MilestoneReached] = []
        self.asked: list[PracticeAdded] = []  # "teach me" taps

    def subscribe(self, bus) -> None:
        bus.subscribe("perception.identity", self.on_identity, replay=True)
        bus.subscribe("perception.person", self.on_person, replay=True)
        bus.subscribe("impact.item.disposed", self.on_disposed, replay=True)  # state only: whose it was
        bus.subscribe("impact.milestone.reached", self.on_milestone, replay=True)
        bus.subscribe("impact.practice.added", self.on_practice, replay=True)
        bus.subscribe("impact.item.disposed", self.celebrate)  # live only: publishes

    # --- state handlers (replay=True) ---

    def on_identity(self, e: Event) -> None:
        self.members = list(getattr(e, "members", []) or [])
        self.recognition = bool(getattr(e, "enabled", False)) and bool(self.members)
        if not self.recognition:
            self.tags.clear()

    def on_person(self, e: Event) -> None:
        if e.type == "perception.person.identified":
            self.tags[e.camera_id, e.track_id] = e.member_id
        elif e.type == "perception.person.lost":
            for key in [k for k in self.tags if k[0] == e.camera_id and (e.track_id is None or k[1] == e.track_id)]:
                del self.tags[key]

    def on_disposed(self, e: ItemDisposed) -> None:
        """Whose disposal: the one recognised person at that camera at that moment. Two people, or nobody
        recognised, and it stays the table's."""
        if not self.recognition:
            return
        camera = e.item_id.rsplit(":", 1)[0]
        there = {m for (cam, _), m in self.tags.items() if cam == camera}
        if len(there) == 1:
            self.who[e.item_id] = next(iter(there))

    def on_milestone(self, e: MilestoneReached) -> None:
        key = (e.week_start, e.kind, e.member_id)
        if key not in self.said or e.threshold > self.said[key].threshold:
            self.said[key] = e
        if all(m.event_id != e.event_id for m in self.milestones):
            self.milestones.append(e)

    def on_practice(self, e: PracticeAdded) -> None:
        self.asked.append(e)

    # --- time ---

    def day_of(self, ts: datetime) -> date:
        return ts.astimezone(self.tz).date()

    def start_of(self, day: date) -> datetime:
        return datetime.combine(day, time(0), self.tz)

    def week_span(self, now: datetime) -> tuple[date, datetime, datetime]:
        """The local week `now` is in, up to `now`: (its Monday, its start, the moment after `now` or its end)."""
        week = week_of(self.day_of(now))
        return week, self.start_of(week), min(self.start_of(week + timedelta(days=7)), now + timedelta(microseconds=1))

    # --- live: milestones ---

    def celebrate(self, e: ItemDisposed) -> list[Event]:
        """After a disposal: has the week's count passed a round number nobody has heard yet?"""
        week, start, end = self.week_span(e.ts)
        out: list[Event] = []
        for kind, value in (("kept_out", self.waste.metrics(since=start, until=end)["recycled"]),
                            ("clean_streak", self.clean_streak(e.ts))):
            passed = [n for n in self.factors.thresholds(kind) if n <= value]
            said = self.said.get((week, kind, None))
            if not passed or (said is not None and said.threshold >= passed[-1]):
                continue
            n = passed[-1]  # the highest one passed: one sentence, not a burst
            m = MilestoneReached(event_id=f"impact-milestone-{week}-{kind}-{n}", ts=e.ts, week_start=week, kind=kind,
                                 threshold=n, value=value, text=milestone_text(kind, n))
            self.said[week, kind, None] = m  # so a second disposal in the same instant does not say it again
            out.append(m)
        return out

    # --- the numbers ---

    def disposals(self, since: datetime | None = None, until: datetime | None = None) -> list[ItemDisposed]:
        return sorted((d for d in self.waste.disposed.values() if (since is None or d.ts >= since) and (until is None or d.ts < until)),
                      key=lambda d: (d.ts, d.item_id))

    def days(self, start: date, end: date, until: datetime | None = None) -> list[dict]:
        """One row per local day in [start, end]: what was thrown away and whether the day was clean."""
        by_day: dict[date, list[ItemDisposed]] = {}
        last = self.start_of(end + timedelta(days=1))
        for d in self.disposals(self.start_of(start), min(last, until) if until else last):
            by_day.setdefault(self.day_of(d.ts), []).append(d)
        out = []
        for i in range((end - start).days + 1):
            day = start + timedelta(days=i)
            ds = by_day.get(day, [])
            right = sum(1 for d in ds if d.correct)
            out.append({"day": day.isoformat(), "disposals": len(ds), "right": right, "wrong": len(ds) - right,
                        "recycled": sum(1 for d in ds if d.bin == "recycling" and d.correct),
                        "contamination": sum(1 for d in ds if d.bin == "recycling" and not d.correct),
                        "clean": bool(ds) and right == len(ds),
                        "right_share": round(right / len(ds), 3) if ds else None})
        return out

    def clean_streak(self, now: datetime) -> int:
        """Days in a row, up to today, on which everything thrown away went in the right bin. A day with
        nothing thrown away neither counts nor breaks it."""
        today = self.day_of(now)
        by_day: dict[date, bool] = {}
        for d in self.disposals(until=now + timedelta(microseconds=1)):
            day = self.day_of(d.ts)
            by_day[day] = by_day.get(day, True) and d.correct
        streak = 0
        for day in sorted(by_day, reverse=True):
            if not by_day[day]:
                break
            streak += 1
        return streak

    def recycled_items(self, since: datetime, until: datetime) -> list[str]:
        """The listed item behind each recyclable that reached recycling in the window (Waste Watch's `recycled`)."""
        seen = {s.item_id for s in self.waste.seen.values() if since <= s.ts < until and s.recyclable}
        return [self.book.item(d.label) for d in self.disposals(since, until) if d.bin == "recycling" and d.item_id in seen]

    def frame_stored(self, ref: str | None) -> bool:
        m = FRAME_REF.fullmatch(ref) if ref else None
        return bool(m and self.frames_dir and ".." not in m[1] and (self.frames_dir / m[1] / f"{m[2]}.jpg").is_file())

    def entry(self, d: ItemDisposed, practising: set[str]) -> dict:
        v = self.book.verdict(d.label, d.bin)
        item = self.book.item(d.label)
        return {"item_id": d.item_id, "ts": d.ts.isoformat(), "day": self.day_of(d.ts).isoformat(), "label": d.label, "item": item,
                "bin": d.bin, "zone": d.zone, "correct": d.correct, "belongs_in": v["belongs_in"], "rule": v["rule"],
                "frame_ref": d.frame_ref, "frame_stored": self.frame_stored(d.frame_ref),
                "member_id": self.who.get(d.item_id), "in_practice": item in practising}

    def feed(self, now: datetime, member: str | None = None, limit: int = FEED_LIMIT) -> list[dict]:
        """The most recent disposals, newest first; one person's when `member` is given."""
        practising = {p["item"] for p in self.practice(now, member)["items"]}
        ds = [d for d in reversed(self.disposals(until=now + timedelta(microseconds=1))) if member is None or self.who.get(d.item_id) == member]
        return [self.entry(d, practising) for d in ds[:limit]]

    # --- the practice set (Sort It) ---

    def practice(self, now: datetime, member: str | None = None) -> dict:
        """GET /api/impact/practice: the items the table got wrong this week, the most mistakes first, then the
        ones someone asked to be taught. Each carries where it belongs and why, in the active rules file's words."""
        week, start, end = self.week_span(now)
        rows: dict[str, dict] = {}

        def row(item: str) -> dict:
            return rows.setdefault(item, {"item": item, "mistakes": 0, "asked": False, "last_wrong_at": None, "went_to": None,
                                          "belongs_in": self.book.rules.bin_for(item), "zone": self.book.rules.zone_for(item),
                                          "rule": self.book.reason(item), "frame_ref": None, "members": []})

        for d in self.disposals(start, end):
            if d.correct or (member is not None and self.who.get(d.item_id) != member):
                continue
            r = row(self.book.item(d.label))
            r["mistakes"] += 1
            r.update(last_wrong_at=d.ts.isoformat(), went_to=d.bin, frame_ref=d.frame_ref)
            if (who := self.who.get(d.item_id)) and who not in r["members"]:
                r["members"].append(who)
        for p in self.asked:
            if start <= p.ts < end and (member is None or p.member_id in (None, member)):
                row(self.book.item(p.label))["asked"] = True
        items = sorted(rows.values(), key=lambda r: (-r["mistakes"], not r["asked"], r["item"]))
        return {"week_start": week.isoformat(), "member_id": member, "size": len(items), "items": items,
                "rules": {"file": self.book.file, "name": self.book.name, "place": self.book.place},
                "how": "Play these first in Sort It: each row is an item to hold up, `zone` is the bin zone that scores, `rule` is what to say after."}

    def add_practice(self, label: str, now: datetime, item_id: str | None = None, member: str | None = None) -> PracticeAdded:
        """The "teach me" tap. Only an item the rules file knows, or one that was really thrown away, can be added."""
        item = self.book.item(label)
        known = item in self.book.rules.items or any(self.book.item(d.label) == item for d in self.waste.disposed.values())
        if not known:
            raise ValueError(f"'{label}' is not an item in {self.book.file} and was never thrown away here")
        if item_id is not None and member is None:
            member = self.who.get(item_id)
        return PracticeAdded(ts=now, label=item, item_id=item_id, member_id=member)

    # --- the view ---

    def view(self, now: datetime, member: str | None = None) -> dict:
        """GET /api/impact/story."""
        week, start, end = self.week_span(now)
        today = self.day_of(now)
        goals = self.factors.goals
        total, this_week = self.waste.metrics(), self.waste.metrics(since=start, until=end)
        days = self.days(week, week + timedelta(days=6), until=end)
        recycled = self.recycled_items(start, end)

        # the goal
        n, done = this_week["recyclable_seen"], this_week["recycled"]
        rate = this_week["diversion_rate"] if n else None
        target = goals["diversion_target"]
        more = 0
        while n and (done + more) / (n + more) < target and more < 10_000:
            more += 1
        goal = {
            "diversion": {"target": target, "value": rate, "recycled": done, "recyclable": n, "met": rate is not None and rate >= target,
                          "progress": None if rate is None else round(min(rate / target, 1.0), 3) if target else 1.0,
                          "to_go": more if n else None,
                          "text": ("Nothing recyclable seen yet this week." if not n else
                                   f"{done} of {n} recyclables reached recycling: the goal is met." if rate >= target else
                                   f"{done} of {n} recyclables reached recycling. The next {more} in the right bin, with none missed, reaches {target:.0%}.")},
            "contamination": {"ceiling": goals["contamination_ceiling"], "value": this_week["contamination"],
                              "met": this_week["contamination"] <= goals["contamination_ceiling"],
                              "left": max(0, goals["contamination_ceiling"] - this_week["contamination"])},
        }

        # momentum
        eligible = [d for d in days if d["disposals"] >= goals["min_day_items"]]
        best = max(eligible, key=lambda d: (d["right_share"], d["disposals"], d["day"]), default=None)
        active = [d for d in days if d["disposals"] and date.fromisoformat(d["day"]) <= today]
        left = max(0, (week + timedelta(days=6) - today).days)
        if len(active) >= 2 and done:
            pace = done / len(active)
            projected = round(done + pace * left)
            projection = {"enough": True, "recycled_by_week_end": projected, "days_with_data": len(active), "days_left": left,
                          "text": (f"At this pace (about {pace:.1f} a day on the {len(active)} days with items), about {projected} "
                                   f"item{'' if projected == 1 else 's'} kept out of landfill by Sunday night.")}
        else:
            projection = {"enough": False, "recycled_by_week_end": None, "days_with_data": len(active), "days_left": left,
                          "text": "Too early to project: it takes two days with items this week."}

        # before and after nudges
        ba = self.waste.before_after()
        min_n = goals["min_n"]
        if ba is None:
            before_after = {"available": False, "enough": False, "min_n": min_n, "nudges_on_at": None, "before": None, "after": None,
                            "text": "Nudges about sorting have not started on this box, so there is no before and after yet."}
        else:
            sides = {k: {"n": m["recyclable_seen"], "recycled": m["recycled"], "contamination": m["contamination"],
                         "diversion_rate": m["diversion_rate"] if m["recyclable_seen"] else None} for k, m in ba.items()}
            enough = all(s["n"] >= min_n for s in sides.values())
            before_after = {"available": True, "enough": enough, "min_n": min_n, "nudges_on_at": self.waste.nudges_on_at.isoformat(),
                            **sides,
                            "text": ("Diversion before the first sorting nudge against after it (n = recyclables seen on each side)." if enough else
                                     f"Not enough data yet: {sides['before']['n']} recyclables before the first sorting nudge and "
                                     f"{sides['after']['n']} after; the bars are drawn once each side has {min_n}.")}

        # milestones
        said = sorted((m for m in self.milestones if m.week_start == week), key=lambda m: (m.ts, m.threshold))
        nxt = next((t for t in self.factors.thresholds("kept_out") if t > done), None)
        milestones = {
            "reached": [m.model_dump(mode="json") for m in said],
            "latest": said[-1].model_dump(mode="json") if said else None,
            "next": None if nxt is None else {"kind": "kept_out", "threshold": nxt, "to_go": nxt - done,
                                              "text": f"{nxt - done} more in recycling makes {nxt} this week."},
        }

        practice = self.practice(now, member)
        wrong_days = {d["day"] for d in days if d["contamination"]}
        good_sharing = sum(d["recycled"] for d in days if d["day"] in wrong_days)
        return {
            "now": now.isoformat(),
            "week": {"start": week.isoformat(), "end": (week + timedelta(days=6)).isoformat(), "today": today.isoformat()},
            "empty": total["items_seen"] == 0,
            "empty_text": EMPTY_TEXT if total["items_seen"] == 0 else None,
            "rules": self.book.summary(),
            "profile": self.profile,
            "scope": {"per_person": self.recognition, "members": self.members if self.recognition else [], "member_id": member,
                      "note": ("Owner recognition is on: a disposal is someone's when they were the only recognised person at that "
                               "camera. The counts stay the table's." if self.recognition else
                               "Owner recognition is off, so everything here is the table's, not any one person's.")},
            "total": total,
            "this_week": this_week,
            "equivalents": self.factors.equivalents(recycled),
            "contamination": self.factors.contamination(this_week["contamination"], good_sharing),
            "feed": self.feed(now, member),
            "goal": goal,
            "days": days,
            "clean_streak": self.clean_streak(now),
            "best_day": best,
            "projection": projection,
            "before_after": before_after,
            "milestones": milestones,
            "practice": {"size": practice["size"], "items": practice["items"]},
        }

    # --- for the League and the evening review ---

    def week_brief(self, now: datetime) -> dict:
        """What the League's team waste line adds: the week's latest milestone and the size of the practice set."""
        week, _, _ = self.week_span(now)
        said = [m for m in self.milestones if m.week_start == week]
        latest = max(said, key=lambda m: (m.ts, m.threshold), default=None)
        return {"milestone": latest.text if latest else None, "practice_size": self.practice(now)["size"]}

    def day_clause(self, day: date, now: datetime | None = None) -> str | None:
        """The day's sorting in one clause, for the evening review: "3 of 4 things you threw away went in the right
        bin; the food container belonged in trash". None when nothing was thrown away that day."""
        end = self.start_of(day + timedelta(days=1))
        ds = self.disposals(self.start_of(day), min(end, now) if now else end)
        if not ds:
            return None
        right = sum(1 for d in ds if d.correct)
        if right == len(ds):
            return (f"all {len(ds)} things you threw away went in the right bin" if len(ds) > 1 else
                    f"the {self.book.item(ds[0].label)} you threw away went in the right bin")
        wrong = [d for d in ds if not d.correct]
        first = wrong[0]
        more = f" (and {len(wrong) - 1} more)" if len(wrong) > 1 else ""
        return (f"{right} of {len(ds)} things you threw away went in the right bin; the {self.book.item(first.label)} "
                f"belonged in {self.book.rules.bin_for(first.label)}{more}")


def milestone_text(kind: str, n: int) -> str:
    if kind == "kept_out":
        return "First item kept out of landfill this week." if n == 1 else f"That's {n} items kept out of landfill this week."
    return f"{n} clean days in a row: nothing in the wrong bin."
