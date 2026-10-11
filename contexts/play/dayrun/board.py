"""The board for a day: what is fixed in the next 24 hours, what is due, and the open slots between.

    board = build_board(cfg, walks, start, items, initial, friends)

`items` are connector items (contracts.memory.Item) read through a MemoryReader:
calendar events become fixed blocks (or optional events, for the campus feed),
assignments become deadlines. Every open slot lists concrete moves with the
walking route from where the previous block leaves you. Pure: no clock, no I/O.
"""

import re
from datetime import datetime, timedelta

from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.models import (
    Board,
    Deadline,
    FixedBlock,
    FriendAt,
    Initial,
    InitialOverride,
    MoveOption,
    OpenSlot,
    OptionalEvent,
    Rules,
)
from contexts.play.dayrun.routes import Walks
from contracts.memory import Item

UNTIL = re.compile(r"until\s+(\d{1,2}):(\d{2})\s*([AP]M)", re.I)
COURSE = re.compile(r"\b([A-Za-z]{3})\s?(\d{4}[A-Za-z]?)\b")


def local(cfg: DayRunConfig, t: datetime) -> datetime:
    return t.replace(tzinfo=cfg.tz) if t.tzinfo is None else t.astimezone(cfg.tz)


def to_tick(cfg: DayRunConfig, t: datetime) -> datetime:
    """`t` in local time, moved up to the next tick boundary."""
    t = local(cfg, t).replace(second=0, microsecond=0)
    over = t.minute % cfg.tick_minutes
    return t + timedelta(minutes=cfg.tick_minutes - over) if over else t


def hm(t: datetime) -> str:
    return t.strftime("%H:%M")


def minutes_between(a: datetime, b: datetime) -> int:
    return round((b - a).total_seconds() / 60)


def risk_of(cfg: DayRunConfig, work_left: float, now: datetime, due: datetime) -> float:
    """Share of the realistic study capacity left before `due` that the remaining work would take (0 to 1)."""
    if work_left <= 0:
        return 0.0
    hours = (due - now).total_seconds() / 3600
    capacity = hours * cfg.capacity_minutes_per_day / 24
    return 1.0 if capacity <= 0 else round(min(1.0, work_left / capacity), 3)


def work_estimate(cfg: DayRunConfig, title: str) -> float:
    return next((w.minutes for w in cfg.work_estimates if re.search(w.match, title, re.I)), cfg.default_work_minutes)


def estimate_of(cfg: DayRunConfig, title: str) -> tuple[float, str, str | None]:
    """(minutes, "kind" | "default", the word in the title that matched): where a deadline's work estimate comes from."""
    for w in cfg.work_estimates:
        if m := re.search(w.match, title, re.I):
            return w.minutes, "kind", m[0].lower()
    return cfg.default_work_minutes, "default", None


def rules_of(cfg: DayRunConfig, w: Walks) -> Rules:
    pack = w.places.packs.get(cfg.pack)
    speed = None
    if pack is not None and getattr(pack, "_net", None) is not None:  # walks are timed along its footpaths
        from contexts.play.domain.walknet import DEFAULT_SPEED_MPS

        speed = pack.walk_speed_mps or DEFAULT_SPEED_MPS
    return Rules(capacity_minutes_per_day=cfg.capacity_minutes_per_day, on_track_risk=cfg.on_track_risk,
                 win_sleep_hours=cfg.win.sleep_hours, win_meals=cfg.win.meals, need_sleep_hours=cfg.sleep.need_hours,
                 walk_meters_per_minute=round(speed * 60, 1) if speed else None)


def course_of(text: str) -> str | None:
    m = COURSE.search(text)
    return f"{m[1]}{m[2]}".upper() if m else None


def initial_state(cfg: DayRunConfig, start: datetime, habits: list[dict] | None, over: InitialOverride | None = None,
                  origin: str = "habits") -> Initial:
    """Where the meters start: last night's sleep and today's meals from the Habits digest, else defaults.

    `origin` names what `habits` really is ("scenario" for a scripted day). `source` on the result says
    where each value came from: one word when they agree, else e.g. "defaults (sleep), habits (meals)"."""
    start = local(cfg, start)
    rows = {r["id"]: r for r in habits or [] if r.get("applies", True)}
    sleep = rows["sleep"]["value"] if "sleep" in rows else None
    sleep_from = rows["sleep"].get("source") if "sleep" in rows and not (over and over.sleep_hours is not None) else None
    meals = rows["meals"]["value"] if "meals" in rows else None
    asked = {"sleep": over.sleep_hours, "meals": over.meals, "energy": over.energy} if over else {}
    came_from: dict[str, list[str]] = {}
    for name, value in (("sleep", sleep), ("meals", meals), ("energy", None)):
        if asked.get(name) is not None:
            came_from.setdefault("request", []).append(name)
        elif name != "energy":  # energy is worked out from the other two unless the request sets it
            came_from.setdefault(origin if value is not None else "defaults", []).append(name)
    source = next(iter(came_from)) if len(came_from) == 1 else ", ".join(f"{s} ({' and '.join(v)})" for s, v in came_from.items())
    sleep = over.sleep_hours if over and over.sleep_hours is not None else sleep if sleep is not None else cfg.start.default_sleep_hours
    meals = int(over.meals if over and over.meals is not None else meals or 0)

    s = cfg.start
    debt = max(0.0, cfg.sleep.need_hours - sleep)
    wake = datetime.combine(start.date(), cfg.sleep.wake, cfg.tz)
    awake_h = min(16.0, max(0.0, (start - wake).total_seconds() / 3600))  # the day so far has cost something too
    energy = s.energy_rested - s.energy_per_missed_hour * debt + cfg.energy_per_hour["idle"] * awake_h
    if meals == 0 and start.time() >= s.no_meal_after:
        energy -= s.no_meal_penalty
    hungry = s.hungry_minutes_no_meal if meals == 0 else s.hungry_minutes_default
    out = Initial(energy=round(max(5.0, min(100.0, energy)), 1), focus=s.focus, sleep_hours=sleep, meals=meals,
                  sleep_debt=round(debt, 2), hungry_minutes=hungry, source=source,
                  sleep_from=sleep_from)
    if over:
        patch = {k: v for k, v in over.model_dump(include={"energy", "focus", "sleep_debt", "hungry_minutes"}).items() if v is not None}
        out = out.model_copy(update=patch)
    return out


def _end_and_place(cfg: DayRunConfig, w: Walks, item: Item, start: datetime) -> tuple[datetime | None, str | None]:
    """A calendar item's end ("until 5:00 PM") and place ("· Library, 3rd floor") from its one-line summary."""
    summary = item.summary or ""
    if summary.lower().startswith("all day"):
        return None, None
    end = start + timedelta(minutes=cfg.default_event_minutes)
    if m := UNTIL.search(summary):
        hour = int(m[1]) % 12 + (12 if m[3].upper() == "PM" else 0)
        end = start.replace(hour=hour, minute=int(m[2]))
        if end <= start:
            end += timedelta(days=1)
    where = " · ".join(part for part in summary.split("·") if not UNTIL.search(part))
    return end, w.locate(where) or w.locate(item.title)


def build_board(
    cfg: DayRunConfig,
    w: Walks,
    start: datetime,
    items: list[Item],
    initial: Initial,
    friends: list[FriendAt] | None = None,
    over: InitialOverride | None = None,
    listed=None,  # (item) -> bool: a public listing, not the person's own commitment (agent.yaml `listings:`, set by the composition root)
) -> Board:
    start = to_tick(cfg, start)
    end = start + timedelta(hours=cfg.window_hours)
    blocks: list[FixedBlock] = []
    events: list[OptionalEvent] = []
    deadlines: list[Deadline] = []
    nearby: list[OptionalEvent] = []
    work_left = over.work_left if over else {}

    for item in sorted((i for i in items if i.when), key=lambda i: (i.when, i.external_id)):
        when = local(cfg, item.when)
        if item.kind == "assignment":
            if start < when <= start + timedelta(hours=cfg.deadline_horizon_hours):
                guess, how, word = estimate_of(cfg, item.title)
                mine = item.external_id in work_left
                left = float(work_left[item.external_id]) if mine else guess
                deadlines.append(Deadline(id=item.external_id, title=item.title, due=when, work_left=left,
                                          in_window=when <= end, risk=risk_of(cfg, left, start, when), url=item.url,
                                          estimate_from="yours" if mine else how, estimate_word=word, estimate_minutes=guess))
            continue
        if item.kind != "calendar_event":
            continue
        ignored = item.external_id.startswith(tuple(cfg.ignored_event_prefixes))
        until, place = _end_and_place(cfg, w, item, when)
        if until is None or until <= start or when >= end:
            continue
        name = w.name(place) if place else None
        if ignored:  # "Near you": never a block or a listed move; kept aside as a suggestion for the gap it falls in
            note = " · ".join(p.strip() for p in (item.summary or "").split("·")[2:] if p.strip()) or None
            nearby.append(OptionalEvent(id=item.external_id, title=item.title, start=when, end=until, place=place,
                                        place_name=name, url=item.url, note=note))
            continue
        # A listing (the campus feed by its id prefix, or anything the box's `listings:` config recognises by its link,
        # feed or connector) is offered as an optional event; it is never a fixed block, so a plan never walks to it
        # as a commitment and the tracker never gives it a leave-by time.
        if any(item.external_id.startswith(p) for p in cfg.optional_event_prefixes) or (listed is not None and listed(item)):
            events.append(OptionalEvent(id=item.external_id, title=item.title, start=when, end=until, place=place,
                                        place_name=name, url=item.url))
            continue
        title = item.title.lower()
        activity = "eat" if any(x in title for x in cfg.meal_words) else "study" if any(x in title for x in cfg.study_words) else "class"
        blocks.append(FixedBlock(id=item.external_id, title=item.title, start=max(when, start), end=min(until, end),
                                 place=place, place_name=name, activity=activity, url=item.url))

    nights = []
    for day in (start.date() - timedelta(days=1), start.date(), end.date()):
        bed = datetime.combine(day, cfg.sleep.bedtime, cfg.tz)
        wake = datetime.combine(day if cfg.sleep.bedtime < cfg.sleep.wake else day + timedelta(days=1), cfg.sleep.wake, cfg.tz)
        if bed < end and wake > start:
            nights.append((max(bed, start), min(wake, end)))
    if cfg.sleep.fixed:
        blocks += [FixedBlock(id=f"sleep:{a.date()}", title="Sleep", start=a, end=b, place=w.ref(cfg.home),
                              place_name=w.name(cfg.home), activity="sleep") for a, b in nights]

    for b in blocks:  # a study block works on the deadline of the same course
        if b.activity == "study":
            code = course_of(b.title)
            b.deadline = next((d.id for d in deadlines if code and code in d.title.replace(" ", "").upper()), None)

    board = Board(
        start=start, end=end, tick_minutes=cfg.tick_minutes, pack=cfg.pack,
        start_place=w.ref((over.start_place if over and over.start_place else None) or cfg.start_place),
        home=w.ref(cfg.home), initial=initial, blocks=blocks, deadlines=deadlines, events=events,
        friends=[f for f in friends or [] if f.start < end and f.end > start],
        sleep_window=max(nights, key=lambda n: n[1] - n[0]) if nights else None,
        suggestions=nearby, rules=rules_of(cfg, w),
    )
    return lay_out(cfg, w, board)


def lay_out(cfg: DayRunConfig, w: Walks, board: Board) -> Board:
    """Order the fixed blocks, cut the open slots between them and list each slot's moves.

    Called again after hard mode moves a block, so slots and routes always match the blocks.
    """
    blocks: list[FixedBlock] = []
    for b in sorted(board.blocks, key=lambda b: (b.start, b.id)):
        if blocks and b.start < blocks[-1].end:  # two things at once: the later one starts when the first ends
            b = b.model_copy(update={"start": blocks[-1].end})
        if b.start < b.end:
            blocks.append(b)

    slots: list[OpenSlot] = []
    cursor, place = board.start, board.start_place
    for nxt in [*blocks, None]:
        until = nxt.start if nxt else board.end
        if until > cursor:
            slots.append(OpenSlot(id=f"s{len(slots) + 1}", start=cursor, end=until, minutes=minutes_between(cursor, until),
                                  from_place=place, to_place=nxt.place if nxt else None, next_block=nxt.id if nxt else None))
        if nxt:
            cursor, place = nxt.end, nxt.place or place
    board = board.model_copy(update={"blocks": blocks, "slots": slots})
    for slot in slots:
        slot.moves = options(cfg, w, board, slot)
    board.suggestions = suggest(cfg, w, board)
    return board


def suggest(cfg: DayRunConfig, w: Walks, board: Board) -> list[OptionalEvent]:
    """Each "Near you" event with the open slot it falls in, the walk there and when to leave; one that fits no slot
    (it overlaps something fixed, or the walks leave no time) is left out."""
    out: list[OptionalEvent] = []
    for e in sorted(board.suggestions, key=lambda e: (e.start, e.id)):
        for slot in board.slots:
            if not (e.start < slot.end and e.end > slot.start):
                continue
            place = e.place or slot.from_place
            there, back = w.walk(slot.from_place, place)[0], w.walk(place, slot.to_place)[0]
            arrive = max(slot.start + timedelta(minutes=there), e.start)
            if minutes_between(arrive, min(e.end, slot.end - timedelta(minutes=back))) < cfg.min_move_minutes:
                continue
            out.append(e.model_copy(update={"slot": slot.id, "walk_minutes": there,
                                            "leave_by": max(slot.start, e.start - timedelta(minutes=there))}))
            break
    return out


def study_packs(cfg: DayRunConfig, w: Walks, board: Board) -> list[str]:
    return list(dict.fromkeys([board.pack, board.home.split(".")[0], w.ref(cfg.desk).split(".")[0]]))


def options(cfg: DayRunConfig, w: Walks, board: Board, slot: OpenSlot) -> list[MoveOption]:
    """The concrete moves a slot can hold, each with its walking minutes and whether it fits."""
    packs = study_packs(cfg, w, board)
    nxt = next((b for b in board.blocks if b.id == slot.next_block), None)
    with_friends = {f.place for f in board.friends if f.start < slot.end and f.end > slot.start}
    out: list[MoveOption] = []

    def option(kind: str, place: str, title: str, least: int, event: OptionalEvent | None = None) -> MoveOption:
        there, trip = w.walk(slot.from_place, place)
        back, _ = w.walk(place, slot.to_place)
        room = slot.minutes - there - back
        if event:
            arrive = max(slot.start + timedelta(minutes=there), event.start)
            room = minutes_between(arrive, min(event.end, slot.end - timedelta(minutes=back)))
        reason = None
        if place in board.closed:
            reason = f"{w.name(place)} is closed today"
        elif room < least:
            after = f" and on to the {hm(nxt.start)} {nxt.title}" if nxt and back else ""
            reason = f"the walk there{after} takes {there + back} minutes and leaves {max(room, 0)} of the {least} needed"
        friends = sorted({f.member for f in board.friends if f.place == place and f.start < slot.end and f.end > slot.start})
        return MoveOption(kind=kind, place=place, place_name=w.name(place), title=title, walk_minutes=there,
                          back_minutes=back, max_minutes=max(room, 0), feasible=reason is None, reason=reason,
                          friends=friends if kind == "study" else [], event=event.id if event else None,
                          route=trip.places if trip else [place])

    for kind, verb in (("study", f"{cfg.vocabulary.activity.capitalize()} at"), ("eat", "Eat at")):
        ranked = sorted(w.of_kind(cfg.kinds[kind], packs), key=lambda r: (w.walk(slot.from_place, r)[0], r))
        keep = [r for n, r in enumerate(ranked) if n < cfg.max_options or (kind == "study" and r in with_friends)]
        out += [option(kind, p, f"{verb} {w.name(p)}", cfg.min_move_minutes) for p in keep]
    out.append(option("sleep", board.home, "Sleep", cfg.sleep.min_block_minutes))
    out.append(option("rest", slot.from_place, f"Rest at {w.name(slot.from_place)}", cfg.min_move_minutes))
    if slot.from_place != board.home:
        out.append(option("rest", board.home, "Rest at home", cfg.min_move_minutes))
    for e in board.events:
        if e.start < slot.end and e.end > slot.start:
            out.append(option("event", e.place or slot.from_place, f"Go to: {e.title}", cfg.min_move_minutes, e))
    return out
