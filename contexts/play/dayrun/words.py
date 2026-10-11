"""Today's Plan in plain words: what is wrong with a plan, and where each number comes from.

Every sentence here is built from the fields the engine actually used (the board's deadlines and rules, the
simulation's checks, totals and routes), so the screen never has to guess. One or two short lines each, 12-hour
times, no ids, ratios or settings names. `contexts/play/tests/test_dayrun_words.py` holds each sentence to its fields.
"""

import re
from datetime import datetime

from contexts.play.dayrun.models import Board, Check, Deadline, Rejected, RouteTaken, SimResult

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBER = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def clock12(t: datetime) -> str:
    """11:15 AM, 2 PM, noon and midnight as a person writes them."""
    if t.minute == 0 and t.hour in (0, 12):
        return "noon" if t.hour == 12 else "midnight"
    return f"{t.hour % 12 or 12}{'' if t.minute == 0 else f':{t.minute:02d}'} {'AM' if t.hour < 12 else 'PM'}"


def day_clock(t: datetime) -> str:
    return f"{DAYS[t.weekday()]} {clock12(t)}"


def dur(minutes: float) -> str:
    """40 min, 1 h, 2 h 30 min: the way the Calendar screen writes a length."""
    total = round(minutes)
    if total < 60:
        return f"{total} min"
    h, m = divmod(total, 60)
    return f"{h} h {m} min" if m else f"{h} h"


def hours(h: float) -> str:
    """7 hours, 5.3 hours, 1 hour."""
    text = f"{h:.1f}".rstrip("0").rstrip(".")
    return f"{text} hour{'' if text == '1' else 's'}"


def span(hours_left: float) -> str:
    """How long until something: 5 h, 1 day 14 h."""
    total = round(hours_left)
    if total < 24:
        return f"{max(total, 0)} h"
    d, h = divmod(total, 24)
    return f"{d} day{'' if d == 1 else 's'}" + (f" {h} h" if h else "")


def short(title: str) -> str:
    """A title a sentence can carry: no course tag in brackets, no "Knight Hacks:" style prefix."""
    title = re.sub(r"\s*\[[^\]]+\]\s*$", "", title).strip()
    head, sep, tail = title.partition(": ")
    return tail.strip() if sep and len(head.split()) <= 3 and tail.strip() else title


def assignment(title: str) -> str:
    """A deadline's title as a person would say it: no course tag, course code, term or dash ("Project Milestone1")."""
    title = re.sub(r"\s*\[[^\]]+\]\s*$", "", title).strip()
    title = re.sub(r"^[A-Z]{3}\s?\d{4}[A-Z]?\s+(?=\S)", "", title)
    title = re.sub(r"\s+(Fall|Spring|Summer)\s?\d{4}$", "", title)
    return re.sub(r"\s+-\s+", " ", title).strip() or title


def count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# --- what needs fixing ---


def headline(problems: int) -> str:
    if problems <= 0:
        return "This plan works"
    return f"{NUMBER[problems] if problems < len(NUMBER) else problems} thing{'' if problems == 1 else 's'} to fix"


def problem_of_check(c: Check, board: Board) -> str:
    if c.id == "sleep":
        return f"{hours(c.value)} of sleep; you need {c.target:g}" if c.value > 0 else f"No sleep planned; you need {hours(c.target)}"
    if c.id == "meals":
        n = int(c.value)
        return f"{'No meals' if n == 0 else 'Only ' + count(n, 'meal')} today; you need {c.target:g}"
    d = next((d for d in board.deadlines if d.id == c.id), None)
    name = assignment(d.title) if d else "A deadline"
    left = dur(c.left_minutes or 0)
    if c.status == "missed":
        return f"{name} is due {day_clock(d.due) if d else 'in this plan'} with {left} left"
    return f"{name} is at risk: {left} left"


def problem_of_rejected(r: Rejected, board: Board, what: str) -> str:
    slot = next((s for s in board.slots if s.id == r.move.slot), None)
    nxt = next((b for b in board.blocks if slot and b.id == slot.next_block), None)
    if r.minutes and nxt:
        return f"You'd miss {short(nxt.title)} by {r.minutes} minutes"
    if r.minutes:
        return f"{what} runs {r.minutes} minutes past the end of the day"
    if "closed" in r.reason:
        return f"{what} can't happen: {r.reason[0].lower()}{r.reason[1:]}"
    return f"{what} doesn't fit where it is"


# --- where a number comes from ---


def came_from(source: str, name: str) -> str:
    """Which origin supplied `name` ("sleep", "meals") in Initial.source: "habits", "defaults", "scenario" or "request"."""
    if "(" not in source:
        return source
    for part in source.split("), "):
        origin, _, names = part.partition(" (")
        if name in names:
            return origin
    return "defaults"


def why_work(d: Deadline) -> str:
    """A deadline's work left: the board takes it from an estimate per kind of work, never from time already spent."""
    amount = dur(d.work_left)
    if d.estimate_from == "yours":
        guess = f" Atlas guessed {dur(d.estimate_minutes)}." if d.estimate_minutes is not None and d.estimate_minutes != d.work_left else ""
        return f"{amount} of work: your own estimate.{guess}"
    if d.estimate_from == "kind" and d.estimate_word:
        about = "an" if d.estimate_word[0] in "aeiou" else "a"
        return f"About {amount}: Atlas's starting guess for {about} {d.estimate_word}, not measured. Change it if you know better."
    return f"About {amount}: Atlas's starting guess for any assignment, not measured. Change it if you know better."


def why_risk(c: Check, board: Board) -> str:
    """Why a deadline is done, on track, at risk or missed: the work this plan does, what is left, and the time for it."""
    left = dur(c.left_minutes or 0)
    does = f"This plan does {dur(c.done_minutes)}" if c.done_minutes else "This plan puts no work into it"
    due = day_clock(c.due) if c.due else "its due time"
    if c.status == "done":
        return f"{does} and finishes it before {due}."
    if c.status == "missed":
        return f"It is due {due}, inside this plan, with {left} of work still left."
    per_day = dur(board.rules.capacity_minutes_per_day) if board.rules else "a few hours"
    if c.status == "on_track":
        return f"{does}. The {left} left fits before {due} at {per_day} a day."
    return f"{does}, leaving {left}. At {per_day} of real work a day, too tight for {due}."


def why_sleep(result: SimResult, board: Board) -> str:
    c = next(c for c in result.outcome.checks if c.id == "sleep")
    i, origin = board.initial, came_from(board.initial.source, "sleep")
    by = {"watch": "from your watch", "camera": "from the desk camera"}.get(i.sleep_from or "", "from your habits")
    last = {"habits": f"Last night: {hours(i.sleep_hours)}, {by}.",
            "scenario": f"Last night: {hours(i.sleep_hours)}, as this scripted day says.",
            "request": f"Last night: {hours(i.sleep_hours)}, the number you gave."}.get(
        origin, f"Last night wasn't measured, so Atlas assumed {hours(i.sleep_hours)}.")
    planned = f"{hours(c.value)} in this plan" if c.value > 0 else "No sleep in this plan"
    return f"{planned}; Atlas asks for {c.target:g}. {last}"


def why_meals(result: SimResult, board: Board) -> str:
    c = next(c for c in result.outcome.checks if c.id == "meals")
    t, origin = result.totals, came_from(board.initial.source, "meals")
    planned = t.meals - t.meals_before
    if t.meals_before:
        seen = {"habits": "already seen at your desk", "scenario": "already eaten on this scripted day"}.get(origin, "already eaten")
        return f"{count(t.meals, 'meal')} today: {t.meals_before} {seen}, {planned} in this plan. Atlas asks for {c.target:g}."
    has = count(planned, "meal") if planned else "No meals"
    return f"{has} in this plan, none seen before it starts. Atlas asks for {c.target:g}."


def why_walk(r: RouteTaken, board: Board) -> str:
    """A walk's minutes: the length along the mapped footpaths at the pack's walking pace, or the pack's set time."""
    minutes = max(1, round(r.minutes))
    if r.meters and board.rules and board.rules.walk_meters_per_minute:
        meters = round(r.meters / 10) * 10
        return f"{count(minutes, 'minute')}: about {meters} metres along the mapped footpaths, at an easy walking pace."
    return f"{count(minutes, 'minute')}: the set walking time between these two places."


def why_points(result: SimResult) -> str:
    p = result.totals.points_parts
    return (f"{result.totals.points} of 100: deadlines {p.get('deadlines', 0)} of 40, sleep {p.get('sleep', 0)} of 25, "
            f"meals {p.get('meals', 0)} of 15, energy left at the end {p.get('energy', 0)} of 20.")


def why_stars(result: SimResult) -> str:
    if not result.outcome.won:
        return "No stars yet: a plan earns its first once sleep, meals and every deadline hold."
    late = any(c.kind == "late" for c in result.changes)
    second = "you never run low on energy or arrive late" if result.stars >= 2 else (
        "you would arrive late somewhere" if late else f"your energy dips to {round(result.totals.energy_min)} of 100")
    if result.stars == 3:
        return "Three stars: it works, you never run low or late, and there's a full night with the deadlines comfortable."
    if result.stars == 2:
        return f"Two stars: it works and {second}. The third needs a longer night and comfortable deadlines."
    return f"One star: it works, but {second}."


def past(problem: str) -> str:
    """A problem as a few words about a day already lived: "missed judging and demos by 11 min"."""
    text = problem.split(";")[0].replace("You'd miss ", "the plan missed ").replace(" minutes", " min")
    return text[0].lower() + text[1:] if text else text
