"""How Atlas says the day (D3): structured facts in, natural language out, in two registers.

    facts = {"moment": "now", "risk": "ok", "drift_min": 20, "next": {"title": "judging", "at": "15:30", ...}}
    text = render(facts, Words())          # Narration(spoken=..., display=...)
    problems = check(rewrite, facts, text.spoken)   # [] when a model's rewrite is faithful to the facts

`spoken` is for the ear: a few sentences, no lists, only the one or two clock times that matter, and numbers the
way people say them ("about six hours", "quarter past three"). `display` is for the screen: a short paragraph
that may keep exact times. Both come from the same `facts`, a small JSON-able dict the narrator
(contexts/day/application/narrator.py) builds from the day's events: local times as "HH:MM" strings, durations
under keys ending `_min`, hours under keys ending `_hours`, names already shortened.

Everything here is pure: no clock, no model, no I/O. The template engine (`render`) is always available and is
what the tests assert. The model engine only ever rewrites a template's `spoken` text, and `check` is the gate
its reply has to pass before anyone hears it:

- every clock time in the reply is one of the facts' times (exact, or rounded to five minutes the way
  `say_time` rounds it);
- every other number is one of the facts' numbers, or a spoken form of one (ninety minutes is an hour and a
  half; 2.2 hours is about two);
- every name (a capitalised word that does not start a sentence, or one that does and is not an ordinary
  sentence opener) appears in the facts or in the template's draft;
- no guarded word (an event, a day, a claim such as "late", "behind", "on track", "cancelled") that the facts
  and the draft do not use;
- one paragraph: no lists, no markdown, no line breaks, within the sentence and length bounds, and a draft
  that ends by asking still ends by asking.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo

from contracts.day import PlanMove

MOMENTS = ("morning", "now", "replan", "coach", "evening", "approval", "checkin", "suggest", "start")
RISKY = ("leave_now", "late", "infeasible")
# Worth saying aloud unasked. "infeasible" is not: the re-plan that follows it says the same thing and what to do.
URGENT_RISKS = ("leave_now", "late")
# Bounds for a spoken narration: (sentences, characters). "Morning." counts as a sentence.
BOUNDS = {"morning": (6, 520), "now": (4, 340), "replan": (5, 420), "coach": (4, 360), "evening": (5, 420), "approval": (3, 240),
          "checkin": (5, 540), "suggest": (4, 340), "start": (3, 260)}
# A check-in says what was measured ("looked strained", "resting heart rate is up") and never what it might mean.
# None of these words may appear in one: not in the template, and a model's rewrite that uses one is rejected.
DIAGNOSIS = frozenset("""
stress stressed stressful sick sickness ill illness unwell dehydrated dehydration anxious anxiety depressed depression
exhausted exhaustion fatigue fatigued tired burnout burned fever feverish unhealthy symptom symptoms diagnosis diagnose
condition disorder overworked overwhelmed upset angry sad afraid worried panic
""".split())
NAP_MAX = 180  # a chosen sleep block shorter than this is a nap, not the night
BREAK_MAX = 30  # a rest this short between two blocks of the main activity is a break inside them
LONG_REST = 60
SHORT_BLOCK = 30
LONG_BLOCK = 90


@dataclass(frozen=True)
class Words:
    """The profile's words (contexts/profiles Vocabulary): a student studies at a desk between classes."""

    activity: str = "study"
    workplace: str = "desk"
    event: str = "class"
    deadline: str = "assignment"


@dataclass(frozen=True)
class Narration:
    spoken: str
    display: str


# ---------- numbers, durations and times the way people say them ----------

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def say_number(n: int) -> str:
    """0 to 999 in words ("twenty-five", "a hundred and twenty"); anything else as digits."""
    n = int(n)
    if n < 0 or n > 999:
        return str(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (f"-{ONES[n % 10]}" if n % 10 else "")
    lead = "a hundred" if n // 100 == 1 else f"{ONES[n // 100]} hundred"
    return lead + (f" and {say_number(n % 100)}" if n % 100 else "")


def _round_to(n: float, step: int) -> int:
    return int((n + step / 2) // step * step)


def say_minutes(minutes: float) -> str:
    """A duration as said aloud: "seven minutes", "about ten minutes", "half an hour", "ninety minutes",
    "an hour and a quarter", "about two and a half hours". "about" whenever rounding changed it."""
    n = round(minutes)
    if n <= 0:
        return "no time"
    if n == 1:
        return "a minute"
    if n < 10:
        return f"{say_number(n)} minutes"
    if n < 58:
        r = _round_to(n, 5)
        words = "half an hour" if r == 30 else f"{say_number(r)} minutes"
    elif n < 128:
        r = _round_to(n, 15)
        words = {60: "an hour", 75: "an hour and a quarter", 90: "ninety minutes", 105: "an hour and three quarters",
                 120: "two hours"}[r]
    else:
        r = _round_to(n, 30)
        whole, half = divmod(r, 60)
        words = f"{say_number(whole)} and a half hours" if half else f"{say_number(whole)} hours"
    return words if r == n else f"about {words}"


def say_hours(hours: float) -> str:
    """Hours as said aloud, to the half hour: "about six hours", "about five and a half hours", "eight hours"."""
    if hours < 0.75:
        return say_minutes(hours * 60)
    r = round(hours * 2) / 2
    whole, half = int(r), r != int(r)
    if whole == 1 and not half:
        words = "an hour"
    elif whole == 1:
        words = "an hour and a half"
    else:
        words = f"{say_number(whole)} and a half hours" if half else f"{say_number(whole)} hours"
    return words if abs(r - hours) < 0.01 else f"about {words}"


def spoken_clock(hour: int, minute: int, down: bool = False) -> tuple[int, int]:
    """(hour, minute) rounded to five minutes: to the nearest, or down (a time to leave by must not move later)."""
    total = hour * 60 + minute
    total = total // 5 * 5 if down else _round_to(total, 5)
    return (total // 60) % 24, total % 60


def say_time(t: datetime | str, down: bool = False) -> str:
    """A clock time as said aloud: "half past three", "quarter to four", "ten to one", "nine o'clock", "noon"."""
    hour, minute = hm_of(t)
    hour, minute = spoken_clock(hour, minute, down)
    if minute == 0:
        return "noon" if hour == 12 else "midnight" if hour == 0 else f"{ONES[hour % 12 or 12]} o'clock"
    this, nxt = ONES[hour % 12 or 12], ONES[(hour + 1) % 12 or 12]
    if nxt == "twelve" and minute > 30:
        nxt = "noon" if (hour + 1) % 24 == 12 else "midnight"
    if minute == 15:
        return f"quarter past {this}"
    if minute == 30:
        return f"half past {this}"
    if minute == 45:
        return f"quarter to {nxt}"
    return f"{say_number(minute)} past {this}" if minute < 30 else f"{say_number(60 - minute)} to {nxt}"


def hm_of(t: datetime | str) -> tuple[int, int]:
    if isinstance(t, str):
        h, m = t.split(":")[:2]
        return int(h), int(m)
    return t.hour, t.minute


def hhmm(t: datetime, tz: tzinfo | None = None) -> str:
    """The form a time takes in `facts`: local "HH:MM"."""
    t = t.astimezone(tz) if tz and t.tzinfo else t
    return f"{t.hour:02d}:{t.minute:02d}"


def clock(t: datetime | str) -> str:
    """A clock time for the screen: "2:40 PM", "9 AM"."""
    hour, minute = hm_of(t)
    suffix = "PM" if hour >= 12 else "AM"
    return f"{hour % 12 or 12}{f':{minute:02d}' if minute else ''} {suffix}"


def say_day(day: date, today: date) -> str:
    gap = (day - today).days
    return {0: "today", 1: "tomorrow", -1: "yesterday"}.get(gap, WEEKDAYS[day.weekday()])


def part_of_day(t: datetime | str) -> str:
    hour = hm_of(t)[0]
    return "morning" if 4 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening" if 17 <= hour < 21 else "night"


def when_part(t: datetime | str) -> str:
    """"this morning", "this afternoon", "this evening", "tonight"."""
    part = part_of_day(t)
    return "tonight" if part == "night" else f"this {part}"


def say_count(n: int, noun: str, plural: str | None = None) -> str:
    return f"{say_number(n)} {noun if n == 1 else plural or noun + 's'}"


def say_share(percent: float) -> str:
    """A share of the plan the way a person puts it: "about a third", "nearly all", "about sixty percent"."""
    p = round(percent)
    for lo, hi, words in ((0, 5, "almost none"), (20, 29, "about a quarter"), (30, 38, "about a third"), (45, 55, "about half"),
                          (62, 70, "about two thirds"), (72, 79, "about three quarters"), (92, 99, "nearly all"), (100, 100, "all")):
        if lo <= p <= hi:
            return words
    return f"about {say_number(_round_to(p, 5))} percent"


# ---------- names ----------

TERM = re.compile(r"\b(?:Fall|Spring|Summer|Winter)\s?\d{4}\b", re.I)
PLAIN_PLACES = ("desk", "kitchen", "bedroom", "front door", "door", "table", "station", "living room", "office", "couch")


def short_title(title: str) -> str:
    """A calendar or Canvas title as a person would say it: "Knight Hacks: mentor check-in" -> "mentor check-in",
    "Project - Milestone1 Fall2026 [COP4533]" -> "Project Milestone 1"."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", title or "")
    t = TERM.sub(" ", t)
    head, sep, tail = t.partition(": ")
    if sep and tail.strip() and len(head.split()) <= 3:
        t = tail if len(tail.split()) >= 2 else f"{head} {tail}"
    t = re.sub(r"\s+-\s+", " ", t)
    t = re.sub(r"(?<=[a-z])(?=\d)", " ", t)  # Milestone1 -> Milestone 1
    return re.sub(r"\s+", " ", t).strip(" -:,.") or (title or "").strip()


def short_place(name: str | None) -> str | None:
    """"Engineering II (Knight Hacks)" -> "Engineering II"; "Desk" -> "the desk"."""
    if not name:
        return None
    n = re.sub(r"\s*\([^)]*\)", "", name).strip() or name.strip()
    return f"the {n.lower()}" if n.lower() in PLAIN_PLACES else n


def at_place(place: str) -> str:
    """"in the kitchen", "at the desk", "at Engineering II"."""
    return f"in {place}" if place in ("the kitchen", "the bedroom", "the living room", "the office") else f"at {place}"


def the(title: str) -> str:
    """"the mentor check-in", but "COP4533 study session" and "Judging" stay as they are."""
    return f"the {title}" if title[:1].islower() else title


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def lower_first(text: str) -> str:
    """Lower the first letter unless the word is a name or an acronym ("COP4533", "I")."""
    first = text.split(" ", 1)[0]
    if first in ("A", "An"):
        return text[:1].lower() + text[1:]
    return text if len(first) < 2 or not first[1:].islower() else text[:1].lower() + text[1:]


# ---------- the shape of a plan: a few beats, not every block ----------


def meal_name(t: datetime | str) -> str:
    hour, minute = hm_of(t)
    at = hour * 60 + minute
    if 4 * 60 <= at < 10 * 60 + 30:
        return "breakfast"
    if 10 * 60 + 30 <= at < 15 * 60:
        return "lunch"
    if 15 * 60 <= at < 17 * 60:
        return "an afternoon meal"
    return "dinner" if 17 * 60 <= at < 21 * 60 + 30 else "a late meal"


def shape(moves: list[PlanMove], now: datetime, tz: tzinfo, skip: str | None = None) -> dict:
    """The rest of the day from `now`, grouped: the chosen blocks before the first fixed event (consecutive blocks
    of the main activity collapsed into one beat, short rests between them absorbed), that event, the fixed
    events after it, tonight's sleep, and what the plan holds for the morning after.

    `skip` drops one fixed block by id (a re-plan's own subject, already spoken about)."""
    blocks = sorted((m for m in moves if m.end > now and m.id != skip), key=lambda m: m.start)
    nights = [m for m in blocks if m.kind == "sleep" and m.minutes >= NAP_MAX]
    night = max(nights, key=lambda m: m.minutes) if nights else None
    today = [m for m in blocks if night is None or m.start < night.start]
    after_night = [m for m in blocks if night is not None and m.start >= night.end]
    fixed = [m for m in today if m.fixed and m.kind != "sleep"]
    first = next((m for m in fixed if m.start > now), None)

    def beats(group: list[PlanMove]) -> list[dict]:
        out: list[dict] = []
        chosen = [m for m in group if not m.fixed]
        for i, m in enumerate(chosen):
            kind = "nap" if m.kind == "sleep" else "meal" if m.kind == "eat" else m.kind
            if kind == "rest" and m.minutes <= BREAK_MAX and out and out[-1]["kind"] == "study" and any(x.kind == "study" for x in chosen[i + 1:i + 2]):
                continue  # a breather between two blocks of the main activity
            same_meal = kind == "meal" and out and out[-1]["kind"] == "meal" and (
                out[-1]["count"] > 1 or meal_name(out[-1]["at"]) == meal_name(hhmm(m.start, tz)))
            if (kind == "study" and out and out[-1]["kind"] == "study") or same_meal:
                out[-1]["count"] += 1
                out[-1]["min"] += m.minutes
                out[-1].pop("place", None)
                continue
            beat = {"kind": kind, "count": 1, "min": m.minutes, "at": hhmm(m.start, tz)}
            if kind not in ("study", "rest", "nap", "meal"):
                beat["title"] = short_title(m.title)
            if m.place_name and kind in ("study", "meal"):
                beat["place"] = short_place(m.place_name)
            out.append(beat)
        return out

    def fixed_fact(m: PlanMove) -> dict:
        out = {"title": short_title(m.title), "at": hhmm(m.start, tz)}
        if m.place_name:
            out["place"] = short_place(m.place_name)
        return out

    before = beats([m for m in today if first is None or m.start < first.start])
    after = beats([m for m in today if first is not None and m.start >= first.start])
    out: dict = {"before": before}
    if first is not None:
        out["first"] = fixed_fact(first)
        later = [m for m in fixed if m.start > first.start]
        if later:
            out["later"] = [fixed_fact(m) for m in later]
            main = max(later, key=lambda m: (m.minutes, -m.start.timestamp()))
            out["main_later"] = fixed_fact(main)
    if after:
        out["after"] = after
    if night is not None:
        out["bed"] = hhmm(night.start, tz)
    morning = [b for b in beats(after_night) if b["kind"] == "study"]
    if morning:
        out["tomorrow"] = morning[0] if len(morning) == 1 else {**morning[0], "count": sum(b["count"] for b in morning), "min": sum(b["min"] for b in morning)}
    return out


def beat_words(b: dict, w: Words, exact: bool = False) -> str:
    """One beat in words: "two short study blocks", "a long rest", "lunch"; with `exact`, the screen's version."""
    kind, n, minutes = b["kind"], b.get("count", 1), b.get("min", 0)
    if kind == "study":
        each = minutes / max(n, 1)
        if exact:
            size = f"{round(each)}-minute " if n > 1 or minutes < 120 else ""
            return f"{'a' if n == 1 else say_number(n)} {size}{w.activity} block{'' if n == 1 else 's'}"
        size = "short " if each <= SHORT_BLOCK else "long " if each >= LONG_BLOCK else ""
        return f"a {size}{w.activity} block" if n == 1 else f"{say_number(n)} {size}{w.activity} blocks"
    if kind == "rest":
        return ("a rest" if exact else "a long rest") if minutes >= LONG_REST else "a break"
    if kind == "nap":
        return "a nap"
    if kind == "meal":
        return meal_name(b["at"]) if n == 1 else f"{say_number(n)} meals"
    return b.get("title") or kind


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _pick(options: tuple[str, ...], seed: str) -> str:
    """A stable choice among phrasings: the same facts always read the same."""
    return options[int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(options)]


def _spoken_beats(beats: list[dict], limit: int = 2) -> list[dict]:
    """The beats worth saying: the first, then the main activity if it is coming, else the first of another kind."""
    if len(beats) <= limit:
        return list(beats)
    rest = beats[1:]
    second = next((b for b in rest if b["kind"] == "study"), None) or next((b for b in rest if b["kind"] != beats[0]["kind"]), rest[0])
    return [beats[0], second]


def _sequence(beats: list[dict], w: Words, exact: bool = False, anchored: bool = False) -> str:
    """"a long rest first, then lunch"; one beat ahead of a fixed event (`anchored`) is just "a nap"."""
    words = [beat_words(b, w, exact) + (f" at {clock(b['at'])}" if exact else "") for b in beats]
    if not words:
        return ""
    if len(words) == 1:
        return words[0] if exact or anchored else f"{words[0]} first"
    return f"{words[0]}, then {_join(words[1:])}" if exact else f"{words[0]} first, then {_join(words[1:])}"


# ---------- the template engine ----------


def render(facts: dict, words: Words = Words()) -> Narration:
    """The template narration of `facts`: deterministic, always available, and the model's draft. Facts with an
    `also` (a second moment due at the same time, see contexts/day/domain/arbiter.py) come out as one sentence."""
    if facts.get("also"):
        return merge({k: v for k, v in facts.items() if k != "also"}, facts["also"], words)
    return {"morning": _morning, "now": _now, "replan": _replan, "coach": _coach, "evening": _evening,
            "approval": _approval, "checkin": _checkin, "suggest": _suggest, "start": _start}[facts["moment"]](facts, words)


def _sleep_clause(hours: float | None, gentle: bool) -> tuple[str, str]:
    if hours is None:
        return "", ""
    shown = f"{hours:.1f}".rstrip("0").rstrip(".")
    if hours < 5:
        tail = "so I've kept the start gentle" if gentle else "so pace yourself"
        return f"You only got {say_hours(hours)}, {tail}", f"You slept {shown} h, so {'the day starts gently' if gentle else 'pace yourself'}"
    if hours < 7:
        return f"You got {say_hours(hours)}, a little short", f"You slept {shown} h, a little short"
    return f"You got {say_hours(hours)} of sleep", f"You slept {shown} h"


def _walk_words(minutes: int) -> str:
    return f"{'an' if say_number(minutes).startswith(('eight', 'eleven')) else 'a'} {say_number(minutes)}-minute walk"


def _leave(first: dict) -> tuple[str, str]:
    if not first.get("leave_by") or first.get("walk_min", 0) < 3:
        return "", ""
    walk = first["walk_min"]
    return (f" Leave by {say_time(first['leave_by'], down=True)} for that; it's {_walk_words(walk)}.",
            f" {walk} min walk: leave by {clock(first['leave_by'])}.")


def _deadline(d: dict | None, w: Words) -> tuple[str, str]:
    if not d:
        return "", ""
    left, planned = d.get("left_min", 0), d.get("planned_min", 0)
    due_say = f"due {d['due_day']}"
    due_show = f"due {d['due_day'] if d['due_day'] in ('today', 'tomorrow') else d['due_day'][:3]} {clock(d['due'])}"
    if left <= 0:
        return f" {d['title']} is {due_say} and the work is done.", f" {d['title']} is {due_show}; nothing left to do on it."
    if planned >= left:
        return (f" {d['title']} is {due_say}, and this plan covers the {say_minutes(left).replace('about ', '')} it still needs.",
                f" {d['title']} is {due_show}, about {left} focused minutes left; this plan covers them.")
    if planned > 0:
        return (f" {d['title']} is {due_say} with {say_minutes(left)} of work left; this plan puts in {say_minutes(planned)}.",
                f" {d['title']} is {due_show}, about {left} focused minutes left; this plan puts in {planned}.")
    return (f" {d['title']} is {due_say} with {say_minutes(left)} of work left.",
            f" {d['title']} is {due_show}, about {left} focused minutes left.")


def _ask(facts: dict) -> tuple[str, str]:
    n = facts.get("blocks", 0)
    if not facts.get("ask") or not n:
        return "", ""
    if facts["ask"] == "plan":  # no calendar connector: approving only keeps the plan
        return " Want me to keep this as today's plan?", f" Approve to keep these {n} blocks as today's plan (no calendar connector on this box, nothing is written)."
    if facts.get("by") == "person":
        return " Want me to put your version on your calendar?", f" Approve to put your {n} blocks on your calendar."
    return " Want me to put this on your calendar?", f" Approve to put {n} block{'' if n == 1 else 's'} on your calendar."


def _plan_body(f: dict, w: Words) -> tuple[str, str, str]:
    """The shape of the day from here, for the ear and for the screen (no greeting, no ask): what is said, the
    leave-by sentence, and the screen's paragraph."""
    before, first = f.get("before") or [], f.get("first")
    said = _spoken_beats(before)
    seq, seq_x = _sequence(said, w, anchored=bool(first)), _sequence(before[:4], w, exact=True)
    main = f.get("main_later")
    if first:
        anchor = f"{the(first['title'])} at {say_time(first['at'])}"
        anchor_x = f"{the(first['title'])} at {clock(first['at'])}" + (f" ({first['place']})" if first.get("place") else "")
        spoken = f"{seq} before {anchor}" if seq else f"first up is {anchor}"
        shown = f"{cap(seq_x)} before {anchor_x}." if seq_x else f"First up is {anchor_x}."
        if main and len(said) < 2:  # room for one more beat: the biggest fixed thing later on
            spoken += f", and {the(main['title'])} {when_part(main['at'])}"
        later = f.get("later") or []
        if later:
            shown += " Later: " + ", ".join(f"{x['title']} {clock(x['at'])}" for x in later[:4]) + (f" and {len(later) - 4} more" if len(later) > 4 else "") + "."
    else:
        spoken = seq or "nothing is fixed and nothing needs adding"
        shown = f"{cap(seq_x)}." if seq_x else "Nothing is fixed today and nothing needs adding."
    leave, leave_x = _leave(first) if first else ("", "")
    if f.get("bed"):
        shown += f" Bed at {clock(f['bed'])}."
    if f.get("tomorrow"):
        shown += f" Tomorrow morning: {beat_words(f['tomorrow'], w, exact=True)}."
    return spoken, leave, _weave(shown, leave_x)


def _weave(shown: str, leave: str) -> str:
    """The leave-by goes right after the sentence that names the first fixed event."""
    if not leave:
        return shown
    m = re.search(r"(?<!\b[A-Z])\. ", shown)  # the first sentence end; not the stop after an initial ("John C. Hitt Library")
    if not m:
        return f"{shown.rstrip('.')}.{leave}"
    head, tail = shown[:m.start()], shown[m.end():]
    return f"{head}.{leave}{' ' + tail if tail else ''}"


def _morning(f: dict, w: Words) -> Narration:
    body, leave, shown = _plan_body(f, w)
    gentle = bool(f.get("before")) and f["before"][0]["kind"] in ("rest", "nap")
    sleep, sleep_x = _sleep_clause(f.get("sleep_hours"), gentle)
    greet = cap(f.get("greeting") or "morning") + "."
    lead = "Your version of the day" if f.get("by") == "person" else ""
    if sleep:
        opening = f"{sleep}: {body}." if not lead else f"{sleep}. {lead}: {body}."
    else:
        opening = f"{lead or _pick(('Here is the plan', 'Today looks like this'), f.get('day', ''))}: {body}."
    deadline, deadline_x = _deadline(f.get("deadline"), w)
    ask, ask_x = _ask(f)
    spoken = f"{greet} {opening}{leave}{deadline}{ask}"
    display = f"{sleep_x + '. ' if sleep_x else ''}{shown}{deadline_x}{ask_x}"
    return Narration(_tidy(spoken), _tidy(display))


def _next_line(nxt: dict | None, urgent: bool = False) -> tuple[str, str]:
    """The next fixed event and when to move: "The mentor check-in is at twenty to three; leave by twenty-five past two."""
    if not nxt:
        return "", ""
    where = f" at {nxt['place']}" if nxt.get("place") else ""
    spoken = f" {cap(the(nxt['title']))} is at {say_time(nxt['at'])}{where if urgent else ''}"
    shown = f" {cap(the(nxt['title']))} is at {clock(nxt['at'])}{where}"
    if nxt.get("leave_by") and nxt.get("walk_min", 0) >= 3:
        spoken += f"; leave by {say_time(nxt['leave_by'], down=True)}"
        shown += f" ({nxt['walk_min']} min walk, leave by {clock(nxt['leave_by'])})"
    return spoken + ".", shown + "."


def _now(f: dict, w: Words) -> Narration:
    risk, nxt, drift = f.get("risk", "ok"), f.get("next"), f.get("drift_min", 0)
    block, until, seen, should = f.get("block"), f.get("until"), f.get("seen_at"), f.get("should_be")
    if risk in RISKY and nxt:
        title, at_s, at_x = the(nxt["title"]), say_time(nxt["at"]), clock(nxt["at"])
        where = f" at {nxt['place']}" if nxt.get("place") else ""
        walk = nxt.get("walk_min", 0)
        walk_s = _walk_words(walk) if walk else "close by"
        if risk == "leave_now":
            soon = nxt.get("leave_in_min", 0)
            go = "leave now" if soon < 3 else f"leave in {say_minutes(soon).replace('about ', '')}"
            return Narration(f"Time to go. {cap(title)} starts at {at_s}{where}, {walk_s} from here; {go}.",
                             f"Leave now: {title} starts at {at_x}{where}" + (f", a {walk} min walk (leave by {clock(nxt['leave_by'])})." if walk and nxt.get("leave_by") else "."))
        if risk == "late":
            still = f" and you're still {at_place(seen)}" if seen else ""
            return Narration(f"{cap(title)} started at {at_s}{still}. Head{' to ' + nxt['place'] if nxt.get('place') else ' over'} now; it's {walk_s}.",
                             f"Late: {title} started at {at_x}{where}{f'; last seen {at_place(seen)}' if seen else ''}. It is a {walk} min walk.")
        origin = f" from {seen}" if seen else ""
        return Narration(f"You won't make {title} at {at_s}{origin}; it's {walk_s}. I'm working out a new plan from where you are.",
                         f"You will miss the start of {title} at {at_x}{origin} ({walk} min walk). A re-plan from here is coming.")
    nxt_s, nxt_x = _next_line(nxt)
    if block:
        doing = f"{block} until {say_time(until)}" if until else block
        doing_x = f"{block} until {clock(until)}" if until else block
        if drift >= 10:
            gap = say_minutes(drift)
            gap = gap if gap.startswith("about") else f"about {gap}"
            away = f": the plan has you {at_place(should)}" + (f" and I last saw you {at_place(seen)}" if seen and seen != should else "") if should else ""
            lead, lead_x = f"You're {gap} behind on {block}{away}.", f"About {drift} min behind on {doing_x}" + (f"; last seen {at_place(seen)}." if seen else ".")
        elif drift <= -10:
            lead, lead_x = f"You're ahead of the plan: {doing}.", f"Ahead of the plan: {doing_x}."
        else:
            lead, lead_x = f"You're on plan: {doing}.", f"On plan: {doing_x}."
    else:
        then = f.get("then")
        if then:
            lead = f"You're free until {then['title']} at {say_time(then['at'])}."
            lead_x = f"Free until {then['title']} at {clock(then['at'])}."
            if nxt and the(nxt["title"]) == then["title"] and nxt["at"] == then["at"]:  # one event, said once, with its leave-by
                nxt_s = f" Leave by {say_time(nxt['leave_by'], down=True)}." if nxt.get("leave_by") and nxt.get("walk_min", 0) >= 3 else ""
                nxt_x = f" {nxt['walk_min']} min walk, leave by {clock(nxt['leave_by'])}." if nxt_s else ""
        else:
            lead, lead_x = "Nothing is planned right now.", "Nothing is planned right now."
    return Narration(_tidy(lead + nxt_s), _tidy(lead_x + nxt_x))


def _replan(f: dict, w: Words) -> Narration:
    ev, seen = f.get("for"), f.get("seen_at")
    body, leave, shown = _plan_body({k: v for k, v in f.items() if k not in ("bed", "tomorrow")}, w)  # the change, not the whole night
    if f.get("added") or f.get("removed"):  # a yes (or a later no) to an optional event: the day with it worked in, or without it
        name = f.get("added") or f["removed"]
        lead = f"Okay, I've worked {name} in" if f.get("added") else f"Okay, {name} is out"
        lead_x = f"With {name} added" if f.get("added") else f"Without {name}"
        spoken, display = f"{lead}: {body}.{leave}", f"{lead_x}. {shown}"
    elif f.get("asked") or not ev:
        lead = f"Here's a fresh plan from {seen}" if seen else "Here's a fresh plan from here"
        lead_x = f"A new plan from {seen or 'here'}, as you asked"
        spoken, display = f"{lead}: {body}.{leave}", f"{lead_x}. {shown}"
    else:
        title, walk = the(ev["title"]), ev.get("walk_min", 0)
        origin = f" from {seen}" if seen else ""
        verb = "started" if ev.get("started") else "starts"
        walk_s = f"; it's {_walk_words(walk)}" if walk else ""
        if ev.get("started"):
            opening = f"{cap(title)} started at {say_time(ev['at'])}" + (f" and you're {_walk_words(walk)} away" if walk else "") + "."
        else:
            opening = f"You won't make the start of {title} at {say_time(ev['at'])}{origin}{walk_s}."
        go = (" Head there now and catch what's left. After that I'd do it this way" if ev.get("kept")
              else " It no longer fits, so I'd carry on from here")
        spoken = f"{opening}{go}: {body}."
        display = (f"{cap(title)} {verb} at {clock(ev['at'])}" + (f" at {ev['place']}" if ev.get("place") else "") + (f", a {walk} min walk{origin}" if walk else "")
                   + f". {'Go now and catch the rest' if ev.get('kept') else 'It no longer fits'}. Then: {lower_first(shown)}")
    n = f.get("blocks", 0)
    if f.get("ask") and n:
        spoken += " Want me to switch to that?"
        display += f" Approve to replace today's blocks with these {n}." if f["ask"] == "calendar" else f" Approve to make these {n} blocks today's plan."
    return Narration(_tidy(spoken), _tidy(display))


TRIGGER_SAID = {"chat": "right after the group chat lit up", "phone": "right after the phone came out",
                "door": "right after someone came in", "food": "right after a drink or snack showed up"}


def _coach(f: dict, w: Words) -> Narration:
    done, planned, n = f.get("desk_min", 0), f.get("planned_min", 0), f.get("breaks", 0)
    ratio = done / planned if planned else 0
    opener = "Nice work" if ratio >= 0.85 else "Decent block" if ratio >= 0.5 else "That one got away from you a little"
    spoken = f"{opener}: {say_number(done)} of {say_minutes(planned).replace('about ', '')} at the {w.workplace}."
    display = f"{cap(f.get('title') or w.activity + ' block')}: {done} of {planned} minutes at the {w.workplace}."
    long = f.get("long_break")
    if n == 0:
        late = f.get("late_start_min", 0)
        spoken += f" No breaks, though you sat down {say_minutes(late)} in." if late >= 5 else " No breaks."
        display += f" No breaks{f'; first sat down {late} min in' if late >= 5 else ''}."
    else:
        why = TRIGGER_SAID.get((long or {}).get("trigger", ""))
        count = say_count(n, "break")
        if n == 1:
            spoken += f" {cap(count)}, {why}." if why else f" {cap(count)}, {say_minutes(long['min']) if long else 'a short one'}."
        else:
            spoken += f" {cap(count)}, and the long one started {why}." if why else f" {cap(count)}, and I can't see what set off the long one."
        display += f" {n} break{'' if n == 1 else 's'}" + (f"; the longest was {long['min']} min{', ' + why if why else ''}." if long else ".")
    if f.get("longest_min"):
        display += f" Longest unbroken stretch: {f['longest_min']} min."
    if f.get("tip"):
        spoken += f" Next time, {lower_first(f['tip'])}."
        display += f" Try this: {lower_first(f['tip'])}."
    nxt = f.get("next")
    if nxt:
        spoken += f" Your next {w.activity} block is {say_minutes(nxt['min'])} at {say_time(nxt['at'])}."
        display += f" Next {w.activity} block: {nxt['min']} min at {clock(nxt['at'])}."
    elif f.get("block_min"):
        spoken += f" I'll plan {say_number(f['block_min'])}-minute blocks from here."
        display += f" Planning {f['block_min']}-minute blocks from here."
    return Narration(_tidy(spoken), _tidy(display))


def _evening(f: dict, w: Words) -> Narration:
    score, stars = f.get("score"), f.get("stars", 0)
    if score is None:
        lead, lead_x = "That's the day. I couldn't measure enough of it to say how it went", "Not enough of the day could be measured to score it"
    else:
        earned = f"earned {say_count(stars, 'star')}" if stars else "no stars this time"
        lead = f"That's the day. You followed {say_share(score)} of the plan and {earned}" if stars else f"That's the day. You followed {say_share(score)} of the plan, {earned}"
        lead_x = f"You followed {score}% of what could be measured: {stars} star{'' if stars == 1 else 's'}, {f.get('points', 0)} points (the plan would have earned {f.get('plan_stars', 0)})"
    win = f.get("win")
    if win and win.get("kind") == "stretch":
        win_s = f" The win: an unbroken {say_minutes(win['min']).replace('about ', '')} in {the(win['title'])}."
        win_x = f" Win: an unbroken {win['min']} min in {win['title']}."
    elif win and win.get("kind") == "kept":
        win_s, win_x = f" The win: {win['title']} happened as planned.", f" Win: {win['title']} happened as planned."
    elif win and win.get("kind") == "streak":
        win_s, win_x = f" That's {say_count(win['days'], 'day')} in a row with a star.", f" Streak: {win['days']} days with a star."
    else:
        win_s, win_x = " No clear win today, and that's okay.", ""
    slipped = f.get("slipped", 0)
    slip_x = f" {slipped} block{'' if slipped == 1 else 's'} slipped." if slipped else ""
    tomorrow = (f.get("tomorrow") or "").strip().rstrip(".")
    tomorrow = re.sub(r"\s+tomorrow\b", "", tomorrow)
    tom_s = f" For tomorrow: {lower_first(tomorrow)}." if tomorrow else ""
    # Waste Watch's clause for the day, when there is one (a fact, so a model's rewrite of it stays grounded)
    waste = (f.get("waste") or "").strip().rstrip(".")
    waste_s = " On sorting: " + re.sub(r"\b\d+\b", lambda m: say_number(int(m.group(0))), waste) + "." if waste else ""
    waste_x = f" Waste Watch: {waste}." if waste else ""
    return Narration(_tidy(f"{lead}.{win_s}{waste_s}{tom_s}"),
                     _tidy(f"{lead_x}.{win_x}{slip_x}{waste_x}{' Tomorrow: ' + lower_first(tomorrow) + '.' if tomorrow else ''}"))


def _approval(f: dict, w: Words) -> Narration:
    kind, n, day = f.get("kind"), f.get("count", 0), f.get("day", "today")
    if kind == "calendar":
        blocks = say_count(n, "block")
        if not f.get("writes", True):
            return Narration(f"I'd like to make these {blocks} {day}'s plan. Okay?" if n != 1 else f"I'd like to make this one block {day}'s plan. Okay?",
                             f"Keep {n} block{'' if n == 1 else 's'} as {day}'s plan. No calendar connector on this box, so nothing is written.")
        if f.get("replace"):
            return Narration(f"I'd like to swap {day}'s calendar blocks for {say_number(n)} new ones. Okay?",
                             f"Replace {day}'s Day Run blocks on your calendar with {n} new one{'' if n == 1 else 's'}.")
        return Narration(f"I'd like to add {blocks} to your calendar for {day}. Okay?",
                         f"Add {n} block{'' if n == 1 else 's'} to your calendar for {day}.")
    if kind == "league_stake":
        return Narration(f"I'd like to set this week's League stake: {lower_first(f['text'])}. Okay?", f"Set this week's League stake: {f['text']}.")
    if kind == "league_dare":
        days = say_count(f.get("days", 1), "day")
        who = f" with {f['who']}" if f.get("who") else ""
        return Narration(f"I'd like to start a League dare: {lower_first(f['title'])} for {days}{who}. Okay?",
                         f"Start a League dare: {f['title']} for {f.get('days', 1)} day{'' if f.get('days', 1) == 1 else 's'}{who}.")
    return Narration("I'd like to post the dare's result to the group chat. Okay?", "Post the League dare's result to the group chat.")


def _series(parts: list[str]) -> str:
    """"a", "a and b", "a, b, and c": clauses that are sentences of their own keep the comma before "and"."""
    return parts[0] if len(parts) == 1 else " and ".join(parts) if len(parts) == 2 else ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _checkin(f: dict, w: Words) -> Narration:
    """The check-in (contexts/day/domain/checkin.py): what was noticed, what happened the last times, what the desk
    camera saw, what to do. Each clause is there only when its fact is; with none of a sentence's facts the
    sentence is not there at all."""
    spoken: list[str] = []
    display: list[str] = []
    # 1. how the person is doing, from the watch and their face
    said, shown = [], []
    if f.get("sleep_hours") is not None:
        hours = float(f["sleep_hours"])
        said.append(f"you slept about {say_hours(hours).removeprefix('about ')}")
        shown.append(f"you slept about {hours:g} h")
    if f.get("resting_hr") == "up":
        said.append("your resting heart rate is up")
        shown.append("your resting heart rate is up")
    if f.get("strained_min"):
        said.append(f"you've looked strained for the last {say_minutes(f['strained_min']).removeprefix('about ')}")
        shown.append(f"you've looked strained for the last {f['strained_min']} minutes")
    if said:
        spoken.append(cap(_series(said)) + ".")
        display.append(cap(_series(shown)) + ".")
    # 2. the earlier days that looked like this one
    past = f.get("history")
    if past and past.get("matches"):
        outcome = (past.get("outcome") or "").strip().rstrip(".")
        by = past.get("by")
        heard = outcome.replace(clock(by), say_time(by)) if by else outcome
        spoken.append(f"The last {say_number(past['matches'])} times a day looked like this, {heard}.")
        display.append(f"The last {past['matches']} times a day looked like this, {outcome}.")
    # 3. at the desk: the sitting, and the drink nobody saw
    said, shown = [], []
    if f.get("sitting_min"):
        said.append(f"you've been at the {w.workplace} for {say_minutes(f['sitting_min'])} straight")
        shown.append(f"you've been at the {w.workplace} for {f['sitting_min']} minutes straight")
    drink = f.get("drink")
    if drink:
        if drink.get("since"):
            said.append(f"I haven't seen you pick up a drink since {say_time(drink['since'])}")
            shown.append(f"I haven't seen you pick up a drink since {clock(drink['since'])}")
        else:
            tail = "in that time" if said else "since you sat down"
            said.append(f"I haven't seen you pick up a drink {tail}")
            shown.append(f"I haven't seen you pick up a drink {tail}")
    if said:
        spoken.append(cap(", and ".join(said)) + ".")
        display.append(cap(", and ".join(shown)) + ".")
    # 4. what to do: plain nudges, and the one offer that goes through an approval card
    todo = []
    for s in f.get("suggest") or []:
        text = (s.get("text") or "").strip().rstrip(".")
        if text:
            todo.append(f"I can {lower_first(text)} if you say yes" if s.get("kind") == "move_block" else lower_first(text))
    if todo:
        line = cap(", and ".join(todo) if len(todo) == 2 and todo[1].startswith("I can ") else _series(todo)) + "."
        spoken.append(line)
        display.append(line)
    return Narration(_tidy(" ".join(spoken)), _tidy(" ".join(display)))


def _people(names: list[str]) -> str:
    """"Riley", "Riley and Sam", "Riley, Sam and Alex"."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


MEALS = ("breakfast", "brunch", "lunch", "dinner")


def _suggest(f: dict, w: Words) -> Narration:
    """An optional event coming up (contexts/day/domain/optional.py): who is going, where and when, the walk, whether
    it fits and what would move, and the offer. A yes only ever raises an approval card."""
    title, who, host = f["title"], list(f.get("who") or []), f.get("host")
    where = f" {at_place(f['place'])}" if f.get("place") else ""
    meal = next((m for m in MEALS if m in title.lower()), None)
    if host:
        opening = f"{host} is hosting {title}{where} at {say_time(f['at'])}" + (f", with {_people(who)}" if who else "")
    elif who:
        verb = "is" if len(who) == 1 else "are"
        opening = f"{_people(who)} {verb} {'having ' + meal if meal else 'going to ' + title}{where} at {say_time(f['at'])}"
    else:
        opening = f"{cap(the(title))} is{where} at {say_time(f['at'])}"
    people = ([host] if host else []) + who
    shown = title + (f" with {_people(people)}" if people else "") + f": {clock(f['at'])}" + (f" to {clock(f['until'])}" if f.get("until") else "")
    shown += f"{where}."
    walk = f.get("walk_min", 0)
    walk_s = f"it's {_walk_words(walk)}" if walk >= 3 else ""
    if walk >= 3:
        shown += f" {walk} min walk" + (f", leave by {clock(f['leave_by'])}." if f.get("leave_by") else ".")
    before, moves, early = f.get("before"), f.get("moves") or [], f.get("early_min", 0)
    if f.get("fits") == "nearly":
        if moves:
            fit_s, fit_x = f"{moves[0]} would have to move to make room", f"Nearly fits: {moves[0]} would move."
        else:
            fit_s = f"you'd leave {say_minutes(early)} early for {f.get('next') or 'what comes after'}"
            fit_x = f"Nearly fits: leave {early} min early" + (f" for {f['next']}." if f.get("next") else ".")
        second = f"{cap(walk_s)}, and {fit_s}." if walk_s else f"{cap(fit_s)}."
    else:
        fit_s = f"it fits before {before['title']}" if before else "it fits the rest of your day"
        fit_x = f"Fits before {before['title']} at {clock(before['at'])}." if before else "Fits your plan."
        second = f"{cap(walk_s)} and {fit_s}." if walk_s else f"{cap(fit_s)}."
    return Narration(_tidy(f"{opening}. {second} Want me to add it?"), _tidy(f"{shown} {fit_x} Add it?"))


def _start(f: dict, w: Words) -> Narration:
    """A planned block is starting (contexts/day/domain/reminders.py): what it is and for how long; the walk when
    there is one (said at the time to leave), or that you are there already; several short blocks said as one."""
    block, at, count, walk = f["block"], f["at"], f.get("count", 1), f.get("walk_min", 0)
    place = f.get("place")
    where = f" {at_place(place)}" if place else ""
    on = f" on {f['deadline']}" if f.get("deadline") else ""
    mins = f.get("min", 0)
    if count > 1:
        what = f"{say_number(count)} short {w.activity} blocks{on}"
        what_x = f"{count} short {w.activity} blocks{on}, {clock(at)} to {clock(f['until'])}"
        span = f"from {say_time(at)}, through to {say_time(f['until'])}"
    else:
        what, what_x, span = f"{block}{on}", f"{block}{on}, {clock(at)}" + (f", {mins} min" if mins else ""), ""
    length = f", {say_minutes(mins)}" if mins and count == 1 else ""
    if walk >= 3:  # said when it is time to leave
        spoken = f"Time for {what}{where}{', ' + span if span else ''}; it's {_walk_words(walk)}."
        shown = f"{cap(what_x)}{where}. {walk} min walk: leave now."
    elif f.get("there"):
        here = at_place(place) if place else "there"
        spoken = f"It's {say_time(at)}: {what}{' from here, through to ' + say_time(f['until']) if span else length}. You're {here} already."
        shown = f"{cap(what_x)}. You're {here} already."
    else:
        spoken = f"It's {say_time(at)}: time for {what}{' ' + span if span else length}{',' if length and where else ''}{where}."
        shown = f"{cap(what_x)}{where}."
    return Narration(_tidy(spoken), _tidy(shown))


def tail(f: dict, w: Words = Words()) -> str | None:
    """A second moment said as the end of another line's sentence ("and Riley's dinner is at quarter past five if
    you want it"), or None when the moment has no short form."""
    if f.get("moment") == "suggest":
        people = ([f["host"]] if f.get("host") else []) + list(f.get("who") or [])
        title = f["title"]
        meal = next((m for m in MEALS if m in title.lower()), None)
        subject = f"{people[0]}'s {meal}" if meal and len(people) == 1 else the(title) + (f" with {_people(people)}" if people else "")
        where = f" {at_place(f['place'])}" if f.get("place") else ""
        return f"and {subject} is{where} at {say_time(f['at'])} if you want it"
    if f.get("moment") == "start":
        on = f" on {f['deadline']}" if f.get("deadline") else ""
        what = f"{say_number(f['count'])} short {w.activity} blocks{on}" if f.get("count", 1) > 1 else f"{f['block']}{on}"
        return f"and {what} {'start' if f.get('count', 1) > 1 else 'starts'} at {say_time(f['at'])}"
    return None


def merge(first: dict, second: dict, w: Words = Words()) -> Narration:
    """Two moments due together as one line: the first said whole, the second as its closing clause (spoken), and
    both shown (display). Falls back to the first alone when the second has no short form or the first asks."""
    a, b = render(first, w), render(second, w)
    end = tail(second, w)
    if end is None or not a.spoken.endswith("."):
        return a
    return Narration(_tidy(f"{a.spoken[:-1]}; {end}."), _tidy(f"{a.display} {b.display}"))


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"([.!?])\.+", r"\1", text)


# ---------- the facts as the model and the cache see them ----------


def facts_json(facts: dict) -> str:
    return json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def facts_hash(facts: dict, tone: str = "") -> str:
    """The same facts (and the same tone) are the same moment: narrated once, never generated twice."""
    return hashlib.sha256((facts_json(facts) + "\n" + tone).encode()).hexdigest()[:16]


SYSTEM_PROMPT = """You are Atlas, a calm, capable assistant who knows this person's day and talks to them like someone who knows them well.
Rewrite the DRAFT so it flows when spoken aloud. Keep its meaning exactly.

Rules:
- Use only what is in FACTS and the DRAFT. Never add, change or guess a time, number, name, place, event or claim.
- Say every time and number the way the DRAFT words it ("twenty to three", "about two hours"). No digits for clock times.
- Keep the order: what matters first. Keep the closing question if the DRAFT ends with one.
- At most {sentences} short sentences and {words} words. Plain speech in the second person, with contractions.
- One paragraph. No lists, no markdown, no emoji, no quotation marks, no preamble.
- If you cannot make it better, return the DRAFT unchanged.
Reply with JSON only: {{"spoken": "<your rewrite>"}}"""


def system_prompt(moment: str, tone: str = "") -> str:
    sentences, chars = BOUNDS[moment]
    prompt = SYSTEM_PROMPT.format(sentences=sentences, words=chars // 6)
    return f"{prompt}\n{tone}" if tone else prompt


def user_prompt(facts: dict, draft: str) -> str:
    return f"FACTS: {facts_json(facts)}\nDRAFT: {draft}"


def reply_schema(moment: str) -> dict:
    """The reply as JSON with a bounded string: llama-server turns it into a grammar, so the model can neither
    think out loud nor run long (the runtime's own cap is 512 tokens; this one is about 80 to 120)."""
    return {"type": "object", "properties": {"spoken": {"type": "string", "maxLength": BOUNDS[moment][1]}},
            "required": ["spoken"], "additionalProperties": False}


def tone_line(words: Words, description: str = "", extra: str = "") -> str:
    """One line of tone per profile, from its vocabulary: the words this person uses for their own day."""
    line = (f"This person's words: \"{words.activity}\" for the main activity, \"{words.workplace}\" for where it happens, "
            f"\"{words.event}\" for a fixed block on the calendar, \"{words.deadline}\" for something due. Use them.")
    if description:
        line = f"Who you are talking to: {description.rstrip('.')}. {line}"
    return f"{line} {extra.strip()}" if extra and extra.strip() else line


def reply_text(content: str) -> str:
    """The rewrite out of a model's reply: the JSON's `spoken`, or the reply itself when it is plain text."""
    text = re.sub(r"<think>.*?(?:</think>|$)", " ", content or "", flags=re.S).strip()
    if text.startswith("{"):
        try:
            value = json.loads(text).get("spoken", "")
            return value.strip() if isinstance(value, str) else ""
        except (ValueError, AttributeError):
            return ""
    return text.strip().strip('"').strip()


# ---------- the grounding check ----------

_NUM_WORDS = {w: i for i, w in enumerate(ONES)} | {w: i * 10 for i, w in enumerate(TENS) if w}
_HOUR = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|noon|midnight|\d{1,2})"
_MIN_WORD = r"(?:quarter|half|five|ten|twenty[- ]five|twenty|\d{1,2})"
_AFTER_HOUR = r"(?:oh[- ]five|fifteen|thirty[- ]five|thirty|forty[- ]five|forty|fifty[- ]five|fifty|twenty[- ]five|twenty|ten|five)"
_NOT_A_UNIT = r"(?![- ]?(?:minute|hour|block|percent|point|star|day|meal|break|more|of\b|new\b))"
TIME_PATTERNS = (
    ("digital", re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?", re.I)),
    ("relative", re.compile(rf"\b(?:a )?({_MIN_WORD})(?: minutes?)? (past|after|to|till|before|of) ({_HOUR})\b{_NOT_A_UNIT}", re.I)),
    ("oclock", re.compile(rf"\b({_HOUR}) o['’]?clock\b", re.I)),
    ("ampm", re.compile(r"\b(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)(?![a-z])", re.I)),
    ("spelled", re.compile(rf"\b({_HOUR})[- ]({_AFTER_HOUR})\b{_NOT_A_UNIT}", re.I)),
    ("named", re.compile(r"\b(noon|midday|midnight)\b", re.I)),
    ("bare", re.compile(rf"\b(?:at|by|until|till|around) ({_HOUR})\b(?=\s*(?:[.,;!?]|$)|\s+(?:am|pm|tonight|today|tomorrow|this|sharp)\b)", re.I)),
)
GUARDED = frozenset("""
meeting class lecture exam quiz midterm gym workout practice appointment shift interview breakfast lunch dinner snack
coffee nap party lab seminar workshop deadline assignment homework presentation demo rehearsal flight doctor dentist
concert club handover chore judging submission checkin
behind ahead late early miss missed skip skipped cancel cancelled canceled moved rescheduled overdue urgent track done
finished complete completed star streak point percent win won lost failed due dropped drop
monday tuesday wednesday thursday friday saturday sunday today tomorrow tonight yesterday weekend
""".split())
OPENERS = frozenset("""
you your you're you've you'll the a an i i've i'd i'll i'm it it's that that's this then after before next now time
leave head want morning afternoon evening good nice solid decent tomorrow today tonight so and but there there's here
here's one two three four five six seven eight nine ten okay ok should shall with for in on at by from if when once
start keep try let let's still just about only no not great well right also plus since because as to do does did can
could would will we we're my nothing everything all most some each every how what where yesterday last first second
later earlier until till get go take grab eat rest sleep wrap finish be don't hold bring plan put tap say heads hey hi
looks sounds quick almost nearly once twenty thirty forty fifty sixty seventy eighty ninety half quarter
""".split())
NOT_NAMES = frozenset({"I", "I'm", "I've", "I'd", "I'll", "Atlas", "AM", "PM", "OK", "Okay"})


def _hour_value(word: str) -> int:
    word = word.lower()
    return 0 if word in ("noon", "midnight", "midday") else (_NUM_WORDS[word] if word in _NUM_WORDS else int(word)) % 12


def _minute_value(word: str) -> int:
    word = word.lower().replace("-", " ")
    if word == "quarter":
        return 15
    if word == "half":
        return 30
    if word.startswith("oh "):
        return _NUM_WORDS.get(word[3:], 0)
    total = sum(_NUM_WORDS[p] for p in word.split() if p in _NUM_WORDS)
    return total if any(p in _NUM_WORDS for p in word.split()) else int(word)


def times_in(text: str) -> tuple[list[tuple[int, int, str]], str]:
    """Every clock time said in `text` as (hour mod 12, minute, the words), and the text with them taken out."""
    found: list[tuple[int, int, str]] = []
    for name, pattern in TIME_PATTERNS:
        def take(m: re.Match, name: str = name) -> str:
            try:
                if name == "digital":
                    hour, minute = int(m.group(1)) % 12, int(m.group(2))
                elif name == "relative":
                    minute, hour = _minute_value(m.group(1)), _hour_value(m.group(3))
                    if m.group(2).lower() in ("to", "till", "before", "of"):
                        hour, minute = (hour - 1) % 12, 60 - minute
                elif name in ("oclock", "ampm", "bare"):
                    hour, minute = _hour_value(m.group(1)), 0
                elif name == "spelled":
                    hour, minute = _hour_value(m.group(1)), _minute_value(m.group(2))
                else:
                    hour, minute = 0, 0
            except (ValueError, KeyError):
                return m.group(0)
            found.append((hour % 12, minute % 60, m.group(0)))
            return " ~ "
        text = pattern.sub(take, text)
    return found, text


def numbers_in(text: str) -> list[tuple[float, str]]:
    """Every other number said in `text` (digits or words) as (value, the words). "one" on its own is a word,
    not a number; "an hour and a half" is 1.5 and "half an hour" is 0.5."""
    t = text.lower()
    found: list[tuple[float, str]] = []

    def take(value: float):
        def inner(m: re.Match) -> str:
            found.append((value(m) if callable(value) else value, m.group(0)))
            return " ~ "
        return inner

    t = re.sub(r"\ban? hour and a half\b", take(1.5), t)
    t = re.sub(r"\ban? hour and a quarter\b", take(1.25), t)
    t = re.sub(r"\ban? hour and three quarters\b", take(1.75), t)
    t = re.sub(r"\bhalf(?: an| of an)?[- ]hour\b", take(0.5), t)
    t = re.sub(r"\b(?:an?|one) hour\b", take(1.0), t)
    t = re.sub(r"\b(?:a|one) hundred(?: and)?\b", " 100plus ", t)
    tokens = re.findall(r"100plus|\d+(?:\.\d+)?|[a-z']+", t.replace("-", " "))
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        value: float | None = None
        words = [tok]
        if tok == "100plus":
            value = 100
        elif re.fullmatch(r"\d+(?:\.\d+)?", tok):
            value = float(tok)
        elif tok in _NUM_WORDS and tok != "one":
            value = _NUM_WORDS[tok]
        elif tok == "couple":
            value = 2
        if value is None:
            i += 1
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if (tok == "100plus" or (tok in _NUM_WORDS and _NUM_WORDS.get(tok, 0) >= 20)) and nxt in _NUM_WORDS and _NUM_WORDS[nxt] < (100 if tok == "100plus" else 10):
            value += _NUM_WORDS[nxt]
            words.append(nxt)
            i += 1
            nxt2 = tokens[i + 1] if i + 1 < len(tokens) else ""
            if tok == "100plus" and _NUM_WORDS.get(nxt, 0) >= 20 and nxt2 in _NUM_WORDS and _NUM_WORDS[nxt2] < 10:
                value += _NUM_WORDS[nxt2]
                words.append(nxt2)
                i += 1
        if tokens[i + 1:i + 4] == ["and", "a", "half"]:
            value += 0.5
            words += ["and a half"]
            i += 3
        found.append((value, " ".join(words).replace("100plus", "a hundred")))
        i += 1
    return found


def _walk(value, key: str = ""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, k)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v, key)
    else:
        yield key, value


def allowed(facts: dict, draft: str = "") -> tuple[set[tuple[int, int]], set[float], str]:
    """What a narration of `facts` may say: its clock times (as said: exact, or rounded to five minutes), its
    numbers with their spoken forms, and every word of its strings and of the template's draft."""
    times: set[tuple[int, int]] = set()
    numbers: set[float] = set()
    blob = [draft]
    for key, value in _walk(facts):
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, str):
            if re.fullmatch(r"\d{1,2}:\d{2}", value):
                h, m = hm_of(value)
                for hh, mm in ((h, m), spoken_clock(h, m), spoken_clock(h, m, down=True)):
                    times.add((hh % 12, mm))
                continue
            blob.append(value)
            numbers.update(float(x) for x in re.findall(r"\d+(?:\.\d+)?", value))
            continue
        n = float(value)
        numbers.add(n)
        if key.endswith("_min"):
            numbers.update({float(_round_to(n, 5)), n / 60, _round_to(n, 15) / 60, _round_to(n, 30) / 60, float(round(n / 60))})
        elif key.endswith("_hours"):
            numbers.update({float(round(n)), round(n * 2) / 2, float(int(n)), round(n * 60)})
        elif key in ("score", "percent"):
            numbers.add(float(_round_to(n, 5)))
    draft_times, rest = times_in(draft)
    times.update((h, m) for h, m, _ in draft_times)
    numbers.update(v for v, _ in numbers_in(rest))
    return times, numbers, " ".join(blob).lower()


def _stem(word: str) -> str:
    word = word.lower().replace("-", "")
    return word[:-1] if word.endswith("s") and len(word) > 3 and not word.endswith("ss") else word


def check(text: str, facts: dict, draft: str) -> list[str]:
    """Why `text` may not be said as a narration of `facts`; an empty list when it may."""
    problems: list[str] = []
    max_sentences, max_chars = BOUNDS[facts["moment"]]
    stripped = text.strip()
    if not stripped:
        return ["empty"]
    if len(stripped) > max_chars:
        problems.append(f"too long ({len(stripped)} characters, at most {max_chars})")
    if len(stripped) < min(40, len(draft) // 3):
        problems.append("too short to carry the facts")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", stripped) if s]
    if len(sentences) > max_sentences:
        problems.append(f"too many sentences ({len(sentences)}, at most {max_sentences})")
    if "\n" in stripped or re.search(r"(^|\s)(?:[-*•]\s|\d{1,2}[.)]\s)", stripped) or re.search(r"[*_#`|]|\[[^\]]*\]\(", stripped):
        problems.append("a list or markdown")
    if draft.rstrip().endswith("?") and not stripped.endswith("?"):
        problems.append("dropped the closing question")

    ok_times, ok_numbers, blob = allowed(facts, draft)
    said_times, rest = times_in(stripped)
    for hour, minute, words in said_times:
        if (hour, minute) not in ok_times:
            problems.append(f"a time that is not in the facts: {words.strip()!r}")
    for value, words in numbers_in(rest):
        if not any(abs(value - n) < 0.01 for n in ok_numbers):
            problems.append(f"a number that is not in the facts: {words!r}")

    if facts["moment"] == "checkin":  # what was measured, never what it might mean
        for word in sorted({w for w in re.findall(r"[a-z]+", stripped.lower()) if w in DIAGNOSIS}):
            problems.append(f"a diagnosis word: {word!r}")

    blob_words = {_stem(w) for w in re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", blob)}
    blob_words |= {_stem(p) for w in re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)+", blob) for p in w.split("-")}
    for i, sentence in enumerate(sentences):
        for j, m in enumerate(re.finditer(r"[A-Za-z0-9'’]+(?:-[A-Za-z0-9'’]+)*", sentence)):
            word = m.group(0).replace("’", "'")
            low = word.lower()
            known = _stem(low) in blob_words or all(_stem(p) in blob_words for p in low.split("-"))
            if _stem(low) in GUARDED and not known:
                problems.append(f"a claim or an event that is not in the facts: {word!r}")
            elif word[:1].isupper() and not known and word not in NOT_NAMES:
                if j > 0 or (low not in OPENERS and low.split("'")[0] not in OPENERS):
                    problems.append(f"a name that is not in the facts: {word!r}")
    return problems
