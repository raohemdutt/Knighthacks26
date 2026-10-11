"""For the ear: how Atlas says a time, a day, a place, a title and a short run of things, wherever it answers.

OWNER'S RULE (Hemdutt): Atlas talks like a person about its work for you. It answers what was asked in one or two
short plain sentences, then at most a brief offer. It never reads records out: no lists, no "10:40 AM to 11:30 AM"
ranges, no ids, source tags, brackets, camera or zone ids.

    ear_time(t)                    "10:40", "2", "noon", "2:30 PM"
    ear_when(t, now)               "at 10:40", "at 8 tonight", "tomorrow at 2", "Tuesday night at 11:59 PM"
    ear_about(t, now)              "around 3", "yesterday evening around 7:40 PM"
    ear_title("COP4533 lecture")   "your COP 4533 lecture"   (names={"COP4533": "algorithms"}: "your algorithms lecture")
    ear_spot("desk", "cam-desk")   "on the desk"
    said_agenda / said_next / said_when / said_free / said_due / said_last_seen / said_visit / said_done

This sits next to the narrator's wording (narration.py) and reuses its names and numbers; it is not a second
narrator: the narrator says moments of the day unprompted, these say the answer to a question. Everything is pure:
no clock, no model, no I/O. The agent's fixed answer paths (contexts/agent/application/schedule.py, qa.py, talk.py)
are written with it, so the wording holds with no model at all and a small model cannot undo it.

A time with minutes after noon keeps its "PM" ("2:30 PM"): the voice reads a bare "2:30" as a 24-hour clock and
would say "2:30 AM" (contexts/voice/text.py and web/lib/speechText.ts). A morning time and a whole hour need none.
"""

import re
from datetime import datetime, timedelta

from contexts.day.domain.narration import WEEKDAYS, at_place, cap, say_number, short_place, short_title, the

MAX_SAID = 3  # things named in one answer unless everything was asked for
MAX_EVERYTHING = 8
OFFER = "What else can I do for you?"
OFFER_QUIET = "What can I do for you?"
MORE = "Want the rest?"  # ends a rundown that named only the first things; "yes" then gets all of it (talk.py)

COURSE_CODE = re.compile(r"\b([A-Z]{2,4}) ?(\d{3,4}[A-Z]?)\b")
CLASS_LIKE = re.compile(r"\b(?:lecture|class|lab|seminar|recitation|section|discussion|exam|midterm|final|quiz|study session|office hours"
                        r"|review session)\b", re.I)


# ---------- times and days ----------

def ear_time(t: datetime) -> str:
    """"10:40", "2", "noon", "midnight", "2:30 PM" (see the module note for the PM)."""
    h, m = t.hour, t.minute
    if m == 0:
        return "noon" if h == 12 else "midnight" if h == 0 else str(h % 12 or 12)
    return f"{h % 12 or 12}:{m:02d}" + (" PM" if h >= 12 else "")


def _clock(t: datetime, today: bool, tonight: bool = True) -> str:
    """The time with the one word that settles which half of the day, when the number alone would not."""
    said = ear_time(t)
    if said in ("noon", "midnight") or said.endswith("PM"):
        return said
    if t.hour < 6:
        return f"{said} AM"
    if t.hour >= 18 and t.minute == 0:
        return f"{said} tonight" if today and tonight else said if today else f"{said} PM"
    return said


def ear_day(t: datetime, now: datetime) -> str:
    """"" for today, "tomorrow", "yesterday", else the weekday."""
    gap = (t.date() - now.date()).days
    return {0: "", 1: "tomorrow", -1: "yesterday"}.get(gap, WEEKDAYS[t.weekday()])


def ear_when(t: datetime, now: datetime, tonight: bool = True) -> str:
    """When something is: "at 10:40", "at 8 tonight", "tomorrow at 2", "Tuesday at 11:59 PM", "Tuesday night at 10"."""
    day = ear_day(t, now)
    if not day:
        return f"at {_clock(t, True, tonight)}"
    if day != "yesterday" and t.hour >= 21 and t.minute == 0:
        return f"{day} night at {ear_time(t)}"
    return f"{day} at {_clock(t, False)}"


def ear_about(t: datetime, now: datetime) -> str:
    """When something was, roughly: "around 3", "around 7:40 PM", "yesterday around 7:40 PM", "yesterday evening around 7"."""
    day = ear_day(t, now)
    rounded = t.replace(second=0, microsecond=0) + timedelta(minutes=round(t.minute / 5) * 5 - t.minute)
    said = ear_time(rounded)
    if not day:
        return f"around {said}" if said.endswith("PM") or rounded.hour >= 6 or said == "midnight" else f"around {said} AM"
    if said.endswith("PM"):
        return f"{day} around {said}"
    part = "morning" if t.hour < 12 else "afternoon" if t.hour < 17 else "evening" if t.hour < 21 else "night"
    return f"{'last night' if (day, part) == ('yesterday', 'night') else f'{day} {part}'} around {said}"


def span_words(span: str) -> str:
    return {"day": "today", "week": "this week", "soon": "in the next few days"}.get(span, span)


# ---------- names and places ----------

def ear_title(title: str, names: dict | None = None, article: bool = True) -> str:
    """A calendar title as it is said: no brackets or tags, a course code as its spoken name when `names` has one
    ("COP4533" -> "algorithms", else "COP 4533"), and "your" or "the" in front where a person would put one."""
    # what is due keeps its whole name ("HW 4: linked lists"); an event loses its organiser ("Knight Hacks: dinner" -> "dinner")
    t = short_title(title) if article else re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", title or "")).strip(" -:,.") or title
    named = False

    def code(m: re.Match) -> str:
        nonlocal named
        name = (names or {}).get(m[1] + m[2]) or (names or {}).get(f"{m[1]} {m[2]}")
        named = named or bool(name)
        return name or f"{m[1]} {m[2]}"

    coded = COURSE_CODE.search(t) is not None
    t = COURSE_CODE.sub(code, t)
    if not article:
        return t
    if CLASS_LIKE.search(t) or coded:
        if t[1:2].islower() and sum(w[:1].isupper() for w in t.split()) == 1:  # "Study session", not "Knight Hacks review session"
            t = t[:1].lower() + t[1:]
        return f"your {t}"
    return the(t)


def ear_spot(zone: str | None, camera: str | None = None, position: str | None = None) -> str:
    """Where a thing was, without the zone or camera id: "on the desk", "by the door", "on the desk, next to the mug"."""
    name = (zone or (camera or "").removeprefix("cam-") or "").replace("_", " ").replace("-", " ").strip().lower()
    if not name or re.fullmatch(r"(?:cam)?\d+", name):
        spot = "in view of the camera"
    elif name in ("desk", "table", "counter", "shelf", "floor", "couch", "bed", "porch"):
        spot = f"on the {name}"
    elif name in ("door", "front door", "window", "sink", "stove", "bins", "recycling", "trash"):
        spot = f"by the {name}"
    else:
        spot = f"in the {name}"
    return f"{spot}, {position.strip().rstrip('.')}" if position and position.strip() else spot


def ear_place(place: str | None) -> str:
    """"at CSE A101", "in the kitchen"; "" when there is none."""
    short = short_place(place)
    return at_place(short) if short else ""


def series(parts: list[str]) -> str:
    """"a", "a, then b", "a, b, then c": things in the order they happen."""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f", then {parts[-1]}"


# ---------- the calendar, said like a person ----------
# An entry is {"title": str, "start": datetime, "end": datetime | None, "place": str | None}; times are local.

def _named(e: dict, now: datetime, names: dict | None, tonight: bool = True, article: bool = True) -> str:
    return f"{ear_title(e['title'], names, article)} {ear_when(e['start'], now, tonight)}"


def said_agenda(entries: list[dict], now: datetime, span: str = "today", *, names: dict | None = None, everything: bool = False,
                yes_no: str = "", earlier: bool = False, listed: bool = False) -> str:
    """What is on in a stretch of the day or week. Up to three things are simply said in order; with more, how many
    there are, the first two, and an offer of the rest. `yes_no` is how the question was put: "have" (do I have
    anything ...?) or "free" (am I free ...?). `listed`: these are things happening nearby (campus events a connector
    lists), not the person's own commitments, so it never says they "have" them."""
    words = span_words(span)
    if not entries:
        nothing = f"nothing {'else ' if earlier else ''}on your calendar {words}"
        return f"Yes, {nothing}." if yes_no == "free" else f"No, {nothing}." if yes_no == "have" else f"{cap(nothing)}."
    tonight = span != "tonight"
    lead = "" if listed else {"free": "No, ", "have": "Yes, "}.get(yes_no, "")
    if len(entries) > MAX_SAID and not everything:
        count = f"there are {say_number(len(entries))} things on campus {words}" if listed else f"you've got {say_number(len(entries))} things {words}"
        first, second = (_named(e, now, names, tonight) for e in entries[:2])
        return f"{cap(lead + count)}. First is {first}, then {second}. {MORE}"
    said = series([_named(e, now, names, tonight) for e in entries[:MAX_EVERYTHING]])
    left = len(entries) - MAX_EVERYTHING
    tail = f" {cap(say_number(left))} more after that." if left > 0 else ""
    if listed:
        return f"On campus {words} there's {said}.{tail}"
    if not lead and span not in ("today", "week", "soon", "day", "this week"):
        lead = f"{cap(words)} "
    if len(entries) == 1 and not yes_no:
        return f"{cap(lead)}you've got {said}, and that's it {'for today' if words == 'today' else words}." if lead else \
            f"You've got {said}, and that's it {'for today' if words == 'today' else words}."
    return f"{lead}you've got {said}.{tail}" if lead else f"You've got {said}.{tail}"


def said_next(entry: dict | None, now: datetime, *, names: dict | None = None, where: bool = False, ends: bool = False) -> str:
    """ "What's next?": one thing."""
    if entry is None:
        return "Nothing else on your calendar today."
    lead = "Next up is" if not ear_day(entry["start"], now) else "Nothing else today. Next up is"
    return f"{lead} {_named(entry, now, names)}.{_extra(entry, where, ends)}"


def said_when(entry: dict | None, now: datetime, thing: str = "", *, names: dict | None = None, where: bool = False,
              ends: bool = False) -> str:
    """ "When is my next class?": the one thing, as a sentence about it."""
    if entry is None:
        return f"I don't see {('an ' if thing[:1] in 'aeiou' else 'a ') + thing if thing else 'that'} on your calendar."
    return f"{cap(ear_title(entry['title'], names))} is {ear_when(entry['start'], now)}.{_extra(entry, where, ends)}"


def _extra(entry: dict, where: bool, ends: bool) -> str:
    out = ""
    if where and entry.get("place"):
        out += f" It's {ear_place(entry['place'])}."
    if ends and entry.get("end"):
        out += f" It runs until {ear_time(entry['end'])}."
    return out


def said_free(at: datetime, clash: dict | None, now: datetime, *, names: dict | None = None, busy: bool = False) -> str:
    """ "Am I free at 3?" (or, with `busy`, "am I busy at 3?"): yes or no, and the one thing in the way."""
    part = "this morning" if at.hour < 12 else "this afternoon" if at.hour < 17 else "this evening" if at.hour < 21 else "tonight"
    when = "right now" if abs(at - now) < timedelta(minutes=1) else f"at {ear_time(at)}" + ("" if ear_time(at) in ("noon", "midnight") else f" {part}")
    if clash is None:
        return f"{'No' if busy else 'Yes'}, you're free {when}."
    until = f", until {ear_time(clash['end'])}" if clash.get("end") else ""
    return f"{'Yes' if busy else 'No'}, you've got {ear_title(clash['title'], names)} then{until}."


def said_due(entries: list[dict], now: datetime, span: str = "this week", *, names: dict | None = None, everything: bool = False) -> str:
    """What is due: "You've got HW 4 due Tuesday night at 11:59 PM, then Exam 1 Thursday at 10." """
    if not entries:
        return f"Nothing due {span_words(span)}."
    shown = entries[:MAX_EVERYTHING if everything else MAX_SAID]
    first, rest = shown[0], shown[1:]
    parts = [f"{ear_title(first['title'], names, article=False)} due {_due_when(first, now)}"]
    parts += [f"{ear_title(e['title'], names, article=False)} {_due_when(e, now)}" for e in rest]
    left = len(entries) - len(shown)
    return f"You've got {series(parts)}." + (f" {cap(say_number(left))} more after that." if left else "")


def _due_when(e: dict, now: datetime) -> str:
    return ear_when(e["start"], now) if ear_day(e["start"], now) else f"today at {_clock(e['start'], True, tonight=False)}"


# ---------- what the cameras know, said like a person ----------

def said_last_seen(label: str, when: datetime, now: datetime, zone: str | None, camera: str | None = None, position: str | None = None) -> str:
    """ "Your keys were last on the desk, next to the coffee mug, around 3." """
    label = label.strip().lower()
    verb = "were" if label.endswith("s") and not label.endswith("ss") else "was"
    return f"Your {label} {verb} last {ear_spot(zone, camera, position)}, {ear_about(when, now)}."


def said_visit(start: datetime, now: datetime, moments: list[str], more: int = 0) -> str:
    """The latest visit: when it started, then what happened in order, as sentences."""
    told = [m.strip().rstrip(".") + "." for m in moments if m and m.strip()]
    tail = f" {cap(say_number(more))} more {'moment' if more == 1 else 'moments'} after that." if more > 0 else ""
    return f"The last visit started {ear_about(start, now)}. " + " ".join(told) + tail


# ---------- what Atlas did for you ----------

def said_done(done: list[str], greeting: bool = False) -> str:
    """ "How's it going?": a mood word, one clause on what Atlas has done for the person so far (at most two things,
    each a past participle: "planned your day"), then the offer. A bare greeting gets the offer alone."""
    if greeting:
        return f"Hey. {OFFER_QUIET}"
    done = [d.strip() for d in done if d and d.strip()][:2]
    if not done:
        return f"Good, all quiet so far. {OFFER_QUIET}"
    return f"Good. I've {' and '.join(done)} so far. {OFFER}"
