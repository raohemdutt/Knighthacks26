"""For the ear: how Atlas says a time, a day, a place, a title and a short run of things, wherever it answers.

OWNER'S RULE (Hemdutt): Atlas talks like a person about its work for you. It answers what was asked in one or two
short plain sentences, then at most a brief offer. It never reads records out: no lists, no "10:40 AM to 11:30 AM"
ranges, no ids, source tags, brackets, camera or zone ids.

    ear_time(t)                    "10:40", "2", "noon", "2:30 PM"
    ear_when(t, now)               "at 10:40", "at 8 tonight", "tomorrow at 2", "Tuesday night at 11:59 PM"
    ear_about(t, now)              "around 3", "yesterday evening around 7:40 PM"
    ear_title("COP4533 lecture")   "your COP 4533 lecture"   (names={"COP4533": "algorithms"}: "your algorithms lecture")
    ear_spot("desk", "cam-desk")   "on the desk"
    said_agenda / said_rundown / said_next / said_when / said_free / said_due / said_last_seen / said_visit / said_done
    day_fact_lines(...)            the same day, as the FACTS a model is given for any other question about it

LENGTH FITS THE QUESTION (the owner's correction). The "about three things, then 'Want the rest?'" cap is only for
casual or narrow questions ("what's on today?", "anything today?", "what's on my calendar", "do I have anything
this afternoon": said_agenda, said_rundown). A request to walk through the day ("what does my day look like", "run
me through my day", "brief me", "what do I have today") is the WHOLE day (said_walkthrough): every own event and
plan block by part of the day, when to leave, what is due, the campus things as one closing clause, then an offer
about the plan. "What's next?" is one thing. An own commitment is never silently dropped.

THE CONTRACT (the owner, on why nothing here is a canned answer): the facts are always fetched live, every time.
The sentences built here are only a FAST LANE for the phrasings the agent's patterns recognise. Any other way of
asking about the day goes to the model, which is given the same live facts in this same wording (`day_fact_lines`),
so Atlas can answer anything about the day from real data. Never a string that ignores the data.

This sits next to the narrator's wording (narration.py) and reuses its names and numbers; it is not a second
narrator: the narrator says moments of the day unprompted, these say the answer to a question. Everything is pure:
no clock, no model, no I/O. The agent's fixed answer paths (contexts/agent/application/schedule.py, qa.py, talk.py)
are written with it, so the wording holds with no model at all and a small model cannot undo it.

A time with minutes after noon keeps its "PM" ("2:30 PM"), so a bare "2:30" is always the morning's. The voice says
a bare "10:40" as written (contexts/voice/text.py and web/lib/speechText.ts): a morning time and a whole hour need
no "AM".
"""

import re
from datetime import datetime, timedelta

from contexts.day.domain.narration import WEEKDAYS, at_place, cap, say_number, short_place, short_title, the

MAX_SAID = 3  # things named in one answer unless everything was asked for
MAX_EVERYTHING = 8
OFFER = "What else can I do for you?"
OFFER_QUIET = "What can I do for you?"
PLAN_OFFER = "Want me to plan your day?"  # ends a rundown of a day with no plan; "yes" starts the Day loop's plan (talk.py)
PLAN_ELSEWHERE = "I can plan it with you on Today's Plan."  # the same offer on a box whose conversation cannot start a plan
PLAN_STARTED = "Okay, I've drafted a plan for today. Have a look on Today's Plan; nothing goes on your calendar until you approve it."
PLAN_CHECK = "Does that look right, or should I change something?"  # ends a walk-through of a day that has a plan
MAX_WALKED = 12  # a walk-through names everything up to this many things; past it the plan's blocks are counted, not named
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
    t = short_title(title) if article else short_title((title or "").replace(": ", " \x00 ")).replace(" \x00 ", ": ")
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
    """One thing and when: "your COP 4533 lecture at 10:40". A block of the plan is already in the narrator's words
    ("a study block at 9", "lunch at noon"); what is due is "HW 4, due at 5"."""
    if e.get("block"):
        return f"{e['title']} {ear_when(e['start'], now, tonight)}"
    if e.get("due"):
        return f"{ear_title(e['title'], names, article=False)}, due {_due_when(e, now)}"
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


# ---------- the whole day: the calendar, the plan, what is due, and what is on nearby ----------
# `things` are the person's own, in time order from now: calendar entries, and blocks of the agreed plan
# ({"title": "a study block", "start": ..., "block": True}, worded by the narrator). `dues` are what is due today or
# tomorrow. `listed` are campus or optional listings: never theirs, only offered. `leave` is {"title", "by"}: when to
# set off for the next thing that needs travel.

def _campus_count(n: int) -> str:
    return "there's one thing on campus if you want it" if n == 1 else f"there are {say_number(n)} things on campus if you want them"


def _due_clause(dues: list[dict], now: datetime, names: dict | None, everything: bool) -> str:
    shown = dues[:MAX_EVERYTHING] if everything else dues[:1]
    if not shown:
        return ""
    first = f"{cap(ear_title(shown[0]['title'], names, article=False))} is due {_due_when(shown[0], now)}"
    return series([first] + [f"{ear_title(e['title'], names, article=False)} {_due_when(e, now)}" for e in shown[1:]])


def said_rundown(things: list[dict], now: datetime, *, dues: list[dict] = (), listed: list[dict] = (), leave: dict | None = None,
                 names: dict | None = None, everything: bool = False, planned: bool = False, earlier: bool = False) -> str:
    """ "What does my day look like?": the whole day in two or three sentences. At most three things are named (the
    first things in order, and the first thing due), then when to leave for the next one, one closing clause for what
    is on nearby, and "Want the rest?" when there is more. With nothing of their own and no plan (`planned`): one
    line saying so and an offer to plan the day. `everything`: all of it, the listings too. `earlier`: something of
    theirs was on earlier today, so an empty rest of the day is "nothing else"."""
    things, dues, listed = list(things), list(dues), list(listed)
    out: list[str] = []
    named: list[dict] = []
    if not things:
        out.append("Nothing else of yours today." if earlier or planned else "Nothing of yours is on today.")
    else:
        room = MAX_EVERYTHING if everything else MAX_SAID - (1 if dues else 0)
        if len(things) <= room:
            named = things
            out.append(f"You've got {series([_named(e, now, names) for e in named])}.")
        elif everything:
            named = things[:room]
            out.append(f"You've got {series([_named(e, now, names) for e in named])}. {cap(say_number(len(things) - room))} more after that.")
        else:
            named = things[:2]
            out.append(f"You've got {say_number(len(things))} things today, starting with {_named(named[0], now, names)}, "
                       f"then {_named(named[1], now, names)}.")
    if leave and any(e["title"] == leave["title"] for e in named):
        last = not everything and named[-1]["title"] == leave["title"]  # just named: "for that one", not its title again
        out.append(f"Leave by {_clock(leave['by'], True, tonight=False)} for {'that one' if last else ear_title(leave['title'], names)}.")
    due = _due_clause(dues, now, names, everything)
    if listed and everything:
        out += [f"{due}."] if due else []
        out.append(f"On campus there's {series([_named(e, now, names) for e in listed[:MAX_EVERYTHING]])}.")
    elif listed and due:
        out.append(f"{due}, and {_campus_count(len(listed))}.")
    elif listed:
        out.append(f"Also, {_campus_count(len(listed))}." if things else f"{cap(_campus_count(len(listed)))}.")
    elif due:
        out.append(f"{due}.")
    if not everything and (len(things) > len(named) or len(dues) > 1):
        out.append(MORE)
    elif not things and not planned and not earlier:
        out.append(PLAN_OFFER)
    return " ".join(out)


def _part(t: datetime) -> str:
    return "morning" if t.hour < 12 else "afternoon" if t.hour < 17 else "tonight"


def said_walkthrough(things: list[dict], now: datetime, *, dues: list[dict] = (), listed: list[dict] = (), leave: dict | None = None,
                     names: dict | None = None, planned: bool = False, earlier: bool = False) -> str:
    """ "Run me through my day": the WHOLE day, in flowing sentences by part of the day. Every thing of the person's
    is said (events, the plan's blocks), then when to leave, what is due today or tomorrow, the campus things as one
    closing clause (how many, and two of them at most: they are not theirs), and an offer about the plan. No cap and
    no "Want the rest?". Only an absurd day (more than `MAX_WALKED` things) is shortened, and then only by counting
    the plan's blocks instead of naming them: an own commitment is never dropped."""
    things, dues, listed = list(things), list(dues), list(listed)
    out: list[str] = []
    if not things:
        out.append("Nothing else of yours today." if earlier or planned else "Nothing of yours is on today.")
    else:
        if len(things) >= 6:
            out.append("Big day.")
        crowded = len(things) > MAX_WALKED
        first = True
        for part in ("morning", "afternoon", "tonight"):
            group = [e for e in things if _part(e["start"]) == part]
            if not group:
                continue
            said = [e for e in group if not (crowded and e.get("block"))] or group
            extra = len(group) - len(said)
            run = series([_named(e, now, names, tonight=part != "tonight") for e in said])
            run += f", with {say_number(extra)} plan {'block' if extra == 1 else 'blocks'} in between" if extra else ""
            lead = "Tonight" if part == "tonight" else f"This {part}"
            verb = "you've got" if first else "there's" if part == "tonight" else "it's"
            out.append(f"{lead} {verb} {run}.")
            first = False
            if leave and any(e["title"] == leave["title"] for e in said):
                last = said[-1]["title"] == leave["title"]
                out.append(f"Leave by {_clock(leave['by'], True, tonight=False)} for {'that one' if last else ear_title(leave['title'], names)}.")
    if due := _due_clause(dues, now, names, everything=True):
        out.append(f"{due}.")
    if listed:
        two = " and ".join(_named(e, now, names) for e in listed[:2])
        lead = "If you want a break, " if things else ""
        if len(listed) <= 2:
            out.append(cap(f"{lead}on campus there's {two}."))
        else:
            out.append(cap(f"{lead}there are {say_number(len(listed))} things on campus, like {two}."))
    if things:
        out.append(PLAN_CHECK if planned else PLAN_OFFER)
    elif not planned and not earlier:
        out.append(PLAN_OFFER)
    return " ".join(out)


def day_fact_lines(things: list[dict], now: datetime, *, dues: list[dict] = (), listed: list[dict] = (), leave: dict | None = None,
                   names: dict | None = None, limit: int = 14) -> list[str]:
    """The same day as `said_rundown`, as the FACTS a model is given for any question about the day the patterns do
    not recognise ("is today going to be rough?", "when can I fit the gym in?"): one line per kind, in this module's
    wording, with the campus listings marked as optional so the model never calls them the person's."""
    def run(entries: list[dict]) -> str:
        more = len(entries) - limit
        return ", ".join(_named(e, now, names) for e in entries[:limit]) + (f", and {say_number(more)} more" if more > 0 else "")

    events, blocks = [e for e in things if not e.get("block")], [e for e in things if e.get("block")]
    lines = [f"Their own events for the rest of today: {run(events)}." if events else "They have no events of their own for the rest of today."]
    if blocks:
        lines.append(f"Blocks of the day plan they agreed to: {run(blocks)}.")
    if leave:
        lines.append(f"To be on time for {ear_title(leave['title'], names)} they should leave by {_clock(leave['by'], True, tonight=False)}.")
    if dues:
        lines.append("Due today or tomorrow: " + ", ".join(f"{ear_title(e['title'], names, article=False)} {_due_when(e, now)}" for e in dues[:limit]) + ".")
    else:
        lines.append("Nothing is due today or tomorrow.")
    if listed:
        lines.append(f"Optional campus events today (public listings, NOT their commitments; they are free during these): {run(list(listed))}.")
    return lines


def said_next(entry: dict | None, now: datetime, *, names: dict | None = None, where: bool = False, ends: bool = False,
              nearby: dict | None = None) -> str:
    """ "What's next?": one thing, and it is the person's own (an event, a class, a block of the plan, something due).
    `nearby`, the next campus listing, is said only when nothing of theirs is next, and as "on campus"."""
    if entry is None:
        if nearby:
            return f"Nothing of yours is next. On campus there's {_named(nearby, now, names)}."
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


def said_free(at: datetime, clash: dict | None, now: datetime, *, names: dict | None = None, busy: bool = False,
              nearby: dict | None = None) -> str:
    """ "Am I free at 3?" (or, with `busy`, "am I busy at 3?"): yes or no, and the one thing in the way. `nearby` is a
    campus listing on at that time: it never makes the person busy, it is offered ("... on campus then if you want it")."""
    part = "this morning" if at.hour < 12 else "this afternoon" if at.hour < 17 else "this evening" if at.hour < 21 else "tonight"
    when = "right now" if abs(at - now) < timedelta(minutes=1) else f"at {ear_time(at)}" + ("" if ear_time(at) in ("noon", "midnight") else f" {part}")
    if clash is None:
        offer = f" There's {ear_title(nearby['title'], names)} on campus then if you want it." if nearby else ""
        return f"{'No' if busy else 'Yes'}, you're free {when}.{offer}"
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

def said_done(done: list[str], greeting: bool = False, full: bool = False) -> str:
    """ "How's it going?": a mood word, one clause on what Atlas has done for the person so far (at most two things,
    each a past participle: "planned your day"), then the offer. A bare greeting gets the offer alone.
    `full` is the answer to "what did you do for me today?": all of it (five things at most) in two sentences,
    then the offer."""
    if greeting:
        return f"Hey. {OFFER_QUIET}"
    done = [d.strip() for d in done if d and d.strip()]
    if full:
        if not done:
            return f"Nothing yet today, it's been quiet. {OFFER_QUIET}"
        also = f" I've also {_and(done[2:5])}." if done[2:] else ""
        return f"Today I've {_and(done[:2])}.{also} {OFFER}"
    if not done:
        return f"Good, all quiet so far. {OFFER_QUIET}"
    return f"Good. I've {' and '.join(done[:2])} so far. {OFFER}"


def _and(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
