"""Questions about the person's calendar, answered like a person and from the data itself (no model call).

OWNER'S RULE (Hemdutt): Atlas speaks naturally and never reads the calendar out. "What does my day look like?" is
"You've got your algorithms lecture at 10:40, then physics at 2.", never "Coming up today: COP4533 lecture, Sun
10:40 AM; PHY2049 lecture, Sun 2:00 PM; ...". Only what was asked: "what's next?" names one thing, "am I free at
3?" says yes or no and the one thing in the way, a rundown names three things at most unless everything is asked for.

    ask = parse(question, now, tz)             # None: not a calendar question
    s = said(ask, memory, now, tz, listings=listings, plan=blocks)   # s.text, s.items (cited), s.found
    d = day(items, now, tz, listings, plan)    # the whole day, gathered: what `said` and the model's FACTS are built from

THE CONTRACT (the owner: "we can't have hard-coded answers, because Atlas has to be prepared for anything"):
  1. The facts are always fetched live, every time: the calendar and what is due from Memory, the agreed plan through
     the Day context's port (`plan`: DayLoop.blocks, wired in contexts/gateway/bootstrap.py). Nothing is cached or canned.
  2. The sentences built in code are only a FAST LANE for the phrasings `parse` recognises.
  3. Any other question about the day ("is today going to be rough?", "when can I fit the gym in?") goes to the
     model, which is given the same facts in the same for-the-ear wording (`day` -> ear.day_fact_lines, talk.py `facts`).
Campus listings are never the person's own: which stored items are listings is read from the item itself against
config/agent.yaml `listings:` (`Listings`), not from a name written in code.

Why this exists: the box's voice (contexts/voice) posts what it heard straight to /api/ask. The phone's Ask screen
used to catch "what's my day" and "what's next" itself and say the narrator's words (web/lib/speechText.ts BRIEF),
so by voice on the box "what's on my calendar today?" got the list written for the screen ("Coming up today: ...;
...") and "what does my day look like?" got small talk. The same sentences now come from here on both paths the
questions already took: QA's calendar pre-fetch (qa.py `upcoming_items`) and the conversation (talk.py `calendar`).
Which question goes to which path is unchanged; only the words are.

The wording is contexts/day/domain/ear.py (pure functions next to the narrator's wording). This module decides
which question was asked (patterns, never a model call) and which calendar items answer it.
"""

import re
from dataclasses import dataclass, replace  # noqa: F401  (`replace` is used by talk.py: the same ask, everything)
from datetime import datetime, timedelta, tzinfo
from typing import NamedTuple
from urllib.parse import urlparse

from contexts.day.domain import ear
from contracts.memory import Item

LEAD = re.compile(r"^(?:(?:hey|hi|hello|yo|ok|okay|so|well|um|uh|oh|and|but|please)\b[ ,!.]*)*(?:atlas\b[ ,!.:]*)?")
WINDOW = r"(?:today|tonight|tomorrow(?: (?:morning|afternoon|evening|night))?|this (?:morning|afternoon|evening|weekend|week)|next week|later(?: today)?|the rest of (?:the|my) day)"
THING = r"(?:class|classes|lecture|lab|meeting|exam|midterm|final|quiz|appointment|interview|event)"
TAIL = rf"(?: (?:for|on|in|at))?(?: (?:{WINDOW}|(?:my|the) (?:calendar|schedule|agenda|day)))*[?.! ]*$"

DUE = re.compile(r"\b(due|deadlines?|homework|assignments?)\b")
# what the fixed path already took (A9): the calendar by name, or a thing plus a time to come
ITEMS_Q = re.compile(r"\b(due|deadlines?|calendar|schedule|coming up|upcoming|happening|going on|on campus)\b")
ITEM_THING = re.compile(r"\b(assignments?|homework|exams?|quiz(zes)?|classes|events?|meetings?)\b")
FUTURE = re.compile(r"\b(tonight|today|tomorrow|this week|next week|this weekend|weekend|upcoming|soon)\b")
ITEM_KINDS = {
    "assignment": re.compile(r"\b(due|deadlines?|assignments?|homework|exams?|quiz)\b"),
    "calendar_event": re.compile(r"\b(calendar|schedule|classes|meetings?|events?|happening|going on|on campus|tonight)\b"),
}

FREE = re.compile(r"\b(?:am i|are we|will i be) (?P<word>free|busy|available|booked|doing anything)\b|\bdo i have (?:time|anything|something) (?:at|around) ")
AT_TIME = re.compile(r"\b(?:at|around|by|until|till) (?:(?P<h>\d{1,2})(?::(?P<m>\d{2}))? ?(?P<ap>a\.?m\.?|p\.?m\.?)?(?![\d:])|(?P<word>noon|midnight))")
NEXT = re.compile(r"\bwhat(?:'s| is| do i have| have i got)(?: up| on| coming)? next\b|\bmy next (?:thing|event)\b|\bwhat(?:'s| is) after (?:this|that)\b"
                  r"|\bwhat(?:'s| is) next\b")
NEXT_THING = re.compile(rf"^(?P<ask>when|what time|where)(?:'s| is| does| do i have)? (?:my|the|our) next (?P<thing>{THING}|thing)\b"
                        rf"|\bwhat(?:'s| is) my next (?P<thing2>{THING})\b")
WHEN = re.compile(rf"^(?P<ask>when|what time|where)(?:'s| is| does| do i have)? (?:my|the|our) (?:[a-z0-9]+ ){{0,2}}?(?P<thing>{THING})\b")
HAVE = re.compile(rf"\bdo (?:i|we) have (?P<what>anything|something|any plans|plans|stuff|much|a lot|(?:a |an |any )?{THING})"
                  rf"(?: on| going on| planned| scheduled| coming up| left)?{TAIL}")
WHAT_HAVE = re.compile(rf"\bwhat (?:do i|have i) (?:have|got)(?: on| going on| planned| scheduled| left| coming up)?{TAIL}")
DAY = re.compile(
    r"\b(?:what does|how does|how's|how is|what's|what is|whats) my (?:day|morning|afternoon|evening|night|week|weekend|tomorrow|schedule|calendar"
    r"|agenda|plan)\b(?! (?:going|been|so far|gone))"
    rf"|\bwhat(?:'s| is) (?:on|planned|scheduled) (?:for )?{WINDOW}"
    r"|\bbrief me\b|\b(?:run|walk|take) me through my (?:day|schedule|calendar|week)\b|\b(?:rundown|run-down|overview) of my (?:day|schedule|week)\b"
    r"|\bgive me (?:a|the|my) (?:brief|briefing|rundown|agenda)\b"
    rf"|\bmy plans? for {WINDOW}|\bwhat(?:'s| is) the plan\b(?: for {WINDOW})?[?.! ]*$"
)
EVERYTHING = re.compile(r"\b(?:everything|all of (?:it|them)|whole|full|entire|every|all my|list)\b")
ENDS = re.compile(r"\bhow long\b|\buntil when\b|\b(?:end|ends|finish|finishes|over)\b")
UNTIL = re.compile(r"\buntil\s+(\d{1,2}):(\d{2})\s*([AP]M)", re.I)
CALENDAR_ONLY = re.compile(r"\bcalendar\b")  # asked for the calendar by name: the calendar, not the whole day
NOT_A_PLACE = re.compile(r"^(?:free|optional|with |hosted by |via |until )", re.I)
CLASS_WORDS = re.compile(r"\b(?:lecture|class|lab|seminar|recitation|section|discussion)\b|\b[A-Z]{2,4} ?\d{3,4}[A-Z]?\b")
NOT_A_CLASS = re.compile(r"\b(?:study|review|office hours|exam|midterm|final|quiz|due)\b", re.I)
DEFAULT_LENGTH = timedelta(hours=1)  # an event whose end the calendar did not give


@dataclass(frozen=True)
class Ask:
    kind: str  # "day" (a stretch of the calendar), "next", "when", "free", "due"
    start: datetime
    end: datetime
    span: str = "today"  # the stretch in words: "today", "this afternoon", "tomorrow", "this week", "soon"
    at: datetime | None = None  # free: the moment asked about
    thing: str = ""  # "class", "exam": the kind of thing asked about
    yes_no: str = ""  # how it was put: "have" (do I have anything ...?), "free" (am I free ...?)
    everything: bool = False
    where: bool = False
    ends: bool = False
    kinds: tuple = ("calendar_event",)  # memory item kinds; (None,) is any kind
    rundown: bool = False  # "what does my day look like?": the whole day (own events, the plan's blocks, what is due, listings)


@dataclass(frozen=True)
class Listings:
    """Which stored items are public listings (campus events a feed publishes) and not the person's own commitments.
    Read from the item's own data against config/agent.yaml `listings:`; no school or site is named in code.

    On the box the campus feed is one of the ics connector's feeds (config/connectors.yaml `calendar.options.feeds`),
    so its items arrive as connector "calendar", kind "calendar_event", exactly like the person's own classes. What
    tells them apart is the feed name the connector puts in front of the external id ("campus:<uid>:<start>") and
    where the link points ("https://events.ucf.edu/event/<id>/<slug>/")."""

    connectors: tuple = ("site", "search", "groupme")  # connectors that only ever list things going on nearby
    feeds: tuple = ()  # calendar feeds by name: the first part of the external id ("campus")
    hosts: tuple = ()  # where a listing's link points ("events.ucf.edu"); subdomains count
    optional: tuple = ("optional",)  # a word in the title or summary that marks an invitation, not a commitment
    own: tuple = ("attending", "registered", "rsvp")  # a word in the summary that makes a listed thing the person's own

    @classmethod
    def of(cls, raw: dict | None = None) -> "Listings":
        raw, d = raw or {}, cls()

        def words(key: str) -> tuple:
            return tuple(str(x).strip().lower() for x in raw[key] or ()) if key in raw else getattr(d, key)

        return cls(**{key: words(key) for key in ("connectors", "feeds", "hosts", "optional", "own")})

    def listed(self, item: Item) -> bool:
        text = f"{item.title} {item.summary or ''}".lower()
        if any(re.search(rf"\b{re.escape(w)}", text) for w in self.own):
            return False
        host = (urlparse(item.url or "").hostname or "").lower()
        return (item.connector.lower() in self.connectors or item.external_id.split(":", 1)[0].lower() in self.feeds
                or any(host == h or host.endswith("." + h) for h in self.hosts)
                or any(re.search(rf"\b{re.escape(w)}\b", text) for w in self.optional))


class Said(NamedTuple):
    text: str
    items: list  # the memory items the sentence rests on (cited by link)
    found: bool  # False: nothing of any kind answered it, so a caller may say something better


def normal(question: str) -> str:
    q = re.sub(r"\s+", " ", question.replace("’", "'").lower()).strip()
    return LEAD.sub("", q, count=1) or q


def window(q: str, now: datetime, tz: tzinfo, default: str = "today") -> tuple[datetime, datetime, str]:
    """The stretch of time a question names, from `now` on: (start, end, its words)."""
    local = now.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)

    def at(day: int, hour: int) -> datetime:
        return midnight + timedelta(days=day, hours=hour)

    if m := re.search(r"\btomorrow(?: (morning|afternoon|evening|night))?", q):
        lo, hi = {"morning": (0, 12), "afternoon": (12, 17), "evening": (17, 24), "night": (17, 27)}.get(m[1], (0, 24))
        return at(1, lo), at(1, hi), m[0]
    if "this morning" in q:
        return local, max(local, at(0, 12)), "this morning"
    if "this afternoon" in q:
        return max(local, at(0, 12)), max(local, at(0, 17)), "this afternoon"
    if "tonight" in q or "this evening" in q:
        return max(local, at(0, 17)), at(1, 3), "tonight" if "tonight" in q else "this evening"
    if "weekend" in q:
        saturday = midnight + timedelta(days=(5 - local.weekday()) % 7)
        return max(local, saturday), saturday + timedelta(days=2), "this weekend"
    if "next week" in q:
        monday = midnight + timedelta(days=7 - local.weekday())
        return monday, monday + timedelta(days=7), "next week"
    if "this week" in q or (default == "week" and "today" not in q and "later" not in q and "rest of" not in q):
        return local, local + timedelta(days=7), "this week" if "this week" in q else "soon"
    return local, at(1, 0), "today"


def moment(q: str, now: datetime, tz: tzinfo) -> datetime | None:
    """ "at 3", "around 2:30 pm", "at noon": that time today. With no am or pm, the one still ahead in waking hours."""
    m = AT_TIME.search(q)
    if not m:
        return None
    local = now.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if m["word"]:
        return midnight + timedelta(hours=12 if m["word"] == "noon" else 24)
    h, minute = int(m["h"]), int(m["m"] or 0)
    if h > 23 or minute > 59:
        return None
    if m["ap"] or h > 12 or h == 0:
        if m["ap"]:
            h = h % 12 + (12 if m["ap"].startswith("p") else 0)
        return midnight + timedelta(hours=h, minutes=minute)
    options = [midnight + timedelta(hours=x, minutes=minute) for x in sorted({h % 12, h % 12 + 12})]
    awake = [t for t in options if t.hour >= 7]
    ahead = [t for t in awake if t >= local]
    return (ahead or awake or options)[0]


def parse(question: str, now: datetime, tz: tzinfo) -> Ask | None:
    """Which calendar question this is; None when it is not one."""
    q = normal(question)
    everything, ends = bool(EVERYTHING.search(q)), bool(ENDS.search(q))
    listed = ITEMS_Q.search(q) or (ITEM_THING.search(q) and FUTURE.search(q))
    if DUE.search(q) and listed:  # "what is due this week?": the assignments
        start, end, span = window(q, now, tz, default="week")
        return Ask("due", start, end, "this week" if span == "soon" and "soon" not in q else span, everything=everything, kinds=_kinds(q))
    if m := FREE.search(q):
        yes_no = "have" if (m["word"] or "") in ("busy", "booked", "doing anything") else "free"
        if at := moment(q, now, tz):
            return Ask("free", at, at, at=at, yes_no=yes_no)
        if re.search(WINDOW, q):
            start, end, span = window(q, now, tz)
            return Ask("day", start, end, span, yes_no=yes_no, everything=everything)
        local = now.astimezone(tz)
        return Ask("free", local, local, at=local, yes_no=yes_no)
    if m := NEXT_THING.search(q):
        thing = m["thing"] or m["thing2"] or ""
        return Ask("when", now, now + timedelta(days=14), thing="" if thing == "thing" else thing, where="where" in q, ends=ends,
                   kinds=_thing_kinds(thing))
    if NEXT.search(q):
        return Ask("next", now, now + timedelta(days=7), where="where" in q, ends=ends)
    if m := WHEN.search(q):
        return Ask("when", now, now + timedelta(days=14), thing=m["thing"], where="where" in q, ends=ends, kinds=_thing_kinds(m["thing"]))
    if m := HAVE.search(q):
        start, end, span = window(q, now, tz)
        what = re.sub(r"^(?:a|an|any) ", "", m["what"])
        thing = what if re.fullmatch(THING, what) else ""
        return Ask("day", start, end, span, thing=thing, yes_no="have", everything=everything, kinds=_thing_kinds(thing))
    if WHAT_HAVE.search(q) or DAY.search(q):
        start, end, span = window(q, now, tz, default="week" if re.search(r"\bmy week\b", q) else "today")
        return Ask("day", start, end, span, everything=everything, rundown=span == "today" and not CALENDAR_ONLY.search(q))
    if listed:  # "what's on my calendar?", "any events this weekend?": as the fixed path always took them
        start, end, span = window(q, now, tz, default="week")
        return Ask("day", start, end, span, everything=everything, kinds=_kinds(q))
    return None


def _kinds(q: str) -> tuple:
    return tuple(k for k, rx in ITEM_KINDS.items() if rx.search(q)) or (None,)


def _thing_kinds(thing: str) -> tuple:
    return ("calendar_event", "assignment") if thing in ("exam", "midterm", "final", "quiz") else ("calendar_event",)


# ---------- the items that answer it ----------

def entry(item: Item, tz: tzinfo, listings: Listings | None = None) -> dict:
    """A calendar item as the wording takes it: local times, the end and the place from its one-line summary
    ("until 11:30 AM · CSE A101 · optional"), and whether it is only a listing (`Listings`)."""
    start = item.when.astimezone(tz)
    end, place = None, None
    for part in [p.strip() for p in (item.summary or "").split("·")]:
        if m := UNTIL.search(part):
            hour = int(m[1]) % 12 + (12 if m[3].upper() == "PM" else 0)
            end = start.replace(hour=hour, minute=int(m[2]), second=0, microsecond=0)
            if end <= start:
                end += timedelta(days=1)
        elif part and place is None and item.kind == "calendar_event" and not NOT_A_PLACE.search(part) and len(part) <= 60:
            place = part
    due = item.kind == "assignment"
    return {"title": item.title, "start": start, "end": end, "place": place, "item": item, "due": due,
            "listed": not due and (listings or Listings()).listed(item)}


def entries_of(items: list[Item], tz: tzinfo, listings: Listings | None = None) -> list[dict]:
    return sorted((entry(i, tz, listings) for i in {i.external_id: i for i in items if i.when}.values()), key=lambda e: e["start"])


def plan_parts(plan: list[dict] | None, local: datetime, end: datetime, tz: tzinfo, own: list[dict]) -> tuple[list[dict], dict | None]:
    """The Day plan's port (DayLoop.blocks) as the wording takes it: the blocks still ahead today, and when to leave
    for the next thing that needs travel. The leave-by time is kept only for one of the person's own events: the
    plan also walks to things that are merely listed, and nobody has to leave for those."""
    blocks, leave = [], None
    for p in plan or []:
        start = p["start"].astimezone(tz)
        if p.get("kind") == "leave":
            if leave is None and start >= local and any(e["title"] == p["title"] for e in own):
                leave = {"title": p["title"], "by": start}
        elif local <= start < end:
            finish = p["end"].astimezone(tz) if p.get("end") else None
            blocks.append({"title": p["title"], "start": start, "end": finish, "place": None, "item": None, "due": False,
                           "listed": False, "block": True})
    return sorted(blocks, key=lambda e: e["start"]), leave


def day(items: list[Item], now: datetime, tz: tzinfo, listings: Listings | None = None, plan: list[dict] | None = None) -> dict:
    """The person's whole day from `now`, gathered live and in time order: `things` (their own calendar events and
    classes, and the blocks of the agreed plan), `dues` (due today or tomorrow), `listed` (campus or optional
    listings today: never theirs), `leave` (when to set off for the next thing that needs travel), `earlier`
    (something of theirs was on before now) and `planned` (a plan exists). The fast lane's sentences
    (ear.said_rundown) and the model's FACTS (ear.day_fact_lines) are both built from this and nothing else."""
    local = now.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = midnight + timedelta(days=1)
    entries = entries_of(items, tz, listings)
    events = [e for e in entries if not e["due"]]
    own = [e for e in events if not e["listed"] and local <= e["start"] < end]
    blocks, leave = plan_parts(plan, local, end, tz, own)
    return {"things": sorted(own + blocks, key=lambda e: e["start"]),
            "dues": [e for e in entries if e["due"] and local <= e["start"] < end + timedelta(days=1)],
            "listed": [e for e in events if e["listed"] and local <= e["start"] < end],
            "leave": leave, "planned": plan is not None,
            "earlier": any(midnight <= e["start"] < local for e in events if not e["listed"])}


def day_from(memory, now: datetime, tz: tzinfo, listings: Listings | None = None, plan: list[dict] | None = None) -> dict:
    """`day`, with the items read from the memory now."""
    midnight = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    items = [i for k in ("calendar_event", "assignment") for i in memory.items(midnight - timedelta(hours=12), midnight + timedelta(days=2), kind=k)]
    return day(items, now, tz, listings, plan)


def matches(e: dict, thing: str) -> bool:
    if not thing:
        return True
    if thing in ("class", "classes", "lecture"):  # a lecture or a lab, or a course by its code; not a study session for it
        return bool(CLASS_WORDS.search(e["title"])) and not e["due"] and not NOT_A_CLASS.search(e["title"])
    return thing.rstrip("s") in e["title"].lower() or (thing == "exam" and re.search(r"\b(?:midterm|final|test)\b", e["title"], re.I) is not None)


def _cited(entries: list[dict]) -> list[Item]:
    return [e["item"] for e in entries if e.get("item") is not None][:10]


def answer(ask: Ask, items: list[Item], now: datetime, tz: tzinfo, names: dict | None = None, listings: Listings | None = None,
           plan: list[dict] | None = None) -> Said:
    """The sentence, the items it rests on, and whether anything answered it. `items` are the memory's items of
    `ask.kinds` from 12 hours before `ask.start` to `ask.end` (an event that began earlier may still be on); a rundown
    and "what's next?" are given what is due as well. `plan` is the Day plan's port (None: there is no plan).

    The person's own calendar comes first. Things a connector only lists (campus events, an optional invitation)
    are never said as theirs: they are named, as "on campus", only when the calendar itself has nothing in that
    stretch or everything was asked for, and they never make the person "busy"."""
    local = now.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if ask.rundown:  # the whole day: own events, the plan's blocks, what is due, and one clause for what is on nearby
        d = day(items, now, tz, listings, plan)
        text = ear.said_rundown(d["things"], local, dues=d["dues"], listed=d["listed"], leave=d["leave"], names=names,
                                everything=ask.everything, planned=d["planned"], earlier=d["earlier"])
        dues = d["dues"] if ask.everything else d["dues"][:1]
        return Said(text, _cited(d["things"] + dues + d["listed"]), bool(d["things"] or d["dues"] or d["listed"]))
    entries = entries_of(items, tz, listings)
    own = [e for e in entries if ask.kind == "due" or not e["listed"]]
    if ask.kind == "free":
        def on_then(e: dict) -> bool:
            return not e["due"] and e["start"] <= ask.at < (e["end"] or e["start"] + DEFAULT_LENGTH)

        clash = next((e for e in own if on_then(e)), None)
        nearby = None if clash else next((e for e in entries if e["listed"] and on_then(e)), None)  # offered, never in the way
        return Said(ear.said_free(ask.at, clash, local, names=names, busy=ask.yes_no == "have", nearby=nearby),
                    _cited([e for e in (clash, nearby) if e]), True)
    ahead = [e for e in own if ask.start <= e["start"] < ask.end]
    if ask.kind == "due":
        return Said(ear.said_due(ahead, local, ask.span, names=names, everything=ask.everything), _cited(ahead), bool(ahead))
    if ask.kind == "next":
        # the person's own next thing: an event or a class, a block of the plan, or something due today. A listing
        # is named only when nothing of theirs is next, and as "on campus".
        tomorrow = midnight + timedelta(days=1)
        events = [e for e in ahead if not e["due"] and e["start"] >= local]
        blocks, _ = plan_parts(plan, local, tomorrow, tz, events)
        dues = [e for e in entries if e["due"] and local <= e["start"] < tomorrow]
        first = min(events[:1] + blocks[:1] + dues[:1], key=lambda e: e["start"], default=None)
        nearby = None if first else next((e for e in entries if e["listed"] and local <= e["start"] < tomorrow), None)
        text = ear.said_next(first, local, names=names, where=ask.where, ends=ask.ends, nearby=nearby)
        return Said(text, _cited([e for e in (first, nearby) if e]), bool(first or nearby))
    if ask.kind == "when":
        found = [e for e in ahead if matches(e, ask.thing)]
        first = found[0] if found else None
        thing = {"classes": "class"}.get(ask.thing, ask.thing)
        return Said(ear.said_when(first, local, thing, names=names, where=ask.where, ends=ask.ends), _cited(found[:1]), bool(first))
    found = [e for e in ahead if matches(e, ask.thing)]
    nearby = [e for e in entries if e["listed"] and ask.start <= e["start"] < ask.end and matches(e, ask.thing)]
    listed = False
    if nearby and (ask.everything or not found) and ask.yes_no != "free":
        listed = not found
        found = sorted(found + nearby, key=lambda e: e["start"])
    earlier = ask.span == "today" and any(midnight <= e["start"] < ask.start for e in own)
    text = ear.said_agenda(found, local, ask.span, names=names, everything=ask.everything, yes_no=ask.yes_no, earlier=earlier, listed=listed)
    behind = sorted({id(e): e for e in found + nearby}.values(), key=lambda e: e["start"])  # cited as the list used to cite them
    return Said(text, _cited(behind), bool(behind))


def said(ask: Ask, memory, now: datetime, tz: tzinfo, names: dict | None = None, listings: Listings | None = None,
         plan: list[dict] | None = None) -> Said:
    """`answer`, with the items read from the memory now (anything with `items(start, end, kind=)`)."""
    midnight = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    since = min(ask.start, midnight) - timedelta(hours=12)
    until = max(ask.end, ask.start + timedelta(minutes=1))
    kinds = ("calendar_event", "assignment") if ask.kinds == (None,) else ask.kinds
    if ask.rundown or ask.kind == "next":  # these also say what is due
        kinds, until = ("calendar_event", "assignment"), max(until, midnight + timedelta(days=2))
    items = [i for k in kinds for i in memory.items(since, until, kind=k)]
    return answer(ask, items, now, tz, names, listings, plan)

