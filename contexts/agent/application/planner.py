"""Cross-source plans (A8): exam prep, and "what should I do tonight?".

When a Canvas or calendar item is an exam or deadline within 48 hours, Atlas
pre-fetches everything a small model needs and asks for one short plan:
- the item itself, and what else is on the calendar before it
- what the web and the school's subreddit say about the course
- what the cameras saw of last night's sleep (desk activity during sleep hours)
- free slots before the deadline, outside sleep hours
Every source is cited. Suggested study blocks are proposed as calendar events
through an approval card when the calendar connector can write.
"""

import re
from datetime import datetime, timedelta
from uuid import uuid4

from contexts.agent.application.qa import QA, Evidence
from contexts.connectors.application.hub import ConnectorHub
from contracts.agent import PlanBlock, PlanCreated
from contracts.connectors import ItemSynced
from contracts.memory import Item
from contracts.types import ChatMessage, Citation

HORIZON = timedelta(hours=48)
CHAT_LOOKBACK = timedelta(days=3)
# Something graded: a Canvas-like assignment, or a calendar entry named as an exam. Never a chat message that
# says "due", a "submissions due" calendar entry, or a notice that only sits in the assignment feed.
EXAM = re.compile(r"\b(exams?|midterms?|finals?|quiz(zes)?|tests?)\b", re.I)
NOTICE = re.compile(r"\b(opt[- ]?(in|out)|surveys?|evaluations?|gatorevals|voluntary|questionnaires?)\b", re.I)
COURSE = re.compile(r"\b([A-Z]{3})\s?(\d{4}[A-Z]?)\b")
EVENING_QUESTION = re.compile(r"\b(tonight|this evening|where should i go|where is everyone)\b", re.I)
STUDY_QUESTION = re.compile(r"\b(prep|prepare|study|exams?|midterms?|finals?|quiz(zes)?|tests?)\b", re.I)
STARTED = timedelta(hours=1)  # an event that began this recently is still on (calendar items carry no end)
EVENING = timedelta(hours=8)  # the window a plan without a deadline always had
MAX_EVENTS = 8
PLAN_QUESTION = re.compile(r"\b(plan|prep|prepare|study for|what should i do|where should i go|where is everyone)\b", re.I)
FOOD_QUESTION = re.compile(r"\b(eat|food|hungry|lunch|dinner|breakfast|snack)\b", re.I)
# Whole words only ("can" used to match "candle" and every can on the desk). EATING is someone having food or a
# drink; FOOD only names one, so it ranks lower. A mug, can or bottle on its own is a container, not a meal.
EATING = re.compile(r"\b(eat(s|ing|en)?|ate|drink(s|ing)|meals?|snack(s|ing)?|lunch|dinner|breakfast)\b", re.I)
FOOD = re.compile(r"\b(food(?!-)|pizza|sandwich(es)?|chips|energy drinks?|coffee(?! (mug|cup)))\b", re.I)
OBJECT_LIST = re.compile(r"^On the [\w ]+: ")  # perception's zone object list ("On the desk: keys (...), mug.")
MAX_MEALS = 6
LEAVING_COOLDOWN = timedelta(minutes=30)
CALENDAR_WRITE = "calendar.create_event"
BLOCK = timedelta(hours=2)
MAX_BLOCKS = 3

PLAN_SYSTEM = (
    "You are Atlas. Using only the evidence, write a short plan in at most 5 bullet points: "
    "what to study (topics from the sources), when (use the free slots given), when to sleep, "
    "and what to eat. Be concrete and kind. Mention times."
)


EVENING_SYSTEM = (
    "You are Atlas. Using only the evidence, say in two or three short sentences where to go or what to do "
    "this evening: name the events with their times and where friends are. A deadline gets one line at most."
)


def is_crunch(item: Item | ItemSynced, now: datetime) -> bool:
    if not item.when or not now <= item.when <= now + HORIZON or NOTICE.search(item.title):
        return False
    return item.kind == "assignment" or (item.kind == "calendar_event" and bool(EXAM.search(item.title)))


def evening_end(now: datetime, sleep) -> datetime:
    """When "tonight" ends: the next bedtime, and never less than EVENING from now (late, or past bedtime)."""
    local = now.astimezone(sleep.tz)
    if overlaps_sleep(sleep, local, local + timedelta(seconds=1)):
        return now + EVENING
    bed = datetime.combine(local.date(), sleep.bedtime, sleep.tz)
    return max(bed if bed > local else bed + timedelta(days=1), now + EVENING)


def course_of(text: str) -> str | None:
    m = COURSE.search(text)
    return f"{m[1]} {m[2]}" if m else None


def overlaps_sleep(sleep, start: datetime, end: datetime) -> bool:
    """Does [start, end) touch any sleep window (bedtime to wake) on the nights around it?"""
    for day in (start.date() - timedelta(days=1), start.date(), end.date()):
        bed = datetime.combine(day, sleep.bedtime, sleep.tz)
        wake = datetime.combine(day if sleep.bedtime < sleep.wake else day + timedelta(days=1), sleep.wake, sleep.tz)
        if bed < end and start < wake:
            return True
    return False


def free_blocks(now: datetime, until: datetime, busy: list[Item], qa: QA) -> list[tuple[datetime, datetime]]:
    """Up to MAX_BLOCKS two-hour blocks before `until`, outside sleep hours and busy events (1 h each)."""
    tz = qa.sleep.tz
    t = now.astimezone(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    taken = [(b.when, b.when + timedelta(hours=1)) for b in busy if b.when]
    blocks: list[tuple[datetime, datetime]] = []
    while t + BLOCK <= until and len(blocks) < MAX_BLOCKS:
        asleep = overlaps_sleep(qa.sleep, t, t + BLOCK)
        clash = any(s < t + BLOCK and t < e for s, e in taken)
        if not asleep and not clash:
            blocks.append((t, t + BLOCK))
            t += BLOCK + timedelta(hours=1)  # leave a break between blocks
        else:
            t += timedelta(hours=1)
    return blocks


class Planner:
    def __init__(
        self,
        qa: QA,
        hub: ConnectorHub | None = None,
        watch_items: tuple[str, ...] = ("keys", "wallet", "charger", "laptop"),
        leave_buffer: timedelta = timedelta(minutes=20),
        door_zone: str = "door",
        campus: str = "UCF",
    ) -> None:
        self.qa = qa
        self.hub = hub
        self.watch_items = watch_items
        self.leave_buffer = leave_buffer
        self.door_zone = door_zone
        self.campus = campus
        self._planned: set[str] = set()
        self._last_leaving: datetime | None = None
        self.presence = None  # PresenceBoard (G5), set by the composition root when the Circle is on
        self.habits = None  # HabitTracker (H1), set by the composition root: meals, drinks, sleep so far today
        self.places = None  # async (kind) -> nearby places (K8), set by the composition root
        self.place_set = None  # PlaceSet (S5): walking minutes from home to the next event
        self.home_place: str | None = None  # where the box is, e.g. "apartment.door" (config: leaving.home_place)
        self.accepted_optional = lambda: ()  # ids of optional events the person said yes to and approved; set by the composition root

    async def on_item(self, e: ItemSynced) -> list:
        """Bus handler for connector.item.synced: plan once per crunch item."""
        now = self.qa.clock()
        if e.external_id in self._planned or not is_crunch(e, now):
            return []
        self._planned.add(e.external_id)
        item = Item(**e.model_dump(include={"connector", "kind", "external_id", "title", "when", "url", "summary"}))
        return await self.plan(item.title, item, now)

    def wants(self, question: str) -> bool:
        return is_plan_question(question) or bool(FOOD_QUESTION.search(question))

    async def answer(self, question: str) -> PlanCreated:
        """For a plan-shaped question: plan around the next crunch item, or the evening if there is none."""
        if FOOD_QUESTION.search(question) and not is_plan_question(question):
            return await self.food(question)
        now = self.qa.clock()
        if is_evening_question(question):
            return await self.evening(question, now)
        upcoming = [i for i in self.qa.memory.items(now, now + HORIZON) if is_crunch(i, now)]
        events = await self.plan(question, upcoming[0] if upcoming else None, now)
        return next(e for e in events if isinstance(e, PlanCreated))

    async def evening(self, question: str, now: datetime) -> PlanCreated:
        """ "Where should I go tonight?": what is on from now until bedtime (`evening_end`) and where friends are, each cited.
        A deadline in the next 48 hours is one line of it, never the whole answer, and no study blocks."""
        ev = Evidence()
        end = evening_end(now, self.qa.sleep)
        # up to and including an event that starts at bedtime
        events = self.qa.memory.items(now - STARTED, end + timedelta(minutes=1), kind="calendar_event")
        for i in events[:MAX_EVENTS]:
            began = ", already started" if i.when < now else ""
            ev.add(f"Coming up: {i.title} at {self.qa._t(i.when)}{began}" + (f" ({i.summary})" if i.summary else ""),
                   Citation(url=i.url, title=i.title, ts=i.when))
        if not events:
            ev.lines.append(f"Nothing on the calendar between now and {self.qa._t(end)}.")
        self._friends(now, ev)
        due = next((i for i in self.qa.memory.items(now, now + HORIZON) if is_crunch(i, now)), None)
        if due:
            ev.add(f"Deadline coming up: {due.title} at {self.qa._t(due.when)}", Citation(url=due.url, title=due.title, ts=due.when))
        ev.lines.append(f"Usual bedtime {self.qa.sleep.bedtime:%I:%M %p}")
        prompt = "Evidence:\n" + "\n".join(f"- {line}" for line in ev.lines) + f"\n\nQuestion: {question}"
        r = await self.qa._chat([ChatMessage(role="system", content=EVENING_SYSTEM), ChatMessage(role="user", content=prompt)])
        return _plan_event(now, question, r.content.strip(), ev, r.model_version)

    async def plan(self, subject: str, item: Item | None, now: datetime) -> list:
        ev = Evidence()
        until = item.when if item and item.when else now + timedelta(hours=8)
        if item:
            ev.add(f"{item.title} at {self.qa._t(item.when)}" + (f" ({item.summary})" if item.summary else ""),
                   Citation(url=item.url, title=item.title, ts=item.when))
        busy = [i for i in self.qa.memory.items(now, until) if i.kind == "calendar_event" and (not item or i.external_id != item.external_id)]
        for b in busy[:5]:
            ev.add(f"Also on the calendar: {b.title} at {self.qa._t(b.when)}", Citation(url=b.url, title=b.title, ts=b.when))
        if not item:  # "what should I do tonight?" / "where should I go?": events, plans, friends
            for i in self.qa.memory.items(now, until)[:5]:
                ev.add(f"{i.title} at {self.qa._t(i.when)}", Citation(url=i.url, title=i.title, ts=i.when))
            self._friends(now, ev)

        course = course_of(item.title if item else subject)
        if course:  # what the group chats say about the course (GroupMe items, K4)
            code = course.replace(" ", "").lower()
            chat = [m for m in self.qa.memory.items(now - CHAT_LOOKBACK, now + timedelta(seconds=1), kind="message")
                    if code in m.title.replace(" ", "").lower() or code in (m.summary or "").replace(" ", "").lower()]
            for m in chat[-5:]:
                ev.add(f"Group chat, {self.qa._t(m.when)}: {m.summary or m.title}", Citation(url=m.url, title=m.title, ts=m.when))
        if course and self.qa.web:
            for search in (self.qa.web_search(f"{course} {item.title if item else ''} UCF".strip(), ev),
                           self.qa.reddit_search(course, None, ev)):
                try:
                    await search
                except Exception as exc:  # the web is optional for a plan
                    ev.lines.append(f"(web lookup failed: {exc})")

        start, end = self.qa.sleep.last_night(now)
        nightly = self.qa.memory.episodes(start, end, camera=None)
        late = [e for e in nightly if e.labels and "person" in e.labels]
        ev.lines.append(
            f"Last night ({self.qa._t(start)} to {self.qa._t(end)}): "
            + (f"activity at the desk {len(late)} times, last at {self.qa._t(late[-1].start)}" if late else "no activity at the desk")
        )
        ev.lines.append(f"Usual bedtime {self.qa.sleep.bedtime:%I:%M %p}, wake {self.qa.sleep.wake:%I:%M %p}")
        self._habit_lines(now, ev)

        blocks = free_blocks(now, until, busy, self.qa)
        topic = f"Study: {course or (item.title if item else 'focus time')}"
        plan_blocks = [PlanBlock(title=topic, start=s, end=e) for s, e in blocks]
        if plan_blocks:
            ev.lines.append("Free slots: " + ", ".join(f"{self.qa._t(b.start)}-{b.end.astimezone(self.qa.sleep.tz):%I:%M %p}" for b in plan_blocks))

        prompt = "Evidence:\n" + "\n".join(f"- {line}" for line in ev.lines) + f"\n\nMake a plan for: {subject}"
        r = await self.qa._chat([ChatMessage(role="system", content=PLAN_SYSTEM), ChatMessage(role="user", content=prompt)])
        plan = PlanCreated(
            ts=now,
            plan_id=uuid4().hex,
            subject=item.title if item else subject,
            text=r.content.strip(),
            citations=ev.citations,
            blocks=plan_blocks,
            model_version=r.model_version,
        )
        out: list = [plan]
        if plan_blocks and self.hub and self.hub.is_write(CALENDAR_WRITE):
            out.append(
                self.hub.propose_write(
                    CALENDAR_WRITE,
                    {"events": [b.model_dump(mode="json") for b in plan_blocks]},
                    f"Add {len(plan_blocks)} study block(s) for {plan.subject}",
                    now,
                )
            )
        return out


    def _habit_lines(self, now: datetime, ev: Evidence, only: tuple[str, ...] | None = None) -> None:
        """Today's habit numbers from Habits (H1): meals, drinks, last night's sleep, desk time."""
        if not self.habits:
            return
        try:
            rows = self.habits.digest(now=now).get("habits", [])
        except Exception:  # habits are context, never a reason to fail a plan
            return
        for r in rows:
            if only and r["id"] not in only or not r.get("applies", True):
                continue
            state = "met" if r["met"] else "not met yet"
            ev.lines.append(f"Habit {r['name']}: {r['value']} {r['unit']} today (target {r['direction'].replace('_', ' ')} {r['target']}, {state}, streak {r['streak']})")

    def _meals_counted(self, now: datetime) -> float:
        """Meals Habits has counted today (0 when Habits is off or has no meals habit)."""
        try:
            rows = self.habits.digest(now=now).get("habits", []) if self.habits else []
        except Exception:
            return 0
        return next((r["value"] for r in rows if r["id"] == "meals"), 0) or 0

    def _friends(self, now: datetime, ev: Evidence) -> None:
        """Where friends said they are (opt-in, coarse, unexpired)."""
        if not self.presence:
            return
        for p in self.presence.friends(now):
            until = p.expires_at.astimezone(self.qa.sleep.tz).strftime("%I:%M %p").lstrip("0")
            ev.add(f"{p.member_id} is at {p.place} (shared until {until})", Citation(title=f"{p.member_id} at {p.place}", ts=p.ts))

    # --- trigger 3: leaving checklist at the door ---

    def leaving(self, now: datetime) -> PlanCreated:
        """Deterministic, so it is instant at the door: next event, when to leave, where your things are."""
        ev = Evidence()
        lines = []
        upcoming = [i for i in self.qa.memory.items(now, now + timedelta(hours=8)) if i.kind in ("calendar_event", "assignment")
                    and (not i.external_id.startswith("nearby:") or i.external_id in self.accepted_optional())]  # an optional "near you" event is not what you are leaving for, unless accepted
        if upcoming:
            nxt = upcoming[0]
            walk, dest = self._walk_minutes(nxt)
            margin = timedelta(minutes=5) if walk is not None else self.leave_buffer
            leave_by = nxt.when - timedelta(minutes=walk or 0) - margin
            when = "now" if leave_by <= now else f"by {self.qa._t(leave_by).split(' ', 1)[1]}"
            trip = f" ({walk:.0f} min walk to {dest})" if walk is not None else ""
            lines.append(f"Next: {nxt.title} at {self.qa._t(nxt.when).split(' ', 1)[1]}; leave {when}{trip}.")
            ev.add(lines[-1], Citation(url=nxt.url, title=nxt.title, ts=nxt.when))
        for label in self.watch_items:
            s = self.qa.memory.last_seen(label)
            if s and s.zone_id != self.door_zone:
                where = f"{s.zone_id} zone" if s.zone_id else s.camera_id
                pos = f", {s.position}" if s.position else ""
                lines.append(f"{label.capitalize()}: last seen in the {where}{pos} at {self.qa._t(s.ts).split(' ', 1)[1]}.")
                ev.add(lines[-1], Citation(frame_ref=s.frame_ref, ts=s.ts))
        text = " ".join(lines) or "Nothing on the calendar and nothing left behind. Have a good one."
        return _plan_event(now, "Leaving", text, ev, "rules")

    def _walk_minutes(self, item) -> tuple[float | None, str | None]:
        """Walking minutes from home to where `item` is (its title or summary names a place), or (None, None)."""
        if not (self.place_set and self.home_place):
            return None, None
        dest = self.place_set.locate(f"{item.title} {item.summary or ''}")
        if not dest or dest == self.home_place:
            return None, None
        try:
            _, minutes = self.place_set.route(self.home_place, dest)
        except ValueError:
            return None, None
        pack, pid = dest.split(".", 1)
        return minutes, self.place_set.packs[pack].places[pid].name.split(" (")[0]

    def on_zone_event(self, e) -> list:
        """Bus handler for perception.zone.event: a person entering the door zone gets the checklist."""
        if e.label != "person" or e.zone_id != self.door_zone or e.kind != "enter":
            return []
        if self._last_leaving and e.ts - self._last_leaving < LEAVING_COOLDOWN:
            return []
        self._last_leaving = e.ts
        return [self.leaving(e.ts)]

    # --- trigger 2: what should I eat? ---

    async def food(self, question: str) -> PlanCreated:
        now = self.qa.clock()
        ev = Evidence()
        start = now.astimezone(self.qa.sleep.tz).replace(hour=0, minute=0, second=0, microsecond=0)
        eaten = meals_seen(self.qa.memory.episodes(start, now))
        for ep, c in eaten:
            ev.add(f"Seen today at {self.qa._t(ep.start)}: {c}",
                   Citation(episode_id=ep.id, frame_ref=ep.keyframe_refs[0] if ep.keyframe_refs else None, ts=ep.start))
        if not eaten and not self._meals_counted(now):  # never beside "Habit Meals: 2 meals today"
            ev.lines.append("No meals seen at the desk today.")
        self._habit_lines(now, ev, only=("meals", "energy_drinks", "sleep"))
        self._friends(now, ev)
        night_start, night_end = self.qa.sleep.last_night(now)
        ev.lines.append(f"Sleep window last night: {self.qa._t(night_start)} to {self.qa._t(night_end)}")
        if self.places:  # K8: real places near the box, from OpenStreetMap
            try:
                for pl in (await self.places("food"))[:6]:
                    ev.add(f"Nearby: {pl['title']} ({pl['summary']})", Citation(url=pl.get("url"), title=pl["title"]))
            except Exception as exc:
                ev.lines.append(f"(nearby places unavailable: {exc})")
        elif self.qa.web:
            try:
                await self.qa.web_search(f"healthy food open now near {self.campus}", ev)
            except Exception as exc:
                ev.lines.append(f"(web lookup failed: {exc})")
        prompt = "Evidence:\n" + "\n".join(f"- {line}" for line in ev.lines) + f"\n\nQuestion: {question}"
        system = "You are Atlas. Suggest what to eat in two or three short sentences, balancing what was already eaten today. Use only the evidence; name a place only if it is in the evidence."
        r = await self.qa._chat([ChatMessage(role="system", content=system), ChatMessage(role="user", content=prompt)])
        return _plan_event(now, question, r.content.strip(), ev, r.model_version)


def meals_seen(episodes: list) -> list[tuple]:
    """(episode, caption) for today's eating and drinking, once per episode, in time order.

    Object lists are skipped (a mug on the desk is not a meal). When there are more than MAX_MEALS,
    captions that show someone eating or drinking are kept before ones that only name a food."""
    found = []
    for ep in episodes:
        captions = [c for c in ep.captions if not OBJECT_LIST.match(c)]
        best = next((c for c in captions if EATING.search(c)), None) or next((c for c in captions if FOOD.search(c)), None)
        if best:
            found.append((not EATING.search(best), ep, best))
    keep = sorted(found, key=lambda f: (f[0], -f[1].start.timestamp()))[:MAX_MEALS]  # eating first, then the latest
    return [(ep, c) for _, ep, c in sorted(keep, key=lambda f: f[1].start)]


def _plan_event(now: datetime, subject: str, text: str, ev: Evidence, model_version: str) -> PlanCreated:
    return PlanCreated(ts=now, plan_id=uuid4().hex, subject=subject, text=text, citations=ev.citations, model_version=model_version)


def is_plan_question(question: str) -> bool:
    return bool(PLAN_QUESTION.search(question))


def is_evening_question(question: str) -> bool:
    """About this evening (where to go, what is on), not about studying for something."""
    return bool(EVENING_QUESTION.search(question)) and not STUDY_QUESTION.search(question)

