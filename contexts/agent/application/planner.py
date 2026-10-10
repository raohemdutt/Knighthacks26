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
CRUNCH = re.compile(r"\b(exam|midterm|final|quiz|test|due|deadline|project|presentation)\b", re.I)
COURSE = re.compile(r"\b([A-Z]{3})\s?(\d{4}[A-Z]?)\b")
PLAN_QUESTION = re.compile(r"\b(plan|prep|prepare|study for|what should i do|where should i go|where is everyone)\b", re.I)
FOOD_QUESTION = re.compile(r"\b(eat|food|hungry|lunch|dinner|breakfast|snack)\b", re.I)
FOOD_WORDS = ("eat", "food", "meal", "pizza", "sandwich", "snack", "lunch", "dinner", "breakfast", "chips", "drink", "coffee", "energy drink", "can", "bottle")
LEAVING_COOLDOWN = timedelta(minutes=30)
CALENDAR_WRITE = "calendar.create_event"
BLOCK = timedelta(hours=2)
MAX_BLOCKS = 3

PLAN_SYSTEM = (
    "You are Atlas. Using only the evidence, write a short plan in at most 5 bullet points: "
    "what to study (topics from the sources), when (use the free slots given), when to sleep, "
    "and what to eat. Be concrete and kind. Mention times."
)


def is_crunch(item: Item | ItemSynced, now: datetime) -> bool:
    if not item.when or not now <= item.when <= now + HORIZON:
        return False
    return item.kind == "assignment" or bool(CRUNCH.search(item.title))


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
        upcoming = [i for i in self.qa.memory.items(now, now + HORIZON) if is_crunch(i, now)]
        events = await self.plan(question, upcoming[0] if upcoming else None, now)
        return next(e for e in events if isinstance(e, PlanCreated))

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


    def _friends(self, now: datetime, ev: Evidence) -> None:
        """Where friends said they are (opt-in, coarse, unexpired)."""
        if not self.presence:
            return
        for p in self.presence.friends(now):
            until = p.expires_at.astimezone(self.qa.sleep.tz).strftime("%I:%M %p").lstrip("0")
            ev.lines.append(f"{p.member_id} is at {p.place} (shared until {until})")

    # --- trigger 3: leaving checklist at the door ---

    def leaving(self, now: datetime) -> PlanCreated:
        """Deterministic, so it is instant at the door: next event, when to leave, where your things are."""
        ev = Evidence()
        lines = []
        upcoming = [i for i in self.qa.memory.items(now, now + timedelta(hours=8)) if i.kind in ("calendar_event", "assignment")]
        if upcoming:
            nxt = upcoming[0]
            leave_by = nxt.when - self.leave_buffer
            when = "now" if leave_by <= now else f"by {self.qa._t(leave_by).split(' ', 1)[1]}"
            lines.append(f"Next: {nxt.title} at {self.qa._t(nxt.when).split(' ', 1)[1]}; leave {when}.")
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
        eaten = [
            (ep, c) for ep in self.qa.memory.episodes(start, now) for c in ep.captions if any(w in c.lower() for w in FOOD_WORDS)
        ]
        for ep, c in eaten[-6:]:
            ev.add(f"Seen today at {self.qa._t(ep.start)}: {c}", Citation(episode_id=ep.id, ts=ep.start))
        if not eaten:
            ev.lines.append("No meals seen at the desk today.")
        self._friends(now, ev)
        night_start, night_end = self.qa.sleep.last_night(now)
        ev.lines.append(f"Sleep window last night: {self.qa._t(night_start)} to {self.qa._t(night_end)}")
        if self.qa.web:
            try:
                await self.qa.web_search(f"healthy food open now near {self.campus}", ev)
            except Exception as exc:
                ev.lines.append(f"(web lookup failed: {exc})")
        prompt = "Evidence:\n" + "\n".join(f"- {line}" for line in ev.lines) + f"\n\nQuestion: {question}"
        system = "You are Atlas. Suggest what to eat in two or three short sentences, balancing what was already eaten today. Use only the evidence; name a place only if it is in the evidence."
        r = await self.qa._chat([ChatMessage(role="system", content=system), ChatMessage(role="user", content=prompt)])
        return _plan_event(now, question, r.content.strip(), ev, r.model_version)

def _plan_event(now: datetime, subject: str, text: str, ev: Evidence, model_version: str) -> PlanCreated:
    return PlanCreated(ts=now, plan_id=uuid4().hex, subject=subject, text=text, citations=ev.citations, model_version=model_version)


def is_plan_question(question: str) -> bool:
    return bool(PLAN_QUESTION.search(question))

