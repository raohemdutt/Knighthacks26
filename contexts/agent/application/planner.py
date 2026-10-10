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
PLAN_QUESTION = re.compile(r"\b(plan|prep|prepare|study for|what should i do)\b", re.I)
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
    def __init__(self, qa: QA, hub: ConnectorHub | None = None) -> None:
        self.qa = qa
        self.hub = hub
        self._planned: set[str] = set()

    async def on_item(self, e: ItemSynced) -> list:
        """Bus handler for connector.item.synced: plan once per crunch item."""
        now = self.qa.clock()
        if e.external_id in self._planned or not is_crunch(e, now):
            return []
        self._planned.add(e.external_id)
        item = Item(**e.model_dump(include={"connector", "kind", "external_id", "title", "when", "url", "summary"}))
        return await self.plan(item.title, item, now)

    def wants(self, question: str) -> bool:
        return is_plan_question(question)

    async def answer(self, question: str) -> PlanCreated:
        """For a plan-shaped question: plan around the next crunch item, or the evening if there is none."""
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
        if not item:  # "what should I do tonight?": campus events and plans tonight
            for i in self.qa.memory.items(now, until)[:5]:
                ev.add(f"{i.title} at {self.qa._t(i.when)}", Citation(url=i.url, title=i.title, ts=i.when))

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


def is_plan_question(question: str) -> bool:
    return bool(PLAN_QUESTION.search(question))

