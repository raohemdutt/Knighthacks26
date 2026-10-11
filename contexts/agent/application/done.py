"""What Atlas has done for the person so far today, for "how's it going?".

OWNER'S RULE (Hemdutt): asked how it is going, Atlas says "Good. I've planned your day and reminded you once so
far. What else can I do for you?": its work for you, never the time, its uptime, the camera log or your schedule.

    done = DoneToday(tz)
    for prefix in DoneToday.PREFIXES:
        bus.subscribe(prefix, done.on_event, replay=True)      # a projection: it rebuilds from the log
    talk.done_for_you = lambda: done.clauses(datetime.now(UTC))  # ["planned your day", "reminded you once"]

Every clause is counted from real events, never guessed: a plan or re-plan the Day loop proposed, something the
person approved that was then carried out, something still waiting for their OK (said as waiting, its contents
are not read out), reminders (habit nudges), a check-in, alerts, and questions answered. `clauses` returns them
most significant first as past participles that follow "I've"; the wording of the whole sentence is the Day
context's ear.py (`said_done`). Other contexts' events are read by their type and fields only: nothing of theirs
is imported but the Day context's pure wording helpers.
"""

from datetime import datetime, timedelta, tzinfo

from contexts.day.domain.narration import say_number

DAY_STARTS_AT = 4  # o'clock: what happened after midnight belongs to the night before
WAITING_FOR = timedelta(minutes=10)  # a card not decided after this long is no longer "waiting" (cards expire)
KEEP = 400
# an approved action that was carried out, by kind: (said for one, the noun for several: "shared three things with your circle")
ACTION_SAID = {"circle_share": ("shared one thing with your circle", "shared {n} things with your circle"),
               "send_message": ("sent the message you approved", "sent the {n} messages you approved"),
               "connector_write": ("saved the change you approved", "saved the {n} changes you approved"),
               "browse": ("opened the page you approved", "opened the {n} pages you approved"),
               "export_training_data": ("exported the examples you approved", "exported the examples you approved")}
ONE_DONE, SEVERAL_DONE = "done the one thing you approved", "done the {n} things you approved"


def times(n: int) -> str:
    return "once" if n == 1 else "twice" if n == 2 else f"{say_number(n)} times"


class DoneToday:
    PREFIXES = ("day.plan.proposed", "day.replan.proposed", "day.checkin.raised", "agent.action.proposed", "agent.answer.generated",
                "agent.alert.raised", "habits.nudge.raised", "workspace.action.completed", "experience.approval.decided")

    def __init__(self, tz: tzinfo) -> None:
        self.tz = tz
        self.events: list[tuple[datetime, str, str]] = []  # (when, type, action id or kind)
        self.kinds: dict[str, str] = {}  # action id -> kind
        self.decided: set[str] = set()

    def on_event(self, e) -> None:
        action = getattr(e, "action_id", "") or ""
        if e.type == "agent.action.proposed":
            self.kinds[action] = getattr(e, "kind", "")
        elif e.type == "experience.approval.decided":
            self.decided.add(action)
            return
        elif e.type == "workspace.action.completed":
            self.decided.add(action)
            if getattr(e, "status", "") != "done":
                return
        self.events.append((e.ts, e.type, action))
        del self.events[:-KEEP]

    def since(self, now: datetime) -> datetime:
        local = now.astimezone(self.tz)
        start = local.replace(hour=DAY_STARTS_AT, minute=0, second=0, microsecond=0)
        return start if local >= start else start - timedelta(days=1)

    def clauses(self, now: datetime) -> list[str]:
        """What it did since the day began, most significant first; [] on a quiet day."""
        start = self.since(now)
        today = [(ts, kind, action) for ts, kind, action in self.events if start <= ts <= now + timedelta(seconds=5)]

        def count(kind: str) -> int:
            return sum(1 for _, k, _ in today if k == kind)

        out = []
        if count("day.replan.proposed"):
            out.append("re-planned your day")
        elif count("day.plan.proposed"):
            out.append("planned your day")
        waiting = [a for ts, k, a in today if k == "agent.action.proposed" and a not in self.decided and now - ts <= WAITING_FOR]
        if waiting:
            out.append("left one thing waiting for your OK" if len(waiting) == 1 else f"left {say_number(len(waiting))} things waiting for your OK")
        carried = [self.kinds.get(a, "") for _, k, a in today if k == "workspace.action.completed"]
        if carried:
            one, several = ACTION_SAID.get(carried[0], (ONE_DONE, SEVERAL_DONE)) if len(set(carried)) == 1 else (ONE_DONE, SEVERAL_DONE)
            out.append(one if len(carried) == 1 else several.format(n=say_number(len(carried))))
        if n := count("habits.nudge.raised"):
            out.append(f"reminded you {times(n)}")
        if count("day.checkin.raised"):
            out.append("checked in on you")
        if n := count("agent.alert.raised"):
            out.append(f"flagged {say_number(n)} {'thing' if n == 1 else 'things'} for you")
        if n := count("agent.answer.generated"):
            out.append(f"answered {say_number(n)} {'question' if n == 1 else 'questions'}")
        return out
