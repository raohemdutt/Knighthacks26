"""Plans the Circle agreed, as fixed blocks of the day.

    agreed = AgreedPlans(me="hemdutt")
    bus.subscribe("circle.event.agreed", agreed.on_agreed, replay=True)   # state only: rebuilt from the log
    dayrun.agreed = agreed

Once everyone invited said free, every attendee's box publishes its own `circle.event.agreed`
(contexts/circle/application/protocol.py), so each member's day gets the block from its own log: no calendar
connector, no invite and no network needed. `items` hands them to the board as calendar items, the shape it
already turns into fixed blocks ("until 5:00 PM · the Library").

If the proposer's calendar connector later syncs the invite it created, the same plan would be there twice;
`items` leaves out an agreed plan that a calendar item already covers (same title, same start).
"""

from datetime import datetime, tzinfo

from contracts.circle import EventAgreed
from contracts.memory import Item

CONNECTOR = "circle"
PREFIX = "circle:"  # a block id that came from an agreement: "circle:<thread_id>"


class AgreedPlans:
    def __init__(self, me: str | None = None) -> None:
        self.me = me  # only plans this member is in; None keeps every agreement this box saw
        self.events: dict[str, EventAgreed] = {}  # thread id -> the first agreement

    def on_agreed(self, e: EventAgreed) -> None:
        self.events.setdefault(e.thread_id, e)

    def items(self, lo: datetime, hi: datetime, tz: tzinfo, known: list[Item] | None = None) -> list[Item]:
        """Agreed plans that overlap [lo, hi) as calendar items, soonest first."""
        taken = {(i.title.strip().lower(), i.when.astimezone(tz).replace(second=0, microsecond=0))
                 for i in known or [] if i.kind == "calendar_event" and i.when}
        out = []
        for e in self.events.values():
            ev = e.event
            if self.me is not None and self.me not in e.attendees:
                continue
            if ev.end <= lo or ev.start >= hi:
                continue
            start, end = ev.start.astimezone(tz), ev.end.astimezone(tz)
            if (ev.title.strip().lower(), start.replace(second=0, microsecond=0)) in taken:
                continue
            until = "until " + end.strftime("%I:%M %p").lstrip("0")
            out.append(Item(connector=CONNECTOR, kind="calendar_event", external_id=PREFIX + e.thread_id, title=ev.title,
                            when=start, summary=f"{until} · {ev.location}" if ev.location else until))
        return sorted(out, key=lambda i: (i.when, i.external_id))
