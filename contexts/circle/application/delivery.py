"""After the Circle agrees (G1 step 4): create the event and tell the group.

On circle.event.agreed the proposer's Atlas proposes two actions, each behind its
own approval card: the calendar invite (calendar.create_event, with every
attendee) and a group chat message (groupme.post) with the time and a link to
the invite file. Only writes the connectors actually declare are proposed, so
a box without GroupMe just proposes the invite.
"""

from zoneinfo import ZoneInfo

from contexts.connectors.application.hub import ConnectorHub
from contracts.agent import ActionProposed
from contracts.circle import EventAgreed

CALENDAR_WRITE = "calendar.create_event"
CHAT_WRITE = "groupme.post"


class AgreedEventDelivery:
    def __init__(self, hub: ConnectorHub, me: str, tz: ZoneInfo, emails: dict[str, str] | None = None,
                 chat_group: str = "friends", public_url: str | None = None) -> None:
        self.hub = hub
        self.me = me
        self.tz = tz
        self.emails = emails or {}  # member id -> email for the invite's attendees (config)
        self.chat_group = chat_group
        self.public_url = public_url  # where the gateway serves /api/invites/<file>

    def on_agreed(self, e: EventAgreed) -> list[ActionProposed]:
        ev = e.event
        when = ev.start.astimezone(self.tz).strftime("%a %b %d, %I:%M %p").replace(" 0", " ")
        out = []
        if self.hub.is_write(CALENDAR_WRITE):
            attendees = [self.emails.get(m, m) for m in e.attendees if m != self.me]
            out.append(self.hub.propose_write(
                CALENDAR_WRITE,
                {"title": ev.title, "start": ev.start.isoformat(), "end": ev.end.isoformat(),
                 "location": ev.location, "attendees": attendees},
                f"Everyone in the circle said yes: add \"{ev.title}\" ({when}) to the calendar and invite them",
                e.ts,
            ))
        if self.hub.is_write(CHAT_WRITE):
            link = f" Invites: {self.public_url}/api/invites" if self.public_url else ""
            place = f" at {ev.location}" if ev.location else ""
            out.append(self.hub.propose_write(
                CHAT_WRITE,
                {"text": f"{ev.title}: {when}{place}. Everyone's in.{link}", "group": self.chat_group},
                f"Tell the group chat that \"{ev.title}\" is set",
                e.ts,
            ))
        return out
