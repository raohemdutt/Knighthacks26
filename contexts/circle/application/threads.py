"""The agents' conversation, as the phone reads it: one thread per plan, one plain line per message.

    threads = Threads()
    threads.subscribe(bus)                       # replay=True: rebuilt from the log
    view = threads.view("team", "hemdutt", names, members, now, tz, extra=store_rows)

What this box's log holds about a negotiation (contexts/circle/application/protocol.py):

- its own proposals: `agent.action.proposed` (kind circle_share, params.kind event_proposal) with the thread
  id, then `workspace.action.completed` once the card was approved and the proposal sent;
- every other member's proposals and replies: `circle.message.received`;
- the agreement: `circle.event.agreed`.

It does NOT hold this box's own replies (the protocol posts them straight to the store). They come from the
store rows passed as `extra` when the Circle is reachable; without them an agreed thread still shows this
member as free, because an agreement means every attendee answered free.

A counter-offer the proposer accepts is sent as a new proposal with a new thread id. Here it is one
conversation: a proposal by the same member, with the same title, at exactly the time a reply in an earlier,
unagreed thread suggested, continues that thread.

Lines are written from the structured fields only. A reply carries one word (free / busy / counter) and, for a
counter, a time: so a line never gives a reason. Pure functions of (log state, extra rows, now): no I/O here.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo
from typing import Literal

from contracts.agent import ActionProposed
from contracts.circle import EventAgreed, MessageReceived, PlaceShared, Proposal, Reply
from contracts.events import Event
from contracts.workspace import ActionCompleted, ApprovalRequested

MAX_THREADS = 30
CARD_TTL = timedelta(minutes=5)  # an approval card's life when its workspace.approval.requested was not seen
State = Literal["proposed", "waiting", "countered", "agreed", "declined"]


def display(member_id: str, names: dict[str, str]) -> str:
    return names.get(member_id) or member_id.replace("_", " ").replace("-", " ").title()


# --- plain language, from the structured fields only ---


def clock_time(t: datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def day_word(t: datetime, now: datetime) -> str:
    days = (t.date() - now.date()).days
    return {0: "today", 1: "tomorrow", -1: "yesterday"}.get(days) or f"{t:%a %b} {t.day}"


def span(start: datetime, end: datetime) -> str:
    """"2:00 to 4:00 PM", or both halves of the day when it crosses noon or midnight."""
    if start.date() == end.date() and start.strftime("%p") == end.strftime("%p"):
        return f"{start.strftime('%I:%M').lstrip('0')} to {clock_time(end)}"
    return f"{clock_time(start)} to {clock_time(end)}"


def _all(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def proposal_line(who: str, p: Proposal, asked: list[str], everyone: bool, now: datetime, tz: tzinfo,
                  follow_up: bool = False) -> str:
    start, end = p.start.astimezone(tz), p.end.astimezone(tz)
    when = f"{day_word(start, now.astimezone(tz))} {span(start, end)}"
    audience = "everyone" if everyone or not asked else _all(asked)
    if follow_up:
        return f"{who}'s Atlas: Can {audience} do {when} instead?"
    where = f" at {p.location}" if p.location else ""
    return f"{who}'s Atlas: Can {audience} do {p.title}, {when}{where}?"


def reply_line(who: str, r: Reply, asked: Proposal | None, now: datetime, tz: tzinfo) -> str:
    """One word, and for a counter the suggested time. Never a reason: the reply does not carry one."""
    if r.answer == "free":
        return f"{who}'s Atlas: Free."
    if r.answer == "counter" and r.counter is not None:
        start = r.counter.start.astimezone(tz)
        same_day = asked is not None and asked.start.astimezone(tz).date() == start.date()
        when = clock_time(start) if same_day else f"{day_word(start, now.astimezone(tz))} {clock_time(start)}"
        return f"{who}'s Atlas: Busy then. Suggests {when}."
    return f"{who}'s Atlas: Busy then."


def agreed_line(e: EventAgreed, now: datetime, tz: tzinfo) -> str:
    start = e.event.start.astimezone(tz)
    where = f", {e.event.location}" if e.event.location else ""
    n = len(e.attendees)
    return f"Agreed: {day_word(start, now.astimezone(tz))} {clock_time(start)}{where}, {n} of {n} in."


# --- the projection ---


@dataclass
class Card:
    """This box's own proposal, from the approval card to the send."""

    action_id: str
    thread_id: str
    proposal: Proposal
    ts: datetime
    expires_at: datetime
    status: str = "proposed"  # contracts.workspace.ActionStatus
    done_at: datetime | None = None


@dataclass
class _Msg:
    id: str
    thread_id: str
    from_member: str
    kind: str  # proposal | reply
    at: datetime
    proposal: Proposal | None = None
    reply: Reply | None = None
    source: str = "log"  # log | store | agreement


@dataclass
class _Leg:
    """One store thread: a proposal and the replies to it."""

    thread_id: str
    proposal: _Msg | None = None
    pending: Card | None = None  # my card for it, not approved yet
    replies: dict[str, _Msg] = field(default_factory=dict)  # member -> their first reply
    parent: str | None = None
    proposer: str = ""  # who sent the proposal; for a card that is not approved yet, this box's member

    @property
    def asked(self) -> Proposal:
        return self.proposal.proposal if self.proposal else self.pending.proposal

    @property
    def started(self) -> datetime:
        return self.proposal.at if self.proposal else self.pending.ts


class Threads:
    """Rebuilt from the log. `view` is a pure read; nothing here sends or writes."""

    def __init__(self) -> None:
        self.cards: dict[str, Card] = {}  # action id -> my proposal card
        self.received: dict[str, MessageReceived] = {}  # message id -> a friend's message
        self.agreed: dict[str, EventAgreed] = {}  # thread id -> the first agreement

    def subscribe(self, bus) -> None:
        bus.subscribe("agent.action.proposed", self.on_proposed, replay=True)
        bus.subscribe("workspace.approval.requested", self.on_requested, replay=True)
        bus.subscribe("workspace.action.completed", self.on_completed, replay=True)
        bus.subscribe("circle.message.received", self.on_message, replay=True)
        bus.subscribe("circle.event.agreed", self.on_agreed, replay=True)

    def on_proposed(self, e: ActionProposed) -> None:
        if e.kind != "circle_share" or e.params.get("kind") != "event_proposal" or e.action_id in self.cards:
            return
        try:
            proposal = Proposal.model_validate(e.params["proposal"])
            thread_id = str(e.params["thread_id"])
        except (KeyError, ValueError):
            return
        self.cards[e.action_id] = Card(e.action_id, thread_id, proposal, e.ts, e.ts + CARD_TTL)

    def on_requested(self, e: ApprovalRequested) -> None:
        if (card := self.cards.get(e.action_id)) is not None:
            card.expires_at = e.expires_at

    def on_completed(self, e: ActionCompleted) -> None:
        card = self.cards.get(e.action_id)
        if card is not None and card.status == "proposed":
            card.status, card.done_at = e.status, e.ts

    def on_message(self, e: MessageReceived) -> None:
        self.received.setdefault(e.message_id, e)

    def on_agreed(self, e: EventAgreed) -> None:
        self.agreed.setdefault(e.thread_id, e)

    # --- read ---

    def view(self, circle_id: str, me: str, names: dict[str, str], members: list[str], now: datetime, tz: tzinfo,
             extra: list | None = None) -> list[dict]:
        """Every conversation, latest activity first, each with its ordered messages.

        `extra` are the store's rows (protocol.AgentMessage) when the Circle could be read: they add what the
        log lacks (this box's own replies, messages from before a lost log) and never replace a logged one."""
        legs = self._legs(circle_id, me, now, extra or [])
        _link(legs, set(self.agreed))
        roots: dict[str, list[_Leg]] = {}
        for leg in sorted(legs.values(), key=lambda g: (g.started, g.thread_id)):
            root, seen = leg, {leg.thread_id}
            while root.parent and root.parent not in seen:
                root = legs[root.parent]
                seen.add(root.thread_id)
            roots.setdefault(root.thread_id, []).append(leg)
        out = [self._conversation(circle_id, me, names, members, now, tz, chain) for chain in roots.values()]
        out.sort(key=lambda t: (t["last_activity"], t["thread_id"]), reverse=True)
        return out[:MAX_THREADS]

    def _legs(self, circle_id: str, me: str, now: datetime, extra: list) -> dict[str, _Leg]:
        legs: dict[str, _Leg] = {}

        def add(m: _Msg) -> None:
            leg = legs.setdefault(m.thread_id, _Leg(m.thread_id))
            if m.kind == "proposal" and m.proposal is not None:
                if leg.proposal is None:
                    leg.proposal = m
            elif m.kind == "reply" and m.reply is not None:
                leg.replies.setdefault(m.from_member, m)

        pending: list[Card] = []
        for c in self.cards.values():
            if c.status == "done":
                add(_Msg(f"card:{c.action_id}", c.thread_id, me, "proposal", c.done_at or c.ts, proposal=c.proposal))
            elif c.status == "proposed" and now < c.expires_at:
                pending.append(c)
        for e in sorted(self.received.values(), key=lambda e: (e.ts, e.message_id)):
            if e.circle_id == circle_id:
                add(_Msg(e.message_id, e.thread_id, e.from_member, e.kind, e.ts, e.proposal, e.reply))
        for m in sorted(extra, key=lambda m: (m.created_at, m.id)):
            if m.circle_id == circle_id and m.id not in self.received:
                add(_Msg(m.id, m.thread_id, m.from_member, m.kind, m.created_at, m.proposal, m.reply, source="store"))
        for c in pending:
            leg = legs.setdefault(c.thread_id, _Leg(c.thread_id))
            if leg.proposal is None:
                leg.pending = c
        legs = {t: leg for t, leg in legs.items() if leg.proposal is not None or leg.pending is not None}
        for leg in legs.values():
            leg.proposer = leg.proposal.from_member if leg.proposal else me
        return legs

    def _conversation(self, circle_id: str, me: str, names: dict[str, str], members: list[str], now: datetime,
                      tz: tzinfo, chain: list[_Leg]) -> dict:
        def who(m: str) -> dict:
            return {"member_id": m, "display_name": display(m, names), "mine": m == me}

        first, last = chain[0], chain[-1]
        proposer = first.proposer
        messages: list[dict] = []
        agreement: EventAgreed | None = None

        def say(id_: str, leg: _Leg, member: str | None, kind: str, at: datetime, line: str, answer: str | None = None,
                p: Proposal | None = None, source: str = "log") -> None:
            messages.append({
                "id": id_, "thread_id": leg.thread_id, "kind": kind, "answer": answer, "at": at.isoformat(), "line": line,
                "from_member": member, "display_name": display(member, names) if member else None,
                "mine": member == me, "starts_at": p.start.isoformat() if p else None,
                "ends_at": p.end.isoformat() if p else None, "location": (p.location or None) if p else None,
                "source": source})

        invited: set[str] = set()
        for n, leg in enumerate(chain):
            asked = leg.asked
            invited = _invitees(asked, leg.proposer, members, leg.replies)
            everyone = not asked.invitees or (bool(members) and invited >= set(members) - {leg.proposer})
            line = proposal_line(display(leg.proposer, names), asked, [display(m, names) for m in sorted(invited)],
                                 everyone, now, tz, follow_up=n > 0)
            if leg.proposal is None:  # my card, not approved: nothing was sent
                say(f"card:{leg.pending.action_id}", leg, me, "note", leg.pending.ts,
                    f"Not sent yet. {line.split(': ', 1)[1]} is waiting for {display(me, names)}'s approval.", p=asked)
                continue
            say(leg.proposal.id, leg, leg.proposer, "proposal", leg.proposal.at, line, p=asked, source=leg.proposal.source)
            for r in sorted(leg.replies.values(), key=lambda r: (r.at, r.from_member)):
                say(r.id, leg, r.from_member, "reply", r.at, reply_line(display(r.from_member, names), r.reply, asked, now, tz),
                    answer=r.reply.answer, p=r.reply.counter if r.reply.answer == "counter" else None, source=r.source)
            e = self.agreed.get(leg.thread_id)
            if e is not None and e.circle_id == circle_id:
                agreement = e
                # agreement means every attendee answered free: show the ones whose reply this box did not see
                for m in sorted(set(e.attendees) - set(leg.replies) - {leg.proposer}):
                    say(f"agreed:{leg.thread_id}:{m}", leg, m, "reply", e.ts, reply_line(display(m, names), Reply(answer="free"), asked, now, tz),
                        answer="free", source="agreement")
                say(f"agreed:{leg.thread_id}", leg, None, "agreed", e.ts, agreed_line(e, now, tz), p=e.event)

        answers = {m: r.reply.answer for m, r in last.replies.items() if m in invited}
        if agreement is not None:
            answers = {m: "free" for m in agreement.attendees if m != proposer}
            invited = set(answers)
        waiting_on = sorted(invited - set(answers)) if last.proposal is not None else []
        counter = next((r.reply.counter for r in sorted(last.replies.values(), key=lambda r: (r.at, r.from_member))
                        if r.from_member in invited and r.reply.answer == "counter" and r.reply.counter), None)

        if agreement is not None:
            state: State = "agreed"
        elif last.proposal is None:
            state = "proposed" if len(chain) == 1 else "countered"
        elif "busy" in answers.values():
            state = "declined"
        elif "counter" in answers.values():
            state = "countered"
        else:
            state = "waiting"

        shown = agreement.event if agreement is not None else last.asked
        mine = proposer == me
        pending = last.pending
        return {
            "thread_id": first.thread_id, "legs": [g.thread_id for g in chain], "title": shown.title,
            "proposer": who(proposer), "mine": mine,
            "starts_at": shown.start.isoformat(), "ends_at": shown.end.isoformat(), "location": shown.location or None,
            "state": state,
            "invitees": [{**who(m), "answer": answers.get(m)} for m in sorted(invited)],
            "waiting_on": [who(m) for m in waiting_on],
            "asked_at": last.proposal.at.isoformat() if last.proposal else None,
            "last_activity": max(m["at"] for m in messages),
            "pending_action_id": pending.action_id if pending else None,
            "action": _action(state, mine, pending, last, counter, waiting_on, names, now, tz),
            "messages": messages,
        }


def _invitees(p: Proposal, proposer: str, members: list[str], replies: dict) -> set[str]:
    """Who must say free, as protocol.CircleAgent._invitees: the named invitees, else every other member."""
    if p.invitees:
        return set(p.invitees) - {proposer}
    return (set(members) or set(replies)) - {proposer}


def _link(legs: dict[str, _Leg], agreed: set[str]) -> None:
    """A proposal at exactly the time an earlier, unagreed thread's reply suggested, by the same proposer and
    with the same title, continues that thread (protocol.CircleAgent._follow_up gives it a new thread id)."""
    for leg in legs.values():
        p = leg.asked
        best: _Leg | None = None
        for other in legs.values():
            if (other is leg or other.proposal is None or other.proposer != leg.proposer or other.thread_id in agreed
                    or other.asked.title != p.title or other.started > leg.started):
                continue
            if any(r.reply.answer == "counter" and r.reply.counter is not None and r.reply.counter.start == p.start
                   and r.reply.counter.end == p.end and r.at <= leg.started for r in other.replies.values()):
                if best is None or (other.started, other.thread_id) > (best.started, best.thread_id):
                    best = other
        leg.parent = best.thread_id if best else None


def _action(state: str, mine: bool, pending: Card | None, last: _Leg, counter: Proposal | None, waiting_on: list[str],
            names: dict[str, str], now: datetime, tz: tzinfo) -> dict:
    """What this person can do next, in words; `proposal` is what "ask again" would send (through a new card)."""

    def again(p: Proposal) -> dict:
        return {"title": p.title, "start": p.start.isoformat(), "end": p.end.isoformat(), "location": p.location or None,
                "invitees": list(last.asked.invitees)}

    if pending is not None:
        what = "the new time" if state == "countered" else "this plan"
        return {"kind": "approve", "action_id": pending.action_id, "proposal": None,
                "text": f"Approve the card to ask their Atlases about {what}. Nothing is sent until you do."}
    if state == "agreed":
        return {"kind": "none", "action_id": None, "proposal": None, "text": "Everyone is in. It is on the shared calendar."}
    if state == "countered" and mine and counter is not None:
        at = clock_time(counter.start.astimezone(tz))
        return {"kind": "propose", "action_id": None, "proposal": again(counter),
                "text": f"Ask everyone about {at} instead. You approve the card before anything is sent."}
    if state == "declined" and mine:
        return {"kind": "propose", "action_id": None, "proposal": again(last.asked),
                "text": "Someone is busy then and their Atlas found no other time that day. Propose another time."}
    if state in ("countered", "declined"):
        whose = display(last.proposer, names)
        return {"kind": "none", "action_id": None, "proposal": None, "text": f"{whose} decides whether to ask about another time."}
    if waiting_on:
        return {"kind": "none", "action_id": None, "proposal": None,
                "text": "Waiting on " + _all([display(m, names) + "'s Atlas" for m in waiting_on]) + "."}
    return {"kind": "none", "action_id": None, "proposal": None, "text": "Everyone answered free. The proposer's Atlas confirms it next."}


def summary(thread: dict) -> dict:
    """A thread without its messages, plus the latest line: what the list shows."""
    out = {k: v for k, v in thread.items() if k != "messages"}
    out["last_line"] = thread["messages"][-1]["line"]
    out["messages_count"] = len(thread["messages"])
    return out


# --- who is in the Circle, and what this box has seen of them ---


class Members:
    """Rebuilt from the log: which members this box has heard from through a box of their own.

    The Circle's `members.has_box` lives in the shared database; from the log alone a member has a box once
    their Atlas sent a message or a summary, or their desk camera shared a place. Otherwise it is unknown."""

    def __init__(self) -> None:
        self.box_seen: dict[str, str] = {}  # member id -> what showed it: "agent" | "summary" | "desk"
        self.places: dict[str, PlaceShared] = {}

    def subscribe(self, bus) -> None:
        bus.subscribe("circle.message.received", lambda e: self.box_seen.setdefault(e.from_member, "agent"), replay=True)
        bus.subscribe("circle.event.agreed", self.on_agreed, replay=True)
        bus.subscribe("play.league.summary.received", self.on_summary, replay=True)
        bus.subscribe("circle.place.shared", self.on_place, replay=True)

    def on_agreed(self, e: EventAgreed) -> None:
        if e.created_by:
            self.box_seen.setdefault(e.created_by, "agent")

    def on_summary(self, e: Event) -> None:
        if member := getattr(e, "member_id", None):
            self.box_seen.setdefault(member, "summary")

    def on_place(self, e: PlaceShared) -> None:
        prev = self.places.get(e.member_id)
        if prev is None or e.ts >= prev.ts:
            self.places[e.member_id] = e
        if e.source == "desk":
            self.box_seen.setdefault(e.member_id, "desk")

    def view(self, circle_id: str, me: str, names: dict[str, str], members: list[str], now: datetime) -> list[dict]:
        ids = list(dict.fromkeys([me, *members, *self.box_seen, *(m for m, p in self.places.items() if p.circle_id == circle_id)]))
        out = []
        for m in ids:
            p = self.places.get(m)
            here = p is not None and p.circle_id == circle_id and p.expires_at > now
            out.append({
                "member_id": m, "display_name": display(m, names), "mine": m == me,
                "has_box": True if m == me or m in self.box_seen else None,
                "box_seen": "this box" if m == me else self.box_seen.get(m),
                "place": {"place": p.place, "source": p.source, "shared_at": p.ts.isoformat(),
                          "expires_at": p.expires_at.isoformat()} if here else None})
        return out
