"""A believable Circle on one box, with no network and no second machine.

    python -m contexts.demo circle --data-dir /tmp/atlas-circle
    # then, exactly as it prints:
    ATLAS_DATA_DIR=/tmp/atlas-circle ATLAS_CONFIG_DIR=/tmp/atlas-circle/config ATLAS_MEMORY_FIXTURE= \
        ATLAS_CONNECTORS=0 python -m contexts.gateway

The demo night (contexts/demo/seed.py) plus three friends' worth of Circle events, all in the log:

- three members with names (Hemdutt on this box, Maya and Dominick), two of them sharing a place;
- "COP4533 exam review": Hemdutt's Atlas asks, Maya's counters an hour later, Hemdutt approves the new
  time, both say free, agreed;
- a meal ("Dinner", or the next mealtime): Maya's Atlas asks everyone, agreed directly;
- "Gym": Hemdutt's Atlas asked ten minutes ago; Dominick is free, Maya's Atlas has not answered.

The committed config never names a member (`circle.member_id` stays unset in config/agent.yaml), so the
Circle settings live in a copy of the config directory made here (`config_dir`), next to the data.

Only what another box, or this box's phone, would have published is written here (messages received, the
phone's approvals, the agreements, places shared); approval requests, completions and the "everyone's in"
alerts are the app's own. Every event here has a fixed id and a time relative to the story's end, so seeding
twice adds nothing. The gaps between a send and its replies are the ones measured between two real boxes
(a free reply about 28 s after approval, the agreement about 57 s).
"""

import shutil
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from contexts.demo.story import LOCAL, ROOT
from contracts.agent import ActionProposed
from contracts.circle import EventAgreed, MessageReceived, PlaceShared, Proposal, Reply
from contracts.events import Event
from contracts.experience import ApprovalDecided

CIRCLE_ID = "demo-circle"
ME = "hemdutt"
NAMES = {"hemdutt": "Hemdutt", "maya": "Maya", "dominick": "Dominick"}
MEAL_AFTER = timedelta(hours=7)
REPLY_S, AGREED_S = 28, 57  # measured between two boxes on Neon: first free reply, and the agreement, after approval


def config_dir(dest: str | Path, base: str | Path = ROOT / "config") -> Path:
    """A copy of the config directory whose agent.yaml joins the demo Circle as Hemdutt. Nothing else changes."""
    dest = Path(dest)
    shutil.copytree(base, dest, dirs_exist_ok=True)
    agent = yaml.safe_load((dest / "agent.yaml").read_text(encoding="utf-8")) or {}
    agent["circle"] = {**(agent.get("circle") or {}), "member_id": ME, "circle_id": CIRCLE_ID, "names": dict(NAMES),
                       "share_presence": True, "share_summaries": False, "emails": {}}
    (dest / "agent.yaml").write_text(
        "# Written by `python -m contexts.demo circle`: config/agent.yaml plus the demo Circle. Not for a real box.\n"
        + yaml.safe_dump(agent, sort_keys=False), encoding="utf-8")
    return dest


def _eid(name: str) -> str:
    return f"demo-circle-{name}"


def _proposal_card(name: str, ts: datetime, p: Proposal, reason: str) -> ActionProposed:
    return ActionProposed(event_id=_eid(f"{name}-card"), ts=ts, action_id=_eid(name), kind="circle_share", target=CIRCLE_ID,
                          reason=reason, params={"kind": "event_proposal", "thread_id": _eid(name), "proposal": p.model_dump(mode="json")})


def _approve(name: str, ts: datetime) -> ApprovalDecided:
    return ApprovalDecided(event_id=_eid(f"{name}-approved"), ts=ts, action_id=_eid(name), decision="approve")


def _reply(thread: str, member: str, ts: datetime, reply: Reply) -> MessageReceived:
    return MessageReceived(event_id=_eid(f"{thread}-{member}-received"), ts=ts, message_id=_eid(f"{thread}-{member}"),
                           circle_id=CIRCLE_ID, thread_id=_eid(thread), from_member=member, kind="reply", reply=reply)


def circle_events(now: datetime, tz: ZoneInfo = LOCAL) -> list[Event]:
    """The Circle's story relative to `now`, oldest first."""
    local = now.astimezone(tz)
    today = local.date()

    def on(days: int, hour: int, minute: int = 0) -> datetime:
        return datetime.combine(today + timedelta(days=days), time(hour, minute), tz)

    def s(seconds: int) -> timedelta:
        return timedelta(seconds=seconds)

    out: list[Event] = []
    friends = ["dominick", "maya"]
    everyone = sorted(NAMES)

    # 1. A meal: Maya's Atlas asks everyone; agreed directly. The first mealtime at least seven hours away: past
    #    the story's own calendar (which fills the hours after `now`) and always inside this box's next 24 hours,
    #    so it shows up as a fixed block of the day.
    meals = [("Lunch", on(0, 12, 30)), ("Dinner", on(0, 18, 30)), ("Breakfast", on(1, 8, 30)), ("Lunch", on(1, 12, 30)),
             ("Dinner", on(1, 18, 30))]
    meal, meal_at = next((m for m in meals if m[1] - local >= MEAL_AFTER), meals[-1])
    plan = Proposal(title=meal, start=meal_at, end=meal_at + timedelta(hours=1), location="the Student Union")
    t = now - timedelta(hours=26)
    out += [
        MessageReceived(event_id=_eid("meal-maya-received"), ts=t, message_id=_eid("meal-maya"), circle_id=CIRCLE_ID,
                        thread_id=_eid("meal"), from_member="maya", kind="proposal", proposal=plan),
        _reply("meal", "dominick", t + s(30), Reply(answer="free")),
        EventAgreed(event_id=_eid("meal-agreed"), ts=t + s(60), circle_id=CIRCLE_ID, thread_id=_eid("meal"),
                    event=plan.model_copy(update={"invitees": ["dominick", ME]}), attendees=everyone, created_by="maya"),
    ]

    # 2. Exam review: this box asks, Maya's Atlas counters an hour later, the new time is approved and agreed.
    study = Proposal(title="COP4533 exam review", start=on(1, 14), end=on(1, 16), location="the Library", invitees=friends)
    later = study.model_copy(update={"start": on(1, 15), "end": on(1, 17)})
    t = now - timedelta(hours=3)
    sent = t + s(12)
    countered = sent + s(AGREED_S)
    resent = countered + s(20)
    suggested = later.start.astimezone(tz).strftime("%a %I:%M %p").replace(" 0", " ")
    out += [
        _proposal_card("study", t, study, "Ask the circle about COP4533 exam review"),
        _approve("study", sent),
        _reply("study", "dominick", sent + s(REPLY_S), Reply(answer="free")),
        _reply("study", "maya", countered, Reply(answer="counter", counter=later.model_copy(update={"invitees": []}))),
        # what protocol.CircleAgent._follow_up raises when a friend counters this box's proposal
        _proposal_card("study-2", countered, later,
                       f"maya is busy for \"{study.title}\" and suggested {suggested}: ask the circle about the new time?"),
        _approve("study-2", resent),
        _reply("study-2", "dominick", resent + s(REPLY_S), Reply(answer="free")),
        _reply("study-2", "maya", resent + s(AGREED_S), Reply(answer="free")),
        EventAgreed(event_id=_eid("study-2-agreed"), ts=resent + s(AGREED_S), circle_id=CIRCLE_ID, thread_id=_eid("study-2"),
                    event=later, attendees=everyone, created_by=ME),
    ]

    # 3. Gym: asked ten minutes ago. Dominick's Atlas answered; Maya's has not.
    gym = Proposal(title="Gym", start=on(2, 10), end=on(2, 11), location="the Gym", invitees=friends)
    t = now - timedelta(minutes=10)
    out += [
        _proposal_card("gym", t, gym, "Ask the circle about Gym"),
        _approve("gym", t + s(10)),
        _reply("gym", "dominick", t + s(10 + REPLY_S), Reply(answer="free")),
    ]

    # Where the friends are: named places only, each with an expiry.
    for member, place, ago_min, hours in (("maya", "Library", 25, 8), ("dominick", "Student Union", 40, 6)):
        ts = now - timedelta(minutes=ago_min)
        out.append(PlaceShared(event_id=_eid(f"place-{member}"), ts=ts, circle_id=CIRCLE_ID, member_id=member, place=place,
                               source="phone", expires_at=ts + timedelta(hours=hours)))
    return sorted(out, key=lambda e: e.ts)
