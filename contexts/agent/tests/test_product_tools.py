"""Ask Atlas's tools (product_tools.py): pure functions over fake ports, each with its schema and a dry run."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from contexts.agent.application.product_tools import (
    CALENDAR_WRITE,
    CAMERA_PAUSE,
    TOOLS,
    Ports,
    ToolRegistry,
    parse_when,
)
from contracts.memory import Item

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 11, 12, 15, tzinfo=UTC)  # Sun 8:15 AM in New York


def item(title: str, when: datetime, kind: str = "calendar_event", connector: str = "calendar") -> Item:
    return Item(connector=connector, kind=kind, external_id=title, title=title, when=when, url=None)


ITEMS = [
    item("Theatre Open House", datetime(2026, 10, 11, 15, 0, tzinfo=UTC)),
    item("Demo prep", datetime(2026, 10, 12, 19, 0, tzinfo=UTC)),
    item("Physics homework", datetime(2026, 10, 13, 3, 59, tzinfo=UTC), kind="assignment", connector="canvas"),
    item("A Reddit post", datetime(2026, 10, 11, 14, 0, tzinfo=UTC), kind="post", connector="search"),
]


def ports(**kw) -> Ports:
    base = dict(tz=TZ, items=lambda s, e: [i for i in ITEMS if s <= i.when < e])
    return Ports(**{**base, **kw})


REG = ToolRegistry()


def run(name: str, args: dict | None = None, p: Ports | None = None, dry_run: bool = False):
    return REG.run(name, args or {}, p or ports(), NOW, dry_run=dry_run)


# --- the registry ---

def test_every_tool_has_a_short_description_and_an_object_schema_the_model_can_call():
    names = [s["function"]["name"] for s in REG.schemas()]
    assert names == [t.name.replace(".", "_") for t in TOOLS]
    assert len(set(names)) == len(names) == 17
    for s in REG.schemas():
        assert s["type"] == "function"
        assert s["function"]["parameters"]["type"] == "object"
        assert 10 < len(s["function"]["description"]) <= 140
        assert all(r in s["function"]["parameters"]["properties"] for r in s["function"]["parameters"]["required"])


def test_the_registry_finds_a_tool_by_either_name_and_says_so_for_an_unknown_one():
    assert REG.get("calendar_add") is REG.get("calendar.add")
    assert "calendar_list" in REG and "nope" not in REG
    r = run("nope")
    assert not r.ok and "nope" in r.text


def test_a_port_that_raises_fails_the_tool_not_the_answer():
    def boom(*_):
        raise RuntimeError("store locked")
    r = run("calendar.list", p=ports(items=boom))
    assert not r.ok and r.data["error"] == "store locked"


@pytest.mark.parametrize("tool", [t.name for t in TOOLS])
def test_every_tool_runs_with_no_ports_and_never_proposes_on_a_dry_run(tool):
    args = {"title": "Demo prep", "start": "3 pm tomorrow", "text": "call mom", "at": "6", "label": "keys", "q": "keys",
            "query": "demo", "part": "morning", "screen": "calendar", "action": "pause"}
    for p in (Ports(tz=TZ), ports(can_pause=lambda: True)):
        r = REG.run(tool, args, p, NOW, dry_run=True)
        assert r.proposal is None
        assert r.text


# --- time words ---

@pytest.mark.parametrize("words,expect", [
    ("3 pm tomorrow", datetime(2026, 10, 12, 15, 0)),
    ("tomorrow at 15:00", datetime(2026, 10, 12, 15, 0)),
    ("6", datetime(2026, 10, 11, 18, 0)),  # "call mom at 6": the evening
    ("at 7am", datetime(2026, 10, 12, 7, 0)),  # already past today: tomorrow
    ("at 9am", datetime(2026, 10, 11, 9, 0)),
    ("noon friday", datetime(2026, 10, 16, 12, 0)),
    ("2026-10-14T10:30:00", datetime(2026, 10, 14, 10, 30)),
    ("Add a meeting with Sam at 3 pm tomorrow called demo prep", datetime(2026, 10, 12, 15, 0)),
    ("Set a reminder to call mom at 6", datetime(2026, 10, 11, 18, 0)),
])
def test_parse_when_reads_the_ways_people_and_small_models_write_a_time(words, expect):
    got = parse_when(words, NOW, TZ)
    assert got is not None and got.astimezone(TZ).replace(tzinfo=None) == expect


def test_parse_when_is_none_without_a_time_or_day():
    assert parse_when("", NOW, TZ) is None
    assert parse_when("call mom", NOW, TZ) is None


# --- the calendar ---

def test_calendar_list_today_says_the_owners_words_and_returns_the_events_for_the_card():
    asked = []
    r = run("calendar.list", {"day": "2026-10-11"}, ports(say_calendar=lambda d: asked.append(d) or "Theatre Open House at 11."))
    assert asked == ["today"]  # a date that is today is said as "today"
    assert r.text == "Theatre Open House at 11."
    assert [e["title"] for e in r.data["events"]] == ["Theatre Open House"]  # the Reddit post is not a calendar item
    assert r.link == "/now/"


def test_calendar_list_without_the_fast_lane_says_it_in_one_sentence():
    r = run("calendar.list", {"day": "tomorrow"})
    assert r.text.startswith("You have 2 things:") and "Demo prep" in r.text
    r = run("calendar.list", {"day": "2026-10-20"})
    assert r.text.startswith("Nothing on your calendar") and r.data["events"] == []


def test_calendar_add_proposes_an_approval_card_and_writes_nothing():
    r = run("calendar.add", {"title": "demo prep", "start": "3 pm tomorrow", "attendees": ["Sam"]})
    assert r.ok and r.proposal is not None
    p = r.proposal
    assert (p.kind, p.target) == ("connector_write", CALENDAR_WRITE)
    event = p.params["events"][0]
    assert event["title"] == "demo prep" and event["attendees"] == ["Sam"]
    assert datetime.fromisoformat(event["start"]).astimezone(TZ).hour == 15
    assert datetime.fromisoformat(event["end"]) - datetime.fromisoformat(event["start"]) == timedelta(hours=1)
    assert r.text == "I have proposed demo prep with Sam for your calendar, tomorrow at 3. Approve it on your phone."
    assert r.data["action_id"] == p.action_id and r.link == f"/approvals/?focus={p.action_id}"


def test_calendar_add_is_idempotent_and_its_dry_run_proposes_nothing():
    args = {"title": "demo prep", "start": "3 pm tomorrow"}
    assert run("calendar.add", args).proposal.action_id == run("calendar.add", args).proposal.action_id
    dry = run("calendar.add", args, dry_run=True)
    assert dry.proposal is None and dry.data["dry_run"] and dry.text.startswith("I would propose")


def test_calendar_add_needs_a_title_and_a_time():
    assert not run("calendar.add", {"start": "3 pm"}).ok
    assert not run("calendar.add", {"title": "x"}).ok


def test_calendar_add_says_when_there_is_no_calendar_connector():
    r = run("calendar.add", {"title": "x", "start": "3 pm"}, ports(calendar_writes=lambda: False))
    assert "No calendar connector" in r.proposal.reason


def test_calendar_find_names_the_next_match():
    r = run("calendar.find", {"query": "demo"})
    assert r.text == "Demo prep is tomorrow at 3." and r.data["events"][0]["title"] == "Demo prep"
    assert "couldn't find" in run("calendar.find", {"query": "dentist"}).text


def test_reminders_add_proposes_a_short_calendar_event():
    r = run("reminders.add", {"text": "call mom", "at": "6"})
    event = r.proposal.params["events"][0]
    assert event["title"] == "Reminder: call mom"
    assert datetime.fromisoformat(event["start"]).astimezone(TZ).hour == 18
    assert datetime.fromisoformat(event["end"]) - datetime.fromisoformat(event["start"]) == timedelta(minutes=15)
    assert r.text == "I have proposed a reminder to call mom at 6 tonight. Approve it on your phone."
    assert not run("reminders.add", {"text": "call mom"}).ok


# --- memory ---

def test_memory_tools_read_through_their_ports():
    p = ports(last_seen=lambda label: f"Your {label} was last on the desk, around 7:50." if label == "phone" else None,
              search=lambda q: ["Around 8: a mug on the desk."], recap=lambda s, e: f"{s.hour}-{e.hour}")
    assert run("memory.last_seen", {"label": "my phone"}, p).text.startswith("Your phone was last")
    assert run("memory.last_seen", {"label": "wallet"}, p).text == "I haven't seen your wallet."
    assert run("memory.search", {"q": "mug"}, p).text == "Around 8: a mug on the desk."
    r = run("memory.recap", {"part": "morning"}, p)
    start = datetime.fromisoformat(r.data["start"]).astimezone(TZ)
    end = datetime.fromisoformat(r.data["end"]).astimezone(TZ)
    assert (start.hour, end.hour, end.minute) == (5, 8, 15)  # this morning so far, not from midnight
    assert "isn't evening yet" in run("memory.recap", {"part": "evening"}, p).text


# --- the room, people, the cameras ---

def test_room_map_opens_the_map_and_says_what_is_on_it():
    r = run("room.map", p=ports(room=lambda: {"title": "Dorm room", "cameras": ["cam-door", "cam-desk"], "zones": ["desk", "door"]}))
    assert r.text == "Here is the map of the dorm room: 2 cameras and 2 zones, like desk and door." and r.link == "/map/"
    assert run("room.map").link == "/map/"


def test_who_is_here_says_no_names_unless_the_owner_is_signed_in():
    people = lambda: [{"name": "Ilan", "known": True}, {"name": None, "known": False}]  # noqa: E731
    out = run("people.who_is_here", p=ports(people=people, owner_signed_in=lambda: False))
    assert out.text == "I can see 2 people. Sign in as the owner to hear names." and out.data["names"] == []
    owner = run("people.who_is_here", p=ports(people=people, owner_signed_in=lambda: True))
    assert owner.text == "I can see Ilan and 1 other."


def test_camera_status_and_pause_through_an_approval():
    status = run("camera.status", p=ports(cameras=lambda: {"cameras": ["a", "b"], "paused": False, "running": True}))
    assert status.text == "2 cameras are on."
    assert not run("camera.pause").ok  # no executor wired: points at the Privacy screen
    r = run("camera.pause", {"action": "pause"}, ports(can_pause=lambda: True))
    assert (r.proposal.kind, r.proposal.target, r.proposal.params) == ("device_control", CAMERA_PAUSE, {"resume": False})
    back = run("camera.pause", {"action": "resume"}, ports(can_pause=lambda: True))
    assert back.proposal.params == {"resume": True} and "back on" in back.text


# --- the box, the lab, the Circle, screens ---

def test_box_model_energy_learning_and_the_shared_brain():
    m = run("box.model", p=ports(box_model=lambda: {"version": "atlas-tool-calling-v2", "base": "Qwen3.5-4B"}))
    assert m.text == "The box is serving atlas-tool-calling-v2, built on Qwen3.5-4B."
    assert "don't have a power reading" in run("box.energy_last_answer").text
    e = run("box.energy_last_answer", p=ports(energy=lambda: {"joules": 41.6, "seconds": 3.2, "watts": 13.0}))
    assert e.text == "The last answer took about 42 joules over 3.2 seconds, at about 13 watts."
    lab = run("learning.status", p=ports(learning=lambda: {"run": "L36", "status": "deployed", "score": 0.91}))
    assert lab.text == "The latest lab run is L36, deployed, scoring 0.91."
    assert run("friends.shared_brain", p=ports(shared_brain=lambda: {"enabled": True})).text == "The Circle's shared brain is on."


@pytest.mark.parametrize("screen,path", [("calendar", "/now/"), ("the approvals screen", "/approvals/"),
                                         ("room map", "/map/"), ("learning lab", "/lab/"), ("Box stats page", "/stats/"),
                                         ("inspection", "/spot/"), ("the plant floor", "/spot/")])
def test_navigate_screen_returns_the_apps_path(screen, path):
    r = run("navigate.screen", {"screen": screen})
    assert r.link == path and r.data["path"] == path


def test_navigate_screen_says_so_for_an_unknown_screen():
    assert not run("navigate.screen", {"screen": "kitchen sink"}).ok


# --- Inspection: the plant floor, in the Inspection context's own words ---

SHIFT = {"status": "SPOT has run 27 rounds. 2 things need you. 1 to keep an eye on. Everything else looks normal.",
         "watch": 1, "do_first_total": 2,
         "do_first": [{"headline": "Send SPOT back to Heat exchanger 2 (HX-202)", "sentence": "The alarm is not confirmed. Send SPOT back."},
                      {"headline": "Send a technician to Heat exchanger 4 (HX-204)", "sentence": "The alarm stands."},
                      {"headline": "Send SPOT back to Heat exchanger 13 (HX-213)", "sentence": "No check here has worked for 32 hours."}]}


def test_inspection_status_says_the_status_then_the_first_jobs_in_order_from_the_ports_own_words():
    r = run("inspection.status", p=ports(inspection=lambda: SHIFT))
    assert r.ok and r.link == "/spot/"
    assert r.text == (SHIFT["status"] + " 1. Send SPOT back to Heat exchanger 2 (HX-202). The alarm is not confirmed. Send SPOT back."
                      " 2. Send a technician to Heat exchanger 4 (HX-204). The alarm stands.")
    assert "HX-213" not in r.text and len(r.data["do_first"]) == 3  # the third job is on the screen and in the data
    assert (r.data["watch"], r.data["do_first_total"], r.data["available"]) == (1, 2, True)
    quiet = run("inspection.status", p=ports(inspection=lambda: {"status": "SPOT has run 3 rounds. Nothing needs you right now.", "do_first": []}))
    assert quiet.text == "SPOT has run 3 rounds. Nothing needs you right now."
    assert run("inspection.status", p=ports(inspection=lambda: SHIFT), dry_run=True).proposal is None  # read-only


def test_inspection_status_says_it_is_not_here_without_the_port_or_without_a_kit():
    for p in (ports(), ports(inspection=lambda: None)):
        r = run("inspection.status", p=p)
        assert not r.ok and r.text == "Inspection isn't set up on this box." and r.data == {"available": False}
