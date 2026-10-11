"""The optional-event demo: an invented day through the real app plays the morning plan, a block start, the
suggestion, a yes behind an approval card, the event as a block of the day and the time to leave; it prints the same
lines for any directory, a gateway on its directory shows the block, and a restart on that log says nothing again."""

import asyncio
import json
import shutil
from datetime import UTC, date, datetime, time

import pytest
from fastapi.testclient import TestClient

from contexts.demo import optional
from contexts.demo.__main__ import main
from contexts.demo.seed import build_app, settle
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.workspace.domain.approvals import ActionRequest

DAY = date(2026, 10, 12)
OFFER = "nearby:friends-dinner:2026-10-12T17:15:00-04:00"
BLOCK = "optional:friends-dinner:2026-10-12T17:15:00-04:00"
FRIENDS = ("Riley", "Sam", "Alex")
APPROVE = "Approve (demo, approved locally: no calendar connector on this box, nothing is written)"
MORNING = ("Morning. Today looks like this: breakfast first, then a long study block before COP4533 lecture at twenty to eleven. Leave by half "
           "past ten for that; it's an eight-minute walk. Project Milestone 1 is due tomorrow, and this plan covers the five hours it still "
           "needs. Want me to keep this as today's plan?")
STARTS = "It's one o'clock: your study block on Project Milestone 1, ninety minutes. You're at the desk already."
SUGGESTS = ("Riley, Sam and Alex are having dinner at '63 South at quarter past five. It's a sixteen-minute walk and it fits before your study "
            "block. Want me to add it?")
WORKED_IN = ("Okay, I've worked Friends' dinner in: first up is Friends' dinner at quarter past five. Leave by five to five for that; it's a "
             "sixteen-minute walk. Want me to switch to that?")
TIME_TO_GO = "Time to go. Friends' dinner starts at quarter past five at '63 South, a sixteen-minute walk from here; leave in five minutes."


def at(hour: int, minute: int = 0) -> datetime:
    return datetime.combine(DAY, time(hour, minute), optional.LOCAL)


@pytest.fixture(scope="module")
def shown(tmp_path_factory):
    out = asyncio.run(optional.show(tmp_path_factory.mktemp("optional"), DAY))
    yield out
    out.app.projection.store.close()


def log(app) -> list:
    return list(app.bus.read_log())


def test_the_day_plays_through_and_every_line_atlas_says_is_the_apps_own(shown):
    assert shown.ok and shown.stopped is None
    said = [(optional.clock(when), who, what) for when, who, what in shown.lines if who != "Plan"]
    assert said == [("8:05 AM", "Atlas", MORNING), ("8:06 AM", "You", APPROVE), ("1:01 PM", "Atlas", STARTS), ("4:00 PM", "Atlas", SUGGESTS),
                    ("4:02 PM", "You", "Add it (yes to Friends' dinner)"), ("4:02 PM", "Atlas", WORKED_IN), ("4:03 PM", "You", APPROVE),
                    ("4:50 PM", "Atlas", TIME_TO_GO)]
    plan = [(optional.clock(when), what) for when, who, what in shown.lines if who == "Plan"]
    assert plan[-1] == ("4:03 PM", "Friends' dinner, 5:15 PM to 6:15 PM at '63 South, is a block of the day")
    assert plan[:-1] and all(when == "4:02 PM" and what.startswith("would change: ") for when, what in plan[:-1])
    # each of those lines is a `day.narrated` the app published: nothing was written by the scene
    told = {e.spoken: e for e in log(shown.app) if e.type == "day.narrated"}
    assert (told[STARTS].moment, told[SUGGESTS].moment, told[WORKED_IN].moment, told[TIME_TO_GO].moment) == ("start", "suggest", "replan", "now")
    assert told[SUGGESTS].ref == OFFER and told[SUGGESTS].ts == at(16) and told[TIME_TO_GO].urgent and told[TIME_TO_GO].ts == at(16, 50)
    assert (shown.started, shown.suggested, shown.leave) == (told[STARTS], told[SUGGESTS], told[TIME_TO_GO])


def test_the_yes_changed_nothing_until_its_card_was_approved(shown):
    events = log(shown.app)
    types = [e.type for e in events]
    assert types.count("day.optional.answered") == 1 and types.count("day.replan.proposed") == 1 and types.count("day.plan.committed") == 2
    answer = next(e for e in events if e.type == "day.optional.answered")
    replan = next(e for e in events if e.type == "day.replan.proposed")
    morning, accepted = [e for e in events if e.type == "day.plan.committed"]
    assert (answer.id, answer.answer, answer.ts, answer.plan_id, answer.action_id) == (OFFER, "yes", at(16, 2), replan.plan_id, replan.action_id)
    assert replan.for_optional == OFFER and replan.asked and [m.origin for m in replan.moves if m.id == BLOCK] == ["optional"]
    assert (morning.ts, accepted.ts, accepted.plan_id) == (at(8, 6), at(16, 3), replan.plan_id)
    assert types.index("day.optional.answered") < types.index("day.replan.proposed") < len(types) - 1 - types[::-1].index("day.plan.committed")
    # between the yes and the approval the day was still measured against the morning plan
    statuses = [e for e in events if e.type == "day.status.changed" and at(16, 2) <= e.ts < at(16, 3)]
    assert all(e.plan_id == morning.plan_id for e in statuses)
    # both approvals were of plan cards on a box that cannot write to a calendar
    cards = [e for e in events if e.type == "agent.action.proposed"]
    assert [(c.kind, c.target) for c in cards] == [("connector_write", "calendar.create_event")] * 2
    assert not shown.app.hub.is_write("calendar.create_event") and all("nothing is written" in c.reason for c in cards)
    decided = [e for e in events if e.type == "experience.approval.decided"]
    assert [(e.decision, e.ts) for e in decided] == [("approve", at(8, 6)), ("approve", at(16, 3))]


def test_a_friend_is_named_only_in_the_calendar_item_and_in_what_is_said_aloud(shown):
    for e in log(shown.app):
        body = e.model_dump(mode="json")
        if e.type == "connector.item.synced" and e.external_id == OFFER:
            continue  # the item as the calendar connector synced it
        if e.type == "day.narrated" and e.moment == "suggest":
            body.pop("spoken")  # said aloud on the owner's own phone
        assert not [n for n in FRIENDS if n in json.dumps(body)], e.type


def test_the_scene_approves_only_a_card_that_stays_on_the_box(shown):
    app = shown.app
    card = ActionRequest(id="a1", kind="connector_write", target="calendar.create_event", reason="r", expires_at=at(16, 8))
    assert optional.stays_on_the_box(app, card)
    assert not optional.stays_on_the_box(app, None)
    assert not optional.stays_on_the_box(app, card.model_copy(update={"status": "expired"}))
    assert not optional.stays_on_the_box(app, card.model_copy(update={"kind": "send_message"}))
    assert not optional.stays_on_the_box(app, card.model_copy(update={"target": "groupme.post_message"}))

    class Writes:
        def is_write(self, tool: str) -> bool:
            return tool == "calendar.create_event"

    hub, app.hub = app.hub, Writes()
    try:
        assert not optional.stays_on_the_box(app, card)  # a box with a calendar connector: the card is the phone's to answer
    finally:
        app.hub = hub


def test_two_runs_into_two_empty_directories_print_the_same_lines(tmp_path, capsys):
    assert main(["optional", "--data-dir", str(tmp_path / "a"), "--day", "2026-10-12"]) == 0
    first = capsys.readouterr()
    assert main(["optional", "--data-dir", str(tmp_path / "b"), "--day", "2026-10-12"]) == 0
    second = capsys.readouterr()
    assert first.out == second.out and first.out.startswith("An invented Monday (2026-10-12), through the real app.")
    for line in (STARTS, SUGGESTS, WORKED_IN, TIME_TO_GO, "Add it (yes to Friends' dinner)", APPROVE):
        assert line in first.out
    assert str(tmp_path) not in first.out  # nothing printed there depends on the directory
    assert str(tmp_path / "a") in first.err and str(tmp_path / "b") in second.err and "python -m contexts.gateway" in first.err
    # the same directory again is refused, and nothing is added to it
    before = sum(1 for _ in (tmp_path / "a" / "events").glob("*.jsonl"))
    assert main(["optional", "--data-dir", str(tmp_path / "a"), "--day", "2026-10-12"]) == 1
    assert "already has an event log" in capsys.readouterr().err and sum(1 for _ in (tmp_path / "a" / "events").glob("*.jsonl")) == before


def test_a_gateway_started_on_the_scenes_directory_shows_the_block(shown, tmp_path):
    shutil.copytree(shown.app.settings.data_dir / "events", tmp_path / "events")
    settings = Settings(data_dir=tmp_path, token=None, connectors=False, ntfy_url=None, llm_url=None, memory_fixture="")
    with TestClient(create_app(settings)) as c:
        plan = c.get("/api/day/plan", params={"day": "2026-10-12"}).json()
        assert plan["committed"] is True and plan["plan_id"] == shown.replan.plan_id
        [block] = [b for b in plan["blocks"] if b["origin"] == "optional"]
        assert (block["id"], block["title"], block["fixed"], block["place"]) == (BLOCK, "Friends' dinner", True, "ucf.dining_63")
        assert (block["start"], block["end"]) == (at(17, 15).isoformat(), at(18, 15).isoformat())
        brief = c.get("/api/day/brief").json()["latest"]
        assert brief["suggest"]["ref"] == OFFER and brief["suggest"]["spoken"] == SUGGESTS and brief["start"]["spoken"] == STARTS
        store = c.app.state.atlas.projection.store
    store.close()


def test_a_restart_on_that_log_knows_the_day_and_says_nothing_again(shown, tmp_path):
    shutil.copytree(shown.app.settings.data_dir / "events", tmp_path / "events")

    async def restarted():
        app = build_app(tmp_path)
        heard: list = []
        app.bus.subscribe("day.", heard.append)
        await app.bus.replay_into()
        for t in (at(13, 2), at(16, 51), at(16, 52)):  # a block start and a suggestion would both still be in their windows
            await app.bus.publish_all(app.day.tick(t))
            await settle()
        return app, heard

    app, heard = asyncio.run(restarted())
    try:
        assert app.day.optional.accepted == {OFFER} and app.day.optional.answers[OFFER].answer == "yes"
        assert app.day.optional.suggested == {OFFER: at(16)} and app.day.planner.current[DAY].plan_id == shown.replan.plan_id
        assert BLOCK in [b.id for b in app.dayrun.board(at(16, 51)).blocks]  # the Day Run board carries it
        assert not [e for e in heard if e.type == "day.narrated" and e.moment in ("suggest", "start", "replan", "morning")]
        status = app.day.tracker.status
        assert status.next_event.id == BLOCK and status.next_event.leave_by == at(16, 59) and status.risk == "leave_now"
    finally:
        app.projection.store.close()


def test_the_default_day_is_today_where_the_box_is():
    assert optional.default_day(datetime(2026, 10, 13, 2, 30, tzinfo=optional.LOCAL)) == date(2026, 10, 13)
    assert optional.default_day(datetime(2026, 10, 13, 3, 30, tzinfo=UTC)) == date(2026, 10, 12)  # still the 12th in New York
