"""Near you: the optional events of fixtures/demo/nearby.yaml, seeded with the night and never a commitment."""

import asyncio
import json
import re
from datetime import UTC, datetime, time, timedelta

import pytest

from contexts.circle.application.protocol import busy
from contexts.demo import nearby
from contexts.demo.nearby import PREFIX, nearby_events
from contexts.demo.seed import seed
from contexts.demo.story import LOCAL, ROOT, load
from contexts.play.domain.places import PlaceSet

NOW = load().now  # Sunday 2026-10-11 09:30 EDT
IDS = [
    "nearby:board-games:2026-10-11T20:00:00-04:00",
    "nearby:hack-snack:2026-10-11T22:15:00-04:00",
    "nearby:resume-workshop:2026-10-12T12:30:00-04:00",
    "nearby:friends-dinner:2026-10-12T17:15:00-04:00",
    "nearby:cop4533-review:2026-10-12T18:30:00-04:00",
    "nearby:pickup-soccer:2026-10-13T16:30:00-04:00",
    "nearby:food-truck-night:2026-10-13T18:30:00-04:00",
]
HERO = IDS[4]
SHAPE = re.compile(r"until \d{1,2}:\d{2} [AP]M · [^·]+( · [^·]+)+")
ANDRIA = "class:1760100010"


def events(now=NOW):
    return sorted((e for e in nearby_events(now) if e.kind == "calendar_event"), key=lambda e: e.when)


class Without:
    """The memory as it would be had the nearby events never been synced."""

    def __init__(self, memory):
        self.memory = memory

    def items(self, *a, **kw):
        return [i for i in self.memory.items(*a, **kw) if not i.external_id.startswith(PREFIX)]

    def __getattr__(self, name):
        return getattr(self.memory, name)


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    out = asyncio.run(seed(tmp_path_factory.mktemp("nearby")))
    yield out
    out.app.projection.store.close()


def log_lines(app) -> list[dict]:
    return [json.loads(line) for f in sorted((app.settings.data_dir / "events").glob("*.jsonl"))
            for line in f.read_text(encoding="utf-8").splitlines()]


# --- the data ---

def test_the_events_have_the_ids_kind_and_summary_shape_the_calendar_screen_asked_for():
    found = events()
    assert [e.external_id for e in found] == IDS
    for e in found:
        assert (e.type, e.connector, e.kind) == ("connector.item.synced", "calendar", "calendar_event")
        assert e.external_id == f"{PREFIX}{e.external_id.split(':')[1]}:{e.when.isoformat()}"  # the id carries the real start
        assert SHAPE.fullmatch(e.summary), e.summary
    hero = next(e for e in found if e.external_id == HERO)
    assert hero.title == "COP4533 review session (optional)"
    assert hero.summary == "until 8:00 PM · John C. Hitt Library · hosted by Andria · with Maya, Priya · optional · via GroupMe"
    by_id = {e.external_id: e.summary for e in found}
    assert by_id[IDS[3]] == "until 6:15 PM · '63 South · with Riley, Sam, Alex · optional · via GroupMe"
    assert by_id[IDS[1]] == "until 11:00 PM · Engineering II · hosted by Jordan · free · optional · via GroupMe"
    assert sum("· free ·" in s for s in by_id.values()) == 4  # a mix of free and not


def test_andria_posts_the_review_in_the_class_chat_in_the_groupme_item_format():
    msg = next(e for e in nearby_events(NOW) if e.kind == "message")
    assert (msg.connector, msg.external_id, msg.when, msg.ts) == ("groupme", ANDRIA, NOW - timedelta(minutes=1), NOW)
    assert msg.summary == ("Andria in COP4533 class chat: running a review session for quiz 5 (greedy algorithms) monday 6:30pm "
                           "at the library, about 90 min. totally optional. maya and priya are in")
    assert msg.title == "Andria: " + msg.summary.split(": ", 1)[1][:72] and len(msg.title) == 80
    moved = next(e for e in nearby_events(NOW + timedelta(days=2)) if e.kind == "message")
    assert "wednesday 6:30pm" in moved.summary  # worded from the moved start, like the rest of the story


def test_every_start_is_after_now_within_three_days_and_at_the_hour_the_script_says_whenever_the_story_ends():
    script = {e.slug: e for e in nearby.load().events}
    for step in range(0, 4 * 24 * 60, 37):  # every time of day, both sides of midnight
        for now in (NOW + timedelta(minutes=step), (NOW + timedelta(minutes=step)).astimezone(UTC)):
            found = events(now)
            assert len({e.external_id for e in found}) == len(found) == len(IDS)
            assert all(e.ts <= now for e in nearby_events(now))  # nothing is published after the story's end
            for e in found:
                assert now < e.when < now + timedelta(days=3), (now, e.external_id)
                assert e.when.astimezone(LOCAL).time() == script[e.external_id.split(":")[1]].at
            hero = next(e for e in found if ":cop4533-review:" in e.external_id).when.astimezone(LOCAL)
            assert hero.date() == now.astimezone(LOCAL).date() + timedelta(days=1) and hero.time() == time(18, 30)  # tomorrow evening
    late = events(NOW.replace(hour=21))  # tonight's board games have started: they are tomorrow's
    assert "nearby:board-games:2026-10-12T20:00:00-04:00" in [e.external_id for e in late]
    assert "nearby:hack-snack:2026-10-11T22:15:00-04:00" in [e.external_id for e in late]


def test_every_place_is_one_of_the_ucf_pack_and_the_summary_finds_it():
    places = PlaceSet.load(ROOT / "config" / "places")
    script = {e.slug: e for e in nearby.load().events}
    for e in events():
        slug = e.external_id.split(":")[1]
        assert places.locate(f"{e.title} {e.summary}") == f"ucf.{script[slug].place}", e.external_id  # the leaving checklist's lookup
        where = " · ".join(part for part in e.summary.split("·") if "until" not in part)
        assert places.locate(where) == f"ucf.{script[slug].place}", e.external_id  # the Day Run board's
        assert places.route("apartment.desk", f"ucf.{script[slug].place}")[1] > 0  # there is a walk there


def test_no_title_reads_as_an_exam_to_the_planner():
    from contexts.agent.application.planner import is_crunch

    assert not any(is_crunch(e, NOW) for e in nearby_events(NOW))


# --- seeded through the real app ---

def test_the_seed_puts_them_in_memory_where_the_calendar_screen_reads_them(seeded):
    items = seeded.app.memory.items(NOW, NOW + timedelta(days=3), kind="calendar_event")  # GET /memory/items
    near = [i for i in items if i.external_id.startswith(PREFIX)]
    assert [i.external_id for i in near] == IDS and all(i.connector == "calendar" and i.when for i in near)
    assert next(i for i in near if i.external_id == HERO).summary.endswith("hosted by Andria · with Maya, Priya · optional · via GroupMe")
    chat = seeded.app.memory.items(NOW - timedelta(hours=1), NOW + timedelta(seconds=1), kind="message")
    assert [m.external_id for m in chat if m.title.startswith("Andria:")] == [ANDRIA]
    assert seeded.written["connector.item.synced"] == 29  # was 21: seven events and Andria's message


def test_nothing_the_night_derives_mentions_them_and_the_day_ends_as_it_did(seeded):
    lines = log_lines(seeded.app)
    assert not [e["type"] for e in lines if e["type"] != "connector.item.synced" and PREFIX in json.dumps(e)]
    status = [e for e in lines if e["type"] == "day.status.changed"][-1]
    assert status["ts"] == NOW.isoformat() and status["reason"] == "Knight Hacks: mentor check-in starts"
    assert (status["next_event"]["id"], status["next_event"]["leave_by"]) == (
        "outlook:kh-mentor:2026-10-11T09:20:00-04:00", "2026-10-11T09:10:00-04:00")
    assert status["next_block"]["title"] == "Study at John C. Hitt Library"
    assert sum(e["type"] == "day.status.changed" for e in lines) == 27 and sum(e["type"] == "agent.plan.created" for e in lines) == 12


def test_the_day_board_the_leaving_checklist_and_free_busy_are_what_they_would_be_without_them(seeded):
    app = seeded.app
    real, without = app.memory, Without(app.memory)
    moments = [NOW + timedelta(hours=h) for h in (0, 5, 9, 12, 26, 32, 33, 50, 56)]  # before each event, and during the hero

    def picture(memory) -> list:
        app.dayrun.memory = app.qa.memory = memory
        try:
            return [(app.dayrun.board(t).model_dump(mode="json"), app.qa.planner.leaving(t).text) for t in moments]
        finally:
            app.dayrun.memory = app.qa.memory = real

    assert picture(real) == picture(without)
    board = app.dayrun.board(NOW)
    assert not [x.id for x in [*board.blocks, *board.events] if x.id.startswith(PREFIX)]
    assert "Board game" not in app.qa.planner.leaving(NOW + timedelta(hours=9)).text  # 6:30 PM: it would have been "next"

    for e in events():  # a friend's Atlas asking about that hour still hears "free"
        end = e.when + timedelta(minutes=45)
        assert busy(real, e.when, end) == busy(without, e.when, end), e.external_id
    hero = next(e for e in events() if e.external_id == HERO)
    assert not busy(real, hero.when, hero.when + timedelta(minutes=90))
    assert busy(real, NOW + timedelta(hours=3), NOW + timedelta(hours=4))  # judging and demos is still a commitment

    assert not [k for k in app.habits.calendar if PREFIX in k]  # nothing to be "leave ready" for
    assert any("outlook:kh-mentor" in k for k in app.habits.calendar)


def test_seeding_twice_adds_none_of_them_again(seeded):
    app = seeded.app
    before = len(log_lines(app))
    app.projection.store.close()
    again = asyncio.run(seed(app.settings.data_dir, force=True))
    try:
        assert len(log_lines(again.app)) == before and not again.written
        near = [i for i in again.app.memory.items(NOW, NOW + timedelta(days=3)) if i.external_id.startswith(PREFIX)]
        assert [i.external_id for i in near] == IDS
    finally:
        again.app.projection.store.close()
