import asyncio
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.agent.api import digest_router
from contexts.agent.application.digest import MAX_MOMENTS, Digester
from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import QA
from contracts.memory import Episode
from runtime import FakeRuntime

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 11, 18, 0, tzinfo=NY)  # Sunday 6 PM


def ep(id, at, captions=(), camera="cam-desk", labels=("person",), summary=None):
    return Episode(id=id, camera_id=camera, start=at, end=at + timedelta(minutes=4), labels=list(labels),
                   captions=list(captions), summary=summary, keyframe_refs=[f"frames:{camera}:{id}"])


def day(h, m=0, d=11):
    return datetime(2026, 10, d, h, m, tzinfo=NY)


EPISODES = [
    ep("late", day(23, 30, d=10), ["Yesterday, a person turns off the lamp."]),  # Saturday: not today
    ep("e1", day(0, 20), ["A person sits down with a laptop."], labels=("person", "laptop")),  # just after midnight
    ep("e2", day(9, 5), ["Keys dropped on the desk."], labels=("person", "keys")),
    ep("e3", day(9, 40), ["A person picks up the keys."], labels=("person", "keys")),
    ep("e4", day(9, 55), ["Someone drinks coffee."], labels=("person", "cup")),
    ep("e5", day(13, 10), ["A courier leaves a box at the door."], camera="cam-door", labels=("person", "package")),
    ep("e6", day(15, 0), summary="the table is empty; a cup is left by the laptop", labels=("cup", "laptop")),
    ep("future", day(19, 0), ["After now: not in today's digest yet."]),
]


def digester(episodes=EPISODES, now=NOW):
    return Digester(FakeMemory(episodes=episodes), lambda: now.astimezone(ZoneInfo("UTC")), NY)


def test_today_counts_and_reads_in_time_order():
    d = digester().build()
    assert d.day == date(2026, 10, 11)
    assert d.episodes == 6  # midnight local to now; not Saturday night, not after now
    assert d.cameras == {"cam-desk": 5, "cam-door": 1}
    assert d.seen_most[0] == ("cup", 2) or d.seen_most[0] == ("keys", 2) or d.seen_most[0] == ("laptop", 2)
    assert "person" not in dict(d.seen_most)  # counted separately: "People in 5 of them."
    assert d.people_episodes == 5
    assert d.busiest_hour == day(9)
    assert d.text.startswith("Today: 6 moments on 2 cameras, from 12:20 AM to 3:00 PM.")
    assert "Busiest hour: 9 AM (3)." in d.text
    assert "People in 5 of them." in d.text
    # moments: one per active hour, local hour names, the summary when there is one
    assert [m.hour.hour for m in d.moments] == [0, 9, 13, 15]
    assert "9 AM: Keys dropped on the desk." in d.text
    assert "3 PM: The table is empty; a cup is left by the laptop." in d.text
    assert d.text.index("12 AM") < d.text.index("9 AM:") < d.text.index("1 PM:") < d.text.index("3 PM:")


def test_every_episode_is_cited_and_moments_carry_their_frame():
    d = digester().build()
    assert {c.episode_id for c in d.citations} == {"e1", "e2", "e3", "e4", "e5", "e6"}
    assert d.citations[0].episode_id == "e1" and d.citations[0].frame_ref == "frames:cam-desk:e1"
    assert len(d.citations) == len({c.episode_id for c in d.citations})


def test_a_long_day_keeps_the_busiest_hours_and_says_how_many_were_left_out():
    busy = [ep(f"h{h}-{i}", day(h, i * 10), [f"hour {h} moment {i}"]) for h in range(8) for i in range(1 + (h % 3))]
    d = digester(busy, now=day(12)).build()
    assert len(d.moments) == MAX_MOMENTS
    hours = [m.hour.hour for m in d.moments]
    assert hours == sorted(hours)
    assert {2, 5} <= set(hours)  # the three-episode hours are kept
    assert "And 2 quieter hours." in d.text


def test_another_day_and_an_empty_day():
    sat = digester().build(date(2026, 10, 10))
    assert sat.episodes == 1 and sat.text.startswith("Saturday, October 10: 1 moment on 1 camera")
    empty = digester().build(date(2026, 10, 1))
    assert empty.episodes == 0 and empty.text == "Nothing on the cameras on Thursday, October 1." and not empty.citations
    assert digester([]).build().text == "Nothing on the cameras today so far."
    assert digester().build(date(2026, 10, 12)).episodes == 0  # tomorrow has nothing yet


def qa():
    rt = FakeRuntime()
    return QA(rt, FakeMemory(episodes=EPISODES), clock=lambda: NOW.astimezone(ZoneInfo("UTC"))), rt


def test_asking_for_the_digest_answers_from_the_data_without_the_model():
    for q in ("Give me my daily digest", "digest", "Recap my day", "Can you sum up today?", "summary of the day"):
        q_a, rt = qa()
        a = asyncio.run(q_a.ask(q))
        assert a.text.startswith("Today: 6 moments"), q
        assert {c.episode_id for c in a.citations} >= {"e1", "e5"}
        assert not rt.calls, q  # no chat call: a 4B model garbles a list this long


def test_yesterdays_digest():
    q_a, _ = qa()
    a = asyncio.run(q_a.ask("yesterday's digest"))
    assert a.text.startswith("Saturday, October 10: 1 moment")


def test_the_overnight_recap_is_not_taken_over():
    q_a, _ = qa()
    assert q_a.prefetch("Give me a recap of last night").direct is None  # still the recap prompt (SYSTEM_RECAP)
    assert q_a.prefetch("What happened today?").direct is None  # the time-window path, as before


def test_the_endpoint():
    app = FastAPI()
    app.include_router(digest_router(digester()))
    c = TestClient(app)
    r = c.get("/api/digest").json()
    assert r["episodes"] == 6 and r["text"].startswith("Today:") and r["moments"][0]["citation"]["episode_id"] == "e1"
    assert c.get("/api/digest", params={"day": "2026-10-10"}).json()["episodes"] == 1
    assert c.get("/api/digest", params={"day": "yesterday"}).status_code == 422
