"""GET /api/circle/events and /api/circle/sharing on fake stores: no network, no real database."""

import asyncio
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus.bus import EventBus
from contexts.circle.api import circle_router
from contexts.circle.application.calendar import AgreedEvents, from_rows, upcoming
from contexts.circle.application.presence import PresenceBoard
from contexts.circle.application.sharing import CircleSharing
from contexts.circle.infrastructure.neon import CircleBoard, NeonDb, NeonShares
from contexts.circle.infrastructure.outbox import CircleOutbox
from contexts.circle.tests.neon_helpers import FakeNeon
from contracts.circle import EventAgreed, PlaceShared, Proposal
from contracts.habits import DayClosed
from contracts.play import RoundScored

NOW = datetime(2026, 10, 10, 16, tzinfo=UTC)
ME = SimpleNamespace(circle_id="team", member_id="hemdutt")
EVENT_COLUMNS = ["id", "thread_id", "title", "starts_at", "ends_at", "location", "created_by", "member_id",
                 "display_name", "answer"]


def agreed(thread="t1", title="Study session", hours=24, circle="team", location="Library", created_by=None):
    start = NOW + timedelta(hours=hours)
    return EventAgreed(ts=NOW, circle_id=circle, thread_id=thread, attendees=["dom", "hemdutt"], created_by=created_by,
                       event=Proposal(title=title, start=start, end=start + timedelta(hours=2), location=location,
                                      invitees=["dom"]))


def row(event="t9", title="Dinner", hours=3, member="dom", name="Dom", answer="free"):
    start = NOW + timedelta(hours=hours)
    return (event, event, title, start, start + timedelta(hours=1), "Student Union", "dom", member, name, answer)


def app(tmp_path, **kw):
    bus = EventBus(tmp_path / "events")
    api = FastAPI()
    api.include_router(circle_router(bus, clock=lambda: NOW, **kw))
    return bus, TestClient(api)


def board(neon):
    return CircleBoard(NeonDb(neon.connect))


# --- the shared calendar ---


def test_without_a_circle_the_calendar_is_empty_and_says_so(tmp_path):
    _, c = app(tmp_path)
    assert c.get("/api/circle/events").json() == {"configured": False, "circle_id": None, "source": "none",
                                                  "offline": False, "events": []}


def test_without_neon_the_calendar_is_what_this_box_saw_agreed(tmp_path):
    bus, c = app(tmp_path, circle=ME)
    for e in [agreed("late", "Hackathon", hours=48), agreed("soon", "Study session", hours=2),
              agreed("past", "Yesterday's lunch", hours=-30), agreed("other", "Not ours", circle="elsewhere")]:
        asyncio.run(bus.publish(e))
    body = c.get("/api/circle/events").json()
    assert (body["configured"], body["source"], body["offline"]) == (True, "local", False)
    assert [e["title"] for e in body["events"]] == ["Study session", "Hackathon"]  # soonest first, upcoming, ours
    first = body["events"][0]
    assert first["starts_at"] == "2026-10-10T18:00:00+00:00" and first["location"] == "Library"
    assert first["attendees"] == [{"member_id": "dom", "display_name": "dom", "answer": "free"},
                                  {"member_id": "hemdutt", "display_name": "hemdutt", "answer": "free"}]


def test_the_local_calendar_shows_configured_names_and_who_proposed(tmp_path):
    (tmp_path / "agent.yaml").write_text("circle:\n  circle_id: team\n  member_id: hemdutt\n  names:\n    hemdutt: Hemdutt Rao\n")
    bus, c = app(tmp_path, circle=ME, config_dir=tmp_path)
    asyncio.run(bus.publish(agreed(created_by="dom")))
    [ev] = c.get("/api/circle/events").json()["events"]
    assert ev["created_by"] == "dom"  # carried by circle.event.agreed
    assert ev["attendees"] == [{"member_id": "dom", "display_name": "dom", "answer": "free"},  # no name configured
                               {"member_id": "hemdutt", "display_name": "Hemdutt Rao", "answer": "free"}]


def test_the_local_calendar_is_rebuilt_from_the_log_after_a_restart(tmp_path):
    bus, _ = app(tmp_path, circle=ME)
    asyncio.run(bus.publish(agreed()))
    bus, c = app(tmp_path, circle=ME)  # a new process on the same log
    assert c.get("/api/circle/events").json()["events"] == []
    asyncio.run(bus.replay_into())
    assert [e["title"] for e in c.get("/api/circle/events").json()["events"]] == ["Study session"]


def test_neon_rows_fold_into_events_with_everyones_answer(tmp_path):
    neon = FakeNeon(columns=EVENT_COLUMNS, rows=[row(), row(member="hemdutt", name="Hemdutt", answer="counter"),
                                                 row("t10", "Gym", hours=30, member=None, name=None, answer=None)])
    bus, c = app(tmp_path, circle=ME, board=board(neon))
    asyncio.run(bus.publish(agreed("t9", "Dinner (as this box saw it)", hours=3)))  # same thread: Neon's row wins
    asyncio.run(bus.publish(agreed("t11", "Only on this box", hours=5)))
    body = c.get("/api/circle/events").json()
    assert (body["source"], body["offline"]) == ("neon", False)
    assert [e["title"] for e in body["events"]] == ["Dinner", "Only on this box", "Gym"]
    assert body["events"][0]["attendees"] == [{"member_id": "dom", "display_name": "Dom", "answer": "free"},
                                              {"member_id": "hemdutt", "display_name": "Hemdutt", "answer": "counter"}]
    assert body["events"][2]["attendees"] == []  # an event nobody has an RSVP row for
    sql, params = neon.statements[0]
    assert "e.ends_at > %s" in sql and "ORDER BY e.starts_at" in sql and params == ("team", NOW)


def test_neon_offline_falls_back_to_this_boxs_events_without_an_error(tmp_path):
    neon = FakeNeon(columns=EVENT_COLUMNS)
    neon.down = True
    bus, c = app(tmp_path, circle=ME, board=board(neon))
    asyncio.run(bus.publish(agreed()))
    for _ in range(2):  # the second read skips Neon: the first one already found it down
        r = c.get("/api/circle/events")
        assert r.status_code == 200
        assert (r.json()["source"], r.json()["offline"]) == ("local", True)
        assert [e["title"] for e in r.json()["events"]] == ["Study session"]
    assert neon.connects == 0


def test_a_refused_read_is_not_an_error_either():
    class Broken:
        def upcoming_events(self, circle_id, now):
            raise RuntimeError('relation "events" does not exist')

    out = upcoming("team", NOW, AgreedEvents(), Broken())
    assert out == {"circle_id": "team", "source": "local", "offline": True, "events": []}


def test_times_without_a_zone_are_read_as_utc():
    naive = datetime(2026, 10, 11, 9)
    [ev] = from_rows([dict(zip(EVENT_COLUMNS, ("e", None, "Run", naive, "2026-10-11T10:00:00Z", "", "dom", None, None, None),
                               strict=True))])
    assert (ev["starts_at"], ev["ends_at"], ev["location"]) == ("2026-10-11T09:00:00+00:00", "2026-10-11T10:00:00+00:00", None)


# --- what leaves the box ---


def test_without_a_circle_nothing_is_shared(tmp_path):
    _, c = app(tmp_path)
    body = c.get("/api/circle/sharing").json()
    assert (body["configured"], body["sharing_now"], body["database"]) == (False, False, "not_configured")
    assert (body["queued"], body["sent"], body["last_synced_at"], body["place"]) == (0, 0, None, None)
    assert {k["kind"]: k["on"] for k in body["kinds"]} == {"habit_summary": False, "impact_count": False,
                                                          "game_score": False, "presence": False, "event_plan": False}


def test_a_circle_without_neon_reports_the_setting_but_shares_nothing(tmp_path):
    (tmp_path / "agent.yaml").write_text("circle:\n  member_id: hemdutt\n  circle_id: team\n  share_summaries: true\n")
    _, c = app(tmp_path, circle=ME, config_dir=tmp_path,
               presence=PresenceBoard("team", "hemdutt", ["Library"], opted_in=True))
    body = c.get("/api/circle/sharing").json()
    assert (body["configured"], body["circle_id"], body["member_id"]) == (True, "team", "hemdutt")
    assert (body["share_summaries"], body["share_presence"], body["sharing_now"]) == (True, True, False)
    assert body["database"] == "not_configured" and not any(k["on"] for k in body["kinds"])


def test_queued_and_sent_counts_follow_the_outbox(tmp_path):
    neon = FakeNeon()
    db = NeonDb(neon.connect, retry_s=0)  # no wait between attempts: the test brings Neon back at once
    sharing = CircleSharing(CircleOutbox(tmp_path / "circle.sqlite"), NeonShares(db), "team", "hemdutt", opted_in=True)
    presence = PresenceBoard("team", "hemdutt", ["Library"], opted_in=False)
    _, c = app(tmp_path, circle=ME, sharing=sharing, presence=presence, board=CircleBoard(db))

    assert c.get("/api/circle/sharing").json()["database"] == "unknown"  # nothing has tried Neon yet
    sharing.on_day_closed(DayClosed(ts=NOW, habit_id="sleep", day=date(2026, 10, 9), met=True, value=7.5, streak=3))
    sharing.on_round_scored(RoundScored(ts=NOW, round_id="r1", session_id="s1", players=2, item="can", correct=True,
                                        seconds=6.5, path_len=7, score=88.0))
    body = c.get("/api/circle/sharing").json()
    assert (body["sharing_now"], body["queued"], body["sent"], body["last_synced_at"]) == (True, 2, 0, None)
    assert body["oldest_queued_at"] == "2026-10-10T16:00:00.000000+00:00"

    neon.down = True
    assert sharing.sync(NOW + timedelta(minutes=1)) == []
    body = c.get("/api/circle/sharing").json()
    assert (body["database"], body["queued"], body["sent"]) == ("offline", 2, 0)
    assert "unreachable" in body["database_note"]

    neon.down = False
    assert len(sharing.sync(NOW + timedelta(minutes=5))) == 2
    body = c.get("/api/circle/sharing").json()
    assert (body["database"], body["database_note"], body["queued"], body["sent"]) == ("connected", None, 0, 2)
    assert body["last_synced_at"] == "2026-10-10T16:05:00.000000+00:00" and body["last_reached_at"]
    kinds = {k["kind"]: k for k in body["kinds"]}
    assert kinds["habit_summary"] == {"kind": "habit_summary", "on": True, "queued": 0, "sent": 1,
                                      "last_synced_at": "2026-10-10T16:05:00.000000+00:00"}
    assert (kinds["impact_count"]["sent"], kinds["game_score"]["sent"]) == (0, 1)
    assert (kinds["presence"]["on"], kinds["event_plan"]["on"]) == (False, True)  # places are off for this member


def test_summaries_off_means_nothing_queues_and_the_status_says_off(tmp_path):
    db = NeonDb(FakeNeon().connect)
    sharing = CircleSharing(CircleOutbox(tmp_path / "circle.sqlite"), NeonShares(db), "team", "hemdutt", opted_in=False)
    _, c = app(tmp_path, circle=ME, sharing=sharing, board=CircleBoard(db))
    sharing.on_day_closed(DayClosed(ts=NOW, habit_id="sleep", day=date(2026, 10, 9), met=True, value=7.5, streak=3))
    body = c.get("/api/circle/sharing").json()
    assert (body["share_summaries"], body["sharing_now"], body["queued"]) == (False, False, 0)
    assert not any(k["on"] for k in body["kinds"] if k["kind"] in ("habit_summary", "impact_count", "game_score"))


def test_the_place_i_am_sharing_shows_until_it_expires(tmp_path):
    presence = PresenceBoard("team", "hemdutt", ["Library"], opted_in=True)
    _, c = app(tmp_path, circle=ME, presence=presence, board=board(FakeNeon()))
    assert c.get("/api/circle/sharing").json()["place"] is None
    presence.on_shared(presence.share("Library", "phone", NOW - timedelta(minutes=10)))
    presence.on_shared(PlaceShared(ts=NOW, circle_id="team", member_id="dom", place="Gym", source="phone",
                                   expires_at=NOW + timedelta(hours=1)))  # a friend's place is not mine
    body = c.get("/api/circle/sharing").json()
    assert body["place"] == {"place": "Library", "source": "phone", "expires_at": "2026-10-10T17:50:00+00:00"}
    assert {k["kind"]: k["on"] for k in body["kinds"]}["presence"] is True
    presence.on_shared(presence.share("Library", "phone", NOW - timedelta(hours=3)))  # older: ignored
    presence.current["hemdutt"] = presence.share("Library", "phone", NOW - timedelta(hours=3))
    assert c.get("/api/circle/sharing").json()["place"] is None  # expired an hour ago


def test_outbox_stats_are_per_kind(tmp_path):
    outbox = CircleOutbox(tmp_path / "circle.sqlite")
    assert outbox.stats() == []
    outbox.enqueue("team", "hemdutt", "game_score", {"score": 1}, NOW, share_id="a")
    outbox.enqueue("team", "hemdutt", "game_score", {"score": 2}, NOW + timedelta(minutes=1), share_id="b")
    outbox.mark_synced("a", NOW + timedelta(minutes=2))
    assert outbox.stats() == [{"kind": "game_score", "queued": 1, "sent": 1,
                               "last_synced_at": "2026-10-10T16:02:00.000000+00:00",
                               "oldest_queued_at": "2026-10-10T16:01:00.000000+00:00"}]


def test_neon_db_remembers_whether_it_last_got_through():
    neon = FakeNeon()
    db = NeonDb(neon.connect, retry_s=0)  # no wait between attempts: the test brings Neon back at once
    assert (db.last_ok, db.offline) == (None, None)
    CircleBoard(db).top_scores("team")
    assert db.last_ok is not None and db.offline is None
    neon.down, neon.drops = True, 1
    try:
        CircleBoard(db).top_scores("team")
    except Exception:
        pass
    assert "unreachable" in db.offline
    neon.down = False
    CircleBoard(db).top_scores("team")
    assert db.offline is None


def test_the_gateway_serves_both_when_the_box_has_no_circle(tmp_path, monkeypatch):
    from contexts.circle.infrastructure.neon import ENV_URL
    from contexts.gateway.app import create_app
    from contexts.gateway.bootstrap import Settings

    monkeypatch.delenv(ENV_URL, raising=False)
    settings = Settings(data_dir=tmp_path, token=None, connectors=False, ntfy_url=None, llm_url=None)
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/circle/events").json()["events"] == []
        body = c.get("/api/circle/sharing").json()
        assert (body["configured"], body["sharing_now"], body["database"]) == (False, False, "not_configured")


# --- starting a plan from the phone ---

PLAN = {"title": "Study session", "start": "2026-10-11T14:00:00-04:00", "end": "2026-10-11T16:00:00-04:00",
        "location": "Library", "invitees": ["dom", "xander"]}


def test_a_proposal_from_the_phone_is_an_approval_card_and_sends_nothing(tmp_path):
    """CircleAgent.propose had no caller outside tests: "find a time for all of us" could not be started."""
    from contexts.agent.application.memory_port import FakeMemory
    from contexts.circle.application.protocol import CircleAgent, InMemoryCircleStore
    from contexts.workspace.domain.approvals import Approvals

    store = InMemoryCircleStore()
    me = CircleAgent("hemdutt", "team", store, FakeMemory(), clock=lambda: NOW)
    bus, c = app(tmp_path, circle=me)
    approvals = Approvals({"circle_share": me.share_executor})
    bus.subscribe("agent.action.proposed", approvals.on_proposed)

    r = c.post("/api/circle/proposals", json=PLAN)
    assert r.status_code == 200 and set(r.json()) == {"action_id", "thread_id"}
    [card] = approvals.pending()
    assert (card.id, card.kind, card.target, card.reason) == (r.json()["action_id"], "circle_share", "team",
                                                              "Ask the circle about Study session")
    assert card.params["kind"] == "event_proposal" and card.params["thread_id"] == r.json()["thread_id"]
    assert Proposal.model_validate(card.params["proposal"]) == Proposal.model_validate(PLAN)
    assert store.messages == []  # nothing has left the box: that waits for the card

    minimal = c.post("/api/circle/proposals", json={k: PLAN[k] for k in ("title", "start", "end")})
    assert minimal.status_code == 200 and approvals.requests[minimal.json()["action_id"]].params["proposal"]["invitees"] == []
    for bad in ({**PLAN, "end": PLAN["start"]}, {**PLAN, "title": "  "}, {**PLAN, "start": "2026-10-11T14:00:00"},
                {k: PLAN[k] for k in ("title", "start")}):
        assert c.post("/api/circle/proposals", json=bad).status_code == 422
    assert len(approvals.pending()) == 2 and store.messages == []


def test_a_proposal_without_a_circle_says_so_and_writes_need_the_token(tmp_path, monkeypatch):
    import shutil

    import yaml

    from contexts.circle.infrastructure.neon import ENV_URL
    from contexts.gateway.app import create_app
    from contexts.gateway.bootstrap import Settings

    monkeypatch.delenv(ENV_URL, raising=False)
    off = Settings(data_dir=tmp_path / "off", token=None, connectors=False, ntfy_url=None, llm_url=None)
    with TestClient(create_app(off)) as c:
        r = c.post("/api/circle/proposals", json=PLAN)
        assert (r.status_code, r.json()["detail"]) == (404, "the Circle is not configured")
        assert c.get("/api/approvals").json() == []

    config = tmp_path / "config"
    shutil.copytree("config", config)
    agent_yaml = yaml.safe_load((config / "agent.yaml").read_text())
    agent_yaml["circle"] = {**(agent_yaml.get("circle") or {}), "member_id": "hemdutt", "circle_id": "team"}
    (config / "agent.yaml").write_text(yaml.safe_dump(agent_yaml))
    on = Settings(data_dir=tmp_path / "on", config_dir=config, token="s3cret", connectors=False, ntfy_url=None, llm_url=None)
    with TestClient(create_app(on)) as c:
        assert c.post("/api/circle/proposals", json=PLAN).status_code == 401
        r = c.post("/api/circle/proposals", json=PLAN, headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200
        [card] = c.get("/api/approvals").json()
        assert (card["id"], card["kind"], card["status"]) == (r.json()["action_id"], "circle_share", "proposed")
        assert c.app.state.atlas.circle.store.messages == []  # not approved, so not sent
