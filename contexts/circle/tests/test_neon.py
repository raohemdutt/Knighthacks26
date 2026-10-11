"""NeonCircleStore: unit tests on a fake connection, then the same behaviour checked on
InMemoryCircleStore and (when NEON_DATABASE_URL is set) on the real database."""

from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from contexts.agent.application.memory_port import FakeMemory
from contexts.circle.application.protocol import AgentMessage, CircleAgent, InMemoryCircleStore
from contexts.circle.application.sharing import CircleOffline
from contexts.circle.infrastructure.neon import ENV_URL, CircleBoard, NeonCircleStore, NeonDb
from contexts.circle.tests.neon_helpers import FakeNeon, drop_test_circle, needs_neon, open_test_circle
from contexts.workspace.domain.approvals import ActionRequest
from contracts.circle import EventAgreed, Proposal, Reply

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 10, 12, tzinfo=NY)
STUDY = Proposal(title="Study session", start=NOW + timedelta(days=1), end=NOW + timedelta(days=1, hours=2),
                 location="Library", invitees=["hemdutt"])


def message(circle, n, member="dom", thread="t1", kind="proposal", minutes=0):
    body = {"proposal": STUDY} if kind == "proposal" else {"reply": Reply(answer="free")}
    return AgentMessage(id=f"{circle}-m{n}", circle_id=circle, thread_id=f"{circle}-{thread}",
                        from_member=f"{circle}-{member}", kind=kind, created_at=NOW + timedelta(minutes=minutes), **body)


# --- unit: no network ---


def test_post_creates_the_circle_and_member_rows_it_references():
    neon = FakeNeon()
    NeonCircleStore(NeonDb(neon.connect, names={"c-dom": "Dom"})).post(message("c", 1))
    assert neon.tables() == ["circles", "members", "agent_messages"]
    assert neon.statements[1][1] == ("c-dom", "c", "Dom")
    assert all("ON CONFLICT" in sql for sql, _ in neon.statements)  # safe to repeat
    assert neon.commits == 1


def test_rows_come_back_as_agent_messages():
    row = ("m1", "c", "t1", "dom", "proposal", STUDY.model_dump_json(), None, NOW)  # jsonb as text or as a dict
    reply = ("m2", "c", "t1", "hemdutt", "reply", None, {"answer": "free", "counter": None}, NOW)
    neon = FakeNeon(rows=[row, reply])
    first, second = NeonCircleStore(NeonDb(neon.connect)).since("c", NOW)
    assert first.proposal == STUDY and second.reply.answer == "free"
    sql, params = neon.statements[0]
    assert "created_at > %s" in sql and "ORDER BY created_at" in sql and params == ("c", NOW)


def test_record_event_writes_the_event_and_everyone_in_it_as_one_statement():
    neon = FakeNeon()
    invited = STUDY.model_copy(update={"invitees": ["hemdutt", "ana"]})
    assert NeonCircleStore(NeonDb(neon.connect)).record_event("c", "t1", invited, "dom") is None
    assert neon.commits == 1 and neon.tables() == ["circles", "members", "members", "members"]  # then the one statement
    sql, params = neon.statements[-1]
    assert sql.startswith("WITH new_event AS (INSERT INTO events ") and sql.count("INSERT INTO") == 2
    assert "ON CONFLICT DO NOTHING RETURNING id" in sql and "ON CONFLICT (event_id, member_id) DO NOTHING" in sql
    assert "NOT EXISTS (SELECT 1 FROM rsvps" in sql  # an event left with nobody listed is filled in by a repeat
    assert params == ("t1", "c", "t1", "Study session", invited.start, invited.end, "Library", "dom", "t1", "c",
                      ["ana", "dom", "hemdutt"])


def test_a_dropped_connection_is_reopened_once():
    neon = FakeNeon()
    store = NeonCircleStore(NeonDb(neon.connect))
    store.post(message("c", 1))
    neon.drops = 1
    store.post(message("c", 2))
    assert neon.connects == 2 and neon.commits == 2


def test_unreachable_neon_raises_circle_offline():
    neon = FakeNeon()
    neon.down = True
    with pytest.raises(CircleOffline):
        NeonCircleStore(NeonDb(neon.connect)).thread("c", "t1")
    neon.down, neon.drops = False, 2  # connects, but the link keeps dropping
    with pytest.raises(CircleOffline):
        NeonCircleStore(NeonDb(neon.connect)).thread("c", "t1")


def test_missing_url_is_offline_not_a_crash(monkeypatch):
    monkeypatch.delenv(ENV_URL, raising=False)
    with pytest.raises(CircleOffline):
        NeonCircleStore(NeonDb()).since("c", None)


def test_board_rows_are_dicts_by_column():
    neon = FakeNeon(rows=[("dom", "Dom", "sleep", 4)], columns=["member_id", "display_name", "habit_id", "streak"])
    board = CircleBoard(NeonDb(neon.connect))
    assert board.habit_leaderboard("c") == [{"member_id": "dom", "display_name": "Dom", "habit_id": "sleep", "streak": 4}]
    board.top_scores("c", limit=3)
    assert neon.statements[-1][1] == ("c", 3)


# --- parity: the port behaves the same in memory and on Neon ---


@pytest.fixture(params=["memory", pytest.param("neon", marks=needs_neon)])
def circle_store(request):
    if request.param == "memory":
        yield InMemoryCircleStore(), f"test-{uuid4().hex[:12]}"
        return
    db, circle = open_test_circle()
    try:
        yield NeonCircleStore(db), circle
    finally:
        assert drop_test_circle(db, circle) == 0


def test_post_is_idempotent_on_message_id(circle_store):
    store, c = circle_store
    store.post(message(c, 1))
    store.post(message(c, 1, minutes=5))  # same id again: ignored
    [only] = store.since(c, None)
    assert only.created_at == NOW and only.proposal == STUDY


def test_since_returns_newer_messages_in_time_order(circle_store):
    store, c = circle_store
    for n, minutes in ((1, 0), (2, 1), (3, 2)):
        store.post(message(c, n, minutes=minutes))
    assert [m.id for m in store.since(c, None)] == [f"{c}-m1", f"{c}-m2", f"{c}-m3"]
    assert [m.id for m in store.since(c, NOW)] == [f"{c}-m2", f"{c}-m3"]  # strictly after
    assert store.since(c, NOW + timedelta(minutes=2)) == []
    assert store.since(f"{c}-other", None) == []


def test_thread_holds_the_proposal_and_its_replies(circle_store):
    store, c = circle_store
    store.post(message(c, 1))
    store.post(message(c, 2, member="hemdutt", kind="reply", minutes=1))
    store.post(message(c, 3, thread="t2", minutes=2))
    thread = store.thread(c, f"{c}-t1")
    assert [(m.kind, m.from_member) for m in thread] == [("proposal", f"{c}-dom"), ("reply", f"{c}-hemdutt")]
    assert thread[1].reply == Reply(answer="free") and thread[1].proposal is None


def test_record_event_keeps_the_first_record_for_a_thread(circle_store):
    store, c = circle_store
    invited = STUDY.model_copy(update={"invitees": [f"{c}-hemdutt"]})
    store.record_event(c, f"{c}-t1", invited, f"{c}-dom")
    store.record_event(c, f"{c}-t1", invited.model_copy(update={"title": "Changed"}), f"{c}-dom")
    if isinstance(store, InMemoryCircleStore):
        assert store.events[f"{c}-t1"].title == "Study session"
        return

    def read(cur):
        cur.execute("SELECT title, starts_at, location FROM events WHERE circle_id = %s", (c,))
        events = cur.fetchall()
        cur.execute("SELECT member_id, answer FROM rsvps WHERE event_id = %s ORDER BY member_id", (f"{c}-t1",))
        return events, cur.fetchall()

    events, rsvps = store.db.run(read)
    assert events == [("Study session", invited.start, "Library")]
    assert rsvps == [(f"{c}-dom", "free"), (f"{c}-hemdutt", "free")]


@needs_neon
def test_real_record_event_fills_in_an_event_left_with_nobody_listed():
    """The record the https transport could leave before the event and its RSVPs were one statement. The member
    ids hold what an array parameter must carry unharmed: a comma, quotes, braces, a backslash, non-ASCII."""
    db, c = open_test_circle()
    try:
        store, thread = NeonCircleStore(db), f"{c}-t1"
        awkward = [f"{c}-Lee, Ana", f'{c}-the "boss"', f"{c}-{{curly}}", f"{c}-back\\slash", f"{c}-Zoë 日本"]
        invited = STUDY.model_copy(update={"invitees": awkward})
        everyone = sorted([*awkward, f"{c}-dom"])

        def half_written(cur):  # the event alone, as the first of the two old requests left it
            db.ensure(cur, c, *everyone)
            cur.execute("INSERT INTO events (id, circle_id, thread_id, title, starts_at, ends_at, location, created_by)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (thread, c, thread, "Study session", invited.start, invited.end, "Library", f"{c}-dom"))

        def read(cur):
            cur.execute("SELECT title FROM events WHERE circle_id = %s", (c,))
            titles = [t for (t,) in cur.fetchall()]
            cur.execute("SELECT member_id, answer FROM rsvps WHERE event_id = %s", (thread,))
            return titles, sorted(cur.fetchall())

        db.run(half_written)
        assert db.run(read) == (["Study session"], [])
        [nobody] = CircleBoard(db).upcoming_events(c, invited.start)
        assert nobody["member_id"] is None  # what the calendar showed

        store.record_event(c, thread, invited.model_copy(update={"title": "Changed"}), f"{c}-dom")
        assert db.run(read) == (["Study session"], [(m, "free") for m in everyone])  # listed; the first title stays
        store.record_event(c, thread, STUDY.model_copy(update={"invitees": [f"{c}-someone-else"]}), f"{c}-dom")
        assert db.run(read) == (["Study session"], [(m, "free") for m in everyone])  # a complete record is never added to
        assert len(CircleBoard(db).upcoming_events(c, invited.start)) == len(everyone)
    finally:
        assert drop_test_circle(db, c) == 0


@needs_neon
def test_the_real_driver_reconnects_after_the_connection_is_lost():
    db, c = open_test_circle()
    try:
        store = NeonCircleStore(db)
        store.post(message(c, 1))
        db._conn.close()  # what an idle timeout or a network blip leaves behind
        assert [m.id for m in store.since(c, None)] == [f"{c}-m1"]
    finally:
        assert drop_test_circle(db, c) == 0


class PrefixedIds(NeonCircleStore):
    """CircleAgent makes message ids with uuid4; keep them recognisable as test rows."""

    def post(self, msg):
        super().post(msg.model_copy(update={"id": f"test-{msg.id}"}))


@needs_neon
def test_two_atlases_agree_through_neon():
    db, c = open_test_circle()
    try:
        store, ticks = PrefixedIds(db), iter(range(1, 100))
        clock = lambda: NOW + timedelta(seconds=next(ticks))  # noqa: E731
        dom = CircleAgent(f"{c}-dom", c, store, FakeMemory(items=[]), clock)
        hem = CircleAgent(f"{c}-hemdutt", c, store, FakeMemory(items=[]), clock)
        proposal = STUDY.model_copy(update={"invitees": [f"{c}-hemdutt"]})
        dom.share_executor(ActionRequest(id="a1", kind="circle_share", target=c, reason="Study", expires_at=NOW, params={
            "kind": "event_proposal", "thread_id": f"{c}-t1", "proposal": proposal.model_dump(mode="json")}))

        # hemdutt's box reads the proposal from Neon and replies free; as the only invitee it also sees the agreement
        received, heard = hem.poll()
        assert received.kind == "proposal" and received.proposal == proposal and isinstance(heard, EventAgreed)
        [agreed] = [e for e in dom.poll() if isinstance(e, EventAgreed)]
        assert agreed.attendees == [f"{c}-dom", f"{c}-hemdutt"]
        assert dom.poll() == [] and hem.poll() == []  # nothing new on either side
        assert [m.kind for m in store.thread(c, f"{c}-t1")] == ["proposal", "reply"]
    finally:
        assert drop_test_circle(db, c) == 0
