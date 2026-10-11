"""How the box uses the shared database: stopping a place share reaches it, an unsent place waits on disk, a slow
or unreachable database never holds the box up, and a box only cleans up its own rows.

Unit tests run on the in-memory remote and the fake connection. The tests marked `needs_neon` run the same SQL on a
real Postgres when NEON_DATABASE_URL is set, each under its own throwaway `test-…` circle."""

import asyncio
import logging
import threading
from datetime import UTC, date, datetime, timedelta

import pytest

from contexts.agent.application.memory_port import FakeMemory
from contexts.circle.application.presence import PresenceBoard
from contexts.circle.application.presence_sync import InMemoryPresenceRemote, PresenceSync
from contexts.circle.application.protocol import AgentMessage, CircleAgent, InMemoryCircleStore
from contexts.circle.application.sharing import CircleOffline, OutboxRow
from contexts.circle.infrastructure.neon import (
    CircleBoard,
    NeonCircleStore,
    NeonCleanup,
    NeonDb,
    NeonPresence,
    NeonShares,
)
from contexts.circle.infrastructure.neon_sealed import NeonSealedStore
from contexts.circle.infrastructure.outbox import PresenceQueue
from bus import EventBus
from contexts.circle import schema
from contexts.circle.application.sharing import CircleSharing
from contexts.circle.infrastructure.outbox import CircleOutbox
from contexts.circle.tests import face_share_helpers as faces
from contexts.circle.tests.neon_helpers import FakeNeon, drop_test_circle, needs_neon, open_test_circle
from contexts.demo.nearby import nearby_events
from contexts.gateway import bootstrap
from contexts.play.league.infrastructure.neon import NeonLeague
from contracts.circle import PlaceShared, Proposal, SealedMessage
from contracts.play import LeagueStakeSet, Stake

NOW = datetime(2026, 10, 10, 23, tzinfo=UTC)
PLACES = ["Our table", "Library", "Dorm"]


def share(member, place, at=0, hours=2.0, circle="team"):
    ts = NOW + timedelta(minutes=at)
    return PlaceShared(ts=ts, circle_id=circle, member_id=member, place=place, source="phone",
                       expires_at=ts + timedelta(hours=hours))


def stop(prev: PlaceShared, at: int) -> PlaceShared:
    """What DELETE /api/presence publishes: the same place, ending the moment it starts."""
    ts = NOW + timedelta(minutes=at)
    return prev.model_copy(update={"event_id": f"stop-{at}", "ts": ts, "expires_at": ts})


class Flaky(InMemoryPresenceRemote):
    """Neon that can be switched off."""

    def __init__(self):
        super().__init__()
        self.down = False

    def push(self, row):
        if self.down:
            raise CircleOffline("no network")
        super().push(row)

    def active(self, circle_id, now):
        if self.down:
            raise CircleOffline("no network")
        return super().active(circle_id, now)


def box(remote, me, queue=None):
    """One box: the sync and the board its events feed, as the bus wires them."""
    sync, board = PresenceSync(remote, "team", me, queue=queue), PresenceBoard("team", me, PLACES, opted_in=True)

    def tick(at: int) -> list[PlaceShared]:
        out = sync.sync(NOW + timedelta(minutes=at))
        for e in out:
            board.on_shared(e)
        return out

    def publish(e: PlaceShared) -> None:
        board.on_shared(e)
        sync.on_shared(e)

    return sync, board, tick, publish


# --- stop sharing reaches the database, and friends' boxes stop showing the place ---


def test_stop_sharing_reaches_the_remote_and_the_friends_board():
    remote = InMemoryPresenceRemote()
    _, _, dom_tick, dom_publish = box(remote, "dom")
    _, hem_board, hem_tick, _ = box(remote, "hemdutt")

    first = share("dom", "Library")
    dom_publish(first)
    dom_tick(1)
    hem_tick(2)
    assert [p.place for p in hem_board.friends(NOW + timedelta(minutes=2))] == ["Library"]

    dom_publish(stop(first, at=5))
    dom_tick(6)  # sent although it is already over: that is the point of it
    assert remote.active("team", NOW + timedelta(minutes=6)) == []
    [ended] = hem_tick(7)
    assert (ended.member_id, ended.place) == ("dom", "Library") and ended.expires_at <= NOW + timedelta(minutes=7)
    assert hem_board.friends(NOW + timedelta(minutes=7)) == []
    assert hem_tick(8) == []  # said once


def test_a_restarted_box_still_ends_a_place_its_owner_stopped():
    remote = InMemoryPresenceRemote()
    _, _, dom_tick, dom_publish = box(remote, "dom")
    _, _, hem_tick, _ = box(remote, "hemdutt")
    first = share("dom", "Library")
    dom_publish(first)
    dom_tick(1)
    [seen] = hem_tick(2)
    dom_publish(stop(first, at=5))
    dom_tick(6)

    restarted, board, tick, _ = box(remote, "hemdutt")  # a new process: the log replays the place it had shown
    board.on_shared(seen)
    restarted.on_friend(seen)
    assert [p.place for p in board.friends(NOW + timedelta(minutes=7))] == ["Library"]
    assert [e.member_id for e in tick(7)] == ["dom"]
    assert board.friends(NOW + timedelta(minutes=7)) == []


def test_a_place_that_ran_out_by_itself_needs_no_ending():
    remote = InMemoryPresenceRemote()
    _, _, dom_tick, dom_publish = box(remote, "dom")
    _, _, hem_tick, _ = box(remote, "hemdutt")
    dom_publish(share("dom", "Library", hours=1))
    dom_tick(1)
    assert len(hem_tick(2)) == 1
    assert hem_tick(90) == []  # the board already hides it: nothing to say


# --- a place shared while the database is unreachable waits, on disk, and goes out when it is back ---


def test_an_unsent_place_survives_a_restart_and_is_delivered(tmp_path):
    remote, queue = Flaky(), PresenceQueue(tmp_path / "circle.sqlite")
    remote.down = True
    sync, _, tick, publish = box(remote, "dom", queue)
    publish(share("dom", "Library"))
    assert tick(1) == [] and sync.pending is not None

    again, _, tick, _ = box(remote, "dom", PresenceQueue(tmp_path / "circle.sqlite"))  # the box restarted, still offline
    assert again.pending.place == "Library"
    tick(2)
    remote.down = False
    tick(3)
    assert remote.rows["dom"]["place"] == "Library" and again.pending is None
    assert PresenceQueue(tmp_path / "circle.sqlite").load("team", "dom") is None  # delivered: nothing left to send


def test_a_place_that_expired_while_offline_is_never_sent(tmp_path):
    remote, queue = Flaky(), PresenceQueue(tmp_path / "circle.sqlite")
    remote.down = True
    sync, _, tick, publish = box(remote, "dom", queue)
    publish(share("dom", "Gym", hours=1))
    tick(1)
    remote.down = False
    tick(61)
    assert remote.rows == {} and sync.pending is None and queue.load("team", "dom") is None


def test_a_stop_made_offline_is_delivered_however_late(tmp_path):
    remote, queue = Flaky(), PresenceQueue(tmp_path / "circle.sqlite")
    sync, _, tick, publish = box(remote, "dom", queue)
    first = share("dom", "Library")
    publish(first)
    tick(1)
    remote.down = True
    publish(stop(first, at=5))
    tick(6)
    assert queue.load("team", "dom").event_id == "stop-5"  # waiting on disk
    remote.down = False
    tick(30)
    assert remote.active("team", NOW + timedelta(minutes=30)) == [] and sync.pending is None


def test_only_the_latest_unsent_place_is_kept(tmp_path):
    queue = PresenceQueue(tmp_path / "circle.sqlite")
    older, newer = share("dom", "Library"), share("dom", "Dorm", at=3)
    queue.save(older)
    queue.save(newer)
    assert queue.load("team", "dom").place == "Dorm"
    queue.clear(older)  # the older one finished sending after the newer was queued
    assert queue.load("team", "dom").place == "Dorm"
    queue.clear(newer)
    assert queue.load("team", "dom") is None and queue.load("team", "hemdutt") is None


def test_offline_presence_is_logged_once_not_every_tick(caplog):
    remote = Flaky()
    remote.down = True
    _, _, tick, publish = box(remote, "dom")
    publish(share("dom", "Library"))
    with caplog.at_level(logging.WARNING, logger="atlas.circle"):
        tick(1), tick(2), tick(3)
        remote.down = False
        tick(4)
    assert [r.message.split(",")[0] for r in caplog.records] == ["presence sync paused", "presence sync resumed"]


# --- the presence table: expiry on the database's clock ---


def test_presence_is_stored_and_read_on_the_database_clock():
    neon = FakeNeon(rows=[], columns=["member_id", "place", "source", "shared_at", "expires_at"])
    presence = NeonPresence(NeonDb(neon.connect))
    payload = {"place": "Library", "source": "phone", "shared_at": NOW.isoformat(),
               "expires_at": (NOW + timedelta(hours=2)).isoformat(), "sent_at": (NOW + timedelta(minutes=30)).isoformat()}
    presence.push(OutboxRow("e1", "team", "dom", "presence", payload, NOW))
    sql, params = neon.statements[-1]
    assert "now() + (%s::timestamptz - %s::timestamptz)" in sql and "presence.shared_at <= EXCLUDED.shared_at" in sql
    assert params[-2:] == (payload["expires_at"], payload["sent_at"])  # what is left when it is sent

    presence.active("team", NOW)
    sql, params = neon.statements[-1]
    assert "expires_at > now()" in sql and "(expires_at - now())" in sql and params == (NOW, "team")


# --- a slow or unreachable database never holds the box up ---


def test_once_offline_calls_fail_fast_until_the_retry_window_passes():
    neon, clock = FakeNeon(), [100.0]
    neon.down = True
    db = NeonDb(neon.connect, retry_s=15, monotonic=lambda: clock[0])
    with pytest.raises(CircleOffline):
        db.run(lambda cur: cur.execute("SELECT 1"))
    neon.down = False  # back, but the box does not know yet
    clock[0] += 5
    with pytest.raises(CircleOffline):  # no second connect timeout for the next job in the same tick
        db.run(lambda cur: cur.execute("SELECT 1"))
    assert neon.connects == 0
    clock[0] += 11
    db.run(lambda cur: cur.execute("SELECT 1"))
    assert neon.connects == 1 and db.offline is None


def test_a_caller_does_not_queue_behind_a_stuck_statement():
    db = NeonDb(FakeNeon().connect, busy_timeout_s=0.01)
    db._lock.acquire()  # another thread is in the middle of a statement that does not come back
    try:
        with pytest.raises(CircleOffline, match="slow to answer"):
            db.run(lambda cur: cur.execute("SELECT 1"))
    finally:
        db._lock.release()
    assert db.offline is None  # busy is not offline: the next call simply tries
    db.run(lambda cur: cur.execute("SELECT 1"))


def test_a_tick_job_that_gets_no_answer_counts_as_offline(monkeypatch):
    monkeypatch.setattr(bootstrap, "NEON_JOB_TIMEOUT_S", 0.05)
    gate = threading.Event()  # a statement that does not come back

    async def run() -> str:
        try:
            with pytest.raises(CircleOffline, match="no answer"):
                await bootstrap.off_loop(gate.wait, 5)
        finally:
            gate.set()
        return await bootstrap.off_loop(lambda: "ok")

    assert asyncio.run(run()) == "ok"


def test_a_store_failure_part_way_through_a_poll_is_retried_on_the_next_one():
    class DropsOnce(InMemoryCircleStore):
        fail = True

        def post(self, msg):
            if msg.kind == "reply" and self.fail:
                self.fail = False
                raise CircleOffline("the connection dropped")
            super().post(msg)

    store, ticks = DropsOnce(), iter(range(1, 100))
    hem = CircleAgent("hemdutt", "team", store, FakeMemory(items=[]), lambda: NOW + timedelta(seconds=next(ticks)))
    plan = Proposal(title="Study", start=NOW + timedelta(days=1), end=NOW + timedelta(days=1, hours=1), invitees=["hemdutt"])
    InMemoryCircleStore.post(store, AgentMessage(id="m1", circle_id="team", thread_id="t1", from_member="dom",
                                                 kind="proposal", proposal=plan, created_at=NOW))
    with pytest.raises(CircleOffline):
        hem.poll()
    assert [m.kind for m in store.thread("team", "t1")] == ["proposal"]
    events = hem.poll()  # the proposal is read again, surfaced and answered
    assert [e.type for e in events] == ["circle.message.received", "circle.event.agreed"]
    assert [(m.kind, m.from_member) for m in store.thread("team", "t1")] == [("proposal", "dom"), ("reply", "hemdutt")]
    assert hem.poll() == []


# --- cleanup: bounded, and only this box's own rows ---


def test_cleanup_only_names_this_members_rows_in_this_circle():
    neon = FakeNeon()
    removed = NeonCleanup(NeonDb(neon.connect), limit=50).sweep("team", "dom", timedelta(days=30))
    assert set(removed) == {"agent_messages", "shares", "presence"}
    assert len(neon.statements) == 3
    for sql, params in neon.statements:
        assert sql.startswith("DELETE FROM") and "circle_id = %s" in sql and "now() - %s" in sql
        assert params[:2] == ("team", "dom") and timedelta(days=30) in params
    assert "from_member = %s" in neon.statements[0][0] and "LIMIT %s" in neon.statements[0][0]
    assert neon.statements[0][1][-1] == 50


def test_cleanup_is_off_at_zero_days():
    neon = FakeNeon()
    assert NeonCleanup(NeonDb(neon.connect)).sweep("team", "dom", timedelta(0)) == {}
    assert neon.statements == [] and neon.connects == 0


# --- the same SQL on a real Postgres (skipped without NEON_DATABASE_URL) ---


@needs_neon
def test_real_presence_round_trip_stop_and_database_clock():
    db, c = open_test_circle()
    try:
        remote = NeonPresence(db)
        now = datetime.now(UTC)
        wrong = now + timedelta(hours=5)  # a's clock is five hours fast: the share must still last two hours for b
        a, b = PresenceSync(remote, c, f"{c}-a"), PresenceSync(remote, c, f"{c}-b")
        first = PlaceShared(ts=wrong, circle_id=c, member_id=f"{c}-a", place="Library", source="phone",
                            expires_at=wrong + timedelta(hours=2))
        a.on_shared(first)
        a.sync(wrong)
        [seen] = b.sync(now)
        assert seen.place == "Library" and abs((seen.expires_at - now) - timedelta(hours=2)) < timedelta(seconds=30)
        assert b.sync(now) == []

        a.on_shared(first.model_copy(update={"event_id": "stop", "ts": wrong + timedelta(minutes=1),
                                             "expires_at": wrong + timedelta(minutes=1)}))
        a.sync(wrong + timedelta(minutes=1))
        assert remote.active(c, now) == []
        [ended] = b.sync(now + timedelta(minutes=1))
        assert ended.member_id == f"{c}-a" and ended.expires_at <= now + timedelta(minutes=1)
    finally:
        assert drop_test_circle(db, c) == 0


@needs_neon
def test_real_cleanup_removes_only_this_members_old_rows():
    db, c = open_test_circle()
    try:
        store, me, friend = NeonCircleStore(db), f"{c}-a", f"{c}-b"
        old, plan = datetime.now(UTC) - timedelta(days=40), Proposal(title="t", start=NOW, end=NOW + timedelta(hours=1))
        for n, (member, at) in enumerate([(me, old), (me, datetime.now(UTC)), (friend, old)]):
            store.post(AgentMessage(id=f"{c}-m{n}", circle_id=c, thread_id=f"{c}-t", from_member=member, kind="proposal",
                                    proposal=plan, created_at=at))
        NeonShares(db).push(OutboxRow(f"{c}-s0", c, me, "game_score", {
            "round_id": f"{c}-r0", "session_id": "s", "players": 1, "item": "can", "correct": True, "seconds": 1.0,
            "path_len": 0, "score": 1.0}, old))
        removed = NeonCleanup(db).sweep(c, me, timedelta(days=30))
        assert removed == {"agent_messages": 1, "shares": 1, "presence": 0}
        assert [m.id for m in store.since(c, None)] == [f"{c}-m2", f"{c}-m1"]  # the friend's old row and my new one stay
        assert len(CircleBoard(db).top_scores(c)) == 1  # the score itself is history: kept
    finally:
        assert drop_test_circle(db, c) == 0


@needs_neon
def test_real_sealed_messages_and_league_tables_take_our_rows():
    db, c = open_test_circle()
    try:
        sealed, me = NeonSealedStore(db), f"{c}-a"
        row = SealedMessage(id=f"{c}-k", circle_id=c, from_member=me, kind="box_key", body={"k": "v"}, created_at=NOW)
        sealed.post(row)
        sealed.post(row)  # a retry: still one row
        [back] = sealed.inbox(c, f"{c}-b")
        assert back.body == {"k": "v"} and back.to_member is None
        sealed.delete(c, [row.id])
        assert sealed.inbox(c, f"{c}-b") == []

        league = NeonLeague(db, CircleBoard(db))
        stake = LeagueStakeSet(ts=NOW, circle_id=c, week_start=date(2026, 10, 5), stake=Stake(kind="pays_for_food"), set_by=me)
        league.push_stake(stake)
        league.push_stake(stake)
        assert league.habit_summaries(c, date(2026, 9, 1)) == [] and league.impact_counters(c) == []

        def count(cur):
            cur.execute("SELECT count(*) FROM league_stakes WHERE circle_id = %s", (c,))
            return cur.fetchall()[0][0]

        assert db.run(count) == 1
    finally:
        assert drop_test_circle(db, c) == 0


@needs_neon
def test_real_face_share_key_exchange_share_accept_revoke_and_expiry(tmp_path):
    """The whole face share flow of test_face_share.py, with the two boxes meeting only in the real table.
    The "faces" are the fake recogniser's synthetic vectors; the days are shortened to one."""
    db, c = open_test_circle()
    try:
        store, wall = NeonSealedStore(db), faces.Wall()
        a, b = f"{c}-a", f"{c}-b"
        names = {a: "Test A", b: "Test B"}
        faces.TEMPLATE_NAME.update({a: "Zebulon Quartermaine", b: "Ottoline Pemberwick"})

        def box(member):
            made = faces.Box(tmp_path, member, store, wall, names=names, days=1)
            made.share.circle_id = c  # the helper's boxes live in "team"; these rows must stay under the test circle
            return made

        def kinds():
            def work(cur):
                cur.execute("SELECT kind, count(*) FROM sealed_messages WHERE circle_id = %s GROUP BY 1 ORDER BY 1", (c,))
                return dict(cur.fetchall())
            return db.run(work)

        ana, bob = box(a), box(b)
        ana.enrol_self()
        bob.identity.set_enabled(True)
        faces.share_and_accept(ana, bob)  # keys published and read, the share sealed, the accept card approved
        assert kinds() == {"box_key": 2}  # the ciphertext row is deleted once it is used
        assert len(bob.identity.members()) == 1 and bob.sees(a.encode()) != {}

        ana.publish(ana.share.revoke(b))
        assert kinds() == {"box_key": 2, "face_revoke": 1}
        bob.poll()
        assert bob.identity.members() == [] and kinds() == {"box_key": 2}

        faces.share_and_accept(ana, bob)  # again, and this time it runs out by itself
        assert len(bob.identity.members()) == 1
        wall.pass_(days=1, hours=1)
        bob.poll()
        assert bob.identity.members() == [] and bob.states()[-1] == ("in", a, "expired")
    finally:
        faces.TEMPLATE_NAME.pop(f"{c}-a", None), faces.TEMPLATE_NAME.pop(f"{c}-b", None)
        assert drop_test_circle(db, c) == 0


# --- what must never reach the database ---


def test_near_you_demo_events_never_reach_the_database(tmp_path):
    neon, bus = FakeNeon(), EventBus(tmp_path / "events")
    db = NeonDb(neon.connect)
    sharing = CircleSharing(CircleOutbox(tmp_path / "circle.sqlite"), NeonShares(db), "team", "dom", opted_in=True)
    presence = PresenceSync(NeonPresence(db), "team", "dom", queue=PresenceQueue(tmp_path / "circle.sqlite"))
    bus.subscribe("habits.day.closed", sharing.on_day_closed)  # the subscriptions bootstrap.py makes for Neon
    bus.subscribe("impact.count.updated", sharing.on_count_updated)
    bus.subscribe("play.round.scored", sharing.on_round_scored)
    bus.subscribe("circle.place.shared", presence.on_shared)
    events = nearby_events(NOW)
    assert events and {e.type for e in events} == {"connector.item.synced"}
    asyncio.run(bus.publish_all(events))
    assert sharing.outbox.pending() == [] and presence.pending is None
    sharing.sync(NOW)
    assert [sql for sql, _ in neon.statements if not sql.startswith("SELECT")] == []  # nothing written


# --- python -m contexts.circle.schema ---


def test_the_schema_tool_reads_schema_sql_and_runs_only_its_creates():
    want = schema.expected(schema.SCHEMA.read_text())
    assert {"presence", "agent_messages", "sealed_messages", "league_weeks"} <= set(want["tables"])
    assert want["tables"]["presence"] == ["member_id", "circle_id", "place", "source", "shared_at", "expires_at"]
    assert want["indexes"]["presence_circle_expiry"] == "presence" and "habit_leaderboard" in want["views"]
    kinds = {kind for kind, _, _ in schema.statements(schema.SCHEMA.read_text())}
    assert kinds == {"table", "index", "view"}
    for bad in ("ALTER TABLE presence ADD COLUMN x text;", "DROP TABLE presence;", "DELETE FROM presence;",
                "CREATE TABLE presence (x text);"):
        with pytest.raises(ValueError):
            schema.statements(bad)


def test_the_schema_tool_never_prints_the_connection_string(monkeypatch, capsys):
    monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://someone:s3cret@127.0.0.1:1/nothing?connect_timeout=1")
    assert schema.main(["--check"]) == 1
    said = capsys.readouterr()
    assert "could not connect" in said.out and "s3cret" not in said.out + said.err and "127.0.0.1" not in said.out + said.err


@needs_neon
def test_real_schema_check_reads_the_database_without_writing():
    db = NeonDb()
    try:
        conn = db._connect()
        try:
            conn.read_only = True  # the check can only read; --apply is for a person to run
            assert isinstance(schema.audit(conn, out=lambda *a: None), list)
        finally:
            conn.close()
    finally:
        db.close()
