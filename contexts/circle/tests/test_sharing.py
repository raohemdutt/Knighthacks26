"""Outbox, sync worker and share handlers (N1). The last test runs against real Neon when
NEON_DATABASE_URL is set."""

import sqlite3
from datetime import UTC, date, datetime, timedelta

from contexts.circle.application.sharing import CircleSharing
from contexts.circle.infrastructure.neon import CircleBoard, NeonDb, NeonShares
from contexts.circle.infrastructure.outbox import CircleOutbox
from contexts.circle.tests.neon_helpers import FakeNeon, drop_test_circle, needs_neon, open_test_circle
from contracts.circle import ShareQueued, ShareSynced
from contracts.habits import DayClosed
from contracts.impact import CountUpdated
from contracts.play import RoundScored

NOW = datetime(2026, 10, 10, 16, tzinfo=UTC)
LATER = NOW + timedelta(minutes=5)


def day_closed(streak=3, **kw):
    return DayClosed(ts=NOW, habit_id="sleep", day=date(2026, 10, 9), met=True, value=7.5, streak=streak, **kw)


def count_updated(recycled=4, ts=NOW, **kw):
    return CountUpdated(ts=ts, items_seen=9, recyclable_seen=5, recycled=recycled, contamination=1,
                        diversion_rate=recycled / 5, **kw)


def round_scored(round_id="r1", score=88.0, **kw):
    return RoundScored(ts=NOW, round_id=round_id, session_id="s1", players=2, item="can", correct=True, seconds=6.5,
                       path_len=7, score=score, **kw)


def sharing(tmp_path, neon=None, opted_in=True, circle="team", member="dom"):
    neon = neon or FakeNeon()
    return CircleSharing(CircleOutbox(tmp_path / "circle.sqlite"), NeonShares(NeonDb(neon.connect)), circle, member, opted_in)


def test_enqueue_writes_locally_and_returns_share_queued(tmp_path):
    outbox = CircleOutbox(tmp_path / "nested" / "circle.sqlite")
    queued = outbox.enqueue("team", "dom", "habit_summary", {"habit_id": "sleep"}, NOW)
    assert isinstance(queued, ShareQueued) and (queued.type, queued.ts) == ("circle.share.queued", NOW)
    [row] = CircleOutbox(tmp_path / "nested" / "circle.sqlite").pending()  # survives a restart
    assert (row.share_id, row.circle_id, row.member_id, row.kind) == (queued.share_id, "team", "dom", "habit_summary")
    assert row.payload == {"habit_id": "sleep"} and row.created_at == NOW
    columns = [c[1] for c in sqlite3.connect(tmp_path / "nested" / "circle.sqlite").execute("PRAGMA table_info(circle_outbox)")]
    assert columns == ["share_id", "circle_id", "member_id", "kind", "payload", "created_at", "synced_at"]


def test_the_same_share_is_never_queued_twice(tmp_path):
    outbox = CircleOutbox(tmp_path / "circle.sqlite")
    assert outbox.enqueue("team", "dom", "game_score", {}, NOW, share_id="s1")
    assert outbox.enqueue("team", "dom", "game_score", {}, NOW, share_id="s1") is None
    assert len(outbox.pending()) == 1


def test_nothing_is_queued_without_consent(tmp_path):
    off = sharing(tmp_path, opted_in=False)
    assert off.on_day_closed(day_closed()) is None
    assert off.on_count_updated(count_updated()) is None
    assert off.on_round_scored(round_scored()) is None
    assert off.outbox.pending() == [] and off.sync(LATER) == []


def test_handlers_queue_only_the_contract_fields(tmp_path):
    on = sharing(tmp_path)
    event = day_closed()
    queued = on.on_day_closed(event)
    assert (queued.kind, queued.share_id, queued.circle_id) == ("habit_summary", event.event_id, "team")
    assert queued.payload == {"habit_id": "sleep", "day": "2026-10-09", "met": True, "value": 7.5, "streak": 3}
    assert on.on_day_closed(event) is None  # the same event again
    assert on.on_count_updated(count_updated()).payload.keys() == {
        "items_seen", "recyclable_seen", "recycled", "contamination", "diversion_rate"}  # fmt: skip
    assert on.on_round_scored(round_scored()).payload["round_id"] == "r1"


def test_sync_pushes_each_kind_to_its_table_and_marks_it_synced(tmp_path):
    neon = FakeNeon()
    on = sharing(tmp_path, neon)
    on.on_day_closed(day_closed())
    on.on_count_updated(count_updated(ts=NOW + timedelta(seconds=1)))
    on.on_round_scored(round_scored().model_copy(update={"ts": NOW + timedelta(seconds=2)}))
    on.outbox.enqueue("team", "dom", "reply", {"answer": "free"}, NOW + timedelta(seconds=3))

    synced = on.sync(LATER)
    assert len(synced) == 4 and all(isinstance(e, ShareSynced) and e.ts == LATER for e in synced)
    assert neon.tables() == [
        "circles", "members", "shares", "habit_summaries",
        "circles", "members", "shares", "impact_counters",
        "circles", "members", "shares", "game_scores",
        "circles", "members", "shares",
    ]  # fmt: skip
    assert all("ON CONFLICT" in sql for sql, _ in neon.statements)  # a retry never duplicates
    assert on.outbox.pending() == [] and on.sync(LATER) == []


def test_offline_leaves_rows_queued_and_returns_nothing(tmp_path):
    neon = FakeNeon()
    neon.down = True
    on = sharing(tmp_path, neon)
    on.on_day_closed(day_closed())
    on.on_round_scored(round_scored())
    assert on.sync(LATER) == []  # no exception for the ticker
    assert len(on.outbox.pending()) == 2
    neon.down = False
    assert len(on.sync(LATER)) == 2 and on.outbox.pending() == []


def test_a_refused_share_does_not_block_the_ones_behind_it(tmp_path):
    on = sharing(tmp_path)
    on.outbox.enqueue("team", "dom", "habit_summary", {"habit_id": "sleep"}, NOW, share_id="bad")  # fields missing
    on.on_round_scored(round_scored().model_copy(update={"ts": LATER}))
    [synced] = on.sync(LATER)
    assert synced.share_id != "bad"
    assert [r.share_id for r in on.outbox.pending()] == ["bad"]


@needs_neon
def test_shares_reach_neon_once_and_show_on_the_leaderboards(tmp_path):
    db, c = open_test_circle()
    try:
        me, board = f"{c}-dom", CircleBoard(db)
        on = CircleSharing(CircleOutbox(tmp_path / "circle.sqlite"), NeonShares(NeonDb(names={me: "Dom (test)"})), c, me, True)
        on.on_day_closed(day_closed(event_id=f"{c}-habit"))
        on.on_count_updated(count_updated(event_id=f"{c}-count"))
        on.on_count_updated(count_updated(recycled=5, ts=LATER, event_id=f"{c}-count2"))
        on.on_round_scored(round_scored(f"{c}-r1", 88.0, event_id=f"{c}-score1"))
        on.on_round_scored(round_scored(f"{c}-r2", 95.0, event_id=f"{c}-score2"))
        assert len(on.sync(LATER)) == 5

        # a second box file with the same shares (a retry after a lost acknowledgement): nothing duplicates
        again = CircleSharing(CircleOutbox(tmp_path / "retry.sqlite"), NeonShares(db), c, me, True)
        again.on_day_closed(day_closed(streak=4, event_id=f"{c}-habit"))
        again.on_count_updated(count_updated(recycled=1, event_id=f"{c}-count"))  # older than what Neon has
        again.on_round_scored(round_scored(f"{c}-r1", 1.0, event_id=f"{c}-score1"))
        assert len(again.sync(LATER)) == 3

        def counts(cur):
            out = []
            for table in ("shares", "habit_summaries", "impact_counters", "game_scores"):
                cur.execute(f"SELECT count(*) FROM {table} WHERE circle_id = %s", (c,))
                out.append(cur.fetchall()[0][0])
            return out

        assert db.run(counts) == [5, 1, 1, 2]
        assert board.habit_leaderboard(c) == [
            {"member_id": me, "display_name": "Dom (test)", "habit_id": "sleep", "streak": 4, "day": date(2026, 10, 9)}]
        [impact] = board.impact_leaderboard(c)
        assert (impact["recycled"], impact["items_seen"], impact["diversion_rate"]) == (5, 9, 1.0)
        assert [(s["round_id"], s["score"], s["display_name"]) for s in board.top_scores(c, limit=1)] == [
            (f"{c}-r2", 95.0, "Dom (test)")]
        assert [s["score"] for s in board.top_scores(c)] == [95.0, 88.0]
    finally:
        assert drop_test_circle(db, c) == 0
