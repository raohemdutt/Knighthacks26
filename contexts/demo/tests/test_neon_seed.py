"""`python -m contexts.demo neon`: demo members in the Circle's shared database, written through the boxes' own stores.

On the fake connection (both transports) these show what is sent. The test marked `needs_neon` runs the whole thing
(apply, the leaderboards, apply again, a real member beside it, remove) on whatever Postgres NEON_DATABASE_URL
names, under a throwaway `test-…` circle."""

from argparse import Namespace
from collections import Counter
from datetime import UTC, date, datetime, timedelta

import pytest

from contexts.circle.application.protocol import AgentMessage
from contexts.circle.application.sharing import OutboxRow
from contexts.circle.infrastructure.neon import CircleBoard, NeonCircleStore, NeonDb, NeonShares
from contexts.circle.tests.neon_helpers import FAKE_URL_PARTS, FakeNeon, drop_test_circle, needs_neon, open_test_circle
from contexts.demo import neon_seed
from contexts.demo.__main__ import main as demo_main
from contexts.demo.neon_seed import Found, SeedRefused, apply, build, judge, look, manifest_path, remove, removal
from contracts.circle import Proposal

END = date(2026, 10, 11)  # a Sunday: the last complete week is Sep 28 to Oct 4
TRANSPORTS = ["tcp", "https"]


def args(tmp_path, **more) -> Namespace:
    base = dict(circle="kh-judges", name=None, members=None, days=None, end=END, dry_run=False, apply=False, remove=False,
                into_existing=False, data_dir=tmp_path)
    return Namespace(**(base | more))


def run(tmp_path, neon=None, **more) -> tuple[int, str]:
    said: list[str] = []
    code = neon_seed.main(args(tmp_path, **more), out=said.append, connect=neon.connect if neon else None)
    return code, "\n".join(said)


def as_found(plan, name=None) -> list[Found]:
    """The database right after --apply: every planned row, owned as the plan says."""
    first = next(iter(plan.members))
    owner = {"circles": lambda key: None, "league_weeks": lambda key: None, "rsvps": lambda key: key.split("|")[1],
             "members": lambda key: key, "shares": lambda key: key.split(":")[0], "habit_summaries": lambda key: key.split("|")[0],
             "impact_counters": lambda key: key, "game_scores": lambda key: key.split(":")[2]}
    return [Found(table, key, owner.get(table, lambda key: first)(key), plan.circle,
                  (name or plan.name) if table == "circles" else plan.week_stamps.get(key))
            for table, keys in plan.rows().items() for key in keys]


# --- the plan ---


def test_the_plan_depends_only_on_the_circle_the_members_the_days_and_the_date():
    one, two = build("kh-judges", end=END), build("kh-judges", end=END)
    assert one.rows() == two.rows() and [s.what for s in one.steps] == [s.what for s in two.steps]
    assert build("kh-judges", end=END + timedelta(days=1)).rows() != one.rows()
    assert one.days[0] == date(2026, 9, 28) and one.days[-1] == date(2026, 10, 10)  # the whole last week, up to yesterday
    rows = one.rows()
    assert {t: len(k) for t, k in rows.items()} == {
        "circles": 1, "members": 3, "shares": 279, "habit_summaries": 234, "impact_counters": 3, "game_scores": 6,
        "agent_messages": 6, "events": 2, "rsvps": 6, "league_stakes": 1, "league_weeks": 1, "league_challenges": 1}
    assert set(rows["members"]) == set(neon_seed.DEFAULT_MEMBERS)
    assert all(share.split(":")[0] in neon_seed.DEFAULT_MEMBERS for share in rows["shares"])  # every share names its member
    assert all(key.startswith("demo-seed:kh-judges:") for t in ("game_scores", "agent_messages", "events", "league_challenges")
               for key in rows[t])
    assert "presence" not in rows and "sealed_messages" not in rows


def test_the_league_rows_are_declared_by_the_real_projection_and_the_first_member_leads():
    plan = build("kh-judges", end=END)
    week = next(s for s in plan.steps if "league_weeks" in s.rows)
    assert week.what.startswith("League week of 2026-09-28: Dominick had the worst week") and "Hemdutt picks the place" in week.what
    dare = next(s for s in plan.steps if "league_challenges" in s.rows)
    assert dare.what == "League dare: Sort it right, 5 days: Hemdutt made it."  # the first preset somebody kept all week
    monday = build("kh-judges", end=date(2026, 10, 12))  # a Monday: last week is declared all the same
    assert monday.rows()["league_weeks"] == ["kh-judges|2026-10-05"] and len(monday.days) == 7


def test_a_plan_always_names_its_invitees_so_no_real_box_is_asked(tmp_path):
    captured = FakeNeon()
    apply(build("kh-judges", end=END), NeonDb(captured.connect), tmp_path)
    proposals = [p for sql, p in captured.statements if sql.startswith("INSERT INTO agent_messages") and p[4] == "proposal"]
    assert len(proposals) == 2
    for p in proposals:
        invited = Proposal.model_validate_json(p[5]).invitees
        assert invited and set(invited) < set(neon_seed.DEFAULT_MEMBERS)
    with pytest.raises(SeedRefused):
        build("kh-judges", members={"demo-solo": "Solo"}, end=END)  # nobody to invite: every box would be asked
    with pytest.raises(SeedRefused):
        build("kh-judges", name="kh-judges", end=END)  # the name is the mark of a seeded circle: it must differ


def test_explicit_members_are_used_as_they_are():
    assert neon_seed.parse_members(None) == {"demo-hemdutt": "Hemdutt", "demo-ilan": "Ilan", "demo-dominick": "Dominick"}
    assert neon_seed.parse_members("demo-maya, zed=Zed The Great") == {"demo-maya": "Maya", "zed": "Zed The Great"}
    assert set(build("c", members=neon_seed.parse_members("a,b"), end=END).rows()["members"]) == {"a", "b"}


# --- the command, on the fake connection ---


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_the_default_is_a_dry_run_that_only_reads(tmp_path, transport):
    neon = FakeNeon(transport=transport)
    code, said = run(tmp_path, neon)
    assert code == 0 and "dry run; nothing was sent. --apply would make sure these 543 rows are there:" in said
    assert "  habit_summaries     234  " in said and "demo-hemdutt|sleep|2026-10-10" in said  # the counts and the ids
    assert neon.statements and all(sql.startswith("SELECT") for sql, _ in neon.statements)
    assert not (tmp_path / "neon-seed").exists()  # no manifest either
    code, said = run(tmp_path, neon, remove=True)
    assert code == 0 and "dry run of the removal; nothing was changed" in said
    assert all(sql.startswith("SELECT") for sql, _ in neon.statements)


def test_without_a_database_the_dry_run_still_prints_the_plan(tmp_path, monkeypatch):
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.setenv("ATLAS_ENV_FILE", str(tmp_path / "no-such-env"))
    code, said = run(tmp_path)
    assert code == 0 and "  shares              279  " in said and "the database was not looked at" in said
    code, said = run(tmp_path, apply=True)
    assert code == 2 and "NEON_DATABASE_URL is not set" in said and not (tmp_path / "neon-seed").exists()


def test_the_command_is_reachable_as_python_m_contexts_demo_neon(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.setenv("ATLAS_ENV_FILE", str(tmp_path / "no-such-env"))
    assert demo_main(["neon", "--circle", "kh-judges", "--end", "2026-10-11", "--data-dir", str(tmp_path)]) == 0
    assert "--apply would make sure these 543 rows are there" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        demo_main(["neon", "--circle", "kh-judges", "--dry-run", "--apply"])  # one or the other


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_apply_writes_only_what_the_boxes_own_stores_write(tmp_path, transport):
    neon = FakeNeon(transport=transport)
    code, said = run(tmp_path, neon, apply=True)
    assert code == 0 and not any(part in said for part in FAKE_URL_PARTS)
    writes = [(sql, p) for sql, p in neon.statements if not sql.startswith("SELECT")]
    plan = build("kh-judges", end=END)
    made = Counter(sql.split()[2] for sql, _ in writes if sql.startswith("INSERT INTO"))
    assert {t: made[t] for t in ("shares", "habit_summaries", "impact_counters", "game_scores", "agent_messages",
                                 "league_stakes", "league_weeks", "league_challenges")} == {
        "shares": 279, "habit_summaries": 234, "impact_counters": 39, "game_scores": 6, "agent_messages": 6,
        "league_stakes": 1, "league_weeks": 1, "league_challenges": 1}  # a counter per day; the latest one stays
    assert sum(sql.startswith("WITH new_event AS (INSERT INTO events") for sql, _ in writes) == 2  # record_event

    # every statement text is one a store class sends for a box (plus the named circle, which has ensure's own shape)
    box = FakeNeon()
    db = NeonDb(box.connect)
    NeonShares(db).push(OutboxRow("s", "c", "m", "habit_summary", {"habit_id": "sleep", "day": "2026-10-10", "met": True, "value": 7.0, "streak": 1}, datetime.now(UTC)))
    NeonShares(db).push(OutboxRow("s", "c", "m", "impact_count", {"items_seen": 1, "recyclable_seen": 1, "recycled": 1, "contamination": 0, "diversion_rate": 1.0}, datetime.now(UTC)))
    NeonShares(db).push(OutboxRow("s", "c", "m", "game_score", {"round_id": "r", "session_id": "s", "players": 1, "item": "can", "correct": True, "seconds": 1.0, "path_len": 1, "score": 1.0}, datetime.now(UTC)))
    plan.steps[-3].run(neon_seed.Stores.on(db)), plan.steps[-2].run(neon_seed.Stores.on(db)), plan.steps[-1].run(neon_seed.Stores.on(db))
    study = Proposal(title="t", start=datetime.now(UTC), end=datetime.now(UTC) + timedelta(hours=1), invitees=["b"])
    NeonCircleStore(db).post(AgentMessage(id="i", circle_id="c", thread_id="t", from_member="a", kind="proposal", proposal=study,
                                          created_at=datetime.now(UTC)))
    NeonCircleStore(db).record_event("c", "t", study, "a")
    theirs = {sql for sql, _ in box.statements}
    assert {sql for sql, _ in writes} == theirs and not any("DELETE" in sql or "UPDATE circles" in sql for sql in theirs)

    circle = next(p for sql, p in writes if sql.startswith("INSERT INTO circles"))
    assert circle == ("kh-judges", "kh-judges (demo seed)")  # the first write: the circle under the seed's name
    assert all(p[1] == p[0] for sql, p in writes[1:] if sql.startswith("INSERT INTO circles"))  # ensure's, which then do nothing
    names = {p[0]: p[2] for sql, p in writes if sql.startswith("INSERT INTO members")}
    assert names == {"demo-hemdutt": "Hemdutt", "demo-ilan": "Ilan", "demo-dominick": "Dominick"}

    manifest = manifest_path(tmp_path, "kh-judges")
    assert manifest.exists() and not any(part in manifest.read_text() for part in FAKE_URL_PARTS)
    assert "to remove from any machine: python -m contexts.demo neon --circle kh-judges --end 2026-10-11 --days 13 --remove --apply" in said


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_an_unreachable_database_says_so_without_saying_where(tmp_path, transport):
    neon = FakeNeon(transport=transport)
    neon.down = True
    code, said = run(tmp_path, neon, apply=True)
    assert code == 1 and "could not be reached" in said and not any(part in said for part in FAKE_URL_PARTS)


# --- whose rows are whose ---


def test_an_empty_database_and_a_seeded_one_are_both_fine():
    plan = build("kh-judges", end=END)
    fresh = judge(plan, [])
    assert fresh.refusal is None and fresh.ours == [] and fresh.circle_is_ours
    again = judge(plan, as_found(plan))
    assert again.refusal is None and len(again.ours) == 543 and again.others == [] and again.circle_is_ours


def test_a_circle_with_someone_elses_rows_needs_into_existing_and_their_rows_are_never_ours():
    plan = build("kh-judges", end=END)
    real = [Found("circles", "kh-judges", None, "kh-judges", "kh-judges"),  # a box named it after its id
            Found("members", "hemdutt", "hemdutt", "kh-judges"),
            Found("habit_summaries", "hemdutt|sleep|2026-10-10", "hemdutt", "kh-judges"),
            Found("league_weeks", "kh-judges|2026-09-28", None, "kh-judges", datetime(2026, 10, 5, 7, tzinfo=UTC))]
    refused = judge(plan, real)
    assert "already exists and is not this seed's" in refused.refusal and "--into-existing" in refused.refusal
    seeded = [f for f in as_found(plan) if f.table not in ("circles", "league_weeks")]  # those two were already theirs
    beside = judge(plan, real + seeded, into_existing=True)
    assert beside.refusal is None and not beside.circle_is_ours and len(beside.others) == 3
    assert beside.skip == {"kh-judges|2026-09-28"} and "left alone" in beside.notes[0]  # their declared week stays theirs
    assert not any(f.owner == "hemdutt" or f.table == "circles" for f in beside.ours)

    statements = removal(plan, beside)
    assert all(sql.startswith("DELETE FROM ") for sql, _ in statements)
    assert not any(sql.startswith(("DELETE FROM circles", "DELETE FROM league_weeks")) for sql, _ in statements)
    assert not any("hemdutt" == value or (isinstance(value, list) and "hemdutt" in value) for _, params in statements for value in params)
    assert all("= ANY(%s)" in sql or "= %s" in sql for sql, _ in statements)  # by key, never a whole table


def test_a_member_id_that_is_someone_elses_is_refused_whatever_the_flags():
    plan = build("kh-judges", end=END)
    elsewhere = judge(plan, [Found("members", "demo-ilan", "demo-ilan", "another-circle")], into_existing=True)
    assert "already belongs to another circle" in elsewhere.refusal
    in_use = judge(plan, as_found(plan) + [Found("habit_summaries", "demo-ilan|sleep|2026-08-01", "demo-ilan", "kh-judges")],
                   into_existing=True)
    assert "already has rows this run would not write" in in_use.refusal and "demo-ilan" in in_use.refusal


def test_removal_is_by_exact_key_and_owner_and_a_member_or_circle_goes_only_when_nothing_is_left_in_it():
    plan = build("kh-judges", end=END)
    statements = removal(plan, judge(plan, as_found(plan)))
    tables = [sql.split()[2] for sql, _ in statements]
    assert tables[0] == "rsvps" and tables[-2:] == ["members", "circles"]  # what refers to a row goes before the row
    assert set(tables) == {"rsvps", "events", "agent_messages", "shares", "game_scores", "league_challenges", "habit_summaries",
                           "impact_counters", "league_stakes", "league_weeks", "members", "circles"}
    members_sql, members_params = statements[-2]
    assert members_params[0] == "kh-judges" and set(members_params[1]) == set(plan.members)
    assert members_sql.count("NOT EXISTS") == 11 and "x.from_member = m.id" in members_sql and "x.set_by = m.id" in members_sql
    circle_sql, circle_params = statements[-1]
    assert circle_params == ("kh-judges", "kh-judges (demo seed)") and "c.name = %s" in circle_sql
    assert circle_sql.count("NOT EXISTS") == 12 and "FROM members x WHERE x.circle_id = c.id" in circle_sql
    for sql, params in statements[:-2]:
        assert "circle_id = %s" in sql or sql.startswith("DELETE FROM rsvps"), sql
    # an explicit id that is not `demo-…` is deleted as a member only when this machine's manifest says the seed made it
    explicit = build("kh-judges", members={"ana": "Ana", "bo": "Bo"}, end=END)
    found = as_found(explicit)
    assert not any(sql.startswith("DELETE FROM members") for sql, _ in removal(explicit, judge(explicit, found)))
    assert any(sql.startswith("DELETE FROM members") for sql, _ in removal(explicit, judge(explicit, found), ["ana", "bo"]))


# --- the whole thing on a real Postgres (skipped without NEON_DATABASE_URL) ---


@needs_neon
def test_real_seed_fills_the_leaderboards_adds_nothing_twice_and_removes_only_its_own(tmp_path):
    db, c = open_test_circle()
    try:
        a, b, real = f"{c}-a", f"{c}-b", f"{c}-real"
        plan = build(c, members={a: "Test A", b: "Test B"}, end=date(2026, 10, 12))  # a Monday: seven days, one League week
        db.names = dict(plan.members)
        counts = {t: len(keys) for t, keys in plan.rows().items()}
        assert look(db, c, [a, b]) == []

        assert apply(plan, db, tmp_path)["new"] == counts
        board = CircleBoard(db)
        sleep = [r for r in board.habit_leaderboard(c) if r["habit_id"] == "sleep"]
        assert [(r["member_id"], r["display_name"], r["streak"]) for r in sleep] == [(a, "Test A", 7), (b, "Test B", 4)]
        impact = board.impact_leaderboard(c)
        assert [r["member_id"] for r in impact] == [a, b] and impact[0]["diversion_rate"] > impact[1]["diversion_rate"] > 0.5
        assert [r["member_id"] for r in board.top_scores(c)][:2] == [a, b] and len(board.top_scores(c)) == 4
        tonight = board.upcoming_events(c, datetime(2026, 10, 12, 16, tzinfo=UTC))
        assert [(r["title"], r["member_id"], r["answer"]) for r in tonight] == [
            ("Dinner", a, "free"), ("Dinner", b, "free"), ("Board game night", a, "free"), ("Board game night", b, "free")]
        assert [m.kind for m in NeonCircleStore(db).since(c, None)] == ["proposal", "reply", "proposal", "reply"]

        assert apply(plan, db, tmp_path)["new"] == {}  # a second apply adds nothing
        gone = remove(plan, db, tmp_path)
        assert gone["removed"] == counts and gone["left"] == [] and look(db, c, [a, b]) == []  # the database as it was
        assert not manifest_path(tmp_path, c).exists()

        # a real member's box is in the same circle: its rows are there before, and exactly the same after
        def join(cur):
            db.ensure(cur, c, real)

        db.run(join)
        NeonShares(db).push(OutboxRow(f"{real}:habits-day-sleep-2026-10-11", c, real, "habit_summary",
                                      {"habit_id": "sleep", "day": "2026-10-11", "met": True, "value": 8.0, "streak": 9},
                                      datetime(2026, 10, 12, 4, 5, tzinfo=UTC)))
        asked = Proposal(title="Real plan", start=datetime(2026, 10, 13, 16, tzinfo=UTC), end=datetime(2026, 10, 13, 17, tzinfo=UTC),
                         invitees=[real])
        NeonCircleStore(db).post(AgentMessage(id=f"{c}-real-m1", circle_id=c, thread_id=f"{c}-real-t", from_member=real,
                                              kind="proposal", proposal=asked, created_at=datetime(2026, 10, 12, 15, tzinfo=UTC)))
        NeonCircleStore(db).record_event(c, f"{c}-real-t", asked, real)
        theirs = sorted((f.table, f.key, f.owner) for f in look(db, c, [a, b]))
        assert len(theirs) == 7  # the circle, the member, a share and its summary, a message, an event and its RSVP

        with pytest.raises(SeedRefused, match="--into-existing"):
            apply(plan, db, tmp_path)
        assert sorted((f.table, f.key, f.owner) for f in look(db, c, [a, b])) == theirs  # refused: nothing written
        beside = apply(plan, db, tmp_path, into_existing=True)
        assert beside["new"] == {t: n for t, n in counts.items() if t != "circles"}  # the circle was already theirs
        assert [r["member_id"] for r in board.habit_leaderboard(c) if r["habit_id"] == "sleep"] == [real, a, b]
        assert apply(plan, db, tmp_path, into_existing=True)["new"] == {}

        gone = remove(plan, db, tmp_path)
        assert gone["removed"] == {t: n for t, n in counts.items() if t != "circles"} and gone["left"] == []
        assert sorted((f.table, f.key, f.owner) for f in look(db, c, [a, b])) == theirs  # the real member's rows, untouched
        assert len(gone["others"]) == 6  # all of them but the circle row itself
    finally:
        assert drop_test_circle(db, c) == 0
