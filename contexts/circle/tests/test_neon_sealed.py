"""NeonSealedStore on the fake connection, once per transport (conftest.py), the table it needs, and the expiry
the database side acts on. The same on a real Postgres: the `needs_neon` tests in test_neon_pass.py."""

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contexts.circle.infrastructure.neon import NeonCleanup, NeonDb
from contexts.circle.infrastructure.neon_sealed import NeonSealedStore, expiry
from contexts.circle.tests.neon_helpers import FAKE_URL_PARTS, FakeNeon
from contracts.circle import SealedMessage

NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)
SCHEMA = Path("contexts/circle/infrastructure/schema.sql").read_text()


def row(kind="face_share", to="bob", id_="s1"):
    return SealedMessage(id=id_, circle_id="team", from_member="ana", to_member=to, kind=kind,
                         body={"header": {"v": 1}, "ciphertext": "AAAA"}, created_at=NOW)


def test_post_writes_one_row_and_is_safe_to_repeat():
    neon = FakeNeon()
    NeonSealedStore(NeonDb(neon.connect, names={"ana": "Ana"})).post(row())
    assert neon.tables() == ["circles", "members", "sealed_messages"] and neon.commits == 1
    sql, params = neon.statements[-1]
    assert "ON CONFLICT (id) DO NOTHING" in sql and "created_at, expires_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)" in sql
    assert params == ("s1", 1, "team", "ana", "bob", "face_share", '{"header": {"v": 1}, "ciphertext": "AAAA"}', NOW, None)
    assert all(table in ("circles", "members", "sealed_messages") for table in neon.tables())  # never agent_messages or shares


UNTIL = NOW + timedelta(days=30)


def sealed_row(header: dict, kind="face_share", id_="s1"):
    return SealedMessage(id=id_, circle_id="team", from_member="ana", to_member="bob", kind=kind,
                         body={"header": header, "ciphertext": "AAAA", "sig": "BBBB"}, created_at=NOW)


def test_post_stores_the_expiry_that_is_in_the_messages_own_header():
    neon = FakeNeon()
    store = NeonSealedStore(NeonDb(neon.connect))
    store.post(sealed_row({"v": 1, "kind": "face_share", "expires_at": UNTIL.isoformat()}))
    sql, params = neon.statements[-1]
    assert params[-1] == UNTIL and json.loads(params[6])["header"]["expires_at"] == UNTIL.isoformat()  # the same value
    assert expiry(sealed_row({"expires_at": "2026-11-10T08:00:00-04:00"})) == UNTIL  # another offset, the same moment


@pytest.mark.parametrize("header", [
    {"v": 1, "kind": "face_revoke", "issued_at": NOW.isoformat()},  # a revoke has no expiry
    {"expires_at": "2026-11-10T12:00:00"},  # no offset: not a moment
    {"expires_at": "soon"}, {"expires_at": 1794398400}, {"expires_at": None}, "not a header", None,
])
def test_a_row_without_a_usable_expiry_is_stored_with_none(header):
    neon = FakeNeon()
    NeonSealedStore(NeonDb(neon.connect)).post(sealed_row(header))
    assert neon.statements[-1][1][-1] is None
    key = SealedMessage(id="key:ana:abcd", circle_id="team", from_member="ana", kind="box_key", body={"sign": "x"}, created_at=NOW)
    assert expiry(key) is None


def test_the_inbox_leaves_out_what_is_past_its_expiry_on_the_databases_clock():
    neon = FakeNeon()
    NeonSealedStore(NeonDb(neon.connect)).inbox("team", "bob")
    sql, params = neon.statements[0]
    assert "AND (expires_at IS NULL OR expires_at > now()) ORDER BY" in sql and params == ("team", "bob")


# --- a database from before the column: found out once, then used the way it was ---


def test_without_the_expiry_column_the_store_works_as_it_did_before_it(caplog):
    neon = FakeNeon(rows=[("s1", 1, "team", "ana", "bob", "face_share", {"header": {"v": 1}, "ciphertext": "AAAA"}, NOW)])
    neon.lacks = {"sealed_messages.expires_at"}
    db = NeonDb(neon.connect, names={"ana": "Ana"})
    store = NeonSealedStore(db)
    with caplog.at_level(logging.INFO, logger="atlas.circle"):
        store.post(sealed_row({"expires_at": UNTIL.isoformat()}))
        assert neon.refused == 1 and neon.tables() == ["circles", "members", "sealed_messages"]  # nothing of the refused try
        sql, params = neon.statements[-1]
        assert "expires_at" not in sql and sql.endswith("VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING")
        assert params[:6] == ("s1", 1, "team", "ana", "bob", "face_share") and params[-1] == NOW and len(params) == 8

        assert store.inbox("team", "bob") == [row()]
        assert "expires_at" not in neon.statements[-1][0] and "(to_member IS NULL OR to_member = %s) ORDER BY" in neon.statements[-1][0]
        store.post(sealed_row({"expires_at": UNTIL.isoformat()}, id_="s2"))
        store.inbox("team", "bob")
        NeonCleanup(db, sealed_keep=timedelta(days=30)).sweep("team", "ana", timedelta(days=30))
    assert neon.refused == 1 and db.lacks == {"sealed_messages.expires_at"}  # found out once, not once per call
    assert db.offline is None  # a refused statement is not "offline"
    said = [r.getMessage() for r in caplog.records if "not there yet" in r.getMessage()]
    assert len(said) == 1 and "sealed_messages.expires_at" in said[0] and "contexts.circle.schema --apply" in said[0]
    assert not any(part in caplog.text for part in (*FAKE_URL_PARTS, "does not exist", "relation"))  # ours, not the server's


def test_another_refusal_is_not_taken_for_a_missing_column():
    class Broken(Exception):
        sqlstate = "42P01"  # the table itself is missing: the store cannot work around that

    def connect():
        raise Broken('relation "sealed_messages" does not exist')

    db = NeonDb(connect)
    with pytest.raises(Broken):
        NeonSealedStore(db).inbox("team", "bob")
    assert db.lacks == set()


# --- the sweep (NeonCleanup): dead sealed rows this member addressed to someone ---


def test_the_sweep_leaves_sealed_messages_alone_unless_face_share_is_on():
    neon = FakeNeon()
    removed = NeonCleanup(NeonDb(neon.connect)).sweep("team", "ana", timedelta(days=30))
    assert "sealed_messages" not in removed and not any("sealed_messages" in sql for sql, _ in neon.statements)


def test_the_sweep_removes_this_members_sealed_rows_that_are_past_their_expiry_or_too_old_to_have_one():
    neon = FakeNeon()
    removed = NeonCleanup(NeonDb(neon.connect), limit=50, sealed_keep=timedelta(days=45)).sweep("team", "ana", timedelta(days=30))
    assert set(removed) == {"agent_messages", "shares", "presence", "sealed_messages"} and len(neon.statements) == 4
    sql, params = neon.statements[-1]
    assert sql.startswith("DELETE FROM sealed_messages WHERE id IN (SELECT id FROM sealed_messages WHERE circle_id = %s"
                          " AND from_member = %s AND to_member IS NOT NULL AND ")  # its own rows; never a key record
    assert "(expires_at < now() OR (expires_at IS NULL AND created_at < now() - %s))" in sql and "LIMIT %s" in sql
    assert params == ("team", "ana", timedelta(days=45), 50)  # face_share_days, not keep_days


def test_without_the_expiry_column_the_sweep_goes_by_age_alone():
    neon = FakeNeon()
    neon.lacks = {"sealed_messages.expires_at"}
    removed = NeonCleanup(NeonDb(neon.connect), sealed_keep=timedelta(days=30)).sweep("team", "ana", timedelta(days=30))
    assert set(removed) == {"agent_messages", "shares", "presence", "sealed_messages"} and neon.refused == 1
    assert [sql.split()[2] for sql, _ in neon.statements] == ["agent_messages", "shares", "presence", "sealed_messages"]
    sql, params = neon.statements[-1]  # the three other deletes were kept: the refusal cost only its own statement
    assert "expires_at" not in sql and "to_member IS NOT NULL AND created_at < now() - %s ORDER BY" in sql
    assert params == ("team", "ana", timedelta(days=30), 500)


def test_keep_days_zero_still_means_this_box_removes_nothing():
    neon = FakeNeon()
    assert NeonCleanup(NeonDb(neon.connect), sealed_keep=timedelta(days=30)).sweep("team", "ana", timedelta(0)) == {}
    assert neon.statements == []


def test_the_inbox_is_the_circles_keys_and_what_is_addressed_to_one_member():
    key = ("key:ana:abcd", 1, "team", "ana", None, "box_key", '{"header": {"v": 1}}', NOW)  # jsonb as text or as a dict
    share = ("s1", 1, "team", "ana", "bob", "face_share", {"header": {"v": 1}, "ciphertext": "AAAA"}, NOW)
    newer = ("s2", 2, "team", "ana", "bob", "something_new", {"x": 1}, NOW)
    neon = FakeNeon(rows=[key, share, newer])
    first, second, third = NeonSealedStore(NeonDb(neon.connect)).inbox("team", "bob")
    assert (first.kind, first.to_member, first.body) == ("box_key", None, {"header": {"v": 1}})
    assert second == row() and (third.v, third.kind) == (2, "something_new")  # unknown kinds and versions still parse
    sql, params = neon.statements[0]
    assert "circle_id = %s AND (to_member IS NULL OR to_member = %s)" in sql and "ORDER BY created_at, id LIMIT 500" in sql
    assert params == ("team", "bob")


def test_delete_is_by_id_within_the_circle_and_nothing_for_nothing():
    neon = FakeNeon()
    store = NeonSealedStore(NeonDb(neon.connect))
    store.delete("team", [])
    assert neon.statements == [] and neon.connects == 0
    store.delete("team", ["s1", "s2"])
    assert neon.statements == [("DELETE FROM sealed_messages WHERE circle_id = %s AND id = ANY(%s)", ("team", ["s1", "s2"]))]


def test_the_schema_has_the_sealed_table_and_leaves_the_others_as_they_were():
    table = re.search(r"CREATE TABLE IF NOT EXISTS sealed_messages \((.*?)\n\);", SCHEMA, re.S).group(1)
    for column in ("id ", "v ", "circle_id", "from_member", "to_member", "kind", "body", "created_at"):
        assert re.search(rf"^\s+{column}", table, re.M), column
    assert "CHECK (kind IN" not in table  # an unknown kind must be storable: later kinds stay additive
    assert "CHECK (kind IN ('proposal', 'reply'))" in SCHEMA  # agent_messages is untouched
    assert "CREATE TABLE " not in SCHEMA.replace("CREATE TABLE IF NOT EXISTS ", "")
