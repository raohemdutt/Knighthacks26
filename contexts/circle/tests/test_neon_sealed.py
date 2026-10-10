"""NeonSealedStore on the fake connection (the real database is not used by this change), and the table it needs."""

import re
from datetime import UTC, datetime
from pathlib import Path

from contexts.circle.infrastructure.neon import NeonDb
from contexts.circle.infrastructure.neon_sealed import NeonSealedStore
from contexts.circle.tests.neon_helpers import FakeNeon
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
    assert "ON CONFLICT (id) DO NOTHING" in sql
    assert params == ("s1", 1, "team", "ana", "bob", "face_share", '{"header": {"v": 1}, "ciphertext": "AAAA"}', NOW)
    assert all(table in ("circles", "members", "sealed_messages") for table in neon.tables())  # never agent_messages or shares


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
