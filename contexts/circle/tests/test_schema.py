import re
from pathlib import Path

from contracts.circle import ShareKind

SCHEMA = Path("contexts/circle/infrastructure/schema.sql").read_text()
TABLES = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA))


def test_every_shared_table_exists():
    assert TABLES >= {
        "circles", "members", "events", "rsvps", "habit_summaries",
        "impact_counters", "game_scores", "agent_messages", "shares", "presence",
    }  # fmt: skip


def test_schema_is_safe_to_rerun():
    assert "CREATE TABLE " not in SCHEMA.replace("CREATE TABLE IF NOT EXISTS ", "")
    assert "CREATE INDEX " not in SCHEMA.replace("CREATE INDEX IF NOT EXISTS ", "")


def test_share_kinds_match_the_contract():
    kinds = re.search(r"kind\s+text NOT NULL CHECK \(kind IN \(([^)]*)\)\),\n\s+payload", SCHEMA).group(1)
    assert set(re.findall(r"'(\w+)'", kinds)) == set(ShareKind.__args__)
