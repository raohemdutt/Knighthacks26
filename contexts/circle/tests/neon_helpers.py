"""Helpers for the Neon tests: a fake connection for unit tests, and a throwaway
`test-…` circle for the tests that run against the real database."""

import os
from uuid import uuid4

import pytest

from contexts.circle.infrastructure.neon import ENV_URL, NeonDb

needs_neon = pytest.mark.skipif(not os.environ.get(ENV_URL), reason=f"{ENV_URL} is not set")

OWNED_TABLES = ["members", "shares", "agent_messages", "events", "habit_summaries", "impact_counters",
                "game_scores", "presence", "league_stakes", "league_weeks", "league_challenges", "sealed_messages"]


class FakeNeon:
    """Stands in for the driver: records statements, returns canned rows, can drop or refuse connections."""

    def __init__(self, rows=(), columns=()):
        self.rows, self.columns = list(rows), list(columns)
        self.statements: list[tuple[str, tuple]] = []
        self.connects = self.commits = self.drops = 0
        self.down = False

    def connect(self):
        if self.down:
            raise ConnectionError("no route to host")
        self.connects += 1
        return _FakeConnection(self)

    def tables(self) -> list[str]:
        """Tables written to, in order."""
        return [sql.split()[2] for sql, _ in self.statements if sql.startswith("INSERT INTO")]


class _FakeConnection:
    rowcount = 1

    def __init__(self, neon: FakeNeon):
        self.neon = neon

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        if self.neon.drops:
            self.neon.drops -= 1
            raise ConnectionError("server closed the connection unexpectedly")
        self.neon.statements.append((sql, tuple(params)))

    @property
    def description(self):
        return [(c,) for c in self.neon.columns]

    def fetchall(self):
        return self.neon.rows

    def commit(self):
        self.neon.commits += 1

    def rollback(self):
        pass

    def close(self):
        pass


def open_test_circle() -> tuple[NeonDb, str]:
    return NeonDb(), f"test-{uuid4().hex[:12]}"


def drop_test_circle(db: NeonDb, circle_id: str) -> int:
    """Delete the throwaway circle (everything in it cascades) and return how many of its rows are left."""

    def work(cur):
        cur.execute("DELETE FROM circles WHERE id = %s", (circle_id,))
        left = 0
        for table in OWNED_TABLES:
            cur.execute(f"SELECT count(*) FROM {table} WHERE circle_id = %s", (circle_id,))
            left += cur.fetchall()[0][0]
        return left

    try:
        return db.run(work)
    finally:
        db.close()
