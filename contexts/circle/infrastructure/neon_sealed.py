"""The Circle's sealed channel in Neon Postgres: the `sealed_messages` table (schema.sql in this folder).

    store = NeonSealedStore(NeonDb())      # the SealedStore port of contexts/circle/application/face_share.py

Same behaviour as InMemorySealedStore. A row is a box's public key record, or a signed header with ciphertext
only one member's box can read, or a signed revoke. The database never holds anything it could read a face from.
Not run against the real database as part of this change: the unit tests use the fake connection.
"""

import json

from contexts.circle.infrastructure.neon import NeonDb, _json
from contracts.circle import SealedMessage

COLUMNS = "id, v, circle_id, from_member, to_member, kind, body, created_at"
MAX_ROWS = 500


class NeonSealedStore:
    def __init__(self, db: NeonDb) -> None:
        self.db = db

    def post(self, msg: SealedMessage) -> None:
        def work(cur):
            self.db.ensure(cur, msg.circle_id, msg.from_member)
            cur.execute(
                f"INSERT INTO sealed_messages ({COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (msg.id, msg.v, msg.circle_id, msg.from_member, msg.to_member, msg.kind, json.dumps(msg.body), msg.created_at),
            )

        self.db.run(work)

    def inbox(self, circle_id: str, member_id: str) -> list[SealedMessage]:
        """The Circle's public key records and what is addressed to `member_id`, oldest first."""

        def work(cur):
            cur.execute(
                f"SELECT {COLUMNS} FROM sealed_messages WHERE circle_id = %s AND (to_member IS NULL OR to_member = %s)"
                f" ORDER BY created_at, id LIMIT {MAX_ROWS}",
                (circle_id, member_id),
            )
            return cur.fetchall()

        return [SealedMessage(id=i, v=v, circle_id=c, from_member=f, to_member=t, kind=k, body=_json(b), created_at=at)
                for i, v, c, f, t, k, b, at in self.db.run(work)]

    def delete(self, circle_id: str, ids: list[str]) -> None:
        def work(cur):
            cur.execute("DELETE FROM sealed_messages WHERE circle_id = %s AND id = ANY(%s)", (circle_id, list(ids)))

        if ids:
            self.db.run(work)
