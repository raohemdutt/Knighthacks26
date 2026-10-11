"""The Circle's sealed channel in Neon Postgres: the `sealed_messages` table (schema.sql in this folder).

    store = NeonSealedStore(NeonDb())      # the SealedStore port of contexts/circle/application/face_share.py

Same behaviour as InMemorySealedStore, with one addition below. A row is a box's public key record, or a signed
header with ciphertext only one member's box can read, or a signed revoke. The database never holds anything it
could read a face from.

Expiry on the database's side. A face_share carries its expiry inside the signed header, which only the two boxes
act on. `post` copies that same value into the row's `expires_at` column, so that without reading a body:
- `inbox` leaves out a row that is past it (on the database's clock): a share nobody accepted in time is no longer
  handed to anyone, whether or not a box ever deletes it;
- the sender's box sweeps it (NeonCleanup in neon.py), so ciphertext addressed to a friend whose box never polls
  does not stay for good.
A key record and a revoke have no expiry and are stored with NULL, as is every row written by a box from before
the column. Nothing a box decides rests on the column: the friend's box still checks the signed header itself.

A database that does not have the column yet (the owner has not run `python -m contexts.circle.schema --apply`)
is found out on the first statement and used the way it was before it: see NeonDb.run_with.
"""

import json
from datetime import datetime

from contexts.circle.infrastructure.neon import SEALED_EXPIRY, NeonDb, _json
from contracts.circle import SealedMessage

COLUMNS = "id, v, circle_id, from_member, to_member, kind, body, created_at"
MAX_ROWS = 500


def expiry(msg: SealedMessage) -> datetime | None:
    """The expiry in the message's own header (a time with an offset), or None when it has none."""
    header = msg.body.get("header")
    text = header.get("expires_at") if isinstance(header, dict) else None
    try:
        at = datetime.fromisoformat(text) if isinstance(text, str) else None
    except ValueError:
        return None
    return at if at is not None and at.tzinfo is not None else None


class NeonSealedStore:
    def __init__(self, db: NeonDb) -> None:
        self.db = db

    def post(self, msg: SealedMessage) -> None:
        row = (msg.id, msg.v, msg.circle_id, msg.from_member, msg.to_member, msg.kind, json.dumps(msg.body), msg.created_at)

        def insert(columns: str, values: tuple):
            def work(cur):
                self.db.ensure(cur, msg.circle_id, msg.from_member)
                cur.execute(f"INSERT INTO sealed_messages ({columns}) VALUES ({', '.join(['%s'] * len(values))})"
                            " ON CONFLICT (id) DO NOTHING", values)

            return work

        self.db.run_with(SEALED_EXPIRY, insert(COLUMNS + ", expires_at", (*row, expiry(msg))), before=insert(COLUMNS, row))

    def inbox(self, circle_id: str, member_id: str) -> list[SealedMessage]:
        """The Circle's public key records and what is addressed to `member_id`, oldest first. A row past its
        `expires_at` is left out."""

        def select(alive: str):
            def work(cur):
                cur.execute(
                    f"SELECT {COLUMNS} FROM sealed_messages WHERE circle_id = %s AND (to_member IS NULL OR to_member = %s)"
                    f"{alive} ORDER BY created_at, id LIMIT {MAX_ROWS}",
                    (circle_id, member_id),
                )
                return cur.fetchall()

            return work

        rows = self.db.run_with(SEALED_EXPIRY, select(" AND (expires_at IS NULL OR expires_at > now())"), before=select(""))
        return [SealedMessage(id=i, v=v, circle_id=c, from_member=f, to_member=t, kind=k, body=_json(b), created_at=at)
                for i, v, c, f, t, k, b, at in rows]

    def delete(self, circle_id: str, ids: list[str]) -> None:
        def work(cur):
            cur.execute("DELETE FROM sealed_messages WHERE circle_id = %s AND id = ANY(%s)", (circle_id, list(ids)))

        if ids:
            self.db.run(work)
