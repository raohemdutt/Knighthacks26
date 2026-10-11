"""View tickets: how the owner's own signed-in screen asks for the live picture with faces showing.

    tickets = ViewTickets()
    ticket, expires_at = tickets.issue(now)      # POST /vault/view-ticket, behind the gateway's token check
    tickets.valid(ticket, now)                   # GET /capture/<camera>/snapshot.jpg?ticket=...

A picture tag cannot send the `Authorization` header the rest of the app signs in with, so the page first asks
for a ticket with that header and then puts the ticket in the picture's address. A ticket is
`<expiry, seconds>.<random id>.<HMAC-SHA256 of the two>`: nothing about a person, a camera or the box is in it.
It is good for TICKET_SECONDS and then the page asks again.

The secret is made when the process starts and never written anywhere, so a restart ends every ticket. A
ticket that is wrong in any way (expired, cut short, signed by another process, not text) is simply not a
ticket: `valid` answers False and the caller serves the blurred picture, the same as with no ticket at all.
Nothing tells the asker which of those it was.

A ticket only ever unblurs the live picture, and only while the policy's `live_faces` is "clear". It changes
nothing that is saved, exported or sent.
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

TICKET_SECONDS = 600
MAX_TICKET_CHARS = 200  # ours are about 100: anything longer is not looked at


class ViewTickets:
    def __init__(self, lifetime_s: int = TICKET_SECONDS, secret: bytes | None = None) -> None:
        self.lifetime_s = lifetime_s
        self._secret = secret or secrets.token_bytes(32)

    def _sign(self, body: str) -> str:
        return hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()

    def issue(self, now: datetime) -> tuple[str, datetime]:
        expires_at = now + timedelta(seconds=self.lifetime_s)
        body = f"{int(expires_at.timestamp())}.{secrets.token_urlsafe(12)}"
        return f"{body}.{self._sign(body)}", expires_at

    def valid(self, ticket: object, now: datetime) -> bool:
        if not isinstance(ticket, str) or not 0 < len(ticket) <= MAX_TICKET_CHARS or not ticket.isascii():
            return False
        body, _, signature = ticket.rpartition(".")
        if not hmac.compare_digest(signature, self._sign(body)):  # constant time; checked before anything is parsed
            return False
        expiry = body.partition(".")[0]
        return expiry.isdigit() and int(expiry) > now.timestamp()
