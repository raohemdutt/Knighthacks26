"""Boxes for the face share tests: each has its own data folder, vault, identity service (the fake recogniser), event
log and approvals, wired the way the gateway wires them. They meet only in one in-memory sealed store. No network, no
real database, no real face: a "face" is a few bytes the fake recogniser turns into a vector."""

import asyncio
import json
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bus import EventBus
from contexts.circle.application.face_share import FaceShare
from contexts.perception.tests.identity_helpers import Frame, Rig, face_in, person
from contexts.vault import KeyRing, Vault
from contexts.workspace.domain.approvals import Approvals
from contracts.experience import ApprovalDecided

T0 = datetime(2026, 10, 11, 12, tzinfo=UTC)
NAMES = {"ana": "Ana", "bob": "Bob", "mal": "Mal"}  # the Circle's roster (config): member id -> display name
# The name kept WITH a template. It is not the roster's name, so finding it anywhere in the open is a leak.
TEMPLATE_NAME = {"ana": "Zebulon Quartermaine", "bob": "Ottoline Pemberwick", "mal": "Thaddeus Wrexham"}


class Wall:
    """The one wall clock both boxes read."""

    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def pass_(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class Down:
    """A Circle store that cannot be reached, and that says so if anything touches it."""

    def __init__(self) -> None:
        self.calls = 0

    def _fail(self, *a, **kw):
        self.calls += 1
        raise ConnectionError("no route to host")

    post = inbox = delete = _fail


class Box:
    def __init__(self, root: Path, member: str, store, wall: Wall, *, enabled: bool = True, names=NAMES, days: int = 30,
                 **rig) -> None:
        self.member, self.dir, self.wall = member, root / member, wall
        self.rig = Rig(self.dir, **rig)
        self.identity = self.rig.service
        self.identity._now = wall
        self.bus = EventBus(self.dir / "events")
        self.share = FaceShare("team", member, store, self.identity, Vault(KeyRing(self.dir)), self.dir, enabled=enabled,
                               names=names, clock=wall, days=days, publish=self.bus.publish)
        self.approvals = Approvals({"circle_share": self.share.executor})
        self.bus.subscribe("agent.action.proposed", self.approvals.on_proposed, replay=True)
        self.events: list = []
        self.bus.subscribe("", self.events.append)

    # --- what the person and the timer do ---

    def enrol_self(self) -> str:
        """Enrol this box's own person (with consent) and say which Circle member they are."""
        member_id = f"m-{self.member}"
        self.publish(self.rig.enroll(member_id, TEMPLATE_NAME[self.member], self.member.encode()))
        self.publish(self.identity.update_member(member_id, circle_member_id=self.member))
        return member_id

    def publish(self, events) -> None:
        asyncio.run(self.bus.publish_all(events))

    def poll(self) -> list:
        events = self.share.poll()
        self.publish(events)
        return events

    def ask_to_share(self, friend: str):
        card = self.share.propose_share(friend)
        self.publish([card])
        return card

    def cards(self) -> list:
        return self.approvals.pending()

    def decide(self, decision: str = "approve") -> list:
        """Tap Approve (or Deny) on every waiting card. Returns how each ended."""
        async def tap():
            done = []
            for card in self.cards():
                out = await self.approvals.on_decided(ApprovalDecided(ts=self.wall(), action_id=card.id, decision=decision))
                await self.bus.publish_all(out)
                done += out
            return done

        return asyncio.run(tap())

    def sees(self, who: bytes, looks: int = 3) -> dict:
        """Someone walks past this box's camera: which track ended up tagged as which member."""
        track = person(7)
        for i in range(looks):
            self.publish(self.rig.look(Frame(face_in(track, who + b"#desk%d" % i)), [track]))
        return self.identity.tagged().get("cam-desk", {})

    def states(self, direction: str | None = None) -> list[tuple]:
        return [(e.direction, e.peer, e.state) for e in self.events
                if e.type == "circle.face_share.changed" and direction in (None, e.direction)]


def share_and_accept(a: Box, b: Box) -> None:
    """a lets b's box recognise them, and b accepts: the whole happy path."""
    a.poll(), b.poll(), a.poll()  # each publishes its key and learns the other's
    a.ask_to_share(b.member)
    assert [r.status for r in a.decide()] == ["done"]
    b.poll()
    assert [r.status for r in b.decide()] == ["done"]


def needles(name: str, vectors: list[list[float]]) -> dict[str, bytes]:
    """Every way a template (or the name kept with it) could sit somewhere in plain text."""
    out = {"name": name.encode(), "name, lower case": name.lower().encode(), "first name": name.split()[0].encode(),
           "last name": name.split()[-1].encode()}
    for i, v in enumerate(vectors):
        out[f"vector {i} as float32"] = struct.pack(f"<{len(v)}f", *v)
        out[f"vector {i} as float64"] = struct.pack(f"<{len(v)}d", *v)
        out[f"vector {i} as float32, start"] = struct.pack("<4f", *v[:4])
        out[f"vector {i} as float64, start"] = struct.pack("<2d", *v[:2])
        out[f"vector {i} as float32, big endian"] = struct.pack(">4f", *v[:4])
        out[f"vector {i} as float64, big endian"] = struct.pack(">2d", *v[:2])
        out[f"vector {i} as JSON"] = json.dumps(v).encode()
        out[f"vector {i} as compact JSON"] = json.dumps(v, separators=(",", ":")).encode()
        for j, x in enumerate(v[:6]):
            out[f"vector {i}[{j}] as text"] = repr(x).encode()
            out[f"vector {i}[{j}] to 6 places"] = f"{x:.6f}".encode()
    return out
