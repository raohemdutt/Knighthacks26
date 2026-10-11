"""Face share: a person enrolled on their own box lets named friends' boxes recognise them, without a second
enrolment. Off by default (`circle.face_share` in config/agent.yaml, or ATLAS_CIRCLE_FACE_SHARE=1).

    share = FaceShare(circle_id, member_id, store, identity, vault, data_dir, enabled=..., names={...}, clock=...)
    share.fallback = approvals.executors.get("circle_share")     # the League wraps the same executor this way
    approvals.executors["circle_share"] = share.executor
    bus.subscribe("vault.wipe.requested", bus.in_background(share.on_wipe))   # live only
    share.poll()                                                  # on the app's timer: returns events to publish

The rules, each of which has a test (contexts/circle/tests/test_face_share*.py):

- Only the template, only one's own. What leaves is the mean vector, the exemplar vectors, the display name and
  the Circle member id of the ONE person enrolled on this box, with consent, as this box's own Circle member
  (`IdentityService.own_template`). Never a picture, a frame, a crop or expression history, never someone else who
  enrolled here, and never a template this box itself received.
- Sealed to one friend's box (contexts/circle/application/sealed.py). The shared database and every other member
  see ciphertext. Each share is signed, addressed, numbered and dated; any change is refused.
- Both people decide. "Let <friend>'s Atlas recognise me" raises an approval card on the sharer's box; nothing is
  sent until Approve. An incoming share raises a card on the friend's box; only on Accept is it decrypted and
  stored (as a member with `source: "circle"`), and it is used only while recognition is on there.
- It can be taken back and it runs out. `revoke(friend)` posts a signed revoke and removes the ciphertext; the
  friend's box deletes the template at its next poll. Every share expires (30 days by default): the friend's box
  deletes it then unless it was renewed, also when the Circle cannot be reached. The friend can forget it at any
  time, and a wipe of everything removes it with every other template.
- Nowhere else. Events, logs, errors and API answers carry ids, states and times only.

Switched off: no key is made or published, nothing is read from or written to the Circle on a box that never had
it on, and the endpoints answer 409. On a box that had it on, switching it off takes everything back at the next
poll: this member's shares are revoked, templates received from friends are deleted, and a share that still
arrives is ignored and reported as ignored (`circle.face_share.changed`, state "ignored", reason "switched_off").

The sealed channel is its own store (`SealedStore`: `InMemorySealedStore` here, the `sealed_messages` table in
infrastructure/neon_sealed.py). It is not `agent_messages`: `CircleAgent.poll` never sees these rows, so a box
without this code, or with it switched off, is not affected by them.

What this box keeps for it is one vault blob, <data_dir>/vault/circle/face_share.bin: the box's two private keys,
the friends' public keys as first seen, what was shared with whom until when, and per friend the highest counter
seen and revoked (so an old share cannot be played back).
"""

import asyncio
import inspect
import json
import logging
import os
import threading
from base64 import b64decode
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from contexts.circle.application import sealed
from contexts.circle.application.sealed import BoxKeys, PublicKeys, SealError
from contexts.circle.application.sharing import CircleOffline
from contexts.perception.application.identity import IdentityError
from contracts.agent import ActionProposed
from contracts.circle import SEALED_VERSION, FaceShareChanged, SealedMessage
from contracts.events import Event

log = logging.getLogger("atlas.circle.face_share")

CONTEXT = "circle.face_share"  # the vault context of the state file
DEFAULT_DAYS, MAX_DAYS = 30, 90
MAX_PLAINTEXT = 256 * 1024  # a template as JSON: 13 vectors of 128 numbers are about 35 kB
MAX_SEALED = 400_000  # a row's body as JSON; anything larger is not read
MAX_ROWS = 500  # rows looked at per poll
MAX_ID = 128  # characters in a row's id or member ids: the database lets anyone write anything, so longer is not read
FUTURE = timedelta(days=1)  # how far ahead of this box's clock a share may be dated
KEEP_IGNORED = 1000  # more than MAX_ROWS, so a row is reported once, not once a poll
TEMPLATE_FIELDS = {"v", "circle_member_id", "name", "model", "vector", "vectors"}
EXPLANATION = ("Your face template goes only to the friends you choose, encrypted so only their box can read it. "
               "You can take it back at any time.")
OFF = ("face sharing between friends' boxes is switched off on this box (circle.face_share in config/agent.yaml, "
       "or ATLAS_CIRCLE_FACE_SHARE=1)")
NO_CIRCLE = "this box has not joined a Circle"


class FaceShareOff(Exception):
    """The switch is off (or there is no Circle). The message is the plain reason."""


class FaceShareError(Exception):
    """Something asked for that cannot be done now. The message is a fixed sentence, safe to show and to log."""


class SealedStore(Protocol):
    """The Circle's sealed channel. Reads give the public key records of the Circle and what is addressed to one
    member; nothing else."""

    def post(self, msg: SealedMessage) -> None: ...

    def inbox(self, circle_id: str, member_id: str) -> list[SealedMessage]: ...

    def delete(self, circle_id: str, ids: list[str]) -> None: ...


class InMemorySealedStore:
    """Two boxes in one process (tests, one laptop). Same behaviour as the Neon table, with one difference: it has
    no clock, so a share past its expiry is still handed out (the Neon inbox leaves it out). Either way the
    receiving box checks the signed header's own expiry before it offers or accepts anything."""

    def __init__(self) -> None:
        self.rows: list[SealedMessage] = []

    def post(self, msg: SealedMessage) -> None:
        if all(r.id != msg.id for r in self.rows):
            self.rows.append(msg)

    def inbox(self, circle_id: str, member_id: str) -> list[SealedMessage]:
        return sorted((r for r in self.rows if r.circle_id == circle_id and r.to_member in (None, member_id)),
                      key=lambda r: (r.created_at, r.id))

    def delete(self, circle_id: str, ids: list[str]) -> None:
        self.rows = [r for r in self.rows if not (r.circle_id == circle_id and r.id in ids)]


class IdentityPort(Protocol):
    """What face share needs from perception (IdentityService fits). Templates cross only through these."""

    def settings(self) -> dict: ...

    def own_template(self, circle_member_id: str) -> dict: ...

    def shared_members(self) -> list: ...

    def receive_shared(self, shared_by: str, name: str, model: str, vector: list, vectors: list, *, expires_at: datetime,
                       share_id: str) -> list[Event]: ...

    def forget_shared(self, shared_by: str) -> list[Event]: ...

    def expire_shared(self, now: datetime) -> tuple[list[str], list[Event]]: ...


def _when(text: Any) -> datetime:
    at = datetime.fromisoformat(text) if isinstance(text, str) else None
    if at is None or at.tzinfo is None:
        raise ValueError("not a time with an offset")
    return at


class FaceShare:
    def __init__(self, circle_id: str, member_id: str, store: SealedStore, identity: IdentityPort, vault: Any,
                 data_dir: str | Path, *, enabled: bool = False, names: dict[str, str] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC), days: int = DEFAULT_DAYS, tz: Any = UTC,
                 publish: Callable[[Event], Any] | None = None) -> None:
        self.circle_id, self.member_id = circle_id, member_id
        self.store, self.identity, self.vault = store, identity, vault
        self.enabled = bool(enabled)
        self.names = {str(k): str(v) for k, v in (names or {}).items() if v}
        self.clock, self.tz = clock, tz
        self.days = min(max(int(days), 1), MAX_DAYS)
        self.publish = publish  # how the executor and the wipe publish what they did (bus.publish)
        self.fallback: Callable | None = None  # the circle_share executor for everything that is not a face share
        self.path = Path(data_dir) / "vault" / "circle" / "face_share.bin"
        self._lock = threading.RLock()  # the timer's thread and the API's both come through here
        self._state: dict | None = None
        self._keys: BoxKeys | None = None
        self._cards: dict[str, tuple[str, datetime]] = {}  # friend -> the share card waiting on this box (memory only)
        self._carry: list[Event] = []  # events of a poll that then failed at the network: published by the next one
        self.skipped = 0  # rows of a kind or version this box does not know, seen by the last poll

    # --- what this box keeps: one vault blob ---

    def _load(self) -> dict:
        if self._state is None:
            state: dict = {}
            try:
                state = json.loads(self.vault.decrypt(CONTEXT, self.path.read_bytes()))
            except FileNotFoundError:
                pass
            except Exception as err:  # another master key, or a damaged file: the old box key is gone with it
                log.warning("the face share state could not be read (%s); it is set aside and a new box key is made",
                            type(err).__name__)
                os.replace(self.path, self.path.with_name(self.path.name + ".unreadable"))
            for key in ("pins", "out", "in", "ignored"):
                state.setdefault(key, {})
            state.setdefault("seq", 0)
            self._state = state
        return self._state

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(self.vault.encrypt(CONTEXT, json.dumps(self._load()).encode()))
        os.replace(tmp, self.path)

    def _used(self) -> bool:
        """This box had face share on at some point (it has a box key). A box that never did stays silent."""
        return self._state is not None and "keys" in self._state or self.path.is_file() and "keys" in self._load()

    def _box(self) -> BoxKeys:
        """This box's key pair, made the first time face share is used while switched on."""
        if self._keys is None:
            state = self._load()
            if "keys" in state:
                self._keys = BoxKeys.from_bytes(b64decode(state["keys"]))
            else:
                self._keys = BoxKeys.generate()
                state["keys"] = sealed.b64(self._keys.to_bytes())
                self._save()
        return self._keys

    def _seq(self, now: datetime) -> int:
        """A counter that only goes up, also across restarts and a clock set back: every share and revoke takes one."""
        state = self._load()
        state["seq"] = max(int(state["seq"]) + 1, int(now.timestamp() * 1000))
        return state["seq"]

    def _in(self, friend: str) -> dict:
        return self._load()["in"].setdefault(friend, {"seen": 0, "revoked": 0, "accepted": 0, "pending": None})

    def _pin(self, friend: str) -> PublicKeys | None:
        pin = self._load()["pins"].get(friend)
        return PublicKeys(b64decode(pin["sign"]), b64decode(pin["enc"])) if pin else None

    def _name(self, member: str) -> str:
        return self.names.get(member, member)

    def _day(self, at: datetime) -> str:
        local = at.astimezone(self.tz)
        return f"{local:%b} {local.day}"

    def _changed(self, now: datetime, peer: str, direction: str, state: str, **more: Any) -> FaceShareChanged:
        return FaceShareChanged(ts=now, circle_id=self.circle_id, member_id=self.member_id, peer=peer,
                                direction=direction, state=state, **more)

    def _on(self) -> None:
        if not self.enabled:
            raise FaceShareOff(OFF)

    def _friend(self, friend: str) -> str:
        if friend == self.member_id or (self.names and friend not in self.names):
            raise KeyError(friend)
        return friend

    def _delete(self, ids: list[str]) -> None:
        """Take rows out of the shared store. Tidying only: nothing depends on it having worked."""
        if not ids:
            return
        try:
            self.store.delete(self.circle_id, ids)
        except Exception as err:
            log.warning("face share: %d row(s) could not be removed from the Circle store (%s)", len(ids), type(err).__name__)

    # --- the sharer's side ---

    def propose_share(self, friend: str) -> ActionProposed:
        """The card on the sharer's own box. Nothing is sealed or sent until it is approved."""
        with self._lock:
            self._on()
            self._friend(friend)
            now = self.clock()
            keys = self._usable_key(friend)
            self._own()  # refuses here, before a card, when there is nothing of one's own to share
            until = now + timedelta(days=self.days)
            card = ActionProposed(
                ts=now, action_id=uuid4().hex, kind="circle_share", target=self.circle_id,
                reason=(f"Let {self._name(friend)}'s Atlas recognise you: send your face template (no photos), encrypted so "
                        f"only their box can read it, until {self._day(until)}. Their box key: {keys.fingerprint}."),
                params={"kind": "face_share", "to": friend, "fingerprint": keys.fingerprint})
            self._cards[friend] = (card.action_id, now + timedelta(minutes=5))
            return card

    def _own(self) -> dict:
        try:
            return self.identity.own_template(self.member_id)
        except IdentityError as err:
            raise FaceShareError(str(err)) from None

    def _usable_key(self, friend: str) -> PublicKeys:
        pin = self._load()["pins"].get(friend)
        if pin is None:
            raise FaceShareError(f"{self._name(friend)}'s box has not published a key yet: face sharing has to be "
                                 "switched on there too")
        if pin.get("other"):
            raise FaceShareError(f"{self._name(friend)}'s box key changed: compare the new fingerprint with them and trust "
                                 "it first")
        return self._pin(friend)

    def _send(self, friend: str, fingerprint: str) -> tuple[str, list[Event]]:
        self._on()
        now = self.clock()
        keys = self._usable_key(friend)
        if keys.fingerprint != fingerprint:
            raise FaceShareError(f"{self._name(friend)}'s box key is not the one you were shown: nothing was sent")
        mine = self._own()
        plaintext = sealed.canonical({"v": SEALED_VERSION, "circle_member_id": self.member_id, "name": mine["name"],
                                      "model": mine["model"], "vector": mine["vector"], "vectors": mine["vectors"]})
        if len(plaintext) > MAX_PLAINTEXT:
            raise FaceShareError("the template is too large to share")
        share_id, until = uuid4().hex, now + timedelta(days=self.days)
        header = {"v": SEALED_VERSION, "kind": "face_share", "circle_id": self.circle_id, "from": self.member_id, "to": friend,
                  "share_id": share_id, "seq": self._seq(now), "issued_at": now.isoformat(), "expires_at": until.isoformat(),
                  "model": mine["model"]}
        body = sealed.seal(self._box(), header, plaintext, keys)
        del plaintext, mine
        state = self._load()
        before = state["out"].get(friend)
        self.store.post(SealedMessage(id=share_id, circle_id=self.circle_id, from_member=self.member_id, to_member=friend,
                                      kind="face_share", body=body, created_at=now))
        state["out"][friend] = {"share_id": share_id, "seq": header["seq"], "shared_at": header["issued_at"],
                                "expires_at": header["expires_at"]}
        self._save()
        self._cards.pop(friend, None)
        if before:
            self._delete([before["share_id"]])  # a renewal replaces the older ciphertext
        return (f"sent to {self._name(friend)}'s Atlas, which can recognise you until {self._day(until)} once they accept",
                [self._changed(now, friend, "out", "shared", share_id=share_id, expires_at=until)])

    def revoke(self, friend: str) -> list[Event]:
        """Take a share back: a signed revoke the friend's box obeys at its next poll, and the ciphertext removed."""
        with self._lock:
            self._on()
            if friend not in self._load()["out"]:
                raise KeyError(friend)
            return self._revoke(friend, self.clock())

    def _revoke(self, friend: str, now: datetime) -> list[Event]:
        state = self._load()
        shared = state["out"][friend]
        header = {"v": SEALED_VERSION, "kind": "face_revoke", "circle_id": self.circle_id, "from": self.member_id,
                  "to": friend, "seq": self._seq(now), "issued_at": now.isoformat()}
        self.store.post(SealedMessage(id=uuid4().hex, circle_id=self.circle_id, from_member=self.member_id, to_member=friend,
                                      kind="face_revoke", body=sealed.sign_revoke(self._box(), header), created_at=now))
        del state["out"][friend]
        self._save()
        self._delete([shared["share_id"]])
        return [self._changed(now, friend, "out", "revoked", share_id=shared["share_id"])]

    def _revoke_all(self, now: datetime) -> list[Event]:
        return [e for friend in list(self._load()["out"]) for e in self._revoke(friend, now)]

    # --- the receiver's side ---

    def review(self, friend: str) -> ActionProposed:
        """Raise the Accept card again for an offer that is still waiting (a card lasts five minutes)."""
        with self._lock:
            self._on()
            pending = self._in(self._friend(friend))["pending"]
            if pending is None:
                raise KeyError(friend)
            return self._card(friend, pending, self.clock())

    def _card(self, friend: str, pending: dict, now: datetime) -> ActionProposed:
        settings = self.identity.settings()
        keys = self._pin(friend)
        notes = ""
        if not settings.get("identity_enabled"):
            notes += " Recognition is switched off on this box, so it will not be used until you switch it on."
        if settings.get("recognizer") and pending["model"] != settings["recognizer"]:
            notes += " It was made with a different face model than this box uses, so it cannot be used here yet."
        return ActionProposed(
            ts=now, action_id=uuid4().hex, kind="circle_share", target=self.circle_id,
            reason=(f"{self._name(friend)} wants your Atlas to recognise them. Accept? Their face template (no photos) is "
                    f"stored encrypted on this box and deleted on {self._day(_when(pending['expires_at']))} unless they "
                    f"renew it. Their box key: {keys.fingerprint if keys else 'unknown'}.{notes}"),
            params={"kind": "face_accept", "from": friend, "share_id": pending["share_id"]})

    def _accept(self, friend: str, share_id: str) -> tuple[str, list[Event]]:
        self._on()
        now = self.clock()
        gone = FaceShareError(f"{self._name(friend)}'s offer is no longer there: it was taken back, replaced or it ran out")
        entry = self._in(friend)
        pending, keys = entry["pending"], self._pin(friend)
        if pending is None or pending["share_id"] != share_id or pending["seq"] <= entry["revoked"] or keys is None:
            raise gone
        if self._load()["pins"][friend].get("other"):
            raise FaceShareError(f"{self._name(friend)}'s box key changed: compare the new fingerprint with them first")
        row = next((r for r in self.store.inbox(self.circle_id, self.member_id)
                    if r.id == share_id and r.kind == "face_share" and r.from_member == friend), None)
        if row is None:
            raise gone
        try:
            header = sealed.verify_share(row.body, keys)
            if header != pending["header"] or _when(header["expires_at"]) <= now:
                raise gone
            raw = sealed.open_sealed(self._box(), row.body, keys)
            if len(raw) > MAX_PLAINTEXT:
                raise ValueError
            template = json.loads(raw)
            del raw
            if not isinstance(template, dict) or set(template) != TEMPLATE_FIELDS or template["v"] != SEALED_VERSION \
                    or template["circle_member_id"] != friend or template["model"] != header["model"]:
                raise ValueError  # a box may only share its own person: the template has to say it is the sender
            events = self.identity.receive_shared(friend, template["name"], template["model"], template["vector"],
                                                  template["vectors"], expires_at=_when(header["expires_at"]), share_id=share_id)
            del template
        except (SealError, IdentityError) as err:  # fixed sentences
            raise FaceShareError(str(err)) from None
        except (ValueError, TypeError, KeyError):  # never the text of what was inside
            raise FaceShareError("the shared template is not well formed: nothing was stored") from None
        entry.update(pending=None, accepted=pending["seq"])
        self._save()
        self._delete([share_id])  # it is on this box now, under this box's own key: the copy in the Circle can go
        until = _when(pending["expires_at"])
        text = f"{self._name(friend)} will be recognised by this box until {self._day(until)}"
        if not self.identity.settings().get("identity_enabled"):
            text += " once recognition is switched on"
        return text, [*events, self._changed(now, friend, "in", "received", share_id=share_id, expires_at=until)]

    def forget(self, friend: str) -> list[Event]:
        """This box's own "forget": delete what the friend shared, or turn down an offer that is waiting."""
        with self._lock:
            self._on()
            entry = self._load()["in"].get(friend) or {}
            pending = entry.get("pending")
            events = self.identity.forget_shared(friend)
            if not events and pending is None:
                raise KeyError(friend)
            now = self.clock()
            if pending is not None:
                entry["pending"] = None
                self._save()
                self._delete([pending["share_id"]])
            return [*events, self._changed(now, friend, "in", "forgotten" if events else "declined",
                                           share_id=pending["share_id"] if pending and not events else None)]

    def trust_key(self, friend: str, fingerprint: str) -> None:
        """After the two friends compared it: take the changed key as the friend's key from now on."""
        with self._lock:
            self._on()
            pin = self._load()["pins"].get(self._friend(friend))
            other = pin and pin.get("other")
            if not other:
                raise KeyError(friend)
            if PublicKeys(b64decode(other["sign"]), b64decode(other["enc"])).fingerprint != fingerprint.strip().lower():
                raise FaceShareError("that is not the fingerprint of the new key")
            pin.update(sign=other["sign"], enc=other["enc"], other=None, first_seen=self.clock().isoformat())
            self._save()

    # --- the approval executor: runs only after the person tapped Approve ---

    async def executor(self, req: Any) -> str:
        kind = req.params.get("kind")
        if kind not in ("face_share", "face_accept"):
            if self.fallback is None:
                raise ValueError(f"unsupported share {kind}")
            out = self.fallback(req)
            return await out if inspect.isawaitable(out) else out
        try:
            text, events = await asyncio.to_thread(self._approved, kind, req.params)
        except (FaceShareError, FaceShareOff, CircleOffline):
            raise  # fixed sentences: they end up in the log as the card's result
        except Exception as err:  # anything else could quote what it was handling: say only that it failed
            log.warning("face share: an approved %s failed (%s)", kind, type(err).__name__)
            raise FaceShareError("the face share could not be completed: nothing was changed") from None
        for event in events:
            await self._publish(event)
        return text

    def _approved(self, kind: str, params: dict) -> tuple[str, list[Event]]:
        with self._lock:
            if kind == "face_share":
                return self._send(str(params.get("to")), str(params.get("fingerprint")))
            return self._accept(str(params.get("from")), str(params.get("share_id")))

    async def _publish(self, event: Event) -> None:
        if self.publish is not None:
            out = self.publish(event)
            if inspect.isawaitable(out):
                await out

    async def on_wipe(self, e: Event) -> list[Event]:
        """LIVE ONLY (`vault.wipe.requested`, scope "all"): the templates friends shared went with every other
        template; this takes this member's own shares back too, and forgets what was waiting."""
        if getattr(e, "scope", None) != "all" or not self._used():
            return []
        return await asyncio.to_thread(self._wiped)

    def _wiped(self) -> list[Event]:
        with self._lock:
            now, out = self.clock(), []
            try:
                out = self._revoke_all(now)
            except Exception as err:  # offline: what was shared still runs out on its own
                log.warning("face share: the shares could not be taken back at the wipe (%s); they expire", type(err).__name__)
            state = self._load()
            state["out"], state["ignored"] = {}, {}
            for entry in state["in"].values():
                entry["pending"] = None
            self._cards.clear()
            self._save()
            return out

    # --- the timer ---

    def poll(self) -> list[Event]:
        """Run on the app's timer. In this order: delete what ran out (no network needed), publish this box's key,
        obey revokes, raise a card for each new share. Returns the events to publish."""
        with self._lock:
            now = self.clock()
            out, self._carry = self._carry + self._expire(now), []
            if not self.enabled and not self._used():
                return out  # never switched on here: nothing is read, written or kept
            try:
                self._sync(now, out)
            except Exception as err:
                if self._state is not None:
                    self._save()
                if not self.enabled:  # switched off: the Circle is tried again next time, quietly
                    log.debug("face share (off): the Circle could not be read (%s)", type(err).__name__)
                    return out
                self._carry = out  # what already happened on this box is said by the next poll
                raise
            return out

    def _expire(self, now: datetime) -> list[Event]:
        whose, events = self.identity.expire_shared(now)
        out: list[Event] = [*events, *(self._changed(now, friend, "in", "expired") for friend in whose)]
        if self._state is None and not self.path.is_file():
            return out
        state, dirty = self._load(), False
        for friend, shared in list(state["out"].items()):
            if _when(shared["expires_at"]) <= now:
                del state["out"][friend]
                dirty = True
                self._delete([shared["share_id"]])
                out.append(self._changed(now, friend, "out", "expired", share_id=shared["share_id"]))
        for friend, entry in state["in"].items():
            pending = entry.get("pending")
            if pending and _when(pending["expires_at"]) <= now:
                entry["pending"], dirty = None, True
                state["ignored"][pending["share_id"]] = "expired"  # said here; the row is not reported a second time
                self._delete([pending["share_id"]])
                out.append(self._changed(now, friend, "in", "expired", share_id=pending["share_id"]))
        if dirty:
            self._save()
        return out

    def _sync(self, now: datetime, out: list[Event]) -> None:
        """Adds to `out` as it goes, so what already happened is still said when the network then fails."""
        state = self._load()
        if not self.enabled:  # switched off after it was on: everything is taken back, this box's own side first
            for m in self.identity.shared_members():
                out += [*self.identity.forget_shared(m.shared_by), self._changed(now, m.shared_by, "in", "forgotten")]
            for entry in state["in"].values():
                entry["pending"] = None
            self._save()
            out += self._revoke_all(now)
        else:
            try:
                self.identity.own_template(self.member_id)
            except Exception:  # this member is no longer enrolled here as themself: what they shared goes too
                out += self._revoke_all(now)
        rows = self.store.inbox(self.circle_id, self.member_id)[:MAX_ROWS]
        known = [r for r in rows if r.v == SEALED_VERSION and r.kind in ("box_key", "face_share", "face_revoke")
                 and max(len(r.id), len(r.from_member), len(r.to_member or "")) <= MAX_ID]
        self.skipped = len(rows) - len(known)  # a newer version's rows, or not rows of ours at all: left alone
        if self.enabled:
            self._publish_key(known, now)
        else:  # no key of a box that has it switched off stays published
            self._delete([r.id for r in known if r.kind == "box_key" and r.from_member == self.member_id])
        self._read_keys(known, now)
        mine = [r for r in known if r.to_member == self.member_id and r.from_member != self.member_id]
        for row in mine:
            if row.kind == "face_revoke":
                out += self._revoked(row, now)
        for row in mine:
            if row.kind == "face_share":
                out += self._offered(row, now)
        self._save()

    def _publish_key(self, rows: list[SealedMessage], now: datetime) -> None:
        keys = self._box()
        row_id = f"key:{self.member_id}:{keys.public.fingerprint.replace(' ', '')}"
        own = [r for r in rows if r.kind == "box_key" and r.from_member == self.member_id]
        if all(r.id != row_id for r in own):
            self.store.post(SealedMessage(id=row_id, circle_id=self.circle_id, from_member=self.member_id, kind="box_key",
                                          body=sealed.key_record(keys, self.circle_id, self.member_id, now.isoformat()),
                                          created_at=now))
        stale = [r.id for r in own if r.id != row_id]
        if stale:
            self._delete(stale)  # an older key of this member (the box was set up again)

    def _read_keys(self, rows: list[SealedMessage], now: datetime) -> None:
        """Trust on first use: the first key seen for a member is kept. A different one later is never used
        silently: it is held as `other` until the person compares fingerprints and trusts it."""
        pins = self._load()["pins"]
        seen: dict[str, list[PublicKeys]] = {}
        for row in rows:
            if row.kind != "box_key" or row.from_member == self.member_id or (self.names and row.from_member not in self.names):
                continue
            try:
                keys = sealed.read_key_record(row.body, self.circle_id, row.from_member)
            except SealError:
                continue  # not signed by itself, or about someone else: not a key
            seen.setdefault(row.from_member, []).append(keys)
        for member, found in seen.items():
            if member not in pins:
                first = found[0]
                pins[member] = {"sign": sealed.b64(first.sign), "enc": sealed.b64(first.enc), "first_seen": now.isoformat(),
                                "other": None}
            pinned = self._pin(member)
            other = next((k for k in reversed(found) if k != pinned), None)
            pins[member]["other"] = {"sign": sealed.b64(other.sign), "enc": sealed.b64(other.enc)} if other else None
        for member, pin in pins.items():
            if member not in seen:
                pin["other"] = None

    def _ignore(self, row: SealedMessage, reason: str, now: datetime) -> list[Event]:
        """Reported once per row. Only "switched_off" is looked at again later (when it is switched on)."""
        ignored = self._load()["ignored"]
        if ignored.get(row.id) == reason:
            return []
        ignored[row.id] = reason
        for old in list(ignored)[:-KEEP_IGNORED]:
            del ignored[old]
        log.info("face share: a share from %s was ignored (%s)",
                 row.from_member if row.from_member in self.names else "someone not in the Circle's member list", reason)
        return [self._changed(now, row.from_member, "in", "ignored", share_id=row.id, reason=reason)]

    def _offered(self, row: SealedMessage, now: datetime) -> list[Event]:
        ignored = self._load()["ignored"]
        if not self.enabled:
            return self._ignore(row, "switched_off", now)
        if ignored.get(row.id, "switched_off") != "switched_off":
            return []
        if self.names and row.from_member not in self.names:
            return self._ignore(row, "unknown_member", now)
        if len(sealed.canonical(row.body)) > MAX_SEALED:
            return self._ignore(row, "too_large", now)
        keys = self._pin(row.from_member)
        if keys is None:
            return []  # their key record has not been seen yet: looked at again at the next poll
        try:
            header = sealed.verify_share(row.body, keys)
        except SealError:
            return self._ignore(row, "bad_signature", now)
        try:
            issued, until, seq = _when(header["issued_at"]), _when(header["expires_at"]), header["seq"]
            well_formed = (header["v"] == SEALED_VERSION and header["kind"] == "face_share"
                           and header["circle_id"] == self.circle_id and header["from"] == row.from_member
                           and header["to"] == self.member_id and header["share_id"] == row.id and type(seq) is int and seq > 0
                           and isinstance(header["model"], str) and len(header["model"]) <= 64
                           and issued <= now + FUTURE and until - issued <= timedelta(days=MAX_DAYS))
        except (KeyError, ValueError, TypeError):
            well_formed = False
        if not well_formed:
            return self._ignore(row, "malformed", now)
        if until <= now:
            return self._ignore(row, "expired", now)
        entry = self._in(row.from_member)
        if seq <= max(entry["seen"], entry["revoked"]):
            if entry["pending"] and entry["pending"]["share_id"] == row.id:
                return []  # the offer already waiting for an answer
            return self._ignore(row, "replayed", now)  # older than what was already seen or taken back
        ignored.pop(row.id, None)
        entry["seen"] = seq
        entry["pending"] = {"share_id": row.id, "seq": seq, "issued_at": header["issued_at"], "expires_at": header["expires_at"],
                            "model": header["model"], "offered_at": now.isoformat(), "header": header}
        return [self._changed(now, row.from_member, "in", "offered", share_id=row.id, expires_at=until),
                self._card(row.from_member, entry["pending"], now)]

    def _revoked(self, row: SealedMessage, now: datetime) -> list[Event]:
        keys = self._pin(row.from_member)
        if keys is None:
            return []
        try:
            header = sealed.verify_revoke(row.body, keys)
            seq = header["seq"]
            if not (header["v"] == SEALED_VERSION and header["kind"] == "face_revoke" and header["circle_id"] == self.circle_id
                    and header["from"] == row.from_member and header["to"] == self.member_id and type(seq) is int):
                return []
        except (SealError, KeyError, TypeError):
            return []  # not from the friend's key: nobody else can make this box delete something
        entry = self._in(row.from_member)
        self._delete([row.id])
        if seq <= entry["revoked"]:
            return []
        entry["revoked"] = seq
        out: list[Event] = []
        pending = entry["pending"]
        if pending and pending["seq"] < seq:
            entry["pending"] = None
        if entry["accepted"] < seq:
            out += self.identity.forget_shared(row.from_member)
        if out or pending:
            out.append(self._changed(now, row.from_member, "in", "revoked", share_id=pending["share_id"] if pending else None))
        return out

    # --- what the screen shows ---

    def status(self) -> dict:
        """Who can recognise whom, both ways, with key fingerprints. Ids, names from the config, times, states."""
        with self._lock:
            self._on()
            now, state = self.clock(), self._load()
            settings = self.identity.settings()
            me: dict[str, Any] = {"fingerprint": self._box().public.fingerprint, "member": None, "can_share": False,
                                  "why_not": None}
            try:
                own = self.identity.own_template(self.member_id)
                me.update(member={"member_id": own["member_id"], "name": own["name"]}, can_share=True)
                del own
            except Exception as err:
                me["why_not"] = str(err)
            received = {m.shared_by: m for m in self.identity.shared_members()}
            everyone = dict.fromkeys([*self.names, *state["pins"], *state["out"], *state["in"], *received])
            friends = []
            for member in everyone:
                if member == self.member_id:
                    continue
                pin, keys = state["pins"].get(member), self._pin(member)
                other = pin and pin.get("other")
                shared, entry, got = state["out"].get(member), state["in"].get(member) or {}, received.get(member)
                card = self._cards.get(member)
                outgoing = {"state": "none", "share_id": None, "shared_at": None, "expires_at": None, "action_id": None}
                if shared:
                    outgoing.update(state="shared", share_id=shared["share_id"], shared_at=shared["shared_at"],
                                    expires_at=shared["expires_at"])
                if card and card[1] > now:
                    outgoing.update(state="waiting_approval" if not shared else "shared", action_id=card[0])
                incoming = {"state": "none", "received_at": None, "expires_at": None, "offered_at": None, "usable": None}
                pending = entry.get("pending")
                if got:
                    incoming.update(state="accepted", received_at=got.received_at, expires_at=got.expires_at,
                                    usable=bool(settings.get("identity_enabled")) and got.model == settings.get("recognizer"))
                if pending:  # an offer (or a renewal) waiting for this box's person
                    incoming.update(state="offered" if not got else "accepted", offered_at=pending["offered_at"],
                                    **({} if got else {"expires_at": pending["expires_at"]}))
                friends.append({
                    "member_id": member, "display_name": self._name(member),
                    "key": {"known": keys is not None, "fingerprint": keys.fingerprint if keys else None,
                            "changed": bool(other),
                            "new_fingerprint": PublicKeys(b64decode(other["sign"]), b64decode(other["enc"])).fingerprint if other else None},
                    "outgoing": outgoing, "incoming": incoming, "renewal_offered": bool(pending and got)})
            return {"enabled": True, "circle_id": self.circle_id, "member_id": self.member_id, "expiry_days": self.days,
                    "recognition_enabled": bool(settings.get("identity_enabled")),
                    "recognition_available": bool(settings.get("available")), "me": me, "friends": friends,
                    "explanation": EXPLANATION}
