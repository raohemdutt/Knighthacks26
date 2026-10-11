"""Privacy policy (V3): face blur, paused cameras, offline mode and retention.

    policy = Policy(data_dir)
    policy.subscribe(bus)              # replay=True: the log rebuilds it
    policy.blur_faces                  # default ON: faces are blurred in every picture that is saved
    policy.live_faces                  # "clear", "owner" or "blurred": the live picture on the owner's own signed-in screen
    policy.live_clear                  # may a ticket holder see every face?
    policy.live_owner_only             # may a ticket holder see the scanned owner's own face, and nobody else's?
    policy.train_faces                 # "blurred" (default) or "owner_clear": the owner's own face in a training export
    policy.is_paused("cam-door")

`live_faces` only ever matters to a request that carries a valid view ticket (contexts/vault/application/tickets.py):
everyone else gets the blurred live picture whatever it says, and it changes nothing about what is saved. It is
"clear" on every box unless the owner switched it off: a policy with nothing recorded for it (no policy yet, a
policy.json without the field, a `vault.policy.changed` without it) is "clear", and only a recorded "blurred"
keeps the owner's own screen blurred. `effective_live_faces` is the one place that rule is written. "owner" is
the choice in between: only the face of the person who did the face scan is clear there, every other face is
blurred, and a face the box cannot tell is blurred too (contexts/capture/application/service.py). A value this
version does not know (a newer box wrote the file or the event) is read as "blurred": cannot tell means blur.

`train_faces` is about one thing that leaves the box, the training export, and is off ("blurred") unless a scanned
owner switched it on. It decides nothing by itself: each export still needs its own approval card, and the outbound
gate (outbound.py) keeps a face clear only when it matches the owner's template. Everything else that leaves the
box is blurred whatever this says.

The event log is the source of truth. <data_dir>/vault/policy.json is a
snapshot of the last applied events: it is what the policy is before replay
has run (and after a wipe has emptied the log), and every applied event
rewrites it, so after replay the file says what the log says.
"""

import json
import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from bus import EventBus
from contracts.types import CameraId
from contracts.vault import PolicyChanged, RetentionChanged

log = logging.getLogger("atlas.vault")


LIVE_FACES = ("clear", "owner", "blurred")
TRAIN_FACES = ("blurred", "owner_clear")


def effective_live_faces(recorded: str | None) -> str:
    """What `live_faces` is when this is what was recorded: nothing recorded means "clear", and a value this
    version does not know means "blurred"."""
    if recorded is None:
        return "clear"
    return recorded if recorded in LIVE_FACES else "blurred"


def effective_train_faces(recorded: str | None) -> str:
    """Nothing recorded, or a value this version does not know: "blurred"."""
    return recorded if recorded in TRAIN_FACES else "blurred"


class PolicyState(BaseModel):
    blur_faces: bool = True
    paused_cameras: list[CameraId] = []
    offline: bool = False
    keep_days: int | None = None  # None = keep forever
    live_faces: Literal["blurred", "clear", "owner"] = "clear"  # the owner's own signed-in live view; see the module docstring
    train_faces: Literal["blurred", "owner_clear"] = "blurred"  # the owner's own face in a training export they approve

    @field_validator("live_faces", mode="before")
    @classmethod
    def _nothing_recorded_is_clear(cls, v):
        return effective_live_faces(v)

    @field_validator("train_faces", mode="before")
    @classmethod
    def _nothing_recorded_is_blurred(cls, v):
        return effective_train_faces(v)

    @property
    def live_clear(self) -> bool:
        """A request holding the owner's view ticket may see every face in the live picture."""
        return self.live_faces == "clear"

    @property
    def live_owner_only(self) -> bool:
        """A request holding the owner's view ticket may see the scanned owner's own face, and nobody else's."""
        return self.live_faces == "owner"


class Policy:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.path = Path(data_dir) / "vault" / "policy.json"
        self.state = self._load()

    @property
    def blur_faces(self) -> bool:
        return self.state.blur_faces

    @property
    def live_faces(self) -> str:
        return self.state.live_faces

    @property
    def live_clear(self) -> bool:
        return self.state.live_clear

    @property
    def live_owner_only(self) -> bool:
        return self.state.live_owner_only

    @property
    def train_faces(self) -> str:
        return self.state.train_faces

    @property
    def offline(self) -> bool:
        return self.state.offline

    @property
    def keep_days(self) -> int | None:
        return self.state.keep_days

    def is_paused(self, camera_id: CameraId) -> bool:
        return camera_id in self.state.paused_cameras

    def snapshot(self) -> dict:
        return self.state.model_dump()

    def subscribe(self, bus: EventBus) -> None:
        """Both rebuild state from the log, so both replay."""
        bus.subscribe("vault.policy.changed", self.on_policy_changed, replay=True)
        bus.subscribe("vault.retention.changed", self.on_retention_changed, replay=True)

    def on_policy_changed(self, e: PolicyChanged) -> None:
        self._apply(blur_faces=e.blur_faces, paused_cameras=sorted(set(e.paused_cameras)), offline=e.offline,
                    live_faces=effective_live_faces(getattr(e, "live_faces", None)),  # an event from before the field: clear
                    train_faces=effective_train_faces(getattr(e, "train_faces", None)))  # and from before this one: blurred

    def on_retention_changed(self, e: RetentionChanged) -> None:
        self._apply(keep_days=e.keep_days)

    def _apply(self, **changes) -> None:
        new = self.state.model_copy(update=changes)
        if new != self.state or not self.path.exists():
            self.state = new
            self._save()

    def _load(self) -> PolicyState:
        try:
            raw = json.loads(self.path.read_text())
            return PolicyState.model_validate(raw)
        except FileNotFoundError:
            return PolicyState()
        except ValueError:  # unreadable snapshot: fall back to the private defaults; replay corrects it
            log.warning("ignoring unreadable %s", self.path)
            return PolicyState()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2) + "\n")
        os.replace(tmp, self.path)  # never a half-written policy
