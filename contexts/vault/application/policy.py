"""Privacy policy (V3): face blur, paused cameras, offline mode and retention.

    policy = Policy(data_dir)
    policy.subscribe(bus)              # replay=True: the log rebuilds it
    policy.blur_faces                  # default ON
    policy.is_paused("cam-door")

The event log is the source of truth. <data_dir>/vault/policy.json is a
snapshot of the last applied events: it is what the policy is before replay
has run (and after a wipe has emptied the log), and every applied event
rewrites it, so after replay the file says what the log says.
"""

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel

from bus import EventBus
from contracts.types import CameraId
from contracts.vault import PolicyChanged, RetentionChanged

log = logging.getLogger("atlas.vault")


class PolicyState(BaseModel):
    blur_faces: bool = True
    paused_cameras: list[CameraId] = []
    offline: bool = False
    keep_days: int | None = None  # None = keep forever


class Policy:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.path = Path(data_dir) / "vault" / "policy.json"
        self.state = self._load()

    @property
    def blur_faces(self) -> bool:
        return self.state.blur_faces

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
        self._apply(blur_faces=e.blur_faces, paused_cameras=sorted(set(e.paused_cameras)), offline=e.offline)

    def on_retention_changed(self, e: RetentionChanged) -> None:
        self._apply(keep_days=e.keep_days)

    def _apply(self, **changes) -> None:
        new = self.state.model_copy(update=changes)
        if new != self.state or not self.path.exists():
            self.state = new
            self._save()

    def _load(self) -> PolicyState:
        try:
            return PolicyState.model_validate_json(self.path.read_text())
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
