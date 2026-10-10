"""Opt-in owner recognition (P9): enrol a face once, then tag that person's tracks. Off by default.

    identity = IdentityService(IdentityStore(data_dir, vault), build_recognizer())
    bus.subscribe("perception.identity", identity.on_changed, replay=True)
    identity.attach(capture.pipelines, loop, bus)      # after capture.start(): one hook per camera

While it is off, or the recognizer is missing, the hook does nothing. While it is on, about once a second
per camera and only when an untagged person is in view, the faces YuNet finds are matched to the enrolled
templates; a match on two looks in a row tags the person track the face is in, published as
`perception.person.identified` (ids only). Everyone else stays what they were: an anonymous track.

Enrolling: `start_enrollment(member_id, name)`, then the next `frames` looks that show exactly one face,
facing the camera, are averaged into the template and stored encrypted. The frame is only read here:
nothing in this module changes it, stores it or publishes anything taken from it. Unmatched vectors are
dropped as soon as they were compared. Blurring of stored keyframes is not touched.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from contexts.perception.domain.identity import (
    IdentityChanged, PersonLost, TrackTags, Vector, average, cosine, face_box, track_for_face,
)
from contexts.perception.infrastructure.face_counter import is_facing
from contexts.perception.infrastructure.identity_store import MEMBER_ID, IdentityStore, Member
from contexts.perception.infrastructure.recognizer import Recognizer
from contracts.events import Event
from contracts.types import CameraId, Track

log = logging.getLogger("atlas.perception.identity")

ENROLL_FRAMES = 5
ENROLL_TIMEOUT_S = 30.0
ENROLL_EVERY_S = 0.3
MIN_FACE_SCORE = 0.8  # YuNet's confidence: below this a face is neither enrolled nor matched


class IdentityError(Exception):
    """Something the caller asked for that cannot be done now. The message is safe to show."""


class IdentityService:
    def __init__(self, store: IdentityStore, recognizer: Recognizer | None, *, frames: int = ENROLL_FRAMES,
                 timeout_s: float = ENROLL_TIMEOUT_S, clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = lambda: datetime.now(UTC), **tag_settings: float) -> None:
        self.store, self.recognizer = store, recognizer
        self.frames, self.timeout_s = frames, timeout_s
        self._clock, self._now, self._tag_settings = clock, now, tag_settings
        self._lock = threading.Lock()
        self._tags: dict[CameraId, TrackTags] = {}
        self._enroll: dict | None = None
        self._enroll_looked: dict[CameraId, float] = {}
        self._logged: tuple[bool, tuple[str, ...]] | None = None  # the last state the event log holds
        self.errors = 0

    # --- the switch and the members ---

    @property
    def available(self) -> bool:
        return self.recognizer is not None

    @property
    def enabled(self) -> bool:
        return self.available and self.store.enabled

    def settings(self) -> dict:
        out = {"identity_enabled": self.enabled, "available": self.available,
               "recognizer": self.recognizer.name if self.recognizer else None}
        if not self.available:
            out["reason"] = "the face recognition model is not installed on this box"
        return out

    def set_enabled(self, on: bool) -> list[Event]:
        if on and not self.available:
            raise IdentityError("the face recognition model is not installed on this box")
        self.store.set_enabled(on)
        if not on:
            with self._lock:
                self._enroll = None
        return self._changed()

    def members(self) -> list[dict]:
        """Who is enrolled. Never the template."""
        return [{"member_id": m.member_id, "name": m.name, "enrolled_at": m.enrolled_at} for m in self.store.members()]

    def remove(self, member_id: str) -> list[Event]:
        if not self.store.delete(member_id):
            raise KeyError(member_id)
        return self._changed()

    def _state(self) -> tuple[bool, tuple[str, ...]]:
        return self.enabled, tuple(m.member_id for m in self.store.members())

    def _changed(self) -> list[Event]:
        enabled, members = self._state()
        return [IdentityChanged(ts=self._now(), enabled=enabled, members=list(members))]

    def on_changed(self, e: Event) -> None:
        """replay=True: remember what the log last said, so `announce` only speaks when it is out of date."""
        self._logged = (bool(e.enabled), tuple(e.members))

    def announce(self) -> list[Event]:
        """At start-up, after replay: say the current state if the log does not have it (the switch and the
        templates live in the vault folder, which survives a wipe of the log)."""
        state = self._state()
        if state == (self._logged or (False, ())):
            return []
        return self._changed()

    # --- enrolment ---

    def start_enrollment(self, member_id: str | None, name: str, camera_id: CameraId | None = None) -> dict:
        if not self.enabled:
            raise IdentityError("recognition is switched off")
        member_id = member_id or f"m-{uuid4().hex[:8]}"
        name = name.strip()
        if not MEMBER_ID.match(member_id):
            raise ValueError("member_id must be 1 to 32 of a-z, 0-9, - and _")
        if not name or len(name) > 60:
            raise ValueError("name must be 1 to 60 characters")
        with self._lock:
            self._enroll = {"state": "collecting", "member_id": member_id, "name": name, "camera_id": camera_id,
                            "vectors": [], "hint": "look_at_camera", "error": None,
                            "deadline": self._clock() + self.timeout_s}
        return self.enrollment_status()

    def enrollment_status(self) -> dict:
        """Progress for the phone. No name and nothing from the picture."""
        with self._lock:
            e = self._enroll
            if e is None:
                return {"state": "idle", "frames_needed": self.frames}
            if e["state"] == "collecting" and self._clock() > e["deadline"]:
                e.update(state="failed", error="timeout", vectors=[])
            return {"state": e["state"], "member_id": e["member_id"], "camera_id": e["camera_id"],
                    "frames_done": len(e["vectors"]) if e["state"] == "collecting" else (self.frames if e["state"] == "done" else 0),
                    "frames_needed": self.frames, "hint": e["hint"] if e["state"] == "collecting" else None,
                    "error": e["error"]}

    def _enroll_step(self, camera_id: CameraId, frame: Any, rows: list[Any]) -> list[Event]:
        with self._lock:
            e = self._enroll
            if e is None or e["state"] != "collecting" or e["camera_id"] not in (None, camera_id):
                return []
            if self._clock() > e["deadline"]:
                e.update(state="failed", error="timeout", vectors=[])
                return []
            good = [r for r in rows if float(r[-1]) >= MIN_FACE_SCORE]
            if len(good) != 1:
                e["hint"] = "no_face" if not good else "one_person_only"
                return []
            if not is_facing(good[0]):
                e["hint"] = "look_at_camera"
                return []
            vector = self.recognizer.embed(self.recognizer.crop(frame, good[0]))
            if e["vectors"] and cosine(vector, e["vectors"][0]) < self.recognizer.threshold:
                e["hint"] = "hold_still"  # not the face this enrolment began with
                return []
            e["vectors"].append(vector)
            e["camera_id"], e["hint"] = camera_id, "hold_still"  # the camera that saw them first finishes the job
            if len(e["vectors"]) < self.frames:
                return []
            self.store.save(Member(member_id=e["member_id"], name=e["name"], enrolled_at=self._now().isoformat(),
                                   model=self.recognizer.name, frames=self.frames, vector=average(e["vectors"])))
            e.update(state="done", vectors=[], name="")
        return self._changed()

    # --- one look at one camera's frame ---

    def look(self, camera_id: CameraId, frame: Any, tracks: list[Track], finder: Any) -> list[Event]:
        """What this frame changes: tags gained and lost, and an enrolment that finished. `finder.faces(frame)`
        gives YuNet rows. Called from the camera's own thread, with the frame the detector just saw."""
        out = self._look(camera_id, frame, tracks, finder)
        seen_at = self._now() if out else None  # one time for everything this frame showed
        return [e.model_copy(update={"ts": seen_at}) for e in out]

    def _look(self, camera_id: CameraId, frame: Any, tracks: list[Track], finder: Any) -> list[Event]:
        tags = self._tags.get(camera_id)
        if not self.enabled:
            if tags is None:
                return []
            del self._tags[camera_id]
            return tags.reset()  # switched off: every tag ends
        now = self._clock()
        out: list[Event] = []
        if tags is None:  # first look since start-up or since it was switched on: track ids from before mean nothing
            tags = self._tags[camera_id] = TrackTags(camera_id, **self._tag_settings)
            out += tags.reset()
        templates = self._templates()
        out += tags.drop(set(templates)) + tags.seen(tracks, now)
        due = tags.due(now) if templates else set()
        enrolling = self._enrolling(camera_id, now)
        if not due and not enrolling:
            return out
        rows = finder.faces(frame)
        if enrolling:
            self._enroll_looked[camera_id] = now
            out += self._enroll_step(camera_id, frame, rows)
        if not due:
            return out
        height, width = frame.shape[:2]
        faces: dict[int, list[Any]] = {}
        for row in rows:
            track = track_for_face(face_box(row, width, height), tracks) if float(row[-1]) >= MIN_FACE_SCORE else None
            if track is not None and track.track_id in due:
                faces.setdefault(track.track_id, []).append(row)
        for track_id in due:
            mine = faces.get(track_id, [])
            if len(mine) != 1:  # no face to see, or two faces in one person's box: not a look worth trusting
                tags.missed(track_id, now)
                continue
            vector: Vector = self.recognizer.embed(self.recognizer.crop(frame, mine[0]))
            out += tags.looked(track_id, self.recognizer.match(vector, templates), now)
        return out

    def _enrolling(self, camera_id: CameraId, now: float) -> bool:
        with self._lock:
            e = self._enroll
            waiting = e is not None and e["state"] == "collecting" and e["camera_id"] in (None, camera_id)
        return waiting and now - self._enroll_looked.get(camera_id, float("-inf")) >= ENROLL_EVERY_S

    def _templates(self) -> dict[str, Vector]:
        return {m.member_id: m.vector for m in self.store.members() if m.model == self.recognizer.name}

    def tagged(self) -> dict[CameraId, dict[int, str]]:
        return {camera: tags.members for camera, tags in self._tags.items()}

    # --- the hook into a camera's pipeline ---

    def watch(self, camera_id: CameraId, detector: Callable[[Any], list[Track]], finder: Any,
              publish: Callable[[Event], Any]) -> Callable[[Any], list[Track]]:
        """`detector` unchanged, plus a look at each frame it just handled. A failure here is counted and
        never reaches the pipeline; what failed is not logged, only that it did."""

        def detect(frame: Any) -> list[Track]:
            tracks = detector(frame)
            try:
                for event in self.look(camera_id, frame, tracks, finder):
                    publish(event)
            except Exception as err:
                self.errors += 1
                if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                    log.warning("recognition failed on %s (%s), %d so far", camera_id, type(err).__name__, self.errors)
            return tracks

        return detect

    def attach(self, pipelines: dict[CameraId, Any], loop: asyncio.AbstractEventLoop, bus: Any) -> None:
        """Put the hook on every camera's tracking pipeline (its `detector` is a public attribute). The face
        finder is the pipeline's own YuNet (the one that counts and blurs faces) when it has one."""
        if not self.available:
            return

        def publish(event: Event) -> None:
            asyncio.run_coroutine_threadsafe(bus.publish(event), loop)

        for camera_id, pipeline in pipelines.items():
            finder = getattr(getattr(pipeline, "faces", None), "finder", None)
            if finder is None:
                from contexts.perception.infrastructure.face_blur import FaceBlur

                finder = FaceBlur()
            pipeline.detector = self.watch(camera_id, pipeline.detector, finder, publish)
