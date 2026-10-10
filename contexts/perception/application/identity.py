"""Opt-in owner recognition (P9) and the face profile built on it. Everything here is off by default.

    identity = IdentityService(IdentityStore(data_dir, vault), build_recognizer(),
                               expression_reader=build_expression_reader(), photo_reader=build_photo_reader())
    bus.subscribe("perception.identity", identity.on_changed, replay=True)
    bus.subscribe("perception.expression", identity.on_expression, replay=True)
    identity.attach(capture.pipelines, loop, bus)      # after capture.start(): one hook per camera

While it is off, or the recognizer is missing, the hook does nothing. While it is on, about once a second
per camera and only when an untagged person is in view, the faces YuNet finds are matched to the enrolled
templates; a match on two looks in a row tags the person track the face is in, published as
`perception.person.identified` (ids only). Everyone else stays what they were: an anonymous track.

Enrolling takes the consent of the person being enrolled (`consent_ack`), and happens in one of two ways:
- live: `start_enrollment(member_id, name, consent_ack=True)`, then the next `frames` looks that show
  exactly one face, facing the camera, become the template;
- from photos: `enroll_photos(member_id, name, photos, consent_ack=True)` with 3 to 12 pictures. Each is
  decoded in memory, must show exactly one face that is large, sharp and lit well enough, and the faces
  must agree with each other. Adding photos to a member who is already enrolled is how the box "learns
  your face": the new looks join the stored exemplars (the newest MAX_EXEMPLARS are kept) and the template
  is their mean. No network is trained or fine-tuned: a template is exact, instant and deletable, and
  fine-tuning a face model on one person's photos would be none of those.
A photo is never written to disk and never logged; only the vectors made from it are kept, encrypted, in
the same vault file as before. The frame of a live look is only read: nothing in this module changes it,
stores it or publishes anything taken from it. Unmatched vectors are dropped as soon as they were compared.
Blurring of stored keyframes is not touched by any switch here.

Matching compares a look with the mean of a member's exemplars. Matching against the single closest
exemplar was measured (synthetic vectors, contexts/perception/tests/test_face_profile.py) and is worse:
one exemplar carries a whole look's noise, and the best of twelve lets more strangers through. A template
is not refreshed from live matches: a template that rewrites itself from what the camera sees can drift
onto someone else, and a biometric that changes without the person doing anything is not what they agreed to.

Apparent expression (off by default, twice): a member can switch "read my expressions" on for themselves;
then, every `expression_every_s`, the face of a track tagged as them is read by the expression model and
counted. The "everyone" switch (needs `confirm`) also counts faces that matched no member, together and
anonymously per camera: no track, no vector, no identity. Looks are added up over windows
(contexts/perception/domain/expression.py) and published as `perception.expression.observed` on a change
or every few minutes, never per frame, and never with a picture, a vector or a landmark in it.

Forgetting a member removes their template, exemplars and settings at once, and their expression history
from what this service answers. The expression events already in the event log stay there, as every event
does, until a wipe of everything.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from contexts.perception.domain.expression import LOOKS, STRAINED, ExpressionRun, Summary, head_pose, top
from contexts.perception.domain.identity import (
    IdentityChanged, PersonLost, TrackTags, Vector, average, consistent, cosine, face_box, track_for_face,
)
from contexts.perception.infrastructure.expression import ExpressionReader
from contexts.perception.infrastructure.face_counter import is_facing
from contexts.perception.infrastructure.identity_store import MAX_EXEMPLARS, MEMBER_ID, IdentityStore, Member
from contexts.perception.infrastructure.photo import PhotoReader
from contexts.perception.infrastructure.recognizer import Recognizer
from contracts.events import Event
from contracts.perception import ExpressionObserved
from contracts.types import CameraId, Track

log = logging.getLogger("atlas.perception.identity")

ENROLL_FRAMES = 5
ENROLL_TIMEOUT_S = 30.0
ENROLL_EVERY_S = 0.3
MIN_FACE_SCORE = 0.8  # YuNet's confidence: below this a face is neither enrolled nor matched

# --- photos ---
MIN_PHOTOS, MAX_PHOTOS = 3, MAX_EXEMPLARS  # a new member needs MIN_PHOTOS that agree; more can be added one at a time
MAX_PHOTO_BYTES = 8 * 1024 * 1024
MIN_FACE_PX = 96  # the shorter side of the face box; the recognizer wants 112, so less than this is mostly guessed pixels
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 50.0, 215.0  # mean grey of the face, 0..255: darker loses the features, brighter burns them out
MIN_SHARPNESS = 40.0  # variance of the Laplacian of the face at 112x112: below it there are too few edges to tell faces apart
MIN_EYE_RATIO, MAX_NOSE_OFF = 0.3, 0.6  # a slight turn is wanted (it is what the exemplars are for); a profile is not
REASONS = {
    "unreadable": "could not read this picture (JPEG or PNG, up to 8 MB)",
    "no_face": "no clear face found",
    "two_faces": "two faces: only the person enrolling should be in the picture",
    "too_small": "too small: move closer",
    "turned": "turned too far: face the camera (a slight turn is fine)",
    "too_dark": "too dark: find better light",
    "too_bright": "too bright: avoid light straight on the face",
    "blurry": "blurry: hold still and tap to focus",
    "different": "does not look like the same person as the other photos",
    "not_them": "does not look like the person already enrolled",
}

# --- apparent expression ---
EXPRESSION_EVERY_S = 2.0
KEEP_EXPRESSIONS = timedelta(hours=48)  # what GET /identity/expressions can answer about


class IdentityError(Exception):
    """Something the caller asked for that cannot be done now. The message is safe to show."""


class IdentityService:
    def __init__(self, store: IdentityStore, recognizer: Recognizer | None, *, frames: int = ENROLL_FRAMES,
                 timeout_s: float = ENROLL_TIMEOUT_S, clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = lambda: datetime.now(UTC), expression_reader: ExpressionReader | None = None,
                 photo_reader: PhotoReader | None = None, photo_finder: Any = None,
                 expression_every_s: float = EXPRESSION_EVERY_S, expression_settings: dict | None = None,
                 **tag_settings: float) -> None:
        self.store, self.recognizer = store, recognizer
        self.expression_reader, self.photo_reader, self._photo_finder = expression_reader, photo_reader, photo_finder
        self.frames, self.timeout_s = frames, timeout_s
        self.expression_every_s, self._run_settings = expression_every_s, expression_settings or {}
        self._clock, self._now, self._tag_settings = clock, now, tag_settings
        self._lock = threading.Lock()
        self._models = threading.Lock()  # one face at a time through the recognizer and the expression model: several cameras share them
        self._photos = threading.Lock()  # one photo enrolment at a time
        self._tags: dict[CameraId, TrackTags] = {}
        self._enroll: dict | None = None
        self._enroll_looked: dict[CameraId, float] = {}
        self._runs: dict[tuple[CameraId, str | None], ExpressionRun] = {}  # live only: looks being added up
        self._read_at: dict[CameraId, float] = {}
        self._observed: list[ExpressionObserved] = []  # rebuilt from the log: what GET /identity/expressions answers from
        self._logged: tuple | None = None  # the last state the event log holds
        self.errors = 0

    # --- the switches and the members ---

    @property
    def available(self) -> bool:
        return self.recognizer is not None

    @property
    def enabled(self) -> bool:
        return self.available and self.store.enabled

    @property
    def everyone(self) -> bool:
        return self.enabled and self.expression_reader is not None and self.store.everyone

    def settings(self) -> dict:
        out = {"identity_enabled": self.enabled, "available": self.available,
               "recognizer": self.recognizer.name if self.recognizer else None,
               "everyone": self.everyone, "expressions_available": self.expression_reader is not None,
               "expression_reader": self.expression_reader.name if self.expression_reader else None,
               "photos_available": self.available and self.photo_reader is not None}
        if not self.available:
            out["reason"] = "the face recognition model is not installed on this box"
        return out

    def set_enabled(self, on: bool) -> list[Event]:
        if on and not self.available:
            raise IdentityError("the face recognition model is not installed on this box")
        self.store.set_enabled(on)  # off also ends the "everyone" mode
        if not on:
            with self._lock:
                self._enroll = None
                self._runs.clear()
        return self._changed()

    def set_everyone(self, on: bool, confirm: bool = False) -> list[Event]:
        """The anonymous room mode. Switching it on has to be meant: `confirm` must be true."""
        if on:
            if not self.enabled:
                raise IdentityError("recognition is switched off")
            if self.expression_reader is None:
                raise IdentityError("the expression model is not installed on this box")
            if confirm is not True:
                raise ValueError("confirm must be true: this counts how everyone in the room looked, anonymously")
        self.store.set_everyone(on)
        if not on:
            with self._lock:
                for key in [k for k in self._runs if k[1] is None]:
                    del self._runs[key]
        return self._changed()

    def members(self) -> list[dict]:
        """Who is enrolled. Never the template."""
        return [self._row(m) for m in self.store.members()]

    @staticmethod
    def _row(m: Member) -> dict:
        return {"member_id": m.member_id, "name": m.name, "enrolled_at": m.enrolled_at, "updated_at": m.updated_at,
                "photos": len(m.exemplars), "source": m.source, "expressions": m.expressions}

    def update_member(self, member_id: str, *, expressions: bool | None = None, name: str | None = None) -> list[Event]:
        member = self.store.get(member_id)
        if member is None:
            raise KeyError(member_id)
        if expressions and self.expression_reader is None:
            raise IdentityError("the expression model is not installed on this box")
        if name is not None:
            name = self._name(name)
        self.store.save(replace(member, expressions=member.expressions if expressions is None else bool(expressions),
                                name=member.name if name is None else name))
        if expressions is False:
            self._drop_runs(member_id)
        return self._changed()

    def remove(self, member_id: str) -> list[Event]:
        if not self.store.delete(member_id):
            raise KeyError(member_id)
        self._drop_runs(member_id)
        self._observed = [e for e in self._observed if e.member_id != member_id]
        return self._changed()

    def _drop_runs(self, member_id: str) -> None:
        with self._lock:
            for key in [k for k in self._runs if k[1] == member_id]:
                del self._runs[key]  # dropped, not published: what was being counted is gone

    def _state(self) -> tuple:
        members = self.store.members()
        return (self.enabled, tuple(m.member_id for m in members), self.everyone,
                tuple(m.member_id for m in members if m.expressions))

    def _changed(self, **enrolment: Any) -> list[Event]:
        enabled, members, everyone, expressions = self._state()
        return [IdentityChanged(ts=self._now(), enabled=enabled, members=list(members), everyone=everyone,
                                expressions=list(expressions), **enrolment)]

    def on_changed(self, e: Event) -> None:
        """replay=True: remember what the log last said, so `announce` only speaks when it is out of date,
        and forget the expression history of anyone who is no longer enrolled."""
        self._logged = (bool(e.enabled), tuple(e.members), bool(getattr(e, "everyone", False)),
                        tuple(getattr(e, "expressions", ())))
        self._observed = [o for o in self._observed if o.member_id is None or o.member_id in e.members]

    def announce(self) -> list[Event]:
        """At start-up, after replay: say the current state if the log does not have it (the switch and the
        templates live in the vault folder, which survives a wipe of the log)."""
        state = self._state()
        if state == (self._logged or (False, (), False, ())):
            return []
        return self._changed()

    @staticmethod
    def _name(name: str) -> str:
        name = name.strip()
        if not name or len(name) > 60:
            raise ValueError("name must be 1 to 60 characters")
        return name

    @staticmethod
    def _consent(consent_ack: bool) -> None:
        if consent_ack is not True:
            raise ValueError("consent_ack must be true: the person being enrolled has to agree to it")

    # --- enrolment, live ---

    def start_enrollment(self, member_id: str | None, name: str, camera_id: CameraId | None = None, *,
                         consent_ack: bool = False) -> dict:
        if not self.enabled:
            raise IdentityError("recognition is switched off")
        member_id = member_id or f"m-{uuid4().hex[:8]}"
        if not MEMBER_ID.match(member_id):
            raise ValueError("member_id must be 1 to 32 of a-z, 0-9, - and _")
        name = self._name(name)
        self._consent(consent_ack)
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
            vector = self._embed(frame, good[0])
            if e["vectors"] and cosine(vector, e["vectors"][0]) < self.recognizer.threshold:
                e["hint"] = "hold_still"  # not the face this enrolment began with
                return []
            e["vectors"].append(vector)
            e["camera_id"], e["hint"] = camera_id, "hold_still"  # the camera that saw them first finishes the job
            if len(e["vectors"]) < self.frames:
                return []
            before = self.store.get(e["member_id"])  # enrolling again replaces the template; their own switch stays
            at = self._now().isoformat()
            self.store.save(Member(member_id=e["member_id"], name=e["name"], enrolled_at=at, model=self.recognizer.name,
                                   frames=self.frames, vector=average(e["vectors"]), vectors=list(e["vectors"]),
                                   expressions=bool(before and before.expressions), updated_at=at, source="live", consent_at=at))
            member_id = e["member_id"]
            e.update(state="done", vectors=[], name="")
        return self._changed(enrolled_member=member_id, enrolled_via="live", consent_ack=True)

    # --- enrolment, from photos ---

    def enroll_photos(self, member_id: str | None, name: str | None, photos: list[bytes], *,
                      consent_ack: bool = False) -> tuple[dict, list[Event]]:
        """Enrol someone from uploaded pictures, or add pictures to someone already enrolled (how the box
        "learns your face": more exemplars, a better mean; no model is trained). Returns what happened to
        each photo and the events to publish. Slow (a detector and a recognizer per photo): call it off the
        event loop. Nothing from a photo is stored, logged or returned except the vectors kept in the vault."""
        if not self.enabled:
            raise IdentityError("recognition is switched off")
        if self.photo_reader is None:
            raise IdentityError("this box cannot read photos (OpenCV is not installed)")
        if member_id is not None and not MEMBER_ID.match(member_id):
            raise ValueError("member_id must be 1 to 32 of a-z, 0-9, - and _")
        existing = self.store.get(member_id) if member_id else None
        if existing is None or name is not None:
            name = self._name(name or "")
        self._consent(consent_ack)
        least = 1 if existing else MIN_PHOTOS
        if not least <= len(photos) <= MAX_PHOTOS:
            raise ValueError(f"send {least} to {MAX_PHOTOS} photos")
        member_id = member_id or f"m-{uuid4().hex[:8]}"
        with self._photos:
            results = [{"index": i, "accepted": False, "reason": None, "message": None} for i in range(len(photos))]
            vectors: dict[int, Vector] = {}
            for i, data in enumerate(photos):
                reason, vector = self._read_photo(data)
                if vector is None:
                    results[i].update(reason=reason, message=REASONS[reason])
                else:
                    vectors[i] = vector
            threshold = self.recognizer.threshold
            same_model = existing is not None and existing.model == self.recognizer.name
            if same_model:  # more photos of someone enrolled: each has to be them
                for i in [i for i, v in vectors.items() if cosine(v, existing.vector) < threshold]:
                    del vectors[i]
                    results[i].update(reason="not_them", message=REASONS["not_them"])
            order = sorted(vectors)
            agree = {order[k] for k in consistent([vectors[i] for i in order], threshold)}
            for i in order:
                if i not in agree:
                    del vectors[i]
                    results[i].update(reason="different", message=REASONS["different"])
            for i in vectors:
                results[i]["accepted"] = True
            out = {"state": "failed", "member_id": member_id, "accepted": len(vectors), "rejected": len(photos) - len(vectors),
                   "needed": least, "photos": results, "exemplars": len(existing.exemplars) if existing else 0, "error": None}
            if len(vectors) < least:
                out["error"] = "not_enough_photos"
                return out, []
            new = [vectors[i] for i in sorted(vectors)]
            mean = average(new)
            other = self.recognizer.match(mean, {m.member_id: m.vector for m in self.store.members()
                                                 if m.member_id != member_id and m.model == self.recognizer.name})
            if other is not None:  # the same face under two ids would make every tag a coin toss
                out["error"] = "already_enrolled"
                return out, []
            exemplars = ([*existing.exemplars, *new] if same_model else new)[-MAX_EXEMPLARS:]
            at = self._now().isoformat()
            self.store.save(Member(
                member_id=member_id, name=name if name is not None else existing.name,
                enrolled_at=existing.enrolled_at if existing else at, model=self.recognizer.name, frames=len(exemplars),
                vector=average(exemplars), vectors=exemplars, expressions=bool(existing and existing.expressions),
                updated_at=at, source="photos" if not same_model else existing.source, consent_at=at))
            out.update(state="done", exemplars=len(exemplars))
        return out, self._changed(enrolled_member=member_id, enrolled_via="photos", consent_ack=True)

    def _read_photo(self, data: bytes) -> tuple[str | None, Vector | None]:
        """(why not, None) or (None, the face's vector). The picture lives only inside this call."""
        reader = self.photo_reader
        try:
            image = reader.decode(data) if 0 < len(data) <= MAX_PHOTO_BYTES else None
            if image is None:
                return "unreadable", None
            rows = self._photo_faces(image)
            good = [r for r in rows if float(r[-1]) >= MIN_FACE_SCORE]
            if len(good) != 1:
                return ("no_face" if not good else "two_faces"), None
            face = good[0]
            if min(float(face[2]), float(face[3])) < MIN_FACE_PX:
                return "too_small", None
            pose = head_pose(face)
            if pose.eye_ratio < MIN_EYE_RATIO or pose.nose_off > MAX_NOSE_OFF:
                return "turned", None
            brightness, sharpness = reader.measure(image, face)
            if brightness < MIN_BRIGHTNESS:
                return "too_dark", None
            if brightness > MAX_BRIGHTNESS:
                return "too_bright", None
            if sharpness < MIN_SHARPNESS:
                return "blurry", None
            return None, self._embed(image, face)
        except Exception as err:  # a picture the libraries choke on: say so, and never say what was in it
            log.warning("a photo could not be read (%s)", type(err).__name__)
            return "unreadable", None

    def _photo_faces(self, image: Any) -> list[Any]:
        """YuNet rows in `image`'s pixels. The detector looks at a small copy first (it misses a face that
        fills a large picture) and at the picture itself only when that found nothing."""
        if self._photo_finder is None:  # its own detector: the cameras' one is busy in their threads
            from contexts.perception.infrastructure.face_blur import FaceBlur

            self._photo_finder = FaceBlur()
        small, scale = self.photo_reader.small(image)
        rows = self._photo_finder.faces(small)
        if rows and scale != 1.0:
            return [[*(float(v) / scale for v in r[:14]), float(r[-1])] for r in rows]
        return rows if rows or scale == 1.0 else self._photo_finder.faces(image)

    def _embed(self, frame: Any, face_row: Any) -> Vector:
        with self._models:
            return self.recognizer.embed(self.recognizer.crop(frame, face_row))

    # --- one look at one camera's frame ---

    def look(self, camera_id: CameraId, frame: Any, tracks: list[Track], finder: Any) -> list[Event]:
        """What this frame changes: tags gained and lost, an enrolment that finished, and a stretch of
        expression looks that ended. `finder.faces(frame)` gives YuNet rows. Called from the camera's own
        thread, with the frame the detector just saw."""
        out = self._look(camera_id, frame, tracks, finder)
        seen_at = self._now() if out else None  # one time for everything this frame showed
        return [e.model_copy(update={"ts": seen_at}) for e in out]

    def _look(self, camera_id: CameraId, frame: Any, tracks: list[Track], finder: Any) -> list[Event]:
        tags = self._tags.get(camera_id)
        if not self.enabled:
            if tags is None:
                return []
            del self._tags[camera_id]
            with self._lock:
                for key in [k for k in self._runs if k[0] == camera_id]:
                    del self._runs[key]
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
        reading = self._reading(camera_id, tags, tracks, now)
        if not due and not enrolling and reading is None:
            return out + self._idle(camera_id)
        rows = finder.faces(frame)
        if enrolling:
            self._enroll_looked[camera_id] = now
            out += self._enroll_step(camera_id, frame, rows)
        if due:
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
                out += tags.looked(track_id, self.recognizer.match(self._embed(frame, mine[0]), templates), now)
        if reading is not None:
            self._read_at[camera_id] = now
            out += self._read_expressions(camera_id, frame, tracks, rows, tags, reading, bool(templates))
        return out + self._idle(camera_id)

    def _enrolling(self, camera_id: CameraId, now: float) -> bool:
        with self._lock:
            e = self._enroll
            waiting = e is not None and e["state"] == "collecting" and e["camera_id"] in (None, camera_id)
        return waiting and now - self._enroll_looked.get(camera_id, float("-inf")) >= ENROLL_EVERY_S

    def _templates(self) -> dict[str, Vector]:
        return {m.member_id: m.vector for m in self.store.members() if m.model == self.recognizer.name}

    def tagged(self) -> dict[CameraId, dict[int, str]]:
        return {camera: tags.members for camera, tags in self._tags.items()}

    # --- apparent expression: looks in, one fact per stretch out ---

    def _reading(self, camera_id: CameraId, tags: TrackTags, tracks: list[Track], now: float) -> set[str] | None:
        """The members whose expressions may be read on this camera now (an empty set: only the anonymous
        room), or None when there is nothing to read or it is not time yet."""
        if self.expression_reader is None or now - self._read_at.get(camera_id, float("-inf")) < self.expression_every_s:
            return None
        if not any(t.label == "person" for t in tracks):
            return None
        wanted = {m.member_id for m in self.store.members() if m.expressions} & set(tags.members.values())
        return wanted if wanted or self.everyone else None

    def _read_expressions(self, camera_id: CameraId, frame: Any, tracks: list[Track], rows: list[Any], tags: TrackTags,
                          wanted: set[str], enrolled: bool) -> list[Event]:
        height, width = frame.shape[:2]
        everyone, strangers, tagged = self.everyone, tags.strangers(), tags.members
        at = self._now()
        seen: dict[str | None, list[tuple[str | None, float, bool, bool]]] = {}
        for row in rows:
            if float(row[-1]) < MIN_FACE_SCORE:
                continue
            track = track_for_face(face_box(row, width, height), tracks)
            member = tagged.get(track.track_id) if track is not None else None
            if member in wanted:
                key = member
            elif everyone and member is None and (not enrolled or (track is not None and track.track_id in strangers)):
                key = None  # matched nobody: counted with the room, as nobody in particular
            else:
                continue  # an enrolled person who did not switch this on, or a face not yet compared: not read
            pose = head_pose(row)
            label, confidence = None, 0.0
            if not pose.away:  # the model was trained on faces towards the camera: a profile is not read
                with self._models:
                    label, confidence = top(self.expression_reader.read(frame, row))
            seen.setdefault(key, []).append((label, confidence, pose.away, pose.down))
        out: list[Event] = []
        expressive = {m.member_id for m in self.store.members() if m.expressions}
        with self._lock:
            for key in [k for k in self._runs if k[0] == camera_id and (not everyone if k[1] is None else k[1] not in expressive)]:
                del self._runs[key]  # switched off since: what was being counted is dropped, not published
            for member, looks in seen.items():
                run = self._runs.setdefault((camera_id, member), ExpressionRun(**self._run_settings))
                for label, confidence, away, down in looks:
                    out += [self._observed_event(camera_id, member, s)
                            for s in run.add(at, label, confidence, away=away, down=down, people=len(looks))]
        return out

    def _idle(self, camera_id: CameraId) -> list[Event]:
        """Stretches nobody added to for a while end, and are published."""
        if not self._runs:
            return []
        at, out = self._now(), []
        with self._lock:
            for (camera, member), run in list(self._runs.items()):
                if camera == camera_id:
                    out += [self._observed_event(camera, member, s) for s in run.idle(at)]
        return out

    @staticmethod
    def _observed_event(camera_id: CameraId, member: str | None, s: Summary) -> ExpressionObserved:
        return ExpressionObserved(camera_id=camera_id, member_id=member, window_start=s.start, window_end=s.end,
                                  dominant=s.dominant, shares=s.shares, samples=s.samples, confidence=s.confidence,
                                  looks=s.looks, head_down=round(s.down / s.looks, 3), turned_away=round(s.away / s.looks, 3),
                                  people=s.people if member is None else None)

    def on_expression(self, e: ExpressionObserved) -> None:
        """replay=True: the series the phone asks for. Kept for KEEP_EXPRESSIONS of event time."""
        self._observed.append(e)
        if self._observed[0].window_end < e.window_end - KEEP_EXPRESSIONS - timedelta(hours=1):
            self._observed = [o for o in self._observed if o.window_end >= e.window_end - KEEP_EXPRESSIONS]

    def on_wipe(self, e: Event) -> None:
        """LIVE ONLY (`vault.wipe.requested`): a wipe of everything empties the log this series came from,
        so it goes from memory too, and whatever was being counted is dropped."""
        if getattr(e, "scope", None) == "all":
            self._observed = []
            with self._lock:
                self._runs.clear()
                self._enroll = None

    def expressions(self, member_id: str | None, hours: float = 6.0, now: datetime | None = None) -> dict:
        """The stretches of the last `hours` for one member (they must be enrolled), or for the anonymous
        room when `member_id` is None. Words for people are "looked ...": see LOOKS."""
        if member_id is not None and self.store.get(member_id) is None:
            raise KeyError(member_id)
        since = (now or self._now()) - timedelta(hours=hours)
        rows = sorted((o for o in self._observed if o.member_id == member_id and o.window_end >= since), key=lambda o: o.window_start)
        counts: dict[str, float] = {}
        for o in rows:
            for label, share in o.shares.items():
                counts[LOOKS.get(label, "neutral")] = counts.get(LOOKS.get(label, "neutral"), 0.0) + share * o.samples
        total = sum(counts.values())
        strained = sum(share * o.samples for o in rows for label, share in o.shares.items() if label in STRAINED)
        return {
            "member_id": member_id, "hours": hours, "samples": sum(o.samples for o in rows),
            "looked": {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])} if total else {},
            "strained_share": round(strained / total, 3) if total else None,
            "windows": [{"start": o.window_start, "end": o.window_end, "camera_id": o.camera_id, "dominant": o.dominant,
                         "looked": LOOKS.get(o.dominant, "neutral"), "shares": o.shares, "samples": o.samples,
                         "confidence": o.confidence, "head_down": o.head_down, "turned_away": o.turned_away,
                         **({"people": o.people} if member_id is None else {})} for o in rows],
            "note": "Apparent expression: how a face looked to a small model, added up over many looks. "
                    "It is a noisy signal, not an emotion and nothing about health.",
        }

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
