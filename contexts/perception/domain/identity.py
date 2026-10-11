"""Owner recognition (P9), the pure part: the events, the vector maths, which track a face belongs to,
and which tracks carry a member's tag.

    perception.person.identified   track N on camera C is member M (ids only: no name, no picture, no vector)
    perception.person.lost         that track's tag is over; track_id None = every track on the camera
    perception.identity.changed    recognition switched on or off, or the set of enrolled member ids changed

The three event classes live in contracts/perception.py (registered, so the log gives them back typed) and
are re-exported here, where they were first defined.

A face belongs to the person track whose box contains it. A track is tagged after `confirm` matches for the
same member in a row; a face that matches nobody never tags anything; a tagged track keeps its tag while it
lasts, also when the face turns away. While the face is visible the tag is checked again now and then, and
dropped after `strikes` looks that did not match (the tracker swapped two people's ids).
"""

import math
from dataclasses import dataclass
from typing import Any

from contracts.events import Event
from contracts.perception import IdentityChanged, PersonIdentified, PersonLost
from contracts.types import Box, CameraId, Track

__all__ = ["IdentityChanged", "PersonIdentified", "PersonLost", "TrackTags", "Vector", "average", "best_match", "consistent",
           "cosine", "face_box", "normalize", "track_for_face"]

Vector = list[float]


# --- vectors ---


def normalize(v: Vector) -> Vector:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


def cosine(a: Vector, b: Vector) -> float:
    if len(a) != len(b) or not a:
        return -1.0  # vectors from different models never match
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else -1.0


def average(vectors: list[Vector]) -> Vector:
    """One template from several looks: each look counts the same, whatever its length."""
    units = [normalize(v) for v in vectors]
    return normalize([sum(col) / len(units) for col in zip(*units, strict=True)])


def best_match(vector: Vector, templates: dict[str, Vector], threshold: float) -> str | None:
    """The member whose template is closest, if it is at least `threshold` close. Otherwise nobody."""
    scored = sorted(((cosine(vector, t), m) for m, t in templates.items()), reverse=True)
    return scored[0][1] if scored and scored[0][0] >= threshold else None


def consistent(vectors: list[Vector], threshold: float) -> list[int]:
    """Which of these looks are the same face, as indexes. Each look is compared with the mean of the
    others; while one is less than `threshold` close, the one furthest from the rest is taken out and the
    rest are asked again, so a stray picture cannot drag the mean. Looks that agree with nobody leave
    nothing. With fewer than two there is nothing to compare, and all are kept."""
    keep = list(range(len(vectors)))
    while len(keep) >= 2:
        scores = {i: cosine(vectors[i], average([vectors[j] for j in keep if j != i])) for i in keep}
        worst = min(keep, key=lambda i: (scores[i], -i))
        if scores[worst] >= threshold:
            return keep
        keep.remove(worst)
    return keep if len(vectors) < 2 else []


# --- faces and tracks ---


def face_box(row: Any, width: int, height: int) -> Box:
    """A YuNet row ([x, y, w, h, ...] in pixels) as a normalized box."""
    x, y, w, h = (float(v) for v in row[:4])
    return Box(x1=x / width, y1=y / height, x2=(x + w) / width, y2=(y + h) / height)


def track_for_face(face: Box, tracks: list[Track]) -> Track | None:
    """The person whose box contains the face's centre. With several (people overlapping), the one whose
    head (top-centre of the box) is nearest the face."""
    cx, cy = (face.x1 + face.x2) / 2, (face.y1 + face.y2) / 2
    half = (face.y2 - face.y1) / 2
    inside = [t for t in tracks if t.label == "person" and t.box.x1 <= cx <= t.box.x2 and t.box.y1 <= cy <= t.box.y2]
    return min(inside, key=lambda t: math.hypot((t.box.x1 + t.box.x2) / 2 - cx, t.box.y1 + half - cy), default=None)


@dataclass
class _Tag:
    member: str | None = None  # None until confirmed
    candidate: str | None = None
    votes: int = 0
    tries: int = 0
    unknown: int = 0  # looks at this face that matched no member, in a row
    strikes: int = 0
    next_look: float = 0.0
    last_seen: float = 0.0


class TrackTags:
    """Which tracks on one camera are which member. Times are a monotonic clock in seconds."""

    def __init__(self, camera_id: CameraId, *, confirm: int = 2, retry_s: float = 1.0, patient_tries: int = 10,
                 slow_retry_s: float = 5.0, verify_every_s: float = 10.0, strikes: int = 3,
                 forget_after_s: float = 5.0) -> None:
        self.camera_id, self.confirm = camera_id, confirm
        self.retry_s, self.patient_tries, self.slow_retry_s = retry_s, patient_tries, slow_retry_s
        self.verify_every_s, self.strikes, self.forget_after_s = verify_every_s, strikes, forget_after_s
        self._tags: dict[int, _Tag] = {}

    @property
    def members(self) -> dict[int, str]:
        return {tid: t.member for tid, t in self._tags.items() if t.member}

    def seen(self, tracks: list[Track], now: float) -> list[Event]:
        """Note who is in view; tags of tracks gone for `forget_after_s` end."""
        for t in tracks:
            if t.label == "person":
                self._tags.setdefault(t.track_id, _Tag(next_look=now)).last_seen = now
        out = []
        for tid in [tid for tid, tag in self._tags.items() if now - tag.last_seen >= self.forget_after_s]:
            if self._tags.pop(tid).member:
                out.append(PersonLost(camera_id=self.camera_id, track_id=tid))
        return out

    def strangers(self) -> set[int]:
        """Tracks whose face was looked at `confirm` times or more and matched no member: not an enrolled
        person, as far as the box can tell."""
        return {tid for tid, t in self._tags.items() if t.member is None and t.candidate is None and t.unknown >= self.confirm}

    def due(self, now: float) -> set[int]:
        """Tracks worth a look at their face now: untagged ones, and tagged ones due a check."""
        return {tid for tid, tag in self._tags.items() if tag.last_seen == now and now >= tag.next_look}

    def looked(self, track_id: int, member: str | None, now: float) -> list[Event]:
        """One look at this track's face matched `member` (None: nobody enrolled)."""
        tag = self._tags.get(track_id)
        if tag is None:
            return []
        if tag.member:  # a check of a tag that already stands
            tag.next_look = now + self.verify_every_s
            tag.strikes = 0 if member == tag.member else tag.strikes + 1
            if tag.strikes < self.strikes:
                return []
            self._tags[track_id] = _Tag(next_look=now + self.retry_s, last_seen=tag.last_seen)
            return [PersonLost(camera_id=self.camera_id, track_id=track_id)]
        self.missed(track_id, now)
        if member is None:
            tag.candidate, tag.votes, tag.unknown = None, 0, tag.unknown + 1
            return []
        tag.votes, tag.unknown = (tag.votes + 1 if member == tag.candidate else 1), 0
        tag.candidate = member
        if tag.votes < self.confirm:
            return []
        tag.member, tag.strikes, tag.next_look = member, 0, now + self.verify_every_s
        return [PersonIdentified(camera_id=self.camera_id, track_id=track_id, member_id=member)]

    def missed(self, track_id: int, now: float) -> None:
        """A look that found no face for this track: try again later, less often for someone who stays unknown."""
        tag = self._tags.get(track_id)
        if tag is None:
            return
        if tag.member:
            tag.next_look = now + self.retry_s  # the face turned away: the tag stays, the check waits for the face
            return
        tag.tries += 1
        tag.next_look = now + (self.retry_s if tag.tries < self.patient_tries else self.slow_retry_s)

    def drop(self, keep: set[str]) -> list[Event]:
        """End the tags of members who are no longer enrolled."""
        out = []
        for tid, tag in self._tags.items():
            if tag.member and tag.member not in keep:
                self._tags[tid] = _Tag(next_look=tag.next_look, last_seen=tag.last_seen)
                out.append(PersonLost(camera_id=self.camera_id, track_id=tid))
        return out

    def reset(self) -> list[Event]:
        """Forget every track: one `lost` for the whole camera."""
        self._tags.clear()
        return [PersonLost(camera_id=self.camera_id)]
