from datetime import datetime
from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import CameraId, FrameRef, ObjectHit, Track, ZoneId


@register
class ObservationRecorded(TypedEvent):
    """Sampled ~1/s per camera."""

    type: Literal["perception.observation.recorded"] = "perception.observation.recorded"
    camera_id: CameraId
    tracks: list[Track]
    frame_ref: FrameRef | None = None


@register
class ZoneEvent(TypedEvent):
    type: Literal["perception.zone.event"] = "perception.zone.event"
    camera_id: CameraId
    track_id: int
    label: str
    zone_id: ZoneId
    kind: Literal["enter", "exit", "dwell"]
    dwell_s: float | None = None  # set when kind == "dwell"
    frame_ref: FrameRef | None = None


@register
class SceneDescribed(TypedEvent):
    type: Literal["perception.scene.described"] = "perception.scene.described"
    camera_id: CameraId
    caption: str
    objects: list[ObjectHit] = []
    zone_id: ZoneId | None = None  # set when the frame is a zone crop (watch-item finder)
    frame_ref: FrameRef
    model_version: str


# --- owner recognition (P9, opt-in): ids only, never a name, a picture or a face vector ---


@register
class PersonIdentified(TypedEvent):
    """Track `track_id` on this camera is the enrolled member `member_id`."""

    type: Literal["perception.person.identified"] = "perception.person.identified"
    camera_id: CameraId
    track_id: int
    member_id: str
    circle_member_id: str | None = None  # the Circle member this enrolled person was linked to, if any: an id only


@register
class PersonLost(TypedEvent):
    """That track's tag is over."""

    type: Literal["perception.person.lost"] = "perception.person.lost"
    camera_id: CameraId
    track_id: int | None = None  # None: every track on this camera (the pipeline restarted, or recognition went off)


@register
class IdentityChanged(TypedEvent):
    """Recognition was switched on or off, or the set of enrolled members changed."""

    type: Literal["perception.identity.changed"] = "perception.identity.changed"
    enabled: bool
    members: list[str] = []  # enrolled member ids
    everyone: bool = False  # the anonymous room mode: faces that match nobody add to a room-level count, never to a person
    expressions: list[str] = []  # the enrolled members who switched "read my expressions" on for themselves
    # When this event records an enrolment: who (the id only), how, and that the person was confirmed to agree.
    # `ts` is when. None on every other change (a switch, a removal).
    enrolled_member: str | None = None
    enrolled_via: Literal["live", "photos", "scan"] | None = None  # the box's camera, uploaded photos, a scan on the phone
    consent_ack: bool | None = None
    circle_member_id: str | None = None  # the Circle member the enrolled person was linked to, if any: an id only


# --- apparent expression (opt-in per member; anonymous for a whole room only in the "everyone" mode) ---


@register
class ExpressionObserved(TypedEvent):
    """How a face looked over a stretch of time, added up from many looks. Never one frame, and never a
    picture, a face vector or a landmark. "Apparent expression" is a noisy reading of a face's shape by a
    small classifier: it is not an emotion, a mood or anything about health.

    `member_id` is the enrolled member who switched this on for themselves, or None for the anonymous room
    aggregate of the "everyone" mode (every face that matched no member, counted together: no track, no id)."""

    type: Literal["perception.expression.observed"] = "perception.expression.observed"
    camera_id: CameraId
    member_id: str | None = None
    window_start: datetime
    window_end: datetime
    dominant: str  # the most frequent of: angry, disgust, fearful, happy, neutral, sad, surprised
    shares: dict[str, float] = {}  # label -> fraction of the counted looks (they add up to 1)
    samples: int  # counted looks: the face was towards the camera and the classifier was sure enough
    confidence: float  # the classifier's mean certainty over the counted looks, 0 to 1
    looks: int | None = None  # every look in the window, also those not counted
    head_down: float | None = None  # fraction of the looks with the head tipped down, relative to the camera
    turned_away: float | None = None  # fraction of the looks turned away from the camera
    people: int | None = None  # anonymous room only: the most faces seen in one look
