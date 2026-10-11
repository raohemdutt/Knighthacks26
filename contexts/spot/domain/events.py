"""The inspection context's events.

    spot.replay.seeked        the replay was moved to a position (0 = rewound); the live picture is rebuilt up to it
    spot.mission.started      a mission began
    spot.action.started       the robot began an action at a waypoint
    spot.action.completed     an action succeeded (its reading, if any, rides along)
    spot.action.failed        an action failed, with its failure type
    spot.mission.completed    a mission ended, with how many actions failed inside it
    spot.anomaly.raised       an anomaly row attached to an action that just ended
    spot.alert.raised         something an operator should see now (a failure, an anomaly, a mission that hid failures)
    spot.work_order.created   an approved re-inspection or work order was written to this box's disk

These are NOT in contracts/ (adding there needs a second person's OK), so they are not in the contract registry:
they are published as the typed classes below, and from the log they come back as a plain `Event` with the same
fields, which is all the projection needs (the way contexts/perception first shipped its identity events).

`ts` is when the event was published, so the log keeps the order things were shown in across re-runs and seeks.
`data_ts` is the kit's own event time; durations and ages are always computed from `data_ts`.
"""

from typing import ClassVar, Literal

from contracts.events import TypedEvent


class SpotEvent(TypedEvent):
    seq: int = -1  # position in the replay stream; -1 for events that are not part of it
    data_ts: str | None = None  # ISO time from the kit, no zone
    source: Literal["replay", "tables"] = "replay"  # "tables": rebuilt from the CSVs because the stream stops short


class ReplaySeeked(SpotEvent):
    TYPE: ClassVar[str] = "spot.replay.seeked"
    type: Literal["spot.replay.seeked"] = "spot.replay.seeked"
    index: int  # how many stream events are in the picture after the seek


class MissionStarted(SpotEvent):
    TYPE: ClassVar[str] = "spot.mission.started"
    type: Literal["spot.mission.started"] = "spot.mission.started"
    mission_id: str
    scenario: str = ""
    configuration_version: str = ""
    battery_pct: float | None = None


class ActionStarted(SpotEvent):
    TYPE: ClassVar[str] = "spot.action.started"
    type: Literal["spot.action.started"] = "spot.action.started"
    mission_id: str
    action_id: str
    location_id: str = ""
    asset_id: str = ""
    action_type: str = ""
    battery_pct: float | None = None
    sequence: int | None = None


class ActionCompleted(SpotEvent):
    TYPE: ClassVar[str] = "spot.action.completed"
    type: Literal["spot.action.completed"] = "spot.action.completed"
    mission_id: str
    action_id: str
    location_id: str = ""
    asset_id: str = ""
    action_type: str = ""
    reading: dict | None = None  # {reading_id, measurement_type, value, unit, quality}


class ActionFailed(SpotEvent):
    TYPE: ClassVar[str] = "spot.action.failed"
    type: Literal["spot.action.failed"] = "spot.action.failed"
    mission_id: str
    action_id: str
    location_id: str = ""
    asset_id: str = ""
    action_type: str = ""
    failure_type: str = ""
    reading: dict | None = None


class MissionCompleted(SpotEvent):
    TYPE: ClassVar[str] = "spot.mission.completed"
    type: Literal["spot.mission.completed"] = "spot.mission.completed"
    mission_id: str
    status: str  # COMPLETED | FAILED, as the stream gives it
    end_battery_pct: float | None = None
    failed_actions: int = 0
    anomalies: int = 0


class AnomalyRaised(SpotEvent):
    TYPE: ClassVar[str] = "spot.anomaly.raised"
    type: Literal["spot.anomaly.raised"] = "spot.anomaly.raised"
    mission_id: str
    action_id: str
    anomaly_id: str
    asset_id: str = ""
    location_id: str = ""
    anomaly_type: str = ""
    severity: str = ""
    confirmed: bool = False  # raised by an action that succeeded


class AlertRaised(SpotEvent):
    TYPE: ClassVar[str] = "spot.alert.raised"
    type: Literal["spot.alert.raised"] = "spot.alert.raised"
    alert_id: str
    level: Literal["info", "warn", "high", "critical"]
    message: str
    mission_id: str = ""
    action_id: str | None = None
    anomaly_id: str | None = None
    asset_id: str | None = None
    location_id: str | None = None


class WorkOrderCreated(SpotEvent):
    TYPE: ClassVar[str] = "spot.work_order.created"
    type: Literal["spot.work_order.created"] = "spot.work_order.created"
    work_order_id: str
    kind: Literal["reinspect", "work_order", "mission_request"]  # mission_request: a round planned in Shift Run
    asset_id: str = ""
    anomaly_id: str | None = None
    reason: str = ""
    path: str = ""  # the file on this box
    action_id: str = ""  # the approval that allowed it
