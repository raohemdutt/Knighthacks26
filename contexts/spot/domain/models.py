"""The SPOT challenge kit as typed rows, joined: MISSION -> ACTION -> READING / ANOMALY, with LOCATION and ASSET.

Rows are frozen value objects; `Kit` holds them with the indexes the analysis needs. Nothing here reads a file
(see infrastructure/kit.py) or does statistics (see domain/analysis.py).

Timestamps in the kit carry no time zone. They are kept as naive datetimes and only ever compared with each
other, so no zone is assumed for the analysis.
"""

from collections import defaultdict
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
CRITICALITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
INSPECTION_PREFIX = "NAVIGATION"  # every action type except this one looks at an asset


class Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class Mission(Row):
    mission_id: str
    mission_name: str = ""
    scenario: str = ""
    configuration_version: str = ""
    start_time: datetime
    end_time: datetime
    duration_min: float
    start_battery_pct: float
    end_battery_pct: float
    mission_status: str
    operator_mode: str = ""


class Action(Row):
    action_id: str
    mission_id: str
    sequence: int
    timestamp: datetime
    location_id: str
    asset_id: str
    action_type: str
    battery_pct: float
    status: str  # SUCCESS | FAILED
    failure_type: str = ""
    duration_sec: float
    scenario: str = ""
    configuration_version: str = ""

    @property
    def failed(self) -> bool:
        return self.status != "SUCCESS"

    @property
    def is_inspection(self) -> bool:
        return self.action_type != INSPECTION_PREFIX


class Reading(Row):
    reading_id: str
    action_id: str
    mission_id: str
    asset_id: str
    location_id: str
    timestamp: datetime
    measurement_type: str
    value: float
    unit: str = ""
    quality: str = ""  # GOOD | QUESTIONABLE

    @property
    def good(self) -> bool:
        return self.quality == "GOOD"


class Anomaly(Row):
    anomaly_id: str
    action_id: str
    mission_id: str
    location_id: str
    asset_id: str
    timestamp: datetime
    anomaly_type: str
    severity: str
    operator_review_required: bool = False
    description: str = ""


class Location(Row):
    location_id: str
    location_name: str = ""
    zone_type: str = ""
    floor: int = 1
    x_coord: float = 0.0
    y_coord: float = 0.0
    environment: str = ""


class Asset(Row):
    asset_id: str
    asset_name: str = ""
    asset_type: str = ""
    location_id: str = ""
    criticality: str = "LOW"


class ReplayEvent(Row):
    """One line of replay.ndjson, or one rebuilt from the tables when the stream stops short (`source="tables"`)."""

    event_id: str
    mission_id: str
    timestamp: datetime
    event_type: str  # MISSION_STARTED | ACTION_STARTED | ACTION_COMPLETED | ACTION_FAILED | MISSION_COMPLETED
    details: dict = {}
    source: Literal["replay", "tables"] = "replay"


class Issue(BaseModel):
    """Something in the files a person should know about before trusting a number."""

    code: str
    level: Literal["info", "warn", "error"]
    message: str
    count: int = 0
    ids: list[str] = []


class Kit:
    """The loaded kit. `issues` is what the loader found; `validation` is the sponsor's validation_results.json."""

    def __init__(self, *, missions: list[Mission], actions: list[Action], readings: list[Reading], anomalies: list[Anomaly],
                 locations: list[Location], assets: list[Asset], replay: list[ReplayEvent], issues: list[Issue] | None = None,
                 validation: dict | None = None, path: str = "", asset_files: list[str] | None = None) -> None:
        self.missions = sorted(missions, key=lambda m: (m.start_time, m.mission_id))
        self.actions = sorted(actions, key=lambda a: (a.timestamp, a.mission_id, a.sequence))
        self.readings = sorted(readings, key=lambda r: (r.timestamp, r.reading_id))
        self.anomalies = sorted(anomalies, key=lambda a: (a.timestamp, a.anomaly_id))
        self.locations = sorted(locations, key=lambda x: x.location_id)
        self.assets = sorted(assets, key=lambda x: x.asset_id)
        self.replay = replay
        self.issues = issues or []
        self.validation = validation
        self.path = path
        self.asset_files = asset_files or []  # image paths relative to assets_dir
        self.assets_dir: str | None = None

        self.mission = {m.mission_id: m for m in self.missions}
        self.action = {a.action_id: a for a in self.actions}
        self.location = {x.location_id: x for x in self.locations}
        self.asset = {x.asset_id: x for x in self.assets}
        self.anomaly = {a.anomaly_id: a for a in self.anomalies}
        self.actions_of: dict[str, list[Action]] = defaultdict(list)
        for a in sorted(self.actions, key=lambda a: (a.mission_id, a.sequence)):
            self.actions_of[a.mission_id].append(a)
        self.readings_of: dict[str, list[Reading]] = defaultdict(list)  # by action
        for r in self.readings:
            self.readings_of[r.action_id].append(r)
        self.anomalies_of: dict[str, list[Anomaly]] = defaultdict(list)  # by action
        for an in self.anomalies:
            self.anomalies_of[an.action_id].append(an)

    @property
    def now(self) -> datetime | None:
        """The newest moment in the data. "Recent" and "old" are measured back from here, never from the wall clock."""
        stamps = [m.end_time for m in self.missions] + [a.timestamp for a in self.actions]
        return max(stamps) if stamps else None

    def anomalies_in(self, mission_id: str) -> list[Anomaly]:
        return [an for an in self.anomalies if an.mission_id == mission_id]

    def failed_in(self, mission_id: str) -> list[Action]:
        return [a for a in self.actions_of.get(mission_id, []) if a.failed]
