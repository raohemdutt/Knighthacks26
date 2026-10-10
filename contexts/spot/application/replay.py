"""The replay as a live operational picture.

    stream = build_stream(kit)                 # replay.ndjson in event-time order, completed from the tables where it stops short
    live = SpotLive(kit, stream)               # the picture: mission in progress, where the robot is, counts, alerts
    bus.subscribe("spot.", live.on_event, replay=True)   # rebuilt from the log after a restart
    player = Player(kit, stream, bus, live)    # publishes the stream on the bus at a chosen speed
    await player.start(); await player.pause(); await player.seek(120); player.set_speed(240)

One stream event becomes one or more bus events (`events_for`): the mission or action event itself, an
`spot.anomaly.raised` for each anomaly row on an action that just ended, and an `spot.alert.raised` for anything
an operator should see at that moment. `SpotLive` is a pure fold over those events. A seek publishes one
`spot.replay.seeked`; the picture is then rebuilt by folding the same `events_for` output up to that position, so
playing to a point and seeking to it give the same picture, and so does a restart that replays the log.

Speed is data seconds per wall second. Gaps longer than `MAX_WAIT_S` of wall time are cut (missions are eight
hours apart). The stream gives an action's start and end the same timestamp, so an action is shown as running
for its own `duration_sec` (capped), to let a person see it happen.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import datetime

from contexts.spot.domain.events import (ActionCompleted, ActionFailed, ActionStarted, AlertRaised, AnomalyRaised, MissionCompleted,
                                         MissionStarted, ReplaySeeked, SpotEvent)
from contexts.spot.domain.models import Kit, ReplayEvent

SPEEDS = (30.0, 60.0, 240.0, 1200.0)
DEFAULT_SPEED = 60.0
MAX_WAIT_S = 1.5  # the longest a gap in the data is allowed to take on screen
MAX_DWELL_S = 1.0  # the longest an action is shown as running
MIN_DWELL_S = 0.05
MAX_ALERTS = 40
TRAIL = 12
ACTION_END = ("ACTION_COMPLETED", "ACTION_FAILED")
ALERT_LEVEL = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "warn", "LOW": "info"}


def _words(s: str) -> str:
    return s.replace("_", " ").lower()


def build_stream(kit: Kit) -> list[ReplayEvent]:
    """Every mission's events in event-time order: the kit's replay lines, plus what the tables say is missing."""
    have_start = {e.mission_id for e in kit.replay if e.event_type == "MISSION_STARTED"}
    have_end = {e.mission_id for e in kit.replay if e.event_type == "MISSION_COMPLETED"}
    begun = {e.details.get("action_id") for e in kit.replay if e.event_type == "ACTION_STARTED"}
    ended = {e.details.get("action_id") for e in kit.replay if e.event_type in ACTION_END}
    rebuilt: list[ReplayEvent] = []

    def add(mission_id: str, ts: datetime, kind: str, details: dict) -> None:
        rebuilt.append(ReplayEvent(event_id=f"T{len(rebuilt) + 1:05d}", mission_id=mission_id, timestamp=ts, event_type=kind,
                                   details=details, source="tables"))

    for m in kit.missions:
        if m.mission_id not in have_start:
            add(m.mission_id, m.start_time, "MISSION_STARTED",
                {"scenario": m.scenario, "configuration_version": m.configuration_version, "battery_pct": m.start_battery_pct})
        for a in kit.actions_of.get(m.mission_id, []):
            if a.action_id not in begun:
                add(m.mission_id, a.timestamp, "ACTION_STARTED", {"action_id": a.action_id, "location_id": a.location_id,
                                                                  "asset_id": a.asset_id, "action_type": a.action_type, "battery_pct": a.battery_pct})
            if a.action_id not in ended:
                add(m.mission_id, a.timestamp, "ACTION_FAILED" if a.failed else "ACTION_COMPLETED",
                    {"action_id": a.action_id, "status": a.status, "failure_type": a.failure_type})
        if m.mission_id not in have_end:
            add(m.mission_id, m.end_time, "MISSION_COMPLETED", {"status": m.mission_status, "end_battery_pct": m.end_battery_pct})

    file_order = {e.event_id: i for i, e in enumerate(kit.replay)}

    def rank(e: ReplayEvent) -> tuple:
        if e.event_type == "MISSION_STARTED":
            step = 0
        elif e.event_type == "MISSION_COMPLETED":
            step = 10_000
        else:
            a = kit.action.get(str(e.details.get("action_id", "")))
            step = (a.sequence * 2 + (1 if e.event_type in ACTION_END else 0)) if a else 5_000
        m = kit.mission.get(e.mission_id)
        return (e.timestamp, m.start_time if m else e.timestamp, e.mission_id, step, file_order.get(e.event_id, 1_000_000))

    return sorted([*kit.replay, *rebuilt], key=rank)


def events_for(kit: Kit, e: ReplayEvent, seq: int, failed_so_far: int = 0, anomalies_so_far: int = 0) -> list[SpotEvent]:
    """The bus events one stream event turns into. Pure: the same inputs give the same events (ids aside)."""
    base = {"seq": seq, "data_ts": e.timestamp.isoformat(), "source": e.source}
    d = e.details
    if e.event_type == "MISSION_STARTED":
        return [MissionStarted(**base, mission_id=e.mission_id, scenario=d.get("scenario", ""),
                               configuration_version=d.get("configuration_version", ""), battery_pct=d.get("battery_pct"))]
    if e.event_type == "MISSION_COMPLETED":
        status = str(d.get("status", ""))
        out: list[SpotEvent] = [MissionCompleted(**base, mission_id=e.mission_id, status=status, end_battery_pct=d.get("end_battery_pct"),
                                                 failed_actions=failed_so_far, anomalies=anomalies_so_far)]
        if status == "COMPLETED" and failed_so_far:
            out.append(AlertRaised(**base, alert_id=f"{e.mission_id}-hidden", level="warn", mission_id=e.mission_id,
                                   message=f"{e.mission_id} ended COMPLETED with {failed_so_far} failed action(s) inside."))
        elif status != "COMPLETED":
            out.append(AlertRaised(**base, alert_id=f"{e.mission_id}-status", level="critical", mission_id=e.mission_id,
                                   message=f"{e.mission_id} ended {status or 'without a status'} ({failed_so_far} failed action(s))."))
        return out
    action_id = str(d.get("action_id", ""))
    a = kit.action.get(action_id)
    where = {"mission_id": e.mission_id, "action_id": action_id, "location_id": d.get("location_id") or (a.location_id if a else ""),
             "asset_id": d.get("asset_id") or (a.asset_id if a else ""), "action_type": d.get("action_type") or (a.action_type if a else "")}
    if e.event_type == "ACTION_STARTED":
        return [ActionStarted(**base, **where, battery_pct=d.get("battery_pct"), sequence=a.sequence if a else None)]
    if e.event_type not in ACTION_END:
        return []
    readings = kit.readings_of.get(action_id, [])
    reading = ({"reading_id": readings[0].reading_id, "measurement_type": readings[0].measurement_type, "value": readings[0].value,
                "unit": readings[0].unit, "quality": readings[0].quality} if readings else None)
    failed = e.event_type == "ACTION_FAILED"
    out = []
    if failed:
        failure = str(d.get("failure_type") or "")
        out.append(ActionFailed(**base, **where, failure_type=failure, reading=reading))
        out.append(AlertRaised(**base, alert_id=f"{action_id}-failed", level="warn", **{k: where[k] for k in ("mission_id", "action_id", "asset_id", "location_id")},
                               message=f"{action_id}: {_words(where['action_type'])} of {where['asset_id']} at {where['location_id']} failed "
                                       f"({_words(failure) or 'no failure type'})."))
    else:
        out.append(ActionCompleted(**base, **where, reading=reading))
    for x in kit.anomalies_of.get(action_id, []):
        out.append(AnomalyRaised(**base, mission_id=e.mission_id, action_id=action_id, anomaly_id=x.anomaly_id, asset_id=x.asset_id,
                                 location_id=x.location_id, anomaly_type=x.anomaly_type, severity=x.severity, confirmed=not failed))
        crit = kit.asset[x.asset_id].criticality if x.asset_id in kit.asset else "?"
        out.append(AlertRaised(**base, alert_id=x.anomaly_id, level=ALERT_LEVEL.get(x.severity, "info"), mission_id=e.mission_id,
                               action_id=action_id, anomaly_id=x.anomaly_id, asset_id=x.asset_id, location_id=x.location_id,
                               message=f"{x.anomaly_id}: {x.severity} {_words(x.anomaly_type)} on {x.asset_id} ({crit} asset) at {x.location_id}"
                                       + (", from a successful inspection." if not failed else ", raised by a failed action: unverified.")))
    return out


class SpotLive:
    """The picture so far. Events in, state out; nothing here publishes, reads a clock or touches a file."""

    def __init__(self, kit: Kit | None, stream: list[ReplayEvent] | None = None) -> None:
        self.kit = kit
        self.stream = stream if stream is not None else (build_stream(kit) if kit else [])
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.data_ts: str | None = None
        self.mission: dict | None = None
        self.robot: dict = {"location_id": None, "trail": []}
        self.counts = {"missions_started": 0, "missions_completed": 0, "missions_failed": 0, "missions_hiding_failures": 0,
                       "actions_ok": 0, "actions_failed": 0, "anomalies": 0, "anomalies_serious": 0}
        self.by_location: dict[str, dict] = {}
        self.by_asset: dict[str, dict] = {}
        self.alerts: list[dict] = []
        self.ended: list[dict] = []  # missions that have ended, oldest first

    # --- the fold ---

    def on_event(self, e) -> None:
        kind = e.type
        if kind == "spot.replay.seeked":
            self.rebuild(int(getattr(e, "index", 0)))
            return
        if not kind.startswith("spot.") or kind == "spot.work_order.created":
            return
        seq = int(getattr(e, "seq", -1))
        if seq >= 0:
            self.index = max(self.index, seq + 1)
        self.data_ts = getattr(e, "data_ts", None) or self.data_ts
        if kind == "spot.mission.started":
            self.counts["missions_started"] += 1
            self.mission = {"mission_id": e.mission_id, "status": "RUNNING", "scenario": getattr(e, "scenario", ""),
                            "configuration_version": getattr(e, "configuration_version", ""), "started": self.data_ts, "ended": None,
                            "battery_pct": getattr(e, "battery_pct", None), "actions": [], "failed": 0, "anomalies": 0,
                            "source": getattr(e, "source", "replay")}
            self.robot = {"location_id": self.robot.get("location_id"), "trail": []}
        elif kind == "spot.action.started":
            m = self._mission(e.mission_id)
            m["actions"].append({"action_id": e.action_id, "sequence": getattr(e, "sequence", None), "action_type": getattr(e, "action_type", ""),
                                 "location_id": getattr(e, "location_id", ""), "asset_id": getattr(e, "asset_id", ""), "status": "RUNNING",
                                 "failure_type": "", "battery_pct": getattr(e, "battery_pct", None), "ts": self.data_ts, "reading": None,
                                 "anomalies": []})
            m["battery_pct"] = getattr(e, "battery_pct", None) if getattr(e, "battery_pct", None) is not None else m["battery_pct"]
            loc = getattr(e, "location_id", "")
            if loc:
                trail = self.robot["trail"]
                if not trail or trail[-1] != loc:
                    trail.append(loc)
                self.robot = {"location_id": loc, "trail": trail[-TRAIL:]}
        elif kind in ("spot.action.completed", "spot.action.failed"):
            failed = kind == "spot.action.failed"
            act = self._action(e.mission_id, e.action_id, e)
            act["status"] = "FAILED" if failed else "SUCCESS"
            act["failure_type"] = getattr(e, "failure_type", "") if failed else ""
            act["reading"] = getattr(e, "reading", None)
            self.counts["actions_failed" if failed else "actions_ok"] += 1
            self._mission(e.mission_id)["failed"] += int(failed)
            loc = self.by_location.setdefault(act["location_id"], {"ok": 0, "failed": 0})
            loc["failed" if failed else "ok"] += 1
            asset = self.by_asset.setdefault(act["asset_id"], {"ok": 0, "failed": 0, "anomalies": 0, "worst": None})
            asset["failed" if failed else "ok"] += 1
        elif kind == "spot.anomaly.raised":
            self.counts["anomalies"] += 1
            self.counts["anomalies_serious"] += int(e.severity in ("HIGH", "CRITICAL"))
            self._mission(e.mission_id)["anomalies"] += 1
            self._action(e.mission_id, e.action_id, e)["anomalies"].append(
                {"anomaly_id": e.anomaly_id, "anomaly_type": getattr(e, "anomaly_type", ""), "severity": e.severity,
                 "confirmed": bool(getattr(e, "confirmed", False))})
            asset = self.by_asset.setdefault(getattr(e, "asset_id", ""), {"ok": 0, "failed": 0, "anomalies": 0, "worst": None})
            asset["anomalies"] += 1
            order = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            if asset["worst"] is None or order.index(e.severity) > order.index(asset["worst"]):
                asset["worst"] = e.severity
        elif kind == "spot.mission.completed":
            m = self._mission(e.mission_id)
            m["status"], m["ended"] = e.status, self.data_ts
            m["battery_pct"] = getattr(e, "end_battery_pct", None) if getattr(e, "end_battery_pct", None) is not None else m["battery_pct"]
            self.counts["missions_completed" if e.status == "COMPLETED" else "missions_failed"] += 1
            self.counts["missions_hiding_failures"] += int(e.status == "COMPLETED" and m["failed"] > 0)
            self.ended.append({"mission_id": m["mission_id"], "status": m["status"], "failed": m["failed"], "anomalies": m["anomalies"]})
        elif kind == "spot.alert.raised":
            self.alerts.append({"alert_id": e.alert_id, "level": e.level, "message": e.message, "data_ts": self.data_ts, "seq": seq,
                                "mission_id": getattr(e, "mission_id", ""), "action_id": getattr(e, "action_id", None),
                                "anomaly_id": getattr(e, "anomaly_id", None), "asset_id": getattr(e, "asset_id", None),
                                "location_id": getattr(e, "location_id", None)})
            del self.alerts[:-MAX_ALERTS]

    def _mission(self, mission_id: str) -> dict:
        if self.mission is None or self.mission["mission_id"] != mission_id:  # an event for a mission we never saw start
            self.mission = {"mission_id": mission_id, "status": "RUNNING", "scenario": "", "configuration_version": "", "started": self.data_ts,
                            "ended": None, "battery_pct": None, "actions": [], "failed": 0, "anomalies": 0, "source": "replay"}
        return self.mission

    def _action(self, mission_id: str, action_id: str, e) -> dict:
        m = self._mission(mission_id)
        for act in m["actions"]:
            if act["action_id"] == action_id:
                return act
        act = {"action_id": action_id, "sequence": None, "action_type": getattr(e, "action_type", ""), "location_id": getattr(e, "location_id", ""),
               "asset_id": getattr(e, "asset_id", ""), "status": "RUNNING", "failure_type": "", "battery_pct": None, "ts": self.data_ts,
               "reading": None, "anomalies": []}
        m["actions"].append(act)
        return act

    def rebuild(self, index: int) -> None:
        """The picture after the first `index` stream events, folded from the kit. No events are published."""
        self.reset()
        if self.kit is None:
            return
        index = max(0, min(index, len(self.stream)))
        for seq in range(index):
            for ev in self.expand(seq):
                self.on_event(ev)
        self.index = index

    def expand(self, seq: int) -> list[SpotEvent]:
        """The bus events for stream position `seq`, given the picture so far (the mission's running counts)."""
        e = self.stream[seq]
        m = self.mission if self.mission and self.mission["mission_id"] == e.mission_id else None
        return events_for(self.kit, e, seq, failed_so_far=m["failed"] if m else 0, anomalies_so_far=m["anomalies"] if m else 0)

    # --- what the screen reads ---

    def marks(self) -> list[dict]:
        """Where each mission sits in the stream, for the scrubber. From the kit, not from what has been played."""
        if self.kit is None:
            return []
        spans: dict[str, dict] = {}
        for i, e in enumerate(self.stream):
            s = spans.setdefault(e.mission_id, {"mission_id": e.mission_id, "start": i, "end": i, "failures": [], "source": e.source})
            s["end"] = i
            if e.event_type == "ACTION_FAILED":
                s["failures"].append(i)
            if e.source == "tables":
                s["source"] = "tables"
        out = []
        for mid, s in spans.items():
            m = self.kit.mission.get(mid)
            out.append({**s, "status": m.mission_status if m else "", "version": m.configuration_version if m else "",
                        "failed": len(self.kit.failed_in(mid)), "anomalies": len(self.kit.anomalies_in(mid)),
                        "started": m.start_time.isoformat() if m else None})
        return out

    def view(self) -> dict:
        rebuilt_from = next((i for i, e in enumerate(self.stream) if e.source == "tables"), None)
        loc = self.kit.location.get(self.robot["location_id"]) if self.kit and self.robot["location_id"] else None
        return {"index": self.index, "total": len(self.stream), "data_ts": self.data_ts, "mission": self.mission,
                "robot": {**self.robot, "floor": loc.floor if loc else None}, "counts": self.counts, "by_location": self.by_location,
                "by_asset": self.by_asset, "alerts": list(reversed(self.alerts)), "ended": self.ended[-30:],
                "rebuilt_from": rebuilt_from, "at_end": self.index >= len(self.stream)}


class Player:
    """Publishes the stream on the bus, one stream event at a time, at a chosen speed."""

    def __init__(self, kit: Kit | None, stream: list[ReplayEvent], bus, live: SpotLive, *, speed: float = DEFAULT_SPEED,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.kit, self.stream, self.bus, self.live = kit, stream, bus, live
        self.speed = speed
        self.sleep = sleep
        self.playing = False
        self._task: asyncio.Task | None = None

    def wait_before(self, index: int) -> float:
        """Wall seconds to wait before stream event `index` is shown."""
        if index <= 0 or index >= len(self.stream):
            return 0.0
        e, prev = self.stream[index], self.stream[index - 1]
        gap = max(0.0, (e.timestamp - prev.timestamp).total_seconds())
        wait = min(gap / self.speed, MAX_WAIT_S)
        if e.event_type in ACTION_END and self.kit is not None:
            a = self.kit.action.get(str(e.details.get("action_id", "")))
            wait += max(MIN_DWELL_S, min((a.duration_sec if a else 30) / self.speed, MAX_DWELL_S))
        return wait

    async def step(self) -> bool:
        """Show the next stream event. False when the stream is finished."""
        if self.kit is None or self.live.index >= len(self.stream):
            return False
        for ev in self.live.expand(self.live.index):
            await self.bus.publish(ev)
        return True

    async def _run(self) -> None:
        try:
            while self.playing and self.live.index < len(self.stream):
                await self.sleep(self.wait_before(self.live.index))
                if not self.playing:
                    break
                await self.step()
        finally:
            self.playing = False

    async def start(self) -> None:
        if self.kit is None or self.playing:
            return
        if self.live.index >= len(self.stream):  # at the end: play again from the top
            await self.seek(0)
        self.playing = True
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def pause(self) -> None:
        self.playing = False
        task, self._task = self._task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.playing = False

    async def seek(self, index: int) -> None:
        was_playing = self.playing
        await self.pause()
        index = max(0, min(int(index), len(self.stream)))
        await self.bus.publish(ReplaySeeked(index=index, data_ts=self.stream[index - 1].timestamp.isoformat() if index and self.stream else None))
        if was_playing and index < len(self.stream):
            await self.start()

    def index_of_mission(self, mission_id: str) -> int | None:
        return next((i for i, e in enumerate(self.stream) if e.mission_id == mission_id), None)

    def set_speed(self, speed: float) -> float:
        self.speed = max(1.0, min(float(speed), 36_000.0))
        return self.speed

    def view(self) -> dict:
        return {**self.live.view(), "playing": self.playing, "speed": self.speed, "speeds": list(SPEEDS), "marks": self.live.marks()}
