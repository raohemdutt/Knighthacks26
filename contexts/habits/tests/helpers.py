"""Scripted events for the Habits tests. Times are local (America/New_York), October 2026."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from contexts.habits.domain.config import HabitsConfig, load_config
from contexts.habits.domain.tracker import HabitTracker
from contracts.connectors import ItemSynced
from contracts.impact import ItemDisposed
from contracts.perception import ObservationRecorded, SceneDescribed, ZoneEvent
from contracts.types import ObjectHit
from contracts.vault import PolicyChanged

NY = ZoneInfo("America/New_York")


def at(day: int, clock: str = "00:00") -> datetime:
    h, m = clock.split(":")
    return datetime(2026, 10, day, int(h), int(m), tzinfo=NY)


def day(n: int) -> date:
    return date(2026, 10, n)


def tracker(cfg: HabitsConfig | None = None) -> HabitTracker:
    return HabitTracker(cfg or load_config())


def zone(ts, kind, zone_id="desk", track=1, dwell_s=None, label="person"):
    camera = "cam-desk" if zone_id == "desk" else "cam-door"
    return ZoneEvent(ts=ts, camera_id=camera, track_id=track, label=label, zone_id=zone_id, kind=kind, dwell_s=dwell_s,
                     frame_ref=f"frames:{camera}:{int(ts.timestamp() * 1000)}")


def sit(d: int, start: str, end: str, track=1) -> list[ZoneEvent]:
    """A person at the desk from `start` to `end`."""
    return [zone(at(d, start), "enter", track=track), zone(at(d, end), "exit", track=track)]


def scene(ts, caption="a tidy desk", objects=(), zone_id=None, camera="cam-desk"):
    return SceneDescribed(ts=ts, camera_id=camera, caption=caption, zone_id=zone_id, model_version="atlas-test",
                          objects=[ObjectHit(label=o, position="on the desk") for o in objects],
                          frame_ref=f"frames:{camera}:{int(ts.timestamp() * 1000)}")


def watch(start: datetime, end: datetime, camera="cam-desk", every_s=60) -> list[ObservationRecorded]:
    """The camera reporting an empty room from `start` to `end` (a real one reports every second)."""
    n = int((end - start).total_seconds() // every_s)
    return [ObservationRecorded(event_id=f"obs-{camera}-{int((start + timedelta(seconds=i * every_s)).timestamp())}",
                                ts=start + timedelta(seconds=i * every_s), camera_id=camera, tracks=[]) for i in range(n + 1)]


def paused(ts, *cameras) -> PolicyChanged:
    """The Vault policy as of `ts`: these cameras paused, every other one on."""
    return PolicyChanged(ts=ts, blur_faces=True, paused_cameras=list(cameras))


def in_order(*events) -> list:
    """Several scripts as one log: by time, and at the same instant the camera's report before what was made of it."""
    flat = [e for part in events for e in (part if isinstance(part, list) else [part])]
    return sorted(flat, key=lambda e: (e.ts, e.type != "perception.observation.recorded"))


def assignment(ts, title, due, external_id="a1"):
    return ItemSynced(ts=ts, connector="canvas", kind="assignment", external_id=external_id, title=title, when=due,
                      url=f"https://canvas.test/{external_id}")


def calendar(ts, title, start, external_id="c1"):
    return ItemSynced(ts=ts, connector="calendar", kind="calendar_event", external_id=external_id, title=title, when=start)


def disposed(ts, correct, item_id="cam-door:1"):
    return ItemDisposed(ts=ts, item_id=item_id, label="can", zone="recycling" if correct else "trash",
                        bin="recycling" if correct else "trash", correct=correct)


def feed(t: HabitTracker, events) -> list:
    """Route events the way the bus does; returns everything the tracker published."""
    handlers = {"perception.zone.event": t.on_zone_event, "perception.scene.described": t.on_scene,
                "connector.item.synced": t.on_item, "impact.item.disposed": t.on_disposed,
                "perception.observation.recorded": t.on_observation, "vault.policy.changed": t.on_policy}
    out = []
    for e in events:
        out += handlers[e.type](e)
    return out


def signals(out, kind) -> list[float]:
    return [e.value for e in out if e.type == "habits.signal.recorded" and e.kind == kind]


def closed(out, habit_id) -> list[tuple]:
    return [(e.day.day, e.value, e.met, e.streak) for e in out if e.type == "habits.day.closed" and e.habit_id == habit_id]


def one_day(n: int, meals: int = 2) -> list:
    """A day with `meals` meals at the desk and nothing else."""
    return [scene(at(n, f"{8 + 5 * i}:00"), caption="a plate of pasta next to the laptop") for i in range(meals)]
