"""Scripted events for the Habits tests. Times are local (America/New_York), October 2026."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from contexts.habits.domain.config import HabitsConfig, load_config
from contexts.habits.domain.tracker import HabitTracker
from contracts.connectors import ItemSynced
from contracts.impact import ItemDisposed
from contracts.perception import SceneDescribed, ZoneEvent
from contracts.types import ObjectHit

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
                "connector.item.synced": t.on_item, "impact.item.disposed": t.on_disposed}
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
