"""P9 end to end without a camera: synthetic frames -> recognition -> tagged tracks -> zone events -> habits
for the enrolled person only, the same after a replay of the log, and nothing about a face in that log."""

import asyncio
from datetime import timedelta

from bus import EventBus
from contexts.habits.domain.config import load_config
from contexts.habits.domain.members import MemberHabits
from contexts.habits.domain.tracker import HabitTracker
from contexts.perception.domain.zones import ZoneEngine, load_zones
from contexts.perception.tests.identity_helpers import T0, Frame, Rig, face_in, person, row
from contracts.perception import ObservationRecorded, SceneDescribed

NAME = "Zebulon Quartermaine"


class Box:
    """Zones, habits and (live only) recognition on one bus, subscribed the way the gateway does it."""

    def __init__(self, data_dir, live: bool):
        self.bus = EventBus(data_dir / "events")
        self.engine = ZoneEngine(load_zones("config/zones.yaml"))
        self.habits = MemberHabits(HabitTracker(load_config()))
        self.rig = Rig(data_dir) if live else None
        self.bus.subscribe("perception.observation", self.engine.on_observation, replay=True)
        for prefix, handler in (("perception.zone", self.habits.on_zone_event), ("perception.scene", self.habits.on_scene),
                                ("habits.day.closed", self.habits.on_day_closed),
                                ("perception.identity", self.habits.on_identity), ("perception.person", self.habits.on_person)):
            self.bus.subscribe(prefix, handler, replay=True)
        if live:
            self.bus.subscribe("perception.identity", self.rig.service.on_changed, replay=True)

    async def second(self, frame: Frame, tracks: list) -> None:
        """One second of camera: the recognition hook sees the frame, then the observation is published."""
        hook_said = self.rig.look(frame, tracks)
        await self.bus.publish_all(hook_said)
        await self.bus.publish(ObservationRecorded(ts=self.now, camera_id="cam-desk", tracks=tracks))

    @property
    def now(self):
        return T0 + timedelta(seconds=self.rig.clock.t)

    def scene(self, caption: str) -> SceneDescribed:
        return SceneDescribed(ts=self.now, camera_id="cam-desk", caption=caption, model_version="atlas-test",
                              frame_ref=f"frames:cam-desk:{int(self.now.timestamp() * 1000)}")


def picture(habits: MemberHabits) -> dict:
    day = habits.sched.local_day(T0)
    return {m: (t.digest(day, T0 + timedelta(hours=1)), t.days.get(day)) for m, t in {"": habits.base, **habits.members}.items()}


def test_only_the_enrolled_person_gets_habits_and_a_replay_gives_the_same(tmp_path):
    owner, guest = person(1, 0.15, 0.2, 0.45, 0.9), person(2, 0.55, 0.2, 0.85, 0.9)  # both sitting in the desk zone

    async def live() -> Box:
        box = Box(tmp_path, live=True)
        await box.bus.publish_all(box.rig.service.set_enabled(True))
        member = box.rig.service.start_enrollment(None, NAME)["member_id"]
        for i in range(5):  # the owner enrols alone at the desk
            await box.second(Frame((b"zeb#enrol%d" % i, row(300, 100))), [])
        assert box.rig.service.enrollment_status()["state"] == "done"
        for i in range(60):  # then works for a minute next to a guest
            await box.second(Frame(face_in(owner, b"zeb#%d" % i), face_in(guest, b"guest#%d" % i)), [owner, guest])
        await box.bus.publish(box.scene("a plate of pasta between two laptops"))
        for i in range(10):  # the owner gets up; the guest stays and eats
            await box.second(Frame(face_in(guest, b"guest#x%d" % i)), [guest])
        await box.bus.publish(box.scene("a burger and an energy drink"))
        assert box.rig.service.tagged() == {"cam-desk": {}} and box.habits.tags == {}  # the owner's tag ended with the track
        box.member = member
        return box

    async def rebuilt() -> Box:
        box = Box(tmp_path, live=False)
        await box.bus.replay_into()
        return box

    first, again = asyncio.run(live()), asyncio.run(rebuilt())
    day = first.habits.sched.local_day(T0)
    mine = first.habits.members[first.member].days[day]
    assert (mine.meals, mine.drinks) == (1, 0) and 0.9 <= mine.desk_min <= 1.1  # the minute at the desk, the pasta
    assert first.habits.base.days.get(day) is None  # nothing about the guest, and nothing left on "this desk"
    assert list(first.habits.members) == [first.member] and first.habits.view() is first.habits.members[first.member]
    assert picture(again.habits) == picture(first.habits) and list(again.habits.members) == [first.member]

    log = "".join(p.read_text() for p in (tmp_path / "events").glob("*.jsonl"))
    assert log.count("perception.person.identified") == 1 and log.count("perception.person.lost") == 2  # reset, then the track
    assert first.member in log and NAME not in log and "zeb" not in log and "Quartermaine" not in log
    assert "habits.signal.recorded" not in log  # no signal about anyone is published while people are told apart
