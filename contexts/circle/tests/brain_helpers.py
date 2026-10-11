"""The world the group brain tests run in: the teacher's pod C behind the fake hub proxy (lab/teacher_fake.py), a box's
event log in a temporary folder, and the real GroupBrain and TeacherLink wired to both."""

import asyncio

from bus import EventBus
from contexts.circle.application.group_brain import GroupBrain
from contexts.circle.infrastructure.teacher import TeacherLink
from lab.hub_fake import FakeClock
from lab.teacher_fake import FakeTeacher, Turn

TOKEN = "0123456789abcdef0123456789abcdef"
USER = "roscon01"
ENV = {"AUP_HUB_USER_C": USER, "AUP_HUB_TOKEN_C": TOKEN, "AUP_HUB_URL_C": "https://teacher.test",
       "AUP_HUB_USER_D": "roscon02", "AUP_HUB_TOKEN_D": "fedcba9876543210fedcba9876543210",
       "LAB_POD_ROLES": "C:pod-a,D:pod-b"}
QUESTION = "Where should we meet tonight?"
FACTS = ["Maya is at the Library until 9:00 PM", "Dominick is at the Student Union until 8:00 PM",
         "Knight Hacks midnight snack at 11:00 PM in the Student Union"]
PLAN = "Meet at the Student Union at 7:30 PM; Maya can join from the Library, and the snack is at 11."
USER_TEXT = ("Facts:\n- Maya is at the Library until 9:00 PM\n- Dominick is at the Student Union until 8:00 PM\n"
             "- Knight Hacks midnight snack at 11:00 PM in the Student Union\n\nQuestion: Where should we meet tonight?")


def run(coro):
    return asyncio.run(coro)


class World:
    """The teacher's pod C behind the fake hub proxy, a box's log in tmp_path, and the group brain wired to both."""

    def __init__(self, tmp_path, *, aliases=("group-brain",), answer=PLAN, policy=None):
        self.log_dir = tmp_path / "events"
        self.clock = FakeClock()
        self.answer = answer
        self.asked_on_disk = []      # for each chat request: the circle.brain.asked events in the log when it arrived
        self.fake = FakeTeacher(policy or self.policy, user=USER, token=TOKEN, aliases=aliases, clock=self.clock)
        self.link = TeacherLink.from_env(environ=ENV, env_files=[], transport=self.fake.transport)
        self.bus = EventBus(self.log_dir)
        self.brain = GroupBrain("team", "ilan", self.link, self.bus.publish, timer=self.clock.time)
        self.bus.subscribe("circle.brain.consent", self.brain.on_consent, replay=True)

    def policy(self, body):
        self.asked_on_disk.append([e for e in self.logged() if e.type == "circle.brain.asked"])
        return Turn(content=self.answer)

    def logged(self):
        return list(EventBus(self.log_dir).read_log())

    def chats(self):
        return [seen for seen in self.fake.requests if seen.method == "POST"]

    def model_lists(self):
        return [seen for seen in self.fake.requests if seen.method == "GET"]


def opted_in(tmp_path, **options):
    world = World(tmp_path, **options)
    consent = run(world.brain.set_consent(True, "phone"))
    return world, consent
