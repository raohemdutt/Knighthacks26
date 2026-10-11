"""Ask Atlas's router (ask_router.py): the model decides, the network brain answers when the box's model chose
nothing for a question only a tool can answer, then the box's rules. A fake model (a scripted tool choice), a fake
network brain (an httpx mock transport) and a fake bus (a list)."""

import asyncio
import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from contexts.agent.application.ask_router import BOX, NETWORK, RULES, AskRouter, NetworkBrain, calls_in_text, rules
from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.product_tools import Ports
from contexts.agent.application.qa import QA
from contracts.agent import ActionProposed
from contracts.experience import QuestionAsked
from contracts.types import ChatResult, ToolCall
from runtime import FakeRuntime

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 11, 12, 15, tzinfo=UTC)

NINE = [
    "What is on my calendar today?",
    "Add a meeting with Sam at 3 pm tomorrow called demo prep",
    "What did I do this morning?",
    "Where is my phone?",
    "Set a reminder to call mom at 6",
    "Turn off the camera",
    "Show me the room map",
    "What is the box's model right now?",
    "How much energy did my last answer cost?",
]


def picks(name: str, **args) -> ChatResult:
    return ChatResult(content="", tool_calls=[ToolCall(id="c1", name=name, arguments=args)], model_version="box-4b")


def says(text: str) -> ChatResult:
    return ChatResult(content=text, model_version="box-4b")


def make(replies, network=None, **kw) -> tuple[AskRouter, list, FakeRuntime]:
    bus: list = []

    async def publish(e) -> None:
        bus.append(e)

    rt = FakeRuntime(chat_replies=list(replies))
    p = Ports(tz=TZ, items=lambda s, e: [], can_pause=lambda: True,
              box_model=lambda: {"version": "box-4b"}, room=lambda: {"title": "Dorm room", "cameras": ["a"], "zones": []})
    return AskRouter(rt, p, lambda: NOW, publish=publish, network=network, **kw), bus, rt


def route(router: AskRouter, q: str):
    return asyncio.run(router.route(q))


def teacher(reply: dict | Exception, seen: list | None = None) -> NetworkBrain:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(200, json={"choices": [{"message": reply}]})
    return NetworkBrain("https://hub.example/user/c/proxy/8000/v1", {"Authorization": "token t"},
                        transport=httpx.MockTransport(handle))


# --- the box's model decides ---

def test_the_box_picks_calendar_add_and_the_router_publishes_the_approval_card():
    router, bus, rt = make([picks("calendar_add", title="demo prep", start="3 pm tomorrow", attendees=["Sam"])])
    r = route(router, NINE[1])
    assert (r.answered_by, r.tool) == (BOX, "calendar.add")
    assert "Approve it on your phone" in r.text
    assert len(bus) == 1 and isinstance(bus[0], ActionProposed) and bus[0].target == "calendar.create_event"
    assert r.model_version == "fake-0"


def test_the_box_gets_every_tool_with_the_choice_left_to_it():
    router, _, rt = make([picks("room_map")])
    route(router, "Show me the room map")
    msgs = rt.calls[0][1]
    assert msgs[0].role == "system" and "NONE" in msgs[0].content and "2026-10-11" in msgs[0].content
    assert msgs[-1].content == "Show me the room map"


def test_a_question_for_no_tool_goes_back_to_qa():
    router, bus, _ = make([says("NONE")])
    assert route(router, "How are you doing today?") is None and bus == []


def test_a_dry_run_publishes_nothing():
    router, bus, _ = make([picks("calendar_add", title="x", start="3 pm")])
    r = asyncio.run(router.route("add x at 3 pm", dry_run=True))
    assert r.result.proposal is None and bus == []


def test_a_tool_call_written_as_text_is_still_a_tool_call():
    router, _, _ = make([says('<tool_call>{"name": "box_model", "arguments": {}}</tool_call>')])
    r = route(router, "What is the box's model right now?")
    assert (r.tool, r.answered_by) == ("box.model", BOX)
    assert calls_in_text('{"name": "room_map", "arguments": "{}"}')[0].name == "room_map"
    assert calls_in_text("NONE") == []


def test_a_missing_argument_from_the_box_is_filled_from_the_words():
    router, bus, _ = make([picks("calendar_add", start="3 pm tomorrow")])  # the 4B left the title out
    r = route(router, "Add a meeting with Sam at 3 pm tomorrow called demo prep")
    assert r.result.ok and bus[0].params["events"][0]["title"] == "demo prep with Sam"


# --- the network brain, then the rules ---

def test_no_tool_from_the_box_for_a_tool_question_goes_to_the_network_brain():
    seen: list = []
    net = teacher({"content": None, "tool_calls": [{"id": "n1", "function": {"name": "camera_pause", "arguments": '{"action": "pause"}'}}]}, seen)
    router, bus, _ = make([says("NONE")], network=net)
    r = route(router, "Turn off the camera")
    assert (r.answered_by, r.tool) == (NETWORK, "camera.pause") and r.model_version == "teacher"
    body = json.loads(seen[0].content)
    assert body["model"] == "teacher" and body["tool_choice"] == "auto" and len(body["tools"]) == 17
    assert seen[0].headers["Authorization"] == "token t"
    assert bus[0].kind == "device_control"


def test_the_network_brain_writing_its_call_as_text_is_understood():
    net = teacher({"content": '<tool_call>{"name": "room_map", "arguments": {}}</tool_call>'})
    router, _, _ = make([says("I can't show maps.")], network=net)
    assert route(router, "Show me the room map").answered_by == NETWORK


def test_an_unreachable_network_brain_falls_back_to_the_rules():
    router, _, _ = make([says("NONE")], network=teacher(httpx.ConnectError("no route")))
    r = route(router, "Set a reminder to call mom at 6")
    assert (r.answered_by, r.tool) == (RULES, "reminders.add")
    assert r.result.proposal.params["events"][0]["title"] == "Reminder: call mom"


def test_rules_off_and_no_network_leave_it_to_qa():
    router, _, _ = make([says("NONE")], use_rules=False)
    assert route(router, "Turn off the camera") is None


def test_a_failing_model_does_not_fail_the_answer():
    class Broken(FakeRuntime):
        def chat(self, *a, **k):
            raise RuntimeError("llama-server down")
    router, _, _ = make([])
    router.runtime = Broken()
    assert route(router, "Show me the room map").answered_by == RULES
    assert route(router, "How are you?") is None


@pytest.mark.parametrize("question,tool", [
    (NINE[0], "calendar.list"), (NINE[1], "calendar.add"), (NINE[2], "memory.recap"), (NINE[4], "reminders.add"),
    (NINE[5], "camera.pause"), (NINE[6], "room.map"), (NINE[7], "box.model"), (NINE[8], "box.energy_last_answer"),
    ("Turn the cameras back on", "camera.pause"), ("Who is here?", "people.who_is_here"),
    ("Open the approvals screen", "navigate.screen"), ("How is the learning lab doing?", "learning.status"),
    ("What needs me on the plant floor?", "inspection.status"), ("How is SPOT doing?", "inspection.status"),
    ("Take me to inspection", "navigate.screen"),
    ("Open the plant floor", "navigate.screen"),
])
def test_the_rules_name_the_tool_for_each_question(question, tool):
    assert rules(question).name == tool


def test_the_rules_leave_a_question_about_an_object_to_qa():
    assert rules("Where is my phone?") is None
    assert rules("How are you?") is None


def test_rules_arguments_for_the_demo_questions():
    add = rules(NINE[1]).arguments
    assert add["title"] == "demo prep with Sam" and add["attendees"] == ["Sam"]
    assert rules(NINE[0]).arguments == {"day": "today"}
    assert rules("Turn the cameras back on").arguments == {"action": "resume"}


# --- the network brain's address ---

def test_network_brain_from_env():
    assert NetworkBrain.from_env({}) is None
    c = {"AUP_HUB_URL_C": "https://sjc.example/", "AUP_HUB_USER_C": "acct", "AUP_HUB_TOKEN_C": "x"}
    nb = NetworkBrain.from_env(c)
    assert nb.base_url == "https://sjc.example/user/acct/proxy/8000/v1" and nb.headers["Authorization"] == "token x"
    assert nb.model == "teacher"
    assert NetworkBrain.from_env({**c, "ATLAS_NETWORK_BRAIN": "off"}) is None
    direct = NetworkBrain.from_env({"ATLAS_NETWORK_BRAIN_URL": "http://pod:8000/v1/", "ATLAS_NETWORK_BRAIN_TOKEN": "y"})
    assert direct.base_url == "http://pod:8000/v1" and direct.headers == {"Authorization": "Bearer y"}


# --- QA: the router first, then everything as before ---

def qa_with(router: AskRouter | None, replies=()) -> QA:
    qa = QA(FakeRuntime(chat_replies=list(replies)), FakeMemory(), clock=lambda: NOW)
    qa.router = router
    return qa


def test_qa_answers_from_the_tool_and_says_who_answered():
    router, bus, _ = make([picks("calendar_add", title="demo prep", start="3 pm tomorrow")])
    qa = qa_with(router)
    streamed: list[str] = []
    qa.on_delta = lambda qid, text: streamed.append(text)
    e = asyncio.run(qa.on_question(QuestionAsked(question_id="q1", text=NINE[1])))
    assert e.answered_by == BOX and e.tool == "calendar.add" and e.action_id == bus[0].action_id
    assert e.tool_result["ok"] and e.tool_result["event"]["title"] == "demo prep"
    assert e.link.startswith("/approvals/") and streamed == [e.answer]
    assert qa.last_window is not None


def test_qa_without_a_tool_answers_as_before_and_marks_the_box():
    router, _, _ = make([says("NONE")])
    qa = qa_with(router, [says("Hi there.")])
    e = asyncio.run(qa.on_question(QuestionAsked(question_id="q2", text="Where did I leave my keys?")))
    assert e.tool is None and e.answered_by == "the box" and e.answer == "I didn't see that."  # the pre-fetch, as before


def test_qa_with_no_router_leaves_the_new_fields_empty():
    e = asyncio.run(qa_with(None).on_question(QuestionAsked(question_id="q3", text="Where did I leave my keys?")))
    assert e.answered_by is None and e.tool is None and e.answer == "I didn't see that."
