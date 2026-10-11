"""The Ask panel's comparison: the box's answer and the 122B pair's, side by side, both from the box's memory.

Every model server, the pair and the sampler are fakes; nothing reaches a pod or the box."""

import asyncio
import json
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning import lab_deploy
from contexts.learning.lab_deploy import DeployRunner, lab_deploy_router

PICK = "<tool_call>\n<function=last_seen>\n<parameter=label>\nwater bottle\n</parameter>\n</function>\n</tool_call>"


def tool_call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def said(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class FakeLlama:
    """llama-server on the box: the eval's completion path and the chat path, with scripted chat turns."""

    def __init__(self, chat_turns: list[dict]) -> None:
        self.turns = list(chat_turns)
        self.chats: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/props":
            return httpx.Response(200, json={"model_path": "/data/models/atlas-base/model.gguf"})
        if path == "/completion":
            return httpx.Response(200, json={"tokens": [1], "content": "x"})
        if path == "/detokenize":
            return httpx.Response(200, json={"content": PICK + "<|im_end|>"})
        if path == "/v1/chat/completions":
            self.chats.append(json.loads(request.content))
            return httpx.Response(200, json=self.turns.pop(0) if self.turns else said("done"))
        return httpx.Response(404)


class FakeQA:
    """The agent's tool executor against a memory that saw the bottle on the desk at 2:05 PM."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict]] = []

    async def _run_tool(self, call, ev) -> None:
        self.ran.append((call.name, call.arguments))
        if call.name == "last_seen":
            ev.lines.append(f"Most recent: {call.arguments.get('label')} on the desk, Sun 2:05 PM (cam-desk)")


class FakePair:
    configured = True

    def __init__(self, turns: list[dict], delay: float = 0.0, fail: Exception | None = None) -> None:
        self.turns = list(turns)
        self.bodies: list[dict] = []
        self.delay = delay
        self.fail = fail

    async def chat(self, body, timeout_s=60.0):
        self.bodies.append(dict(body))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        return self.turns.pop(0) if self.turns else said("done")

    async def window_energy(self, start, end):
        return {"joules": 1234.5, "watts_avg": 130.2, "seconds": round(end - start, 2),
                "pods": {"C": {"joules": 700.0, "watts_avg": 74.0, "idle_w": 10.27, "samples": 9},
                         "D": {"joules": 534.5, "watts_avg": 56.2, "idle_w": 10.27, "samples": 9}}}


class FakeMeter:
    async def measure(self, start, end):
        return {"joules": 41.3, "idle_w": 15.76, "seconds": 2.4, "sampler": "tegrastats", "samples": 7,
                "above_idle_j": 3.5, "low_confidence": True}


def client(tmp_path, llama: FakeLlama, **kwargs) -> TestClient:
    app = FastAPI()
    runner = DeployRunner(tmp_path / "models", tmp_path / "state", llm_url="http://llm:8080")
    app.include_router(lab_deploy_router(runner, ask_transport=httpx.MockTransport(llama), **kwargs))
    return TestClient(app)


def ask(c: TestClient, question: str) -> dict:
    r = c.post("/api/learning/lab/deploy/ask", json={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_held_out_question_runs_the_picked_tool_on_memory_and_both_answer_from_it(tmp_path):
    llama = FakeLlama([said("Your water bottle was on the desk at 2:05 PM.")])
    qa = FakeQA()
    pair = FakePair([tool_call("last_seen", {"label": "water bottle"}), said("On the desk, at 2:05 PM.")])
    body = ask(client(tmp_path, llama, qa=qa, pair=pair, meter=FakeMeter()), "Have you seen my water bottle?")
    box, brain = body["box"], body["brain"]
    assert box["answer"] == "Your water bottle was on the desk at 2:05 PM."
    assert box["calls"] == [{"name": "last_seen", "arguments": {"label": "water bottle"}}]
    assert box["prompt"] == "eval" and box["serving"] == "atlas-base" and box["energy"]["joules"] == 41.3
    assert brain["answer"] == "On the desk, at 2:05 PM." and brain["serving"] == "Qwen3.5-122B across pods C and D"
    assert brain["calls"] == [{"name": "last_seen", "arguments": {"label": "water bottle"}}]
    assert brain["energy"]["pods"]["C"]["joules"] == 700.0 and brain["energy"]["joules"] == 1234.5
    assert qa.ran == [("last_seen", {"label": "water bottle"})] * 2  # the same memory, the same executor, both sides
    # the box's continuation saw the tool's result before it answered
    tool_msgs = [m for m in llama.chats[0]["messages"] if m["role"] == "tool"]
    assert tool_msgs and "on the desk, Sun 2:05 PM" in tool_msgs[0]["content"]
    # the brain was asked the 122B way: served name "teacher", thinking off, tools offered not forced
    assert pair.bodies[0]["model"] == "teacher" and pair.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert pair.bodies[0]["tool_choice"] == "auto" and pair.bodies[0]["tools"]
    # the card from before still reads its fields
    assert body["calls"] == box["calls"] and body["serving"] == "atlas-base" and body["energy"]["joules"] == 41.3


def test_a_free_question_gets_a_plain_answer_with_no_tool(tmp_path):
    llama = FakeLlama([said("<think>\n\n</think>\nI'm doing well, watching the room. How can I help?")])
    qa = FakeQA()
    pair = FakePair([said("Doing well, thanks for asking!")])
    body = ask(client(tmp_path, llama, qa=qa, pair=pair), "how are you")
    assert body["box"]["answer"] == "I'm doing well, watching the room. How can I help?"
    assert body["box"]["calls"] == [] and body["box"]["prompt"] == "chat"
    assert body["brain"]["answer"] == "Doing well, thanks for asking!" and body["brain"]["calls"] == []
    assert qa.ran == []
    asked = llama.chats[0]
    assert asked["tool_choice"] == "auto" and {t["function"]["name"] for t in asked["tools"]} >= {"last_seen", "search_memory"}
    assert "how are you" == asked["messages"][-1]["content"]


def test_tool_rounds_are_capped_at_three_then_it_must_answer(tmp_path):
    loop = [tool_call("search_memory", {"q": "keys"}, f"c{i}") for i in range(5)]
    llama = FakeLlama(loop + [said("I looked three times and found nothing.")])
    body = ask(client(tmp_path, llama, qa=FakeQA(), compare=False), "where did I leave my keys")
    assert len(body["box"]["calls"]) == 3
    assert "tools" not in llama.chats[3]  # the fourth turn offers no tools
    assert body["brain"] is None


def test_a_brain_that_fails_says_so_and_the_box_still_answers(tmp_path):
    llama = FakeLlama([said("Hello.")])
    body = ask(client(tmp_path, llama, qa=FakeQA(), pair=FakePair([], fail=RuntimeError("the head pod answered HTTP 502"))),
               "hello")
    assert body["box"]["answer"] == "Hello."
    assert body["brain"]["error"] == "RuntimeError: the head pod answered HTTP 502"


def test_a_side_that_takes_too_long_is_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(lab_deploy, "SIDE_TIMEOUT_S", 0.05)
    llama = FakeLlama([said("Hello.")])
    body = ask(client(tmp_path, llama, qa=FakeQA(), pair=FakePair([said("late")], delay=6)), "hello")
    assert body["box"]["answer"] == "Hello." and body["brain"]["error"].startswith("no answer in")


def test_the_flag_turns_the_brain_off(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_ASK_COMPARE", "off")
    pair = FakePair([said("never")])
    body = ask(client(tmp_path, FakeLlama([said("Hello.")]), qa=FakeQA(), pair=pair), "hello")
    assert body["brain"] is None and pair.bodies == []


def test_without_keys_for_the_pods_the_brain_says_so(tmp_path):
    # a test app has no app.state.atlas, so the box's .env is never read here
    body = ask(client(tmp_path, FakeLlama([said("Hello.")]), qa=FakeQA()), "hello")
    assert body["brain"]["error"].startswith("this box has no keys for the pods")


def test_a_text_tool_call_from_a_server_without_a_parser_still_runs(tmp_path):
    pair = FakePair([said(PICK), said("On the desk.")])
    qa = FakeQA()
    body = ask(client(tmp_path, FakeLlama([said("Hi.")]), qa=qa, pair=pair), "where is my water bottle now")
    assert body["brain"]["calls"] == [{"name": "last_seen", "arguments": {"label": "water bottle"}}]
    assert body["brain"]["answer"] == "On the desk."


def test_nothing_answering_is_still_503(tmp_path):
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    app = FastAPI()
    runner = DeployRunner(tmp_path / "models", tmp_path / "state", llm_url="http://llm:8080")
    app.include_router(lab_deploy_router(runner, ask_transport=httpx.MockTransport(down), qa=FakeQA(), compare=False))
    r = TestClient(app).post("/api/learning/lab/deploy/ask", json={"question": "Have you seen my water bottle?"})
    assert r.status_code == 503
