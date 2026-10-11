"""The live chat beside the power graph: tokens as Server-Sent Events, then the joules for that answer.

Every model server and sampler is a fake; nothing reaches a pod or the box."""

import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.circle.brain_pair import Pod
from contexts.learning import live_chat
from contexts.learning.lab_deploy import DeployRunner, lab_deploy_router
from contexts.learning.tests.test_ask_compare import FakeMeter, FakeQA


def chunk(content: str | None = None, tool: dict | None = None, usage: int | None = None) -> str:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if tool is not None:
        delta["tool_calls"] = [tool]
    body: dict = {"choices": [{"index": 0, "delta": delta}]} if (content is not None or tool is not None) else {"choices": []}
    if usage is not None:
        body["usage"] = {"completion_tokens": usage}
    return f"data: {json.dumps(body)}\n\n"


def sse_body(parts: list[str]) -> bytes:
    return ("".join(parts) + "data: [DONE]\n\n").encode()


class FakeStreamServer:
    """An OpenAI chat server that streams scripted turns, recording each request body and path."""

    def __init__(self, turns: list[list[str]], status: int = 200) -> None:
        self.turns = list(turns)
        self.bodies: list[dict] = []
        self.urls: list[str] = []
        self.headers: list[dict] = []
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        self.headers.append(dict(request.headers))
        if not request.url.path.endswith("/v1/chat/completions"):
            return httpx.Response(404)
        self.bodies.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, text="nope")
        parts = self.turns.pop(0) if self.turns else [chunk("done")]
        return httpx.Response(200, content=sse_body(parts), headers={"content-type": "text/event-stream"})


class FakeStreamPair:
    """The pair's shape that live_chat reads: the head pod's proxy base and token, the transport, the samplers."""

    configured = True
    reason = None
    user_agent = "atlas-test"

    def __init__(self, server: FakeStreamServer, energy: dict | None = None) -> None:
        self.head = Pod("pod C", "head", "pod-a", "https://hub.test/user/u/proxy", "head-token")
        self.transport = httpx.MockTransport(server)
        self.energy = energy
        self.windows: list[tuple[float, float]] = []

    async def window_energy(self, start, end):
        self.windows.append((start, end))
        return self.energy


PAIR_ENERGY = {"joules": 812.4, "watts_avg": 190.1, "seconds": 4.27,
               "pods": {"C": {"joules": 430.0, "watts_avg": 100.7, "idle_w": 10.27, "samples": 5},
                        "D": {"joules": 382.4, "watts_avg": 89.6, "idle_w": 10.27, "samples": 5}}}


def client(tmp_path, box: FakeStreamServer | None = None, **kwargs) -> TestClient:
    app = FastAPI()
    runner = DeployRunner(tmp_path / "models", tmp_path / "state", llm_url="http://llm:8080")
    transport = httpx.MockTransport(box or FakeStreamServer([]))
    app.include_router(lab_deploy_router(runner, ask_transport=transport, **kwargs))
    return TestClient(app)


def events(c: TestClient, body: dict) -> list[tuple[str, dict]]:
    r = c.post("/api/learning/lab/live-chat", json=body)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    out = []
    for block in r.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_the_pair_streams_tokens_then_done_with_both_pods_joules(tmp_path):
    server = FakeStreamServer([[chunk("Hello"), chunk(" there,"), chunk(" friend."), chunk(usage=3)]])
    pair = FakeStreamPair(server, PAIR_ENERGY)
    got = events(client(tmp_path, qa=FakeQA(), pair=pair), {"question": "how are you?", "target": "pair"})
    tokens = [d["text"] for e, d in got if e == "token"]
    assert tokens == ["Hello", " there,", " friend."]
    assert got[-1][0] == "done"
    done = got[-1][1]
    assert done["answer"] == "Hello there, friend." and done["tokens"] == 3 and done["target"] == "pair"
    assert done["energy"] == PAIR_ENERGY and len(pair.windows) == 1
    assert done["seconds"] >= 0 and done["ttft_s"] is not None
    # through the hub proxy, port 8000, the head's token, served name "teacher", streamed, tools offered not forced
    assert server.urls[0] == "https://hub.test/user/u/proxy/8000/v1/chat/completions"
    assert server.headers[0]["authorization"] == "token head-token"
    body = server.bodies[0]
    assert body["model"] == "teacher" and body["stream"] is True and body["tool_choice"] == "auto" and body["tools"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_streamed_tool_call_runs_on_memory_then_the_answer_streams(tmp_path):
    call = [chunk(tool={"index": 0, "id": "c1", "function": {"name": "last_seen", "arguments": "{\"label\": \"wat"}}),
            chunk(tool={"index": 0, "function": {"arguments": "er bottle\"}"}})]
    answer = [chunk("On the desk"), chunk(" at 2:05 PM.")]
    server = FakeStreamServer([call, answer])
    qa = FakeQA()
    got = events(client(tmp_path, qa=qa, pair=FakeStreamPair(server, PAIR_ENERGY)),
                 {"question": "Have you seen my water bottle?", "target": "pair"})
    kinds = [e for e, _ in got]
    assert kinds == ["tool", "token", "token", "done"]
    assert got[0][1] == {"name": "last_seen", "arguments": {"label": "water bottle"}}
    assert qa.ran == [("last_seen", {"label": "water bottle"})]
    done = got[-1][1]
    assert done["answer"] == "On the desk at 2:05 PM." and done["tokens"] == 4
    assert done["calls"] == [{"name": "last_seen", "arguments": {"label": "water bottle"}}]
    # one round: the answer turn has the tool's result and no tools offered
    second = server.bodies[1]
    assert "tools" not in second and any(m["role"] == "tool" and "2:05 PM" in m["content"] for m in second["messages"])


def test_tool_markup_and_thinking_in_text_never_reach_the_transcript(tmp_path):
    text_call = [chunk("<think>hm</think>"), chunk("<tool_call>\n<function=last_seen>\n<parameter=label>\nkeys"),
                 chunk("\n</parameter>\n</function>\n</tool_call>")]
    server = FakeStreamServer([text_call, [chunk("By the door.")]])
    got = events(client(tmp_path, box=server, qa=FakeQA(), meter=FakeMeter()), {"question": "my keys?", "target": "box"})
    assert [d["text"] for e, d in got if e == "token"] == ["By the door."]
    assert got[0] == ("tool", {"name": "last_seen", "arguments": {"label": "keys"}})


def test_the_box_streams_from_llama_server_with_the_jetson_joules(tmp_path):
    server = FakeStreamServer([[chunk("Fine,"), chunk(" thanks.")]])
    got = events(client(tmp_path, box=server, qa=FakeQA(), meter=FakeMeter()), {"question": "how are you", "target": "box"})
    done = got[-1][1]
    assert server.urls[0] == "http://llm:8080/v1/chat/completions" and server.bodies[0]["stream"] is True
    assert done["answer"] == "Fine, thanks." and done["tokens"] == 2
    assert done["energy"]["joules"] == 41.3 and done["energy"]["idle_w"] == 15.76
    assert done["energy"]["watts_avg"] == round(41.3 / 2.4, 1)


def test_meters_offline_gives_null_energy_not_a_made_up_one(tmp_path):
    server = FakeStreamServer([[chunk("Hi.")]])
    got = events(client(tmp_path, qa=FakeQA(), pair=FakeStreamPair(server, None)), {"question": "hi", "target": "pair"})
    assert got[-1][0] == "done" and got[-1][1]["energy"] is None


def test_a_failing_model_server_is_an_error_event(tmp_path):
    server = FakeStreamServer([], status=502)
    got = events(client(tmp_path, qa=FakeQA(), pair=FakeStreamPair(server, PAIR_ENERGY)), {"question": "hi", "target": "pair"})
    assert got == [("error", {"error": "the model server answered HTTP 502", "seconds": got[0][1]["seconds"]})]


def test_no_keys_for_the_pods_is_an_error_event(tmp_path):
    got = events(client(tmp_path, qa=FakeQA()), {"question": "hi", "target": "pair"})
    assert got[0][0] == "error" and "keys" in got[0][1]["error"]


def test_bad_requests_are_400(tmp_path):
    c = client(tmp_path, qa=FakeQA())
    assert c.post("/api/learning/lab/live-chat", json={"question": " "}).status_code == 400
    assert c.post("/api/learning/lab/live-chat", json={"question": "hi", "target": "cloud"}).status_code == 400


def test_the_timeout_is_an_error_event():
    import asyncio

    async def slow(*args, **kwargs):
        await asyncio.sleep(5)

    async def run():
        original = live_chat.run_chat
        live_chat.run_chat = slow
        try:
            return [x async for x in live_chat.event_stream("hi", "box", timeout_s=0.2)]
        finally:
            live_chat.run_chat = original

    out = asyncio.run(run())
    assert len(out) == 1 and out[0].startswith("event: error") and "no answer in 0 s" in out[0]


def test_visible_holds_back_a_tag_that_is_still_arriving():
    assert live_chat.visible("Hello <") == "Hello "
    assert live_chat.visible("Hello <tool_") == "Hello "
    assert live_chat.visible("a < b") == "a < b"
    assert live_chat.visible("<think>x") == ""
    assert live_chat.visible("<think>x</think>Yes.") == "Yes."
