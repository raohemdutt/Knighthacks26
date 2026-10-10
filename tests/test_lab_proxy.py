"""lab.proxy: the box's call path to a pod's model server through the hub proxy.

Every test drives the real PodLink, ModelServer and ProxyTransport (and, under the transport, the real
LlamaServerRuntime). The only double is the network between the box and the pod, lab.proxy_fake.FakePodProxy: an httpx
transport on a fake clock that answers with the shapes recorded in lab/refine/research/plumbing/proxy-path.md, so
nothing here opens a socket, sleeps for real, reads a .env file or calls a pod."""
from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import httpx
import pytest

from contracts.types import ChatMessage, ModelVersion, ToolCall
from lab import proxy
from lab.hub import Account, Hub
from lab.hub_fake import FakeClock
from lab.proxy import (BadToken, Blocked, GatewayTimeout, HubDown, ModelServerDown, ModelServerError, NotFound,
                       PodLink, PodStarting, PodStopped, PodUnresponsive, ProxyError, ProxyTimeout, ProxyTransport,
                       StreamInterrupted, TooLarge, Unauthorized, Unreachable, collect_chat, parse_sse,
                       proxied_runtime)
from lab.proxy_fake import FakePodProxy

REPO = Path(__file__).resolve().parents[1]
TOKEN = "0123456789abcdef0123456789abcdef"
OTHER_TOKEN = "fedcba9876543210fedcba9876543210"
USER = "roscon01"
URL = "https://hub.test"
ENV = {"AUP_HUB_USER_C": USER, "AUP_HUB_TOKEN_C": TOKEN, "LAB_HUB_URL": URL}
CHAT = "/proxy/8000/v1/chat/completions"


def messages(text="Hello"):
    return [{"role": "user", "content": text}]


class World:
    """One account's pod on a fake proxy, a teacher listening on port 8000, and a link wired to both."""

    def __init__(self, *, user_agent=None, **fake_options):
        self.clock = FakeClock()
        self.fake = FakePodProxy(self.clock, **fake_options)
        self.account = self.fake.add_account("C", user=USER, token=TOKEN)
        self.teacher = self.fake.serve_model("C", 8000)
        self.records = []
        self.link = self.make_link(self.account, user_agent=user_agent)
        self.server = self.link.server(8000)

    def make_link(self, account, *, user_agent=None, base_url=None):
        hub = Hub(account, base_url=base_url or self.fake.url, transport=self.fake.transport, sleep=self.clock.sleep,
                  environ={}, user_agent=user_agent)
        return PodLink(hub, transport=self.fake.transport, sleep=self.clock.sleep, clock=self.clock.time,
                       observer=self.records.append)

    def runtime(self, **options):
        return proxied_runtime(ModelVersion(version="teacher-0", base="Qwen/Qwen3.8-27B"), self.link, 8000,
                               model_name="teacher", **options)


# --- the address and the wire ------------------------------------------------------------------

def test_a_model_server_is_reached_at_the_plain_proxy_address_with_the_token_and_a_browser_agent():
    w = World()

    assert w.server.models() == ["teacher"]

    call = w.fake.calls[0]
    assert (call.method, call.path) == ("GET", f"/user/{USER}/proxy/8000/v1/models")
    assert call.headers["authorization"] == f"token {TOKEN}"
    assert call.headers["user-agent"].startswith("Mozilla/5.0")
    assert call.headers["accept"] == "application/json"
    assert w.link.proxy_url(8000, "v1/models") == f"{URL}/user/{USER}/proxy/8000/v1/models"
    assert w.server.url() == f"{URL}/user/{USER}/proxy/8000/"
    assert all("/proxy/absolute/" not in c.path for c in w.fake.calls)


def test_the_hub_address_is_the_one_lab_hub_settled_on_not_one_named_here():
    clock = FakeClock()
    fake = FakePodProxy(clock, host="sjc.test")
    account = fake.add_account("D", user="roscon02", token=OTHER_TOKEN)
    fake.serve_model("D", 8000)
    hub = Hub(account, base_url="https://sjc.test", transport=fake.transport, sleep=clock.sleep, environ={})
    link = PodLink(hub, transport=fake.transport, sleep=clock.sleep, clock=clock.time)

    assert link.server(8000).models() == ["teacher"]
    assert link.proxy_url(8000) == "https://sjc.test/user/roscon02/proxy/8000/"


@pytest.mark.parametrize("port", [0, 65536, -1, "8000", 8000.0, True, None])
def test_a_port_must_be_a_port_number(port):
    w = World()
    with pytest.raises(ValueError):
        w.link.server(port)
    with pytest.raises(ValueError):
        w.link.proxy_url(port)


def test_a_wrong_token_is_a_bad_token_and_the_redirect_is_never_followed():
    w = World()
    link = w.make_link(Account("C", USER, OTHER_TOKEN))

    with pytest.raises(BadToken) as caught:
        link.server(8000).models()

    assert (caught.value.status, caught.value.kind, caught.value.retryable) == (302, "bad_token", False)
    assert len(w.fake.calls) == 1
    assert w.fake.calls[0].headers["authorization"] == f"token {OTHER_TOKEN}"
    assert "oauth2" not in str(caught.value)


def test_a_wrong_token_on_a_post_is_403_and_the_call_is_not_repeated():
    w = World()
    link = w.make_link(Account("C", USER, OTHER_TOKEN))

    with pytest.raises(BadToken) as caught:
        link.server(8000).chat(messages())

    assert caught.value.status == 403
    assert len(w.fake.calls) == 1
    assert w.teacher.requests == []


def test_a_right_token_on_another_accounts_address_is_a_bad_token():
    w = World()
    w.fake.add_account("D", user="roscon02", token=OTHER_TOKEN)
    link = w.make_link(Account("C", "roscon02", TOKEN))

    with pytest.raises(BadToken) as caught:
        link.server(8000).models()

    assert caught.value.status == 302
    assert w.fake.calls[0].path == "/user/roscon02/proxy/8000/v1/models"


def test_a_user_agent_the_front_end_refuses_is_blocked_not_a_bad_token():
    w = World(user_agent="python-httpx/0.28")

    with pytest.raises(Blocked) as caught:
        w.server.models()

    assert (caught.value.kind, caught.value.status, caught.value.retryable) == ("blocked", 403, False)
    assert "$LAB_HUB_USER_AGENT" in str(caught.value)


def test_no_message_or_record_holds_the_token_or_the_account_name():
    w = World()
    w.fake.stop_pod("C")
    reply_before = None
    with pytest.raises(PodStopped) as stopped:
        w.server.models()
    w.fake.start_pod("C")
    reply_before = w.server.get("v1/models")
    w.fake.queue("GET", "/proxy/8000/health", 401, body={"error": f"bad key for /user/{USER}/ with {TOKEN}"})
    with pytest.raises(Unauthorized) as unauthorized:
        w.server.get("health")

    for error in (stopped.value, unauthorized.value):
        for text in (str(error), error.detail, error.where, repr(error)):
            assert TOKEN not in text
            assert USER not in text
    assert "<account>" in stopped.value.detail
    assert TOKEN not in repr(reply_before) and TOKEN not in repr(w.link)
    for record in w.records:
        assert TOKEN not in repr(record) and USER not in repr(record)


def test_the_observer_sees_each_call_without_the_address():
    w = World()
    w.server.models()
    w.fake.queue("GET", "/proxy/8000/health", 424, body={"message": "gone"})
    with pytest.raises(PodStopped):
        w.server.get("health")

    ok, failed = w.records
    assert (ok.method, ok.where, ok.status, ok.kind, ok.attempts) == ("GET", "/proxy/8000/v1/models", 200, "ok", 1)
    assert (failed.where, failed.status, failed.kind) == ("/proxy/8000/health", 424, "pod_stopped")
    assert ok.first_byte_s is not None and ok.elapsed_s >= 0


def test_an_observer_that_raises_does_not_break_the_call():
    w = World()

    def broken(record):
        raise RuntimeError("observer bug")

    link = PodLink(Hub(w.account, base_url=URL, transport=w.fake.transport, environ={}), transport=w.fake.transport,
                   sleep=w.clock.sleep, clock=w.clock.time, observer=broken)

    assert link.server(8000).models() == ["teacher"]


# --- every failure has its error -----------------------------------------------------------------

HTML = {"content-type": "text/html"}
JUPYTER_500 = "<html><body><h1>500 : Internal Server Error</h1></body></html>"
FAILURES = [
    # status, headers, JSON body, text, error class, kind, retryable, detail that must show
    (302, {"location": "/hub/api/oauth2/authorize?client_id=x"}, None, "", BadToken, "bad_token", False, ""),
    (303, {"location": "/hub/spawn-pending/roscon01"}, None, "", PodStarting, "pod_starting", True, ""),
    (401, {}, {"error": "Unauthorized"}, None, Unauthorized, "unauthorized", False, "Unauthorized"),
    (403, {}, {"message": "Forbidden"}, None, BadToken, "bad_token", False, "Forbidden"),
    (403, HTML, None, "Error 1010: browser signature banned", Blocked, "blocked", False, "browser signature banned"),
    (404, {}, {"detail": "Not Found"}, None, NotFound, "not_found", False, "Not Found"),
    (413, HTML, None, "<html><body>413 Request Entity Too Large</body></html>", TooLarge, "too_large", False,
     "413 Request Entity Too Large"),
    (424, {}, {"status": 424, "message": "server is not running"}, None, PodStopped, "pod_stopped", False,
     "server is not running"),
    (500, HTML, None, JUPYTER_500, ModelServerDown, "model_server_down", False, "500 : Internal Server Error"),
    (500, {}, {"object": "error", "message": "boom", "code": 500}, None, ModelServerError, "model_server_error", False,
     "boom"),
    (400, {}, {"error": {"message": "max_tokens is too large"}}, None, ModelServerError, "model_server_error", False,
     "max_tokens is too large"),
    (429, {}, {"message": "slow down"}, None, ModelServerError, "model_server_error", False, "slow down"),
    (502, HTML, None, "<html>Bad gateway</html>", HubDown, "hub_down", True, "Bad gateway"),
    (503, {}, None, "Service Unavailable", PodUnresponsive, "pod_unresponsive", True, "Service Unavailable"),
    (504, HTML, None, "<html>Gateway time-out</html>", HubDown, "hub_down", True, "Gateway time-out"),
    (521, HTML, None, "<html>Web server is down</html>", HubDown, "hub_down", True, "Web server is down"),
    (524, HTML, None, "<html>A timeout occurred</html>", GatewayTimeout, "gateway_timeout", False, "A timeout occurred"),
    (599, {}, None, "HTTP 599: Timeout while connecting", ProxyTimeout, "proxy_timeout", False, "Timeout while connecting"),
]


@pytest.mark.parametrize("status,headers,body,text,cls,kind,retryable,detail", FAILURES)
def test_every_failure_status_becomes_its_own_error(status, headers, body, text, cls, kind, retryable, detail):
    w = World()
    w.fake.queue("GET", "/proxy/8000/v1/models", status, body=body, text=text, headers=headers, times=10)

    with pytest.raises(ProxyError) as caught:
        w.server.models()

    error = caught.value
    assert type(error) is cls
    assert (error.kind, error.status, error.retryable) == (kind, status, retryable)
    assert isinstance(error, httpx.HTTPError)
    assert detail in error.detail
    assert str(error).startswith(f"pod C: GET /proxy/8000/v1/models answered {status} ({kind}): {cls.hint}")
    assert TOKEN not in str(error) and USER not in str(error)
    assert len(w.fake.calls) == (4 if status in (502, 503, 504) else 1)


def test_a_port_with_nothing_listening_is_a_model_server_down():
    w = World()

    with pytest.raises(ModelServerDown) as caught:
        w.link.server(9999).models()

    assert caught.value.status == 500
    assert w.fake.calls[0].path == f"/user/{USER}/proxy/9999/v1/models"


def test_a_stopped_pod_answers_503_while_its_route_lingers_and_then_424():
    w = World()
    w.fake.stop_pod("C", route_lingers_s=30.0)

    with pytest.raises(PodUnresponsive):
        w.server.models()
    assert w.clock.elapsed == pytest.approx(7.0, abs=1e-3)       # 1 + 2 + 4 seconds of pauses, then it gives up
    w.clock.sleep(30.0)
    with pytest.raises(PodStopped) as caught:
        w.server.models()

    assert caught.value.status == 424


def test_a_pod_that_is_starting_sends_the_caller_to_the_spawn_pending_page():
    w = World()
    w.fake.start_pod("C", ready=False)

    with pytest.raises(PodStarting) as caught:
        w.server.models()

    assert caught.value.retryable is True
    assert len(w.fake.calls) == 1


def test_the_hubs_front_door_down_is_hub_down_after_the_retries():
    w = World()
    w.fake.hub_down = True

    with pytest.raises(HubDown) as caught:
        w.server.models()

    assert (caught.value.status, caught.value.retryable) == (502, True)
    assert len(w.fake.calls) == 4


def test_a_request_body_over_the_edge_cap_is_too_large():
    w = World(edge_body_cap=2000)

    with pytest.raises(TooLarge) as caught:
        w.server.post_json("v1/chat/completions", {"messages": messages("x" * 5000)})

    assert caught.value.status == 413


# --- retries ---------------------------------------------------------------------------------------

def test_a_get_is_tried_again_after_a_gateway_error_with_growing_pauses():
    w = World()
    w.fake.queue("GET", "/proxy/8000/v1/models", 502, text="<html>Bad gateway</html>", headers=HTML)
    w.fake.queue("GET", "/proxy/8000/v1/models", 503, text="Service Unavailable")

    reply = w.server.get("v1/models")

    assert (reply.status, reply.attempts) == (200, 3)
    assert w.clock.elapsed == pytest.approx(3.0, abs=1e-3)       # 1 s, then 2 s
    assert [record.status for record in w.records] == [502, 503, 200]


def test_a_dropped_connection_on_a_get_is_tried_again_and_then_reported_as_unreachable():
    w = World()
    w.fake.queue_dropped("GET", "/proxy/8000/v1/models")
    assert w.server.models() == ["teacher"]

    w.fake.queue_dropped("GET", "/proxy/8000/v1/models", times=10)
    with pytest.raises(Unreachable) as caught:
        w.server.models()

    assert (caught.value.status, caught.value.retryable) == (None, True)
    assert "ReadError" in str(caught.value)


def test_a_post_is_never_sent_again_and_a_token_problem_is_never_retried():
    w = World()
    w.fake.queue("POST", CHAT, 502, text="<html>Bad gateway</html>", headers=HTML, times=5)
    with pytest.raises(HubDown):
        w.server.chat(messages())
    assert len(w.fake.calls) == 1

    w.fake.queue("GET", "/proxy/8000/health", 424, body={"message": "gone"}, times=5)
    with pytest.raises(PodStopped):
        w.server.get("health")
    assert len(w.fake.calls) == 2


# --- streaming ---------------------------------------------------------------------------------------

def test_a_chat_streams_with_the_exact_accept_header_and_the_pieces_arrive_as_the_model_writes_them():
    w = World()
    w.teacher.reply_tokens = 20
    seen = []

    done = w.server.chat(messages(), on_text=lambda piece: seen.append((piece, w.clock.elapsed)))

    assert [piece for piece, _ in seen] == ["w0"] + [f" w{i}" for i in range(1, 20)]
    assert [moment for _, moment in seen] == pytest.approx([0.55 + i / 7.7 for i in range(20)], abs=1e-3)
    assert done.content == " ".join(f"w{i}" for i in range(20))
    assert done.first_token_s == pytest.approx(0.55, abs=1e-3)
    assert list(done.chunk_times) == pytest.approx([0.55 + i / 7.7 for i in range(20)], abs=1e-3)
    assert (done.finish_reason, done.model) == ("stop", "teacher")
    assert done.usage == {"prompt_tokens": 1, "completion_tokens": 20, "total_tokens": 21}
    assert w.teacher.accepts == ["text/event-stream"]
    assert w.teacher.requests[0]["stream"] is True
    assert w.fake.calls[0].headers["accept-encoding"] == "identity"


def test_the_control_a_reply_asked_for_with_any_other_accept_header_is_held_and_arrives_in_one_burst():
    w = World()
    w.teacher.reply_tokens = 20

    with w.link.stream("POST", w.server.rel("v1/chat/completions"), accept="*/*",
                       json={"messages": messages(), "stream": True}) as handle:
        done = collect_chat(handle)

    assert len(set(done.chunk_times)) == 1
    assert done.first_token_s == pytest.approx(0.55 + 19 / 7.7, abs=1e-3)
    assert done.content == " ".join(f"w{i}" for i in range(20))
    assert w.teacher.accepts == ["*/*"]


def test_an_answer_of_150_seconds_streams_through_where_a_held_one_is_cut_at_125():
    w = World()
    w.teacher.reply_tokens = 1150

    done = w.server.chat(messages())

    assert len(done.content.split(" ")) == 1150
    assert done.elapsed_s == pytest.approx(0.55 + 1149 / 7.7, abs=0.01)
    assert done.first_token_s == pytest.approx(0.55, abs=1e-3)

    held = World()
    held.teacher.reply_tokens = 1150
    with pytest.raises(GatewayTimeout) as caught:
        with held.link.stream("POST", held.server.rel("v1/chat/completions"), accept="*/*",
                              json={"messages": messages(), "stream": True}) as handle:
            collect_chat(handle)
    assert caught.value.status == 524
    assert held.clock.elapsed == pytest.approx(125.0, abs=1e-3)


def test_a_reply_that_is_not_streamed_is_cut_at_125_seconds_and_the_error_says_to_stream():
    w = World()
    w.teacher.reply_tokens = 1150

    with pytest.raises(GatewayTimeout) as caught:
        w.server.post_json("v1/chat/completions", {"messages": messages(), "stream": False})

    assert (caught.value.kind, caught.value.status, caught.value.retryable) == ("gateway_timeout", 524, False)
    assert "stream true" in str(caught.value)
    assert w.clock.elapsed == pytest.approx(125.0, abs=1e-3)
    assert len(w.fake.calls) == 1


def test_tool_calls_are_put_together_from_the_fragments_the_server_streams():
    w = World()
    arguments = {"query": "keys", "hours": 3}
    w.teacher.tool_calls = [("search_memory", arguments)]
    tools = [{"type": "function", "function": {"name": "search_memory", "parameters": {"type": "object"}}}]

    done = w.server.chat(messages("Where are my keys?"), tools=tools)

    assert done.tool_calls == [{"id": "call_0", "type": "function",
                                "function": {"name": "search_memory", "arguments": json.dumps(arguments)}}]
    assert json.loads(done.tool_calls[0]["function"]["arguments"]) == arguments
    assert (done.content, done.finish_reason) == ("", "tool_calls")
    assert w.teacher.requests[0]["tools"] == tools
    assert done.message() == {"role": "assistant", "content": "", "tool_calls": done.tool_calls}


def test_the_request_carries_what_chat_was_given():
    w = World()
    w.teacher.reply_tokens = 10
    schema = {"type": "object"}

    done = w.server.chat(messages(), model="teacher", max_tokens=3, temperature=0.2, thinking=False,
                         json_schema=schema, extra={"top_p": 0.9})

    sent = w.teacher.requests[0]
    assert sent == {
        "messages": messages(), "stream": True, "stream_options": {"include_usage": True}, "model": "teacher",
        "max_tokens": 3, "temperature": 0.2, "top_p": 0.9, "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}}
    assert (done.content, done.finish_reason) == ("w0 w1 w2", "length")


def test_a_model_the_server_does_not_serve_is_not_found_with_the_servers_words():
    w = World()

    with pytest.raises(NotFound) as caught:
        w.server.chat(messages(), model="nope")

    assert caught.value.detail == "The model `nope` does not exist."


def test_an_error_the_model_server_answers_with_is_a_model_server_error_with_its_message():
    w = World()
    w.teacher.fail_next.append((500, {"object": "error", "message": "CUDA out of memory", "type": "InternalServerError"}))

    with pytest.raises(ModelServerError) as caught:
        w.server.chat(messages())

    assert (caught.value.status, caught.value.detail) == (500, "CUDA out of memory")
    assert w.server.chat(messages()).content == "Hello from the fake teacher."


def test_a_stream_that_breaks_off_is_interrupted_and_keeps_the_text_that_arrived():
    w = World()
    w.teacher.reply_tokens = 20
    w.teacher.break_stream_after = 6          # the role chunk and five pieces
    pieces = []

    with pytest.raises(StreamInterrupted) as caught:
        w.server.chat(messages(), on_text=pieces.append)

    assert caught.value.partial == "w0 w1 w2 w3 w4"
    assert pieces == ["w0", " w1", " w2", " w3", " w4"]
    assert (caught.value.kind, caught.value.retryable) == ("stream_interrupted", True)
    assert TOKEN not in str(caught.value) and USER not in str(caught.value)


def test_a_stream_that_ends_before_done_is_interrupted_not_taken_for_a_whole_answer():
    w = World()
    event = 'data: {"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": null}]}\n\n'
    w.fake.queue("POST", CHAT, 200, text=event, headers={"content-type": "text/event-stream"})

    with pytest.raises(StreamInterrupted) as caught:
        w.server.chat(messages())

    assert caught.value.partial == "Hel"
    assert "before [DONE]" in str(caught.value)


def test_a_stream_that_ends_with_a_finish_reason_but_no_done_is_a_whole_answer():
    w = World()
    event = 'data: {"choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}]}\n\n'
    w.fake.queue("POST", CHAT, 200, text=event, headers={"content-type": "text/event-stream"})

    done = w.server.chat(messages())

    assert (done.content, done.finish_reason) == ("Hi", "stop")


def test_an_error_event_inside_a_stream_is_a_model_server_error():
    w = World()
    w.fake.queue("POST", CHAT, 200, text='data: {"error": {"message": "out of memory"}}\n\n',
                 headers={"content-type": "text/event-stream"})

    with pytest.raises(ModelServerError) as caught:
        w.server.chat(messages())

    assert caught.value.detail == "out of memory"


def test_an_event_that_is_not_json_interrupts_the_stream():
    w = World()
    w.fake.queue("POST", CHAT, 200, text="data: this is not json\n\n", headers={"content-type": "text/event-stream"})

    with pytest.raises(StreamInterrupted):
        w.server.chat(messages())


def test_models_that_arrive_as_something_else_are_a_model_server_error():
    w = World()
    w.fake.queue("GET", "/proxy/8000/v1/models", 200, body={"unexpected": True})

    with pytest.raises(ModelServerError) as caught:
        w.server.models()

    assert "not a model list" in str(caught.value)


def test_get_reads_any_path_whole_and_a_proxy_reply_has_no_request_in_it():
    w = World()

    reply = w.server.get("metrics", accept="text/plain")

    assert reply.status == 200
    assert reply.text == "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
    assert not hasattr(reply, "request")
    assert w.teacher.accepts == ["text/plain"]


def test_the_hub_token_reaches_the_model_server_because_the_proxy_forwards_every_header():
    w = World()

    w.server.chat(messages())

    assert w.teacher.authorizations == [f"token {TOKEN}"]


# --- server-sent events as text ---------------------------------------------------------------------------

def test_parse_sse_follows_the_event_stream_format():
    lines = [": a comment", "event: ping", "id: 7", "data: first", "data: second", "", "data:tight", "",
             "ignored: field", "", "data: unfinished"]
    ticks = iter(range(100))

    events = list(parse_sse(lines, lambda: float(next(ticks))))

    assert [(e.data, e.event, e.id, e.t) for e in events] == [
        ("first\nsecond", "ping", "7", 0.0), ("tight", None, "7", 1.0), ("unfinished", None, "7", 2.0)]


# --- the stock runtime over the proxy --------------------------------------------------------------------------

def test_the_stock_runtime_answers_through_the_proxy_with_the_token_the_model_name_and_a_stream():
    w = World()
    runtime = w.runtime()

    text = runtime.summarize(["the keys were on the desk", "someone left"])

    assert text == "Hello from the fake teacher."
    sent = w.teacher.requests[0]
    assert sent["model"] == "teacher" and sent["stream"] is True
    assert w.teacher.accepts == ["text/event-stream"]
    assert w.teacher.authorizations == [f"token {TOKEN}"]
    call = w.fake.calls[0]
    assert call.path == f"/user/{USER}/proxy/8000/v1/chat/completions"
    assert call.headers["user-agent"].startswith("Mozilla/5.0")


def test_a_reply_the_runtime_waits_for_survives_the_125_second_cut_because_it_is_streamed_underneath():
    w = World()
    w.teacher.tokens_per_s = 3.0
    w.teacher.reply_tokens = 600           # the runtime asks for 512 tokens: 171 seconds

    result = w.runtime().chat([ChatMessage(role="user", content="Tell me everything")])

    assert len(result.content.split(" ")) == 512
    assert result.model_version == "teacher-0"
    assert w.clock.elapsed == pytest.approx(0.55 + 511 / 3.0, abs=0.01)

    held = World()
    held.teacher.tokens_per_s = 3.0
    held.teacher.reply_tokens = 600
    with pytest.raises(GatewayTimeout):
        held.runtime(upgrade=False).chat([ChatMessage(role="user", content="Tell me everything")])
    assert held.clock.elapsed == pytest.approx(125.0, abs=1e-3)


def test_the_runtimes_streaming_chat_delivers_pieces_as_they_are_written():
    w = World()
    w.teacher.reply_tokens = 8
    seen = []

    result = w.runtime().chat([ChatMessage(role="user", content="Hi")],
                              on_text=lambda piece: seen.append((piece, w.clock.elapsed)))

    assert result.content == " ".join(f"w{i}" for i in range(8))
    assert [moment for _, moment in seen] == pytest.approx([0.55 + i / 7.7 for i in range(8)], abs=1e-3)
    assert w.teacher.accepts == ["text/event-stream"]


def test_the_runtimes_tool_calls_come_back_whole():
    w = World()
    w.teacher.tool_calls = [("search_memory", {"query": "keys"})]
    tools = [{"type": "function", "function": {"name": "search_memory", "parameters": {"type": "object"}}}]

    result = w.runtime().chat([ChatMessage(role="user", content="Where are my keys?")], tools=tools)

    assert result.tool_calls == [ToolCall(id="call_0", name="search_memory", arguments={"query": "keys"})]
    assert result.content == ""


def test_the_runtime_raises_the_mapped_errors_which_are_also_httpx_errors():
    w = World()
    runtime = w.runtime()
    w.fake.stop_pod("C")

    with pytest.raises(PodStopped) as caught:
        runtime.summarize(["a"])

    assert isinstance(caught.value, httpx.HTTPError)
    assert caught.value.status == 424


def test_other_requests_through_the_transport_carry_the_token_and_keep_their_own_accept_header():
    w = World()
    runtime = w.runtime()

    reply = runtime.http.get("/v1/models")

    assert reply.json()["data"][0]["id"] == "teacher"
    assert w.teacher.authorizations == [f"token {TOKEN}"]
    assert w.fake.calls[0].headers["accept"] == "*/*"
    assert w.fake.calls[0].path == f"/user/{USER}/proxy/8000/v1/models"


def test_the_transport_will_not_send_the_token_to_any_other_host():
    from runtime.llama import LlamaServerRuntime

    w = World()
    runtime = LlamaServerRuntime(ModelVersion(version="x", base="y"), base_url="https://evil.test/",
                                 transport=ProxyTransport(w.link))

    with pytest.raises(ValueError, match="only to the hub it was built for"):
        runtime.summarize(["a"])

    assert w.fake.calls == []


# --- the command line --------------------------------------------------------------------------------------

def run_cli(w, *argv, environ=None):
    out, err = io.StringIO(), io.StringIO()
    code = proxy.main(list(argv), environ=ENV if environ is None else environ, env_files=[], transport=w.fake.transport,
                      sleep=w.clock.sleep, clock=w.clock.time, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_probe_prints_one_json_line_with_the_models_and_the_health():
    w = World()

    code, out, _ = run_cli(w, "probe", "--account", "c", "--port", "8000")

    assert code == 0
    assert json.loads(out) == {"ok": True, "models": ["teacher"], "models_s": 0.0, "health_status": 200, "port": 8000,
                               "edge": {"server": "cloudflare", "cf_ray": None}}
    assert TOKEN not in out and USER not in out


def test_probe_with_a_wrong_token_reports_the_failure_the_hub_answers_with_and_never_sends_the_real_token():
    w = World()

    code, out, _ = run_cli(w, "probe", "--account", "C", "--wrong-token")

    assert code == 1
    report = json.loads(out)
    assert (report["ok"], report["kind"], report["status"], report["retryable"]) == (False, "bad_token", 302, False)
    assert all(call.headers["authorization"] != f"token {TOKEN}" for call in w.fake.calls)
    assert TOKEN not in out and USER not in out


def test_chat_streams_the_text_to_stderr_and_prints_the_timing():
    w = World()
    w.teacher.reply_tokens = 5

    code, out, err = run_cli(w, "chat", "--account", "C", "--prompt", "Say hello", "--max-tokens", "4", "--no-thinking")

    report = json.loads(out)
    assert code == 0 and err == "w0 w1 w2 w3\n"
    assert report["text"] == "w0 w1 w2 w3" and report["finish_reason"] == "length" and report["pieces"] == 4
    assert report["first_token_s"] == pytest.approx(0.55, abs=1e-3)
    assert w.teacher.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert w.teacher.requests[0]["max_tokens"] == 4


def test_a_pod_that_is_down_is_exit_1_with_the_kind_in_the_report():
    w = World()
    w.fake.stop_pod("C")

    code, out, _ = run_cli(w, "probe", "--account", "C")

    report = json.loads(out)
    assert code == 1
    assert (report["kind"], report["status"]) == ("pod_stopped", 424)


def test_missing_keys_are_exit_2_and_name_the_keys_not_values():
    w = World()

    code, out, _ = run_cli(w, "probe", "--account", "C", environ={"AUP_HUB_USER_C": USER})

    report = json.loads(out)
    assert code == 2 and report["kind"] == "missing_keys" and report["missing"] == ["AUP_HUB_TOKEN_C"]
    assert w.fake.calls == []


def test_an_account_is_one_letter():
    w = World()

    code, out, _ = run_cli(w, "probe", "--account", "roscon01")

    assert code == 2 and json.loads(out)["kind"] == "usage"


# --- the code the Jetson runs stays on what Python 3.10 has ---------------------------------------------------

THREE_ELEVEN_ONLY = {"UTC", "Self", "StrEnum", "tomllib", "TaskGroup", "ExceptionGroup", "BaseExceptionGroup",
                     "LiteralString", "assert_never"}


@pytest.mark.parametrize("name", ["proxy", "transfer", "proxy_fake"])
def test_the_new_lab_modules_parse_as_python_3_10_and_use_nothing_added_in_3_11(name):
    source = (REPO / "lab" / f"{name}.py").read_text(encoding="utf-8")

    tree = ast.parse(source, feature_version=(3, 10))

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            used.update(alias.name for alias in node.names)
            used.add(node.module or "")
        elif isinstance(node, ast.Import):
            used.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    assert used & THREE_ELEVEN_ONLY == set()
    assert source.isascii()
