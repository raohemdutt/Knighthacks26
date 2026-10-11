"""The shared brain as two pods (G6): GET /api/circle/shared-brain read from a fake hub proxy, never the real pods."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.circle.brain_pair import (CACHE_S, LINK, MODEL, BrainPair, Pod, build_brain_pair, parse_metrics,
                                        shared_brain_router)

HUB = "https://hub.example"
T0 = 1_791_700_000.0  # the pods' clock, epoch seconds

METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{engine="0",model_name="teacher"}} {running}
vllm:time_to_first_token_seconds_sum{{engine="0",model_name="teacher"}} {ttft_sum}
vllm:time_to_first_token_seconds_count{{engine="0",model_name="teacher"}} {ttft_count}
vllm:time_to_first_token_seconds_bucket{{engine="0",le="0.5",model_name="teacher"}} {ttft_count}
vllm:inter_token_latency_seconds_sum{{engine="0",model_name="teacher"}} {itl_sum}
vllm:inter_token_latency_seconds_count{{engine="0",model_name="teacher"}} {itl_count}
"""


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


class FakeProxy:
    """The hub's Jupyter proxy in front of both pods: the head's vLLM (8000) and each pod's sampler (8790)."""

    def __init__(self) -> None:
        self.users = {"roscon01": "token-c", "roscon02": "token-d"}
        self.down: set[str] = set()  # "<user>:<port>" that do not answer
        self.slow: set[str] = set()  # "<user>:<port>" that answer after 10 s
        self.calls: list[str] = []
        self.pod_now = T0 + 100
        self.watts = {"roscon01": 61.0, "roscon02": 48.0}
        self.metrics = {"running": 1, "ttft_sum": 2.9, "ttft_count": 10, "itl_sum": 50.0, "itl_count": 500}

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")  # "", "user", <user>, "proxy", <port>, ...
        user, port, path = parts[2], parts[4], "/" + "/".join(parts[5:])
        self.calls.append(f"{user}:{port}{path}")
        if request.headers.get("Authorization") != f"token {self.users.get(user)}":
            return httpx.Response(302, headers={"Location": "/hub/login"})
        if f"{user}:{port}" in self.slow:
            await asyncio.sleep(10)
        if f"{user}:{port}" in self.down:
            return httpx.Response(503, text="proxy error")
        if port == "8000" and user == "roscon01":
            if path == "/v1/models":
                return httpx.Response(200, json={"object": "list", "data": [{"id": "teacher"}, {"id": "group-brain"}]})
            if path == "/metrics":
                return httpx.Response(200, text=METRICS.format(**self.metrics))
        if port == "8790":
            last = 100
            stamp = {"pod_received": iso(self.pod_now), "pod_sent": iso(self.pod_now)}
            if path == "/status":
                return httpx.Response(200, json={"ok": True, "last_seq": last, **stamp})
            if path == "/samples":
                after = int(request.url.params.get("after", 0))
                samples = [{"seq": q, "event": {"ts": iso(T0 + q), "socket_w": self.watts[user]}}
                           for q in range(after + 1, last + 1)]
                return httpx.Response(200, json={"ok": True, "last_seq": last, "samples": samples, **stamp})
        return httpx.Response(404, json={"error": "no such path"})


def pair(proxy: FakeProxy, **kwargs) -> BrainPair:
    head = Pod("pod C", "head", "pod-a", f"{HUB}/user/roscon01/proxy", "token-c")
    half = Pod("pod D", "half", "pod-b", f"{HUB}/user/roscon02/proxy", "token-d")
    now = kwargs.pop("now", [0.0])
    return BrainPair(head, half, idle={"pod-a": 10.27, "pod-b": 10.27}, transport=httpx.MockTransport(proxy),
                     clock=lambda: T0, monotonic=lambda: now[0], **kwargs)


def snap(p: BrainPair) -> dict:
    return asyncio.run(p.snapshot())


def test_both_pods_up_gives_the_whole_card():
    body = snap(pair(FakeProxy()))
    assert body["configured"] is True and body["reason"] is None
    assert body["model"] == {**MODEL, "served": ["teacher", "group-brain"]}
    assert body["model"]["params_b"] == 122 and body["model"]["active_b"] == 10 and body["model"]["quant"] == "int4"
    head, half = body["members"]
    assert head == {"name": "pod C", "role": "head", "node": "pod-a", "layers": 24, "online": True, "watts_now": 61.0,
                    "watts_idle": 10.27, "vram_gb": 34.5}
    assert half == {"name": "pod D", "role": "half", "node": "pod-b", "layers": 24, "online": True, "watts_now": 48.0,
                    "watts_idle": 10.27, "vram_gb": 34.5}
    assert body["link"] == LINK == {"kind": "tunnel", "rtt_ms": 0.36, "mbps": 100}
    # lifetime means on the first read: 2.9 s / 10 first tokens, 50 s / 500 gaps -> 10 tokens a second
    assert body["serving"] == {"tok_s": 10.0, "ttft_s": 0.29, "requests_running": 1}
    assert body["updated"] == iso(T0)
    json.dumps(body)  # plain JSON


def test_serving_follows_the_requests_since_the_previous_read():
    proxy = FakeProxy()
    now = [0.0]
    p = pair(proxy, now=now)
    snap(p)
    proxy.metrics.update(ttft_sum=2.9 + 0.5, ttft_count=11, itl_sum=50.0 + 20 * 0.125, itl_count=520, running=0)
    now[0] += CACHE_S + 0.1
    body = snap(p)
    assert body["serving"] == {"tok_s": 8.0, "ttft_s": 0.5, "requests_running": 0}
    now[0] += CACHE_S + 0.1  # nothing new finished: the last request's figures stay
    assert snap(p)["serving"] == {"tok_s": 8.0, "ttft_s": 0.5, "requests_running": 0}


def test_answers_are_cached_for_two_seconds():
    proxy = FakeProxy()
    now = [0.0]
    p = pair(proxy, now=now)
    snap(p)
    first = len(proxy.calls)
    now[0] += CACHE_S - 0.1
    snap(p)
    assert len(proxy.calls) == first  # served from the cache
    now[0] += 0.2
    snap(p)
    # once a sampler's sequence is known each pod costs one /samples read, the head two more (models, metrics)
    assert len(proxy.calls) == first + 4
    assert not any(c.endswith("/status") for c in proxy.calls[first:])


def test_the_tokens_go_to_their_own_pod_and_never_into_the_answer():
    proxy = FakeProxy()
    body = snap(pair(proxy))
    text = json.dumps(body)
    assert "token-c" not in text and "token-d" not in text
    assert all(c.startswith(("roscon01:", "roscon02:")) for c in proxy.calls)
    assert any(c.startswith("roscon02:8790") for c in proxy.calls)
    assert not any(c.startswith("roscon02:8000") for c in proxy.calls)  # pod D is headless: only its sampler is read


def test_the_half_without_its_sampler_is_online_while_the_model_serves_but_has_no_watts():
    proxy = FakeProxy()
    proxy.down.add("roscon02:8790")
    head, half = snap(pair(proxy))["members"]
    assert head["online"] is True and head["watts_now"] == 61.0
    assert half["online"] is True and half["watts_now"] is None and half["watts_idle"] == 10.27


def test_both_down_is_offline_with_null_fields_not_an_error():
    proxy = FakeProxy()
    proxy.down |= {"roscon01:8000", "roscon01:8790", "roscon02:8790"}
    body = snap(pair(proxy))
    assert body["configured"] is True
    assert body["model"]["served"] == []
    assert [m["online"] for m in body["members"]] == [False, False]
    assert all(m["watts_now"] is None and m["vram_gb"] is None for m in body["members"])
    assert body["serving"] == {"tok_s": None, "ttft_s": None, "requests_running": None}


def test_a_source_that_does_not_answer_in_time_is_null_after_the_timeout():
    proxy = FakeProxy()
    proxy.slow.add("roscon01:8000")
    body = snap(pair(proxy, timeout_s=0.2))
    assert body["model"]["served"] == [] and body["serving"]["tok_s"] is None
    head, half = body["members"]
    assert head["online"] is True and head["watts_now"] == 61.0  # its sampler still answered
    assert half["online"] is True and half["watts_now"] == 48.0


def test_a_stale_sample_is_not_watts_now():
    proxy = FakeProxy()
    proxy.pod_now = T0 + 100 + 60  # the newest sample is a minute old by the pod's own clock
    head, _ = snap(pair(proxy))["members"]
    assert head["online"] is True and head["watts_now"] is None


def test_a_sampler_that_restarted_is_read_from_its_new_end():
    proxy = FakeProxy()
    now = [0.0]
    p = pair(proxy, now=now)
    snap(p)
    p._last_seq["pod C"] = 5000  # as if the sampler had counted far past 100 before it restarted
    proxy.watts["roscon01"] = 70.0
    now[0] += CACHE_S + 0.1
    assert snap(p)["members"][0]["watts_now"] == 70.0


def test_a_wrong_token_reads_as_offline():
    proxy = FakeProxy()
    proxy.users["roscon02"] = "another"
    proxy.down |= {"roscon01:8000"}
    _, half = snap(pair(proxy))["members"]
    assert half["online"] is False and half["watts_now"] is None


def test_parse_metrics_sums_label_sets_and_skips_comments():
    text = 'vllm:num_requests_running{engine="0"} 2.0\nvllm:num_requests_running{engine="1"} 1.0\n# x 9\nother 4\n'
    assert parse_metrics(text, ["vllm:num_requests_running", "other"]) == {"vllm:num_requests_running": 3.0,
                                                                          "other": 4.0}
    assert parse_metrics("vllm:x NaN\n", ["vllm:x"]) == {}


# --- configuration from the environment ---

ENV = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_C": "roscon01", "AUP_HUB_TOKEN_C": "token-c",
       "AUP_HUB_URL_C": HUB, "AUP_HUB_USER_D": "roscon02", "AUP_HUB_TOKEN_D": "token-d", "AUP_HUB_URL_D": HUB}


def test_build_reads_both_accounts_the_way_lab_hub_does():
    p = build_brain_pair(ENV, env_files=[], transport=httpx.MockTransport(FakeProxy()))
    assert p.configured and p.reason is None
    assert (p.head.name, p.head.base) == ("pod C", f"{HUB}/user/roscon01/proxy")
    assert (p.half.name, p.half.base) == ("pod D", f"{HUB}/user/roscon02/proxy")
    assert "token" not in repr(p.head)
    assert p.idle["pod-a"] == 10.27 and p.idle["pod-b"] == 10.27  # config/telemetry_idle.yaml
    body = asyncio.run(p.snapshot())
    assert [m["online"] for m in body["members"]] == [True, True]


def test_build_without_the_half_is_configured_and_names_the_missing_keys():
    env = {k: v for k, v in ENV.items() if not k.endswith("_D")}
    p = build_brain_pair(env, env_files=[])
    assert p.configured and p.half.base is None
    assert p.reason == "missing keys: AUP_HUB_USER_D, AUP_HUB_TOKEN_D"


def test_build_without_any_key_is_not_configured_and_fetches_nothing():
    proxy = FakeProxy()
    p = build_brain_pair({"LAB_POD_ROLES": "C:pod-a,D:pod-b"}, env_files=[], transport=httpx.MockTransport(proxy))
    body = asyncio.run(p.snapshot())
    assert body["configured"] is False and proxy.calls == []
    assert body["reason"] == "missing keys: AUP_HUB_USER_C, AUP_HUB_TOKEN_C, AUP_HUB_USER_D, AUP_HUB_TOKEN_D"
    assert [m["name"] for m in body["members"]] == ["pod C", "pod D"]
    assert all(m["online"] is False and m["watts_now"] is None for m in body["members"])


def test_build_with_unreadable_roles_is_not_configured():
    p = build_brain_pair({"LAB_POD_ROLES": "nonsense"}, env_files=[])
    assert not p.configured and "LAB_POD_ROLES" in p.reason


# --- the route ---

def client(p: BrainPair | None) -> TestClient:
    app = FastAPI()
    app.include_router(shared_brain_router(p))
    return TestClient(app)


def test_the_route_answers_the_card():
    r = client(pair(FakeProxy())).get("/api/circle/shared-brain")
    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is True and [m["name"] for m in body["members"]] == ["pod C", "pod D"]
    assert body["serving"]["tok_s"] == 10.0


def test_the_route_answers_200_without_wiring_or_when_reading_breaks():
    r = client(None).get("/api/circle/shared-brain")
    assert r.status_code == 200 and r.json()["configured"] is False

    p = pair(FakeProxy())

    async def broken():
        raise RuntimeError("a bug")

    p._build = broken  # type: ignore[method-assign]
    r = client(p).get("/api/circle/shared-brain")
    assert r.status_code == 200 and r.json()["configured"] is True
    assert [m["online"] for m in r.json()["members"]] == [False, False]
