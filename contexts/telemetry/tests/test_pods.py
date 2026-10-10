"""The pod reader: the pods' samples through the hub proxy, with the Jetson's receive time, into the log.

The hub proxy is the boundary: an httpx MockTransport answers as lab/pod_power.py's endpoint does, from the
recorded samples in fixtures/telemetry/pod_power.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from bus import EventBus
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.infrastructure.pods import PodEndpoint, PodReader, pod_endpoints
from contexts.telemetry.wiring import build_telemetry
from contracts import parse_event
from contracts.telemetry import EnergyCheck, GpuPower

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry" / "pod_power"
TOKEN = "hub-token-for-tests"
NOW = datetime(2026, 10, 10, 18, 2, tzinfo=UTC)
ENV = {"AUP_HUB_USER_A": "acct a", "AUP_HUB_TOKEN_A": TOKEN, "AUP_HUB_USER_B": "acct-b", "AUP_HUB_TOKEN_B": "other"}


def recorded(name: str = "samples_gpu-a.jsonl") -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


class FakePod:
    """GET /samples as lab/pod_power.py answers it, over a list of (seq, event) it holds."""

    def __init__(self, events: list[dict], started: str = "2026-10-10T18:00:00.000Z", first_seq: int = 1) -> None:
        self.held = list(enumerate(events, first_seq))
        self.started = started
        self.requests: list[httpx.Request] = []
        self.fail_with: int | Exception | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.fail_with, Exception):
            raise self.fail_with
        if self.fail_with is not None:
            return httpx.Response(self.fail_with, text="Bad Gateway")
        after, limit = int(request.url.params["after"]), int(request.url.params["limit"])
        page = [(seq, e) for seq, e in self.held if seq > after][:limit]
        return httpx.Response(200, json={
            "ok": True, "node": "gpu-a", "sampler_started": self.started,
            "first_seq": self.held[0][0] if self.held else None, "last_seq": self.held[-1][0] if self.held else 0,
            "samples": [{"seq": seq, "event": e} for seq, e in page]})


def collect(reader: PodReader) -> list:
    async def run():
        return [e async for e in reader.samples()]

    return asyncio.run(run())


def reader_for(pod: FakePod, **kw) -> PodReader:
    kw.setdefault("rounds", 1)
    return PodReader([PodEndpoint("gpu-a", "https://hub.test/user/acct/proxy/8790", TOKEN)],
                     transport=httpx.MockTransport(pod), clock=lambda: NOW, sleep=lambda _: asyncio.sleep(0), **kw)


def test_the_samples_come_back_as_gpu_power_events_stamped_with_the_jetsons_receive_time():
    events = recorded()[:5]
    pod = FakePod(events)
    got = collect(reader_for(pod))
    assert got == [parse_event({**e, "received_ts": NOW.isoformat()}) for e in events]
    assert all(isinstance(e, GpuPower) and e.received_ts == NOW for e in got)
    assert got[0].ts < got[0].received_ts  # the pod's time of the reading, and when the Jetson had it


def test_the_token_goes_only_in_the_authorization_header_and_the_url_asks_for_what_comes_after():
    pod = FakePod(recorded()[:3])
    collect(reader_for(pod, rounds=2))
    first, second = pod.requests
    assert first.headers["Authorization"] == f"token {TOKEN}"
    assert TOKEN not in str(first.url)
    assert (str(first.url), str(second.url)) == (
        "https://hub.test/user/acct/proxy/8790/samples?after=0&limit=1000",
        "https://hub.test/user/acct/proxy/8790/samples?after=3&limit=1000")


def test_polls_page_through_a_backlog_and_then_ask_for_only_the_new_samples():
    pod = FakePod(recorded()[:25])
    reader = reader_for(pod, rounds=4, limit=10)
    got = collect(reader)
    assert len(got) == 25
    assert [r.url.params["after"] for r in pod.requests] == ["0", "10", "20", "25"]
    assert reader.status()["gpu-a"]["fetched"] == 25


def test_a_sampler_restart_is_followed_from_its_first_sample():
    events = recorded()
    pod = FakePod(events[:48], started="2026-10-10T18:00:00.000Z")
    reader = reader_for(pod, rounds=1)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(pod)) as client:
            first = await reader.fetch(client, reader.pods[0])
            pod.held = list(enumerate(events[48:], 1))  # the new process numbers from 1 again
            pod.started = "2026-10-10T18:00:53.330Z"
            second = await reader.fetch(client, reader.pods[0])
            return first, second

    first, second = asyncio.run(run())
    assert (len(first), len(second)) == (48, 12)
    assert [e.event_id for e in second] == [e["event_id"] for e in events[48:]]
    assert (reader.status()["gpu-a"]["restarts"], reader.status()["gpu-a"]["after"]) == (1, 12)


def test_samples_the_pod_dropped_before_they_were_read_are_counted_as_lost(caplog):
    events = recorded()[:30]
    pod = FakePod(events[:10])
    reader = reader_for(pod, rounds=1)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(pod)) as client:
            await reader.fetch(client, reader.pods[0])
            pod.held = list(enumerate(events[15:], 16))  # samples 11 to 15 rolled out of the pod's buffer
            return await reader.fetch(client, reader.pods[0])

    with caplog.at_level(logging.WARNING, logger="atlas.telemetry.pods"):
        got = asyncio.run(run())
    assert [e.event_id for e in got] == [e["event_id"] for e in events[15:]]
    assert reader.status()["gpu-a"]["lost"] == 5
    assert "5 gpu-a samples were dropped on the pod before they could be read" in caplog.text


def test_only_this_pods_power_samples_pass_and_anything_else_is_counted_as_rejected():
    good = recorded()[0]
    other_node = {**recorded()[1], "node": "gpu-b"}
    not_power = {"event_id": "x", "type": "telemetry.request", "ts": good["ts"], "node": "gpu-a", "request_id": "r",
                 "started": good["ts"], "ended": good["ts"], "tokens_in": 1, "tokens_out": 1, "target": "teacher"}
    broken = {**recorded()[2], "socket_w": "forty"}
    pod = FakePod([other_node, not_power, broken, good])
    reader = reader_for(pod)
    got = collect(reader)
    assert [e.event_id for e in got] == [good["event_id"]]
    assert reader.status()["gpu-a"]["rejected"] == 3


def test_samples_already_in_the_log_are_not_published_again_after_the_app_restarts():
    events = recorded()[:10]
    newest_held = parse_event(events[6]).ts
    reader = reader_for(FakePod(events), skip_until=lambda node: newest_held if node == "gpu-a" else None)
    got = collect(reader)
    assert [e.event_id for e in got] == [e["event_id"] for e in events[7:]]
    assert reader.status()["gpu-a"]["repeats"] == 7


def test_a_pod_that_does_not_answer_is_logged_once_per_reason_and_read_again_when_it_comes_back(caplog):
    pod = FakePod(recorded()[:4])
    pod.fail_with = 502
    reader = reader_for(pod, rounds=1)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(pod)) as client:
            out = [await reader.fetch(client, reader.pods[0]) for _ in range(3)]
            pod.fail_with = httpx.ConnectError("refused")
            out.append(await reader.fetch(client, reader.pods[0]))
            pod.fail_with = None
            out.append(await reader.fetch(client, reader.pods[0]))
            return out

    with caplog.at_level(logging.INFO, logger="atlas.telemetry.pods"):
        results = asyncio.run(run())
    assert [len(r) for r in results] == [0, 0, 0, 0, 4]
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["the gpu-a sampler gave HTTP 502; asking again every 2 s",
                        "the gpu-a sampler gave no answer (ConnectError); asking again every 2 s",
                        "the gpu-a sampler answers again"]
    assert TOKEN not in caplog.text
    assert reader.status()["gpu-a"]["error"] is None


def test_one_pod_down_does_not_hold_up_the_other():
    a, b = FakePod([]), FakePod([{**e, "node": "gpu-b"} for e in recorded("samples_gpu-b_stale.jsonl")[:3]])
    a.fail_with = 502

    def route(request: httpx.Request) -> httpx.Response:
        return (a if "/acct-a/" in request.url.path else b)(request)

    reader = PodReader([PodEndpoint("gpu-a", "https://hub.test/user/acct-a/proxy/8790", TOKEN),
                        PodEndpoint("gpu-b", "https://hub.test/user/acct-b/proxy/8790", TOKEN)],
                       transport=httpx.MockTransport(route), clock=lambda: NOW, rounds=1)
    got = collect(reader)
    assert [e.node for e in got] == ["gpu-b"] * 3
    assert reader.status()["gpu-a"]["error"] == "HTTP 502"


def test_the_reader_takes_only_pods_each_once():
    with pytest.raises(ValueError, match="jetson is not a pod"):
        PodReader([PodEndpoint("jetson", "https://hub.test/x", TOKEN)])
    with pytest.raises(ValueError, match="each pod may appear once"):
        PodReader([PodEndpoint("gpu-a", "https://hub.test/x", TOKEN), PodEndpoint("gpu-a", "https://hub.test/y", TOKEN)])
    assert TOKEN not in repr(PodEndpoint("gpu-a", "https://hub.test/x", TOKEN))


# --- which pods, from the environment ---------------------------------------------------------------------------------

def test_the_pods_come_from_the_hub_account_keys_with_the_account_name_quoted_into_the_proxy_url():
    pods = pod_endpoints(ENV)
    assert [(p.node, p.base_url, p.token) for p in pods] == [
        ("gpu-a", "https://dub.aupcloud.io/user/acct%20a/proxy/8790", TOKEN),
        ("gpu-b", "https://dub.aupcloud.io/user/acct-b/proxy/8790", "other")]
    moved = pod_endpoints({**ENV, "ATLAS_HUB_URL": "https://hub.test/", "ATLAS_POD_POWER_PORT": "9001"})
    assert moved[1].base_url == "https://hub.test/user/acct-b/proxy/9001"


def test_one_account_is_enough_a_half_set_one_names_its_missing_key_and_none_at_all_is_refused():
    only_b = pod_endpoints({"AUP_HUB_USER_B": "acct-b", "AUP_HUB_TOKEN_B": "other"})
    assert [p.node for p in only_b] == ["gpu-b"]
    with pytest.raises(ValueError) as caught:
        pod_endpoints({"AUP_HUB_USER_A": "acct-a", "AUP_HUB_USER_B": "acct-b", "AUP_HUB_TOKEN_B": "other"})
    assert str(caught.value) == "set AUP_HUB_TOKEN_A too, or neither key of that account"
    with pytest.raises(ValueError, match="the pods source needs AUP_HUB_USER_A and AUP_HUB_TOKEN_A"):
        pod_endpoints({"AUP_HUB_TOKEN_B": " "})


# --- the pods source, end to end --------------------------------------------------------------------------------------

def test_the_pods_source_publishes_both_pods_samples_into_the_log_and_the_projection(tmp_path):
    a_events = recorded()[:40]
    b_events = recorded("samples_gpu-b_stale.jsonl")
    a, b = FakePod(a_events), FakePod(b_events)

    def route(request: httpx.Request) -> httpx.Response:
        return (a if "/acct%20a/" in request.url.raw_path.decode() else b)(request)

    async def run():
        bus = EventBus(tmp_path)
        service = build_telemetry("pods", bus, environ=ENV, transport=httpx.MockTransport(route))
        service.runner.sampler.rounds = 1
        await service.run()
        return service

    service = asyncio.run(run())
    logged = [json.loads(line) for path in sorted(tmp_path.glob("*.jsonl")) for line in path.read_text().splitlines()]
    assert [e["event_id"] for e in logged] == [e["event_id"] for e in a_events + b_events]
    assert all(e["received_ts"] is not None for e in logged)
    status = service.status()
    assert (status["source"], status["sampler"]["published"], status["sampler"]["nodes"]) == ("pods", 65,
                                                                                              ["gpu-a", "gpu-b"])
    assert status["sampler"]["own"]["gpu-b"]["fetched"] == 25
    nodes = service.projection.snapshot()["nodes"]
    assert nodes["gpu-b"]["sampler"]["stale"] is True
    assert nodes["gpu-b"]["sampler"]["stale_samples"] == 12
    assert nodes["gpu-b"]["sampler"]["counter_ok"] is False
    assert nodes["gpu-a"]["sampler"]["counter_ok"] is True


def test_a_restarted_app_rebuilds_from_the_log_and_does_not_publish_the_pods_buffer_twice(tmp_path):
    events = recorded()[:20]

    async def run_once():
        bus = EventBus(tmp_path)
        service = build_telemetry("pods", bus, environ={"AUP_HUB_USER_A": "acct", "AUP_HUB_TOKEN_A": TOKEN},
                                  transport=httpx.MockTransport(FakePod(events)))
        await bus.replay_into()  # as the app starts: the log rebuilds the projection before the reader runs
        service.runner.sampler.rounds = 1
        await service.run()
        return service

    asyncio.run(run_once())
    again = asyncio.run(run_once())
    logged = [line for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert len(logged) == 20
    assert again.runner.published == 0
    assert again.runner.sampler.status()["gpu-a"]["repeats"] == 20


# --- the projection's view of the pod sampler -------------------------------------------------------------------------

def fold(events: list[GpuPower]) -> TelemetryProjection:
    projection = TelemetryProjection()

    async def run():
        bus = EventBus(persist=False)
        projection.subscribe(bus)
        await bus.publish_all(events)

    asyncio.run(run())
    return projection


def test_the_projection_keeps_the_samplers_restarts_updates_staleness_and_energy_checks():
    events = [parse_event(e) for e in recorded()]
    sampler = fold(events).snapshot()["nodes"]["gpu-a"]["sampler"]
    checks = [(c["ts"], c["phase"], c["agrees"]) for c in sampler["energy_checks"]]
    assert checks == [(datetime(2026, 10, 10, 18, 0, 10, 260000, tzinfo=UTC), "start-up", True),
                      (datetime(2026, 10, 10, 18, 0, 33, 230000, tzinfo=UTC), "load", True),
                      (datetime(2026, 10, 10, 18, 1, 3, 690000, tzinfo=UTC), "start-up", True)]
    assert {k: sampler[k] for k in ("name", "restarts", "polls", "updates", "under_load", "stale", "stale_samples",
                                    "counter_ok")} == {
        "name": "lab/pod_power.py 1", "restarts": 1, "polls": 12, "updates": 12, "under_load": False, "stale": False,
        "stale_samples": 0, "counter_ok": True}
    assert sampler["started"] == datetime(2026, 10, 10, 18, 0, 53, 330000, tzinfo=UTC)


def test_a_late_pod_sample_does_not_overwrite_the_samplers_latest_state_or_count_a_restart():
    events = [parse_event(e) for e in recorded()[:12]]
    late = events[3]
    sampler = fold([*events, late.model_copy(update={"event_id": "late"})]).snapshot()["nodes"]["gpu-a"]["sampler"]
    assert (sampler["polls"], sampler["restarts"], len(sampler["energy_checks"])) == (12, 0, 1)


def test_samples_without_the_pod_samplers_fields_leave_no_sampler_section():
    plain = GpuPower(ts=NOW, node="gpu-a", source="amd-smi", period_ms=1000, socket_w=40.0, gfx_mhz=500,
                     mem_mhz=96, throttled=False)
    assert "sampler" not in fold([plain]).snapshot()["nodes"]["gpu-a"]


def test_the_newest_power_sample_per_node_is_what_the_reader_skips_up_to():
    events = [parse_event(e) for e in recorded()[:5]]
    projection = fold(events)
    assert projection.latest_power_ts("gpu-a") == events[-1].ts
    assert projection.latest_power_ts("gpu-b") is None


def test_an_energy_check_survives_the_log_as_its_own_model(tmp_path):
    closing = parse_event(recorded()[10])
    assert closing.energy_check == EnergyCheck(phase="start-up", window_s=10.06, samples=11, advanced=True,
                                               counter_delta_j=411.918, integrated_j=399.92, ratio=1.03,
                                               tolerance=0.1, agrees=True)

    async def run():
        await EventBus(tmp_path).publish(closing)

    asyncio.run(run())
    assert list(EventBus(tmp_path).read_log()) == [closing]
    with pytest.raises(ValueError, match="phase"):
        EnergyCheck(**{**closing.energy_check.model_dump(), "phase": "idle"})
