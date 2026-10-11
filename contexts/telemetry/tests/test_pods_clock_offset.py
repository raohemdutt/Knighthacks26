"""The pod's clock against the reader's, from the four timestamps of each poll through the hub proxy.

The proxy is an httpx MockTransport answering as lab/pod_power.py does, with its pod_received and pod_sent stamped
on a pod clock that runs 65 s ahead, and a scripted delay on the way out and back."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from contexts.telemetry.infrastructure.clock import best_offset, exchange_of, measure
from contexts.telemetry.infrastructure.pods import PodEndpoint, PodReader
from contracts.telemetry import ClockOffset, GpuPower

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry" / "pod_power"
T0 = datetime(2026, 10, 10, 20, 0, tzinfo=UTC)
AHEAD = timedelta(seconds=65)


def iso(t: datetime) -> str:
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Clocks:
    """This machine's clock, advanced by the fake proxy as the request travels; the pod's is AHEAD of it."""

    def __init__(self, delays: list[tuple[float, float]]) -> None:
        self.now = T0
        self.delays = list(delays)  # (out, back) seconds per request

    def local(self) -> datetime:
        return self.now

    def answer(self, request: httpx.Request, events: list[dict]) -> httpx.Response:
        out, back = self.delays.pop(0) if self.delays else (0.05, 0.05)
        self.now += timedelta(seconds=out)
        received = self.now + AHEAD
        sent = received + timedelta(milliseconds=2)
        self.now += timedelta(milliseconds=2) + timedelta(seconds=back)
        after = int(request.url.params["after"])
        page = [{"seq": n, "event": e} for n, e in enumerate(events, 1) if n > after]
        return httpx.Response(200, json={"ok": True, "node": "gpu-a", "sampler_started": "2026-10-10T19:00:00.000Z",
                                         "first_seq": 1, "last_seq": len(events), "samples": page,
                                         "pod_received": iso(received), "pod_sent": iso(sent)})


def recorded(count: int) -> list[dict]:
    lines = (FIXTURES / "samples_gpu-a.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[:count]]


def run_reader(clocks: Clocks, events: list[dict], rounds: int, **kw) -> tuple[PodReader, list]:
    reader = PodReader([PodEndpoint("pod-a", "https://hub.test/user/acct/proxy/8790", "t")], rounds=rounds,
                       transport=httpx.MockTransport(lambda r: clocks.answer(r, events)), clock=clocks.local,
                       sleep=lambda _: asyncio.sleep(0), **kw)

    async def go():
        return [e async for e in reader.samples()]

    return reader, asyncio.run(go())


def test_the_first_poll_reports_the_pods_offset_before_its_samples():
    clocks = Clocks([(0.08, 0.02)])
    reader, got = run_reader(clocks, recorded(3), rounds=1)
    assert isinstance(got[0], ClockOffset) and all(isinstance(e, GpuPower) for e in got[1:])
    offset = got[0]
    assert (offset.node, offset.reference, offset.via) == ("pod-a", "jetson", "hub proxy /samples")
    assert offset.round_trip_s == pytest.approx(0.10)
    assert offset.error_s == pytest.approx(0.05)
    assert abs(offset.offset_s - 65.0) <= offset.error_s
    assert offset.offset_s == pytest.approx(65.0 + (0.08 - 0.02) / 2)
    assert reader.status()["pod-a"]["clock_offset"]["offset_s"] == offset.offset_s


def test_the_next_offset_comes_after_offset_every_polls_from_the_exchange_with_the_smallest_round_trip():
    delays = [(0.3, 0.3), (0.2, 0.25), (0.01, 0.02), (0.4, 0.1), (0.2, 0.2), (0.05, 0.05)]
    clocks = Clocks(delays)
    _, got = run_reader(clocks, recorded(2), rounds=6, offset_every=3)
    offsets = [e for e in got if isinstance(e, ClockOffset)]
    assert len(offsets) == 2  # poll 1, then poll 4
    assert offsets[0].round_trip_s == pytest.approx(0.6)  # only one exchange so far
    assert offsets[1].round_trip_s == pytest.approx(0.03) and offsets[1].exchanges == 4  # the best of four
    assert abs(offsets[1].offset_s - 65.0) <= offsets[1].error_s


def test_an_answer_without_pod_timestamps_or_running_backwards_is_no_exchange():
    t1, t4 = T0, T0 + timedelta(seconds=0.1)
    assert exchange_of(t1, {"samples": []}, t4) is None
    assert exchange_of(t1, {"pod_received": iso(T0 + AHEAD), "pod_sent": iso(T0 + AHEAD - timedelta(seconds=1))},
                       t4) is None
    with pytest.raises(ValueError):
        best_offset("pod-a", [])


def test_the_one_off_measurement_reads_the_account_keys_and_sends_the_token_only_in_the_header():
    clocks = Clocks([(0.1, 0.1)] * 5)
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return clocks.answer(request, [])

    env = {"AUP_HUB_USER_C": "roscon01", "AUP_HUB_TOKEN_C": "secret-token", "AUP_HUB_URL_C": "https://sjc.test"}
    import contexts.telemetry.infrastructure.clock as clock_module

    real_now = clock_module.datetime
    try:
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return clocks.local()

        clock_module.datetime = FakeDatetime
        event = measure("C", count=5, env=env, transport=httpx.MockTransport(answer))
    finally:
        clock_module.datetime = real_now
    assert event.node == "pod-a" and event.exchanges == 5  # the sampler says gpu-a, its old name
    assert abs(event.offset_s - 65.0) <= event.error_s
    assert all(r.headers["Authorization"] == "token secret-token" and "secret" not in str(r.url) for r in seen)
    assert str(seen[0].url).startswith("https://sjc.test/user/roscon01/proxy/8790/samples?")
