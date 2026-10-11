"""What one answer on the box cost (G6): the Jetson's power sampler over the answer's window, gross, idle beside it.

Every sampler here is a fake: telemetry.jetson.power events folded by the real TelemetryProjection, or a generator of
the same readings around the wall clock for the route."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.answer_energy import AnswerMeter
from contexts.learning.lab_deploy import DeployRunner, lab_deploy_router
from contexts.telemetry.application.idle import IdleFigure
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.domain.power import PowerReading
from contracts.telemetry import JetsonPower

T0 = datetime(2026, 10, 11, 14, 0, tzinfo=UTC)
PERIOD = timedelta(milliseconds=350)
ANSWER = "<tool_call>\n<function=last_seen>\n<parameter=label>\nwater bottle\n</parameter>\n</function>\n</tool_call>"


def sample(t: datetime, watts: float, source: str = "tegrastats") -> JetsonPower:
    return JetsonPower(node="jetson", ts=t, source=source, period_ms=350, rails={"VDD_IN": watts * 1000},
                       nvpmodel="MAXN_SUPER")


def telemetry(projection: TelemetryProjection, idle: float | None = 15.76, source: str = "tegrastats"):
    """The parts of TelemetryService the meter reads: the projection, the source, the calibrated idle."""
    figures = {"jetson": IdleFigure(idle, source="config")} if idle is not None else {}
    return SimpleNamespace(projection=projection, source=source,
                           joules=SimpleNamespace(calibration=SimpleNamespace(idle=figures)))


def fed(start: datetime, end: datetime, watts) -> TelemetryProjection:
    projection = TelemetryProjection()
    t = start
    while t <= end:
        projection.on_event(sample(t, watts(t)))
        t += PERIOD
    return projection


def test_a_steady_20_w_over_2_s_is_40_j_gross_with_the_idle_beside_it():
    projection = fed(T0 - timedelta(seconds=3), T0 + timedelta(seconds=4), lambda t: 20.0)
    energy = asyncio.run(AnswerMeter(telemetry(projection)).measure(T0, T0 + timedelta(seconds=2)))
    assert energy == {"joules": 40.0, "idle_w": 15.76, "seconds": 2.0, "sampler": "tegrastats", "samples": 6,
                      "above_idle_j": 8.5, "low_confidence": True}
    json.dumps(energy)


def test_the_edges_are_clipped_by_interpolation_not_counted_whole():
    # 10 W until T0 + 1 s, then 30 W: a window from T0 + 0.5 s to T0 + 1.5 s is about 20 J, not 30 or 10
    projection = TelemetryProjection()
    for t, w in ((T0, 10.0), (T0 + timedelta(seconds=1), 10.0), (T0 + timedelta(seconds=1, milliseconds=1), 30.0),
                 (T0 + timedelta(seconds=2), 30.0)):
        projection.on_event(sample(t, w))
    energy = asyncio.run(AnswerMeter(telemetry(projection)).measure(T0 + timedelta(seconds=0.5),
                                                                  T0 + timedelta(seconds=1.5)))
    assert abs(energy["joules"] - 20.0) <= 0.1
    assert energy["samples"] == 2 and energy["low_confidence"] is True


def test_a_long_answer_has_enough_samples_to_be_confident():
    projection = fed(T0 - timedelta(seconds=1), T0 + timedelta(seconds=6), lambda t: 18.0)
    energy = asyncio.run(AnswerMeter(telemetry(projection)).measure(T0, T0 + timedelta(seconds=5)))
    assert energy["joules"] == 90.0 and energy["samples"] >= 10 and energy["low_confidence"] is False


def test_it_waits_for_the_first_sample_after_the_answer_ends():
    end = T0 + timedelta(seconds=1)
    projection = fed(T0 - timedelta(seconds=1), end - timedelta(milliseconds=100), lambda t: 20.0)
    waited = []

    async def sleep(s: float) -> None:  # the sampler's next line arrives while the meter waits
        waited.append(s)
        projection.on_event(sample(end + timedelta(milliseconds=250), 20.0))

    energy = asyncio.run(AnswerMeter(telemetry(projection), sleep=sleep).measure(T0, end))
    assert waited and energy["joules"] == 20.0


def test_no_sample_after_the_end_in_time_is_null():
    end = T0 + timedelta(seconds=1)
    projection = fed(T0 - timedelta(seconds=1), end - timedelta(milliseconds=100), lambda t: 20.0)
    clock = [0.0]

    async def sleep(s: float) -> None:
        clock[0] += s

    meter = AnswerMeter(telemetry(projection), sleep=sleep, monotonic=lambda: clock[0])
    assert asyncio.run(meter.measure(T0, end)) is None
    assert clock[0] >= meter.wait_s


def test_sampler_off_is_null():
    assert asyncio.run(AnswerMeter(None).measure(T0, T0 + timedelta(seconds=1))) is None
    empty = telemetry(TelemetryProjection())
    assert asyncio.run(AnswerMeter(empty).measure(T0, T0 + timedelta(seconds=1))) is None
    # samples that stopped a minute before the answer: the sampler is not running now
    old = fed(T0 - timedelta(seconds=70), T0 - timedelta(seconds=60), lambda t: 20.0)
    assert asyncio.run(AnswerMeter(telemetry(old), wait_s=0).measure(T0, T0 + timedelta(seconds=1))) is None


def test_only_the_boxs_own_node_counts():
    projection = TelemetryProjection()
    t = T0 - timedelta(seconds=1)
    while t <= T0 + timedelta(seconds=3):  # a pod's samples are someone else's watts
        projection.on_event(JetsonPower(node="pod-a", ts=t, source="tegrastats", period_ms=350,
                                        rails={"VDD_IN": 50_000}, nvpmodel=None))
        t += PERIOD
    assert asyncio.run(AnswerMeter(telemetry(projection), wait_s=0).measure(T0, T0 + timedelta(seconds=1))) is None


def test_without_an_idle_figure_the_joules_still_show():
    projection = fed(T0 - timedelta(seconds=1), T0 + timedelta(seconds=2), lambda t: 20.0)
    energy = asyncio.run(AnswerMeter(telemetry(projection, idle=None)).measure(T0, T0 + timedelta(seconds=1)))
    assert energy["joules"] == 20.0 and energy["idle_w"] is None and energy["above_idle_j"] is None


def test_a_fake_source_says_so():
    projection = fed(T0 - timedelta(seconds=1), T0 + timedelta(seconds=2), lambda t: 20.0)
    energy = asyncio.run(AnswerMeter(telemetry(projection, source="fake")).measure(T0, T0 + timedelta(seconds=1)))
    assert energy["sampler"] == "fake replay of tegrastats"


# --- the ask route ---

class LiveSampler:
    """A Jetson sampler on the wall clock: 22 W, one reading every 350 ms up to now, as the projection holds them."""

    def __init__(self, watts: float = 22.0) -> None:
        self.watts = watts
        self.origin = datetime.now(UTC) - timedelta(seconds=10)

    def readings(self, node, start=None, end=None, source=None):
        now = datetime.now(UTC)
        out, t = [], self.origin
        while t <= now:
            if node == "jetson" and (start is None or t >= start) and (end is None or t <= end):
                out.append(PowerReading(node="jetson", ts=t, watts=self.watts, source="tegrastats", boundary="module"))
            t += PERIOD
        return out


def llm(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/props":
        return httpx.Response(200, json={"model_path": "/data/models/atlas-base/model.gguf"})
    if request.url.path == "/completion":
        return httpx.Response(200, json={"tokens": [1, 2], "content": "ignored"})
    if request.url.path == "/detokenize":
        return httpx.Response(200, json={"content": ANSWER + "<|im_end|>"})
    return httpx.Response(404)


def client(tmp_path, meter) -> TestClient:
    app = FastAPI()
    runner = DeployRunner(tmp_path / "models", tmp_path / "state", llm_url="http://llm:8080")
    app.include_router(lab_deploy_router(runner, ask_transport=httpx.MockTransport(llm), meter=meter))
    return TestClient(app)


def ask(c: TestClient) -> dict:
    r = c.post("/api/learning/lab/deploy/ask", json={"question": "Have you seen my water bottle?"})
    assert r.status_code == 200
    return r.json()


def test_the_answer_carries_what_it_cost_on_the_box(tmp_path):
    live = SimpleNamespace(projection=LiveSampler(22.0), source="tegrastats",
                           joules=SimpleNamespace(calibration=SimpleNamespace(idle={"jetson": IdleFigure(15.76, "config")})))
    body = ask(client(tmp_path, AnswerMeter(live)))
    assert body["calls"] == [{"name": "last_seen", "arguments": {"label": "water bottle"}}]  # the answer is unchanged
    energy = body["energy"]
    assert set(energy) >= {"joules", "idle_w", "seconds", "sampler"}
    assert energy["sampler"] == "tegrastats" and energy["idle_w"] == 15.76
    assert abs(energy["joules"] - 22.0 * energy["seconds"]) <= 0.2  # gross: 22 W for the answer's seconds


def test_sampler_off_leaves_energy_null(tmp_path):
    assert ask(client(tmp_path, AnswerMeter(None)))["energy"] is None
    assert ask(client(tmp_path, None))["energy"] is None


def test_a_meter_that_breaks_never_costs_the_answer(tmp_path):
    class Broken:
        async def measure(self, start, end):
            raise RuntimeError("a bug")

    body = ask(client(tmp_path, Broken()))
    assert body["energy"] is None and body["calls"][0]["name"] == "last_seen"
