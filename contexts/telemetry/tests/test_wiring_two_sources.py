"""Two samplers in one app (tegrastats and the pods, ATLAS_TELEMETRY_SOURCE=tegrastats,pods), and joules per answer
computed and logged by the running service."""

import asyncio
import json
from pathlib import Path

import pytest

from bus import EventBus
from contexts.gateway.bootstrap import build
from contexts.telemetry.infrastructure.pods import PodReader
from contexts.telemetry.tests.test_jetson_wiring import a_jetson, jetson_env, run_app_until, settings
from contexts.telemetry.wiring import build_telemetry

JOULES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry" / "joules"
PODS_ENV = {"AUP_HUB_USER_A": "acct", "AUP_HUB_TOKEN_A": "t", "AUP_HUB_USER_B": "acct-b", "AUP_HUB_TOKEN_B": "t2"}


def test_two_sources_share_one_projection_and_each_keeps_its_own_runner():
    service = build_telemetry("tegrastats,pods", EventBus(persist=False), on_jetson=lambda: True,
                              env={**PODS_ENV, "ATLAS_TEGRASTATS": "tegrastats"})
    assert service.source == "tegrastats,pods"
    assert [sorted(r.nodes) for r in service.runners] == [["jetson"], ["pod-a", "pod-b"]]
    assert isinstance(service.runners[1].sampler, PodReader)
    assert set(service.status()["samplers"]) == {"tegrastats", "pods"}


@pytest.mark.parametrize("source, error", [
    ("tegrastats,nope", "unknown telemetry source 'nope'"),
    ("pods,pods", "names a source twice"),
    ("fake,pods", "the fake telemetry source runs alone"),
    (" , ", "no telemetry source named"),
])
def test_a_bad_source_list_is_refused_by_name(source, error):
    with pytest.raises(ValueError, match=error):
        build_telemetry(source, EventBus(persist=False), on_jetson=lambda: True, env=PODS_ENV)


def test_the_app_runs_tegrastats_and_the_pods_reader_side_by_side(tmp_path, monkeypatch):
    a_jetson(tmp_path, monkeypatch)
    jetson_env(monkeypatch)
    monkeypatch.setenv("ATLAS_TELEMETRY_SOURCE", "tegrastats,pods")
    for key, value in {**PODS_ENV, "AUP_HUB_URL_A": "http://127.0.0.1:9", "AUP_HUB_URL_B": "http://127.0.0.1:9",
                       "ATLAS_POD_POWER_POLL_S": "0.05"}.items():
        monkeypatch.setenv(key, value)
    atlas = build(settings(tmp_path))  # bootstrap passes the setting through: no second wiring block
    jetson, pods = atlas.telemetry.runners
    run_app_until(atlas, lambda: jetson.published >= 3 and pods.sampler.status()["pod-a"]["error"] is not None)
    assert jetson.published >= 3
    assert pods.sampler.status()["pod-a"]["error"].startswith("no answer")  # nothing listens there: it keeps asking


def test_the_running_service_logs_joules_per_answer_once_the_windows_settle(tmp_path):
    async def run():
        bus = EventBus(tmp_path)
        service = build_telemetry("fake", bus, fixture=JOULES / "conditions.jsonl", speed=0, loop=False,
                                  on_jetson=lambda: False)
        await service.run()
        return service

    service = asyncio.run(run())
    lines = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    values = [line for line in lines if line["type"] == "telemetry.joules_per_answer"]
    assert sorted(v["request_id"] for v in values) == sorted(
        ["jetson-clean", "jetson-modes", "jetson-oc", "jetson-short", "pod-throttled", "pod-train"])
    assert all(v["basis"] == "gross" and v["attribution"] == "token_proportional" for v in values)
    stats = service.stats()["nodes"]
    assert stats["jetson"]["answers"] == 4 and stats["pod-b"]["answers"] == 2

    async def again():  # a restart: the log rebuilds the values, and none is computed or logged a second time
        bus = EventBus(tmp_path)
        service = build_telemetry("fake", bus, fixture=JOULES / "conditions.jsonl", speed=0, loop=False,
                                  on_jetson=lambda: False)
        await bus.replay_into()
        return service, service.joules.ready()

    rebuilt, ready = asyncio.run(again())
    assert ready == [] and rebuilt.stats()["nodes"]["jetson"]["answers"] == 4
