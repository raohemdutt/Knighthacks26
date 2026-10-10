"""Building telemetry from one setting, and the gateway's wiring: off unless ATLAS_TELEMETRY_SOURCE is set."""

import asyncio
import json

import pytest

from bus import EventBus
from contexts.gateway.bootstrap import Settings, build
from contexts.telemetry.infrastructure import host
from contexts.telemetry.wiring import build_telemetry


def telemetry_lines(log_dir) -> list[dict]:
    lines = [json.loads(line) for path in sorted(log_dir.glob("*.jsonl")) for line in path.read_text().splitlines()]
    return [line for line in lines if line["type"].startswith("telemetry.")]


def test_the_fake_source_publishes_the_recorded_minute_into_the_log_and_the_projection(tmp_path):
    async def run():
        bus = EventBus(tmp_path)
        service = build_telemetry("fake", bus, speed=0, loop=False, on_jetson=lambda: False)
        await service.run()
        return service

    service = asyncio.run(run())
    assert len(telemetry_lines(tmp_path)) == 206
    assert sum(service.projection.counts.values()) == 206
    assert service.status() == {"source": "fake", "sampler": {"nodes": ["gpu-a", "gpu-b", "jetson"],
                                                              "published": 206, "dropped": 0, "errors": 0,
                                                              "last_error": None}}


def test_an_unknown_source_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown telemetry source 'tegrastat'; known sources: fake, tegrastats$"):
        build_telemetry("tegrastat", EventBus(persist=False), on_jetson=lambda: False)


def test_the_fake_source_refuses_to_run_on_a_jetson():
    with pytest.raises(RuntimeError, match="does not run on a Jetson"):
        build_telemetry("fake", EventBus(persist=False), on_jetson=lambda: True)


def test_the_jetson_check_looks_for_the_l4t_release_file(tmp_path):
    release = tmp_path / "nv_tegra_release"
    assert not host.on_jetson(release)
    release.write_text("# R36 (release), REVISION: 4.4\n")
    assert host.on_jetson(release)


def run_app_briefly(atlas, until=lambda: True, timeout_s: float = 5.0) -> None:
    async def run():
        await atlas.start()
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_s
            while not until() and loop.time() < deadline:
                await asyncio.sleep(0.01)
        finally:
            await atlas.stop()

    asyncio.run(run())


def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, connectors=False, ntfy_url=None, llm_url=None)


def test_the_app_has_no_telemetry_and_writes_none_unless_a_source_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_TELEMETRY_SOURCE", raising=False)
    atlas = build(settings(tmp_path))
    assert atlas.telemetry is None
    run_app_briefly(atlas)
    assert telemetry_lines(tmp_path / "events") == []


def test_with_the_fake_source_the_running_app_samples_into_its_log_and_projection(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_TELEMETRY_SOURCE", "fake")
    monkeypatch.setattr(host, "TEGRA_RELEASE", tmp_path / "not-a-jetson")
    atlas = build(settings(tmp_path))
    assert atlas.telemetry is not None and atlas.telemetry.source == "fake"
    run_app_briefly(atlas, until=lambda: atlas.telemetry.projection.counts)
    lines = telemetry_lines(tmp_path / "events")
    assert lines and sum(atlas.telemetry.projection.counts.values()) == len(lines)
    assert {line["node"] for line in lines} <= {"jetson", "gpu-a", "gpu-b"}
    assert all(task.done() for task in atlas.tasks)  # stop() cancelled the sampler with the ticks


def test_the_app_will_not_start_the_fake_source_on_a_jetson(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_TELEMETRY_SOURCE", "fake")
    release = tmp_path / "nv_tegra_release"
    release.write_text("# R36 (release), REVISION: 4.4\n")
    monkeypatch.setattr(host, "TEGRA_RELEASE", release)
    with pytest.raises(RuntimeError, match="does not run on a Jetson"):
        build(settings(tmp_path))
