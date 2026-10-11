"""The tegrastats source as the gateway runs it: only on a Jetson, settings from the environment, samples in the
app's log and projection, and sampling again after the app restarts (what the atlas-gateway systemd unit does).
tegrastats and nvpmodel are the fake tools replaying recorded Orin Nano output."""

import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest

from bus import EventBus
from contexts.gateway.bootstrap import Settings, build
from contexts.telemetry.infrastructure import host
from contexts.telemetry.wiring import build_telemetry, jetson_power_sampler

FAKE = Path(__file__).resolve().parents[1] / "fakes" / "jetson_tools.py"


def fake_command(tool: str, *args: str) -> str:
    assert FAKE.exists(), f"expected the fake Jetson tools at {FAKE}"
    return shlex.join([Path(sys.executable).as_posix(), FAKE.as_posix(), tool, *args])


def jetson_power_lines(log_dir: Path) -> list[dict]:
    lines = [json.loads(line) for path in sorted(log_dir.glob("*.jsonl")) for line in path.read_text().splitlines()]
    return [line for line in lines if line["type"] == "telemetry.jetson.power"]


def a_jetson(tmp_path: Path, monkeypatch) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text("# R36 (release), REVISION: 4.4\n")
    monkeypatch.setattr(host, "TEGRA_RELEASE", release)


def jetson_env(monkeypatch, interval_ms: str = "20") -> None:
    monkeypatch.setenv("ATLAS_TELEMETRY_SOURCE", "tegrastats")
    monkeypatch.setenv("ATLAS_TELEMETRY_JETSON_INTERVAL_MS", interval_ms)
    monkeypatch.setenv("ATLAS_TEGRASTATS", fake_command("tegrastats"))
    monkeypatch.setenv("ATLAS_NVPMODEL", fake_command("nvpmodel"))


def run_app_until(atlas, until, timeout_s: float = 10.0) -> None:
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


def test_the_tegrastats_source_refuses_to_run_anywhere_but_a_jetson():
    with pytest.raises(RuntimeError, match="runs only on a Jetson"):
        build_telemetry("tegrastats", EventBus(persist=False), on_jetson=lambda: False, env={})


def test_the_app_will_not_start_the_tegrastats_source_off_a_jetson(tmp_path, monkeypatch):
    jetson_env(monkeypatch)
    monkeypatch.setattr(host, "TEGRA_RELEASE", tmp_path / "not-a-jetson")
    with pytest.raises(RuntimeError, match="runs only on a Jetson"):
        build(settings(tmp_path))


def test_on_a_jetson_the_running_app_samples_power_into_its_log_and_projection(tmp_path, monkeypatch):
    a_jetson(tmp_path, monkeypatch)
    jetson_env(monkeypatch)
    atlas = build(settings(tmp_path))
    assert atlas.telemetry is not None and atlas.telemetry.source == "tegrastats"
    run_app_until(atlas, lambda: atlas.telemetry.runner.published >= 5)
    lines = jetson_power_lines(tmp_path / "events")
    assert len(lines) == atlas.telemetry.runner.published >= 5
    assert {(line["node"], line["source"], line["period_ms"], line["nvpmodel"]) for line in lines} == {
        ("jetson", "tegrastats", 20.0, "MAXN_SUPER")}
    assert lines[0]["rails"] == {"VDD_IN": 7732.0, "VDD_CPU_GPU_CV": 2120.0, "VDD_SOC": 2524.0}
    power = atlas.telemetry.projection.snapshot()["nodes"]["jetson"]["power"]["tegrastats"]
    assert power["samples"] == len(lines) and power["period_ms"] == 20.0
    assert atlas.telemetry.runner.sampler.tegrastats.process.returncode is not None  # stop() ended tegrastats


def test_after_a_restart_the_app_samples_again_and_rebuilds_the_projection_from_the_log(tmp_path, monkeypatch):
    a_jetson(tmp_path, monkeypatch)
    jetson_env(monkeypatch)
    first = build(settings(tmp_path))
    run_app_until(first, lambda: first.telemetry.runner.published >= 3)
    before = len(jetson_power_lines(tmp_path / "events"))

    again = build(settings(tmp_path))  # what systemd's restart of atlas-gateway amounts to: a new process, same box
    run_app_until(again, lambda: again.telemetry.runner.published >= 3)
    after = jetson_power_lines(tmp_path / "events")
    power = again.telemetry.projection.snapshot()["nodes"]["jetson"]["power"]["tegrastats"]
    assert before >= 3 and len(after) == before + again.telemetry.runner.published
    assert power["samples"] == len(after)  # the replayed samples and the new ones


def test_the_sampler_settings_come_from_the_environment_with_350_ms_by_default():
    sampler = jetson_power_sampler({})
    assert (sampler.interval_ms, sampler.tegrastats.command, sampler.nvpmodel.command) == (
        350, ["tegrastats"], ["/usr/sbin/nvpmodel"])
    sampler = jetson_power_sampler({"ATLAS_TELEMETRY_JETSON_INTERVAL_MS": "250",
                                    "ATLAS_TEGRASTATS": "/usr/bin/tegrastats --verbose",
                                    "ATLAS_NVPMODEL": "/opt/nvpmodel"})
    assert (sampler.interval_ms, sampler.tegrastats.command, sampler.nvpmodel.command) == (
        250, ["/usr/bin/tegrastats", "--verbose"], ["/opt/nvpmodel"])
    assert jetson_power_sampler({"ATLAS_TELEMETRY_JETSON_INTERVAL_MS": "250"}, interval_ms=500).interval_ms == 500


@pytest.mark.parametrize("value, error", [("fast", "must be a whole number of ms, not 'fast'"),
                                          ("0", "at least 1 ms")])
def test_a_bad_interval_setting_is_refused_by_name(value, error):
    with pytest.raises(ValueError, match=error):
        jetson_power_sampler({"ATLAS_TELEMETRY_JETSON_INTERVAL_MS": value})
