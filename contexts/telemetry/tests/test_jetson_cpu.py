"""Measuring the Jetson power sampler's CPU cost, and the box-side command that runs the first live steps.
The snapshots and the CPU numbers here are made up; the sampler runs over the fake tools replaying recorded
Orin Nano output."""

import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest

from contexts.telemetry import jetson_power
from contexts.telemetry.application.jetson_cpu import CpuSnapshot, cpu_report, measure_sampler_cpu
from contexts.telemetry.application.jetson_sampler import JetsonPowerSampler
from contexts.telemetry.infrastructure.jetson_sysfs import JetsonSysfs
from contexts.telemetry.infrastructure.jetson_tools import NvpmodelQuery, TegrastatsProcess

FAKE = Path(__file__).resolve().parents[1] / "fakes" / "jetson_tools.py"


def fake(*args: str) -> list[str]:
    assert FAKE.exists(), f"expected the fake Jetson tools at {FAKE}"
    return [sys.executable, str(FAKE), *args]


def snap(wall: float, busy: float, own: float) -> CpuSnapshot:
    return CpuSnapshot(wall_s=wall, busy_s=busy, own_s=own)


def test_the_cost_is_the_box_busy_cpu_with_the_sampler_on_minus_off_as_percent_of_one_core():
    off = (snap(0, 1000.0, 1.0), snap(600, 1060.0, 1.0))  # 10 percent of a core busy without the sampler
    on = (snap(600, 1060.0, 1.0), snap(1200, 1123.0, 4.6))  # 10.5 percent with it; it used 3.6 s itself
    assert cpu_report(off, on, interval_ms=100, samples=6000) == {
        "metric": "telemetry.sampler_cpu_pct", "node": "jetson", "value": 0.5, "unit": "percent of one core",
        "method": "busy CPU of the whole box (/proc/stat), sampler on minus sampler off",
        "own_pct": 0.6, "system_off_pct": 10.0, "system_on_pct": 10.5, "off_s": 600.0, "on_s": 600.0,
        "interval_ms": 100, "samples": 6000, "limit_pct": 1.0, "within_limit": True, "suggested_interval_ms": None}


def test_over_the_budget_the_report_says_so_and_suggests_a_period_that_would_fit():
    off = (snap(0, 1000.0, 0.0), snap(600, 1060.0, 0.0))
    on = (snap(600, 1060.0, 0.0), snap(1200, 1135.0, 12.0))  # 2.5 percent of a core on the box, 2.0 its own
    report = cpu_report(off, on, interval_ms=100, samples=6000)
    assert (report["value"], report["own_pct"], report["within_limit"], report["suggested_interval_ms"]) == (
        2.5, 2.0, False, 250)


def test_its_own_cpu_time_over_the_budget_fails_it_even_when_the_box_looks_quiet():
    off = (snap(0, 1000.0, 0.0), snap(600, 1070.0, 0.0))  # the box was busier while off: the difference is noise
    on = (snap(600, 1070.0, 0.0), snap(1200, 1130.0, 9.0))
    report = cpu_report(off, on, interval_ms=100, samples=6000)
    assert (report["value"], report["own_pct"], report["within_limit"], report["suggested_interval_ms"]) == (
        -1.667, 1.5, False, 150)


def test_a_stretch_of_no_time_is_refused():
    with pytest.raises(ValueError, match="longer than zero seconds"):
        cpu_report((snap(5, 0, 0), snap(5, 0, 0)), (snap(5, 0, 0), snap(6, 0, 0)), interval_ms=100, samples=1)


def test_the_measurement_runs_off_then_on_and_publishes_every_sample_into_a_throwaway_log(tmp_path):
    readings = iter([snap(0, 100.0, 0.0), snap(1, 100.1, 0.0), snap(1, 100.1, 0.0), snap(2, 100.205, 0.004)])
    pids = []

    def snapshot(pid):
        pids.append(pid)
        return next(readings)

    def make_sampler():
        return JetsonPowerSampler(TegrastatsProcess(fake("tegrastats")), JetsonSysfs(tmp_path / "sysfs"),
                                  NvpmodelQuery(fake("nvpmodel")), interval_ms=20, nvpmodel_every_s=0)

    async def wait(seconds):
        await asyncio.sleep(0.5)  # long enough for the fake tegrastats to print a few lines

    report = asyncio.run(measure_sampler_cpu(make_sampler, snapshot, seconds=1, log_dir=tmp_path / "log", sleep=wait))
    logged = [json.loads(line) for path in (tmp_path / "log").glob("*.jsonl") for line in path.read_text().splitlines()]
    assert pids[:3] == [None, None, None] and isinstance(pids[3], int)  # tegrastats was running at the end
    assert report["samples"] == len(logged) == report["sampler"]["published"] >= 3
    assert {line["type"] for line in logged} == {"telemetry.jetson.power"}
    assert (report["metric"], report["value"], report["own_pct"], report["interval_ms"], report["within_limit"]) == (
        "telemetry.sampler_cpu_pct", 0.5, 0.4, 20, True)


def test_a_sampler_that_publishes_nothing_is_not_reported_as_cheap(tmp_path):
    readings = iter([snap(0, 0, 0), snap(1, 0, 0), snap(1, 0, 0), snap(2, 0, 0)])

    def make_sampler():
        return JetsonPowerSampler(TegrastatsProcess([str(tmp_path / "no-such-tegrastats")]), JetsonSysfs(tmp_path),
                                  NvpmodelQuery([str(tmp_path / "no-such-nvpmodel")]), interval_ms=20,
                                  nvpmodel_every_s=0)

    async def wait(seconds):
        await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError, match="published nothing"):
        asyncio.run(measure_sampler_cpu(make_sampler, lambda pid: next(readings), seconds=1, log_dir=tmp_path / "log",
                                        sleep=wait))


@pytest.mark.parametrize("argv", [["probe"], ["cpu", "--minutes", "0.01"]])
def test_the_box_side_command_refuses_to_run_off_a_jetson(argv, capsys):
    assert jetson_power.main(argv, on_jetson=lambda: False) == 2
    assert "refusing: this is not a Jetson" in capsys.readouterr().err


def test_probe_prints_each_sample_and_a_summary_and_writes_no_log(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_TEGRASTATS", shlex.join([Path(sys.executable).as_posix(), FAKE.as_posix(), "tegrastats"]))
    monkeypatch.setenv("ATLAS_NVPMODEL", shlex.join([Path(sys.executable).as_posix(), FAKE.as_posix(), "nvpmodel"]))
    assert jetson_power.main(["probe", "--count", "4", "--interval-ms", "20"], on_jetson=lambda: True) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    samples, summary = out[:4], out[4]
    assert [s["rails"]["VDD_IN"] for s in samples] == [7732.0, 17730.0, 19955.0, 19955.0]
    assert {(s["type"], s["period_ms"], s["nvpmodel"]) for s in samples} == {("telemetry.jetson.power", 20.0, "MAXN_SUPER")}
    assert (summary["summary"], summary["samples"], summary["period_ms"], summary["nvpmodel"]) == ("probe", 4, 20, "MAXN_SUPER")
    assert summary["measured_period_ms"] > 0 and len(out) == 5
    assert not (tmp_path / "data").exists()


def test_cpu_refuses_while_another_tegrastats_runs(tmp_path, capsys):
    comm = tmp_path / "proc" / "812" / "comm"
    comm.parent.mkdir(parents=True)
    comm.write_text("tegrastats\n")
    (tmp_path / "proc" / "813").mkdir()
    (tmp_path / "proc" / "813" / "comm").write_text("python3\n")
    assert jetson_power.main(["cpu", "--minutes", "0.01"], on_jetson=lambda: True, sysfs=JetsonSysfs(tmp_path)) == 2
    assert "tegrastats is already running (pid 812)" in capsys.readouterr().err
