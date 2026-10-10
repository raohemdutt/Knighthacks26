"""Running tegrastats and nvpmodel as child processes. The children are the fake tools in
contexts/telemetry/fakes/jetson_tools.py, which replay the recorded Orin Nano output."""

import asyncio
import sys
from datetime import timezone
from pathlib import Path

import pytest

from bus import EventBus
from contexts.telemetry.application.jetson_sampler import JetsonPowerSampler
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.infrastructure.jetson_sysfs import JetsonSysfs
from contexts.telemetry.infrastructure.jetson_tools import NvpmodelQuery, TegrastatsProcess

FAKE = Path(__file__).resolve().parents[1] / "fakes" / "jetson_tools.py"


def fake(*args: str) -> list[str]:
    assert FAKE.exists(), f"expected the fake Jetson tools at {FAKE}"
    return [sys.executable, str(FAKE), *args]


def test_tegrastats_lines_arrive_stamped_in_utc_and_its_exit_is_an_error_with_the_code():
    tegrastats = TegrastatsProcess(fake("tegrastats", "--count", "3", "--exit-code", "3"))

    async def run():
        got, error = [], None
        try:
            async for ts, line in tegrastats.lines(20):
                got.append((ts, line))
        except RuntimeError as exc:
            error = exc
        return got, error

    got, error = asyncio.run(run())
    assert str(error) == "tegrastats exited with code 3"
    assert [line.split(" RAM ")[1][:20] for _, line in got] == ["5609/7620MB (lfb 9x1", "5988/7620MB (lfb 1x2",
                                                                 "6002/7620MB (lfb 1x4"]
    assert all(ts.tzinfo == timezone.utc for ts, _ in got) and got[0][0] < got[1][0] < got[2][0]
    assert tegrastats.pid is None and tegrastats.process.returncode == 3


def test_tegrastats_is_started_with_the_interval_asked_for():
    tegrastats = TegrastatsProcess(fake("tegrastats", "--count", "3"))

    async def run():
        stamps = []
        with pytest.raises(RuntimeError):
            async for ts, _ in tegrastats.lines(150):
                stamps.append(ts)
        return [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]

    gaps = asyncio.run(run())
    assert len(gaps) == 2 and all(0.12 < gap < 1.0 for gap in gaps)  # paced by --interval 150, not flooding


def test_stopping_early_terminates_tegrastats():
    tegrastats = TegrastatsProcess(fake("tegrastats"))  # endless

    async def run():
        lines = tegrastats.lines(20)
        await lines.__anext__()
        running = tegrastats.process.returncode is None
        await lines.aclose()
        return running

    assert asyncio.run(run()) is True
    assert tegrastats.process.returncode is not None and tegrastats.pid is None


def test_cancelling_the_sampler_runner_terminates_tegrastats(tmp_path):
    tegrastats = TegrastatsProcess(fake("tegrastats"))
    sampler = JetsonPowerSampler(tegrastats, JetsonSysfs(tmp_path), NvpmodelQuery(fake("nvpmodel")), interval_ms=20,
                                 nvpmodel_every_s=0)

    async def run():
        runner = SamplerRunner(sampler, EventBus(tmp_path / "events").publish, nodes=["jetson"])
        task = asyncio.create_task(runner.run())
        while runner.published < 3:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return tegrastats.process.returncode

    assert asyncio.run(run()) is not None
    assert len(list(EventBus(tmp_path / "events").read_log())) >= 3


def test_a_tegrastats_that_dies_is_started_again_by_the_runner(tmp_path):
    tegrastats = TegrastatsProcess(fake("tegrastats", "--count", "2", "--exit-code", "1"))
    sampler = JetsonPowerSampler(tegrastats, JetsonSysfs(tmp_path), NvpmodelQuery(fake("nvpmodel")), interval_ms=20,
                                 nvpmodel_every_s=0)

    async def run():
        runner = SamplerRunner(sampler, EventBus(tmp_path / "events").publish, nodes=["jetson"], retry_s=0)
        task = asyncio.create_task(runner.run())
        while runner.published < 6:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return runner

    runner = asyncio.run(run())
    assert runner.errors >= 2 and runner.last_error == "RuntimeError: tegrastats exited with code 1"
    logged = list(EventBus(tmp_path / "events").read_log())
    assert len(logged) == runner.published >= 6
    assert {(e.type, e.node, e.source, e.nvpmodel) for e in logged} == {
        ("telemetry.jetson.power", "jetson", "tegrastats", "MAXN_SUPER")}


def test_nvpmodel_q_gives_the_mode_name():
    assert asyncio.run(NvpmodelQuery(fake("nvpmodel"))()) == "MAXN_SUPER"


def test_nvpmodel_that_cannot_run_or_names_no_mode_gives_none_and_counts_the_failure(tmp_path):
    missing = NvpmodelQuery([str(tmp_path / "no-such-nvpmodel")])
    silent = tmp_path / "silent.txt"
    silent.write_text("# nothing\n")
    mute = NvpmodelQuery(fake("nvpmodel", "--file", str(silent)))
    assert (asyncio.run(missing()), asyncio.run(mute())) == (None, None)
    assert (missing.failures, mute.failures) == (1, 1)
