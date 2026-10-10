"""The Jetson power sampler on recorded tegrastats lines: what each sample carries, the period it samples at, and what
it does with lines that are not samples. tegrastats and nvpmodel are replaced by scripted stand-ins; the sysfs
reader is the real one over a directory tree laid out like the Jetson's."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bus import EventBus
from contexts.telemetry.application.jetson_sampler import DEFAULT_INTERVAL_MS, JetsonPowerSampler, sampling_period_ms
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.infrastructure.jetson_sysfs import JetsonSysfs
from contracts.telemetry import JetsonPower

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry"
T0 = datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)
INA = "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1"
TEMPS = {"cpu": 51.562, "soc2": 51.093, "soc0": 50.0, "gpu": 50.875, "tj": 51.562, "soc1": 51.093}


def recorded_lines() -> list[str]:
    path = FIXTURES / "jetson_tegrastats_orin_nano.txt"
    assert path.exists(), f"expected the recorded fixture {path}"
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


LINES = recorded_lines()


def write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def box(root: Path, samples: str = "4", update: str = "26") -> JetsonSysfs:
    """A sysfs tree with an INA3221 (made-up settings) and the recorded overcurrent counters."""
    write(root, f"{INA}/in1_label", "VDD_IN")
    write(root, f"{INA}/samples", samples)
    write(root, f"{INA}/update_interval", update)
    for n, count in (("1", "0"), ("2", "0"), ("3", "194")):
        write(root, f"sys/class/hwmon/hwmon3/oc{n}_event_cnt", count)
        write(root, f"sys/class/hwmon/hwmon3/oc{n}_throt_en", "1")
    return JetsonSysfs(root)


class ScriptedTegrastats:
    """Stands in for the tegrastats process: yields (arrival time, line) pairs, one script per start.

    A callable in a script runs when reached (to change the box between lines); an Exception is raised."""

    def __init__(self, *scripts: list) -> None:
        self.scripts = list(scripts)
        self.intervals: list[int] = []
        self.pid: int | None = None
        self.closed = 0

    async def lines(self, interval_ms: int):
        self.intervals.append(interval_ms)
        script = self.scripts.pop(0) if self.scripts else []
        self.pid = 4242
        try:
            for item in script:
                if isinstance(item, Exception):
                    raise item
                if callable(item):
                    item()
                    continue
                yield item
                await asyncio.sleep(0)
        finally:
            self.pid = None
            self.closed += 1


def every(lines: list[str], start: datetime = T0, step_ms: float = 100) -> list[tuple[datetime, str]]:
    return [(start + timedelta(milliseconds=step_ms * n), line) for n, line in enumerate(lines)]


def nvpmodel(*modes):
    answers = list(modes)

    async def query():
        return answers.pop(0) if len(answers) > 1 else answers[0]

    return query


async def collect(sampler: JetsonPowerSampler, limit: int | None = None) -> list[JetsonPower]:
    out = []
    async for event in sampler.samples():
        out.append(event)
        if limit is not None and len(out) >= limit:
            break
    return out


def test_each_line_becomes_one_sample_with_every_field_the_done_when_names(tmp_path):
    tegrastats = ScriptedTegrastats(every(LINES[:2]))
    sampler = JetsonPowerSampler(tegrastats, box(tmp_path), nvpmodel("MAXN_SUPER"), nvpmodel_every_s=0)
    events = asyncio.run(collect(sampler))
    common = dict(node="jetson", source="tegrastats", period_ms=100.0, nvpmodel="MAXN_SUPER",
                  oc_events={"oc1": 0, "oc2": 0, "oc3": 194}, sensor_avg_samples=4, sensor_update_ms=26.0)
    assert [e.model_dump(exclude={"event_id"}) for e in events] == [
        JetsonPower(ts=T0, rails={"VDD_IN": 7732.0, "VDD_CPU_GPU_CV": 2120.0, "VDD_SOC": 2524.0}, temps_c=TEMPS,
                    **common).model_dump(exclude={"event_id"}),
        JetsonPower(ts=T0 + timedelta(milliseconds=100), rails={"VDD_IN": 17730.0, "VDD_CPU_GPU_CV": 9352.0,
                                                                 "VDD_SOC": 3711.0},
                    temps_c={"cpu": 52.031, "soc2": 52.031, "soc0": 50.5, "gpu": 54.437, "tj": 54.437,
                             "soc1": 51.125}, **common).model_dump(exclude={"event_id"}),
    ]
    assert len({e.event_id for e in events}) == 2
    assert tegrastats.intervals == [100] and tegrastats.closed == 1


def test_the_default_interval_is_100_ms_and_tegrastats_is_started_at_it(tmp_path):
    tegrastats = ScriptedTegrastats(every(LINES[:1]))
    sampler = JetsonPowerSampler(tegrastats, box(tmp_path, update="26"), nvpmodel("15W"), nvpmodel_every_s=0)
    asyncio.run(collect(sampler))
    assert (DEFAULT_INTERVAL_MS, sampler.interval_ms, tegrastats.intervals) == (100, 100, [100])


def test_the_sampler_never_polls_faster_than_the_ina3221_updates(tmp_path):
    tegrastats = ScriptedTegrastats(every(LINES[:2], step_ms=140))
    sampler = JetsonPowerSampler(tegrastats, box(tmp_path, samples="16", update="140"), nvpmodel("MAXN_SUPER"),
                                 interval_ms=100, nvpmodel_every_s=0)
    events = asyncio.run(collect(sampler))
    assert tegrastats.intervals == [140]
    assert [(e.period_ms, e.sensor_update_ms, e.sensor_avg_samples) for e in events] == [(140.0, 140.0, 16)] * 2


@pytest.mark.parametrize("requested, update, period", [(100, None, 100), (100, 26.0, 100), (100, 100.0, 100),
                                                       (100, 140.0, 140), (100, 6.6, 100), (100, 100.4, 101),
                                                       (1000, 140.0, 1000)])
def test_the_sampling_period_is_the_requested_interval_raised_to_the_sensor_update(requested, update, period):
    assert sampling_period_ms(requested, update) == period


@pytest.mark.parametrize("interval", [0, -5])
def test_an_interval_under_1_ms_is_refused_before_anything_starts(tmp_path, interval):
    with pytest.raises(ValueError, match="at least 1 ms"):
        JetsonPowerSampler(ScriptedTegrastats(), box(tmp_path), nvpmodel("15W"), interval_ms=interval)


def test_without_a_power_monitor_the_sample_says_so_with_nulls_and_keeps_the_requested_period(tmp_path):
    sampler = JetsonPowerSampler(ScriptedTegrastats(every(LINES[:1])), JetsonSysfs(tmp_path), nvpmodel("MAXN_SUPER"),
                                 nvpmodel_every_s=0)
    [event] = asyncio.run(collect(sampler))
    assert (event.period_ms, event.sensor_avg_samples, event.sensor_update_ms, event.oc_events) == (100.0, None, None, {})


def test_an_unreadable_power_mode_is_null_not_a_guess(tmp_path):
    sampler = JetsonPowerSampler(ScriptedTegrastats(every(LINES[:1])), box(tmp_path), nvpmodel(None), nvpmodel_every_s=0)
    [event] = asyncio.run(collect(sampler))
    assert event.nvpmodel is None


def test_lines_without_rails_and_lines_that_are_not_tegrastats_are_counted_and_never_published(tmp_path):
    no_rails = "04-24-2025 09:49:12 RAM 1603/7620MB (lfb 9x4MB) SWAP 0/16384MB (cached 0MB) cpu@48.437C"
    script = [(T0, "NVPM WARN: something"), (T0 + timedelta(milliseconds=100), no_rails),
              (T0 + timedelta(milliseconds=200), LINES[9]), (T0 + timedelta(milliseconds=300), "")]
    sampler = JetsonPowerSampler(ScriptedTegrastats(script), box(tmp_path), nvpmodel("MAXN_SUPER"), nvpmodel_every_s=0)
    events = asyncio.run(collect(sampler))
    assert [(e.ts, e.rails["VDD_IN"]) for e in events] == [(T0 + timedelta(milliseconds=200), 4629.0)]
    assert {k: sampler.status()[k] for k in ("lines", "samples", "unparsed", "without_rails")} == \
        {"lines": 4, "samples": 1, "unparsed": 2, "without_rails": 1}


def test_lines_that_arrive_bunched_together_are_counted(tmp_path):
    arrivals = [T0, T0 + timedelta(milliseconds=100), T0 + timedelta(milliseconds=101),
                T0 + timedelta(milliseconds=102), T0 + timedelta(milliseconds=300)]
    sampler = JetsonPowerSampler(ScriptedTegrastats(list(zip(arrivals, LINES))), box(tmp_path), nvpmodel("15W"),
                                 nvpmodel_every_s=0)
    asyncio.run(collect(sampler))
    assert sampler.status()["bunched"] == 2


def test_each_sample_reads_the_overcurrent_counters_at_that_moment(tmp_path):
    sysfs = box(tmp_path)
    bump = lambda: write(tmp_path, "sys/class/hwmon/hwmon3/oc3_event_cnt", "195")  # noqa: E731
    sampler = JetsonPowerSampler(ScriptedTegrastats([(T0, LINES[0]), bump, (T0 + timedelta(milliseconds=100), LINES[1])]),
                                 sysfs, nvpmodel("MAXN_SUPER"), nvpmodel_every_s=0)
    events = asyncio.run(collect(sampler))
    assert [e.oc_events["oc3"] for e in events] == [194, 195]


def test_a_power_mode_switch_shows_up_in_later_samples(tmp_path):
    async def run():
        tegrastats = ScriptedTegrastats()

        async def slow_lines(interval_ms):
            for n, line in enumerate(LINES[:6]):
                yield T0 + timedelta(milliseconds=100 * n), line
                await asyncio.sleep(0.05)

        tegrastats.lines = slow_lines
        sampler = JetsonPowerSampler(tegrastats, box(tmp_path), nvpmodel("15W", "MAXN_SUPER"), nvpmodel_every_s=0.02)
        return await collect(sampler)

    events = asyncio.run(run())
    assert events[0].nvpmodel == "15W" and events[-1].nvpmodel == "MAXN_SUPER"


def test_a_sensor_update_interval_that_grows_past_the_period_restarts_tegrastats_at_the_new_period(tmp_path):
    sysfs = box(tmp_path, update="26")
    slower = lambda: write(tmp_path, f"{INA}/update_interval", "250")  # noqa: E731
    first = [(T0, LINES[0]), slower] + every(LINES[1:4], start=T0 + timedelta(milliseconds=100))
    second = every(LINES[4:6], start=T0 + timedelta(seconds=10), step_ms=250)
    tegrastats = ScriptedTegrastats(first, second)
    sampler = JetsonPowerSampler(tegrastats, sysfs, nvpmodel("MAXN_SUPER"), nvpmodel_every_s=0, settings_every_s=0.2)
    waits = []

    async def no_wait(s):
        waits.append(s)

    async def run():
        bus = EventBus(tmp_path / "events")
        runner = SamplerRunner(sampler, bus.publish, nodes=["jetson"], retry_s=5, sleep=no_wait)
        await runner.run()
        return runner, list(bus.read_log())

    runner, logged = asyncio.run(run())
    assert tegrastats.intervals == [100, 250] and waits == [5]
    assert [(e.period_ms, e.sensor_update_ms) for e in logged] == [(100.0, 26.0)] + [(250.0, 250.0)] * 2
    assert runner.errors == 1 and "longer than the 100 ms sampling period" in runner.last_error
    assert tegrastats.closed == 2


def test_the_runner_reports_what_the_sampler_knows(tmp_path):
    sampler = JetsonPowerSampler(ScriptedTegrastats(every(LINES[:3])), box(tmp_path), nvpmodel("MAXN_SUPER"),
                                 nvpmodel_every_s=0)

    async def run():
        runner = SamplerRunner(sampler, EventBus(persist=False).publish, nodes=["jetson"])
        await runner.run()
        return runner.status()

    assert asyncio.run(run()) == {
        "nodes": ["jetson"], "published": 3, "dropped": 0, "errors": 0, "last_error": None,
        "source": {"requested_ms": 100, "period_ms": 100, "nvpmodel": "MAXN_SUPER", "sensor_avg_samples": 4,
                   "sensor_update_ms": 26.0, "lines": 3, "samples": 3, "unparsed": 0, "without_rails": 0,
                   "bunched": 0, "tegrastats_pid": None}}
