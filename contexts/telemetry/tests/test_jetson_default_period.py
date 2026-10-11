"""T2's two closing clauses: the raised default period still flags under-sampled windows, and the box's gateway
unit turns the sampler on at that period so it samples from boot and after every restart.

The default went from 100 ms to 350 ms under telemetry owner ruling 8 (T13 measured 3.51 percent of one core at
100 ms against a 1 percent budget). At 350 ms a 2.1 s warm caption holds about 6 samples and must stay
low_confidence; a 3.5 s window and T7's mean caption of 9.5 s hold 10 or more and carry no sample-count reason."""

import asyncio
import configparser
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bus import EventBus
from contexts.gateway.__main__ import load_dotenv
from contexts.telemetry.application.jetson_sampler import DEFAULT_INTERVAL_MS
from contexts.telemetry.application.joules import MIN_SAMPLES, Calibration, JoulesPerAnswerProjection
from contexts.telemetry.wiring import jetson_power_sampler
from contracts.telemetry import JetsonPower, RequestWindow

REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "jetson" / "atlas-gateway.service"
T0 = datetime(2026, 10, 11, 5, 0, tzinfo=UTC)
FEW = f"fewer than {MIN_SAMPLES} samples in the window"
WINDOWS = {"warm-caption": (10.0, 2.1), "ruling-window": (20.0, 3.5), "mean-caption": (30.0, 9.5)}


def samples_at_the_default_period(seconds: float) -> list[JetsonPower]:
    step = DEFAULT_INTERVAL_MS / 1000
    return [JetsonPower(node="jetson", ts=T0 + timedelta(seconds=i * step), source="tegrastats",
                        period_ms=float(DEFAULT_INTERVAL_MS), rails={"VDD_IN": 9000.0}, nvpmodel="MAXN_SUPER",
                        oc_events={"oc1": 0, "oc2": 0, "oc3": 0})
            for i in range(int(seconds / step) + 1)]


def window(rid: str, start_s: float, length_s: float) -> RequestWindow:
    start = T0 + timedelta(seconds=start_s)
    end = start + timedelta(seconds=length_s)
    return RequestWindow(node="jetson", ts=end, request_id=rid, started=start, ended=end, tokens_in=100,
                         tokens_out=20, target="student")


def joules_for_each_window() -> dict:
    # a measured error for the Jetson, so the sample count is the only reason a window can be flagged for
    joules = JoulesPerAnswerProjection(Calibration(error={"jetson": 0.05}, settle_s=5.0))
    events = sorted(samples_at_the_default_period(60.0) + [window(r, *w) for r, w in WINDOWS.items()],
                    key=lambda e: e.ts)

    async def run():
        bus = EventBus(persist=False)
        joules.subscribe(bus)
        await bus.publish_all(events)
        return joules.ready()

    return {v.request_id: v for v in asyncio.run(run())}


def test_at_the_default_period_a_warm_caption_is_low_confidence_for_its_sample_count():
    values = joules_for_each_window()
    assert values["warm-caption"].samples < MIN_SAMPLES
    assert values["warm-caption"].low_confidence and values["warm-caption"].reasons == [FEW]


def test_at_the_default_period_the_ruling_window_and_a_mean_caption_hold_enough_samples():
    values = joules_for_each_window()
    for rid in ("ruling-window", "mean-caption"):
        assert values[rid].samples >= MIN_SAMPLES, rid
        assert FEW not in values[rid].reasons and not values[rid].low_confidence, rid


def unit_environment() -> dict[str, str]:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str
    text = UNIT.read_text(encoding="utf-8")
    env: dict[str, str] = {}
    for line in text.splitlines():  # repeated Environment= keys: configparser keeps only the last, so read lines
        if line.startswith("Environment="):
            key, _, value = line.removeprefix("Environment=").partition("=")
            env[key] = value
    parser.read_string(text)
    assert parser.get("Service", "Restart") == "always"  # what "survives a restart" stands on
    return env


def test_the_gateway_unit_turns_the_tegrastats_sampler_on_at_the_default_period():
    env = unit_environment()
    assert env["ATLAS_TELEMETRY_SOURCE"] == "tegrastats"
    assert int(env["ATLAS_TELEMETRY_JETSON_INTERVAL_MS"]) == DEFAULT_INTERVAL_MS
    assert jetson_power_sampler(env).interval_ms == DEFAULT_INTERVAL_MS


def test_the_units_settings_win_over_dotenv_as_the_unit_comment_says(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_TELEMETRY_SOURCE", "tegrastats")  # what systemd puts in the gateway's environment
    dotenv = tmp_path / ".env"
    dotenv.write_text("ATLAS_TELEMETRY_SOURCE=fake\n", encoding="utf-8")
    load_dotenv(dotenv)
    assert os.environ["ATLAS_TELEMETRY_SOURCE"] == "tegrastats"
