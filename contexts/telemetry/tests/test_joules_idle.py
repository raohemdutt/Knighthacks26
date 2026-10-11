"""The idle figure stored beside a joules-per-answer value: the node's calibrated idle, with the measurement's source
row named, and the latest quiet reading only where no calibrated figure exists (T4b; telemetry rulings T10 and T13)."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bus import EventBus
from contexts.telemetry.application.idle import IDLE_ENV, IdleFigure, idle_figures
from contexts.telemetry.application.joules import Calibration
from contexts.telemetry.tests.test_joules import by_id, load, replay
from contexts.telemetry.wiring import build_telemetry
from contracts.telemetry import JoulesPerAnswer

CONFIG = Path(__file__).resolve().parents[3] / "config" / "telemetry_idle.yaml"
POD_ROW = datetime(2026, 10, 10, 22, 32, 58, tzinfo=UTC)
POD_A = IdleFigure(10.27, source="config", metric="pod.idle_watts", measured=POD_ROW)


def test_the_calibrated_path_stores_the_nodes_idle_and_names_the_row_it_came_from():
    values = by_id([v for v in replay(load("idle_sources"), Calibration(idle={"pod-a": POD_A}))[1]
                    if v.node == "pod-a"])
    for rid in ("pod-a-first", "pod-a-second"):  # the second starts on the power tail; the calibrated idle does not care
        v = values[rid]
        assert (v.idle_w, v.idle_source) == (10.27, "calibrated")
        assert (v.idle_metric, v.idle_measured) == ("pod.idle_watts", POD_ROW)
        assert v.idle_ts is None  # a calibrated figure is not a reading taken before this window


def test_the_fallback_path_takes_the_latest_quiet_reading_and_says_so():
    values = [v for v in replay(load("idle_sources"), Calibration(idle={"pod-a": POD_A}))[1] if v.node == "pod-b"]
    second = by_id(values)["pod-b-second"]
    assert (second.idle_w, second.idle_source) == (13.0, "latest_quiet")  # the power tail, the very thing T10 rejected
    assert second.idle_ts is not None and second.idle_ts < second.started
    assert (second.idle_metric, second.idle_measured) == (None, None)


def test_with_no_calibration_at_all_every_value_falls_back_to_the_latest_quiet_reading():
    values = [v for v in replay(load("idle_sources"))[1] if v.node == "pod-a"]
    assert {v.idle_source for v in values} == {"latest_quiet"}
    assert all(v.idle_w is not None for v in values)


def test_a_calibrated_node_is_not_confused_with_another_the_figure_follows_the_node():
    jetson = IdleFigure(15.76, source="config", metric="jetson.idle_watts",
                        measured=datetime(2026, 10, 10, 21, 58, 10, tzinfo=UTC))
    values = replay(load("conditions"), Calibration(idle={"jetson": jetson}))[1]
    by_node = {v.node: v for v in values}
    assert (by_node["jetson"].idle_w, by_node["jetson"].idle_source) == (15.76, "calibrated")
    assert by_node["pod-b"].idle_source == "latest_quiet"


def test_the_new_fields_are_optional_so_a_value_already_in_the_log_still_reads():
    v = replay(load("idle_sources"))[1][0].model_dump(mode="json")
    for key in ("idle_source", "idle_metric", "idle_measured"):
        v.pop(key)
    old = JoulesPerAnswer.model_validate(v)
    assert (old.idle_source, old.idle_metric, old.idle_measured, old.version) == (None, None, None, 1)


# --- the setting ---

def test_the_setting_is_one_figure_per_node_and_names_where_it_came_from():
    figures = idle_figures({IDLE_ENV: "pod-a=10.27,pod-b=10.27, jetson=15.76"}, path=None)
    assert {n: f.watts for n, f in figures.items()} == {"pod-a": 10.27, "pod-b": 10.27, "jetson": 15.76}
    assert figures["pod-a"].source == IDLE_ENV and figures["pod-a"].metric is None


def test_the_old_pod_names_read_as_the_pods():
    assert set(idle_figures({IDLE_ENV: "gpu-a=10.27"}, path=None)) == {"pod-a"}


@pytest.mark.parametrize("raw", ["pod-a", "pod-a=", "pod-a=ten", "pod-a=0", "pod-a=-3", "pod-a=nan", "pod-z=10",
                                 "pod-a=10,pod-a=11"])
def test_a_bad_setting_is_refused_by_name_not_guessed(raw):
    with pytest.raises(ValueError, match=IDLE_ENV):
        idle_figures({IDLE_ENV: raw}, path=None)


def test_the_committed_config_carries_the_calibrated_figures_with_their_source_rows():
    figures = idle_figures({}, path=CONFIG)
    assert set(figures) == {"jetson", "pod-a", "pod-b"}
    assert (figures["pod-a"].watts, figures["pod-a"].metric, figures["pod-a"].measured) == (10.27, "pod.idle_watts", POD_ROW)
    assert (figures["jetson"].watts, figures["jetson"].metric, figures["jetson"].measured) == (
        15.76, "jetson.idle_watts", datetime(2026, 10, 10, 21, 58, 10, tzinfo=UTC))
    assert figures["pod-b"].watts == 10.27 and "pod-a" in figures["pod-b"].note  # same until pod B is measured


def test_the_setting_overrides_the_config_per_node_and_drops_the_row_it_no_longer_matches():
    figures = idle_figures({IDLE_ENV: "pod-b=9.8"}, path=CONFIG)
    assert figures["pod-b"].watts == 9.8 and figures["pod-b"].metric is None and figures["pod-b"].measured is None
    assert figures["pod-a"].watts == 10.27 and figures["pod-a"].metric == "pod.idle_watts"


def test_a_missing_config_file_means_no_figures_not_an_error(tmp_path):
    assert idle_figures({}, path=tmp_path / "nope.yaml") == {}


def test_a_malformed_config_is_refused(tmp_path):
    bad = tmp_path / "idle.yaml"
    bad.write_text("nodes:\n  pod-a: {watts: ten}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pod-a"):
        idle_figures({}, path=bad)


def test_the_service_is_built_with_the_calibrated_idle_from_the_config_and_the_setting():
    service = build_telemetry("fake", EventBus(persist=False), on_jetson=lambda: False, env={IDLE_ENV: "jetson=16.0"})
    idle = service.joules.calibration.idle
    assert idle["jetson"].watts == 16.0 and idle["pod-a"].watts == 10.27


def test_a_calibration_handed_in_is_used_as_it_is():
    given = Calibration(idle={"pod-a": POD_A})
    service = build_telemetry("fake", EventBus(persist=False), on_jetson=lambda: False, calibration=given)
    assert service.joules.calibration.idle == {"pod-a": POD_A}
