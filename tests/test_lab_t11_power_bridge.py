"""T11 off the Jetson: lab/t11_power_bridge.py turns recorded tegrastats lines and a recorded pod sampler answer into
the power events the closed loop reads, and lab/t11_offbox_run.py applies the run's outage rule.

Every input here was recorded on 2026-10-11 from the team Jetson and pod C (fixtures/telemetry/t11_bridge_*); nothing
reaches a network address: the pod answer is served by an httpx MockTransport.
"""

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from contexts.telemetry.application.joules import Calibration, JoulesPerAnswerProjection
from contracts.telemetry import ClockOffset, GpuPower, JetsonPower, RequestWindow
from lab import label as label_mod
from lab import t11_offbox_run as offbox
from lab import t11_power_bridge as bridge
from lab.routing_closed_loop import JsonlFeed

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "telemetry"
TEGRASTATS = FIXTURES / "t11_bridge_tegrastats.txt"
POD_PAGE = FIXTURES / "t11_bridge_pod_page.json"
ENV = {"AUP_HUB_USER_C": "someone", "AUP_HUB_TOKEN_C": "x" * 32, "AUP_HUB_URL_C": "https://hub.example"}


def _recorded_lines() -> list[tuple[datetime, str]]:
    out = []
    for line in TEGRASTATS.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        stamp, text = line.split("\t", 1)
        out.append((datetime.fromisoformat(stamp), text))
    return out


def test_a_recorded_tegrastats_line_becomes_a_jetson_power_event_stamped_on_arrival():
    ts, line = _recorded_lines()[0]
    event = bridge.jetson_event(line, ts, period_ms=350, nvpmodel="MAXN_SUPER")
    assert isinstance(event, JetsonPower)
    assert event.node == "jetson" and event.ts == ts and event.period_ms == 350 and event.nvpmodel == "MAXN_SUPER"
    assert event.rails["VDD_IN"] == 10396.0  # the current reading, never the running average after the slash
    assert set(event.rails) == {"VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"}
    assert event.temps_c["gpu"] == 60.843


def test_a_line_without_rails_is_skipped():
    assert bridge.jetson_event("RAM 6834/7608MB (lfb 11x1MB) cpu@61.1C", datetime.now(UTC), period_ms=350,
                               nvpmodel=None) is None
    assert bridge.jetson_event("ssh: connection closed", datetime.now(UTC), period_ms=350, nvpmodel=None) is None


def test_the_bridge_writes_lines_the_closed_loop_feed_reads_back(tmp_path):
    out = tmp_path / "power.jsonl"
    sink = bridge.JsonlSink(out)
    tb = bridge.TegrastatsBridge(sink, target="nobody@nowhere", interval_ms=350, nvpmodel="MAXN_SUPER")
    lines = _recorded_lines()
    tb.feed(lines + [(datetime.now(UTC), "not tegrastats")])
    sink.close()
    events = JsonlFeed([out]).poll()
    assert len(events) == len(lines) and tb.skipped == 1
    assert all(isinstance(e, JetsonPower) and e.node == "jetson" for e in events)
    assert [e.ts for e in events] == [ts for ts, _ in lines]


def _pod_transport(page: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"].startswith("token ")
        assert request.url.path == "/user/someone/proxy/8790/samples"
        return httpx.Response(200, json={k: v for k, v in page.items() if not k.startswith("_")})
    return httpx.MockTransport(handler)


def test_the_recorded_pod_answer_becomes_pod_a_samples_and_a_clock_offset_against_the_run_clock(tmp_path):
    page = json.loads(POD_PAGE.read_text(encoding="utf-8"))
    rec = page["_recorded"]
    times = iter([datetime.fromisoformat(rec["t1"]), datetime.fromisoformat(rec["t4"])] * 4)
    out = tmp_path / "power.jsonl"
    sink = bridge.JsonlSink(out)
    pod = bridge.PodBridge(sink, ENV, account="C", transport=_pod_transport(page), rounds=1, history_s=0,
                           clock=lambda: next(times))
    pod.run(threading.Event())
    sink.close()
    events = JsonlFeed([out]).poll()
    offsets = [e for e in events if isinstance(e, ClockOffset)]
    samples = [e for e in events if isinstance(e, GpuPower)]
    assert len(offsets) == 1 and offsets[0].node == "pod-a" and offsets[0].reference == "jetson"
    assert offsets[0].offset_s == pytest.approx(64.03, abs=0.05)  # the pod's clock ahead of the workstation's
    assert len(samples) == 15 and all(s.node == "pod-a" and s.received_ts is not None for s in samples)
    assert [s.socket_w for s in samples[:5]] == [11.0, 11.0, 10.0, 10.0, 9.0]


def test_history_before_the_bridge_started_is_not_copied(tmp_path):
    page = json.loads(POD_PAGE.read_text(encoding="utf-8"))
    last = datetime.fromisoformat(page["samples"][-1]["event"]["ts"].replace("Z", "+00:00"))
    sink = bridge.JsonlSink(tmp_path / "p.jsonl")
    # started 1 s after the newest sample's pod time, keeping 5 s of history: only the last five or so samples pass
    pod = bridge.PodBridge(sink, ENV, transport=_pod_transport(page), rounds=1, history_s=5.0,
                           started=last + timedelta(seconds=1))
    pod.run(threading.Event())
    sink.close()
    kept = [e for e in JsonlFeed([tmp_path / "p.jsonl"]).poll() if isinstance(e, GpuPower)]
    assert 0 < len(kept) < 15 and all(e.ts > last - timedelta(seconds=5) for e in kept)


def test_bridged_samples_give_an_answer_its_joules_on_both_nodes(tmp_path):
    """The projection the closed loop uses reads the bridged file: a Jetson window on the run's clock, and a pod
    window moved onto the pod's clock by the bridged offset."""
    out = tmp_path / "power.jsonl"
    sink = bridge.JsonlSink(out)
    lines = _recorded_lines()
    bridge.TegrastatsBridge(sink, target="x", interval_ms=350, nvpmodel="MAXN_SUPER").feed(lines)
    page = json.loads(POD_PAGE.read_text(encoding="utf-8"))
    rec = page["_recorded"]
    times = iter([datetime.fromisoformat(rec["t1"]), datetime.fromisoformat(rec["t4"])] * 4)
    bridge.PodBridge(sink, ENV, transport=_pod_transport(page), rounds=1, history_s=0,
                     clock=lambda: next(times)).run(threading.Event())
    sink.close()
    projection = JoulesPerAnswerProjection(Calibration(settle_s=0.5))
    for event in JsonlFeed([out]).poll():
        projection.on_event(event)
    j0, j1 = lines[1][0], lines[-4][0]  # the last samples run past the settle time
    projection.on_event(RequestWindow(ts=j1, node="jetson", request_id="j", started=j0, ended=j1, tokens_in=None,
                                      tokens_out=10, target="student", kind="label", policy="capability"))
    offset = 64.02903354167938
    p_first = datetime.fromisoformat(page["samples"][2]["event"]["ts"].replace("Z", "+00:00"))
    p_last = datetime.fromisoformat(page["samples"][-3]["event"]["ts"].replace("Z", "+00:00"))
    run_start, run_end = p_first - timedelta(seconds=offset), p_last - timedelta(seconds=offset)
    projection.on_event(RequestWindow(ts=run_end, node="pod-a", request_id="t", started=run_start, ended=run_end,
                                      tokens_in=None, tokens_out=10, target="teacher", kind="label",
                                      policy="capability", clock="jetson"))
    values = {v.node: v for v in projection.ready()}
    assert set(values) == {"jetson", "pod-a"}
    jetson_s = (j1 - j0).total_seconds()
    assert 5.0 * jetson_s < values["jetson"].joules < 20.0 * jetson_s  # VDD_IN between 5 and 20 W
    pod_s = (p_last - p_first).total_seconds()
    assert 8.0 * pod_s < values["pod-a"].joules < 12.0 * pod_s  # the idle pod read 9 to 11 W
    assert values["pod-a"].clock_offset_s == pytest.approx(offset)


# --- the outage rule (lab/t11_offbox_run.py) ---

def _teacher_failing(times: int, error: label_mod.TeacherError) -> tuple:
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("not used")

    teacher = label_mod.Teacher("http://127.0.0.1:1", transport=httpx.MockTransport(handler), retries=0,
                                sleep=lambda s: None, node="jetson")

    def once():
        calls[0] += 1
        if calls[0] <= times:
            raise error
        return "answer"

    return teacher, calls, once


def test_an_outage_is_paused_every_two_minutes_and_logged(tmp_path):
    progress = offbox._Progress(tmp_path / "progress.jsonl")
    progress.current = "007-B"
    slept, now = [], [0.0]

    def sleep(s):
        slept.append(s)
        now[0] += s

    wrapped = offbox.outage_retries(label_mod.Teacher._with_retries, progress, sleep=sleep, clock=lambda: now[0])
    teacher, calls, once = _teacher_failing(3, label_mod.TeacherError("POST: no answer (ConnectError)", retry=True))
    assert wrapped(teacher, once) == "answer"
    assert slept == [120.0, 120.0, 120.0]
    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [r["request"] for r in rows] == ["007-B"] * 3 and all(r["pause"] for r in rows)
    teacher.close()


def test_an_outage_past_twenty_minutes_stops_the_run(tmp_path):
    progress = offbox._Progress(tmp_path / "progress.jsonl")
    now = [0.0]

    def sleep(s):
        now[0] += s

    wrapped = offbox.outage_retries(label_mod.Teacher._with_retries, progress, sleep=sleep, clock=lambda: now[0])
    teacher, calls, once = _teacher_failing(10**6, label_mod.TeacherError("500: boom", 500))
    with pytest.raises(offbox.GaveUp):
        wrapped(teacher, once)
    assert now[0] == 1200.0 and calls[0] == 11
    teacher.close()


@pytest.mark.parametrize("status", [400, 401, 404])
def test_a_4xx_is_not_an_outage(tmp_path, status):
    progress = offbox._Progress(tmp_path / "progress.jsonl")
    wrapped = offbox.outage_retries(label_mod.Teacher._with_retries, progress, sleep=lambda s: None)
    teacher, calls, once = _teacher_failing(1, label_mod.TeacherError(f"{status}: no", status,
                                                                      fatal=status in label_mod.FATAL_STATUSES))
    with pytest.raises(label_mod.TeacherError):
        wrapped(teacher, once)
    assert calls[0] == 1 and not (tmp_path / "progress.jsonl").exists()
    teacher.close()
