"""Review-driven checks for lab/pod_power.py (PR 142), offline, on crafted but realistic inputs."""

from pathlib import Path

from lab import pod_power

# --- t-sampler-alive-check-needs-the-sampler-command-line ----------------------------------------------------------

_SAMPLER_CMDLINE = b"python3\x00/home/jovyan/atlas-power/pod_power.py\x00serve\x00--node\x00gpu-a\x00"
_REUSED_PID_CMDLINE = b"python3\x00-m\x00vllm.entrypoints.openai.api_server\x00"


def _fake_proc(tmp_path: Path, pid: int | None, cmdline: bytes | None) -> Path:
    """A /proc tree under tmp_path; only the pid entry the test names exists in it."""
    proc = tmp_path / "proc"
    proc.mkdir()
    if pid is not None and cmdline is not None:
        (proc / str(pid)).mkdir()
        (proc / str(pid) / "cmdline").write_bytes(cmdline)
    return proc


def _pidfile(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sampler.pid"
    path.write_text(text, encoding="ascii")
    return path


def test_alive_sampler_pid_returns_the_pid_when_its_command_line_is_the_sampler_serving(tmp_path):
    proc = _fake_proc(tmp_path, 4242, _SAMPLER_CMDLINE)
    pidfile = _pidfile(tmp_path, "4242\n")
    assert pod_power.alive_sampler_pid(pidfile, proc=proc) == 4242


def test_alive_sampler_pid_is_none_when_the_pid_was_reused_by_another_program(tmp_path):
    proc = _fake_proc(tmp_path, 4242, _REUSED_PID_CMDLINE)
    pidfile = _pidfile(tmp_path, "4242\n")
    assert pod_power.alive_sampler_pid(pidfile, proc=proc) is None


def test_alive_sampler_pid_is_none_when_the_pid_has_no_proc_entry(tmp_path):
    proc = _fake_proc(tmp_path, None, None)
    pidfile = _pidfile(tmp_path, "4242\n")
    assert pod_power.alive_sampler_pid(pidfile, proc=proc) is None


def test_alive_sampler_pid_is_none_when_the_pid_file_is_absent(tmp_path):
    proc = _fake_proc(tmp_path, 4242, _SAMPLER_CMDLINE)
    assert pod_power.alive_sampler_pid(tmp_path / "no-such.pid", proc=proc) is None


def test_alive_sampler_pid_is_none_when_the_pid_file_is_not_an_integer(tmp_path):
    proc = _fake_proc(tmp_path, 4242, _SAMPLER_CMDLINE)
    pidfile = _pidfile(tmp_path, "not-a-pid\n")
    assert pod_power.alive_sampler_pid(pidfile, proc=proc) is None


def test_alive_subcommand_exits_1_for_a_reused_pid(tmp_path, capsys):
    proc = _fake_proc(tmp_path, 4242, _REUSED_PID_CMDLINE)
    pidfile = _pidfile(tmp_path, "4242\n")
    code = pod_power.main(["alive", "--pidfile", str(pidfile), "--proc", str(proc)])
    assert code == 1
    assert "4242" not in capsys.readouterr().out


def test_alive_subcommand_exits_0_and_prints_the_pid_for_the_running_sampler(tmp_path, capsys):
    proc = _fake_proc(tmp_path, 4242, _SAMPLER_CMDLINE)
    pidfile = _pidfile(tmp_path, "4242\n")
    code = pod_power.main(["alive", "--pidfile", str(pidfile), "--proc", str(proc)])
    assert code == 0
    assert capsys.readouterr().out.split() == ["4242"]


# --- t-gate-refuses-a-9070-xt-that-reports-gfx1201 -----------------------------------------------------------------

_RX_9070_XT_ROCMINFO = "  Marketing Name:          AMD Radeon RX 9070 XT\n  Name:                    gfx1201\n"
_R9700_ROCMINFO = "  Marketing Name:          AMD Radeon AI PRO R9700\n  Name:                    gfx1201\n"


def test_gate_refuses_a_9070_xt_that_reports_gfx1201_and_still_admits_an_r9700():
    refused = pod_power.gate("gpu-a", lambda: _RX_9070_XT_ROCMINFO)
    assert refused is not None, "the RX 9070 XT also reports gfx1201 but is not an R9700 pod"
    assert "9070" in refused
    assert pod_power.gate("gpu-a", lambda: _R9700_ROCMINFO) is None


# --- t-summary-shows-a-receive-gap-the-pod-clock-hides -------------------------------------------------------------

import pytest as _pytest_rx  # noqa: E402
from datetime import UTC as _UTC_RX, datetime as _datetime_rx  # noqa: E402

_RX_T0 = _datetime_rx(2026, 10, 10, 18, 0, tzinfo=_UTC_RX).timestamp()
_RX_PROXY_GAP_S = 60.0
_RX_FIRST_DELAY_S = 0.2


def _rx_gpu_power(n: int, with_received_ts: bool) -> dict:
    """Sample n (0-based): ts n seconds after the start on the pod clock; samples 10 and later reached the Jetson
    only after a 60 s proxy gap, so their received_ts is 60.2 s after ts and the first ten's is 0.2 s."""
    ts = _RX_T0 + n
    delay = _RX_FIRST_DELAY_S if n < 10 else _RX_FIRST_DELAY_S + _RX_PROXY_GAP_S
    event = {
        "event_id": f"rx-{n}", "type": pod_power.GPU_POWER, "version": 1, "ts": pod_power.iso(ts), "node": "gpu-a",
        "source": "amd-smi", "period_ms": 1000, "socket_w": 40.0, "power_field": "SOCKET_POWER",
        "temps_c": {"edge": 45.0}, "sampler": "lab/pod_power.py 1", "sampler_started": pod_power.iso(_RX_T0 - 5),
    }
    if with_received_ts:
        event["received_ts"] = pod_power.iso(ts + delay)
    return event


def test_summary_shows_a_receive_gap_the_pod_clock_hides():
    events = [_rx_gpu_power(n, with_received_ts=True) for n in range(30)]

    summary = pod_power.summarize(events, node="gpu-a")

    # the pod clock saw one sample a second, so its fields are unchanged ...
    assert summary["samples"] == 30
    assert summary["measured_period_ms"] == 1000.0
    assert summary["max_gap_ms"] == 1000.0
    # ... and the Jetson's receive times show the 60 s proxy gap before sample 11 and how late it arrived
    assert summary["max_receive_gap_ms"] == _pytest_rx.approx(61000.0, abs=1.0)
    assert summary["max_receive_delay_s"] == _pytest_rx.approx(60.2, abs=0.01)


def test_summary_without_received_ts_leaves_the_receive_fields_none():
    events = [_rx_gpu_power(n, with_received_ts=False) for n in range(30)]

    summary = pod_power.summarize(events, node="gpu-a")

    assert summary["measured_period_ms"] == 1000.0
    assert summary["max_gap_ms"] == 1000.0
    assert summary["max_receive_gap_ms"] is None
    assert summary["max_receive_delay_s"] is None
