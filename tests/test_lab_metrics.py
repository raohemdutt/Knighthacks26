"""Tests for the metrics baseline file (lab/refine/metrics.jsonl) and its reader (lab/machine/metrics.py)."""

import datetime as dt
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from lab.machine import metrics

ROOT = Path(__file__).resolve().parents[1]
MODEL = "RedHatAI/Qwen3.8-27B-INT4"


def make_row(**overrides):
    row = {
        "metric": "teacher.decode_tok_s",
        "value": 7.7,
        "unit": "tok/s",
        "node": "pod-a",
        "conditions": {"model": MODEL, "batch": 1},
        "source": "unit test",
        "ts": "2026-10-10T10:20:00Z",
        "low_confidence": False,
    }
    row.update(overrides)
    return row


def write_lines(path, *lines):
    path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))


# --- the seeded baseline -------------------------------------------------------------------------

SAMPLER = "lab/pod_power.py 1"  # the T6 rows' conditions name the sampler that read them

IDLE_DEEP = "deep idle: teacher loaded, no request in flight"
IDLE_MCLK_UP = "idle sclk, memory clock still up (mclk 1000 MHz), the seconds after a request"
IDLE_TWO_READERS = (
    "deep idle with a second reader running (rocm-smi --showclocks and amd-smi metric --power forked once a second "
    "beside the sampler)"
)

# (metric, node, conditions that select exactly one row, value, unit, low_confidence)
R26 = "bench-20261010T215122Z-2c65"  # the stock teacher's batching sweep (April build, b1388b1)
R60 = "bench-20261011T044643Z-6240"  # the same sweep on vLLM v0.31.0 with the RDNA hybrid W4A16 kernel
EXPECTED_SEED = [
    ("teacher.decode_tok_s", "pod-a", {"prompt": "image"}, 6.6, "tok/s", False),
    ("teacher.decode_tok_s", "pod-a", {"prompt": "text", "thinking": "off"}, 7.8, "tok/s", False),
    ("teacher.decode_tok_s", "pod-a", {"prompt": "text", "thinking": "on"}, 7.7, "tok/s", False),
    ("teacher.ttft_s", "pod-a", {"prompt": "image"}, 1.09, "s", False),
    ("teacher.ttft_s", "pod-a", {"prompt": "text", "thinking": "on"}, 0.28, "s", False),
    ("teacher.ttft_s", "pod-a", {"prompt": "text", "thinking": "off"}, 0.55, "s", False),
    ("teacher.watts_avg", "pod-a", {"while": "generating"}, 200, "W", False),
    ("teacher.watts_peak", "pod-a", {"while": "generating"}, 300, "W", False),
    ("teacher.engine_init_s", "pod-a", {}, 270, "s", False),
    ("teacher.vram_pct", "pod-a", {}, 84, "%", False),
    ("teacher.max_concurrency", "pod-a", {}, 10, "x", False),
    ("teacher.joules_per_token", "pod-a", {}, 25.97, "J/token", True),
    ("trainer.samples_per_s", "pod-b", {}, 0.51, "samples/s", False),
    ("trainer.peak_mem_gb", "pod-b", {}, 8.7, "GiB", False),
    ("trainer.smoke_loss_first", "pod-b", {}, 2.23, "loss", False),
    ("trainer.smoke_loss_last", "pod-b", {}, 0.16, "loss", False),
    ("pod.idle_watts", "pod-a", {"state": "idle", "range_w": [38, 40]}, 39.0, "W", True),
    ("pod.start_s", "pod-a", {}, 15, "s", False),
    ("pod.credits_per_min", "pod-a", {}, 4, "credits/min", False),
    ("pod.credits_per_min", "pod-b", {}, 4, "credits/min", False),
    ("pod.balance_credits", "pod-a", {}, 59960, "credits", True),
    ("pod.balance_credits", "pod-b", {}, 59960, "credits", True),
    # R29: the teacher on the SJC pod C (Strix Halo, gfx1151)
    ("teacher.ttft_s", "pod-c", {"prompt": "text", "thinking": "off"}, 0.39, "s", False),
    ("teacher.decode_tok_s", "pod-c", {"prompt": "text", "thinking": "off"}, 2.88, "tok/s", False),
    ("teacher.ttft_s", "pod-c", {"prompt": "image"}, 3.51, "s", False),
    ("teacher.decode_tok_s", "pod-c", {"prompt": "image"}, 2.82, "tok/s", False),
    ("teacher.e2e_s", "pod-c", {"prompt": "text"}, 27.51, "s", False),
    ("teacher.e2e_s", "pod-c", {"prompt": "image"}, 12.85, "s", False),
    ("teacher.watts_avg", "pod-c", {"prompt": "text"}, 120.3, "W", False),
    ("teacher.watts_peak", "pod-c", {}, 129, "W", False),
    ("teacher.watts_avg", "pod-c", {"prompt": "image"}, 118.3, "W", False),
    ("pod.idle_watts", "pod-c", {"server": "teacher loaded, no request"}, 9.2, "W", False),
    ("pod.idle_watts", "pod-c", {"server": "none"}, 8, "W", False),
    ("teacher.engine_init_s", "pod-c", {}, 297, "s", False),
    # T6: the pod power sampler's live run on pod C under the teacher load (recorded as node pod-a, the teacher's role);
    # the samples are fixtures/telemetry/pod_power/strixhalo_2026-10-10_teacher_load_*.jsonl
    ("pod.idle_watts", "pod-a", {"sampler": SAMPLER, "under_load_samples": 0}, 10.2, "W", True),
    ("teacher.watts_avg", "pod-a", {"sampler": SAMPLER}, 106.807, "W", True),
    ("teacher.watts_peak", "pod-a", {"sampler": SAMPLER, "value": "peak"}, 129.0, "W", True),
    ("pod.start_s", "pod-d", {"launch_path": "form"}, 5.55, "s", False),
    ("pod.first_request_s", "pod-d", {"launch_path": "form"}, 11.7, "s", False),
    ("pod.keep_alive_min", "pod-d", {"launch_path": "form"}, 70, "min", False),
    ("pod.credits_per_hour", "pod-d", {"billing": "unmetered"}, 0, "credits/h", False),
    # T10: idle watts per observed clock state on pod C (recorded as node pod-a), from
    # fixtures/telemetry/pod_power/strixhalo_2026-10-10_idle_baseline_*.jsonl (recomputed in test_idle_baseline_rows_*)
    ("pod.idle_watts", "pod-a", {"state": IDLE_DEEP}, 10.27, "W", True),
    ("pod.idle_watts", "pod-a", {"state": IDLE_MCLK_UP}, 11.14, "W", True),
    ("pod.idle_watts", "pod-a", {"state": IDLE_TWO_READERS}, 13.84, "W", True),
    ("pod.settle_s", "pod-a", {"sampler": SAMPLER}, 4.0, "s", True),
    ("pod.idle_state_share", "pod-a", {"state": "deep idle (sclk below 700 MHz)"}, 62.15, "%", True),
    ("pod.idle_state_share", "pod-a", {"state": "transition (sclk 700 to 2599 MHz)"}, 6.64, "%", True),
    ("pod.idle_state_share", "pod-a", {"state": "busy (sclk 2600 MHz or more, mclk 1000)"}, 31.22, "%", True),
    # R18: the proxy failure-mode map against pod D (account D on SJC, pod B's role)
    ("proxy.status", "pod-b", {"case": "bad token", "method": "GET"}, 302, "http", False),
    ("proxy.status", "pod-b", {"case": "bad token", "method": "POST"}, 403, "http", False),
    ("proxy.status", "pod-b", {"case": "nothing listening"}, 500, "http", False),
    ("proxy.status", "pod-b", {"case": "mid-restart"}, 500, "http", False),
    ("proxy.restart_to_ready_s", "pod-b", {"case": "mid-restart"}, 294.1, "s", False),
    ("proxy.status", "pod-b", {"case": "model server stopped"}, 500, "http", False),
    ("proxy.stream_first_byte_s", "pod-b", {"accept": "text/event-stream"}, 0.916, "s", False),
    ("proxy.stream_first_byte_s", "pod-b", {"accept": "application/json"}, 31.483, "s", False),
    ("proxy.status", "pod-b", {"case": "5 MB body"}, 400, "http", False),
    ("proxy.concurrent_wall_s", "pod-b", {"case": "20 concurrent"}, 2.998, "s", False),
    # T13: the Jetson sampler's CPU cost and the box's idle watts, over SSH in MAXN_SUPER (fixtures/telemetry/jetson_sampler_cost)
    ("telemetry.sampler_cpu_pct", "jetson", {"period_ms": 100, "load": "idle"}, 3.513, "percent of one core", True),
    ("telemetry.sampler_cpu_pct", "jetson", {"period_ms": 200, "load": "idle"}, 1.833, "percent of one core", True),
    ("telemetry.sampler_cpu_pct", "jetson", {"period_ms": 100, "load": "one caption request in flight"}, 3.28, "percent of one core", True),
    ("telemetry.sampler_cpu_pct", "jetson", {"period_ms": 200, "load": "one caption request in flight"}, 1.703, "percent of one core", True),
    ("jetson.idle_watts", "jetson", {"rail": "VDD_IN", "samples": 3000}, 15.76, "W", True),
    # T7: the Jetson power sampler read once a second for 10 minutes in MAXN_SUPER, idle then under caption load, from
    # fixtures/telemetry/jetson_power_live (recomputed in tests/test_lab_jetson_live_run.py)
    ("jetson.idle_watts", "jetson", {"rail": "VDD_IN", "period_ms": 1000}, 12.62, "W", True),
    ("student.watts_avg", "jetson", {"rail": "VDD_IN", "period_ms": 1000}, 17.536, "W", True),
    ("student.watts_peak", "jetson", {"rail": "VDD_IN", "period_ms": 1000}, 19.127, "W", True),
    # R26: the batching sweep on the pod C teacher, attach mode, 3 repeats per concurrency (node pod-a, gfx1151)
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 1, "session": R26}, 2.7021, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 1, "session": R26}, 2.6172, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 1, "session": R26}, 1.1809, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 1, "session": R26}, 36.6587, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 1, "session": R26}, 96.0046, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 1, "session": R26}, 97.0, "W", True),
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 4, "session": R26}, 2.5218, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 4, "session": R26}, 9.3153, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 4, "session": R26}, 4.8672, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 4, "session": R26}, 10.3071, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 4, "session": R26}, 96.0126, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 4, "session": R26}, 97.0, "W", True),
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 8, "session": R26}, 2.319, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 8, "session": R26}, 16.3616, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 8, "session": R26}, 7.7821, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 8, "session": R26}, 5.8676, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 8, "session": R26}, 96.004, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 8, "session": R26}, 97.0, "W", True),
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 16, "session": R26}, 2.0712, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 16, "session": R26}, 26.946, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 16, "session": R26}, 14.9531, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 16, "session": R26}, 3.5628, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 16, "session": R26}, 96.0052, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 16, "session": R26}, 97.0, "W", True),
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 64, "session": R26}, 0.8854, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 64, "session": R26}, 49.5288, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 64, "session": R26}, 24.3901, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 64, "session": R26}, 1.9384, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 64, "session": R26}, 96.0499, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 64, "session": R26}, 98.0, "W", True),
    # R60: the same sweep after the teacher moved to vLLM v0.31.0 (db9527a46) with the RDNA hybrid W4A16 kernel,
    # attach mode, 3 repeats per concurrency, no 64 arm (node pod-a, gfx1151)
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 1, "session": R60}, 12.3264, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 1, "session": R60}, 10.9589, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 1, "session": R60}, 0.9399, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 1, "session": R60}, 8.8161, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 1, "session": R60}, 96.7243, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 1, "session": R60}, 99.0, "W", True),
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 4, "session": R60}, 9.2679, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 4, "session": R60}, 29.5444, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 4, "session": R60}, 3.8513, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 4, "session": R60}, 3.2719, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 4, "session": R60}, 96.2464, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 4, "session": R60}, 98.0, "W", True),
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 8, "session": R60}, 5.8808, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 8, "session": R60}, 36.5269, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 8, "session": R60}, 6.7616, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 8, "session": R60}, 2.6888, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 8, "session": R60}, 96.1088, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 8, "session": R60}, 97.0, "W", True),
    ("teacher_batching.decode_tok_s", "pod-a", {"batch_size": 16, "session": R60}, 4.244, "tok/s", True),
    ("teacher_batching.decode_tok_s_total", "pod-a", {"batch_size": 16, "session": R60}, 51.0898, "tok/s", True),
    ("teacher_batching.ttft_s", "pod-a", {"batch_size": 16, "session": R60}, 10.4472, "s", True),
    ("teacher_batching.joules_per_token", "pod-a", {"batch_size": 16, "session": R60}, 1.8799, "J/token", True),
    ("teacher_batching.watts_avg", "pod-a", {"batch_size": 16, "session": R60}, 96.0946, "W", True),
    ("teacher_batching.watts_peak", "pod-a", {"batch_size": 16, "session": R60}, 97.0, "W", True),
    ("pod.idle_watts", "pod-a", {"server": "teacher loaded, no request"}, 10.44, "W", False),
    # R21: the stock teacher on pod C compared with its own baseline (the K4 self-match floor), one compare
    ("teacher.output_match_rate", "pod-a", {"chip": "gfx1151", "batch_size": 1, "engine_commit": "b1388b1"}, 0.5, "fraction", True),
    # R60: the new build against the stock baseline (1 of 4; the two image prompts and text-schedule diverge)
    ("teacher.output_match_rate", "pod-a", {"chip": "gfx1151", "batch_size": 1, "engine_commit": "db9527a46"}, 0.25, "fraction", True),
    # R57: the A100 reference on Colab (other chip: NVIDIA A100-SXM4-40GB, stock vLLM 0.31.0), the same INT4 teacher
    # through lab/bench.py, 3 cold spawns per batch size; watts from nvidia-smi over each decode window
    # (fixtures/lab/bench/a100-2026-10-10, recomputed in test_a100_reference_rows_match_the_fixture)
    ("teacher.decode_tok_s", "a100", {"batch_size": 1}, 64.956, "tok/s", True),
    ("teacher.decode_tok_s_total", "a100", {"batch_size": 1}, 61.2077, "tok/s", True),
    ("teacher.ttft_s", "a100", {"batch_size": 1}, 0.1184, "s", True),
    ("teacher.watts_avg", "a100", {"batch_size": 1}, 306.0747, "W", True),
    ("teacher.watts_peak", "a100", {"batch_size": 1}, 407.97, "W", True),
    ("teacher.joules_per_token", "a100", {"batch_size": 1}, 5.0174, "J/token", True),
    ("teacher.decode_tok_s", "a100", {"batch_size": 4}, 59.2302, "tok/s", True),
    ("teacher.decode_tok_s_total", "a100", {"batch_size": 4}, 198.933, "tok/s", True),
    ("teacher.ttft_s", "a100", {"batch_size": 4}, 0.4949, "s", True),
    ("teacher.watts_avg", "a100", {"batch_size": 4}, 302.8075, "W", True),
    ("teacher.watts_peak", "a100", {"batch_size": 4}, 411.5, "W", True),
    ("teacher.joules_per_token", "a100", {"batch_size": 4}, 1.5234, "J/token", True),
    ("teacher.decode_tok_s", "a100", {"batch_size": 8}, 52.6137, "tok/s", True),
    ("teacher.decode_tok_s_total", "a100", {"batch_size": 8}, 321.3743, "tok/s", True),
    ("teacher.ttft_s", "a100", {"batch_size": 8}, 0.8243, "s", True),
    ("teacher.watts_avg", "a100", {"batch_size": 8}, 321.0224, "W", True),
    ("teacher.watts_peak", "a100", {"batch_size": 8}, 418.76, "W", True),
    ("teacher.joules_per_token", "a100", {"batch_size": 8}, 0.999, "J/token", True),
    ("teacher.decode_tok_s", "a100", {"batch_size": 16}, 36.0495, "tok/s", True),
    ("teacher.decode_tok_s_total", "a100", {"batch_size": 16}, 462.3296, "tok/s", True),
    ("teacher.ttft_s", "a100", {"batch_size": 16}, 1.056, "s", True),
    ("teacher.watts_avg", "a100", {"batch_size": 16}, 355.9959, "W", True),
    ("teacher.watts_peak", "a100", {"batch_size": 16}, 412.96, "W", True),
    ("teacher.joules_per_token", "a100", {"batch_size": 16}, 0.7743, "J/token", True),
    ("teacher.decode_tok_s", "a100", {"batch_size": 64}, 11.2975, "tok/s", True),
    ("teacher.decode_tok_s_total", "a100", {"batch_size": 64}, 636.3307, "tok/s", True),
    ("teacher.ttft_s", "a100", {"batch_size": 64}, 1.9381, "s", True),
    ("teacher.watts_avg", "a100", {"batch_size": 64}, 382.1646, "W", True),
    ("teacher.watts_peak", "a100", {"batch_size": 64}, 424.61, "W", True),
    ("teacher.joules_per_token", "a100", {"batch_size": 64}, 0.6003, "J/token", True),
    ("pod.idle_watts", "a100", {"hub": "colab"}, 46.14, "W", True),
    # R20: the baseline matrix on pod D (SJC account D, Strix Halo gfx1151), vLLM b1388b1 stock build, from
    # fixtures/lab/bench/pod-b-2026-10-10 (median of the spawns that ran per batch size; a partial matrix)
    ("teacher.decode_tok_s", "pod-b", {"batch_size": 1, "session": "pod-b-2026-10-10"}, 2.7327, "tok/s", True),
    ("teacher.decode_tok_s_total", "pod-b", {"batch_size": 1, "session": "pod-b-2026-10-10"}, 2.6665, "tok/s", True),
    ("teacher.ttft_s", "pod-b", {"batch_size": 1, "session": "pod-b-2026-10-10"}, 0.9579, "s", True),
    ("teacher.joules_per_token", "pod-b", {"batch_size": 1, "session": "pod-b-2026-10-10"}, 36.4001, "J/token", True),
    ("teacher.watts_avg", "pod-b", {"batch_size": 1, "session": "pod-b-2026-10-10"}, 97.2353, "W", True),
    ("teacher.watts_peak", "pod-b", {"batch_size": 1, "session": "pod-b-2026-10-10"}, 104.9594, "W", True),
    ("teacher.decode_tok_s", "pod-b", {"batch_size": 4, "session": "pod-b-2026-10-10"}, 2.5807, "tok/s", True),
    ("teacher.decode_tok_s_total", "pod-b", {"batch_size": 4, "session": "pod-b-2026-10-10"}, 9.4809, "tok/s", True),
    ("teacher.ttft_s", "pod-b", {"batch_size": 4, "session": "pod-b-2026-10-10"}, 5.1508, "s", True),
    ("teacher.joules_per_token", "pod-b", {"batch_size": 4, "session": "pod-b-2026-10-10"}, 10.33, "J/token", True),
    ("teacher.watts_avg", "pod-b", {"batch_size": 4, "session": "pod-b-2026-10-10"}, 97.9385, "W", True),
    ("teacher.watts_peak", "pod-b", {"batch_size": 4, "session": "pod-b-2026-10-10"}, 104.0, "W", True),
    ("teacher.decode_tok_s", "pod-b", {"batch_size": 8, "session": "pod-b-2026-10-10"}, 2.3677, "tok/s", True),
    ("teacher.decode_tok_s_total", "pod-b", {"batch_size": 8, "session": "pod-b-2026-10-10"}, 16.5352, "tok/s", True),
    ("teacher.ttft_s", "pod-b", {"batch_size": 8, "session": "pod-b-2026-10-10"}, 8.4941, "s", True),
    ("teacher.joules_per_token", "pod-b", {"batch_size": 8, "session": "pod-b-2026-10-10"}, 5.8068, "J/token", True),
    ("teacher.watts_avg", "pod-b", {"batch_size": 8, "session": "pod-b-2026-10-10"}, 96.0158, "W", True),
    ("teacher.watts_peak", "pod-b", {"batch_size": 8, "session": "pod-b-2026-10-10"}, 97.0, "W", True),
    ("teacher.decode_tok_s", "pod-b", {"batch_size": 16, "session": "pod-b-2026-10-10"}, 2.0627, "tok/s", True),
    ("teacher.decode_tok_s_total", "pod-b", {"batch_size": 16, "session": "pod-b-2026-10-10"}, 27.4494, "tok/s", True),
    ("teacher.ttft_s", "pod-b", {"batch_size": 16, "session": "pod-b-2026-10-10"}, 11.9126, "s", True),
    ("teacher.joules_per_token", "pod-b", {"batch_size": 16, "session": "pod-b-2026-10-10"}, 3.4978, "J/token", True),
    ("teacher.watts_avg", "pod-b", {"batch_size": 16, "session": "pod-b-2026-10-10"}, 96.0119, "W", True),
    ("teacher.watts_peak", "pod-b", {"batch_size": 16, "session": "pod-b-2026-10-10"}, 97.0, "W", True),
    ("teacher.decode_tok_s", "pod-b", {"batch_size": 64, "session": "pod-b-2026-10-10"}, 0.8821, "tok/s", True),
    ("teacher.decode_tok_s_total", "pod-b", {"batch_size": 64, "session": "pod-b-2026-10-10"}, 48.9584, "tok/s", True),
    ("teacher.ttft_s", "pod-b", {"batch_size": 64, "session": "pod-b-2026-10-10"}, 24.1934, "s", True),
    ("teacher.joules_per_token", "pod-b", {"batch_size": 64, "session": "pod-b-2026-10-10"}, 1.9619, "J/token", True),
    ("teacher.watts_avg", "pod-b", {"batch_size": 64, "session": "pod-b-2026-10-10"}, 96.0237, "W", True),
    ("teacher.watts_peak", "pod-b", {"batch_size": 64, "session": "pod-b-2026-10-10"}, 97.7339, "W", True),
    # E29: llama.cpp 23b0202 HIP serving the teacher's Q4_K_M GGUF on pod D (gfx1151), cut short for L5: batch 1 and 4
    # over 3 runs, 8 and 16 once, 64 not run; and llama-bench's own pp512 and tg128
    ("teacher_llamacpp.decode_tok_s", "pod-b", {"batch_size": 1}, 12.1185, "tok/s", True),
    ("teacher_llamacpp.decode_tok_s_total", "pod-b", {"batch_size": 1}, 10.4758, "tok/s", True),
    ("teacher_llamacpp.ttft_s", "pod-b", {"batch_size": 1}, 1.4634, "s", True),
    ("teacher_llamacpp.joules_per_token", "pod-b", {"batch_size": 1}, 10.449, "J/token", True),
    ("teacher_llamacpp.watts_avg", "pod-b", {"batch_size": 1}, 109.9128, "W", True),
    ("teacher_llamacpp.watts_peak", "pod-b", {"batch_size": 1}, 117.0, "W", True),
    ("teacher_llamacpp.decode_tok_s", "pod-b", {"batch_size": 4}, 9.0896, "tok/s", True),
    ("teacher_llamacpp.decode_tok_s_total", "pod-b", {"batch_size": 4}, 23.419, "tok/s", True),
    ("teacher_llamacpp.ttft_s", "pod-b", {"batch_size": 4}, 7.1765, "s", True),
    ("teacher_llamacpp.joules_per_token", "pod-b", {"batch_size": 4}, 4.679, "J/token", True),
    ("teacher_llamacpp.watts_avg", "pod-b", {"batch_size": 4}, 109.5083, "W", True),
    ("teacher_llamacpp.watts_peak", "pod-b", {"batch_size": 4}, 122.0, "W", True),
    ("teacher_llamacpp.decode_tok_s", "pod-b", {"batch_size": 8}, 3.7304, "tok/s", True),
    ("teacher_llamacpp.decode_tok_s_total", "pod-b", {"batch_size": 8}, 21.5206, "tok/s", True),
    ("teacher_llamacpp.ttft_s", "pod-b", {"batch_size": 8}, 14.4666, "s", True),
    ("teacher_llamacpp.joules_per_token", "pod-b", {"batch_size": 8}, 4.7024, "J/token", True),
    ("teacher_llamacpp.watts_avg", "pod-b", {"batch_size": 8}, 101.1975, "W", True),
    ("teacher_llamacpp.watts_peak", "pod-b", {"batch_size": 8}, 128.0, "W", True),
    ("teacher_llamacpp.decode_tok_s", "pod-b", {"batch_size": 16}, 4.5329, "tok/s", True),
    ("teacher_llamacpp.decode_tok_s_total", "pod-b", {"batch_size": 16}, 41.9757, "tok/s", True),
    ("teacher_llamacpp.ttft_s", "pod-b", {"batch_size": 16}, 27.5993, "s", True),
    ("teacher_llamacpp.joules_per_token", "pod-b", {"batch_size": 16}, 2.2716, "J/token", True),
    ("teacher_llamacpp.watts_avg", "pod-b", {"batch_size": 16}, 95.3507, "W", True),
    ("teacher_llamacpp.watts_peak", "pod-b", {"batch_size": 16}, 101.0, "W", True),
    ("teacher_llamacpp.llama_bench_pp512_tok_s", "pod-b", {"test": "pp512"}, 390.7811, "tok/s", True),
    ("teacher_llamacpp.llama_bench_tg128_tok_s", "pod-b", {"test": "tg128"}, 12.1822, "tok/s", True),
]

# The numbers the lab setup measured, by metric name.
MEASURED_NAMES = {
    "teacher.decode_tok_s", "teacher.ttft_s", "teacher.watts_avg", "teacher.watts_peak",
    "teacher.engine_init_s", "teacher.vram_pct", "trainer.samples_per_s", "trainer.peak_mem_gb",
    "pod.idle_watts", "pod.start_s", "pod.credits_per_min", "pod.balance_credits",
    "telemetry.sampler_cpu_pct", "jetson.idle_watts",
}


# R63: the dispatch head-to-head on pod-b, every row recomputed from its result file (lab/kernels/dispatch_bench.py)
R63_RESULT = Path(__file__).resolve().parents[1] / "fixtures" / "lab" / "dispatch" / "dispatch-gfx1151-2026-10-11.json"


def _r63_rows() -> list[dict]:
    from lab.kernels import dispatch_bench

    result = json.loads(R63_RESULT.read_text(encoding="utf-8"))
    return dispatch_bench.metric_rows(result, "fixtures/lab/dispatch/dispatch-gfx1151-2026-10-11.json", "D")


def test_r63_rows_match_their_result_file():
    expected = _r63_rows()
    found = [row for row in metrics.load()
             if row["metric"].startswith(("dispatch.", "lut_gemv.mode4.")) and "R63" in row["source"]]
    assert found == [dict(metrics.check_row(row)) for row in expected]


def test_seed_file_has_exactly_the_expected_rows_in_schema_order():
    rows = metrics.load()
    assert len(rows) == len(EXPECTED_SEED) + len(_r63_rows())
    for row in rows:
        assert list(row) == list(metrics.FIELDS)


@pytest.mark.parametrize("metric, node, conditions, value, unit, low_confidence", EXPECTED_SEED)
def test_seed_row_carries_the_traced_number(metric, node, conditions, value, unit, low_confidence):
    matches = metrics.history(metric, node=node, conditions=conditions)
    assert len(matches) == 1
    row = matches[0]
    assert row["value"] == value
    assert row["unit"] == unit
    assert row["low_confidence"] is low_confidence
    assert row["source"].strip() != ""


def test_seed_covers_every_number_the_lab_setup_measured():
    assert MEASURED_NAMES <= {row["metric"] for row in metrics.load()}


def test_latest_on_pod_a_is_the_headline_text_prompt():
    assert metrics.latest("teacher.decode_tok_s", node="pod-a")["value"] == 7.7
    assert metrics.latest("teacher.ttft_s", node="pod-a")["value"] == 0.55


def test_seed_history_on_pod_a_keeps_every_condition_of_a_metric_in_file_order():
    assert [row["value"] for row in metrics.history("teacher.decode_tok_s", node="pod-a")] == [6.6, 7.8, 7.7]
    assert [row["value"] for row in metrics.history("teacher.ttft_s", node="pod-a")] == [1.09, 0.28, 0.55]


def test_derived_joules_per_token_matches_the_inputs_it_names():
    row = metrics.latest("teacher.joules_per_token", node="pod-a")
    assert row["conditions"]["derived"] is True
    inputs = row["conditions"]["inputs"]
    for name, value in inputs.items():
        assert value in [r["value"] for r in metrics.history(name, node="pod-a")], name
    assert row["value"] == round(inputs["teacher.watts_avg"] / inputs["teacher.decode_tok_s"], 2)


def test_every_low_confidence_seed_row_says_why():
    flagged = [row for row in metrics.load() if row["low_confidence"]]
    assert {row["metric"] for row in flagged} == {
        "teacher.joules_per_token", "pod.idle_watts", "pod.balance_credits", "teacher.watts_avg", "teacher.watts_peak",
        "pod.settle_s", "pod.idle_state_share",
        "telemetry.sampler_cpu_pct", "jetson.idle_watts", "student.watts_avg", "student.watts_peak"} | {
        # R26: the sweep's medians are of 3 repeats, under the 5 verdict() asks for
        "teacher_batching." + name
        for name in ("decode_tok_s", "decode_tok_s_total", "ttft_s", "joules_per_token", "watts_avg", "watts_peak")} | {
        "teacher.output_match_rate"} | {
        # R57: the A100 reference medians are of 3 spawns
        "telemetry.sampler_cpu_pct", "jetson.idle_watts", "student.watts_avg", "student.watts_peak",
        "teacher.decode_tok_s", "teacher.decode_tok_s_total", "teacher.ttft_s"} | {
        # E29: llama.cpp on pod D, cut short (1 to 3 runs per batch size) and one llama-bench call
        "teacher_llamacpp." + name
        for name in ("decode_tok_s", "decode_tok_s_total", "ttft_s", "joules_per_token", "watts_avg", "watts_peak",
                     "llama_bench_pp512_tok_s", "llama_bench_tg128_tok_s")}
    for row in flagged:
        if row["conditions"].get("sampler") == SAMPLER:  # the sampler's rows: its error is "unknown" until a reference run
            assert row["conditions"]["error"] == "unknown", row["metric"]
            continue
        reason = row["conditions"].get("low_confidence_reason")
        assert isinstance(reason, str) and reason.strip() != "", row["metric"]


A100_FIXTURE = ROOT / "fixtures" / "lab" / "bench" / "a100-2026-10-10"
A100_SUMMARY_NAMES = ("decode_tok_s", "decode_tok_s_total", "ttft_s", "watts_avg", "watts_peak", "joules_per_token")


def test_a100_reference_rows_match_the_fixture(tmp_path):
    # analyze.py recomputes the joules from the nvidia-smi samples and the harness rows; its table is the rows' values
    for name in ("analyze.py", "rows.jsonl", "nvidia_smi_power.csv", "idle_window.txt"):
        (tmp_path / name).write_bytes((A100_FIXTURE / name).read_bytes())
    subprocess.run([sys.executable, "-I", str(tmp_path / "analyze.py"), str(tmp_path)], check=True, capture_output=True)
    summary = json.loads((tmp_path / "summary_a100.json").read_text(encoding="utf-8"))
    assert summary == json.loads((A100_FIXTURE / "summary_a100.json").read_text(encoding="utf-8"))
    rows = [row for row in metrics.load() if row["node"] == "a100"]
    for batch, entry in summary["per_batch"].items():
        assert entry["runs_ok"] == entry["runs"] == 3
        for name in A100_SUMMARY_NAMES:
            matches = [r for r in rows if r["metric"] == f"teacher.{name}" and r["conditions"]["batch_size"] == int(batch)]
            assert len(matches) == 1, (name, batch)
            assert matches[0]["value"] == round(entry[name], 4), (name, batch)
            assert matches[0]["conditions"]["other_chip"] is True
            assert matches[0]["conditions"]["hub"] == "colab"
    idle = [r for r in rows if r["metric"] == "pod.idle_watts"]
    assert [r["value"] for r in idle] == [round(summary["idle_no_server"]["avg_w"], 2)]


def test_a100_greedy_record_is_an_r21_baseline_beside_the_strix_halo_one():
    from lab import correctness

    a100 = correctness.load_baseline(A100_FIXTURE / "vllm-a100-2026-10-10.json")
    strix = correctness.load_baseline(ROOT / "fixtures" / "lab" / "correctness" / "vllm-gfx1151-2026-10-11.json")
    assert a100["conditions"]["chip"] == "a100" and a100["conditions"]["other_chip"] is True
    assert a100["conditions"]["prompt_set"] == strix["conditions"]["prompt_set"]
    assert [len(o["tokens"]) for o in a100["outputs"]] == [128] * 4
    rows = {r["prompt"]: r["first_divergence"] for r in correctness.compare_outputs(strix["outputs"], a100["outputs"])}
    # the cross-chip picture on 2026-10-10: the kitchen text matches token for token, the rest diverge
    assert rows == {"text-kitchen": None, "image-bars": 0, "text-schedule": 76, "image-shapes": 3}


def test_seed_file_is_plain_ascii_with_unix_line_endings():
    data = metrics.DEFAULT_PATH.read_bytes()
    data.decode("ascii")
    assert b"\r" not in data
    assert data.endswith(b"\n")


# --- add -----------------------------------------------------------------------------------------

def test_add_appends_one_canonical_line_and_returns_the_row_as_written(tmp_path):
    path = tmp_path / "m.jsonl"
    row = make_row()
    written = metrics.add(row, path=path)
    assert path.read_bytes() == (
        '{"metric": "teacher.decode_tok_s", "value": 7.7, "unit": "tok/s", "node": "pod-a", '
        '"conditions": {"model": "RedHatAI/Qwen3.8-27B-INT4", "batch": 1}, "source": "unit test", '
        '"ts": "2026-10-10T10:20:00Z", "low_confidence": false}\n'
    ).encode("utf-8")
    assert written == row


def test_add_puts_fields_in_schema_order_whatever_the_caller_order(tmp_path):
    path = tmp_path / "m.jsonl"
    shuffled = dict(reversed(list(make_row().items())))
    metrics.add(shuffled, path=path)
    assert list(json.loads(path.read_text(encoding="utf-8"))) == list(metrics.FIELDS)


def test_add_keeps_earlier_lines_untouched_and_appends_in_order(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1), path=path)
    first = path.read_bytes()
    metrics.add(make_row(value=2), path=path)
    metrics.add(make_row(value=3), path=path)
    assert path.read_bytes().startswith(first)
    assert [row["value"] for row in metrics.load(path)] == [1, 2, 3]


def test_add_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "m.jsonl"
    metrics.add(make_row(), path=path)
    assert len(metrics.load(path)) == 1


def test_add_stamps_the_current_utc_time_when_ts_is_left_out(tmp_path):
    path = tmp_path / "m.jsonl"
    row = make_row()
    del row["ts"]
    before = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    written = metrics.add(row, path=path)
    after = dt.datetime.now(dt.timezone.utc)
    assert written["ts"].endswith("Z")
    stamped = dt.datetime.strptime(written["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    assert before <= stamped <= after
    assert metrics.load(path)[0]["ts"] == written["ts"]


def test_add_accepts_fractional_seconds_and_integer_values(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(ts="2026-10-10T10:20:00.5Z", value=4), path=path)
    row = metrics.load(path)[0]
    assert row["ts"] == "2026-10-10T10:20:00.5Z"
    assert row["value"] == 4 and isinstance(row["value"], int)


BAD_ROWS = [
    ("missing metric", lambda r: r.pop("metric"), "metric"),
    ("missing value", lambda r: r.pop("value"), "value"),
    ("missing conditions", lambda r: r.pop("conditions"), "conditions"),
    ("missing low_confidence", lambda r: r.pop("low_confidence"), "low_confidence"),
    ("unknown field", lambda r: r.update(extra=1), "extra"),
    ("metric without a dot", lambda r: r.update(metric="decode"), "metric"),
    ("metric in capitals", lambda r: r.update(metric="Teacher.decode"), "metric"),
    ("metric with a space", lambda r: r.update(metric="teacher.decode tok"), "metric"),
    ("value is a bool", lambda r: r.update(value=True), "value"),
    ("value is a string", lambda r: r.update(value="7.7"), "value"),
    ("value is nan", lambda r: r.update(value=float("nan")), "value"),
    ("value is infinite", lambda r: r.update(value=float("inf")), "value"),
    ("unit is empty", lambda r: r.update(unit=""), "unit"),
    ("unit is not text", lambda r: r.update(unit=3), "unit"),
    ("node in capitals", lambda r: r.update(node="Pod-A"), "node"),
    ("node is empty", lambda r: r.update(node=""), "node"),
    ("conditions is a list", lambda r: r.update(conditions=[1]), "conditions"),
    ("conditions not json", lambda r: r.update(conditions={"x": object()}), "conditions"),
    ("source is blank", lambda r: r.update(source="   "), "source"),
    ("low_confidence is an int", lambda r: r.update(low_confidence=0), "low_confidence"),
    ("ts without the Z", lambda r: r.update(ts="2026-10-10T10:20:00"), "ts"),
    ("ts with an offset", lambda r: r.update(ts="2026-10-10T10:20:00+00:00"), "ts"),
    ("ts date only", lambda r: r.update(ts="2026-10-10"), "ts"),
    ("ts impossible date", lambda r: r.update(ts="2026-13-40T10:20:00Z"), "ts"),
]


@pytest.mark.parametrize("label, damage, field", BAD_ROWS, ids=[case[0] for case in BAD_ROWS])
def test_add_rejects_a_bad_row_and_leaves_the_file_as_it_was(tmp_path, label, damage, field):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(), path=path)
    before = path.read_bytes()
    row = make_row()
    damage(row)
    with pytest.raises(metrics.MetricsFormatError, match=field):
        metrics.add(row, path=path)
    assert path.read_bytes() == before


def test_add_refuses_to_append_after_a_line_that_was_cut_short(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_bytes(json.dumps(make_row()).encode("utf-8"))  # no trailing newline
    before = path.read_bytes()
    with pytest.raises(metrics.MetricsFormatError, match="newline"):
        metrics.add(make_row(value=9), path=path)
    assert path.read_bytes() == before


# --- load, latest, history -----------------------------------------------------------------------

def test_load_of_a_missing_file_is_empty_and_creates_nothing(tmp_path):
    path = tmp_path / "absent.jsonl"
    assert metrics.load(path) == []
    assert not path.exists()


def test_load_skips_blank_lines_and_reads_crlf_endings(tmp_path):
    path = tmp_path / "m.jsonl"
    first, second = json.dumps(make_row(value=1)), json.dumps(make_row(value=2))
    path.write_bytes((first + "\r\n\r\n" + second + "\r\n").encode("utf-8"))
    assert [row["value"] for row in metrics.load(path)] == [1, 2]


def test_load_refuses_a_corrupt_line_and_names_it(tmp_path):
    path = tmp_path / "m.jsonl"
    write_lines(path, json.dumps(make_row()), '{"metric": "teacher.decode_tok_s", "value": ', json.dumps(make_row()))
    with pytest.raises(metrics.MetricsFormatError, match=r"line 2"):
        metrics.load(path)


def test_load_numbered_counts_blank_and_crlf_lines_the_way_load_names_them(tmp_path):
    path = tmp_path / "m.jsonl"
    first, second = json.dumps(make_row(value=1)), json.dumps(make_row(value=2))
    good = (first + "\n" + "\n" + second + "\r\n" + "   \n").encode("utf-8")
    path.write_bytes(good + b"{not json\n")

    with pytest.raises(metrics.MetricsFormatError, match=r"line 5\b") as from_load:
        metrics.load(path)
    with pytest.raises(metrics.MetricsFormatError, match=r"line 5\b") as from_numbered:
        metrics.load_numbered(path)
    assert "line 5" in str(from_load.value) and "line 5" in str(from_numbered.value)

    path.write_bytes(good)
    rows = metrics.load(path)
    numbered = metrics.load_numbered(path)
    assert numbered == [(1, rows[0]), (3, rows[1])]
    assert [row for _, row in numbered] == rows
    assert [row["value"] for _, row in numbered] == [1, 2]


def test_load_refuses_bytes_that_are_not_utf8(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_bytes(json.dumps(make_row()).encode("utf-8") + b"\n\xff\xfe\n")
    with pytest.raises(metrics.MetricsFormatError, match=r"line 2"):
        metrics.load(path)


def test_load_refuses_a_row_that_breaks_the_schema_and_names_line_and_field(tmp_path):
    path = tmp_path / "m.jsonl"
    write_lines(path, json.dumps(make_row()), json.dumps(make_row(value=True)))
    with pytest.raises(metrics.MetricsFormatError, match=r"line 2.*value"):
        metrics.load(path)


def test_load_refuses_a_row_with_an_unknown_field(tmp_path):
    path = tmp_path / "m.jsonl"
    write_lines(path, json.dumps({**make_row(), "seq": 1}))
    with pytest.raises(metrics.MetricsFormatError, match=r"line 1.*seq"):
        metrics.load(path)


def test_load_requires_ts_on_every_stored_row(tmp_path):
    path = tmp_path / "m.jsonl"
    row = make_row()
    del row["ts"]
    write_lines(path, json.dumps(row))
    with pytest.raises(metrics.MetricsFormatError, match=r"line 1.*ts"):
        metrics.load(path)


def test_latest_is_the_newest_timestamp_not_the_last_line(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=2, ts="2026-10-10T12:00:00Z"), path=path)
    metrics.add(make_row(value=1, ts="2026-10-10T09:00:00Z"), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 2


def test_latest_orders_fractional_seconds_correctly(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=2, ts="2026-10-10T10:20:00.5Z"), path=path)
    metrics.add(make_row(value=1, ts="2026-10-10T10:20:00Z"), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 2


def test_latest_breaks_a_timestamp_tie_with_the_later_line(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1), path=path)
    metrics.add(make_row(value=2), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 2


def test_latest_filters_by_node_and_by_condition_subset(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1, node="pod-a", conditions={"prompt": "text", "batch": 1}), path=path)
    metrics.add(make_row(value=2, node="pod-a", conditions={"prompt": "image", "batch": 1}), path=path)
    metrics.add(make_row(value=3, node="pod-b", conditions={"prompt": "text", "batch": 4}), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 3
    assert metrics.latest("teacher.decode_tok_s", node="pod-a", path=path)["value"] == 2
    assert metrics.latest("teacher.decode_tok_s", node="pod-a", conditions={"prompt": "text"}, path=path)["value"] == 1
    assert metrics.latest("teacher.decode_tok_s", conditions={"batch": 4}, path=path)["value"] == 3
    assert metrics.latest("teacher.decode_tok_s", node="pod-b", conditions={"prompt": "image"}, path=path) is None
    assert metrics.latest("teacher.decode_tok_s", node="pod-c", path=path) is None


def test_latest_of_an_unknown_metric_is_none_and_its_history_is_empty(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(), path=path)
    assert metrics.latest("teacher.ttft_s", path=path) is None
    assert metrics.history("teacher.ttft_s", path=path) == []


def test_history_is_oldest_first_for_one_metric_only(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=3, ts="2026-10-10T12:00:00Z"), path=path)
    metrics.add(make_row(metric="teacher.ttft_s", unit="s", value=0.5), path=path)
    metrics.add(make_row(value=1, ts="2026-10-10T08:00:00Z"), path=path)
    metrics.add(make_row(value=2, ts="2026-10-10T10:00:00Z"), path=path)
    assert [row["value"] for row in metrics.history("teacher.decode_tok_s", path=path)] == [1, 2, 3]


def test_history_filters_by_node(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1, node="pod-a"), path=path)
    metrics.add(make_row(value=2, node="pod-b", ts="2026-10-10T11:00:00Z"), path=path)
    assert [row["value"] for row in metrics.history("teacher.decode_tok_s", node="pod-b", path=path)] == [2]


def test_a_row_added_after_a_read_shows_up_on_the_next_read(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 1
    metrics.add(make_row(value=2, ts="2026-10-10T11:00:00Z"), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 2


# --- the file lock, across real processes --------------------------------------------------------

HOLD_THE_LOCK = """
import sys
from lab.machine import metrics

lock = metrics.FileLock(sys.argv[1])
if not lock.acquire(timeout=10):
    sys.exit("child could not take the lock")
print("locked", flush=True)
sys.stdin.read()
lock.release()
"""

WRITE_AFTER_GO = """
import os
import sys
import time
from lab.machine import metrics

path, writer, count, go = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
print("ready", flush=True)
while not os.path.exists(go):
    time.sleep(0.001)
for i in range(count):
    metrics.add({"metric": "test.counter", "value": i, "unit": "n", "node": "writer-" + writer,
                 "conditions": {"writer": writer, "i": i}, "source": "concurrency test",
                 "ts": "2026-10-10T10:20:00Z", "low_confidence": False}, path=path)
"""


def start(code, *args, errors, stdin=None):
    """Run `code` in a child python from the repo root; its stderr goes to the open file `errors`, so a child that
    writes a lot can never block on a full pipe."""
    return subprocess.Popen(
        [sys.executable, "-c", code, *map(str, args)], cwd=ROOT, text=True,
        stdin=stdin, stdout=subprocess.PIPE, stderr=errors)


def test_add_and_load_give_up_with_metrics_locked_while_another_process_holds_the_lock(tmp_path):
    path = tmp_path / "m.jsonl"
    errors_path = tmp_path / "children.err"
    with open(errors_path, "w") as errors:
        holder = start(HOLD_THE_LOCK, path, errors=errors, stdin=subprocess.PIPE)
        try:
            assert holder.stdout.readline().strip() == "locked", errors_path.read_text()
            with pytest.raises(metrics.MetricsLocked):
                metrics.add(make_row(), path=path, lock_timeout=0.3)
            with pytest.raises(metrics.MetricsLocked):
                metrics.load(path, lock_timeout=0.3)
            holder.stdin.close()
            assert holder.wait(timeout=30) == 0, errors_path.read_text()
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
    assert path.read_bytes() == b""
    metrics.add(make_row(), path=path)
    assert len(metrics.load(path)) == 1


def test_add_waits_for_the_lock_instead_of_failing_when_it_frees_in_time(tmp_path):
    path = tmp_path / "m.jsonl"
    errors_path = tmp_path / "children.err"
    outcome = {}

    def writer():
        try:
            outcome["row"] = metrics.add(make_row(), path=path, lock_timeout=30)
        except BaseException as exc:  # handed to the assertion below
            outcome["error"] = exc

    with open(errors_path, "w") as errors:
        holder = start(HOLD_THE_LOCK, path, errors=errors, stdin=subprocess.PIPE)
        try:
            assert holder.stdout.readline().strip() == "locked", errors_path.read_text()
            thread = threading.Thread(target=writer)
            thread.start()
            thread.join(timeout=0.5)
            assert thread.is_alive(), "add returned while another process still held the lock"
            holder.stdin.close()  # the holder lets go; the waiting add must now finish
            thread.join(timeout=30)
            assert not thread.is_alive()
            assert holder.wait(timeout=30) == 0, errors_path.read_text()
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
    assert "error" not in outcome, repr(outcome.get("error"))
    assert [row["metric"] for row in metrics.load(path)] == ["teacher.decode_tok_s"]
    assert outcome["row"] == make_row()


def test_concurrent_writers_lose_no_rows_and_never_interleave(tmp_path):
    path = tmp_path / "m.jsonl"
    go = tmp_path / "go"
    errors_path = tmp_path / "children.err"
    writers, per_writer = 4, 40
    with open(errors_path, "w") as errors:
        children = [start(WRITE_AFTER_GO, path, name, per_writer, go, errors=errors) for name in "abcd"]
        try:
            for child in children:
                assert child.stdout.readline().strip() == "ready", errors_path.read_text()
            go.write_text("go")
            for child in children:
                assert child.wait(timeout=120) == 0, errors_path.read_text()
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
    rows = metrics.load(path)
    assert len(rows) == writers * per_writer
    assert path.read_bytes().count(b"\n") == writers * per_writer
    assert {(row["conditions"]["writer"], row["conditions"]["i"]) for row in rows} == {
        (name, i) for name in "abcd" for i in range(per_writer)}


# --- T10: the idle-baseline rows are what the recorded readings give ------------------------------------------------


def _idle_baseline():
    import importlib.util

    path = ROOT / "fixtures" / "telemetry" / "pod_power" / "idle_baseline.py"
    spec = importlib.util.spec_from_file_location("idle_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.report()


def test_idle_baseline_rows_are_what_the_recorded_readings_give():
    report = _idle_baseline()
    deep = metrics.latest("pod.idle_watts", node="pod-a", conditions={"state": IDLE_DEEP})
    assert deep["value"] == report["pure_idle"]["mean"]
    assert deep["conditions"]["samples"] == report["pure_idle"]["n"]
    assert metrics.latest("pod.idle_watts", node="pod-a", conditions={"state": IDLE_MCLK_UP})["value"] == (
        report["deep_idle_by_mclk"]["1000"]["mean"]
    )
    assert metrics.latest("pod.idle_watts", node="pod-a", conditions={"state": IDLE_TWO_READERS})["value"] == (
        report["reader"]["deep_idle"]["mean"]
    )
    settle = metrics.latest("pod.settle_s", node="pod-a", conditions={"sampler": SAMPLER})
    assert settle["value"] == report["settle"]["median_s"]
    assert settle["conditions"]["events"] == report["settle"]["events"]
    shares = {r["conditions"]["state"].split(" ")[0]: r["value"] for r in metrics.history("pod.idle_state_share", node="pod-a")}
    assert shares == {"deep": report["share_pct"]["deep_idle"], "transition": report["share_pct"]["transition"], "busy": report["share_pct"]["busy"]}
    assert round(sum(shares.values()), 1) == 100.0


def test_idle_baseline_rows_carry_the_sampler_period_samples_and_unknown_error():
    for metric in ("pod.idle_watts", "pod.settle_s", "pod.idle_state_share"):
        for row in metrics.history(metric, node="pod-a"):
            if "under_load_samples" in row["conditions"] or row["conditions"].get("state") == "idle":
                continue
            if "state" not in row["conditions"] and metric == "pod.idle_watts":
                continue
            cond = row["conditions"]
            assert cond["sampler"] == SAMPLER and cond["period_ms"] == 1000 and cond["error"] == "unknown"
            assert row["low_confidence"] is True
            if metric != "pod.settle_s":
                assert cond["samples"] > 0
