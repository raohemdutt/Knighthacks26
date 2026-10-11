import importlib.util
import json
from pathlib import Path

import httpx

STRIX = {"gpu": "Radeon 8060S (Strix Halo), gfx1151"}

spec = importlib.util.spec_from_file_location("slide_numbers", Path("lab/slide_numbers.py"))
slide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slide)


def fake_gateway() -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        routes = {
            "/health": {"ok": True, "model": "atlas-v3-desk-objects"},
            "/memory/stats": {"hours_recorded": 19.5, "episodes": 212, "captions": 640},
            "/api/impact": {"total": {"items_seen": 14, "diversion_rate": 0.71},
                            "before_after_nudges": {"before": {"diversion_rate": 0.4}, "after": {"diversion_rate": 0.8}}},
            "/api/stats": {"detector_fps": 17.2, "student_tok_s": 9.1},
        }
        body = routes.get(request.url.path)
        return httpx.Response(200, json=body) if body else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_real_numbers_with_sources_and_tbd_for_missing(tmp_path):
    before, after = tmp_path / "base.json", tmp_path / "cand.json"
    before.write_text(json.dumps({"score": 0.5, "model": "atlas-base"}))
    after.write_text(json.dumps({"score": 0.83, "model": "atlas-v3"}))
    telemetry = tmp_path / "t.jsonl"
    telemetry.write_text('{"ts": "2026-10-11T03:00:00Z", "jetson_watts": 11.4}\n{"ts": "2026-10-11T04:00:00Z", "jetson_watts": 12.0}\n')

    client = fake_gateway()
    rows = slide.gateway_rows(client, "http://box:8000") + slide.eval_rows(str(before), str(after))
    rows += slide.measured_rows(client, "http://box:8000/api/stats", [str(telemetry)], metrics_file=None,
                                ledger_dir=str(tmp_path / "no-ledger"))
    md = slide.table(rows)
    assert "| Hours of memory | 19.5 h | http://box:8000/memory/stats | read " in md
    assert "71% recycled" in md and "40% -> 80%" in md
    assert "0.50 (atlas-base) -> 0.83 (atlas-v3)" in md
    assert "| Detector FPS (Jetson) | 17.2 | http://box:8000/api/stats | unknown | no |" in md  # a bare number has no time
    assert f"| Watts, Jetson | 12.0 | {telemetry}:2 | 2026-10-11T04:00:00Z | no |" in md  # newest telemetry row wins
    assert "| Teacher tok/s (Strix Halo) | [TBD: R5 stats, T1-T4 telemetry or a lab metrics row not recorded yet] |" in md


def test_without_any_source_everything_is_tbd():
    rows = slide.gateway_rows(httpx.Client(), None) + slide.eval_rows(None, None) + slide.measured_rows(httpx.Client(), None, [], metrics_file=None, ledger_dir="no-such-ledger-dir")
    assert all(r.value is None for r in rows)
    assert "[TBD: pass --gateway]" in slide.table(rows)


def test_lab_numbers_cite_their_row_measurement_time_and_low_confidence(tmp_path):
    """Q15 (telemetry rubric TT7): a flagged number never looks like a measured fact."""
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"metric": "teacher.decode_tok_s", "value": 6.6, "unit": "tok/s", "ts": "2026-10-10T10:22:13Z", "low_confidence": False, "conditions": STRIX},
        {"metric": "teacher.decode_tok_s", "value": 7.7, "unit": "tok/s", "ts": "2026-10-10T10:22:13Z", "low_confidence": False, "conditions": STRIX},
        {"metric": "teacher.joules_per_token", "value": 25.97, "unit": "J/token", "ts": "2026-10-10T10:22:13Z", "low_confidence": True, "conditions": STRIX},
        {"metric": "teacher.watts_avg", "value": 200, "unit": "W", "ts": "2026-10-10T09:00:00Z", "low_confidence": False, "conditions": STRIX},
    ]
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    stats = {"teacher_tok_s": 9.9, "gpu_watts": {"value": 210, "ts": "2026-10-10T11:00:00Z", "source": "R5 stats", "low_confidence": False}}

    def handle(request):
        return httpx.Response(200, json=stats)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    md = slide.table(slide.measured_rows(client, "http://box/api/stats", [], metrics_file=metrics, ledger_dir=str(tmp_path)))
    assert "| Teacher tok/s (Strix Halo) | 7.7 tok/s | metrics.jsonl:2 | 2026-10-10T10:22:13Z | no |" in md  # timed row beats a bare number
    assert "| Teacher joules per token | 25.97 J/token | metrics.jsonl:3 | 2026-10-10T10:22:13Z | LOW |" in md
    assert "| Watts, Strix Halo pod | 210 | R5 stats | 2026-10-10T11:00:00Z | no |" in md  # the endpoint's newer measurement wins


def test_a_stats_figure_served_as_null_with_a_reason_is_not_printed_as_a_value(tmp_path):
    """/runtime/stats (R5) serves a figure nothing has recorded as {value: null, reason}: the lab row fills it, or TBD."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"metric": "teacher.decode_tok_s", "value": 7.7, "unit": "tok/s",
                                   "ts": "2026-10-10T10:22:13Z", "low_confidence": False,
                                   "conditions": STRIX}) + "\n")
    stats = {
        "teacher_tok_s": {"value": None, "unit": "tok/s", "ts": None, "source": None, "low_confidence": False,
                          "reason": "not recorded yet: telemetry is off on this box"},
        "detector_fps": {"value": None, "unit": "frames/s", "ts": None, "source": None, "low_confidence": False,
                         "reason": "not recorded yet: no detector.fps row for node jetson"},
    }

    def handle(request):
        return httpx.Response(200, json=stats)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    md = slide.table(slide.measured_rows(client, "http://box/runtime/stats", [], metrics_file=metrics,
                                        ledger_dir=str(tmp_path / "no-ledger")))
    assert "| Teacher tok/s (Strix Halo) | 7.7 tok/s | metrics.jsonl:1 | 2026-10-10T10:22:13Z | no |" in md
    assert "| Detector FPS (Jetson) | [TBD: R5 stats, T1-T4 telemetry or a lab metrics row not recorded yet] |" in md
    assert "reason" not in md


def test_an_old_chip_row_only_fills_a_gap_and_says_so(tmp_path):
    """The lab moved from the R9700s to the Strix Halo pods on Sat 4:25 PM ET: an R9700 number never passes as current."""
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"metric": "teacher.decode_tok_s", "value": 7.7, "unit": "tok/s", "ts": "2026-10-10T10:22:13Z"},  # untagged, before the move
        {"metric": "teacher.decode_tok_s", "value": 2.82, "unit": "tok/s", "ts": "2026-10-10T19:59:08Z", "conditions": STRIX},
        {"metric": "teacher.decode_tok_s", "value": 8.1, "unit": "tok/s", "ts": "2026-10-10T23:00:00Z",
         "conditions": {"gpu": "Radeon AI PRO R9700, gfx1201"}},  # newer, but the old chip
        {"metric": "trainer.samples_per_s", "value": 0.51, "unit": "samples/s", "ts": "2026-10-10T10:23:23Z"},
    ]
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    md = slide.table(slide.measured_rows(httpx.Client(), None, [], metrics_file=metrics, ledger_dir=str(tmp_path / "no-ledger")))
    assert "| Teacher tok/s (Strix Halo) | 2.82 tok/s | metrics.jsonl:2 | 2026-10-10T19:59:08Z | no |" in md
    assert "| Trainer samples/s (Strix Halo) | 0.51 samples/s | metrics.jsonl:4 (R9700, not the current lab) | 2026-10-10T10:23:23Z | LOW |" in md
