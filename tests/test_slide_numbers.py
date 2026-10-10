import importlib.util
import json
from pathlib import Path

import httpx

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
    rows += slide.measured_rows(client, "http://box:8000/api/stats", [str(telemetry)])
    md = slide.table(rows)
    assert "| Hours of memory | 19.5 h | http://box:8000/memory/stats |" in md
    assert "71% recycled" in md and "40% -> 80%" in md
    assert "0.50 (atlas-base) -> 0.83 (atlas-v3)" in md
    assert "| Detector FPS (Jetson) | 17.2 | http://box:8000/api/stats |" in md
    assert "| Watts, Jetson | 12.0 |" in md and "2026-10-11T04:00:00Z" in md  # newest telemetry wins
    assert "| Teacher tok/s (R9700) | [TBD: R5 stats or T1-T4 telemetry not recorded yet] |" in md


def test_without_any_source_everything_is_tbd():
    rows = slide.gateway_rows(httpx.Client(), None) + slide.eval_rows(None, None) + slide.measured_rows(httpx.Client(), None, [])
    assert all(r.value is None for r in rows)
    assert "[TBD: pass --gateway]" in slide.table(rows)
