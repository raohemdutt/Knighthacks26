"""L29: the live loop relay's stage arithmetic and pod D's energy split (tooling/demo/live_loop.py), on the stage
lines and the pod D trace recorded in tonight's rehearsal (fixtures/learning/live-loop-2026-10-11)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures" / "learning" / "live-loop-2026-10-11"


@pytest.fixture(scope="module")
def live_loop():
    spec = importlib.util.spec_from_file_location("live_loop", ROOT / "tooling" / "demo" / "live_loop.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_loop"] = module
    spec.loader.exec_module(module)
    return module


def stages():
    return [json.loads(x) for x in (FIX / "stages.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def test_stage_seconds_match_the_rehearsal(live_loop):
    solo = live_loop.stage_seconds(stages(), "solo")
    duo = live_loop.stage_seconds(stages(), "duo")
    assert solo == {"labeling": 32.1, "waiting": 0.0, "training": 53.9, "packaging": 71.1, "total": 157.0}
    assert duo["labeling"] == 61.8 and duo["training"] == 68.8 and duo["packaging"] == 59.2


def test_pod_d_split_adds_up_to_the_pods_joules(live_loop):
    trace = live_loop.Trace("pod-d")
    for line in (FIX / "power-pod-d.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        trace.points.append((live_loop.iso_s(row["ts"]), float(row["w"])))
    shared, overlaps = live_loop.pod_d_shares(stages(), trace)
    assert set(shared) == {"solo.training", "solo.packaging", "duo.training", "duo.packaging"}
    first = live_loop.window(stages(), "solo", "train", "trained")[0]
    last = live_loop.window(stages(), "duo", "trained", "done")[1]
    whole = live_loop.integrate(trace.points, first, last)
    assert sum(shared.values()) == pytest.approx(whole, rel=1e-3)   # the overlap is split, never counted twice
    assert overlaps["solo.packaging"] > 0 and overlaps["solo.training"] == 0.0


def test_comparison_fixture_reads_two_to_one(live_loop):
    c = json.loads((FIX / "comparison.json").read_text(encoding="utf-8"))
    assert c["acts"]["solo"]["examples"] == 100 and c["acts"]["duo"]["examples"] == 200
    assert c["total_s"] < 240
    assert c["duo_over_solo"]["examples"] == 2.0 and c["duo_over_solo"]["pods_joules"] < 2.0


# --- what the relay posts to the box (L30, L31): through a fake box that runs the real import and compare routes ---

def fake_box(tmp_path):
    """The box's import, lab and compare routes in this process, behind an httpx transport the relay's Box can use."""
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bus import EventBus
    from contexts.learning.application.lab_compare import default_dirs
    from contexts.learning.application.lab_view import LabView
    from contexts.learning.lab_api import lab_router
    from contexts.learning.lab_import import lab_import_router

    view, bus = LabView(), EventBus(log_dir=tmp_path / "events")
    view.subscribe(bus)
    app = FastAPI()
    app.include_router(lab_import_router(view, bus.publish, data_dir=tmp_path / "data"))
    app.include_router(lab_router(view, sample_dir=None, metrics_file=None, loop_dirs=default_dirs(tmp_path / "data")))
    client = TestClient(app)

    def handle(request: httpx.Request) -> httpx.Response:
        r = client.request(request.method, request.url.raw_path.decode(), content=request.content,
                           headers={"Content-Type": request.headers.get("content-type", "")})
        return httpx.Response(r.status_code, content=r.content, headers={"Content-Type": r.headers.get("content-type", "")})

    return client, httpx.MockTransport(handle)


def recorded_traces(live_loop):
    offsets = json.loads((FIX / "comparison.json").read_text(encoding="utf-8"))["clock_offsets_s"]
    traces = {}
    for node in ("pod-c", "pod-d", "box"):
        tr = live_loop.Trace(node)
        tr.offset = offsets[node]
        for line in (FIX / f"power-{node}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            tr.points.append((live_loop.iso_s(row["ts"]), float(row["w"])))
        traces[node] = tr
    return traces


def test_the_relay_posts_stages_power_and_the_comparison_and_the_box_draws_them(live_loop, tmp_path):
    client, transport = fake_box(tmp_path)
    box = live_loop.Box("http://box:8000", None, transport=transport)
    traces = recorded_traces(live_loop)
    where = {"pod-c": {"hub": "sjc.aupcloud.io", "account": "C"}, "pod-d": {"hub": "sjc.aupcloud.io", "account": "D"}}
    doc = json.loads((FIX / "comparison.json").read_text(encoding="utf-8"))
    go = next(s for s in stages() if s["stage"] == "go")
    doc["t0"] = live_loop.iso_s(go["ts"]) - doc["clock_offsets_s"]["pod-d"] - doc["go_s"]
    for act in ("solo", "duo"):
        run_id = doc["acts"][act]["run_id"]
        lines = live_loop.act_lines(stages(), act)
        assert all(x["task"] == live_loop.TASK for x in lines)
        answer = box.post("stage", "".join(json.dumps(x) + "\n" for x in lines).encode(), run_id, {"task": live_loop.TASK})
        assert answer.ok, answer.answer
        said = live_loop.post_windows(box, stages(), act, run_id, traces, where)
        assert said.count(" 200") == 6, said
    assert live_loop.post_comparison(box, doc).startswith("box 200, kept")
    out = client.get("/api/learning/lab/compare").json()
    assert out["asked"] == {"a": "live-1011-040342-solo", "b": "live-1011-040342-duo"}
    assert out["duo_over_solo"] == doc["duo_over_solo"]
    runs = {r["run_id"]: r for r in out["series"]["runs"]}
    assert runs["live-1011-040342-solo"]["start_s"] == 2.7 and runs["live-1011-040342-duo"]["start_s"] == 35.0
    for node, body in out["series"]["nodes"].items():
        assert len(body["points"]) > 50, node
        assert {w["stage"] for w in body["windows"]} >= {"act"}
    pod_c = out["series"]["nodes"]["pod-c"]["windows"][0]
    assert pod_c["account"] == "C" and pod_c["hub"] == "sjc.aupcloud.io" and 900 <= pod_c["period_ms"] <= 1100
    story = client.get("/api/learning/lab/live-1011-040342-solo").json()
    assert story["run_status"] == "finished" and [s["state"] for s in story["stage_strip"]] == ["done"] * 4


def test_a_power_window_carries_the_samples_either_side_and_the_nodes_clock(live_loop):
    traces = recorded_traces(live_loop)
    wins = dict((f"{stage}.{w['node']}", w) for stage, w in live_loop.power_windows(stages(), "solo", "r", traces, {}))
    assert set(wins) == {"labeling.pod-c", "training.pod-d", "packaging.pod-d", "act.pod-c", "act.pod-d", "act.box"}
    lab = wins["labeling.pod-c"]
    assert lab["clock_offset_s"] == 63.88 and lab["idle_w"] == 10.27 and lab["sampler"].startswith("lab/pod_power.py")
    start, end = live_loop.iso_s(lab["start"]), live_loop.iso_s(lab["end"])
    ts = [live_loop.iso_s(s["ts"]) for s in lab["samples"]]
    assert ts[0] <= start and ts[-1] >= end   # the box can bracket both edges
    assert round(end - start, 1) == 32.1


def test_the_local_card_is_off_unless_asked_for(live_loop, monkeypatch):
    seen = {}
    monkeypatch.setattr(live_loop, "run_loop", lambda args: seen.setdefault("card", args.card) and 0 or 0)
    live_loop.main(["--slot", "row"])
    assert seen["card"] is False
    seen.clear()
    live_loop.main(["--slot", "row", "--card"])
    assert seen["card"] is True
