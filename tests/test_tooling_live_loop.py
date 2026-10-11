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
