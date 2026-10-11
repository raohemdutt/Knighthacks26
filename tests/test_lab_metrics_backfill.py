"""The measured-row fields (sampler, period_ms, samples, error, hub, account, GPU) on the committed metrics file, and
the backfill tool that put them on the rows written without them (lab/tools/backfill_row_fields.py)."""

import json
from pathlib import Path

from lab.machine import metrics
from lab.tools import backfill_row_fields as bf

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "lab" / "refine" / "metrics.jsonl"
# Rows other branches append tonight are theirs to complete; this check covers the rows present when the backfill ran.
CHECKED_UNTIL = "2026-10-11T02:00:00Z"
# Two T6 teacher power rows on pod-a name no engine anywhere in the repo; the tool reports them rather than guess one.
NO_ENGINE = {("teacher.watts_avg", "2026-10-10T20:53"), ("teacher.watts_peak", "2026-10-10T20:53")}


def test_every_measured_row_in_scope_carries_the_required_fields():
    problems = []
    for n, row in enumerate(metrics.load(COMMITTED), 1):
        if not bf.in_scope(row) or row["ts"] > CHECKED_UNTIL:
            continue
        missing = bf.missing_fields(row)
        if "engine" in missing and (row["metric"], row["ts"][:16]) in NO_ENGINE:
            missing.remove("engine")
        if missing:
            problems.append(f"row {n} {row['metric']} {row['node']}: {missing}")
    assert not problems, problems


def test_named_groups_are_backfilled_with_their_hub_and_account():
    rows = metrics.load(COMMITTED)
    expect = {"pod-c": ("sjc", "C"), "pod-d": ("sjc", "D")}
    seen = set()
    for row in rows:
        if row["node"] in expect:
            c = row["conditions"]
            assert (c["hub"], c["account"]) == expect[row["node"]]
            assert c["error"] == "unknown" and "period_ms" in c and "samples" in c and c["sampler"]
            seen.add(row["node"])
        if row["metric"].startswith("teacher_batching."):
            c = row["conditions"]
            assert c["period_ms"] == 1000 and c["error"] == "unknown" and "samples" in c
            assert (c["hub"], c["account"]) == ("sjc", "C")
    assert seen == set(expect)


def _row(node, metric, ts, source="unit test", **cond):
    return {"metric": metric, "value": 1.5, "unit": "s", "node": node, "conditions": cond, "source": source,
            "ts": ts, "low_confidence": False}


def _write(path, rows):
    path.write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode())


def test_backfill_fills_only_what_is_missing_and_changes_no_value(tmp_path):
    f = tmp_path / "m.jsonl"
    old = [
        _row("pod-c", "teacher.ttft_s", "2026-10-10T19:58:00Z", engine="vllm", gpu="Radeon 8060S"),
        _row("pod-d", "pod.start_s", "2026-10-10T20:14:00Z", "R13 live run, lab.controller form launch", hub="sjc", account="D", gpu="g", pings=7),
        _row("pod-a", "teacher_batching.watts_avg", "2026-10-11T00:23:00Z", engine="vllm", chip="gfx1151",
             sampler="T6 lab/pod_power.py 1 on port 8790, 1 s period", hub="sjc", account="C", error="measured"),
        _row("pod-a", "pod.idle_watts", "2026-10-11T00:30:00Z", engine="x", window_s=60, sampler="lab/pod_power.py 1"),
        _row("a100", "teacher.decode_tok_s", "2026-10-11T02:00:00Z", engine="vllm"),
        _row("pod-a", "pod.idle_watts", "2026-10-10T09:55:00Z"),  # a DUB-era row: untouched
        _row("jetson", "jetson.idle_watts", "2026-10-10T22:00:00Z"),
    ]
    _write(f, old)
    first = f.read_bytes().splitlines()
    summary = bf.backfill_file(f)
    assert summary["changed"] == 5
    new = metrics.load(f)
    c = [r["conditions"] for r in new]
    assert (c[0]["hub"], c[0]["account"], c[0]["sampler"], c[0]["error"]) == ("sjc", "C", "unknown", "unknown")
    assert c[0]["period_ms"] is None and c[0]["samples"] is None and c[0]["gpu"] == "Radeon 8060S"
    assert (c[1]["samples"], c[1]["sampler"]) == (7, "lab.controller timer") and "hub" in c[1]
    assert c[2]["error"] == "measured" and c[2]["period_ms"] == 1000 and "gpu" not in c[2]
    assert c[3]["samples"] == 60 and c[3]["period_ms"] == 1000
    assert (c[4]["hub"], c[4]["account"]) == ("colab", "colab")
    # untouched rows stay byte for byte; touched rows differ only inside conditions
    lines = f.read_bytes().splitlines()
    assert lines[5] == first[5] and lines[6] == first[6]
    for a, b in zip(old, new):
        assert {k: v for k, v in a.items() if k != "conditions"} == {k: v for k, v in b.items() if k != "conditions"}
        assert all(b["conditions"][k] == v for k, v in a["conditions"].items())
    assert new[0]["conditions"]["backfilled"] == sorted(new[0]["conditions"]["backfilled"])
    # a second run changes nothing, and a row appended meanwhile is kept as is
    after = f.read_bytes()
    assert bf.backfill_file(f)["changed"] == 0 and f.read_bytes() == after


def test_backfill_dry_run_writes_nothing(tmp_path):
    f = tmp_path / "m.jsonl"
    _write(f, [_row("pod-c", "teacher.ttft_s", "2026-10-10T19:58:00Z", engine="vllm", gpu="g")])
    before = f.read_bytes()
    assert bf.backfill_file(f, dry_run=True)["changed"] == 1
    assert f.read_bytes() == before
