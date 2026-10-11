"""The "Speed and energy" tab's read model: every value keeps its row, nothing is invented, and new rows show up."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_bench import METRICS_FILE, NO_REASON, bench, read_rows, sweeps
from contexts.learning.lab_bench_api import lab_bench_router

ROWS, SKIPPED = read_rows(METRICS_FILE)
RAW = {n: json.loads(line) for n, line in enumerate(METRICS_FILE.read_text(encoding="utf-8").splitlines(), 1) if line.strip()}


def row(metric, value, node="pod-b", unit="tok/s", ts="2026-10-11T03:00:00Z", low=False, **conditions):
    return {"metric": metric, "value": value, "unit": unit, "node": node, "conditions": conditions,
            "source": "a test", "ts": ts, "low_confidence": low}


def write(tmp_path, rows, extra_lines=()):
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join([*(json.dumps(r) for r in rows), *extra_lines]) + "\n", encoding="utf-8")
    return path


def every_point(payload):
    for s in payload["series"]:
        yield from s["points"]


def test_every_row_of_the_committed_file_is_served_once_with_its_own_line_value_and_flag():
    payload = bench(ROWS, SKIPPED)
    points = list(every_point(payload))
    assert payload["rows"] == len(RAW) == len(points) and payload["skipped"] == []
    assert sorted(p["line"] for p in points) == sorted(RAW)
    for p in points:
        raw = RAW[p["line"]]
        assert (p["value"], p["unit"], p["metric"], p["node"], p["ts"]) == (raw["value"], raw["unit"], raw["metric"], raw["node"], raw["ts"])
        assert p["source"] == {"kind": "file", "ref": f"lab/refine/metrics.jsonl:{p['line']}", "ids": []}
        assert p["low_confidence"] is raw["low_confidence"]
        # a low-confidence value says why in the row's own words; a confident one carries no reason
        assert p["reason"] == (raw["conditions"].get("low_confidence_reason", NO_REASON) if raw["low_confidence"] else None)


def test_the_batching_sweep_is_the_recorded_one_point_per_batch_size_all_flagged_low_confidence():
    # R57's A100 reference and E29's llama.cpp run on pod B are sweeps of their own
    (sweep,) = [s for s in bench(ROWS)["sweeps"] if not s["other_chip"] and s["family"] == "teacher_batching"]
    assert (sweep["family"], sweep["node"], sweep["batches"]) == ("teacher_batching", "pod-a", [1, 4, 8, 16, 64])
    assert set(sweep["metrics"]) == {"decode_tok_s", "decode_tok_s_total", "joules_per_token", "ttft_s", "watts_avg", "watts_peak"}
    assert sweep["points"] == 30 == sweep["low_confidence"] and sweep["other_metrics"] == []
    per_stream = [p["value"] for p in sweep["metrics"]["decode_tok_s"]]
    total = [p["value"] for p in sweep["metrics"]["decode_tok_s_total"]]
    joules = [p["value"] for p in sweep["metrics"]["joules_per_token"]]
    assert per_stream == sorted(per_stream, reverse=True) and total == sorted(total) and joules == sorted(joules, reverse=True)
    for name, pts in sweep["metrics"].items():
        assert [p["x"] for p in pts] == [1, 4, 8, 16, 64]
        assert all(p["value"] == RAW[p["line"]]["value"] and p["runs"] == 3 and p["sampler"]["sampler"] for p in pts), name
    # what the rows say about trust, not a guess: three repeats, one session, under the five a verdict needs
    assert "median of 3 repeats" in sweep["confidence_note"] and "Every point is marked low confidence" in sweep["confidence_note"]


def test_the_student_reference_is_the_jetson_rows_and_its_joules_per_token_is_marked_derived_with_the_formula():
    student = bench(ROWS)["student"]
    watts, tok = student["watts"], student["tok_s"]
    raw = RAW[watts["line"]]
    assert raw["metric"] == "student.watts_avg" and raw["node"] == "jetson" and watts["value"] == raw["value"]
    assert tok["value"] == raw["conditions"]["decode_tok_s_median"] and "conditions.decode_tok_s_median" in tok["source"]["ref"]
    j = student["j_per_token"]
    assert j["kind"] == "derived" and j["source"]["kind"] == "derived" and j["low_confidence"] is True
    assert j["value"] == round(watts["value"] / tok["value"], 4)
    assert j["formula"] == f"{watts['value']:g} W ÷ {tok['value']:g} tok/s"
    whole = student["j_per_token_request"]
    assert whole["value"] == round(watts["value"] * raw["conditions"]["request_mean_s"] / 80, 4)
    assert "assumes every answer ran to its 80-token limit" in whole["reason"]


def test_power_is_the_newest_idle_and_serving_row_per_node_and_a_node_without_a_row_is_absent_not_zero():
    power = bench(ROWS)["power"]
    by_node = {n["node"]: n for n in power["nodes"]}
    # pod-b has serving watts since E29's llama.cpp rows and no idle row of its own; nothing is absent
    assert {"jetson", "pod-a", "pod-b"} <= set(by_node) and power["absent"] == [] and by_node["pod-b"]["idle"] is None
    for node in by_node.values():
        for key, suffix in (("idle", ".idle_watts"), ("serving", ".watts_avg")):
            if node[key]:
                mine = [r for r in ROWS if r["node"] == node["node"] and r["metric"].endswith(suffix)]
                assert node[key]["ts"] == max(r["ts"] for r in mine) and node[key]["value"] == RAW[node[key]["line"]]["value"]


def test_rows_from_the_r9700s_are_marked_and_never_put_on_the_strix_halo_curve_or_bars(tmp_path):
    payload = bench(ROWS)
    old = [p for p in every_point(payload) if p["other_chip"]]
    assert payload["other_chip_rows"] == len(old) > 0 and all(p["node"] != "jetson" and p["chip"] for p in old)
    assert {p["line"] for p in old} >= {1, 7, 12, 13, 17}  # the batch-1 teacher rows, 200 W, the smoke run, the 39 W idle
    assert not any(p["other_chip"] for s in payload["sweeps"] if not s["other_chip"] for pts in s["metrics"].values() for p in pts)
    assert not any(n[k]["other_chip"] for n in payload["power"]["nodes"] for k in ("idle", "serving", "peak") if n[k])
    assert {p["line"] for p in payload["power"]["other_chip"]} >= {7, 8, 17}
    # an R9700 sweep stays a sweep of its own, after the current lab's, even on the same node and metric family
    tagged = [row("teacher_batching.decode_tok_s", v, node="pod-a", batch_size=b, gpu="Radeon R9700 (gfx1201)", ts="2026-10-11T05:00:00Z") for b, v in ((1, 7.7), (4, 7.0))]
    both = bench(*read_rows(write(tmp_path, [*({k: v for k, v in r.items() if k != "_line"} for r in ROWS), *tagged])))
    assert [(s["node"], s["other_chip"], s["batches"]) for s in both["sweeps"] if s["node"] == "pod-a"] == [("pod-a", False, [1, 4, 8, 16, 64]), ("pod-a", True, [1, 4])]


def test_nothing_is_dropped_silently_every_row_is_charted_or_listed_as_not_charted():
    payload = bench(ROWS)
    listed = {line for x in payload["not_charted"] for line in x["lines"]}
    assert payload["charted_rows"] + len(listed) == payload["rows"]
    assert {(x["metric"], x["node"]) for x in payload["not_charted"]} >= {("proxy.status", "pod-b"), ("teacher.decode_tok_s", "pod-a")}
    assert sum(f["rows"] for f in payload["families"]) == payload["rows"]


def test_a_new_node_and_a_new_metric_family_show_up_without_a_code_change(tmp_path):
    new = [row("teacher_batching.decode_tok_s", v, batch_size=b, session="bench-r20", model="Qwen3.8-27B-INT4") for b, v in ((1, 3.0), (4, 2.8), (16, 2.2))]
    new += [row("teacher_batching.joules_per_token", v, unit="J/token", batch_size=b, session="bench-r20") for b, v in ((1, 30.0), (4, 9.0), (16, 3.0))]
    new += [row("kernel_hybrid.decode_tok_s", 9.5, batch_size=1, session="bench-k"), row("kernel_hybrid.decode_tok_s", 8.0, batch_size=8, session="bench-k"),
            row("quant.perplexity", 6.1, unit="ppl"), row("pod.idle_watts", 11.0, unit="W", ts="2026-10-11T04:00:00Z")]
    path = write(tmp_path, [*({k: v for k, v in r.items() if k != "_line"} for r in ROWS), *new], extra_lines=["{not json", '{"metric": "x.y", "value": true}'])
    rows, skipped = read_rows(path)
    payload = bench(rows, skipped)
    assert [s["problem"] for s in payload["skipped"]] == ["not JSON", "no metric name or no numeric value"]
    ids = {s["id"] for s in payload["sweeps"]}
    assert {"teacher_batching:pod-b:bench-r20", "kernel_hybrid:pod-b:bench-k"} <= ids and len(ids) == 6  # with R57's A100 reference, R20's pod-b vLLM matrix and E29's llama.cpp sweep
    assert payload["sweeps"][0]["ts"] == max(x["ts"] for x in payload["sweeps"])  # newest first (E29's, 05:10)
    assert next(n for n in payload["power"]["nodes"] if n["node"] == "pod-b")["idle"]["value"] == 11.0 and payload["power"]["absent"] == []
    assert ("quant.perplexity", "pod-b") in {(x["metric"], x["node"]) for x in payload["not_charted"]}
    assert {"kernel_hybrid", "quant"} <= {f["family"] for f in payload["families"]}


def test_a_missing_metric_is_absent_and_an_empty_file_says_so(tmp_path):
    payload = bench(*read_rows(write(tmp_path, [row("proxy.status", 200, unit="http")])))
    assert payload["sweeps"] == [] and payload["student"] is None and payload["power"]["nodes"] == []
    assert sweeps([row("teacher_batching.decode_tok_s", 2.0, batch_size=1) | {"_line": 1}]) == []  # one level is not a sweep
    empty = bench(*read_rows(tmp_path / "nothing.jsonl"))
    assert empty["rows"] == 0 and empty["empty_reason"] and empty["series"] == []


def test_the_endpoint_serves_the_file_and_rereads_it_when_it_changes(tmp_path):
    path = write(tmp_path, [row("proxy.status", 200, unit="http")])
    app = FastAPI()
    app.include_router(lab_bench_router(metrics_file=path, state_dir=tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/learning/lab/bench").json()["rows"] == 1
        path.write_text(path.read_text() + json.dumps(row("proxy.status", 500, unit="http")) + "\n")
        assert client.get("/api/learning/lab/bench").json()["rows"] == 2
        real = FastAPI()
        real.include_router(lab_bench_router())
        with TestClient(real) as committed:
            body = committed.get("/api/learning/lab/bench").json()
            assert body["rows"] == len(RAW) and {s["node"] for s in body["sweeps"]} >= {"pod-a", "pod-b"}


def test_each_row_is_named_for_the_chip_it_was_measured_on_and_only_the_current_lab_is_on_its_curve(tmp_path):
    def run(node, session, ts, tok, **conditions):  # a batching run: tokens a second and joules per token at batch 1 and 4
        return [row(f"teacher_batching.{name}", v, node=node, unit=unit, ts=ts, batch_size=b, session=session, model="Qwen3.8-27B-INT4", **conditions)
                for b, scale in ((1, 1.0), (4, 0.9)) for name, unit, v in (("decode_tok_s", "tok/s", tok * scale), ("joules_per_token", "J/token", 90 / tok / b))]

    after, before = "2026-10-11T03:00:00Z", "2026-10-10T12:00:00Z"
    rows = [*run("pod-a", "s-strix", after, 2.7, gpu="Radeon 8060S (gfx1151)", hub="sjc"),
            *run("pod-a", "s-untagged", after, 2.6),                                          # no chip named, after the move
            *run("pod-a", "s-r9700-tag", after, 7.7, gpu="Radeon R9700 (gfx1201)"),           # tagged, even after the move
            *run("pod-b", "s-r9700-date", before, 7.5),                                       # no chip named, before the move
            *run("a100", "s-a100", after, 65.0, gpu="NVIDIA A100-SXM4-40GB, 400 W power limit", hub="colab", other_chip=True),
            *run("rig-7", "s-future", after, 120.0, gpu="NVIDIA H200-NVL 141GB"),             # a chip nobody has written a rule for
            *run("pod-a", "s-unnamed", after, 5.0, other_chip=True),                          # says only that it is another chip
            row("pod.idle_watts", 46.0, node="a100", unit="W", ts=after, gpu="NVIDIA A100-SXM4-40GB", hub="colab"),
            row("student.watts_avg", 14.0, node="jetson", unit="W", ts=before)]
    payload = bench(*read_rows(write(tmp_path, rows)))

    by_session = {s["session"]: s for s in payload["sweeps"]}
    assert len(by_session) == 7 == len({s["id"] for s in payload["sweeps"]})  # one sweep per run: no two chips share a curve
    chip = {name: s["chip"] for name, s in by_session.items()}
    assert {name: (c["key"], c["name"], c["where"], c["role"], c["current"]) for name, c in chip.items()} == {
        "s-strix": ("strix-halo", "Radeon 8060S (Strix Halo)", "SJC", "current", True),
        "s-untagged": ("strix-halo", "Radeon 8060S (Strix Halo)", "pod-a", "current", True),
        "s-r9700-tag": ("r9700", "Radeon R9700", "pod-a", "earlier", False),
        "s-r9700-date": ("r9700", "Radeon R9700", "pod-b", "earlier", False),
        "s-a100": ("nvidia-a100", "NVIDIA A100", "Colab", "reference", False),
        "s-future": ("nvidia-h200", "NVIDIA H200", "rig-7", "reference", False),
        "s-unnamed": ("unnamed", "Another chip, not named", "pod-a", "unnamed", False),
    }
    # each says how it was decided: by the row's own words, or by the date when the row says nothing
    assert "conditions.gpu" in chip["s-strix"]["reason"] and "gfx1201" in chip["s-r9700-tag"]["reason"] and "A100-SXM4" in chip["s-a100"]["reason"]
    assert "names no chip and was measured before" in chip["s-r9700-date"]["reason"]
    assert "names no chip and was measured after" in chip["s-untagged"]["reason"]
    assert "marks itself other_chip" in chip["s-unnamed"]["reason"]

    # the old boolean is kept, on sweeps and on every point, and means "not the current lab"
    points = list(every_point(payload))
    assert all(p["other_chip"] is (not p["chip"]["current"]) for p in points) and len(points) == len(rows)
    assert all(s["other_chip"] is (not s["chip"]["current"]) for s in payload["sweeps"])
    assert payload["other_chip_rows"] == 4 * 5 + 1 and [s["other_chip"] for s in payload["sweeps"]] == [False, False, True, True, True, True, True]
    assert next(p for p in points if p["node"] == "jetson")["chip"]["key"] == "jetson"

    # no point from another chip is in a current sweep, and no current point in another chip's
    for s in payload["sweeps"]:
        assert {p["chip"]["key"] for pts in s["metrics"].values() for p in pts} == {s["chip"]["key"]}, s["id"]

    # a current sweep names the other chips' sweeps that recorded the same metrics at the same batch sizes
    offered = {(c["chip"]["name"], c["chip"]["where"]): c for c in by_session["s-strix"]["comparisons"]}
    assert set(offered) == {("NVIDIA A100", "Colab"), ("NVIDIA H200", "rig-7"), ("Radeon R9700", "pod-a"), ("Radeon R9700", "pod-b"), ("Another chip, not named", "pod-a")}
    a100 = offered[("NVIDIA A100", "Colab")]
    assert (a100["id"], a100["metrics"], a100["batches"], a100["same_model"]) == (by_session["s-a100"]["id"], ["decode_tok_s", "joules_per_token"], [1, 4], True)
    assert all(s["comparisons"] == [] for s in payload["sweeps"] if s["other_chip"])

    # the power bars are the current lab's; the A100's idle row is listed under its own name, not as an R9700
    assert [n["node"] for n in payload["power"]["nodes"]] == ["jetson"]
    assert [(g["name"], g["where"], g["rows"]) for g in payload["power"]["other_chips"]] == [("NVIDIA A100", "Colab", 1)]
    assert {(g["key"], g["where"]): g["rows"] for g in payload["chips"]} == {
        ("strix-halo", None): 8, ("r9700", None): 8, ("nvidia-a100", "Colab"): 5, ("nvidia-h200", "rig-7"): 4, ("unnamed", None): 4}


def test_the_committed_a100_reference_is_a_reference_on_colab_not_an_r9700_and_the_smoke_run_is_an_r9700():
    app = FastAPI()
    app.include_router(lab_bench_router())
    with TestClient(app) as client:
        payload = client.get("/api/learning/lab/bench").json()
    current = next(s for s in payload["sweeps"] if s["family"] == "teacher_batching")
    (reference,) = [s for s in payload["sweeps"] if s["other_chip"]]
    assert (current["chip"]["key"], current["other_chip"]) == ("strix-halo", False)
    assert (reference["node"], reference["chip"]["name"], reference["chip"]["where"], reference["chip"]["role"], reference["other_chip"]) == ("a100", "NVIDIA A100", "Colab", "reference", True)
    assert [c["id"] for c in current["comparisons"]] == [reference["id"]] and current["comparisons"][0]["batches"] == [1, 4, 8, 16, 64]
    assert "median of 3 spawns" in reference["confidence_note"] and all(p["runs"] == 3 for p in reference["metrics"]["decode_tok_s"])
    assert reference["chip_note"] == RAW[99]["conditions"]["other_chip_note"] and "nvidia-smi" in reference["power_source"]
    names = {p["line"]: p["chip"]["name"] for p in every_point(payload)}
    assert {names[n] for n in range(99, 130)} == {"NVIDIA A100"} and {names[n] for n in (1, 7, 13, 17)} == {"Radeon R9700"}
    assert all(p["other_chip"] is True for p in every_point(payload) if p["line"] in (1, 7, 13, 17, 99, 129))
    assert [(g["name"], g["lines"]) for g in payload["power"]["other_chips"] if g["role"] == "earlier"] == [("Radeon R9700", [7, 8, 17])]
