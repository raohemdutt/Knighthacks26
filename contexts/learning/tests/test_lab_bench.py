"""The "Speed and energy" tab's read model: every value keeps its row, nothing is invented, and new rows show up.

The lab keeps appending to lab/refine/metrics.jsonl and relabelling its rows, so every test of behaviour here reads a
small file it writes itself (contexts/learning/tests/lab_rows.py). The committed file is read only by the tests named
`..._committed_...`, which hold it to what is true of any lab file (`check_bench`) and pin no count, node or value."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_bench import METRICS_FILE, NO_REASON, bench, read_rows, repeats, sweeps
from contexts.learning.lab_bench_api import lab_bench_router
from contexts.learning.tests.lab_rows import (
    A100_NOTE,
    BATCHES,
    POD_A_CURVE,
    POD_A_SESSION,
    check_bench,
    every_point,
    pod_a_sweep,
    raw_lines,
    recorded,
    row,
    sweep_rows,
    write,
)

SWEEP_LINES, A100_LINES = range(12, 42), range(43, 74)  # where recorded() puts the pod-a sweep and the A100's rows


def served(tmp_path, rows, extra_lines=()):
    """The payload for a file holding `rows`, and the file's lines; held to the invariants every payload must keep."""
    path = write(tmp_path, rows, extra_lines)
    payload, raw = bench(*read_rows(path)), raw_lines(path)
    check_bench(payload, raw)
    return payload, raw


def test_every_row_of_the_file_is_served_once_with_its_own_line_value_and_flag(tmp_path):
    payload, raw = served(tmp_path, recorded())
    points = list(every_point(payload))
    assert payload["rows"] == len(raw) == len(points) == 73 and payload["skipped"] == []
    assert sorted(p["line"] for p in points) == sorted(raw)
    for p in points:
        r = raw[p["line"]]
        assert (p["value"], p["unit"], p["metric"], p["node"], p["ts"]) == (r["value"], r["unit"], r["metric"], r["node"], r["ts"])
        assert p["source"] == {"kind": "file", "ref": f"lab/refine/metrics.jsonl:{p['line']}", "ids": []}
        assert p["low_confidence"] is r["low_confidence"]
        # a low-confidence value says why in the row's own words; a confident one carries no reason
        assert p["reason"] == (r["conditions"].get("low_confidence_reason", NO_REASON) if r["low_confidence"] else None)
    assert next(p for p in points if p["line"] == 5)["reason"] == NO_REASON


def test_the_committed_file_is_served_whole_and_keeps_every_invariant_whatever_the_lab_has_added():
    rows, skipped = read_rows(METRICS_FILE)
    payload = bench(rows, skipped)
    check_bench(payload, raw_lines(METRICS_FILE))
    assert payload["rows"] == len(rows) and (payload["empty_reason"] is None) is bool(rows)


def test_the_batching_sweep_is_the_recorded_one_point_per_batch_size_all_flagged_low_confidence(tmp_path):
    payload, raw = served(tmp_path, recorded())
    (sweep,) = [s for s in payload["sweeps"] if not s["other_chip"]]  # the A100 reference is a sweep of its own
    assert (sweep["family"], sweep["node"], sweep["batches"]) == ("teacher_batching", "pod-a", [1, 4, 8, 16, 64])
    assert set(sweep["metrics"]) == {"decode_tok_s", "decode_tok_s_total", "joules_per_token", "ttft_s", "watts_avg", "watts_peak"}
    assert sweep["points"] == 30 == sweep["low_confidence"] and sweep["other_metrics"] == []
    assert (sweep["superseded"], sweep["repeated"]) == (0, 0)
    per_stream = [p["value"] for p in sweep["metrics"]["decode_tok_s"]]
    total = [p["value"] for p in sweep["metrics"]["decode_tok_s_total"]]
    joules = [p["value"] for p in sweep["metrics"]["joules_per_token"]]
    assert per_stream == sorted(per_stream, reverse=True) and total == sorted(total) and joules == sorted(joules, reverse=True)
    for name, pts in sweep["metrics"].items():
        assert [p["x"] for p in pts] == [1, 4, 8, 16, 64]
        assert all(p["value"] == raw[p["line"]]["value"] and p["runs"] == 3 and p["sampler"]["sampler"] for p in pts), name
        assert {p["line"] for p in pts} <= set(SWEEP_LINES)
    # what the rows say about trust, not a guess: three repeats, one session, under the five a verdict needs
    assert "median of 3 repeats" in sweep["confidence_note"] and "Every point is marked low confidence" in sweep["confidence_note"]


def test_a_sweep_on_each_pod_is_a_curve_of_its_own_and_the_newest_is_the_one_the_screen_opens_on(tmp_path):
    # the trainer's pod ran the same matrix later, under the plain `teacher` family, with fewer spawns behind its watts
    later = pod_a_sweep(session="pod-b-night", node="pod-b", family="teacher", ts="2026-10-11T04:13:45Z")
    for r in later:
        n = 3 if r["metric"].endswith((".decode_tok_s", ".ttft_s")) and r["conditions"]["batch_size"] == 1 else 2
        r["conditions"].update(statistic=f"median of {n} spawns", low_confidence_reason=f"{n} spawn(s), under the 5 that verdict() needs to call a difference")
        r["conditions"]["stats"]["n"] = n
    payload, raw = served(tmp_path, [*recorded(), *later])
    current = [s for s in payload["sweeps"] if not s["other_chip"]]
    assert [(s["node"], s["family"], s["session"]) for s in current] == [("pod-b", "teacher", "pod-b-night"), ("pod-a", "teacher_batching", POD_A_SESSION)]
    assert payload["sweeps"][0] is current[0]  # the newest sweep on the current lab comes first: the one the tab opens on
    b, a = current
    assert {p["node"] for pts in b["metrics"].values() for p in pts} == {"pod-b"} and {p["node"] for pts in a["metrics"].values() for p in pts} == {"pod-a"}
    assert {p["line"] for pts in a["metrics"].values() for p in pts} == set(SWEEP_LINES) and a["points"] == b["points"] == 30
    assert not {p["line"] for pts in b["metrics"].values() for p in pts} & set(SWEEP_LINES)  # no row is on both curves
    assert b["chip"]["key"] == a["chip"]["key"] == "strix-halo" and (b["chip"]["where"], a["chip"]["where"]) == ("SJC", "SJC")
    reference = next(s for s in payload["sweeps"] if s["other_chip"])
    assert [c["id"] for c in a["comparisons"]] == [c["id"] for c in b["comparisons"]] == [reference["id"]]
    # the note says what the points are, and says so when they are not all the same statistic
    assert b["confidence_note"].startswith("28 of 30 points are the median of 2 spawns; the others are the median of 3 spawns, all from one session (pod-b-night).")
    assert "for 2 different reasons" in b["confidence_note"]
    # the pod-b bars are its own rows, not pod-a's
    by_node = {n["node"]: n for n in payload["power"]["nodes"]}
    assert raw[by_node["pod-b"]["serving"]["line"]]["node"] == "pod-b" and raw[by_node["pod-a"]["serving"]["line"]]["node"] == "pod-a"
    assert by_node["pod-b"]["idle"] is None and payload["power"]["absent"] == []  # no idle row for pod-b: nothing drawn, not zero


def test_a_row_the_file_holds_twice_is_listed_twice_but_is_one_reading(tmp_path):
    # a merge left a second copy of the sweep and of the idle row; the copies lost the fields a backfill had added
    base = recorded()
    for r in base[11:42]:
        r["conditions"].update(error="unknown", backfilled=["error"])
    copies = [{**r, "conditions": {k: v for k, v in r["conditions"].items() if k not in ("error", "backfilled")}} for r in base[11:42]]
    fresh = row("teacher_batching.joules_per_token", 30.0, node="pod-a", unit="J/token", ts="2026-10-11T06:00:00Z", batch_size=1,
                session=POD_A_SESSION, chip="gfx1151")                                       # measured again: a new reading
    other = {**base[11], "conditions": {**base[11]["conditions"], "sampler": "another sampler"}}  # same value, another condition
    payload, raw = served(tmp_path, [*base, *copies, fresh, other])
    first_copy = len(base) + 1
    assert payload["rows"] == len(raw) == len(base) + 33 and payload["repeated_rows"] == 31  # every line is still served
    assert payload["repeats"][0] == {"line": first_copy, "of": 12} and payload["repeats"][-1] == {"line": first_copy + 30, "of": 42}
    assert next(p for p in every_point(payload) if p["line"] == first_copy)["repeat_of"] == 12
    assert next(p for p in every_point(payload) if p["line"] == len(raw))["repeat_of"] is None  # a differing condition: not a repeat
    (sweep,) = [s for s in payload["sweeps"] if not s["other_chip"]]
    assert sweep["points"] == 30 and sweep["repeated"] == 30  # one point per batch size, as before the copy
    assert sweep["superseded"] == 2  # the re-measured batch-1 energy and the row with another sampler; no copy is counted
    assert sweep["metrics"]["joules_per_token"][0]["value"] == 30.0
    rows, _ = read_rows(write(tmp_path, [*base, *copies], name="again.jsonl"))
    assert repeats(rows) == {len(base) + i: 12 + i - 1 for i in range(1, 32)}
    assert repeats([{**r, "ts": None} for r in rows]) == {}  # a row with no time is never called a repeat


def test_the_student_reference_is_the_jetson_rows_and_its_joules_per_token_is_marked_derived_with_the_formula(tmp_path):
    payload, raw = served(tmp_path, recorded())
    student = payload["student"]
    watts, tok = student["watts"], student["tok_s"]
    r = raw[watts["line"]]
    assert r["metric"] == "student.watts_avg" and r["node"] == "jetson" and watts["value"] == r["value"] == 17.536
    assert tok["value"] == r["conditions"]["decode_tok_s_median"] and "conditions.decode_tok_s_median" in tok["source"]["ref"]
    j = student["j_per_token"]
    assert j["kind"] == "derived" and j["source"]["kind"] == "derived" and j["low_confidence"] is True
    assert j["value"] == round(watts["value"] / tok["value"], 4)
    assert j["formula"] == f"{watts['value']:g} W ÷ {tok['value']:g} tok/s"
    whole = student["j_per_token_request"]
    assert whole["value"] == round(watts["value"] * r["conditions"]["request_mean_s"] / 80, 4)
    assert "assumes every answer ran to its 80-token limit" in whole["reason"]


def test_power_is_the_newest_idle_and_serving_row_per_node_and_a_node_without_a_row_is_absent_not_zero(tmp_path):
    payload, raw = served(tmp_path, recorded())
    power = payload["power"]
    by_node = {n["node"]: n for n in power["nodes"]}
    assert set(by_node) == {"jetson", "pod-a"} and power["absent"] == ["pod-b"]
    rows = [r for r in raw.values() if "A100" not in r["conditions"].get("gpu", "")]
    for node in by_node.values():
        for key, suffix in (("idle", ".idle_watts"), ("serving", ".watts_avg")):
            mine = [r for r in rows if r["node"] == node["node"] and r["metric"].endswith(suffix)]
            assert node[key]["ts"] == max(r["ts"] for r in mine) and node[key]["value"] == raw[node[key]["line"]]["value"]
    assert (by_node["pod-a"]["idle"]["value"], by_node["pod-a"]["idle"]["line"]) == (10.44, 42)  # not the 39 W of the R9700s, nor the older 10.2
    assert by_node["jetson"]["idle"]["value"] == 12.62 and by_node["jetson"]["serving"]["value"] == 17.536


def test_rows_from_the_r9700s_are_marked_and_never_put_on_the_strix_halo_curve_or_bars(tmp_path):
    payload, _ = served(tmp_path, recorded())
    old = [p for p in every_point(payload) if p["other_chip"]]
    assert payload["other_chip_rows"] == len(old) > 0 and all(p["node"] != "jetson" and p["chip"] for p in old)
    assert {p["line"] for p in old} == {1, 2, 3, 4, 5, *A100_LINES}  # the batch-1 teacher row, 200 W, the smoke run, the 39 W idle
    assert not any(p["other_chip"] for s in payload["sweeps"] if not s["other_chip"] for pts in s["metrics"].values() for p in pts)
    assert not any(n[k]["other_chip"] for n in payload["power"]["nodes"] for k in ("idle", "serving", "peak") if n[k])
    assert {p["line"] for p in payload["power"]["other_chip"]} >= {2, 3, 5}
    # an R9700 sweep stays a sweep of its own, after the current lab's, even on the same node and metric family
    tagged = [row("teacher_batching.decode_tok_s", v, node="pod-a", batch_size=b, gpu="Radeon R9700 (gfx1201)", ts="2026-10-11T05:00:00Z") for b, v in ((1, 7.7), (4, 7.0))]
    both, _ = served(tmp_path, [*recorded(), *tagged])
    assert [(s["node"], s["other_chip"], s["batches"]) for s in both["sweeps"] if s["node"] == "pod-a"] == [("pod-a", False, [1, 4, 8, 16, 64]), ("pod-a", True, [1, 4])]


def test_nothing_is_dropped_silently_every_row_is_charted_or_listed_as_not_charted(tmp_path):
    payload, _ = served(tmp_path, recorded())
    listed = {line for x in payload["not_charted"] for line in x["lines"]}
    assert payload["charted_rows"] + len(listed) == payload["rows"]
    assert {(x["metric"], x["node"]) for x in payload["not_charted"]} >= {("proxy.status", "pod-b"), ("teacher.decode_tok_s", "pod-a")}
    assert sum(f["rows"] for f in payload["families"]) == payload["rows"]


def test_a_new_node_and_a_new_metric_family_show_up_without_a_code_change(tmp_path):
    new = [row("teacher_batching.decode_tok_s", v, batch_size=b, session="bench-r20", model="Qwen3.8-27B-INT4") for b, v in ((1, 3.0), (4, 2.8), (16, 2.2))]
    new += [row("teacher_batching.joules_per_token", v, unit="J/token", batch_size=b, session="bench-r20") for b, v in ((1, 30.0), (4, 9.0), (16, 3.0))]
    new += [row("kernel_hybrid.decode_tok_s", 9.5, batch_size=1, session="bench-k"), row("kernel_hybrid.decode_tok_s", 8.0, batch_size=8, session="bench-k"),
            row("quant.perplexity", 6.1, unit="ppl"), row("pod.idle_watts", 11.0, unit="W", ts="2026-10-11T04:00:00Z"),
            row("teacher.decode_tok_s", 2.9, node="pod-e", batch_size=1, session="bench-e"), row("teacher.decode_tok_s", 2.6, node="pod-e", batch_size=4, session="bench-e"),
            row("teacher.watts_avg", 95.0, node="pod-e", unit="W")]
    payload, _ = served(tmp_path, [*recorded(), *new], extra_lines=["{not json", '{"metric": "x.y", "value": true}'])
    assert [s["problem"] for s in payload["skipped"]] == ["not JSON", "no metric name or no numeric value"]
    ids = {s["id"] for s in payload["sweeps"]}
    assert {"teacher_batching:pod-b:bench-r20", "kernel_hybrid:pod-b:bench-k", "teacher:pod-e:bench-e"} <= ids and len(ids) == 5  # with pod-a's and the A100's
    assert payload["sweeps"][0]["ts"] == "2026-10-11T03:00:00Z"  # newest first
    assert next(n for n in payload["power"]["nodes"] if n["node"] == "pod-b")["idle"]["value"] == 11.0 and payload["power"]["absent"] == []
    assert next(n for n in payload["power"]["nodes"] if n["node"] == "pod-e")["serving"]["value"] == 95.0  # a pod nobody named in the code
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
        assert client.get("/api/learning/lab/bench?all=1").json()["rows"] == 1
        path.write_text(path.read_text() + json.dumps(row("proxy.status", 500, unit="http")) + "\n")
        assert client.get("/api/learning/lab/bench?all=1").json()["rows"] == 2
        path.write_text("\n".join(json.dumps(r) for r in recorded()) + "\n")
        body = client.get("/api/learning/lab/bench?all=1").json()
        assert body["rows"] == 73 and body["sweeps"][0]["node"] == "pod-a"


def test_the_endpoint_serves_the_committed_file_by_default():
    app = FastAPI()
    app.include_router(lab_bench_router())
    with TestClient(app) as client:
        body = client.get("/api/learning/lab/bench?all=1").json()
    check_bench(body, raw_lines(METRICS_FILE))
    assert body["file"] == "lab/refine/metrics.jsonl" and body["rows"] == len(read_rows(METRICS_FILE)[0])


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
    payload, _ = served(tmp_path, rows)

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


def test_two_sweeps_never_share_an_id_or_a_curve_when_only_the_pod_the_session_or_the_chip_differs(tmp_path):
    two = {"decode_tok_s": ("tok/s", (2.7, 2.5)), "joules_per_token": ("J/token", (36.0, 10.0))}
    strix, old = {"chip": "gfx1151"}, {"gpu": "Radeon R9700 (gfx1201)"}
    rows = [*sweep_rows("teacher", "pod-a", "night", two, batches=(1, 4), **strix),
            *sweep_rows("teacher", "pod-b", "night", two, batches=(1, 4), **strix),    # the same session name on the other pod
            *sweep_rows("teacher", "pod-a", "night-2", two, batches=(1, 4), **strix),  # another session on the same pod
            *sweep_rows("teacher", "pod-a", "night", two, batches=(1, 4), **old),      # the same pod and session, another chip
            *sweep_rows("teacher_batching", "pod-a", "night", two, batches=(1, 4), **strix)]
    payload, _ = served(tmp_path, rows)
    assert len(payload["sweeps"]) == 5 == len({s["id"] for s in payload["sweeps"]})
    assert all(s["points"] == 4 and s["superseded"] == 0 for s in payload["sweeps"])
    assert sorted((s["family"], s["node"], s["session"], s["chip"]["key"]) for s in payload["sweeps"]) == sorted([
        ("teacher", "pod-a", "night", "strix-halo"), ("teacher", "pod-b", "night", "strix-halo"), ("teacher", "pod-a", "night-2", "strix-halo"),
        ("teacher", "pod-a", "night", "r9700"), ("teacher_batching", "pod-a", "night", "strix-halo")])


def test_the_a100_reference_is_a_reference_on_colab_not_an_r9700_and_the_smoke_run_is_an_r9700(tmp_path):
    path = write(tmp_path, recorded())
    app = FastAPI()
    app.include_router(lab_bench_router(metrics_file=path))
    with TestClient(app) as client:
        payload = client.get("/api/learning/lab/bench?all=1").json()
    check_bench(payload, raw_lines(path))
    current, reference = payload["sweeps"]
    assert (current["chip"]["key"], current["other_chip"]) == ("strix-halo", False)
    assert (reference["node"], reference["chip"]["name"], reference["chip"]["where"], reference["chip"]["role"], reference["other_chip"]) == ("a100", "NVIDIA A100", "Colab", "reference", True)
    assert [c["id"] for c in current["comparisons"]] == [reference["id"]] and current["comparisons"][0]["batches"] == list(BATCHES)
    assert current["comparisons"][0]["metrics"] == [m for m in ("decode_tok_s", "decode_tok_s_total", "joules_per_token", "ttft_s", "watts_avg", "watts_peak") if m in POD_A_CURVE]
    assert "median of 3 spawns" in reference["confidence_note"] and all(p["runs"] == 3 for p in reference["metrics"]["decode_tok_s"])
    assert reference["chip_note"] == A100_NOTE and "nvidia-smi" in reference["power_source"]
    names = {p["line"]: p["chip"]["name"] for p in every_point(payload)}
    assert {names[n] for n in A100_LINES} == {"NVIDIA A100"} and {names[n] for n in (1, 2, 4, 5)} == {"Radeon R9700"}
    assert all(p["other_chip"] is True for p in every_point(payload) if p["line"] in (1, 2, 4, 5, 43, 73))
    assert [(g["name"], g["lines"]) for g in payload["power"]["other_chips"] if g["role"] == "earlier"] == [("Radeon R9700", [2, 3, 5])]


def test_the_default_answer_is_a_small_summary_with_each_sessions_headline_and_the_whole_payload_moved_not_lost(tmp_path):
    path = write(tmp_path, recorded())
    app = FastAPI()
    app.include_router(lab_bench_router(metrics_file=path))
    with TestClient(app) as client:
        body = client.get("/api/learning/lab/bench").json()
        whole = client.get("/api/learning/lab/bench?all=1").json()
        assert client.get("/api/learning/lab/bench/rows?all=1").json() == whole
        assert client.get("/api/learning/lab/bench/rows").status_code == 400
        assert client.get("/api/learning/lab/bench/rows?session=nope").status_code == 404
        pod_a = next(s for s in body["sessions"] if s["id"] == POD_A_SESSION)
        one = client.get(pod_a["rows_url"]).json()
    check_bench(whole, raw_lines(path))
    assert body["summary"] is True and body["rows"] == whole["rows"] == 73
    for key in ("file", "low_confidence_rows", "repeated_rows", "other_chip_rows", "newest", "student", "power", "empty_reason"):
        assert body[key] == whole[key], key
    assert all(body[k] == [] for k in body["omitted"]["keys"]) and whole["sweeps"] and whole["series"]
    assert sum(s["rows"] + s["repeated_rows"] for s in body["sessions"]) == 73
    assert pod_a["chip"]["key"] == "strix-halo" and pod_a["lever"] and pod_a["rows"] > 0
    raw = raw_lines(path)
    for cell in pod_a["headline"]:
        assert cell["batch"] in BATCHES and cell["confidence"]["level"] in ("low", "repeated", "single run")
        for name in ("tok_s_stream", "tok_s_total", "ttft_s", "watts", "j_per_token"):
            if cell[name] is not None:  # every headline number is a row of the file, by its line
                assert raw[cell[name]["line"]]["value"] == cell[name]["value"]
    assert any(c["tok_s_stream"] for c in pod_a["headline"])
    assert one["session"] == POD_A_SESSION and one["rows"] == pod_a["rows"] + pod_a["repeated_rows"]


def test_the_summary_is_built_once_per_version_of_the_file(tmp_path, monkeypatch):
    import contexts.learning.lab_bench_api as api
    calls = []
    real = api.summary
    monkeypatch.setattr(api, "summary", lambda *a: calls.append(1) or real(*a))
    path = write(tmp_path, [row("proxy.status", 200, unit="http")])
    app = FastAPI()
    app.include_router(api.lab_bench_router(metrics_file=path))
    with TestClient(app) as client:
        assert client.get("/api/learning/lab/bench").json()["rows"] == 1
        assert client.get("/api/learning/lab/bench").json()["rows"] == 1
        assert len(calls) == 1
        path.write_text(path.read_text() + json.dumps(row("proxy.status", 500, unit="http")) + chr(10))
        assert client.get("/api/learning/lab/bench").json()["rows"] == 2 and len(calls) == 2
