"""Rows in the shape the lab writes to lab/refine/metrics.jsonl, for tests that check how the Learning lab reads
them. The lab keeps appending to the committed file and relabelling its rows, so a test of behaviour writes these
into its own tmp_path and reads that; only the invariant checks at the bottom are run against the committed file.

`recorded()` is a small lab file with one of everything the screens draw: rows from before the move to the Strix
Halo pods (no chip named), the Jetson student's power run, a batching sweep on pod-a, the pod's idle watts, and an
A100 reference run on Colab. The numbers are the ones the lab recorded on 2026-10-10 and 11, frozen here.
"""

import json
from pathlib import Path
from typing import Any

BEFORE, AFTER = "2026-10-10T10:22:13Z", "2026-10-11T00:23:20Z"  # either side of the move to the Strix Halo pods
BATCHES = (1, 4, 8, 16, 64)
POD_A_SESSION, A100_SESSION = "bench-pod-a", "bench-a100"
A100_GPU = "NVIDIA A100-SXM4-40GB, 400 W power limit"
A100_NOTE = "a reference on an NVIDIA A100, not the lab's chip; the lab's teacher numbers are the Strix Halo rows"
THREE_RUNS = "3 spawn(s), under the 5 that verdict() needs to call a difference"
# metric -> (unit, value at batch 1, 4, 8, 16, 64): the pod-a sweep and the A100 reference as they were recorded
POD_A_CURVE = {
    "decode_tok_s": ("tok/s", (2.7021, 2.5218, 2.319, 2.0712, 0.8854)),
    "decode_tok_s_total": ("tok/s", (2.6172, 9.3153, 16.3616, 26.946, 49.5288)),
    "ttft_s": ("s", (1.1809, 4.8672, 7.7821, 14.9531, 24.3901)),
    "joules_per_token": ("J/token", (36.6587, 10.3071, 5.8676, 3.5628, 1.9384)),
    "watts_avg": ("W", (96.0046, 96.0126, 96.004, 96.0052, 96.0499)),
    "watts_peak": ("W", (97.0, 97.0, 97.0, 97.0, 98.0)),
}
A100_CURVE = {
    "decode_tok_s": ("tok/s", (64.956, 59.2302, 52.6137, 36.0495, 11.2975)),
    "decode_tok_s_total": ("tok/s", (61.2077, 198.933, 321.3743, 462.3296, 636.3307)),
    "ttft_s": ("s", (0.1184, 0.4949, 0.8243, 1.056, 1.9381)),
    "joules_per_token": ("J/token", (5.0174, 1.5234, 0.999, 0.7743, 0.6003)),
    "watts_avg": ("W", (306.0747, 302.8075, 321.0224, 355.9959, 382.1646)),
    "watts_peak": ("W", (407.97, 411.5, 418.76, 412.96, 424.61)),
}
STUDENT_LOAD = "back-to-back caption requests, one in flight, to llama-server :8080 (939 prompt tokens, max_tokens 80)"


def row(metric: str, value: float, node: str = "pod-b", unit: str = "tok/s", ts: str = "2026-10-11T03:00:00Z", low: bool = False,
        source: str = "a test", **conditions: Any) -> dict[str, Any]:
    return {"metric": metric, "value": value, "unit": unit, "node": node, "conditions": conditions, "source": source,
            "ts": ts, "low_confidence": low}


def write(tmp_path: Path, rows: list[dict[str, Any]], extra_lines: tuple[str, ...] | list[str] = (), name: str = "metrics.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join([*(json.dumps(r) for r in rows), *extra_lines]) + "\n", encoding="utf-8")
    return path


def sweep_rows(family: str, node: str, session: str, curve: dict[str, tuple[str, tuple[float, ...]]], ts: str = AFTER,
               batches: tuple[int, ...] = BATCHES, low_reason: str | None = THREE_RUNS, **conditions: Any) -> list[dict[str, Any]]:
    """One benchmark session as the lab's bench harness writes it: per batch size, one row per metric."""
    out = []
    for i, b in enumerate(batches):
        for name, (unit, values) in curve.items():
            v = values[i]
            extra = {"low_confidence_reason": low_reason} if low_reason else {}
            out.append(row(f"{family}.{name}", v, node=node, unit=unit, ts=ts, low=bool(low_reason), source=f"lab/bench.py 1 {session}",
                           batch_size=b, session=session, **extra, **conditions))
    return out


def pod_a_sweep(session: str = POD_A_SESSION, node: str = "pod-a", family: str = "teacher_batching", ts: str = AFTER, **conditions: Any) -> list[dict[str, Any]]:
    """The batching sweep on the Strix Halo teacher: 5 batch sizes x 6 metrics, each the median of 3 repeats."""
    rows = sweep_rows(family, node, session, POD_A_CURVE, ts=ts, checkpoint="/home/jovyan/models/Qwen3.8-27B-INT4", engine="vllm",
                      quantization="compressed-tensors-W4A16-INT4", max_tokens=128, prompt_set={"id": "bench-v1", "prompts": 4},
                      statistic="median of 3 repeats", mode="attach: the running teacher measured through the hub proxy",
                      hub="sjc", account="C", chip="gfx1151 (Strix Halo, Radeon 8060S)", sampler="T6 lab/pod_power.py 1 on port 8790, 1 s period",
                      **conditions)
    for r in rows:
        r["conditions"]["stats"] = {"n": 3, "median": r["value"], "min": r["value"] * 0.99, "max": r["value"] * 1.01, "sd": 0.01}
    return rows


def a100_sweep(session: str = A100_SESSION) -> list[dict[str, Any]]:
    """The reference run on an NVIDIA A100 on Colab: the same metrics and batch sizes, on a chip that is not the lab's."""
    rows = sweep_rows("teacher", "a100", session, A100_CURVE, ts="2026-10-11T01:52:01Z",
                      low_reason="3 spawns per batch size, under the 5 that verdict() needs to call a difference",
                      hub="colab", gpu=A100_GPU, engine="vllm", checkpoint="RedHatAI/Qwen3.8-27B-INT4", max_tokens=128, prompt_set="bench-v1",
                      other_chip=True, other_chip_note=A100_NOTE, statistic="median of 3 spawns")
    for r in rows:
        if r["metric"].endswith((".decode_tok_s", ".joules_per_token")):
            r["conditions"]["per_spawn"] = [r["value"] * 0.99, r["value"], r["value"] * 1.01]
        if r["metric"].endswith((".watts_avg", ".watts_peak", ".joules_per_token")):
            r["conditions"]["power_source"] = "nvidia-smi power.draw (board power), sampled once a second beside the harness"
    return rows


def recorded() -> list[dict[str, Any]]:
    """The small lab file described at the top. Line numbers are the list positions counted from 1."""
    old = {"model": "RedHatAI/Qwen3.8-27B-INT4", "engine": "vllm@b1388b1"}  # no chip named: before the move, so the R9700s
    strix = {"gpu": "Radeon 8060S (gfx1151)", "hub": "sjc"}
    return [
        row("teacher.decode_tok_s", 7.7, node="pod-a", ts=BEFORE, batch=1, prompt="text", **old),                               # 1
        row("teacher.watts_avg", 200, node="pod-a", unit="W", ts=BEFORE, **{"while": "serving one request"}, **old),               # 2
        row("teacher.watts_peak", 300, node="pod-a", unit="W", ts=BEFORE, **{"while": "serving one request"}, **old),              # 3
        row("trainer.samples_per_s", 0.51, node="pod-b", unit="samples/s", ts="2026-10-10T10:23:23Z", script="lab/smoke_lora.py"),  # 4
        row("pod.idle_watts", 39.0, node="pod-a", unit="W", ts="2026-10-10T09:55:00Z", low=True, state="idle"),                    # 5: low, no reason given
        row("pod.idle_watts", 10.2, node="pod-a", unit="W", ts="2026-10-10T20:43:00Z", low=True,
            low_confidence_reason="one sample", **strix),                                                                           # 6
        row("proxy.status", 500, node="pod-b", unit="http", ts="2026-10-10T22:14:00Z", **strix),                                   # 7
        row("jetson.idle_watts", 15.76, node="jetson", unit="W", ts="2026-10-10T21:58:00Z", low=True, low_confidence_reason="one run"),  # 8
        row("jetson.idle_watts", 12.62, node="jetson", unit="W", ts="2026-10-10T22:59:00Z", low=True, low_confidence_reason="one run"),  # 9
        row("student.watts_avg", 17.536, node="jetson", unit="W", ts="2026-10-10T23:03:27Z", low=True, rail="VDD_IN",
            sampler="contexts.telemetry.jetson_power probe (tegrastats)", period_ms=1000, samples=265, load=STUDENT_LOAD,
            request_mean_s=9.513, decode_tok_s_median=13.85, low_confidence_reason="one run and no external meter on the box"),      # 10
        row("student.watts_peak", 19.127, node="jetson", unit="W", ts="2026-10-10T23:03:27Z", low=True, low_confidence_reason="one run"),  # 11
        *pod_a_sweep(),                                                                                                             # 12-41
        row("pod.idle_watts", 10.44, node="pod-a", unit="W", ts="2026-10-11T00:30:00Z", server="teacher loaded, no request",
            chip="gfx1151", hub="sjc"),                                                                                             # 42
        *a100_sweep(),                                                                                                              # 43-72
        row("pod.idle_watts", 46.14, node="a100", unit="W", ts="2026-10-11T01:52:01Z", low=True, hub="colab", gpu=A100_GPU,
            other_chip=True, other_chip_note=A100_NOTE, state="idle, no server", low_confidence_reason="one 60 s stretch on one VM"),  # 73
    ]


def raw_lines(path: Path) -> dict[int, Any]:
    """Every non-blank line of a metrics file by its number: the parsed row, or None for a line that is not JSON."""
    out: dict[int, Any] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out[n] = json.loads(line)
            except ValueError:
                out[n] = None
    return out


def every_point(payload: dict[str, Any]):
    for s in payload["series"]:
        yield from s["points"]


def check_bench(payload: dict[str, Any], raw: dict[int, Any]) -> None:
    """What GET /api/learning/lab/bench must hold for ANY lab file, however many rows, nodes, chips and sweeps it
    has. `raw` is `raw_lines` of the file the payload was built from. No count, node name or value is pinned."""
    points = list(every_point(payload))
    skipped = {s["line"] for s in payload["skipped"]}
    # every row is served once, under its own line, with the file's value; a line that is not a row is listed
    assert payload["rows"] == len(points) and sorted(p["line"] for p in points) == sorted(set(raw) - skipped)
    assert skipped <= set(raw)
    for p in points:
        r = raw[p["line"]]
        assert (p["value"], p["metric"], p["unit"], p["node"], p["ts"]) == (r["value"], r["metric"], r.get("unit"), r.get("node"), r.get("ts"))
        assert p["source"] == {"kind": "file", "ref": f"lab/refine/metrics.jsonl:{p['line']}", "ids": []}
        assert p["low_confidence"] is bool(r.get("low_confidence")) and (p["reason"] is not None) is p["low_confidence"]
        # every point names its chip, and the old boolean means "not the current lab"
        chip = p["chip"]
        assert chip["key"] and chip["name"] and chip["reason"] and chip["role"] in ("current", "earlier", "reference", "unnamed")
        assert chip["current"] is (chip["role"] == "current") and p["other_chip"] is (not chip["current"])
        if p["node"] == "jetson":
            assert chip["key"] == "jetson"
        # a row that names a chip is called by that chip: a card that is not the lab's is never the current lab,
        # and is never filed with the R9700s
        said = r.get("conditions") if isinstance(r.get("conditions"), dict) else {}
        named = next((said[f] for f in ("gpu", "chip") if isinstance(said.get(f), str) and said[f].strip()), None)
        if p["node"] != "jetson" and named:
            low = named.lower()
            ours = any(m in low for m in ("gfx1151", "8060s", "strix"))
            earlier = any(m in low for m in ("gfx1201", "r9700"))
            assert chip["role"] == ("current" if ours else "earlier" if earlier else "reference"), (p["line"], named)
            if "a100" in low:
                assert chip["name"] == "NVIDIA A100" and chip["key"] not in ("r9700", "strix-halo") and not chip["current"]
        if chip["role"] == "reference":
            assert chip["key"] not in ("r9700", "strix-halo", "jetson", "unnamed")
    assert payload["other_chip_rows"] == sum(p["other_chip"] for p in points)
    assert payload["low_confidence_rows"] == sum(p["low_confidence"] for p in points)
    assert sum(g["rows"] for g in payload["chips"]) == sum(p["node"] != "jetson" for p in points)
    assert all(len(g["lines"]) == g["rows"] and g["current"] is (g["role"] == "current") for g in payload["chips"])

    # a row written twice points at an earlier line that holds the same reading
    by_line = {p["line"]: p for p in points}
    assert payload["repeated_rows"] == len(payload["repeats"]) == sum(p["repeat_of"] is not None for p in points)
    for x in payload["repeats"]:
        first, again = by_line[x["of"]], by_line[x["line"]]
        assert x["of"] < x["line"] and first["repeat_of"] is None and again["repeat_of"] == x["of"]
        assert (first["metric"], first["node"], first["value"], first["ts"]) == (again["metric"], again["node"], again["value"], again["ts"])

    # sweeps: ids are unique, no sweep mixes chips or nodes, each point is a row of the file, the counts add up
    sweeps = payload["sweeps"]
    assert len({s["id"] for s in sweeps}) == len(sweeps)
    assert [s["other_chip"] for s in sweeps] == sorted(s["other_chip"] for s in sweeps)  # the current lab's first
    for flag in (False, True):
        stamps = [str(s["ts"] or "") for s in sweeps if s["other_chip"] is flag]
        assert stamps == sorted(stamps, reverse=True)  # newest first within each
    for s in sweeps:
        pts = [p for metric in s["metrics"].values() for p in metric]
        assert s["other_chip"] is (not s["chip"]["current"]) and len(s["batches"]) >= 2 and s["batches"] == sorted(set(s["batches"]))
        assert {p["chip"]["key"] for p in pts} == {s["chip"]["key"]} and {p["node"] for p in pts} == {s["node"]}, s["id"]
        assert {p["other_chip"] for p in pts} == {s["other_chip"]}, s["id"]
        assert s["points"] == len(pts) and s["low_confidence"] == sum(p["low_confidence"] for p in pts)
        assert s["superseded"] >= 0 and s["repeated"] >= 0 and s["confidence_note"]
        assert len({p["line"] for p in pts}) == len(pts)
        for name, metric in s["metrics"].items():
            xs = [p["x"] for p in metric]
            assert xs == sorted(set(xs)) and set(xs) <= set(s["batches"]), (s["id"], name)
            for p in metric:
                assert by_line[p["line"]]["value"] == p["value"] and p["metric"] == f"{s['family']}.{name}"
        assert {x for metric in s["metrics"].values() for x in (p["x"] for p in metric)} == set(s["batches"])
        # only the current lab's sweep offers comparisons, and each is another chip's sweep that exists
        others = {o["id"]: o for o in sweeps if o["other_chip"]}
        assert s["comparisons"] == [] or not s["other_chip"]
        for c in s["comparisons"]:
            other = others[c["id"]]
            assert c["chip"] == other["chip"] and c["metrics"] and c["batches"]
            assert set(c["batches"]) <= set(s["batches"]) & set(other["batches"])

    # power: a bar is the current lab's own row on that node; a node with no row is absent, never zero
    power = payload["power"]
    drawn = [n["node"] for n in power["nodes"]]
    assert len(set(drawn)) == len(drawn) and not set(power["absent"]) & set(drawn)
    for n in power["nodes"]:
        assert n["idle"] or n["serving"]
        for key, suffix in (("idle", ".idle_watts"), ("serving", ".watts_avg"), ("peak", ".watts_peak")):
            if n[key]:
                mine = [p for p in points if p["node"] == n["node"] and p["metric"].endswith(suffix) and not p["other_chip"]]
                assert n[key]["node"] == n["node"] and not n[key]["other_chip"] and n[key]["metric"].endswith(suffix)
                assert str(n[key]["ts"] or "") == max(str(p["ts"] or "") for p in mine) and n[key]["value"] == by_line[n[key]["line"]]["value"]
    assert all(p["other_chip"] and p["metric"].endswith((".idle_watts", ".watts_avg", ".watts_peak")) for p in power["other_chip"])
    assert sum(g["rows"] for g in power["other_chips"]) == len(power["other_chip"])

    # nothing is dropped silently: a row is drawn or listed, never both, and the families cover every row
    listed = [line for x in payload["not_charted"] for line in x["lines"]]
    assert len(set(listed)) == len(listed) and set(listed) <= set(by_line)
    assert payload["charted_rows"] + len(listed) == payload["rows"]
    assert all(x["rows"] == len(x["lines"]) <= x["of"] and x["other_chip"] == sum(g["rows"] for g in x["other_chips"]) for x in payload["not_charted"])
    assert sum(f["rows"] for f in payload["families"]) == payload["rows"]
    assert sum(f["charted"] for f in payload["families"]) == payload["charted_rows"]


def check_sessions(cards: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """What the timeline's benchmark-session cards must hold for ANY lab file. `rows` are `read_rows` of that file."""
    in_file: dict[str, list[int]] = {}
    raw = {r["_line"]: r for r in rows}
    for n, r in sorted(raw.items()):
        session = r["conditions"].get("session")
        if isinstance(session, str) and session.strip():
            in_file.setdefault(session, []).append(n)
    assert sorted(c["id"] for c in cards) == sorted(in_file)  # one card per session, ids unique
    for c in cards:
        lines = in_file[c["id"]]
        cited = []
        for run in c["source"]["ref"].split(":", 1)[1].split(", "):
            a, _, b = run.partition("-")
            cited.extend(range(int(a), int(b or a) + 1))
        assert cited == lines, c["id"]  # the card cites its own rows, and only them
        assert c["kind"] == "bench" and c["status"] == "measured" and c["verdict"].startswith("Measured")
        recorded_n, low, *again = (n["value"] for n in c["numbers"])
        assert recorded_n + sum(again) == len(lines) and 0 <= low <= recorded_n and all(a > 0 for a in again)
        nodes = {str(raw[n].get("node")) for n in lines}
        assert c["node"] in nodes if len(nodes) == 1 else c["node"] is None
        assert c["ts"] == max(str(raw[n].get("ts")) for n in lines)
