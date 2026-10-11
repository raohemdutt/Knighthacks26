"""The live loop's arithmetic, once (loop_energy.py): fed the rehearsal's own stage lines, power traces and counts, it
gives back the comparison the relay wrote that night, so the relay and the box compute the same numbers."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from contexts.learning.application import loop_energy as le

REHEARSAL = Path(__file__).resolve().parents[3] / "fixtures" / "learning" / "live-loop-2026-10-11"


def recorded():
    return json.loads((REHEARSAL / "comparison.json").read_text(encoding="utf-8"))


def stages():
    return [json.loads(x) for x in (REHEARSAL / "stages.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def traces():
    offsets = recorded()["clock_offsets_s"]
    out = {}
    for node in ("pod-c", "pod-d", "box"):
        rows = [json.loads(x) for x in (REHEARSAL / f"power-{node}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        out[node] = ([(le.iso_s(r["ts"]), float(r["w"])) for r in rows], offsets[node])
    return out


def facts(row):
    """The counts the relay read from the act's run folder on the pod, as they landed in its comparison."""
    lab, tr = row["labeling"], row["training"]
    return {"outcome": row["outcome"],
            "counts": {k: row[k] for k in ("examples", "train_rows", "held_out")},
            "labels": {k: lab[k] for k in ("examples", "tokens", "completion_tokens")},
            "trainer": {"steps": tr["steps"], "examples_seen": tr["examples_seen"], "tokens": tr["tokens"],
                        "train_s": tr["train_s"], "loss_first": tr["loss_first"], "loss_last": tr["loss_last"],
                        "energy_j": tr["trainer_energy_j"]}}


def rebuilt():
    doc = recorded()
    go = next(s for s in stages() if s["stage"] == "go")
    t0 = le.iso_s(go["ts"]) - doc["clock_offsets_s"]["pod-d"] - doc["go_s"]
    return le.compare(stages(), {a: doc["acts"][a]["run_id"] for a in le.ACTS},
                      {a: facts(doc["acts"][a]) for a in le.ACTS}, traces(), t0, doc["total_s"], doc["stamp"]), doc


def test_stage_seconds_match_the_rehearsal():
    assert le.stage_seconds(stages(), "solo") == {"labeling": 32.1, "waiting": 0.0, "training": 53.9, "packaging": 71.1,
                                                  "total": 157.0}
    assert le.stage_seconds(stages(), "duo")["training"] == 68.8


def test_the_rehearsal_comparison_comes_back_from_its_own_inputs():
    got, doc = rebuilt()
    assert set(doc) - {"t0"} <= set(got)
    assert got["stamp"] == doc["stamp"] and got["go_s"] == doc["go_s"] and got["rule"] == doc["rule"]
    for act in le.ACTS:
        g, d = got["acts"][act], doc["acts"][act]
        assert g["seconds"] == d["seconds"] and g["run_id"] == d["run_id"]
        assert (g["start_s"], g["end_s"]) == (d["start_s"], d["end_s"])
        for stage in ("training", "packaging"):   # pod D against its own clock: exact
            for key in ("joules", "gross_window_j", "overlap_s", "seconds", "idle_j", "peak_w", "samples"):
                assert g[stage][key] == d[stage][key], (act, stage, key)
        assert g["training"]["j_per_step"] == d["training"]["j_per_step"]
        # pod C and the box move through the relay's clock, and the committed offsets are rounded to 10 ms
        assert g["labeling"]["joules"] == pytest.approx(d["labeling"]["joules"], rel=0.01)
        assert g["box"]["joules"] == pytest.approx(d["box"]["joules"], rel=0.02)
        assert g["pods_joules"] == pytest.approx(d["pods_joules"], rel=0.01)
    for key, value in doc["duo_over_solo"].items():
        assert got["duo_over_solo"][key] == pytest.approx(value, abs=0.01), key


def test_the_idle_floors_sit_beside_the_figures():
    got, doc = rebuilt()
    assert got["idle"]["pod-c"] == doc["idle"]["pod-c"] and got["idle"]["pod-d"] == doc["idle"]["pod-d"]
    assert got["idle"]["box"]["calibrated_w"] == 15.76 and "note" in got["idle"]["box"]


def test_pod_d_split_adds_up_to_the_pods_joules():
    points = traces()["pod-d"][0]
    shared, overlaps = le.pod_d_shares(stages(), points)
    first = le.window(stages(), "solo", "train", "trained")[0]
    last = le.window(stages(), "duo", "trained", "done")[1]
    assert sum(shared.values()) == pytest.approx(le.integrate(points, first, last), rel=1e-3)
    assert overlaps["solo.packaging"] > 0 and overlaps["solo.training"] == 0.0


def test_ratios_are_b_over_a_and_none_where_there_is_nothing_to_divide():
    a = {"examples": 100, "seconds": {"total": 10.0}, "labeling": {"joules": 0}}
    b = {"examples": 200, "seconds": {"total": 15.0}, "labeling": {"joules": 5}}
    out = le.ratios(a, b)
    assert out["examples"] == 2.0 and out["act_seconds"] == 1.5 and out["labeling_j"] is None
    assert out["members"] is None and le.ratios(a, None) is None


def test_one_act_alone_gives_a_document_without_the_ratio():
    doc = recorded()
    got = le.compare(stages()[:4], {"solo": doc["acts"]["solo"]["run_id"]}, {}, traces(), 0.0, None, "s")
    assert set(got["acts"]) == {"solo"} and "duo_over_solo" not in got and got["total_s"] is None
    assert got["acts"]["solo"]["seconds"] == {"labeling": 32.1, "waiting": 0.0}


# --- the two-pod teacher and act 2 on both pods (L29) ---------------------------------------------------------------

def rank1_lines():
    """Pod C's rank 1 lines for act 2, on pod C's clock: the rehearsal's act 2 training window moved onto pod C."""
    offsets = recorded()["clock_offsets_s"]
    a, b = le.window(stages(), "duo", "train", "trained")
    d = offsets["pod-c"] - offsets["pod-d"]
    iso = lambda t: datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")  # noqa: E731
    return [{"act": "duo", "stage": "rank1_train", "ts": iso(a + d + 1.0), "node": "pod-c"},
            {"act": "duo", "stage": "rank1_trained", "ts": iso(b + d - 0.5), "node": "pod-c"}]


def pair(split=False):
    doc = recorded()
    go = next(s for s in stages() if s["stage"] == "go")
    t0 = le.iso_s(go["ts"]) - doc["clock_offsets_s"]["pod-d"] - doc["go_s"]
    facts = {a: facts_of(doc["acts"][a]) for a in le.ACTS}
    if split:
        facts["duo"]["trainer_pods"] = {"pod-c": {"run_id": "r.pod-c", "steps": 8, "loss_first": 5.1, "loss_last": 2.2}}
    lines = stages() + (rank1_lines() if split else [])
    return le.compare(lines, {a: doc["acts"][a]["run_id"] for a in le.ACTS}, facts, traces(), t0, doc["total_s"],
                      doc["stamp"], teacher_nodes=("pod-c", "pod-d"))


facts_of = facts


def test_with_the_pair_labeling_counts_both_pods_and_each_pods_stages_add_up_to_its_joules():
    got = pair()
    solo = got["acts"]["solo"]["labeling"]
    assert set(solo["by_node"]) == {"pod-c", "pod-d"} and solo["node"] == "pods C and D (the two-pod teacher)"
    assert solo["joules"] == pytest.approx(solo["by_node"]["pod-c"]["joules"] + solo["by_node"]["pod-d"]["joules"], abs=0.2)
    assert solo["idle_j"] == round(2 * le.IDLE_W["pod-c"] * solo["seconds"], 1)
    shares = le.node_shares(stages(), traces(), ("pod-c", "pod-d"))
    pod_d = traces()["pod-d"][0]
    first = le.window(stages(), "solo", "start", "labeled")[0]
    last = le.window(stages(), "duo", "trained", "done")[1]
    whole = le.integrate(pod_d, first, last)   # pod D's windows cover the run end to end, each stretch once
    assert sum(x["joules"] for x in shares["pod-d"].values()) == pytest.approx(whole, rel=1e-3)
    assert got["pod_d_split"] == le.NODE_SPLIT and got["teacher_nodes"] == ["pod-c", "pod-d"]
    assert got["acts"]["solo"]["training"]["joules"] < recorded()["acts"]["solo"]["training"]["joules"]  # it shares pod D now


def test_act_two_on_both_pods_sums_the_ranks_and_keeps_each_pods_row():
    got = pair(split=True)
    duo, solo = got["acts"]["duo"], got["acts"]["solo"]
    assert duo["pods"] == ["pod-d", "pod-c"] and solo["pods"] == ["pod-d"] and duo["members"] == 2 and solo["members"] == 1
    assert duo["training"]["pods"] == 2 and duo["training"]["node"] == "pods D and C (two-pod LoRA, R65)"
    rows = duo["training_pods"]
    assert set(rows) == {"pod-d", "pod-c"}
    assert duo["training"]["joules"] == pytest.approx(rows["pod-d"]["joules"] + rows["pod-c"]["joules"], abs=0.2)
    assert rows["pod-c"]["steps"] == 8 and rows["pod-c"]["run_id"] == "r.pod-c" and rows["pod-c"]["j_per_step"]
    assert rows["pod-c"]["seconds"] == pytest.approx(rows["pod-d"]["seconds"] - 1.5, abs=0.2)
    assert duo["pods_joules"] == pytest.approx(duo["labeling"]["joules"] + duo["training"]["joules"]
                                              + duo["packaging"]["joules"], abs=0.3)
    assert got["duo_over_solo"]["training_j"] == le.ratio(duo["training"]["joules"], solo["training"]["joules"])
    # pod C's rank shares pod C with nothing else at that time (labeling ended before act 2 trained)
    assert rows["pod-c"]["overlap_s"] == 0.0 and rows["pod-c"]["joules"] == rows["pod-c"]["gross_window_j"]
