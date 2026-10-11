"""T11: the routing hook's closed-loop test (lab/routing_closed_loop.py), run on the teacher and student fakes.

Each test names the clause of T11's Done when it proves. Nothing here reaches a network address: the teacher is
lab/teacher_fake.py, the student lab/student_fake.py, and the power samples come from the fake meter feed.
"""

import json
from pathlib import Path

import pytest

from contexts.telemetry.application.routing import CAPABILITY, ENERGY
from lab import __main__ as dispatcher
from lab import routing_closed_loop as loop
from lab.fixture_dataset import load_expected
from lab.machine import experiments, ledger
from lab.student_fake import FakePowerFeed, student_policy

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "learning" / "tool-calling-fixture"
REFINE = Path(__file__).resolve().parents[1] / "lab" / "refine"


def _refine_state() -> dict[str, float]:
    return {str(p): p.stat().st_mtime for p in REFINE.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def cheap_teacher(tmp_path_factory):
    """A run where the teacher costs fewer joules per answer than the student: policy two should move the work."""
    state = tmp_path_factory.mktemp("state")
    out = tmp_path_factory.mktemp("out")
    before = _refine_state()
    result = loop.run_fake(n=100, seed=11, state_dir=state, out_dir=out,
                           fake=loop.FakeSetup(teacher_watts=20.0, student_watts=150.0, teacher_decode_tok_s=40.0,
                                               student_decode_tok_s=10.0, student_miss_share=0.5))
    return result, state, out, before


@pytest.fixture(scope="module")
def dear_teacher(tmp_path_factory):
    """The fakes' default numbers: the teacher on the Strix Halo costs more per answer than the Jetson student."""
    state = tmp_path_factory.mktemp("state")
    out = tmp_path_factory.mktemp("out")
    result = loop.run_fake(n=100, seed=11, state_dir=state, out_dir=out, fake=loop.FakeSetup())
    return result, state, out


# --- the request set: at least 100 lab labeling requests from the L11 fixture, fixed seed, no golden question ---

def test_the_request_set_is_100_fixture_labeling_requests_drawn_with_a_fixed_seed_and_no_golden_question():
    drawn = loop.draw_requests(FIXTURE, n=100, seed=11)
    expected = load_expected(FIXTURE)
    assert len(drawn) == 100 and len({r.example.id for r in drawn}) == 100
    assert all(expected[r.example.id].origin == "variant" for r in drawn)  # never a golden question
    assert [r.example.id for r in drawn] == [r.example.id for r in loop.draw_requests(FIXTURE, n=100, seed=11)]
    assert [r.example.id for r in drawn] != [r.example.id for r in loop.draw_requests(FIXTURE, n=100, seed=12)]
    route = drawn[0].route
    assert (route.kind, route.source, route.own_space) == ("label", "learning", False)  # work the hook may send out


def test_asking_for_more_requests_than_the_fixture_has_without_golden_questions_is_refused():
    with pytest.raises(loop.LoopError, match="golden"):
        loop.draw_requests(FIXTURE, n=500, seed=11)


def test_fewer_than_100_requests_is_refused_by_the_verb(tmp_path):
    answer = loop.main(["--fake", "--n", "99", "--out", str(tmp_path), "--state-dir", str(tmp_path)])
    assert answer["ok"] is False and "100" in answer["error"]


# --- routed twice through T5's hook, policy one (capability, control) and policy two (energy), interleaved ---

def test_every_request_is_routed_under_both_policies_with_the_arms_interleaved(cheap_teacher):
    result = cheap_teacher[0]
    records = result.records
    assert [r.arm for r in records] == ["A", "B"] * 100  # A, B, A, B: request by request
    for a, b in zip(records[::2], records[1::2], strict=True):
        assert a.example_id == b.example_id
        assert a.policy == CAPABILITY and b.policy == ENERGY
    assert {r.decided_by for r in records if r.arm == "A"} == {CAPABILITY}


def test_policy_two_reads_the_telemetry_snapshot_and_moves_work_to_the_cheaper_node(cheap_teacher):
    records = cheap_teacher[0].records
    a_targets = {r.target for r in records if r.arm == "A"}
    b = [r for r in records if r.arm == "B"]
    assert a_targets == {"student"}  # capability: the student can label
    assert all(r.decided_by == ENERGY for r in b)  # the probes filled the snapshot for both nodes
    assert sum(r.target == "teacher" for r in b) > 90  # the snapshot says the teacher is cheaper
    assert all(r.snapshot_teacher_j is not None and r.snapshot_student_j is not None for r in b)


def test_the_snapshot_is_the_joules_projections_and_every_answer_feeds_it():
    feed = FakePowerFeed({"jetson": 60.0, "pod-a": 20.0}, idle={"jetson": 15.0, "pod-a": 20.0}, jitter=0.0)
    meter = loop.Meter(feed, clock=feed.clock)
    t0 = feed.clock.time()
    feed.clock.sleep(10.0)
    value = meter.measure("jetson", "r1", t0 + 2.0, t0 + 6.0, target="student", kind="label", policy=CAPABILITY,
                          tokens_out=40)
    assert value is not None and value.joules == pytest.approx(60.0 * 4.0, rel=0.02)
    assert meter.snapshot()["nodes"]["jetson"]["by_kind"]["label"]["joules_per_answer"] == value.joules


# --- joules per answer, latency and the label agreement, for both arms, with their spread ---

def test_each_request_records_joules_latency_and_the_label_agreement(cheap_teacher):
    for r in cheap_teacher[0].records:
        assert r.joules is not None and r.joules > 0
        assert r.latency_s > 0
        assert r.quality in (0.0, 1.0)


def test_each_arm_has_a_mean_and_a_spread_for_all_three(cheap_teacher):
    arms = cheap_teacher[0].summary["arms"]
    for arm in ("A", "B"):
        for metric in (loop.JOULES, loop.LATENCY, loop.QUALITY):
            stats = arms[arm][metric]
            assert stats["n"] == 100 and stats["mean"] is not None and stats["sd"] is not None


def test_the_quality_score_is_label_agreement_with_the_fixture_reference():
    expected = load_expected(FIXTURE)
    row = next(e for e in expected.values() if e.origin == "variant")
    right = json.dumps({"tool_calls": [c.model_dump() for c in row.label.tool_calls], "answer": "anything"})
    assert loop.agreement(right, row) == 1.0
    assert loop.agreement(json.dumps({"tool_calls": [], "answer": "I didn't see that."}), row) == 0.0
    assert loop.agreement(None, row) == 0.0  # a rejected or failed label does not agree


def test_the_student_fake_misses_its_share_and_the_teacher_fake_agrees(cheap_teacher):
    records = [r for r in cheap_teacher[0].records if r.arm == "A"] + [r for r in cheap_teacher[0].probes]
    student = [r.quality for r in records if r.target == "student"]
    teacher = [r.quality for r in records if r.target == "teacher"]
    assert 0.3 < sum(student) / len(student) < 0.7  # miss share 0.5
    assert teacher and sum(teacher) == len(teacher)
    with pytest.raises(ValueError):
        student_policy(load_expected(FIXTURE), miss_share=1.5)


# --- the verdict math: did policy two change energy or quality beyond the spread ---

def test_the_verdict_says_policy_two_changed_energy_and_quality_when_the_teacher_is_cheaper(cheap_teacher):
    verdicts = cheap_teacher[0].summary["verdicts"]
    assert verdicts[loop.JOULES]["label"] == "b_ahead"  # fewer joules per answer
    assert verdicts[loop.QUALITY]["label"] == "b_ahead"  # the teacher agrees more
    assert cheap_teacher[0].summary["changed"] == {"energy": True, "quality": True}


def test_the_verdict_says_no_change_when_policy_two_keeps_the_work_on_the_student(dear_teacher):
    summary = dear_teacher[0].summary
    assert all(r.target == "student" for r in dear_teacher[0].records)  # the snapshot says the student is cheaper
    assert summary["verdicts"][loop.JOULES]["label"] == "no_difference"
    assert summary["changed"] == {"energy": False, "quality": False}


# --- the experiment result on T0 with its citation, and a ledger row per arm ---

def test_the_t0_experiment_result_text_is_written_with_its_citation(cheap_teacher):
    result, state, out, _ = cheap_teacher
    text = (out / result.run_id / loop.T0_FILE).read_text(encoding="utf-8")
    assert "Source:" in text and result.run_id in text and "lab/routing_closed_loop.py" in text
    assert "telemetry.joules_per_answer" in text and "learning.label_agreement" in text
    assert "Result" in text
    assert "\u2014" not in text  # plain hyphens
    folded = experiments.load(state)
    assert len(folded) == 2 and all(e["status"] == "done" and e["task"] == "T0" for e in folded.values())


def test_the_two_t0_experiments_are_added_under_the_tests_own_ids_on_a_fresh_ledger(cheap_teacher):
    # experiments.add hands out an E number only from a ledger that already holds one beyond T0's seeds; a fresh
    # state dir holds none, so the test names its two experiments itself
    result, state, _, _ = cheap_teacher
    assert result.experiments == list(loop.EXPERIMENT_IDS)
    assert sorted(experiments.load(state)) == sorted(loop.EXPERIMENT_IDS)


def test_the_ids_t0_gave_replace_the_default_ones_and_a_second_run_reuses_them(tmp_path):
    state, out = tmp_path / "state", tmp_path / "out"
    first = loop.run_fake(n=100, seed=11, state_dir=state, out_dir=out, experiment_ids=("E61", "E62"))
    assert first.experiments == ["E61", "E62"]
    again = loop.run_fake(n=100, seed=11, state_dir=state, out_dir=out, experiment_ids=("E63", "E64"))
    assert again.experiments == ["E61", "E62"] and sorted(experiments.load(state)) == ["E61", "E62"]  # found by title
    with pytest.raises(loop.LoopError, match="two different ids"):
        loop.run_fake(n=100, seed=11, state_dir=tmp_path / "other", out_dir=out, experiment_ids=("E61", "E61"))


# --- the metrics-file rows: what the Learning lab's Routing tab reads as the measured closed loop ---

def test_the_run_goes_into_the_metrics_file_as_one_row_per_policy_and_metric_that_the_routing_tab_reads(tmp_path):
    from contexts.learning.application import lab_bench, lab_routing

    path = tmp_path / "metrics.jsonl"
    told = {"gpu": "Radeon 8060S (Strix Halo), gfx1151", "hub": "sjc", "account": "C"}
    result = loop.run_fake(n=100, seed=11, state_dir=tmp_path / "state", out_dir=tmp_path / "out",
                           metrics_path=path, row_conditions=told)
    assert result.metrics_rows == 6
    rows, skipped = lab_bench.read_rows(path)
    assert not skipped and len(rows) == 6
    for row in rows:
        c = row["conditions"]
        assert c["policy"] in (CAPABILITY, ENERGY) and c["requests"] == 100 and c["session"] == result.run_id
        assert c["gpu"] == told["gpu"] and c["hub"] == "sjc" and row["node"] == "pod-a"
        assert row["low_confidence"] is True and c["low_confidence_reason"]  # the fakes are never a measurement
    measured = lab_routing.closed_loop(rows)
    assert measured is not None and measured["min_requests"] == 100
    assert {m["metric"] for m in measured["rows"]} == {loop.JOULES, loop.LATENCY, loop.QUALITY}
    arms = result.summary["arms"]
    for m in measured["rows"]:
        assert m["capability"]["value"] == round(arms["A"][m["metric"]]["mean"], 4)
        assert m["energy"]["value"] == round(arms["B"][m["metric"]]["mean"], 4)


def test_a_real_runs_rows_are_low_confidence_only_for_joules_and_say_why():
    stats = {"n": 100, "mean": 2.0, "sd": 0.5}
    arm = {"requests": 100, "targets": {"student": 100}, "decided_by": {"capability": 100}, "unmeasured": 3,
           "low_confidence": 97, loop.JOULES: stats, loop.LATENCY: stats, loop.QUALITY: stats}
    summary = {"arms": {"A": arm, "B": arm}, "verdicts": {m: {"label": "no difference"} for m in loop.UNITS}}
    rows = loop.metrics_rows(summary, run_id="t11-x", source="real", conditions={"gpu": "gfx1151"})
    assert [(r["metric"], r["conditions"]["policy"]) for r in rows] == [
        (m, policy) for policy in (CAPABILITY, ENERGY) for m in (loop.JOULES, loop.LATENCY, loop.QUALITY)]
    for row in rows:
        assert row["low_confidence"] is (row["metric"] == loop.JOULES)
        assert ("low_confidence_reason" in row["conditions"]) is (row["metric"] == loop.JOULES)
    assert "97 of 100" in rows[0]["conditions"]["low_confidence_reason"]


def test_no_metrics_row_is_written_unless_a_file_is_named(dear_teacher):
    result, _, _ = dear_teacher
    assert result.metrics_rows == 0


def test_told_row_conditions_are_name_value_pairs_and_a_json_value_is_read_as_one():
    assert loop._told(["gpu=Radeon 8060S (Strix Halo), gfx1151", "requests_per_s=0.5", "quiet=true"]) == {
        "gpu": "Radeon 8060S (Strix Halo), gfx1151", "requests_per_s": 0.5, "quiet": True}
    with pytest.raises(loop.LoopError, match="NAME=VALUE"):
        loop._told(["gpu"])


def test_a_ledger_row_per_arm(cheap_teacher):
    result, state, _, _ = cheap_teacher
    rows = ledger.read(state_dir=state, verb=loop.VERB)
    assert sorted(row["conditions"]["arm"] for row in rows) == ["A", "B"]
    for row in rows:
        assert row["task"] == "T11" and row["refs"]["run_id"] == result.run_id
        assert {loop.JOULES, loop.LATENCY, loop.QUALITY} <= set(row["metrics"])
        assert row["conditions"]["policy"] in (CAPABILITY, ENERGY)
        assert row["conditions"]["n"] == 100 and "sd" in row["conditions"][loop.JOULES]


def test_the_per_request_records_and_the_report_are_written(cheap_teacher):
    result, _, out, _ = cheap_teacher
    folder = out / result.run_id
    lines = [json.loads(line) for line in (folder / loop.REQUESTS_FILE).read_text(encoding="utf-8").splitlines()]
    assert sum(r["arm"] in ("A", "B") for r in lines) == 200
    assert sum(r["arm"] == "probe" for r in lines) == 2  # one probe on each node
    first = lines[0]
    assert {"arm", "target", "joules", "latency_s", "quality", "example_id"} <= set(first)
    report = json.loads((folder / loop.REPORT_FILE).read_text(encoding="utf-8"))
    assert report["run_id"] == result.run_id and report["requests"] == 100


# --- rubric test 6 reaches the rubric only as a finding, never as an edit ---

def test_rubric_test_6_is_a_finding_in_the_report_and_nothing_under_lab_refine_changes(cheap_teacher):
    result, _, out, before = cheap_teacher
    finding = result.summary["rubric_tt6"]
    assert finding["met"] is True and "finding" in finding["text"].lower()
    assert "TT6" in (out / result.run_id / loop.T0_FILE).read_text(encoding="utf-8")
    assert _refine_state() == before


def test_rubric_test_6_is_not_met_when_policy_two_never_read_a_value(tmp_path):
    result = loop.run_fake(n=100, seed=11, state_dir=tmp_path, out_dir=tmp_path, probes=0,
                           fake=loop.FakeSetup(student_miss_share=0.0))
    assert result.summary["rubric_tt6"]["met"] is False  # the snapshot never held the teacher: capability fallback


# --- the verb ---

def test_the_verb_is_registered_and_runs_on_the_fakes(tmp_path, capsys):
    assert dispatcher.VERBS["routing-closed-loop"] == "lab.routing_closed_loop:main"
    names = list(dispatcher.VERBS)
    assert names == sorted(names)
    answer = loop.main(["--fake", "--n", "100", "--out", str(tmp_path), "--state-dir", str(tmp_path)])
    assert answer["ok"] is True and answer["requests"] == 100
    assert Path(answer["t0_text"]).exists()


def test_the_jsonl_feed_reads_only_new_lines(tmp_path):
    log = tmp_path / "events.jsonl"
    feed = FakePowerFeed({"jetson": 30.0}, jitter=0.0)
    feed.clock.sleep(3.0)
    events = feed.poll()
    log.write_text("".join(e.model_dump_json() + "\n" for e in events[:2]), encoding="utf-8")
    reader = loop.JsonlFeed([log])
    assert len(reader.poll()) == 2
    with log.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
        fh.write(events[2].model_dump_json() + "\n")
    assert len(reader.poll()) == 1 and reader.poll() == []


def test_a_snapshot_that_holds_only_the_latest_value_makes_policy_two_follow_the_last_answer(tmp_path):
    """A short student answer (a miss, no tool call) costs few joules: the next decision under policy two then goes
    to the student, and since A answers the same request just before B, B repeats A's miss. The report says so."""
    result = loop.run_fake(n=100, seed=11, state_dir=tmp_path, out_dir=tmp_path,
                           fake=loop.FakeSetup(teacher_watts=20.0, student_watts=60.0, teacher_decode_tok_s=40.0,
                                               student_decode_tok_s=10.0, student_miss_share=0.5))
    b = result.summary["arms"]["B"]["targets"]
    assert b.get("student", 0) > 0 and b.get("teacher", 0) > 0
    assert "latest" in result.summary["rubric_tt6"]["text"]
