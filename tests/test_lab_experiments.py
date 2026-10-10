"""Experiments as ledger rows: the definition and its checks, add and amend, rank by expected gain times
confidence, result read by the verdict math, the T0 checklist (--print), the seeds, the fold's handling of rows it
cannot use, and the verb run through `python -m lab`. Every test works in a temporary state dir; nothing here
touches lab/state.

Where the expected numbers come from (none of them is the code under test):
  * the hand-checked A/B numbers of the phone project's method: 3 of 60 against 12 of 60 gives z 2.484, p 0.013;
    5% to 10% takes 435 trials an arm;
  * R's documented t.test output for its sleep and mtcars data sets (the same references the verdict tests use);
  * the ranked table in lab/refine/research/kernels/w4a16_decode_gfx1201.md section 7 (gain and confidence per
    row), copied below;
  * hand arithmetic, written out in the docstring of the test that uses it.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import experiments as ex
from lab.machine import ledger, record
from lab.machine import verdict as v

REPO_ROOT = Path(__file__).resolve().parents[1]
SPAWN = multiprocessing.get_context("spawn")
JOIN_TIMEOUT = 300.0
START_TIMEOUT = 60.0
ADDS_PER_CHILD = 8

# R's sleep data (extra hours of sleep per patient, two drugs) and mtcars miles per gallon by transmission.
SLEEP_1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP_2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
MPG_AUTOMATIC = [21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 21.5,
                 15.5, 15.2, 13.3, 19.2]
MPG_MANUAL = [21.0, 21.0, 22.8, 32.4, 30.4, 33.9, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4]

# lab/refine/research/kernels/w4a16_decode_gfx1201.md section 7, in the note's printed rank order: (id, gain,
# confidence). Gain there is the expected fractional cut in joules per token.
KERNELS_TABLE = [
    ("KE1", 0.85, 0.75), ("KE2", 0.60, 0.60), ("KE3", 0.45, 0.45), ("KE4", 0.50, 0.30), ("KE6", 0.45, 0.30),
    ("KE5", 0.25, 0.40), ("KE14", 0.40, 0.20), ("KE13", 0.55, 0.15), ("KE7", 0.30, 0.25), ("KE8", 0.15, 0.45),
    ("KE10", 0.08, 0.40), ("KE9", 0.10, 0.30), ("KE12", 0.07, 0.30), ("KE11", 0.015, 0.60),
    ("KE15", 0.03, 0.25), ("KE16", 0.01, 0.30),
]


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    """A temporary state dir for every test, and the sitting's identity in the environment."""
    path = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(path))
    monkeypatch.setenv("LAB_ACTOR", "opus-3")
    monkeypatch.setenv("LAB_TIER", "Opus")
    return path


def cli(capsys, *args):
    """Run `python -m lab experiment ...` in process: (exit code, the one JSON answer)."""
    code = lab_cli.main(["experiment", *args])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 1, f"expected one JSON line, got {out!r}"
    return code, json.loads(lines[0])


def cli_text(capsys, *args):
    code = lab_cli.main(["experiment", *args])
    return code, capsys.readouterr().out


def definition(**changes):
    fields = {"id": "E9", "title": "Batch sweep", "track": "kernels", "hypothesis": "Batching cuts joules per token",
              "lever": "batch size", "arms": {"A": "batch 1", "B": "batch 16"}, "metric": "teacher.joules_per_token",
              "better": "lower", "expected": "26 to 3 J/token", "expected_delta": -0.85, "confidence": 0.75,
              "min_n": 3, "citation": "kernels note KE1"}
    fields.update(changes)
    return fields


def rate_definition(**changes):
    fields = {"id": "R1", "track": "learning", "hypothesis": "The new prompt is answered more often",
              "lever": "the prompt wording", "arms": {"A": "old wording", "B": "new wording", "C": "short wording"},
              "kind": "rate", "metric": "student.answer_rate", "better": "higher", "citation": "test"}
    fields.update(changes)
    return fields


ADD_E9 = ["add", "E9", "--title", "Batch sweep", "--track", "kernels", "--hypothesis", "Batching cuts joules per token",
          "--lever", "batch size", "--arm", "A=batch 1", "--arm", "B=batch 16", "--metric", "teacher.joules_per_token",
          "--better", "lower", "--expected", "26 to 3 J/token", "--expected-delta", "-0.85", "--confidence", "0.75",
          "--min-n", "3", "--citation", "kernels note KE1"]

E9_STORED = {"id": "E9", "title": "Batch sweep", "track": "kernels", "task": "T0",
             "hypothesis": "Batching cuts joules per token", "lever": "batch size",
             "arms": {"A": "batch 1", "B": "batch 16"}, "kind": "measure", "metric": "teacher.joules_per_token",
             "better": "lower", "secondary": [], "expected": "26 to 3 J/token", "expected_delta": -0.85,
             "confidence": 0.75, "min_n": 3, "citation": "kernels note KE1", "node": None, "note": None}


# --- add ------------------------------------------------------------------------------------------------------

def test_add_writes_one_ledger_row_holding_the_definition_and_who_wrote_it(capsys):
    code, answer = cli(capsys, *ADD_E9)
    assert code == 0
    assert (answer["ok"], answer["outcome"], answer["id"], answer["seq"], answer["warnings"]) == \
        (True, "added", "E9", 1, [])
    rows = ledger.read()
    assert len(rows) == 1
    row = rows[0]
    assert {key: row[key] for key in ("seq", "actor", "tier", "task", "verb", "outcome", "metrics", "refs")} == {
        "seq": 1, "actor": "opus-3", "tier": "opus", "task": "T0", "verb": "experiment add", "outcome": "ok",
        "metrics": {}, "refs": {"pr": None, "commit": None, "slot": None, "run_id": None}}
    assert row["experiment"] == E9_STORED


def test_list_shows_an_added_experiment_queued_with_its_gain_and_score(capsys):
    cli(capsys, *ADD_E9)
    code, answer = cli(capsys, "list")
    assert code == 0
    assert (answer["outcome"], answer["count"], answer["problems"]) == ("listed", 1, [])
    shown = answer["experiments"][0]
    assert {key: shown[key] for key in ex.DEFINITION_FIELDS} == E9_STORED
    assert (shown["status"], shown["result"], shown["gain"], shown["score"]) == ("queued", None, 0.85, 0.6375)
    assert (shown["added"]["seq"], shown["added"]["actor"], shown["seqs"]) == (1, "opus-3", [1])


def test_with_no_actor_or_tier_set_an_experiment_row_names_its_writer_as_every_other_ledger_row_does(
        capsys, monkeypatch):
    """record.py stamps the writer of every ledger row; an experiment row written outside a sitting carries the same
    actor and tier as any other row written there, so the digest counts one unattributed writer, not two."""
    monkeypatch.delenv("LAB_ACTOR")
    monkeypatch.delenv("LAB_TIER")
    cli(capsys, *ADD_E9)
    record.record("bench.run", "ok")
    experiment_row, other_row = ledger.read()
    assert (experiment_row["actor"], experiment_row["tier"]) == (other_row["actor"], other_row["tier"])
    assert (experiment_row["actor"], experiment_row["tier"]) == (record.UNKNOWN_ACTOR, None)


def test_add_refuses_a_definition_with_problems_listing_every_one_and_writes_nothing():
    with pytest.raises(ex.ExperimentError) as err:
        ex.add(definition(track="speed", arms={"A": "batch 1"}, metric="Teacher JPT", better="faster",
                          expected_delta=-1.5, confidence=1.5, min_n=1, hypothesis="  ", colour="red"))
    assert err.value.problems == [
        "not experiment fields: colour",
        "track must be one of telemetry, learning, plumbing, kernels, not 'speed'",
        "hypothesis must be non-empty text",
        "arms must name at least two arms, the control first (--arm A=... --arm B=...)",
        "metric must be a dotted lowercase name such as teacher.joules_per_token, not 'Teacher JPT'",
        "better must be one of higher, lower, not 'faster'",
        "expected_delta must be a finite fraction above -1 (-0.85 is 85% lower), not -1.5",
        "confidence must be a probability from 0 to 1, not 1.5",
        "min_n must be a whole number of at least 2 for a measure, not 1",
    ]
    assert ledger.read() == []


def test_the_add_verb_answers_a_refusal_as_json_with_exit_code_1(capsys):
    code, answer = cli(capsys, "add", "E9", "--track", "kernels", "--arm", "A=batch 1", "--arm", "B=batch 16")
    assert code == 1
    assert (answer["ok"], answer["outcome"], answer["id"]) == (False, "refused", "E9")
    assert answer["problems"] == ["hypothesis is missing", "lever is missing", "citation is missing",
                                  "metric is missing", "better is missing: one of higher, lower"]
    assert ledger.read() == []


def test_an_argument_argparse_rejects_exits_2_and_writes_nothing(capsys):
    assert lab_cli.main(["experiment", "add", "E9", "--track", "speed"]) == 2
    assert lab_cli.main(["experiment", "add", "E9", "--arm", "no-equals-sign"]) == 2
    capsys.readouterr()
    assert ledger.read() == []


def test_text_fields_are_kept_on_one_line():
    added = ex.add(definition(hypothesis="Batching\ncuts   joules\n  per token", arms={"A": "batch\n1", "B": "b 16"}))
    assert (added["hypothesis"], added["arms"]) == ("Batching cuts joules per token", {"A": "batch 1", "B": "b 16"})


NO_ID_HERE = ("no id was given, and this ledger holds no E-numbered experiment beyond T0's seeded E1 to E5, so it "
              "cannot know which E numbers T0 has handed out since; give the id, one above the highest E number on T0")


def test_two_ledgers_that_cannot_see_t0s_numbers_never_hand_out_an_id(tmp_path):
    """Two clones with their own state dirs used to hand out the same next E number, E6, each from a ledger that
    holds none of the numbers T0 gave out after the seeds. A ledger with no E number of its own, empty or holding
    only the seeds, now refuses an add without an id and writes nothing."""
    empty, seeded = tmp_path / "clone-a" / "state", tmp_path / "clone-b" / "state"
    ex.seed(state_dir=seeded)
    for state_dir, rows in ((empty, 0), (seeded, len(ex.SEEDS))):
        with pytest.raises(ex.ExperimentError) as err:
            ex.add(definition(id=None), state_dir=state_dir)
        assert err.value.problems == [NO_ID_HERE]
        assert len(ledger.read(state_dir=state_dir)) == rows


def test_the_add_verb_without_an_id_on_a_fresh_ledger_is_refused_with_exit_code_1(capsys):
    no_id = [arg for arg in ADD_E9 if arg != "E9"]
    code, answer = cli(capsys, *no_id)
    assert (code, answer["ok"], answer["outcome"], answer["problems"]) == (1, False, "refused", [NO_ID_HERE])
    assert ledger.read() == []


def test_add_without_an_id_takes_the_next_e_number_past_the_ledgers_own_and_the_seeds(capsys):
    """Once the ledger holds an E number beyond the seeds, an add without an id takes the next one, never one of
    T0's E1 to E5."""
    ex.seed()
    ex.add(definition(id="E12"))
    no_id = [arg for arg in ADD_E9 if arg != "E9"]
    assert cli(capsys, *no_id)[1]["id"] == "E13"
    assert cli(capsys, *no_id)[1]["id"] == "E14"
    assert [row["experiment"]["id"] for row in ledger.read()] == ["E1", "E2", "E3", "E4", "E5", "E12", "E13", "E14"]


def test_add_refuses_an_id_that_is_taken(capsys):
    cli(capsys, *ADD_E9)
    code, answer = cli(capsys, *ADD_E9)
    assert (code, answer["outcome"]) == (1, "refused")
    assert answer["problems"] == ["E9 already exists (status queued); change it with add E9 --amend, or pick "
                                  "another id"]
    assert len(ledger.read()) == 1


def test_min_n_defaults_to_the_verdict_floor_of_its_kind():
    assert ex.add(definition(id="M1", min_n=None))["min_n"] == 5
    assert ex.add(rate_definition(id="M2"))["min_n"] == 50


def test_min_n_is_sized_from_the_spread_and_the_baseline(capsys):
    """sd 5 on a mean of 100, to see a 5% cut, takes 17 runs an arm (verdict.runs_per_arm_relative's documented
    case); 5% to 10% takes 435 trials an arm (the method's hand-checked number); a gap so wide that 2 runs would do
    is held at the floor of 5."""
    sized = [arg for arg in ADD_E9 if arg not in ("--min-n", "3")]
    sized[1] = "S1"
    sized[sized.index("--expected-delta") + 1] = "-0.05"
    assert cli(capsys, *sized, "--spread", "5", "--baseline", "100")[1]["experiment"]["min_n"] == 17
    wide = list(sized)
    wide[1] = "S2"
    wide[wide.index("--expected-delta") + 1] = "-0.5"
    assert cli(capsys, *wide, "--spread", "1", "--baseline", "100")[1]["experiment"]["min_n"] == 5
    rate = ["add", "S3", "--track", "learning", "--hypothesis", "h", "--lever", "l", "--arm", "A=old",
            "--arm", "B=new", "--kind", "rate", "--metric", "student.answer_rate", "--better", "higher",
            "--expected-delta", "1.0", "--confidence", "0.5", "--citation", "c", "--baseline", "0.05"]
    assert cli(capsys, *rate)[1]["experiment"]["min_n"] == 435


def test_sizing_refuses_a_min_n_given_beside_it_and_a_baseline_without_a_delta(capsys):
    code, answer = cli(capsys, *ADD_E9, "--spread", "5", "--baseline", "100")
    assert (code, answer["problems"]) == (1, ["give --min-n or --spread with --baseline, not both"])
    no_delta = [arg for arg in ADD_E9 if arg not in ("--min-n", "3", "--expected-delta", "-0.85")]
    code, answer = cli(capsys, *no_delta, "--spread", "5", "--baseline", "100")
    assert (code, answer["problems"]) == (1, ["sizing min_n needs --baseline and --expected-delta (and --spread for "
                                              "a measure)"])
    assert ledger.read() == []


def test_an_experiment_without_an_estimate_is_added_with_a_warning(capsys):
    no_estimate = [arg for arg in ADD_E9 if arg not in ("--expected-delta", "-0.85", "--confidence", "0.75")]
    code, answer = cli(capsys, *no_estimate)
    assert (code, answer["outcome"]) == (0, "added")
    assert answer["warnings"] == ["no expected_delta: the experiment cannot be ranked until it has one",
                                  "no confidence: the experiment cannot be ranked until it has one"]
    assert (answer["experiment"]["gain"], answer["experiment"]["score"]) == (None, None)


# --- amend ----------------------------------------------------------------------------------------------------

def test_amend_changes_only_the_fields_given(capsys):
    cli(capsys, *ADD_E9)
    code, answer = cli(capsys, "add", "E9", "--amend", "--confidence", "0.5", "--secondary", "teacher.decode_tok_s")
    assert (code, answer["outcome"], answer["seq"]) == (0, "amended", 2)
    changed = {**E9_STORED, "confidence": 0.5, "secondary": ["teacher.decode_tok_s"]}
    assert {key: answer["experiment"][key] for key in ex.DEFINITION_FIELDS} == changed
    assert answer["experiment"]["score"] == 0.425
    row = ledger.read()[1]
    assert (row["verb"], row["outcome"], row["task"]) == ("experiment amend", "ok", "T0")
    assert row["experiment"] == {"id": "E9", "changes": {"confidence": 0.5, "secondary": ["teacher.decode_tok_s"]},
                                 "reason": None}


def test_amend_drops_an_experiment_with_its_reason_and_rank_leaves_it_out(capsys):
    cli(capsys, *ADD_E9)
    code, answer = cli(capsys, "add", "E9", "--amend", "--status", "dropped", "--reason", "merged into E10")
    assert code == 0
    assert (answer["experiment"]["status"], answer["experiment"]["status_note"]) == ("dropped", "merged into E10")
    assert cli(capsys, "rank")[1]["experiments"] == []
    assert [e["id"] for e in cli(capsys, "rank", "--all")[1]["experiments"]] == ["E9"]


def test_amend_refuses_an_unknown_id_no_change_a_broken_definition_and_status_without_amend(capsys):
    cli(capsys, *ADD_E9)
    assert cli(capsys, "add", "E99", "--amend", "--confidence", "0.5")[1]["problems"] == \
        ["no experiment E99; known: E9"]
    assert cli(capsys, "add", "E9", "--amend")[1]["problems"] == ["nothing to change"]
    assert cli(capsys, "add", "E9", "--amend", "--confidence", "2")[1]["problems"] == \
        ["confidence must be a probability from 0 to 1, not 2.0"]
    assert cli(capsys, "add", "--amend", "--confidence", "0.5")[1]["problems"] == \
        ["--amend needs the id of the experiment to change"]
    assert cli(capsys, "add", "E10", "--status", "running")[1]["problems"] == \
        ["--status and --reason change an experiment that exists: use them with --amend"]
    assert lab_cli.main(["experiment", "add", "E9", "--amend", "--status", "done"]) == 2
    capsys.readouterr()
    assert len(ledger.read()) == 1


def test_amend_can_resize_min_n_from_a_spread_using_the_stored_delta(capsys):
    cli(capsys, *ADD_E9)
    code, answer = cli(capsys, "add", "E9", "--amend", "--expected-delta", "-0.05", "--spread", "5",
                       "--baseline", "100")
    assert (code, answer["experiment"]["min_n"]) == (0, 17)


# --- gain and rank --------------------------------------------------------------------------------------------

def test_gain_puts_a_doubling_of_speed_level_with_a_halving_of_energy():
    """For tok/s (higher is better) a delta of +1.0 doubles speed, which halves the time, and at constant power the
    joules, per token: gain 0.5, the same as J/token falling by half. A speed halving doubles the cost: gain -1."""
    assert ex.expected_gain(1.0, "higher") == 0.5
    assert ex.expected_gain(-0.5, "lower") == 0.5
    assert ex.expected_gain(-0.5, "higher") == -1.0
    assert ex.expected_gain(1.0, "lower") == -1.0
    assert ex.expected_gain(None, "lower") is None


def test_rank_orders_the_kernels_research_table_by_gain_times_confidence():
    """The sixteen rows go in as the note prints them. Rank recomputes every score: KE13 (0.55 x 0.15 = 0.0825)
    comes before KE14 (0.40 x 0.20 = 0.0800), where the note's rank column puts KE14 first because its score column
    rounds both to 0.08. Every other position matches the note."""
    for eid, gain, confidence in KERNELS_TABLE:
        ex.add(definition(id=eid, expected_delta=-gain, confidence=confidence))
    ranked = ex.rank(ex.load().values())
    assert [e["id"] for e in ranked] == ["KE1", "KE2", "KE3", "KE4", "KE6", "KE5", "KE13", "KE14", "KE7", "KE8",
                                         "KE10", "KE9", "KE12", "KE11", "KE15", "KE16"]
    assert [e["rank"] for e in ranked] == list(range(1, 17))
    assert [e["score"] for e in ranked[:3]] == [0.6375, 0.36, 0.2025]


def test_rank_breaks_a_tie_by_confidence_then_by_the_order_added():
    ex.add(definition(id="X1", expected_delta=-0.5, confidence=0.4))     # 0.5 x 0.4 = 0.2
    ex.add(definition(id="X2", expected_delta=-0.25, confidence=0.8))    # 0.25 x 0.8 = 0.2, surer
    ex.add(definition(id="X3", expected_delta=-0.5, confidence=0.4))     # the same as X1, added later
    assert [e["id"] for e in ex.rank(ex.load().values())] == ["X2", "X1", "X3"]


def test_rank_puts_unscored_experiments_last_with_what_they_need_and_finished_ones_only_with_all(capsys):
    ex.add(definition(id="U1", expected_delta=None, confidence=None))
    ex.add(definition(id="S1", expected_delta=-0.2, confidence=0.5))
    ex.add(definition(id="U2", confidence=None))
    ex.add(definition(id="D1"))
    ex.amend("D1", {"status": "dropped"})
    ex.add(definition(id="R1", expected_delta=-0.1, confidence=0.5))
    ex.amend("R1", {"status": "running"})
    code, answer = cli(capsys, "rank")
    assert code == 0
    assert [(e["id"], e["rank"], e["status"], e["needs"]) for e in answer["experiments"]] == [
        ("S1", 1, "queued", []), ("R1", 2, "running", []),
        ("U1", None, "queued", ["expected_delta", "confidence"]), ("U2", None, "queued", ["confidence"])]
    with_all = cli(capsys, "rank", "--all")[1]["experiments"]
    assert [e["id"] for e in with_all] == ["S1", "R1", "U1", "U2", "D1"]


def test_rank_and_list_filter_by_track(capsys):
    ex.add(definition(id="K1"))
    ex.add(definition(id="T1", track="telemetry"))
    assert [e["id"] for e in cli(capsys, "rank", "--track", "telemetry")[1]["experiments"]] == ["T1"]
    assert [e["id"] for e in cli(capsys, "list", "--track", "kernels")[1]["experiments"]] == ["K1"]
    assert [e["id"] for e in cli(capsys, "list", "--id", "T1", "--id", "K1")[1]["experiments"]] == ["K1", "T1"]


# --- result ---------------------------------------------------------------------------------------------------

def test_result_reads_two_rates_with_the_two_proportion_test(capsys):
    ex.add(rate_definition(arms={"A": "old wording", "B": "new wording"}))
    code, answer = cli(capsys, "result", "R1", "--arm", "A=3/60", "--arm", "B=12/60")
    assert code == 0
    assert (answer["outcome"], answer["label"], answer["winner"], answer["status"]) == \
        ("recorded", "b_ahead", "B", "done")
    comparison = answer["result"]["comparisons"]["B"]
    assert (round(comparison["statistic"], 3), round(comparison["p"], 3)) == (2.484, 0.013)
    assert answer["result"]["arms"] == {"A": {"x": 3, "n": 60, "rate": 0.05}, "B": {"x": 12, "n": 60, "rate": 0.2}}
    assert answer["summary"] == "b_ahead, winner B: B ahead of A: 20.0% (12/60) against 5.0% (3/60) " \
                                "(z = 2.484, p = 0.013)."
    row = ledger.read()[-1]
    assert (row["verb"], row["outcome"], row["task"]) == ("experiment result", "b_ahead", "T0")
    listed = cli(capsys, "list")[1]["experiments"][0]
    assert (listed["status"], listed["result"]["label"], listed["result"]["seq"]) == ("done", "b_ahead", 2)


def test_result_reads_runs_with_welchs_test(capsys):
    """R documents t.test(mpg ~ am, data = mtcars): t = 3.7671 (manual against automatic), df = 18.332,
    p = 0.001374; and t.test(extra ~ group, data = sleep): t = 1.8608, df = 17.776, p = 0.07939."""
    ex.add(definition(id="W1", metric="car.mpg", better="higher", min_n=5))
    code, answer = cli(capsys, "result", "W1", "--arm", "A=" + ",".join(map(str, MPG_AUTOMATIC)),
                       "--arm", "B=" + ",".join(map(str, MPG_MANUAL)))
    assert code == 0
    comparison = answer["result"]["comparisons"]["B"]
    assert (answer["label"], comparison["test"]) == ("b_ahead", "welch")
    assert (round(comparison["statistic"], 4), round(comparison["df"], 3), round(comparison["p"], 6)) == \
        (3.7671, 18.332, 0.001374)
    assert answer["result"]["arms"]["B"]["values"] == MPG_MANUAL
    ex.add(definition(id="W2", metric="patient.extra_sleep_h", better="higher", min_n=5))
    code, answer = cli(capsys, "result", "W2", "--arm", "A=" + ",".join(map(str, SLEEP_1)),
                       "--arm", "B=" + ",".join(map(str, SLEEP_2)))
    comparison = answer["result"]["comparisons"]["B"]
    assert (answer["label"], answer["status"], round(comparison["statistic"], 4), round(comparison["p"], 5)) == \
        ("no_difference", "done", 1.8608, 0.07939)


def test_result_for_a_metric_where_lower_is_better_calls_the_lower_arm_ahead(capsys):
    """Read as joules per token: the arm with the lower mean is ahead, and its t against the control is negative."""
    ex.add(definition(id="W3", min_n=5))
    code, answer = cli(capsys, "result", "W3", "--arm", "A=" + ",".join(map(str, MPG_MANUAL)),
                       "--arm", "B=" + ",".join(map(str, MPG_AUTOMATIC)))
    comparison = answer["result"]["comparisons"]["B"]
    assert (answer["label"], answer["winner"], round(comparison["statistic"], 4)) == ("b_ahead", "B", -3.7671)


def test_too_early_keeps_the_experiment_running_and_final_closes_it(capsys):
    ex.add(definition(id="T1", metric="teacher.decode_tok_s", better="higher", min_n=5))
    code, answer = cli(capsys, "result", "T1", "--arm", "A=7.7,7.9,7.6", "--arm", "B=8.4,8.6,8.5")
    comparison = answer["result"]["comparisons"]["B"]
    assert (code, answer["label"], answer["status"]) == (0, "too_early", "running")
    assert (comparison["n_a"], comparison["n_b"], comparison["min_per_arm"]) == (3, 3, 5)
    assert cli(capsys, "list")[1]["experiments"][0]["status"] == "running"
    code, answer = cli(capsys, "result", "T1", "--arm", "A=7.7,7.9,7.6", "--arm", "B=8.4,8.6,8.5", "--final",
                       "--note", "the slot ran out")
    assert (answer["label"], answer["status"], answer["result"]["final"], answer["result"]["note"]) == \
        ("too_early", "done", True, "the slot ran out")


def test_the_latest_result_is_the_experiments_result(capsys):
    ex.add(definition(id="T1", metric="teacher.decode_tok_s", better="higher", min_n=5))
    cli(capsys, "result", "T1", "--arm", "A=7.7,7.9,7.6", "--arm", "B=8.4,8.6,8.5")
    code, answer = cli(capsys, "result", "T1", "--arm", "A=7.7,7.9,7.6,7.8,7.7", "--arm", "B=8.4,8.6,8.5,8.5,8.4")
    assert (answer["label"], answer["status"]) == ("b_ahead", "done")
    shown = cli(capsys, "list")[1]["experiments"][0]
    assert (shown["result"]["seq"], shown["result"]["arms"]["A"]["n"], shown["seqs"]) == (3, 5, [1, 2, 3])


def test_several_arms_split_alpha_so_a_borderline_arm_is_not_called_ahead(capsys):
    """3 of 60 against 10 of 60: pooled rate 13/120 = 0.1083, standard error sqrt(0.1083 x 0.8917 x 2/60) = 0.05674,
    z = (10/60 - 3/60) / 0.05674 = 2.056, p = 0.0398. Alone that is under 0.05, so B is ahead; read beside a second
    arm, alpha is 0.05 / 2 = 0.025 and the same reading is no difference."""
    ex.add(rate_definition())
    alone = cli(capsys, "result", "R1", "--arm", "A=3/60", "--arm", "B=10/60")[1]
    assert (alone["label"], alone["result"]["alpha"]) == ("b_ahead", 0.05)
    assert round(alone["result"]["comparisons"]["B"]["p"], 4) == 0.0398
    both = cli(capsys, "result", "R1", "--arm", "A=3/60", "--arm", "B=10/60", "--arm", "C=3/60")[1]
    assert (both["label"], both["winner"], both["result"]["alpha"]) == ("no_difference", None, 0.025)
    assert {name: c["label"] for name, c in both["result"]["comparisons"].items()} == \
        {"B": "no_difference", "C": "no_difference"}


def test_the_winner_among_several_arms_ahead_is_the_best_one(capsys):
    """B (12 of 60: z 2.484, p 0.013) and C (13 of 60: p 0.0072) both clear alpha 0.025; C is higher."""
    ex.add(rate_definition())
    answer = cli(capsys, "result", "R1", "--arm", "A=3/60", "--arm", "B=12/60", "--arm", "C=13/60")[1]
    assert (answer["label"], answer["winner"]) == ("b_ahead", "C")


def test_the_control_beating_every_arm_is_a_ahead_with_the_control_as_winner(capsys):
    ex.add(rate_definition(arms={"A": "old wording", "B": "new wording"}))
    answer = cli(capsys, "result", "R1", "--arm", "A=12/60", "--arm", "B=3/60")[1]
    assert (answer["label"], answer["winner"]) == ("a_ahead", "A")


def test_another_control_can_be_named(capsys):
    ex.add(rate_definition())
    answer = cli(capsys, "result", "R1", "--control", "B", "--arm", "B=3/60", "--arm", "C=12/60")[1]
    assert (answer["result"]["control"], list(answer["result"]["comparisons"]), answer["winner"]) == \
        ("B", ["C"], "C")


def test_result_reads_an_arm_from_the_ledger_rows_of_its_runs(capsys):
    """Each row whose refs.run_id is one of the ids gives one value of the experiment's metric, in ledger order;
    rows of other runs and rows without the metric are not read."""
    ex.add(definition(id="T1", metric="teacher.decode_tok_s", better="higher", min_n=5))

    def measured(run_id, value, metric="teacher.decode_tok_s"):
        ledger.append({"actor": "bench", "tier": None, "task": "R11", "verb": "bench.run", "outcome": "ok",
                       "metrics": {metric: value}, "refs": {"run_id": run_id}, "node": "pod-b"})

    for value in (7.7, 7.9, 7.6, 7.8, 7.7):
        measured("spawn-a1" if value < 7.75 else "spawn-a2", value)
    for value in (8.4, 8.6, 8.5, 8.5, 8.4):
        measured("spawn-b1", value)
    measured("spawn-a1", 201.0, metric="teacher.watts_avg")
    measured("spawn-z9", 99.0)
    code, answer = cli(capsys, "result", "T1", "--arm", "A=run:spawn-a1,spawn-a2", "--arm", "B=run:spawn-b1",
                       "--slot", "gpu-b-0007", "--run-id", "exp-T1-1")
    assert code == 0
    assert answer["result"]["arms"]["A"]["values"] == [7.7, 7.9, 7.6, 7.8, 7.7]
    assert answer["result"]["arms"]["B"]["values"] == [8.4, 8.6, 8.5, 8.5, 8.4]
    assert answer["label"] == "b_ahead"
    assert ledger.read()[-1]["refs"] == {"pr": None, "commit": None, "slot": "gpu-b-0007", "run_id": "exp-T1-1"}


def test_result_refuses_readings_it_cannot_use_and_writes_nothing(capsys):
    ex.add(rate_definition(arms={"A": "old wording", "B": "new wording"}))
    ex.add(definition(id="T1", metric="teacher.decode_tok_s", better="higher"))
    before = len(ledger.read())
    refusals = [
        (("result", "R9", "--arm", "A=3/60", "--arm", "B=12/60"), ["no experiment R9"]),
        (("result", "R1", "--arm", "A=3/60", "--arm", "Z=12/60"), ["arm 'Z' is not an arm of R1 (A, B)"]),
        (("result", "R1", "--arm", "B=12/60"), ["the control A has no reading; give --arm A=..."]),
        (("result", "R1", "--arm", "A=3/60"), ["give the control and at least one other arm"]),
        (("result", "R1", "--arm", "A=3/60", "--arm", "B=1.5,2.5"), ["arm B: a rate experiment reads counts "
                                                                      "such as 12/60"]),
        (("result", "T1", "--arm", "A=3/60", "--arm", "B=8,9"), ["arm A: a measure experiment reads run values, "
                                                                  "not counts"]),
        (("result", "T1", "--arm", "A=7.7,fast", "--arm", "B=8,9"), ["arm A: '7.7,fast' is not a list of "
                                                                      "numbers, x/n counts or run:ID"]),
        (("result", "T1", "--arm", "A=,", "--arm", "B=8,9"), ["arm A: no run values in ','"]),
        (("result", "T1", "--arm", "A=run:nothing-here", "--arm", "B=8,9"),
         ["arm A: no ledger row carries teacher.decode_tok_s under run id nothing-here"]),
        (("result", "R1", "--arm", "A=3/60", "--arm", "A=4/60"), ["an arm is read twice: A, A"]),
        (("result", "R1", "--arm", "A=70/60", "--arm", "B=12/60"),
         ["arm A: '70/60' is not x/n counts (70 successes in 60 trials cannot happen)"]),
    ]
    for args, problems in refusals:
        code, answer = cli(capsys, *args)
        assert (code, answer["outcome"], answer["problems"]) == (1, "refused", problems), args
    assert len(ledger.read()) == before


# --- the T0 checklist -----------------------------------------------------------------------------------------

E9_LINE = ("- [ ] E9 (kernels) Batch sweep. Hypothesis: Batching cuts joules per token. Lever: batch size. "
           "Arms: A (control): batch 1; B: batch 16. Metric: teacher.joules_per_token, lower is better. "
           "Expected: 26 to 3 J/token; delta -85%; confidence 0.75; score 0.64. Minimum 3 runs per arm. "
           "Source: kernels note KE1.")


def test_print_emits_one_t0_checklist_line_per_experiment(capsys):
    cli(capsys, *ADD_E9)
    ex.add(definition(id="E10", title=None, expected=None, expected_delta=None, confidence=None, node="pod-b",
                      note="needs a slot on pod B", secondary=["teacher.decode_tok_s"], kind="rate", min_n=60))
    code, text = cli_text(capsys, "list", "--print")
    assert code == 0
    assert text.splitlines() == [
        E9_LINE + " Status: queued.",
        "- [ ] E10 (kernels) Hypothesis: Batching cuts joules per token. Lever: batch size. "
        "Arms: A (control): batch 1; B: batch 16. "
        "Metric: teacher.joules_per_token, lower is better (also teacher.decode_tok_s). "
        "Expected: unscored, needs expected_delta and confidence. Minimum 60 trials per arm. "
        "Source: kernels note KE1. Node: pod-b. Note: needs a slot on pod B. Status: queued.",
    ]


def test_print_checks_a_finished_line_and_writes_its_result(capsys):
    """A: 26, 25, 27 (mean 26, sd 1); B: 3, 4, 2 (mean 3, sd 1). Welch: t = (3 - 26) / sqrt(1/3 + 1/3) = -28.169,
    df = 1 / (0.5^2 / 2 + 0.5^2 / 2) = 4, p far below 0.0001. J/token, so the lower B is ahead."""
    cli(capsys, *ADD_E9)
    cli(capsys, "result", "E9", "--arm", "A=26,25,27", "--arm", "B=3,4,2")
    day = ledger.read()[-1]["ts"][:10]
    code, text = cli_text(capsys, "rank", "--all", "--print")
    assert code == 0
    assert text.splitlines() == [
        E9_LINE.replace("- [ ]", "- [x]") + f" Result {day}: b_ahead, winner B: B ahead of A: 3 (3 runs) against "
                                            "26 (3 runs) (t = -28.169, df = 4.0, p < 0.0001)."]


def test_print_shows_a_running_experiments_latest_reading_and_a_dropped_ones_reason(capsys):
    cli(capsys, *ADD_E9)
    cli(capsys, "result", "E9", "--arm", "A=26,25", "--arm", "B=3,4")
    ex.add(definition(id="E10"))
    ex.amend("E10", {"status": "dropped"}, reason="merged into E9")
    code, text = cli_text(capsys, "list", "--print")
    lines = text.splitlines()
    assert lines[0] == (E9_LINE + " Status: running. Latest reading: too_early: too early for B against A: 3.5 "
                                  "(2 runs) against 25.5 (2 runs); 1 more for A, 1 more for B.")
    assert lines[1] == E9_LINE.replace("- [ ] E9", "- [x] E10") + " Dropped: merged into E9."


def test_print_with_nothing_to_show_says_so(capsys):
    assert cli_text(capsys, "rank", "--print") == (0, "No experiments match.\n")


# --- the seeds ------------------------------------------------------------------------------------------------

def test_seed_adds_the_five_t0_experiments_once(capsys):
    code, answer = cli(capsys, "seed")
    assert (code, answer["outcome"], answer["added"], answer["present"]) == \
        (0, "seeded", ["E1", "E2", "E3", "E4", "E5"], [])
    rows = ledger.read()
    assert [(row["verb"], row["experiment"]["id"], row["source"]) for row in rows] == \
        [("experiment add", f"E{i}", ex.SEED_SOURCE) for i in range(1, 6)]
    assert cli(capsys, "seed")[1]["added"] == []
    assert cli(capsys, "seed")[1]["present"] == ["E1", "E2", "E3", "E4", "E5"]
    assert len(ledger.read()) == 5
    seeded = {e["id"]: e for e in cli(capsys, "list")[1]["experiments"]}
    assert {eid: (e["track"], e["metric"], e["better"], e["min_n"]) for eid, e in seeded.items()} == {
        "E1": ("plumbing", "trainer.samples_per_s", "higher", 5),
        "E2": ("kernels", "teacher.decode_tok_s", "higher", 5),
        "E3": ("kernels", "teacher.joules_per_token", "lower", 3),
        "E4": ("telemetry", "student.joules_per_token", "lower", 32),
        "E5": ("kernels", "teacher.joules_per_token", "lower", 5)}
    assert list(seeded["E3"]["arms"].values()) == ["batch 1", "batch 4", "batch 16", "batch 64"]


def test_the_seeded_queue_ranks_e3_then_e5_and_leaves_the_unestimated_ones_last(capsys):
    """E3 is KE1 of the kernels note (0.85 x 0.75 = 0.6375); E5's FP8 arm is KE14 (0.40 x 0.20 = 0.08). E1 and E4
    have no estimate in any note, and E2 has an expected 4x speedup but no confidence."""
    cli(capsys, "seed")
    ranked = cli(capsys, "rank")[1]["experiments"]
    assert [(e["id"], e["rank"], e["score"], e["needs"]) for e in ranked] == [
        ("E3", 1, 0.6375, []), ("E5", 2, 0.08, []), ("E1", None, None, ["expected_delta", "confidence"]),
        ("E2", None, None, ["confidence"]), ("E4", None, None, ["expected_delta", "confidence"])]
    code, text = cli_text(capsys, "rank", "--print")
    assert [line[:20] for line in text.splitlines()] == ["- [ ] E3 (kernels) B", "- [ ] E5 (kernels) T",
                                                         "- [ ] E1 (plumbing) ", "- [ ] E2 (kernels) v",
                                                         "- [ ] E4 (telemetry)"]


# --- rows the fold cannot use ---------------------------------------------------------------------------------

def raw_row(verb, payload, **changes):
    row = {"actor": "someone", "tier": None, "task": "T0", "verb": verb, "outcome": "ok", "metrics": {}, "refs": {},
           "experiment": payload}
    row.update(changes)
    return row


def test_a_second_add_of_one_id_is_ignored_and_reported(capsys):
    """Two writers that skipped the experiments lock both added E9: the first definition stands."""
    ledger.append(raw_row("experiment add", definition()))
    ledger.append(raw_row("experiment add", definition(hypothesis="something else")))
    experiments, problems = ex.fold(ledger.read())
    assert experiments["E9"]["hypothesis"] == "Batching cuts joules per token"
    assert problems == ["seq 2: E9 was already added at seq 1; this second add is ignored (use add --amend to "
                        "change it)"]
    assert cli(capsys, "list")[1]["problems"] == problems


def test_rows_for_unknown_ids_without_ids_or_with_broken_changes_are_skipped_and_reported():
    ledger.append(raw_row("experiment result", {"id": "E99", "label": "b_ahead"}))
    ledger.append(raw_row("experiment add", None))
    ledger.append(raw_row("experiment add", definition(confidence=7)))
    ledger.append(raw_row("experiment add", definition()))
    ledger.append(raw_row("experiment amend", {"id": "E9", "changes": {"better": "sideways"}}))
    ledger.append(raw_row("experiment amend", {"id": "E9", "changes": {"status": "done"}}))
    ledger.append(raw_row("bench.run", None, metrics={"teacher.decode_tok_s": 7.7}))
    experiments, problems = ex.fold(ledger.read())
    assert list(experiments) == ["E9"]
    assert experiments["E9"]["better"] == "lower" and experiments["E9"]["status"] == "queued"
    assert problems == [
        "seq 1: experiment result for E99, which was never added; skipped",
        "seq 2: experiment add row carries no experiment id; skipped",
        "seq 3: E9 add is not a valid experiment (confidence must be a probability from 0 to 1, not 7); skipped",
        "seq 5: E9 amend would break the experiment (better must be one of higher, lower, not 'sideways'); skipped",
        "seq 6: E9 amend sets status 'done'; skipped",
    ]


# --- through the dispatcher and across processes --------------------------------------------------------------

def test_the_verb_is_refused_while_the_stop_file_exists(state, capsys):
    state.mkdir(parents=True)
    (state / "STOP").write_text("testing the kill switch", encoding="utf-8")
    code, answer = cli(capsys, "seed")
    assert (code, answer["ok"], answer["outcome"]) == (1, False, "stopped")
    assert ledger.read() == []


def test_python_dash_m_lab_experiment_answers_json_and_prints_the_checklist(state):
    ex.seed()
    env = {**os.environ, "LAB_STATE_DIR": str(state)}
    listed = subprocess.run([sys.executable, "-m", "lab", "experiment", "list"], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert listed.returncode == 0, listed.stderr
    answer = json.loads(listed.stdout)
    assert (answer["outcome"], answer["count"]) == ("listed", 5)
    printed = subprocess.run([sys.executable, "-m", "lab", "experiment", "rank", "--print"], cwd=REPO_ROOT, env=env,
                             capture_output=True, text=True, timeout=120)
    assert printed.returncode == 0, printed.stderr
    assert [line.split(" ")[3] for line in printed.stdout.splitlines()] == ["E3", "E5", "E1", "E2", "E4"]
    helped = subprocess.run([sys.executable, "-m", "lab", "--help"], cwd=REPO_ROOT, env=env, capture_output=True,
                            text=True, timeout=120)
    shown = [line.split(maxsplit=1) for line in helped.stdout.splitlines() if line.strip().startswith("experiment ")]
    assert shown == [["experiment", "Experiments as ledger rows: add, list, result (read by the verdict math), rank, "
                                    "and T0's checklist (--print)."]]


def _add_many(state: str, start, out: str) -> None:
    """Spawned child: wait for the other child, then add ADDS_PER_CHILD experiments without ids."""
    start.wait(START_TIMEOUT)
    ids = [ex.add(definition(id=None), state_dir=state)["id"] for _ in range(ADDS_PER_CHILD)]
    Path(out).write_text(json.dumps(ids), encoding="utf-8")


def test_two_processes_adding_at_once_never_get_the_same_id(tmp_path, state):
    # Only a ledger that already holds an E number beyond the seeds hands out the next one.
    ex.add(definition(id="E6"), state_dir=state)
    start = SPAWN.Barrier(2)
    children = {name: SPAWN.Process(target=_add_many, args=(str(state), start, str(tmp_path / f"{name}.json")),
                                    daemon=True) for name in ("a", "b")}
    for child in children.values():
        child.start()
    for child in children.values():
        child.join(JOIN_TIMEOUT)
    assert {name: child.exitcode for name, child in children.items()} == {"a": 0, "b": 0}
    got = [eid for name in ("a", "b") for eid in json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))]
    expected = [f"E{n}" for n in range(7, 7 + 2 * ADDS_PER_CHILD)]
    assert sorted(got, key=lambda eid: int(eid[1:])) == expected
    assert [row["experiment"]["id"] for row in ledger.read(state_dir=state)] == ["E6", *expected]
    assert ex.fold(ledger.read(state_dir=state))[1] == []
