"""Tests for the review verb (python -m lab review): the owner's morning file for one UTC day.

The main fixtures are the recorded ledger, fixtures/lab/alerts/ledger.jsonl (44 rows written through
ledger.append for the 9th and 10th of October 2026), and fixtures/lab/alerts/open_prs.json, the output of
gh pr list as it was on the 10th. The smaller cases build their own few rows through ledger.append. Every test
gets its own state dir through LAB_STATE_DIR and its own folder of rulings, so none of them reads or writes
the real lab/state or lab/refine.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab import __main__ as lab_cli
from lab.machine import alerts, ledger, review

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDED = REPO_ROOT / "fixtures" / "lab" / "alerts" / "ledger.jsonl"
OPEN_PRS = REPO_ROOT / "fixtures" / "lab" / "alerts" / "open_prs.json"
UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 10, 15, 0, 0, tzinfo=UTC)
DAY = "2026-10-10"

ALERT_LINES = [
    "- 15:00Z **teacher.decode_tok_s regressed** pod-a: 7.8 -> 6.9 tok/s (-11.5%, higher is better); "
    "prompt=text, thinking=off; read 2026-10-10 07:30Z (ledger:3 -> ledger:14)",
    "- 15:00Z **Sitting s-son-2 ended with error** sonnet, task O33, exit code 1, after 15.2 min; "
    "ended 2026-10-10 08:20Z (ledger:22)",
    "- 15:00Z **Slot q-1 ran past its cap** pod-b, task R20: 30 of 15 min, started 2026-10-10 10:10Z, "
    "ended 2026-10-10 10:40Z (ledger:31)",
    "- 15:00Z **Pod pod-b stopped unexpectedly** pod status reported stopped at 2026-10-10 12:00Z and no pod "
    "stop was recorded since it was up (ledger:40)",
    "- 15:00Z **Slot q-3 is past its cap** pod-b, task R20: 150 of 60 min, started 2026-10-10 12:30Z, "
    "still running (ledger:42)",
    "- 15:00Z **Credits on pod-a are projected to run out** 4,667 left at 3.70 credits/min, about 21.0 h "
    "(alert horizon 24 h), from the readings up to 2026-10-10 13:30Z (ledger:26 to ledger:43)",
]


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """A private state dir for every test."""
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    return state


@pytest.fixture
def refine_root(tmp_path):
    """An empty folder of tracks; no ruling is waiting in it."""
    root = tmp_path / "refine"
    root.mkdir()
    return root


def record(state: Path) -> None:
    (state / ledger.LEDGER_FILE).write_bytes(RECORDED.read_bytes())


def put(state: Path, ts: str, verb: str, outcome: str, *, actor: str = "t", tier: str | None = None,
        task: str | None = None, metrics: dict | None = None, refs: dict | None = None, **extra) -> int:
    row = {"ts": ts, "actor": actor, "tier": tier, "task": task, "verb": verb, "outcome": outcome,
           "metrics": metrics or {}, "refs": refs or {}, **extra}
    return ledger.append(row, state_dir=state)


def at(hour: int, minute: int = 0, day: int = 10, second: int = 0) -> str:
    return f"2026-10-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"


def moment(hour: int, minute: int = 0, day: int = 10) -> dt.datetime:
    return dt.datetime(2026, 10, day, hour, minute, tzinfo=UTC)


def build(state: Path, refine_root: Path, *, day: str = DAY, now: dt.datetime = NOW, prs=None, scan: bool = False):
    return review.build(day, state_dir=state, refine_root=refine_root, prs=prs, now=now, scan=scan)


def sections(text: str) -> dict[str, list[str]]:
    """The review's headings (any level) and the non-blank lines under each, up to the next heading."""
    found: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("#"):
            current = line
            found[current] = []
        elif current is not None and line.strip():
            found[current].append(line)
    return found


def run_cli(argv, capsys):
    code = lab_cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def one_json_line(text: str) -> dict:
    lines = text.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    return json.loads(lines[0])


RULING = (
    "## {n}. {question} ({origin})\n"
    "\n"
    "Now: {now}\n"
    "Recommend: {recommend}\n"
    "Source of tension: {tension}\n"
    "Answer: {answer}\n"
)


def ruling(n: int, question: str, *, answer: str = "____", now: str = "the default", recommend: str = "do it",
           origin: str = "pass 1") -> str:
    return RULING.format(n=n, question=question, now=now, recommend=recommend, tension="two sources", answer=answer,
                         origin=origin)


def write_rulings(root: Path, track: str, *entries: str) -> None:
    folder = root / track
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "OWNER_RULINGS.md").write_text(f"# Owner rulings for the {track} track\n\nA header.\n\n"
                                             + "\n".join(entries), encoding="utf-8")


# --- the recorded day ----------------------------------------------------------------------------------

def test_the_recorded_day_reviews_as_these_sections_exactly(isolated_state, refine_root):
    record(isolated_state)
    built = build(isolated_state, refine_root, prs=review.file_open_prs(OPEN_PRS), scan=True)
    found = sections(built.text)
    assert built.text.splitlines()[0] == "# Lab review, 2026-10-10 (UTC)"
    assert built.text.splitlines()[2] == (
        "Status: 35 ledger rows, 3 sittings ended (1 error, 1 ok, 1 timeout), costing at least $3.52 "
        "(1 cost unknown), 6 alerts, 8 metrics moved, 0 rulings waiting, 1 complaint, 4 PRs open.")
    assert list(found) == [
        "# Lab review, 2026-10-10 (UTC)", "## Alerts", "## What happened", "### Sittings", "### Other activity",
        "### Tasks touched", "## Numbers moved", "## GPU slots and complaints", "### Slots", "### Complaints",
        "## Rulings waiting", "## PRs open", "## Problems"]
    assert found["## Alerts"] == ALERT_LINES
    assert found["### Sittings"] == [
        "- 08:20:10Z s-son-2 (sonnet, task O33): error, 15.2 min, $0.42 (ledger:22)",
        "- 08:41:30Z s-opus-1 (opus, task O20): ok, 41.5 min, $3.10 (ledger:23)",
        "- 09:30:00Z s-son-3 (sonnet, task O16): timeout, 30.0 min, cost unknown (ledger:29)",
        "- 09:10:00Z s-opus-4 (opus, task O22): started, no end row (ledger:28)",
    ]
    assert found["### Other activity"] == [
        "- pod status: 8 (running 6, stopped 2)",
        "- bench run: 6 (ok 6)",
        "- slot start: 5 (running 4, refused 1)",
        "- pod launch: 2 (ok 2)",
        "- slot done: 2 (ok 2)",
        "- handoff: 1 (handoff 1)",
        "- pod stop: 1 (ok 1)",
        "- slot complain: 1 (ok 1)",
        "- slot extend: 1 (ok 1)",
        "- slot request: 1 (queued 1)",
    ]
    assert found["### Tasks touched"] == [
        "- R24: 1 row",
        "- R20: 4 rows",
        "- R23: 1 row",
        "- R22: 3 rows",
        "- R21: 1 row",
        "- O16: 2 rows",
        "- O22: 1 row",
        "- O20: 2 rows, PR 61, commit 9a3db18",
        "- O33: 2 rows",
        "- R11: 6 rows",
    ]
    assert found["## Numbers moved"] == [
        "- trainer.samples_per_s (pod-b; script=lab/smoke_lora.py, steps=20): 0.51 -> 0.6 samples/s "
        "(+17.6%, better) (ledger:5 -> ledger:16)",
        "- pod.balance_credits (pod-a): 6000 -> 5000 credits (-16.7%, down) (ledger:26 -> ledger:43)",
        "- teacher.joules_per_token (pod-a; derived=True, formula=watts_avg / decode_tok_s): 25.97 -> 30 J/token "
        "(+15.5%, worse) (low confidence) (ledger:7 -> ledger:18)",
        "- teacher.decode_tok_s (pod-a; prompt=text, thinking=off): 7.8 -> 6.9 tok/s (-11.5%, worse) "
        "(ledger:3 -> ledger:14)",
        "- teacher.vram_pct (pod-a; ctx=16384): 84 -> 90 % (+7.1%, up) (ledger:6 -> ledger:17)",
        "- teacher.ttft_s (pod-a; prompt=text, thinking=off): 0.55 -> 0.57 s (+3.6%, worse) "
        "(ledger:3 -> ledger:14)",
        "- teacher.decode_tok_s (pod-a; image=1280x720 JPEG, prompt=image): 6.6 -> 6.5 tok/s (-1.5%, worse) "
        "(ledger:4 -> ledger:15)",
        "- pod.balance_credits (pod-b): 59000 -> 58500 credits (-0.8%, down) (ledger:27 -> ledger:36)",
        "First readings:",
        "- teacher.watts_avg (pod-a; model=RedHatAI/Qwen3.8-27B-INT4, while=generating): first reading 200 W "
        "(ledger:19)",
    ]
    assert found["### Slots"] == [
        "- slot q-1 on pod-b, task R20: 30 of 15 min, done, past its cap (ledger:31)",
        "- slot q-2 on pod-b, task R22: 25 of 30 min (extended), done (ledger:34)",
        "- slot q-3 on pod-b, task R20: 150 of 60 min, still running, past its cap (ledger:42)",
        "- slot q-5 on pod-b, task R24: 10 of 15 min, still running (ledger:44)",
    ]
    assert found["### Complaints"] == [
        "- 10:20:00Z slot q-9 (sonnet-7): waiting for pod B for 31 minutes behind q-1 (ledger:32)"]
    assert found["## Rulings waiting"] == ["- none"]
    assert found["## PRs open"] == [
        "- #88 O19: Backfill the lab setup numbers and facts into the ledger (ctx/orchestration/ledger-backfill), "
        "by IlanDanial, open 0.6 h, https://github.com/raohemdutt/Knighthacks26/pull/88",
        "- #92 O17: Digest computed from ledger and board (python -m lab digest) "
        "(ctx/orchestration/digest), by IlanDanial, open 0.5 h, https://github.com/raohemdutt/Knighthacks26/pull/92",
        "- #93 O14: Rulings flow (list, apply, mirror to Notion) (ctx/orchestration/rulings-flow), "
        "by IlanDanial, open 0.5 h, https://github.com/raohemdutt/Knighthacks26/pull/93",
        "- #94 O12: Merge log and the checks the merge pass relies on (ctx/lab/refine-merge), "
        "by IlanDanial, open 0.5 h, https://github.com/raohemdutt/Knighthacks26/pull/94",
    ]
    assert found["## Problems"] == ["- none"]


def test_the_verb_writes_the_file_whole_and_answers_with_counts_not_text(isolated_state, refine_root):
    record(isolated_state)
    answer = review.review(DAY, state_dir=isolated_state, refine_root=refine_root,
                           prs=review.file_open_prs(OPEN_PRS), now=NOW)
    assert answer == {
        "ok": True, "outcome": "written", "day": DAY, "file": "review/2026-10-10.md", "problems": 0,
        "counts": {"rows": 35, "sittings": 3, "sittings_never_ended": 1, "alerts": 6, "metrics_moved": 8,
                   "metrics_new": 1, "rulings_waiting": 0, "complaints": 1, "prs_open": 4, "cost_usd": 3.52,
                   "cost_unknown": 1}}
    written = (isolated_state / "review" / "2026-10-10.md").read_text(encoding="utf-8")
    assert written == build(isolated_state, refine_root, prs=review.file_open_prs(OPEN_PRS), scan=True).text
    assert sorted(path.name for path in (isolated_state / "review").iterdir()) == ["2026-10-10.md"]
    assert "\r" not in written


def test_a_review_of_today_first_brings_the_alerts_file_up_to_date(isolated_state, refine_root):
    record(isolated_state)
    assert not (isolated_state / "alerts").exists()
    review.review(DAY, state_dir=isolated_state, refine_root=refine_root, now=NOW)
    assert [alerts.parse_line(line).key for line in alerts.day_lines(DAY, isolated_state)] == [
        "metric_regressed:14:teacher.decode_tok_s", "sitting_error:22", "slot_over_cap:q-1", "pod_stopped:40",
        "slot_over_cap:q-3", "credits_low:pod-a:2026-10-10"]


def test_without_the_scan_the_alerts_file_is_read_as_it_is(isolated_state, refine_root):
    record(isolated_state)
    built = build(isolated_state, refine_root, scan=False)
    assert not (isolated_state / "alerts").exists()
    assert sections(built.text)["## Alerts"] == ["- none"]
    assert "0 alerts" in built.text.splitlines()[2]


def test_a_review_of_an_earlier_day_only_reads(isolated_state, refine_root):
    record(isolated_state)
    review.review("2026-10-09", state_dir=isolated_state, refine_root=refine_root, now=NOW)
    assert not (isolated_state / "alerts").exists()
    found = sections((isolated_state / "review" / "2026-10-09.md").read_text(encoding="utf-8"))
    assert found["### Sittings"] == ["- 08:30:00Z s-opus-0 (opus, task O15): error, 30.0 min, cost unknown (ledger:2)"]
    assert found["## Alerts"] == ["- none"]


def test_the_numbers_of_a_day_with_nothing_before_it_are_all_first_readings(isolated_state, refine_root):
    record(isolated_state)
    built = build(isolated_state, refine_root, day="2026-10-09")
    assert sections(built.text)["## Numbers moved"] == [
        "First readings:",
        "- teacher.decode_tok_s (pod-a; image=1280x720 JPEG, prompt=image): first reading 6.6 tok/s (ledger:4)",
        "- teacher.decode_tok_s (pod-a; prompt=text, thinking=off): first reading 7.8 tok/s (ledger:3)",
        "- teacher.joules_per_token (pod-a; derived=True, formula=watts_avg / decode_tok_s): first reading "
        "25.97 J/token (low confidence) (ledger:7)",
        "- teacher.ttft_s (pod-a; prompt=text, thinking=off): first reading 0.55 s (ledger:3)",
        "- teacher.vram_pct (pod-a; ctx=16384): first reading 84 % (ledger:6)",
        "- trainer.samples_per_s (pod-b; script=lab/smoke_lora.py, steps=20): first reading 0.51 samples/s "
        "(ledger:5)",
    ]
    assert built.text.splitlines()[2].startswith("Status: 9 ledger rows, 1 sitting ended (1 error), "
                                                 "costing at least $0.00 (1 cost unknown), 0 alerts, 0 metrics moved,")


def test_the_same_inputs_give_the_same_file_and_a_new_row_changes_it(isolated_state, refine_root):
    record(isolated_state)
    first = build(isolated_state, refine_root).text
    assert build(isolated_state, refine_root).text == first
    put(isolated_state, at(15, 1), "ledger add", "ok", task="O99")
    again = build(isolated_state, refine_root, now=moment(15, 2)).text
    assert again != first and "- O99: 1 row" in again and "- O99: 1 row" not in first


def test_a_second_run_replaces_the_first_whole(isolated_state, refine_root):
    record(isolated_state)
    review.review(DAY, state_dir=isolated_state, refine_root=refine_root, now=NOW)
    path = isolated_state / "review" / "2026-10-10.md"
    first = path.read_text(encoding="utf-8")
    put(isolated_state, at(15, 1), "ledger add", "ok", task="O99")
    review.review(DAY, state_dir=isolated_state, refine_root=refine_root, now=moment(15, 2))
    second = path.read_text(encoding="utf-8")
    assert second.startswith("# Lab review, 2026-10-10 (UTC)") and second.count("# Lab review") == 1
    assert "36 ledger rows" in second and "35 ledger rows" in first
    assert sorted(item.name for item in path.parent.iterdir()) == ["2026-10-10.md"]


def test_a_day_with_no_rows_and_no_ledger_reviews_as_nothing(isolated_state, refine_root):
    built = build(isolated_state, refine_root)
    assert built.text.splitlines()[2] == (
        "Status: 0 ledger rows, 0 sittings ended, costing $0.00, 0 alerts, 0 metrics moved, 0 rulings waiting, "
        "0 complaints, PRs open unknown.")
    found = sections(built.text)
    for heading in ("## Alerts", "### Sittings", "### Other activity", "### Tasks touched", "## Numbers moved",
                    "### Slots", "### Complaints", "## Rulings waiting", "## Problems"):
        assert found[heading] == ["- none"], heading
    assert found["## PRs open"] == ["- not asked for"]


def test_rows_of_other_days_stay_out_of_the_day(isolated_state, refine_root):
    put(isolated_state, at(23, 59, 9, 59), "sit", "error", tier="opus", task="O1", refs={"run_id": "late"})
    put(isolated_state, at(0, 0, 10), "sit", "ok", tier="opus", task="O2", refs={"run_id": "first"})
    put(isolated_state, at(23, 59, 10, 59), "sit", "ok", tier="opus", task="O3", refs={"run_id": "last"})
    put(isolated_state, at(0, 0, 11), "sit", "ok", tier="opus", task="O4", refs={"run_id": "next"})
    built = build(isolated_state, refine_root, now=moment(6, 0, 11))
    assert sections(built.text)["### Sittings"] == [
        "- 00:00:00Z first (opus, task O2): ok, duration unknown, cost unknown (ledger:2)",
        "- 23:59:59Z last (opus, task O3): ok, duration unknown, cost unknown (ledger:3)"]


def test_rows_after_now_are_not_yet_part_of_the_account(isolated_state, refine_root):
    record(isolated_state)
    built = build(isolated_state, refine_root, now=moment(9, 0))
    assert sections(built.text)["### Sittings"] == [
        "- 08:20:10Z s-son-2 (sonnet, task O33): error, 15.2 min, $0.42 (ledger:22)",
        "- 08:41:30Z s-opus-1 (opus, task O20): ok, 41.5 min, $3.10 (ledger:23)",
        "- 09:00:00Z s-son-3 (sonnet, task O16): started, no end row (ledger:25)"]
    assert built.text.splitlines()[2] == (
        "Status: 17 ledger rows, 2 sittings ended (1 error, 1 ok), costing $3.52, 0 alerts, 6 metrics moved, "
        "0 rulings waiting, 0 complaints, PRs open unknown.")


# --- sittings ----------------------------------------------------------------------------------------

def test_a_sitting_that_started_one_day_and_ended_the_next_is_neither_lost_nor_open(isolated_state, refine_root):
    put(isolated_state, at(23, 50, 9), "sit", "started", tier="opus", task="O1", refs={"run_id": "s-night"})
    put(isolated_state, at(0, 10, 10), "sit", "ok", tier="opus", task="O1", refs={"run_id": "s-night"},
        metrics={"sitting.duration_s": 1200, "sitting.cost_usd": 2.5})
    first_day = build(isolated_state, refine_root, day="2026-10-09")
    second_day = build(isolated_state, refine_root)
    assert sections(first_day.text)["### Sittings"] == ["- none"]
    assert "never ended" not in first_day.text
    assert sections(second_day.text)["### Sittings"] == [
        "- 00:10:00Z s-night (opus, task O1): ok, 20.0 min, $2.50 (ledger:2)"]


def test_a_sitting_with_no_end_row_is_named_in_the_status_and_in_the_list(isolated_state, refine_root):
    put(isolated_state, at(9), "sit", "started", tier="sonnet", refs={"run_id": "s-lost"})
    built = build(isolated_state, refine_root)
    assert sections(built.text)["### Sittings"] == [
        "- 09:00:00Z s-lost (sonnet, no task): started, no end row (ledger:1)"]
    assert "1 sitting started that day and never ended; the launcher may have died." in built.text
    assert built.counts["sittings_never_ended"] == 1 and built.counts["sittings"] == 0


def test_every_cost_known_makes_the_total_exact_and_one_unknown_makes_it_a_floor(isolated_state, refine_root):
    put(isolated_state, at(9), "sit", "ok", refs={"run_id": "a"}, metrics={"sitting.cost_usd": 1.25})
    put(isolated_state, at(10), "sit", "ok", refs={"run_id": "b"}, metrics={"sitting.cost_usd": 0.5})
    exact = build(isolated_state, refine_root)
    put(isolated_state, at(11), "sit", "ok", refs={"run_id": "c"})
    floor = build(isolated_state, refine_root)
    assert "2 sittings ended (2 ok), costing $1.75," in exact.text
    assert "cost_unknown" not in exact.counts
    assert "3 sittings ended (3 ok), costing at least $1.75 (1 cost unknown)," in floor.text
    assert (floor.counts["cost_usd"], floor.counts["cost_unknown"]) == (1.75, 1)


def test_a_sitting_row_without_a_run_id_is_named_by_its_actor_and_never_pairs_with_another(isolated_state, refine_root):
    put(isolated_state, at(9), "sit", "started", actor="lab sit", tier="opus")
    put(isolated_state, at(10), "sit", "ok", actor="lab sit", tier="opus")
    built = build(isolated_state, refine_root)
    assert sections(built.text)["### Sittings"] == [
        "- 10:00:00Z lab_sit (opus, no task): ok, duration unknown, cost unknown (ledger:2)",
        "- 09:00:00Z lab_sit (opus, no task): started, no end row (ledger:1)"]


def test_more_sittings_than_the_cap_are_counted_not_listed(isolated_state, refine_root):
    for index in range(review.MAX_SITTINGS + 4):
        put(isolated_state, at(1 + index // 60, index % 60), "sit", "ok", refs={"run_id": f"s-{index}"})
    built = build(isolated_state, refine_root)
    lines = sections(built.text)["### Sittings"]
    assert len(lines) == review.MAX_SITTINGS + 1
    assert lines[-1] == "- +4 more"
    assert built.counts["sittings"] == review.MAX_SITTINGS + 4


# --- numbers moved -----------------------------------------------------------------------------------

def reading(state, ts, value, *, name="teacher.decode_tok_s", unit="tok/s", node="pod-a", conditions=None, **extra):
    return put(state, ts, "bench run", "ok", node=node, conditions=conditions or {}, metrics={name: value},
               units={name: unit} if unit else {}, **extra)


def moved(state, refine_root, **options):
    return sections(build(state, refine_root, **options).text)["## Numbers moved"]


def test_the_newest_reading_of_the_day_is_set_against_the_newest_one_before_it(isolated_state, refine_root):
    reading(isolated_state, at(20, day=9), 10)
    reading(isolated_state, at(8), 11)                  # not the one compared: a later one is
    reading(isolated_state, at(12), 12)
    assert moved(isolated_state, refine_root) == [
        "- teacher.decode_tok_s (pod-a): 10 -> 12 tok/s (+20.0%, better) (ledger:1 -> ledger:3)"]


def test_with_nothing_before_the_day_the_first_reading_of_the_day_is_the_baseline(isolated_state, refine_root):
    reading(isolated_state, at(8), 10)
    reading(isolated_state, at(12), 8)
    assert moved(isolated_state, refine_root) == [
        "- teacher.decode_tok_s (pod-a): 10 -> 8 tok/s (-20.0%, worse) (ledger:1 -> ledger:2)"]


def test_a_series_that_did_not_change_is_counted_and_not_listed(isolated_state, refine_root):
    reading(isolated_state, at(20, day=9), 10)
    reading(isolated_state, at(8), 10)
    reading(isolated_state, at(20, day=9), 3, name="other.rate", unit="tok/s")
    reading(isolated_state, at(8), 3, name="other.rate", unit="tok/s")
    built = build(isolated_state, refine_root)
    assert sections(built.text)["## Numbers moved"] == ["- 2 series did not change"]
    assert built.counts["metrics_moved"] == 0


def test_the_direction_words_follow_the_metric_and_a_metric_with_no_better_side_reads_up_or_down(isolated_state,
                                                                                                refine_root):
    for name, unit, before, after in (("exp.rate_up", "tok/s", 1, 2), ("exp.rate_down", "tok/s", 2, 1),
                                      ("exp.balance_up", "credits", 10, 20), ("exp.balance_down", "credits", 20, 10)):
        reading(isolated_state, at(20, day=9), before, name=name, unit=unit)
        reading(isolated_state, at(8), after, name=name, unit=unit)
    assert moved(isolated_state, refine_root) == [
        "- exp.balance_up (pod-a): 10 -> 20 credits (+100.0%, up) (ledger:5 -> ledger:6)",
        "- exp.rate_up (pod-a): 1 -> 2 tok/s (+100.0%, better) (ledger:1 -> ledger:2)",
        "- exp.balance_down (pod-a): 20 -> 10 credits (-50.0%, down) (ledger:7 -> ledger:8)",
        "- exp.rate_down (pod-a): 2 -> 1 tok/s (-50.0%, worse) (ledger:3 -> ledger:4)"]


def test_a_baseline_of_zero_has_no_percent(isolated_state, refine_root):
    reading(isolated_state, at(20, day=9), 0, name="exp.errors", unit="count")
    reading(isolated_state, at(8), 3, name="exp.errors", unit="count")
    assert moved(isolated_state, refine_root) == [
        "- exp.errors (pod-a): 0 -> 3 count (from 0, up) (ledger:1 -> ledger:2)"]


def test_the_numbers_of_a_sitting_or_a_slot_are_events_and_never_series(isolated_state, refine_root):
    put(isolated_state, at(20, day=9), "sit", "ok", metrics={"sitting.duration_s": 100})
    put(isolated_state, at(8), "sit", "ok", metrics={"sitting.duration_s": 900})
    put(isolated_state, at(9), "slot done", "ok", metrics={"slot.minutes": 15}, refs={"slot": "x"})
    assert moved(isolated_state, refine_root) == ["- none"]


def test_series_with_other_conditions_or_nodes_are_not_compared_with_each_other(isolated_state, refine_root):
    reading(isolated_state, at(20, day=9), 10, conditions={"prompt": "text"})
    reading(isolated_state, at(8), 5, conditions={"prompt": "image"})
    reading(isolated_state, at(9), 7, node="pod-b", conditions={"prompt": "text"})
    assert moved(isolated_state, refine_root) == [
        "First readings:",
        "- teacher.decode_tok_s (pod-a; prompt=image): first reading 5 tok/s (ledger:2)",
        "- teacher.decode_tok_s (pod-b; prompt=text): first reading 7 tok/s (ledger:3)"]
    counts = build(isolated_state, refine_root).counts
    assert (counts["metrics_moved"], counts["metrics_new"]) == (0, 2)


def test_a_reading_a_later_day_does_not_move_an_earlier_days_numbers(isolated_state, refine_root):
    reading(isolated_state, at(8), 10)
    reading(isolated_state, at(8, day=11), 1)
    assert moved(isolated_state, refine_root, now=moment(9, day=11)) == [
        "First readings:",
        "- teacher.decode_tok_s (pod-a): first reading 10 tok/s (ledger:1)"]


def test_the_moved_list_has_a_cap_and_says_how_many_it_left_out(isolated_state, refine_root):
    for index in range(review.MAX_MOVED + 3):
        reading(isolated_state, at(20, day=9), 10, name=f"exp.m{index:02d}", unit="tok/s")
        reading(isolated_state, at(8), 10 + index + 1, name=f"exp.m{index:02d}", unit="tok/s")
    lines = moved(isolated_state, refine_root)
    assert len(lines) == review.MAX_MOVED + 1
    assert lines[0].startswith("- exp.m17 ") and lines[-1] == "- +3 more"


# --- GPU slots and complaints --------------------------------------------------------------------------

def test_a_slot_that_began_the_day_before_and_ended_today_is_listed_today(isolated_state, refine_root):
    put(isolated_state, at(23, 50, 9), "slot start", "running", node="pod-b", task="R20",
        metrics={"slot.minutes": 15}, refs={"slot": "q-night"})
    put(isolated_state, at(0, 3, 10), "slot done", "ok", refs={"slot": "q-night"})
    assert sections(build(isolated_state, refine_root).text)["### Slots"] == [
        "- slot q-night on pod-b, task R20: 13 of 15 min, done (ledger:1)"]


def test_a_slot_still_running_when_a_past_day_ended_is_measured_to_the_end_of_that_day(isolated_state, refine_root):
    put(isolated_state, at(23, 0, 9), "slot start", "running", node="pod-b", task="R20",
        metrics={"slot.minutes": 15}, refs={"slot": "q-open"})
    built = build(isolated_state, refine_root, day="2026-10-09", now=moment(6, day=10))
    assert sections(built.text)["### Slots"] == [
        "- slot q-open on pod-b, task R20: 60 of 15 min, still running, past its cap (ledger:1)"]


def test_a_complaint_is_clipped_to_one_line_and_a_missing_text_is_said(isolated_state, refine_root):
    put(isolated_state, at(9), "slot complain", "ok", actor="sonnet-1", refs={"slot": "q-1"},
        note="waiting\nfor the card " + "x" * 300)
    put(isolated_state, at(10), "slot complain", "ok", actor="sonnet-2", refs={"slot": "q-2"})
    put(isolated_state, at(11), "slot complain", "ok", actor="sonnet-3", refs={"slot": "q-3"},
        complaint="named in the complaint field")
    lines = sections(build(isolated_state, refine_root).text)["### Complaints"]
    assert lines[0] == "- 09:00:00Z slot q-1 (sonnet-1): waiting for the card " + "x" * 176 + "... (ledger:1)"
    assert len(lines[0].split(": ", 1)[1].rsplit(" (ledger", 1)[0]) == review.LINE_CHARS
    assert lines[1:] == ["- 10:00:00Z slot q-2 (sonnet-2): (no text) (ledger:2)",
                         "- 11:00:00Z slot q-3 (sonnet-3): named in the complaint field (ledger:3)"]


def test_a_complaint_that_the_queue_refused_is_not_a_complaint(isolated_state, refine_root):
    put(isolated_state, at(9), "slot complain", "refused", refs={"slot": "q-1"}, note="never filed")
    built = build(isolated_state, refine_root)
    assert sections(built.text)["### Complaints"] == ["- none"]
    assert built.counts["complaints"] == 0


def test_complaints_have_a_cap_and_the_count_still_counts_all(isolated_state, refine_root):
    for index in range(review.MAX_COMPLAINTS + 2):
        put(isolated_state, at(1, index), "slot complain", "ok", refs={"slot": f"q-{index}"}, note="slow")
    built = build(isolated_state, refine_root)
    lines = sections(built.text)["### Complaints"]
    assert len(lines) == review.MAX_COMPLAINTS + 1 and lines[-1] == "- +2 more"
    assert built.counts["complaints"] == review.MAX_COMPLAINTS + 2


# --- rulings -------------------------------------------------------------------------------------------

def test_only_the_rulings_without_an_answer_are_waiting_and_they_say_what_applies_meanwhile(refine_root):
    write_rulings(refine_root, "kernels",
                  ruling(1, "Does a win beat the spread?", answer="the spread (owner, 2026-10-10)"),
                  ruling(2, "Is a practitioner post a source?", now="measured and paper sources only",
                         recommend="allow it, labeled"))
    write_rulings(refine_root, "learning", ruling(1, "Which split is held out?", now="by day", recommend="by episode"))
    lines, total, problems = review.rulings_section(refine_root)
    assert total == 2 and problems == []
    assert lines == [
        "- kernels ruling 2: Is a practitioner post a source? (applied meanwhile: measured and paper sources only; "
        "recommended: allow it, labeled)",
        "- learning ruling 1: Which split is held out? (applied meanwhile: by day; recommended: by episode)"]


def test_a_rulings_file_that_cannot_be_read_is_a_problem_and_the_others_are_still_listed(isolated_state, refine_root):
    write_rulings(refine_root, "kernels", ruling(1, "Open question?"))
    (refine_root / "plumbing").mkdir()
    (refine_root / "plumbing" / "OWNER_RULINGS.md").write_text("## 3 Missing the period\n\nNow: x\n", encoding="utf-8")
    (refine_root / "empty-track").mkdir()
    built = build(isolated_state, refine_root)
    found = sections(built.text)
    assert found["## Rulings waiting"] == [
        "- kernels ruling 1: Open question? (applied meanwhile: the default; recommended: do it)"]
    assert len(found["## Problems"]) == 1
    assert found["## Problems"][0].startswith("- plumbing/OWNER_RULINGS.md could not be read: FormatError: ")
    assert built.counts["rulings_waiting"] == 1 and len(built.problems) == 1


def test_more_waiting_rulings_than_the_cap_are_counted_not_listed(refine_root):
    write_rulings(refine_root, "kernels", *(ruling(n, f"Question {n}?") for n in range(1, review.MAX_RULINGS + 4)))
    lines, total, _ = review.rulings_section(refine_root)
    assert total == review.MAX_RULINGS + 3 and len(lines) == review.MAX_RULINGS + 1 and lines[-1] == "- +3 more"


def test_a_folder_of_tracks_that_does_not_exist_has_no_rulings(tmp_path):
    assert review.rulings_section(tmp_path / "nowhere") == ([], 0, [])


def test_the_default_rulings_folder_is_the_refine_folder_of_this_repository():
    assert review.REFINE_ROOT == REPO_ROOT / "lab" / "refine"
    assert (review.REFINE_ROOT / "kernels" / "OWNER_RULINGS.md").is_file()


# --- open pull requests ----------------------------------------------------------------------------------

PR = {"number": 7, "title": "O7: A title", "headRefName": "ctx/x/y", "url": "https://example.test/7",
      "createdAt": "2026-10-10T12:00:00Z", "isDraft": False, "author": {"login": "someone"}}


def prs_of(*items):
    return lambda: list(items)


def pr_lines(state, refine_root, *items):
    return sections(build(state, refine_root, prs=prs_of(*items)).text)["## PRs open"]


def test_pull_requests_are_listed_oldest_first_with_branch_author_and_age(isolated_state, refine_root):
    newer = {**PR, "number": 9, "title": "O9: Newer", "createdAt": "2026-10-10T14:00:00Z", "isDraft": True,
             "author": {"login": "other"}}
    assert pr_lines(isolated_state, refine_root, newer, PR) == [
        "- #7 O7: A title (ctx/x/y), by someone, open 3.0 h, https://example.test/7",
        "- #9 O9: Newer (ctx/x/y, draft), by other, open 1.0 h, https://example.test/7"]


def test_a_pull_request_with_fields_missing_is_still_a_line(isolated_state, refine_root):
    assert pr_lines(isolated_state, refine_root, {"number": 3}) == [
        "- #3 (no title) (no branch), by unknown, no url"]


def test_a_pull_request_dated_after_now_is_open_for_zero_hours_not_a_negative_time(isolated_state, refine_root):
    future = {**PR, "createdAt": "2026-10-10T16:00:00Z"}
    assert pr_lines(isolated_state, refine_root, future) == [
        "- #7 O7: A title (ctx/x/y), by someone, open 0.0 h, https://example.test/7"]


def test_a_long_title_is_clipped(isolated_state, refine_root):
    [line] = pr_lines(isolated_state, refine_root, {**PR, "title": "T" * 300})
    assert line.startswith("- #7 " + "T" * 97 + "... (ctx/x/y)")


def test_more_pull_requests_than_the_cap_are_counted_not_listed(isolated_state, refine_root):
    items = [{**PR, "number": number} for number in range(1, review.MAX_PRS + 3)]
    built = build(isolated_state, refine_root, prs=prs_of(*items))
    lines = sections(built.text)["## PRs open"]
    assert len(lines) == review.MAX_PRS + 1 and lines[-1] == "- +2 more"
    assert built.counts["prs_open"] == review.MAX_PRS + 2


def test_no_open_pull_requests_is_none_and_zero(isolated_state, refine_root):
    built = build(isolated_state, refine_root, prs=prs_of())
    assert sections(built.text)["## PRs open"] == ["- none"]
    assert built.counts["prs_open"] == 0 and "0 PRs open." in built.text


def test_a_source_that_fails_is_said_in_the_file_and_never_fatal(isolated_state, refine_root):
    def failing():
        raise review.PrsUnavailable("gh pr list exited with code 4")

    built = build(isolated_state, refine_root, prs=failing)
    assert sections(built.text)["## PRs open"] == ["- could not be listed: gh pr list exited with code 4"]
    assert built.counts["prs_open"] is None and "PRs open unknown." in built.text


def test_the_recorded_gh_output_is_a_list_of_the_shape_the_review_reads():
    listed = review.file_open_prs(OPEN_PRS)()
    assert [item["number"] for item in listed] == [94, 93, 92, 88]
    assert all({"number", "title", "headRefName", "url", "createdAt", "isDraft", "author"} <= set(item)
               for item in listed)


@pytest.mark.parametrize(("content", "message"), [
    (None, "the pull request file could not be read: FileNotFoundError"),
    ("not json", "the pull request list is not JSON"),
    ('{"number": 1}', "the pull request list is dict, not a list"),
])
def test_a_pull_request_file_that_is_missing_or_wrong_is_unavailable(tmp_path, content, message):
    path = tmp_path / "prs.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(review.PrsUnavailable, match=re.escape(message)):
        review.file_open_prs(path)()


def test_items_of_the_pull_request_file_that_are_not_objects_are_skipped(tmp_path):
    path = tmp_path / "prs.json"
    path.write_text(json.dumps([PR, 5, "x", None]), encoding="utf-8")
    assert review.file_open_prs(path)() == [PR]


class FakeRun:
    """The subprocess boundary: records the call and answers what it was told to."""

    def __init__(self, *, returncode=0, stdout="[]", error=None):
        self.returncode, self.stdout, self.error, self.calls = returncode, stdout, error, []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr="")


def test_gh_is_asked_for_the_open_pull_requests_of_the_team_repository_and_its_list_is_returned():
    run = FakeRun(stdout=OPEN_PRS.read_text(encoding="utf-8"))
    listed = review.gh_open_prs(run=run)
    assert [item["number"] for item in listed] == [94, 93, 92, 88]
    [(command, kwargs)] = run.calls
    assert command == review.GH_COMMAND
    assert command[:3] == ("gh", "pr", "list") and "raohemdutt/Knighthacks26" in command
    assert command[command.index("--state") + 1] == "open"
    assert kwargs["timeout"] == review.PR_TIMEOUT and kwargs["capture_output"] is True and kwargs["check"] is False


@pytest.mark.parametrize(("run", "message"), [
    (FakeRun(returncode=1), "gh pr list exited with code 1"),
    (FakeRun(error=FileNotFoundError()), "gh could not run: FileNotFoundError"),
    (FakeRun(error=subprocess.TimeoutExpired("gh", 30)), "gh could not run: TimeoutExpired"),
    (FakeRun(stdout="login required"), "the pull request list is not JSON"),
    (FakeRun(stdout='"a string"'), "the pull request list is str, not a list"),
])
def test_gh_failing_in_any_way_is_unavailable_and_says_why_without_its_output(run, message):
    with pytest.raises(review.PrsUnavailable) as raised:
        review.gh_open_prs(run=run)
    assert str(raised.value) == message


# --- the stop file and the problems ----------------------------------------------------------------------

def test_a_stopped_machine_says_so_in_the_file_with_its_reason(isolated_state, refine_root):
    (isolated_state / "STOP").write_text("pods\nneed a restart\n", encoding="utf-8")
    built = build(isolated_state, refine_root)
    lines = built.text.splitlines()
    assert lines[4] == "The machine is stopped: pods need a restart."
    assert build(isolated_state, refine_root).text == built.text
    (isolated_state / "STOP").unlink()
    assert "stopped" not in build(isolated_state, refine_root).text


def test_a_stop_file_with_no_text_still_says_stopped(isolated_state, refine_root):
    (isolated_state / "STOP").write_text("", encoding="utf-8")
    assert "The machine is stopped: no reason given." in build(isolated_state, refine_root).text


def test_a_failed_alert_scan_is_a_problem_and_the_review_is_still_written(isolated_state, refine_root):
    record(isolated_state)
    (isolated_state / "alerts").write_text("in the way", encoding="utf-8")
    answer = review.review(DAY, state_dir=isolated_state, refine_root=refine_root, now=NOW)
    text = (isolated_state / "review" / "2026-10-10.md").read_text(encoding="utf-8")
    assert answer["ok"] is True and answer["problems"] == 1
    [problem] = sections(text)["## Problems"]
    assert problem.startswith("- the alert scan failed: ")
    assert "35 ledger rows" in text


def test_a_naive_time_is_refused_before_anything_is_written(isolated_state, refine_root):
    with pytest.raises(ValueError, match="UTC offset"):
        review.review(DAY, state_dir=isolated_state, refine_root=refine_root, now=dt.datetime(2026, 10, 10, 15, 0))
    with pytest.raises(ValueError, match="UTC offset"):
        review.build(DAY, state_dir=isolated_state, refine_root=refine_root, now=dt.datetime(2026, 10, 10, 15, 0))
    assert not (isolated_state / "review").exists()


def test_a_day_that_is_not_a_date_is_refused(isolated_state, refine_root):
    with pytest.raises(ValueError):
        review.build("10/10/2026", state_dir=isolated_state, refine_root=refine_root, now=NOW)
    assert not (isolated_state / "review").exists()


def test_the_default_day_is_the_utc_day_of_now(isolated_state, refine_root):
    eastern = dt.timezone(dt.timedelta(hours=-4))
    answer = review.review(None, state_dir=isolated_state, refine_root=refine_root,
                           now=dt.datetime(2026, 10, 10, 22, 30, tzinfo=eastern))
    assert answer["day"] == "2026-10-11" and (isolated_state / "review" / "2026-10-11.md").is_file()


def test_a_review_file_that_cannot_be_replaced_leaves_no_temporary_file(isolated_state, refine_root):
    folder = isolated_state / "review"
    (folder / "2026-10-10.md").mkdir(parents=True)          # a folder where the file goes: the replace fails
    with pytest.raises(OSError):
        review.review(DAY, state_dir=isolated_state, refine_root=refine_root, now=NOW, scan=False)
    assert [item.name for item in folder.iterdir()] == ["2026-10-10.md"]


# --- the verb ------------------------------------------------------------------------------------------

def test_the_verb_is_registered_and_listed_in_the_help(capsys):
    assert lab_cli.VERBS["review"] == "lab.machine.review:main"
    code, out, _ = run_cli(["--help"], capsys)
    assert code == 0
    assert re.search(r"^  review +Write the owner's morning file: what happened, numbers moved, rulings, "
                     r"complaints and open PRs\.$", out, re.MULTILINE)


def test_review_answers_one_json_line_and_writes_the_file(isolated_state, capsys):
    record(isolated_state)
    code, out, _ = run_cli(["review", "--day", DAY, "--now", "2026-10-10T15:00:00Z", "--prs-file", str(OPEN_PRS)],
                           capsys)
    answer = one_json_line(out)
    assert code == 0
    assert {key: value for key, value in answer["counts"].items() if key != "rulings_waiting"} == {
        "rows": 35, "sittings": 3, "sittings_never_ended": 1, "alerts": 6, "metrics_moved": 8, "metrics_new": 1,
        "complaints": 1, "prs_open": 4, "cost_usd": 3.52, "cost_unknown": 1}
    assert (answer["ok"], answer["outcome"], answer["day"], answer["file"]) == (True, "written", DAY,
                                                                              "review/2026-10-10.md")
    assert str(isolated_state) not in out
    text = (isolated_state / "review" / "2026-10-10.md").read_text(encoding="utf-8")
    assert text.startswith("# Lab review, 2026-10-10 (UTC)")
    assert sections(text)["## Alerts"] == ALERT_LINES


def test_the_flags_leave_out_the_scan_and_the_pull_requests(isolated_state, capsys):
    record(isolated_state)
    code, out, _ = run_cli(["review", "--day", DAY, "--now", "2026-10-10T15:00:00Z", "--no-prs", "--no-alerts"],
                           capsys)
    answer = one_json_line(out)
    text = (isolated_state / "review" / "2026-10-10.md").read_text(encoding="utf-8")
    assert code == 0 and answer["counts"]["alerts"] == 0 and answer["counts"]["prs_open"] is None
    assert not (isolated_state / "alerts").exists()
    assert sections(text)["## PRs open"] == ["- not asked for"]


def test_a_pull_request_file_that_is_missing_is_said_in_the_file_not_an_error(isolated_state, capsys, tmp_path):
    code, out, _ = run_cli(["review", "--day", DAY, "--now", "2026-10-10T15:00:00Z", "--no-alerts",
                            "--prs-file", str(tmp_path / "nope.json")], capsys)
    text = (isolated_state / "review" / "2026-10-10.md").read_text(encoding="utf-8")
    assert code == 0 and one_json_line(out)["ok"] is True
    assert sections(text)["## PRs open"] == ["- could not be listed: the pull request file could not be read: "
                                             "FileNotFoundError"]


@pytest.mark.parametrize("argv", [["--day", "yesterday"], ["--day", "2026-13-01"], ["--now", "noon"],
                                  ["--now", "2026-10-10T15:00:00"], ["--no-prs", "--prs-file", "x.json"],
                                  ["--bogus"]])
def test_bad_arguments_exit_2_and_write_nothing(isolated_state, capsys, argv):
    code, out, err = run_cli(["review", *argv], capsys)
    assert code == 2
    assert out == ""
    assert "usage: python -m lab review" in err
    assert not (isolated_state / "review").exists()


def test_review_help_is_answered_without_writing_anything(isolated_state, capsys):
    code, out, _ = run_cli(["review", "-h"], capsys)
    assert code == 0
    assert "usage: python -m lab review" in out and "--day" in out and "--prs-file" in out
    assert not (isolated_state / "review").exists()


def test_the_verb_runs_while_the_stop_file_exists(isolated_state, capsys):
    record(isolated_state)
    (isolated_state / "STOP").write_text("maintenance\n", encoding="utf-8")
    code, out, _ = run_cli(["review", "--day", DAY, "--now", "2026-10-10T15:00:00Z", "--no-prs"], capsys)
    assert (code, one_json_line(out)["outcome"]) == (0, "written")
    text = (isolated_state / "review" / "2026-10-10.md").read_text(encoding="utf-8")
    assert "The machine is stopped: maintenance." in text


def test_a_file_where_the_review_folder_should_be_is_an_error_answer_without_a_path(isolated_state, capsys):
    (isolated_state / "review").write_text("in the way", encoding="utf-8")
    code, out, _ = run_cli(["review", "--day", DAY, "--now", "2026-10-10T15:00:00Z", "--no-prs", "--no-alerts"],
                           capsys)
    answer = one_json_line(out)
    assert code == 1 and answer["ok"] is False and answer["outcome"] == "error"
    assert answer["error"].split(":")[0] in ("FileExistsError", "NotADirectoryError")
    assert str(isolated_state) not in out


def test_python_dash_m_lab_review_works_as_a_real_process(isolated_state):
    record(isolated_state)
    env = {**os.environ, "LAB_STATE_DIR": str(isolated_state)}
    done = subprocess.run([sys.executable, "-m", "lab", "review", "--day", DAY, "--now", "2026-10-10T15:00:00Z",
                           "--prs-file", str(OPEN_PRS)], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["counts"]["rows"] == 35
    assert (isolated_state / "review" / "2026-10-10.md").is_file()
