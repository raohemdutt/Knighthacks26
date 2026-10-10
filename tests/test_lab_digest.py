"""The lab digest (lab/machine/digest.py) on a recorded ledger, board snapshot and rulings
files under fixtures/lab/digest/: what ran in the window, which metrics moved, the GPU queue,
the unanswered rulings, Ilan's lane by status and tier, the next unblocked tasks per tier, the
150-line bound, the never-a-note rule, reading and storing the board snapshot, and the verb
through the dispatcher and as a process. Every test works in a temporary state dir; nothing
here touches lab/state."""
from __future__ import annotations

import builtins
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import lab.__main__ as lab_cli
from lab.machine import digest, handoff

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "fixtures" / "lab" / "digest"
LEDGER = FIXTURES / "ledger.jsonl"
BOARD = FIXTURES / "board.json"
REFINE = FIXTURES / "refine"
EXPECTED = FIXTURES / "expected.txt"
UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 10, 16, 0, tzinfo=UTC)
NOW_TEXT = "2026-10-10T16:00:00Z"
ILAN = "user://fbcc0baf-9942-4661-9414-319688df6910"
DOMINICK = "user://13dd872b-594c-81cd-b409-0002df1ecab7"


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    """The default state dir of every test is a temporary one."""
    folder = tmp_path / "state"
    folder.mkdir()
    monkeypatch.setenv("LAB_STATE_DIR", str(folder))
    return folder


@pytest.fixture
def recorded(state):
    """The state dir holding the recorded ledger."""
    assert LEDGER.is_file(), f"expected the recorded ledger at {LEDGER}"
    shutil.copyfile(LEDGER, state / "ledger.jsonl")
    return state


def build(**changes):
    changes.setdefault("now", NOW)
    changes.setdefault("board", BOARD)
    changes.setdefault("refine_root", REFINE)
    return digest.compute(**changes)


def shown(result: dict) -> str:
    """The text, with the temporary ledger path replaced so it compares across machines."""
    return digest.render(result).replace(result["ledger"]["path"], "<state dir>/ledger.jsonl")


def write_board(path: Path, tasks: list[dict], queue: list[dict] | None = None,
                taken_at: str = "2026-10-10T15:50:00Z") -> Path:
    path.write_text(json.dumps({"schema": 1, "taken_at": taken_at, "taken_by": "fake",
                                "tasks": tasks, "gpu_queue": queue or []}), encoding="utf-8")
    return path


def fake_url(task_id: str) -> str:
    """A Notion page URL: 32 hex characters, as Notion's are."""
    return "https://app.notion.com/p/" + hashlib.md5(task_id.encode()).hexdigest()


def fake_task(task_id: str, **fields) -> dict:
    """A Task Board row in the plain lowercase form of the schema."""
    row = {"id": task_id, "title": f"{task_id}: a task", "status": "Not started", "tier": "Opus",
           "priority": "P1 Strengthens demo", "wave": 1, "locked": False, "stretch": False,
           "assignees": [ILAN], "blocked_by": [], "url": fake_url(task_id)}
    row.update(fields)
    return row


def ledger_line(seq: int, ts: str, **fields) -> str:
    row = {"seq": seq, "ts": ts, "actor": "opus-1", "tier": "opus", "task": None, "verb": "bench",
           "outcome": "ok", "metrics": {}, "refs": {"pr": None, "commit": None, "slot": None, "run_id": None}}
    row.update(fields)
    return json.dumps(row)


# --- the whole digest ----------------------------------------------------------------------------

def test_the_recorded_digest_is_the_expected_text_and_under_150_lines(recorded):
    result = build()
    text = shown(result)
    assert EXPECTED.is_file(), f"expected the recorded digest text at {EXPECTED}"
    assert text == EXPECTED.read_text(encoding="utf-8")
    assert len(text.splitlines()) < 150
    assert text.isascii()


def test_the_json_form_holds_what_the_text_shows(recorded):
    result = build()
    assert json.loads(json.dumps(result)) == result
    assert result["ok"] is True and result["outcome"] == "ok"
    assert result["now"] == NOW_TEXT
    assert result["window"] == {"since": "2026-10-10T14:00:00Z", "until": NOW_TEXT, "hours": 2.0}
    assert result["ledger"]["rows"] == 22
    assert (result["ledger"]["first_seq"], result["ledger"]["last_seq"]) == (1, 22)
    assert result["board"] == {"path": "fixtures/lab/digest/board.json", "taken_at": "2026-10-10T15:45:00Z",
                               "taken_by": "sonnet-3", "age_min": 15, "stale": False, "tasks": 23,
                               "queue_rows": 5}


# --- what ran ----------------------------------------------------------------------------------------

def test_what_ran_counts_the_window_rows_and_lists_failures_newest_first(recorded):
    ran = build()["ran"]
    assert (ran["rows"], ran["actors"], ran["tasks"]) == (12, 5, 5)
    assert ran["outcomes"] == {"ok": 9, "error": 1, "fail": 1, "handoff": 1}
    assert ran["reads"] == {"digest": 2}
    assert [entry["task"] for entry in ran["by_task"]["items"]] == ["L3", "O20", "R11", "L13", "O17"]
    assert ran["by_task"]["more"] == 0
    r11 = ran["by_task"]["items"][2]
    assert r11 == {"task": "R11", "rows": 5, "failures": 0, "verbs": ["bench", "slot done", "slot start"],
                   "pr": None, "last_ts": "2026-10-10T15:00:00Z", "last_verb": "bench", "last_outcome": "ok",
                   "last_actor": "opus-2"}
    assert ran["by_task"]["items"][3]["failures"] == 2
    assert ran["by_task"]["items"][4]["pr"] == "88"
    assert ran["by_verb"] == {"items": [{"verb": "pod status", "rows": 1, "outcomes": {"ok": 1}},
                                        {"verb": "pool tick", "rows": 1, "outcomes": {"ok": 1}}], "more": 0}
    assert ran["failures"] == {"items": [
        {"ts": "2026-10-10T14:45:00Z", "seq": 14, "task": "L13", "verb": "label", "outcome": "error",
         "actor": "sonnet-4", "run_id": "label-0003"},
        {"ts": "2026-10-10T14:30:00Z", "seq": 12, "task": "L13", "verb": "label", "outcome": "fail",
         "actor": "sonnet-4", "run_id": "label-0002"},
    ], "more": 0}


def test_rows_before_the_window_or_after_now_are_left_out(recorded):
    tasks = [entry["task"] for entry in build()["ran"]["by_task"]["items"]]
    assert "O15" not in tasks          # 12:05, before the window
    assert "R12" not in tasks          # 16:30, after now
    wide = build(hours=6)["ran"]
    assert [entry["task"] for entry in wide["by_task"]["items"]] == ["L3", "O20", "R11", "L13", "O17", "O15", "O19"]
    assert wide["rows"] == 19
    since = build(since=dt.datetime(2026, 10, 10, 15, 0, tzinfo=UTC))["ran"]
    assert [entry["task"] for entry in since["by_task"]["items"]] == ["L3", "O20", "R11"]


# --- metrics ------------------------------------------------------------------------------------------

def test_metrics_compare_each_series_with_its_newest_earlier_reading(recorded):
    metrics = build()["metrics"]
    moved = metrics["moved"]["items"]
    assert [(e["metric"], e["node"]) for e in moved] == [
        ("teacher.decode_tok_s", "pod-a"), ("teacher.joules_per_token", "pod-a"), ("pod.balance_credits", "pod-b")]
    decode, joules, balance = moved
    assert (decode["before"], decode["value"], decode["before_seq"], decode["seq"]) == (7.7, 9.1, 1, 9)
    assert decode["delta"] == pytest.approx(1.4)
    assert decode["pct"] == pytest.approx(18.1818, abs=1e-3)
    assert decode["unit"] == "tok/s" and decode["low_confidence"] is False and decode["within_window"] is False
    assert decode["label"] == "output_tokens=160 prompt=text thinking=on"
    assert (joules["before"], joules["value"], joules["low_confidence"]) == (25.97, 23.08, True)
    assert joules["pct"] == pytest.approx(-11.128, abs=1e-3)
    assert (balance["before"], balance["value"], balance["delta"]) == (59960, 59900, -60)
    assert metrics["moved"]["more"] == 0


def test_a_reading_is_never_compared_with_one_taken_under_other_conditions(recorded):
    metrics = build()["metrics"]
    new = {(e["metric"], e["node"]): e for e in metrics["new"]["items"]}
    # watts_avg was 200 W while generating (seq 3); the bench row's 210 W carries the bench's
    # conditions, so it starts a series instead of "moving" by 10 W.
    assert new[("teacher.watts_avg", "pod-a")]["value"] == 210
    assert "before" not in new[("teacher.watts_avg", "pod-a")]
    assert new[("teacher.ttft_s", "pod-a")]["value"] == 0.5
    assert [(e["metric"], e["value"]) for e in metrics["unchanged"]["items"]] == [("trainer.samples_per_s", 0.51)]
    # the image-caption decode series (6.6, seq 2) was not read in the window, and the row
    # stamped after now (99.0, seq 22) is never used
    every = metrics["moved"]["items"] + metrics["new"]["items"] + metrics["unchanged"]["items"]
    assert 6.6 not in [e["value"] for e in every] and 99.0 not in [e["value"] for e in every]


def test_a_series_read_twice_inside_the_window_moves_against_its_first_reading(state):
    conditions = {"model": "m", "batch": 1}
    lines = [ledger_line(1, "2026-10-10T14:10:00Z", metrics={"trainer.samples_per_s": 0.5}, node="pod-b",
                         conditions=conditions),
             ledger_line(2, "2026-10-10T15:10:00Z", metrics={"trainer.samples_per_s": 0.6}, node="pod-b",
                         conditions=conditions)]
    (state / "ledger.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    moved = build()["metrics"]["moved"]["items"]
    assert len(moved) == 1
    assert (moved[0]["before"], moved[0]["value"], moved[0]["within_window"]) == (0.5, 0.6, True)
    assert moved[0]["pct"] == pytest.approx(20.0)


# --- the GPU queue ------------------------------------------------------------------------------------

def test_gpu_queue_shows_overruns_waits_shortest_first_and_complaints(recorded):
    gpu = build()["gpu"]
    assert gpu["summary"] == [{"gpu": "A teacher shared service", "running": 0, "queued": 1},
                              {"gpu": "B exclusive", "running": 1, "queued": 2}]
    assert gpu["running"]["items"] == [{
        "request": "R11: decode sweep batch 1 to 16", "task": "R11", "gpu": "B exclusive", "minutes": 15,
        "started": "2026-10-10T15:40:00Z", "elapsed_min": 20, "cap_min": 15, "extended": False, "over_min": 5}]
    queued = [(e["gpu"], e["task"], e["position"], e["waited_min"], e["complaint_due"]) for e in gpu["queued"]["items"]]
    assert queued == [("A teacher shared service", "T1", 1, 10, False),
                      ("B exclusive", "R12", 1, 40, False),
                      ("B exclusive", "L3", 2, 60, True)]       # R11 on the same card is past its cap
    # the finished row's complaint is history and is not listed
    assert gpu["complaints"]["items"] == [{"request": "R12: HIP kernel microbench", "status": "Queued",
                                           "minutes_since": 40,
                                           "text": "Waited 40 min; the running slot is past its cap"}]


def test_a_queued_row_may_complain_while_a_running_row_on_its_card_is_past_its_cap(recorded):
    # The queue verbs' rule (lab/machine/slots.py, complaint_due): L3 has waited 60 of its 60 minutes, not yet
    # twice them, but R11 has been running 5 minutes past its 15-minute cap on the same card.
    queued = build()["gpu"]["queued"]["items"]
    l3 = next(entry for entry in queued if entry["task"] == "L3")
    assert (l3["gpu"], l3["waited_min"], l3["has_complaint"], l3["complaint_due"]) == ("B exclusive", 60, False, True)


def test_an_overrun_makes_a_complaint_due_on_its_own_card_only_and_not_twice(state, tmp_path):
    queue = [
        {"request": "R11: sweep", "task": "R11", "gpu": "B exclusive", "minutes": 15, "status": "Running",
         "requested": "2026-10-10T15:00:00Z", "started": "2026-10-10T15:30:00Z"},
        {"request": "L3: smoke", "task": "L3", "gpu": "B exclusive", "minutes": 60, "status": "Queued",
         "requested": "2026-10-10T15:50:00Z"},
        {"request": "R12: bench", "task": "R12", "gpu": "B exclusive", "minutes": 15, "status": "Queued",
         "requested": "2026-10-10T15:52:00Z", "complaint": "already said: R11 is past its cap"},
        {"request": "T1: probe", "task": "T1", "gpu": "A teacher shared service", "minutes": 15, "status": "Queued",
         "requested": "2026-10-10T15:55:00Z"},
    ]
    board = write_board(tmp_path / "board.json", [fake_task("O1")], queue)
    gpu = build(board=board)["gpu"]
    assert [(e["task"], e["waited_min"], e["complaint_due"]) for e in gpu["queued"]["items"]] == [
        ("T1", 5, False), ("R12", 8, False), ("L3", 10, True)]
    assert [(e["task"], e["cap_min"], e["elapsed_min"], e["over_min"]) for e in gpu["running"]["items"]] == [
        ("R11", 15, 30, 15)]


def test_a_fake_board_in_the_plain_schema_marks_a_complaint_due(state, tmp_path):
    queue = [
        {"request": "R12: microbench", "task": "R12", "gpu": "B exclusive", "minutes": 15, "status": "Queued",
         "requested": {"start": "2026-10-10T15:20:00+00:00"}, "complaint": ""},
        {"request": "L3: smoke", "task": "L3", "gpu": "B exclusive", "minutes": 60, "status": "Queued",
         "requested": "2026-10-10T11:45:00-04:00"},
        {"request": "R11: sweep", "task": "R11", "gpu": "B exclusive", "minutes": 15, "status": "Running",
         "started": "2026-10-10T15:30:00", "extended": True},
    ]
    board = write_board(tmp_path / "board.json", [fake_task("O1")], queue)
    result = build(board=board)
    queued = result["gpu"]["queued"]["items"]
    assert [(e["task"], e["waited_min"], e["complaint_due"]) for e in queued] == [("R12", 40, True), ("L3", 15, False)]
    running = result["gpu"]["running"]["items"][0]
    assert (running["cap_min"], running["elapsed_min"], running["over_min"], running["extended"]) == (30, 30, 0, True)
    assert "1 queue time without a UTC offset read as UTC" in result["problems"]["items"]
    assert "complaint due" in digest.render(result)


# --- rulings ------------------------------------------------------------------------------------------

def test_unanswered_rulings_are_listed_with_their_default(recorded):
    rulings = build()["rulings"]
    assert [(e["track"], e["number"], e["origin"]) for e in rulings["items"]] == [("kernels", 4, "pass 3"),
                                                                                  ("telemetry", 1, "pass 1")]
    assert rulings["items"][0]["now"] == "the spread, as kernel test 3 says; E3 uses five runs per point."
    assert rulings["answered"] == 1 and rulings["more"] == 0


def test_a_broken_rulings_file_is_named_and_the_others_still_listed(recorded, tmp_path):
    refine = tmp_path / "refine"
    shutil.copytree(REFINE, refine)
    (refine / "plumbing").mkdir()
    (refine / "plumbing" / "OWNER_RULINGS.md").write_text(
        "# Owner rulings for the plumbing track\n\n## 7 a heading without its dot\n\nNow: x\nAnswer: ____\n",
        encoding="utf-8")
    result = build(refine_root=refine)
    assert [(e["track"], e["number"]) for e in result["rulings"]["items"]] == [("kernels", 4), ("telemetry", 1)]
    assert [p for p in result["problems"]["items"] if p.startswith("plumbing rulings unreadable:")]


# --- Ilan's lane and the next tasks ---------------------------------------------------------------

def test_lane_counts_ilans_tasks_by_status_and_tier(recorded):
    table = build()["lane"]
    assert (table["tasks"], table["locked"]) == (22, 6)
    assert table["tiers"] == ["Fable", "Opus", "Sonnet", "none"]
    assert table["counts"] == {
        "Not started": {"Fable": 1, "Opus": 6, "Sonnet": 3, "none": 1},
        "In progress": {"Fable": 0, "Opus": 3, "Sonnet": 2, "none": 0},
        "Blocked": {"Fable": 0, "Opus": 1, "Sonnet": 0, "none": 0},
        "Done": {"Fable": 0, "Opus": 2, "Sonnet": 3, "none": 0},
    }


def test_next_unblocked_per_tier_puts_p0_first_then_wave_and_stretch_last(recorded):
    result = build()
    nxt = {tier: [e["id"] for e in holder["items"]] for tier, holder in result["next"].items()}
    assert nxt == {"Fable": [], "Opus": ["O27", "O18", "O33", "G9"], "Sonnet": ["O30"], "none": ["X1"]}
    assert result["next"]["Opus"]["items"][0] == {"id": "O27", "priority": "P0", "wave": 2, "stretch": False,
                                                  "title": "Experiment schema and queue sync"}
    assert result["held"]["shared"] == {"items": ["R5"], "more": 0}
    assert result["held"]["missing_blocker"] == {"items": [{"id": "O41", "blockers": ["page 000a0012"]}], "more": 0}


def test_a_task_waiting_on_an_unfinished_blocker_is_not_offered(state, tmp_path):
    tasks = [
        fake_task("O15", status="Done"),
        fake_task("O20", status="In progress", locked=True),
        fake_task("O22", blocked_by=["O20"]),                 # by task ID, still running
        fake_task("O27", blocked_by=["O15"]),                 # by task ID, Done
        fake_task("O33", blocked_by=[fake_url("O15")]),        # by page URL, Done
        fake_task("O40", locked=True),                        # unlocked tasks only
        fake_task("O41", assignees=[ILAN, DOMINICK]),         # shared with a teammate
        fake_task("K9", assignees=[DOMINICK]),                # not Ilan's lane at all
        fake_task("O50", tier="sonnet"),                      # the tier is matched without case
    ]
    result = build(board=write_board(tmp_path / "board.json", tasks))
    assert [e["id"] for e in result["next"]["Opus"]["items"]] == ["O27", "O33"]
    assert [e["id"] for e in result["next"]["Sonnet"]["items"]] == ["O50"]
    assert result["held"]["shared"]["items"] == ["O41"]
    assert result["lane"]["tasks"] == 8


# --- never a note -------------------------------------------------------------------------------------

def test_the_digest_never_opens_a_handoff_note_or_copies_a_rows_free_text(recorded, monkeypatch):
    # the note is written by the real handoff verb: the notes file and its ledger row
    answer = handoff.handoff("sit-0001", "NOTE-FILE-SENTINEL: skip the tests next time", state_dir=recorded,
                             actor="opus-8", tier="opus", now=dt.datetime(2026, 10, 10, 15, 55, tzinfo=UTC))
    assert answer["ok"] is True and answer["note_written"] is True
    notes = recorded / "notes"
    assert "NOTE-FILE-SENTINEL" in (notes / "2026-10-10.md").read_text(encoding="utf-8")
    opened: list[str] = []
    listed: list[str] = []
    real_open, real_io_open, real_scandir = builtins.open, io.open, os.scandir

    def spy_open(file, *args, **kwargs):
        opened.append(os.path.abspath(os.fspath(file)) if not isinstance(file, int) else f"fd {file}")
        return real_open(file, *args, **kwargs)

    def spy_scandir(path="."):
        listed.append(os.path.abspath(os.fspath(path)))
        return real_scandir(path)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)
    monkeypatch.setattr(os, "scandir", spy_scandir)
    result = build()
    text = digest.render(result)
    monkeypatch.undo()
    notes_dir = os.path.abspath(notes)
    assert opened, "the spy saw no file opened at all"
    assert os.path.abspath(recorded / "ledger.jsonl") in opened
    assert [p for p in opened + listed if p.startswith(notes_dir)] == []
    # the handoff itself is counted from its ledger row; its text is not
    assert {"verb": "handoff", "rows": 1, "outcomes": {"handoff": 1}} in result["ran"]["by_verb"]["items"]
    as_json = json.dumps(result)
    for secret in ("NOTE-FILE-SENTINEL", "HANDOFF-NOTE-SENTINEL", "lab bench run bench-0001", "hub API balance"):
        assert secret not in text and secret not in as_json


# --- the size bound -----------------------------------------------------------------------------------

def test_the_text_stays_under_150_lines_however_much_happened(state, tmp_path):
    lines = []
    seq = 0
    for n in range(400):
        seq += 1
        ts = (dt.datetime(2026, 10, 10, 14, 0, 5, tzinfo=UTC) + dt.timedelta(seconds=17 * n)).isoformat()
        lines.append(ledger_line(seq, ts, task=f"T{n % 120}", verb=f"verb{n % 30}",
                                 outcome="fail" if n % 7 == 0 else "ok", actor=f"opus-{n % 9}"))
    for n in range(60):
        for ts, value in (("2026-10-10T10:00:00Z", 1.0), ("2026-10-10T15:00:00Z", 2.0 + n)):
            seq += 1
            lines.append(ledger_line(seq, ts, metrics={f"m.series_{n}": value}, node="pod-a"))
    for n in range(40):
        seq += 1
        lines.append(ledger_line(seq, "2026-10-10T15:30:00Z", verb=f"untasked{n}"))
    (state / "ledger.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tiers = ("Fable", "Opus", "Sonnet")
    tasks = [fake_task(f"Q{n}", tier=tiers[n % 3]) for n in range(200)]
    queue = [{"request": f"job {n}", "task": f"Q{n}", "gpu": "B exclusive", "minutes": 15, "status": "Queued",
              "requested": "2026-10-10T14:00:00Z", "complaint": f"waited long {n}"} for n in range(40)]
    refine = tmp_path / "refine"
    (refine / "kernels").mkdir(parents=True)
    rulings = "".join(f"## {n}. Question {n}? (pass 1)\n\nNow: d.\nRecommend: r.\nSource of tension: s.\nAnswer: ____\n\n"
                      for n in range(1, 31))
    (refine / "kernels" / "OWNER_RULINGS.md").write_text("# Rulings\n\n" + rulings, encoding="utf-8")
    result = build(board=write_board(tmp_path / "board.json", tasks, queue), refine_root=refine)
    text = digest.render(result)
    assert len(text.splitlines()) <= digest.MAX_LINES
    # every list still accounts for what it left out
    failures = sum(1 for n in range(400) if n % 7 == 0)
    assert len(result["ran"]["failures"]["items"]) + result["ran"]["failures"]["more"] == failures
    assert len(result["ran"]["by_task"]["items"]) + result["ran"]["by_task"]["more"] == 120
    assert len(result["metrics"]["moved"]["items"]) + result["metrics"]["moved"]["more"] == 60
    assert len(result["gpu"]["queued"]["items"]) + result["gpu"]["queued"]["more"] == 40
    assert len(result["rulings"]["items"]) + result["rulings"]["more"] == 30
    assert sum(len(h["items"]) + h["more"] for h in result["next"].values()) == 200
    assert "more)" in text


def test_fit_trims_the_longest_list_first_and_counts_what_it_dropped(recorded):
    result = build()
    full = len(digest.render(result).splitlines())
    before = {"by_task": len(result["ran"]["by_task"]["items"]), "opus": len(result["next"]["Opus"]["items"])}
    digest.fit(result, full - 1)
    assert len(digest.render(result).splitlines()) <= full - 1
    # by_task (5 entries) is the longest list; its first cut adds a "(+N more)" line, so it
    # takes two cuts to save one line
    assert before["by_task"] == 5
    assert len(result["ran"]["by_task"]["items"]) == 3
    assert result["ran"]["by_task"]["more"] == 2
    assert len(result["next"]["Opus"]["items"]) == before["opus"]


# --- the board snapshot ---------------------------------------------------------------------------------

def test_without_a_snapshot_the_digest_still_answers_and_says_why(recorded, tmp_path):
    result = build(board=tmp_path / "missing.json")
    assert result["board"] is None and "gpu" not in result and "next" not in result
    assert result["ran"]["rows"] == 12
    assert result["problems"]["items"][0].startswith(
        "no board snapshot (python -m lab digest --schema says how to write one): ")
    assert "Board snapshot: none readable" in digest.render(result)


def test_a_snapshot_that_is_not_json_or_has_no_time_is_named(recorded, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    result = build(board=broken)
    assert result["board"] is None
    assert result["problems"]["items"][0].startswith("board snapshot not JSON (")
    untimed = tmp_path / "untimed.json"
    untimed.write_text(json.dumps({"schema": 1, "tasks": []}), encoding="utf-8")
    result = build(board=untimed)
    assert result["board"] is None
    assert result["problems"]["items"] == ["taken_at is missing or not ISO 8601 (when were the Notion reads made?)"]


def test_an_old_snapshot_is_marked_stale(recorded, tmp_path):
    old = write_board(tmp_path / "old.json", [fake_task("O1")], taken_at="2026-10-10T13:00:00Z")
    result = build(board=old)
    assert (result["board"]["age_min"], result["board"]["stale"]) == (180, True)
    assert "STALE: fetch the board fresh before acting on it." in digest.render(result)


def test_the_schema_example_is_a_valid_snapshot():
    example = digest.SCHEMA.split("Example:\n", 1)[1]
    board = digest.parse_board(json.loads(example), "example")
    assert [t.id for t in board.tasks] == ["O17"]
    assert board.tasks[0].blocked_by == ["https://app.notion.com/p/3f596ef5fec581499019c49da7a9d322"]
    assert [(q.status, q.minutes) for q in board.queue] == [("Running", 15.0)]
    assert board.problems == []


def test_save_board_stores_the_checked_snapshot_and_later_digests_read_it(recorded, capsys):
    common = ["--now", NOW_TEXT, "--refine-root", str(REFINE)]
    code = digest.main(["--save-board", str(BOARD), *common])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("LAB DIGEST 2026-10-10T16:00:00Z")
    assert (recorded / "board.json").read_bytes() == BOARD.read_bytes()
    stored = digest.main(["--json", *common])
    assert stored["board"]["taken_by"] == "sonnet-3"
    assert stored["board"]["path"].endswith("board.json")


def test_save_board_refuses_a_snapshot_it_cannot_use_and_stores_nothing(recorded, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": 1, "tasks": "not a list", "taken_at": "2026-10-10T15:00:00Z"}),
                   encoding="utf-8")
    answer = digest.main(["--save-board", str(bad), "--now", NOW_TEXT])
    assert answer["ok"] is False and answer["outcome"] == "invalid_board"
    assert answer["problems"] == ["tasks must be a list of Task Board rows"]
    assert not (recorded / "board.json").exists()


# --- the verb -------------------------------------------------------------------------------------------

def test_the_verb_answers_through_the_dispatcher_in_text_and_in_json(recorded, capsys):
    common = ["--now", NOW_TEXT, "--board", str(BOARD), "--refine-root", str(REFINE)]
    assert lab_cli.main(["digest", *common]) == 0
    text = capsys.readouterr().out
    assert text == digest.render(build())
    assert lab_cli.main(["digest", "--json", *common]) == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 1
    assert json.loads(out) == json.loads(json.dumps(build()))


def test_the_verb_is_registered_and_listed_in_help(capsys):
    assert lab_cli.VERBS["digest"] == "lab.machine.digest:main"
    assert lab_cli.main(["--help"]) == 0
    assert re.search(r"^  digest +What a sitting starts from", capsys.readouterr().out, re.MULTILINE)


def test_the_verb_prints_the_schema(capsys):
    assert lab_cli.main(["digest", "--schema"]) == 0
    assert capsys.readouterr().out == digest.SCHEMA


@pytest.mark.parametrize("argv", [["--hours", "-1"], ["--hours", "two"], ["--since", "2026-10-10T14:00:00"],
                                  ["--hours", "2", "--since", "2026-10-10T14:00:00Z"],
                                  ["--board", "a.json", "--save-board", "b.json"]])
def test_bad_arguments_exit_2(argv, capsys):
    assert lab_cli.main(["digest", *argv]) == 2


def test_a_window_that_starts_after_now_is_refused(recorded):
    answer = digest.main(["--since", "2026-10-10T17:00:00Z", "--now", NOW_TEXT, "--board", str(BOARD)])
    assert answer == {"ok": False, "outcome": "bad_window",
                      "error": "the window starts at 2026-10-10T17:00:00Z, after now (2026-10-10T16:00:00Z)"}


def test_the_verb_is_refused_while_stopped(recorded, capsys):
    (recorded / "STOP").write_text("maintenance\n", encoding="utf-8")
    assert lab_cli.main(["digest", "--now", NOW_TEXT]) == 1
    answer = json.loads(capsys.readouterr().out.splitlines()[0])
    assert (answer["ok"], answer["outcome"], answer["reason"]) == (False, "stopped", "maintenance")


def test_an_empty_ledger_and_no_snapshot_still_give_a_digest(state, tmp_path):
    empty_refine = tmp_path / "refine"
    empty_refine.mkdir()
    text = digest.render(build(board=tmp_path / "none.json", refine_root=empty_refine))
    assert "Ledger: 0 rows, " in text
    assert "Nothing in the window." in text
    assert "No metric was read in the window." in text
    assert "None unanswered (0 answered)." in text


def test_the_text_is_ascii_whatever_the_board_says(recorded, tmp_path):
    board = write_board(tmp_path / "board.json",
                        [fake_task("O1", title="O1: Ilan ↔ Dominick — café “plan”")])
    text = digest.render(build(board=board))
    assert text.isascii()
    assert "O1 P1 w1 Ilan <-> Dominick - caf\\xe9 \"plan\"" in text


def test_python_m_lab_digest_runs_as_a_process(recorded):
    env = dict(os.environ, LAB_STATE_DIR=str(recorded))
    done = subprocess.run([sys.executable, "-m", "lab", "digest", "--now", NOW_TEXT, "--board", str(BOARD),
                           "--refine-root", str(REFINE)],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("LAB DIGEST 2026-10-10T16:00:00Z")
    assert len(done.stdout.splitlines()) < 150
    assert "## Next unblocked" in done.stdout
