"""Tests for lab/machine/refine_track.py, the checks on what refinement passes leave in a track folder.

Every committed track folder is checked as it stands, so a pass that appends an invalid entry, skips its report
or its counter, or cites a metrics-file line for a value it does not hold fails the suite before it can merge.
The synthetic tracks are built from the recorded telemetry dry run, the one pass entry in the repo, so the checks
run on the same text a pass writes, and each broken variant changes exactly one thing.
"""

import re
import shutil
from pathlib import Path

import pytest

from lab.machine import refine_formats as rf
from lab.machine import refine_track as rt

REFINE = Path(__file__).resolve().parents[1] / "lab" / "refine"
METRICS = REFINE / "metrics.jsonl"
RECORDED = REFINE / "telemetry" / "dry_runs" / "2026-10-10-pass-01.md"

LOG_HEADER = "# Refinement log for the telemetry track\n\nOne entry per pass, in the formats of lab/refine/TEMPLATES.md.\n\n---\n"


def recorded_text():
    return RECORDED.read_text(encoding="utf-8").replace("\r\n", "\n")


def entry_text():
    """The pass 1 entry of the recorded dry run, exactly as a live pass would append it to the log."""
    found = re.search(r"^```text dry-run:pass-entry\n(.*?)^```", recorded_text(), flags=re.MULTILINE | re.DOTALL)
    assert found, f"{RECORDED} has no dry-run entry block"
    return found.group(1)


def report_text(entry, track="telemetry", number=1):
    return (
        f"Verbatim report returned by {track} refinement pass {number} (Opus sitting), 2026-10-10. "
        f"The log entry it appended is in lab/refine/{track}/REFINE_LOG.md.\n"
        "Ended: 2026-10-10T11:00:00-04:00\n\n---\n\nOne paragraph.\n\n"
        f"**(a) Log entry appended, verbatim**\n\n```text\n{entry}```\n"
    )


def make_track(tmp_path, entries=(), counter=None, reports=True, track="telemetry"):
    """A track folder as a pass leaves it: header-only log plus entries, counter, reports."""
    folder = tmp_path / track
    (folder / "pass_reports").mkdir(parents=True)
    (folder / "pass_reports" / ".gitkeep").write_text("", encoding="utf-8")
    (folder / "REFINE_LOG.md").write_text(LOG_HEADER + "".join(f"\n{entry}" for entry in entries), encoding="utf-8")
    (folder / "refine_pass.txt").write_text(f"{len(entries) if counter is None else counter}\n", encoding="utf-8")
    if reports:
        for number, entry in enumerate(entries, 1):
            (folder / "pass_reports" / f"pass_{number:02d}.md").write_text(report_text(entry, track, number), encoding="utf-8")
    return folder


def edit(text, old, new):
    assert old in text, f"{old!r} is not in the text being edited"
    return text.replace(old, new, 1)


def put_dry_run(tmp_path, text, name="2026-10-10-pass-01.md", track="telemetry"):
    folder = tmp_path / track / "dry_runs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


# --- the committed tree ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("track", rt.TRACKS)
def test_every_committed_track_folder_holds_together(track):
    assert rt.check_track(REFINE / track) == []


def test_the_telemetry_track_keeps_a_recorded_dry_run_of_pass_one():
    record = rt.parse_dry_run(recorded_text())

    assert (record.track, record.number, record.date) == ("telemetry", 1, "2026-10-10")
    assert record.ended.utcoffset() is not None
    assert rf.check_pass_entry(record.entry) == []
    assert rt.check_dry_run(RECORDED) == []


def test_the_recorded_dry_run_cites_real_rows_of_the_metrics_file():
    entry = rt.parse_dry_run(recorded_text()).entry
    cited = sorted(reading.source for reading in entry.metrics_before if reading.cited)

    assert cited == ["metrics.jsonl:12", "metrics.jsonl:17", "metrics.jsonl:7", "metrics.jsonl:8"]
    assert rt.metric_citation_problems(entry.metrics_before, METRICS.read_text(encoding="utf-8").splitlines(), "x") == []


# --- a track folder -----------------------------------------------------------------------------------------------


def test_a_track_with_no_pass_yet_holds_together(tmp_path):
    assert rt.check_track(make_track(tmp_path)) == []


def test_a_track_after_one_valid_pass_holds_together(tmp_path):
    assert rt.check_track(make_track(tmp_path, entries=[entry_text()])) == []


def test_a_pass_that_did_not_write_the_counter_is_reported(tmp_path):
    problems = rt.check_track(make_track(tmp_path, entries=[entry_text()], counter=0))

    assert problems == [
        "refine_pass.txt reads 0 but the log's pass entries are numbered [1]; "
        "they must run from 1 to the counter, which a pass writes last"
    ]


def test_a_counter_ahead_of_the_log_is_reported(tmp_path):
    problems = rt.check_track(make_track(tmp_path, counter=1))

    assert problems == [
        "refine_pass.txt reads 1 but the log's pass entries are numbered []; "
        "they must run from 1 to the counter, which a pass writes last"
    ]


def test_a_pass_without_its_report_is_reported(tmp_path):
    problems = rt.check_track(make_track(tmp_path, entries=[entry_text()], reports=False))

    assert problems == ["pass 1 has no report at pass_reports/pass_01.md"]


def test_a_report_without_its_ended_line_is_reported(tmp_path):
    folder = make_track(tmp_path, entries=[entry_text()])
    report = folder / "pass_reports" / "pass_01.md"
    report.write_text(edit(report.read_text(encoding="utf-8"), "Ended: 2026-10-10T11:00:00-04:00\n", ""), encoding="utf-8")

    assert rt.check_track(folder) == [
        "pass_reports/pass_01.md: the second line must state when the pass ended: 'Ended: <date>T<time><offset>'"
    ]


def test_a_report_ended_time_without_an_offset_is_reported(tmp_path):
    folder = make_track(tmp_path, entries=[entry_text()])
    report = folder / "pass_reports" / "pass_01.md"
    report.write_text(edit(report.read_text(encoding="utf-8"), "11:00:00-04:00", "11:00:00"), encoding="utf-8")

    problems = rt.check_track(folder)

    assert len(problems) == 1
    assert problems[0].startswith("pass_reports/pass_01.md: 'Ended: 2026-10-10T11:00:00' has no UTC offset")


def test_a_report_written_for_another_pass_is_reported(tmp_path):
    folder = make_track(tmp_path, entries=[entry_text()])
    report = folder / "pass_reports" / "pass_01.md"
    report.write_text(edit(report.read_text(encoding="utf-8"), "telemetry refinement pass 1 (", "telemetry refinement pass 2 ("), encoding="utf-8")

    assert rt.check_track(folder) == [
        "pass_reports/pass_01.md: the first line must read 'Verbatim report returned by telemetry refinement pass 1 (...'"
    ]


def test_a_report_that_does_not_quote_its_entry_is_reported(tmp_path):
    folder = make_track(tmp_path, entries=[entry_text()])
    report = folder / "pass_reports" / "pass_01.md"
    report.write_text(edit(report.read_text(encoding="utf-8"), "## Pass 1, 2026-10-10\n", ""), encoding="utf-8")

    assert rt.check_track(folder) == [
        "pass_reports/pass_01.md: does not quote its entry; no line reads '## Pass 1, 2026-10-10'"
    ]


def test_an_entry_without_its_metrics_before_block_is_reported(tmp_path):
    text = entry_text()
    start = text.index("Metrics before (read before changing anything):")
    end = text.index("Walked (task or experiment")
    folder = make_track(tmp_path, entries=[text[:start] + text[end:]])

    assert rt.check_track(folder) == ["REFINE_LOG.md: pass 1: 'Metrics before' needs a table with at least one reading"]


def test_a_change_without_a_source_is_reported(tmp_path):
    text = entry_text()
    source = re.search(r"^   SOURCE: AMD SMI 7\.0\.2 reference.*\n", text, flags=re.MULTILINE).group(0)
    folder = make_track(tmp_path, entries=[edit(text, source, "")])

    assert rt.check_track(folder) == ["REFINE_LOG.md: pass 1: change 2 has no SOURCE"]


def test_a_metrics_line_cited_for_another_metric_is_reported(tmp_path):
    folder = make_track(tmp_path, entries=[edit(entry_text(), "| metrics.jsonl:12 |", "| metrics.jsonl:13 |")])

    assert rt.check_track(folder) == [
        "REFINE_LOG.md pass 1: teacher.joules_per_token = 25.97 on pod-a cites metrics.jsonl:13, "
        "which holds metric 'trainer.samples_per_s', node 'pod-b', value 0.51"
    ]


def test_a_metrics_line_cited_for_another_value_is_reported(tmp_path):
    folder = make_track(tmp_path, entries=[edit(entry_text(), "| teacher.watts_avg | 200 |", "| teacher.watts_avg | 202 |")])

    assert rt.check_track(folder) == [
        "REFINE_LOG.md pass 1: teacher.watts_avg = 202 on pod-a cites metrics.jsonl:7, which holds value 200"
    ]


def test_a_metrics_line_past_the_end_of_the_file_is_reported(tmp_path):
    folder = make_track(tmp_path, entries=[edit(entry_text(), "| metrics.jsonl:17 |", "| metrics.jsonl:999 |")])

    assert rt.check_track(folder) == ["REFINE_LOG.md pass 1: pod.idle_watts cites metrics.jsonl:999, which holds no row"]


def test_a_reading_with_no_value_stored_or_a_ledger_row_is_not_looked_up(tmp_path):
    entry = rt.parse_dry_run(recorded_text()).entry
    absent = [reading for reading in entry.metrics_before if reading.absent]
    ledger = [rf.Reading("pod.idle_watts", "39.0", "W", "pod-a", "idle", "ledger:41")]

    assert len(absent) == 7
    assert rt.metric_citation_problems([*absent, *ledger], [], "x") == []


def test_an_unreadable_log_is_reported_once(tmp_path):
    folder = make_track(tmp_path)
    (folder / "REFINE_LOG.md").write_text(LOG_HEADER + "\n## Pass one, someday\n", encoding="utf-8")

    problems = rt.check_track(folder)

    assert len(problems) == 1
    assert problems[0].startswith("REFINE_LOG.md: unrecognized log heading '## Pass one, someday'")


def test_a_counter_that_is_not_a_number_is_reported_once(tmp_path):
    folder = make_track(tmp_path)
    (folder / "refine_pass.txt").write_text("one\n", encoding="utf-8")

    assert rt.check_track(folder) == ["refine_pass.txt: refine_pass.txt must hold one non-negative integer, found 'one\\n'"]


# --- dry-run records ----------------------------------------------------------------------------------------------


def test_a_dry_run_does_not_count_as_a_pass(tmp_path):
    folder = make_track(tmp_path)
    put_dry_run(tmp_path, recorded_text())

    assert rt.check_track(folder) == []


def test_a_broken_dry_run_in_a_track_folder_is_reported_by_the_track_check(tmp_path):
    folder = make_track(tmp_path)
    put_dry_run(tmp_path, edit(recorded_text(), "| metrics.jsonl:8 |", "| metrics.jsonl:9 |"))

    assert rt.check_track(folder) == [
        "dry_runs/2026-10-10-pass-01.md: teacher.watts_peak = 300 on pod-a cites metrics.jsonl:9, "
        "which holds metric 'teacher.engine_init_s', value 270"
    ]


def test_a_dry_run_named_for_another_pass_is_reported(tmp_path):
    path = put_dry_run(tmp_path, recorded_text(), name="2026-10-10-pass-02.md")

    assert rt.check_dry_run(path) == [
        "2026-10-10-pass-02.md: the name says pass 2 on 2026-10-10 but the record rehearses pass 1 on 2026-10-10"
    ]


def test_a_dry_run_with_a_name_outside_the_pattern_is_reported(tmp_path):
    path = put_dry_run(tmp_path, recorded_text(), name="rehearsal.md")

    assert rt.check_dry_run(path) == ["rehearsal.md: a dry-run record is named <YYYY-MM-DD>-pass-<NN>.md"]


def test_a_dry_run_in_another_tracks_folder_is_reported(tmp_path):
    path = put_dry_run(tmp_path, recorded_text(), track="kernels")

    assert rt.check_dry_run(path) == [
        "2026-10-10-pass-01.md: the record rehearses the telemetry track but sits in the kernels folder"
    ]


def test_a_dry_run_for_a_track_that_does_not_exist_is_reported(tmp_path):
    text = edit(recorded_text(), "Dry run of telemetry refinement pass 1", "Dry run of speed refinement pass 1")
    path = put_dry_run(tmp_path, text, track="speed")

    assert rt.check_dry_run(path) == [
        "2026-10-10-pass-01.md: 'speed' is not a track; the tracks are telemetry, learning, plumbing, kernels"
    ]


def test_a_dry_run_whose_entry_is_another_pass_is_reported(tmp_path):
    path = put_dry_run(tmp_path, edit(recorded_text(), "## Pass 1, 2026-10-10", "## Pass 2, 2026-10-10"))

    assert rt.check_dry_run(path) == [
        "2026-10-10-pass-01.md: the record rehearses pass 1 on 2026-10-10 but its entry is pass 2 on 2026-10-10"
    ]


def test_a_dry_run_without_its_header_line_is_reported(tmp_path):
    text = recorded_text()
    path = put_dry_run(tmp_path, "Rehearsal." + text[text.index("\n"):])

    assert rt.check_dry_run(path) == [
        "2026-10-10-pass-01.md: a dry-run record starts 'Dry run of <track> refinement pass <N> (<who>), <YYYY-MM-DD>.'"
    ]


def test_a_dry_run_without_its_ended_line_is_reported(tmp_path):
    text = recorded_text()
    ended = re.search(r"^Ended: .*\n", text, flags=re.MULTILINE).group(0)
    path = put_dry_run(tmp_path, edit(text, ended, ""))

    assert rt.check_dry_run(path) == [
        "2026-10-10-pass-01.md: the second line of a dry-run record states when it ended: 'Ended: <date>T<time><offset>'"
    ]


def test_a_dry_run_without_its_entry_block_is_reported(tmp_path):
    path = put_dry_run(tmp_path, edit(recorded_text(), "```text dry-run:pass-entry\n", "```text\n"))

    assert rt.check_dry_run(path) == [
        "2026-10-10-pass-01.md: a dry-run record holds exactly one ```text dry-run:pass-entry block, found 0"
    ]


def test_a_dry_run_block_holding_two_entries_is_reported(tmp_path):
    text = recorded_text()
    path = put_dry_run(tmp_path, edit(text, "Checks run (commands and their actual output):", "## Pass 2, 2026-10-10\n\nChecks run (commands and their actual output):"))

    problems = rt.check_dry_run(path)

    assert problems == [
        "2026-10-10-pass-01.md: the dry-run block holds one '## Pass N, YYYY-MM-DD' entry and nothing else, found 2 entries"
    ]


def test_a_dry_run_entry_with_an_uncited_change_is_reported(tmp_path):
    text = recorded_text()
    source = re.search(r"^   SOURCE: lab/refine/research/telemetry/power-sampling\.md#3.*\n", text, flags=re.MULTILINE).group(0)
    path = put_dry_run(tmp_path, edit(text, source, ""))

    assert rt.check_dry_run(path) == ["2026-10-10-pass-01.md: pass 1: change 5 has no SOURCE"]


def test_a_dry_run_with_crlf_line_endings_parses_like_lf(tmp_path):
    path = tmp_path / "telemetry" / "dry_runs" / "2026-10-10-pass-01.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(recorded_text().replace("\n", "\r\n").encode("utf-8"))

    assert rt.check_dry_run(path) == []
    assert rt.parse_dry_run(path.read_text(encoding="utf-8")).entry.number == 1
