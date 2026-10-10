"""Tests for the owner-rulings flow: parsing the four tracks, recording answers in the logs, mirroring to Notion.

The rulings come from files built the way the tracks' own files are built (the committed headers, the
rulings and corrections shown in lab/refine/TEMPLATES.md), parsed by the real parsers. Only Notion is
faked, at its boundary: a page store that behaves like the page-markdown endpoints, and a transport
under the REST client that answers like the API reference says it does.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab import __main__ as dispatcher
from lab.machine import refine_formats as rf
from lab.machine import rulings
from lab.machine.lock import FileLock

REPO = Path(__file__).resolve().parents[1]
REFINE = REPO / "lab" / "refine"
TRACKS = ("telemetry", "learning", "plumbing", "kernels")

HEART = "\U0001F916"
DOT = "\u00b7"
NOW = datetime(2026, 10, 10, 14, 30, tzinfo=timezone.utc)  # Saturday 10:30 AM Eastern
SIGN = f"{HEART} Ilan's agent {DOT} Sat 10:30 AM ET:"
TOKEN = "ntn_test_value_0123456789"

RULINGS_HEADER = (REFINE / "telemetry" / "OWNER_RULINGS.md").read_text(encoding="utf-8")
LOG_HEADER = (REFINE / "telemetry" / "REFINE_LOG.md").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    """Keep the lock file and the STOP file of every test out of the repository's lab/state."""
    path = tmp_path / "lab-state"
    monkeypatch.setenv("LAB_STATE_DIR", str(path))
    return path


def example(name: str) -> str:
    """The body of the ```text example:<name> block in lab/refine/TEMPLATES.md."""
    text = (REFINE / "TEMPLATES.md").read_text(encoding="utf-8")
    found = re.search(rf"^```text example:{re.escape(name)}\n(.*?)^```\s*$", text, flags=re.MULTILINE | re.DOTALL)
    assert found, f"lab/refine/TEMPLATES.md has no ```text example:{name} block"
    return found.group(1)


def ruling(number, question, *, origin="pass 3", now="the plan does X.", recommend="Do Y.", tension="A against B.", answer="____"):
    return (
        f"## {number}. {question} ({origin})\n\n"
        f"Now: {now}\nRecommend: {recommend}\nSource of tension: {tension}\nAnswer: {answer}\n"
    )


def make_tree(root: Path, tracks: dict) -> Path:
    """A refine folder: per track the committed header plus the given ruling blocks, a header-only log and a counter."""
    refine = root / "refine"
    for name in TRACKS:
        folder = refine / name
        folder.mkdir(parents=True)
        blocks = tracks.get(name, [])
        (folder / "OWNER_RULINGS.md").write_text(RULINGS_HEADER + "".join("\n" + block for block in blocks), encoding="utf-8", newline="\n")
        (folder / "REFINE_LOG.md").write_text(LOG_HEADER, encoding="utf-8", newline="\n")
        (folder / "refine_pass.txt").write_text("0\n", encoding="utf-8", newline="\n")
    return refine


T1 = ruling(
    1,
    "Should the power sampler run at 1 Hz or 10 Hz?",
    origin="pass 2",
    now="1 Hz, the rate the hub reports.",
    recommend="10 Hz, because a 15-minute slot has no time for a coarse curve.",
    tension="RUBRIC.md TT2 against the sampler overhead in lab/README.md.",
)
T2_QUESTION = "Is idle power part of the joules per token?"
T2_OPEN = ruling(2, T2_QUESTION, origin="merge 1", now="No, only generation time counts.", recommend="Yes, report both.", tension="TT3 against the telemetry spec.")
T2_ANSWER = "report both (owner, 2026-10-10)"
T2_ANSWERED = ruling(2, T2_QUESTION, origin="merge 1", now="No, only generation time counts.", recommend="Yes, report both.", tension="TT3 against the telemetry spec.", answer=T2_ANSWER)
L1 = ruling(
    1,
    "Which base model does the first adapter train on?",
    origin="pass 1",
    now="The 4B student.",
    recommend="The 4B student.",
    tension="The design doc against lab/README.md.",
)
K4_OPEN = example("ruling")  # kernels ruling 4, unanswered


def k4_answered() -> str:
    return K4_OPEN.replace("Answer: ____", example("ruling-answered").strip())


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def log_path(refine: Path, track: str) -> Path:
    return refine / track / "REFINE_LOG.md"


def rulings_path(refine: Path, track: str) -> Path:
    return refine / track / "OWNER_RULINGS.md"


def answer_ruling(refine: Path, track: str, number: int, answer: str) -> None:
    """Fill in a ruling's Answer line the way the owner would, by editing the file."""
    path = rulings_path(refine, track)
    text = read(path)
    block = re.search(rf"^## {number}\. .*?(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    assert block, f"ruling {number} is not in {path}"
    write(path, text.replace(block.group(0), re.sub(r"^Answer:.*$", f"Answer: {answer}", block.group(0), flags=re.MULTILINE)))


def run(refine: Path, *argv: str, **kwargs):
    return rulings.main([argv[0], "--refine-dir", str(refine), *argv[1:]], now=NOW, **kwargs)


def open_items(refine: Path):
    rows, errors, warnings = rulings.collect(refine)
    assert (errors, warnings) == ([], [])
    return [row for row in rows if row.state == "open"]


# ------------------------------------------------------------------ the committed files


def test_the_four_committed_tracks_are_found_and_load_without_errors():
    assert rulings.discover(REFINE) == list(TRACKS)
    result = rulings.main(["list", "--all"])
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["unanswered"] + result["answered"] == len(result["rulings"])


def test_every_committed_open_ruling_is_listed_with_its_default_and_recommendation():
    result = rulings.main(["list"])

    for row in result["rulings"]:
        assert row["now"] and row["recommend"] and row["tension"]


# ------------------------------------------------------------------ list


def test_list_returns_every_open_ruling_of_every_track_with_default_and_recommendation(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED], "learning": [L1], "kernels": [K4_OPEN]})

    result = run(refine, "list")

    assert result["ok"] is True
    assert (result["unanswered"], result["answered"]) == (3, 1)
    assert result["rulings"] == [
        {
            "track": "telemetry",
            "number": 1,
            "question": "Should the power sampler run at 1 Hz or 10 Hz?",
            "origin": "pass 2",
            "now": "1 Hz, the rate the hub reports.",
            "recommend": "10 Hz, because a 15-minute slot has no time for a coarse curve.",
            "tension": "RUBRIC.md TT2 against the sampler overhead in lab/README.md.",
        },
        {
            "track": "learning",
            "number": 1,
            "question": "Which base model does the first adapter train on?",
            "origin": "pass 1",
            "now": "The 4B student.",
            "recommend": "The 4B student.",
            "tension": "The design doc against lab/README.md.",
        },
        {
            "track": "kernels",
            "number": 4,
            "question": "Must a kernel win beat the run-to-run spread, or a fixed percentage?",
            "origin": "pass 3",
            "now": "the spread, as kernel test 3 says; E3 uses five runs per point.",
            "recommend": (
                "the spread. A fixed percentage ignores how noisy a pod is on a given day, "
                "and the spread costs only the repeat runs already added."
            ),
            "tension": (
                'Notion task O4, "Kernel tests" 3 (a win must exceed the spread) against the 15-minute slot the GPU queue '
                "gives a benchmark, which five runs at five batch sizes may not fit."
            ),
        },
    ]
    assert result["errors"] == []


def test_list_all_adds_the_answer_and_where_each_ruling_stands(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED]})

    result = run(refine, "list", "--all")

    assert [(r["number"], r["state"], r.get("answer")) for r in result["rulings"]] == [
        (1, "open", "____"),
        (2, "new", T2_ANSWER),
    ]


def test_list_can_be_limited_to_tracks_and_refuses_a_track_that_does_not_exist(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1], "learning": [L1]})

    only = run(refine, "list", "--track", "learning")
    unknown = run(refine, "list", "--track", "learning", "--track", "speed")

    assert [(r["track"], r["number"]) for r in only["rulings"]] == [("learning", 1)]
    assert unknown["ok"] is False
    assert unknown["outcome"] == "unknown_track"
    assert unknown["tracks"] == list(TRACKS)


def test_list_reports_a_track_whose_file_breaks_the_format_and_still_lists_the_others(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1], "learning": [L1]})
    broken = rulings_path(refine, "telemetry")
    write(broken, read(broken) + "\n## 3. A ruling with no Answer line (pass 4)\n\nNow: x\nRecommend: y\nSource of tension: z\n")

    result = run(refine, "list")

    assert result["ok"] is False
    assert [(r["track"], r["number"]) for r in result["rulings"]] == [("learning", 1)]
    assert [e["track"] for e in result["errors"]] == ["telemetry"]
    assert "OWNER_RULINGS.md" in result["errors"][0]["error"]
    assert "ruling 3 has no 'Answer:' line" in result["errors"][0]["error"]


def test_a_log_that_breaks_the_format_is_a_warning_and_does_not_hide_the_open_rulings(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED]})
    broken = log_path(refine, "telemetry")
    write(broken, read(broken) + "\n## Something else entirely\n\ntext\n")

    plain = run(refine, "list")
    everything = run(refine, "list", "--all")

    assert plain["ok"] is True
    assert [(r["track"], r["number"]) for r in plain["rulings"]] == [("telemetry", 1)]
    assert [w["track"] for w in plain["warnings"]] == ["telemetry"]
    assert "unrecognized log heading" in plain["warnings"][0]["error"]
    assert [(r["number"], r["state"]) for r in everything["rulings"]] == [(1, "open"), (2, "unknown")]


def test_a_folder_without_any_rulings_file_is_not_a_track(tmp_path):
    result = rulings.main(["list", "--refine-dir", str(tmp_path)])

    assert result["ok"] is False
    assert result["outcome"] == "no_tracks"


# ------------------------------------------------------------------ apply


def test_apply_appends_an_owner_correction_that_cites_the_ruling_and_reads_back(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED]})
    before = read(log_path(refine, "telemetry"))

    result = run(refine, "apply", "--date", "2026-10-10")

    after = read(log_path(refine, "telemetry"))
    assert result["ok"] is True
    assert result["recorded"] == [{"track": "telemetry", "ruling": 2, "state": "new"}]
    assert result["written"] == [str(log_path(refine, "telemetry"))]
    assert after.startswith(before)
    appended = after[len(before):]
    digest = re.search(r"answer digest ([0-9a-f]{8}) for ruling 2", appended).group(1)
    assert appended.replace(digest, "DIGEST") == (
        "\n## Owner correction, 2026-10-10 (no increment)\n\n"
        "Rulings applied: 2\n\n"
        f'Ruling 2 ({T2_QUESTION}) was answered by the owner and is now an owner rule at the top of the authority order. '
        f'Answer, as written: "{T2_ANSWER}" (answer digest DIGEST for ruling 2). '
        'Until this answer the plan followed the default: "No, only generation time counts." '
        "No plan text changed here. The next pass makes the plan follow the answer, keeps it that way, "
        "and cites lab/refine/telemetry/OWNER_RULINGS.md 2 (owner, 2026-10-10) in the SOURCE line of each change it makes for it.\n\n"
        "---\n"
    )
    entry = rf.parse_log(after).entries[-1]
    assert isinstance(entry, rf.OwnerCorrection)
    assert (entry.date, entry.rulings, entry.changes) == ("2026-10-10", (2,), ())


def test_apply_prints_what_the_next_pass_must_change_with_the_source_to_cite(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED], "kernels": [k4_answered()]})

    result = run(refine, "apply", "--date", "2026-10-11")

    assert result["next_pass_must_change"] == [
        {
            "track": "telemetry",
            "ruling": 2,
            "question": T2_QUESTION,
            "state": "new",
            "was": "No, only generation time counts.",
            "owner_rule": T2_ANSWER,
            "recommend": "Yes, report both.",
            "source": "lab/refine/telemetry/OWNER_RULINGS.md 2 (owner, 2026-10-10)",
            "instruction": (
                "Make the plan follow the owner's answer wherever it still follows the default in 'was', and cite "
                "lab/refine/telemetry/OWNER_RULINGS.md 2 (owner, 2026-10-10) in the SOURCE line of each change. "
                "A change to RUBRIC.md waits for the merge pass, which cites the same source."
            ),
        },
        {
            "track": "kernels",
            "ruling": 4,
            "question": "Must a kernel win beat the run-to-run spread, or a fixed percentage?",
            "state": "new",
            "was": "the spread, as kernel test 3 says; E3 uses five runs per point.",
            "owner_rule": "the spread, but three runs per point so the sweep fits one slot (owner, 2026-10-10)",
            "recommend": (
                "the spread. A fixed percentage ignores how noisy a pod is on a given day, "
                "and the spread costs only the repeat runs already added."
            ),
            "source": "lab/refine/kernels/OWNER_RULINGS.md 4 (owner, 2026-10-10)",
            "instruction": (
                "Make the plan follow the owner's answer wherever it still follows the default in 'was', and cite "
                "lab/refine/kernels/OWNER_RULINGS.md 4 (owner, 2026-10-10) in the SOURCE line of each change. "
                "A change to RUBRIC.md waits for the merge pass, which cites the same source."
            ),
        },
    ]


def test_an_answer_without_a_date_is_cited_with_the_day_it_was_recorded(tmp_path):
    refine = make_tree(tmp_path, {"learning": [ruling(1, "Which base?", answer="the 4B student")]})

    result = run(refine, "apply", "--date", "2026-10-12")

    assert result["next_pass_must_change"][0]["source"] == "lab/refine/learning/OWNER_RULINGS.md 1 (owner, 2026-10-12)"


def test_apply_changes_neither_the_rulings_files_nor_the_counters(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED], "kernels": [k4_answered()]})
    watched = [p for track in TRACKS for p in (rulings_path(refine, track), refine / track / "refine_pass.txt")]
    before = {p: p.read_bytes() for p in watched}

    run(refine, "apply")

    assert {p: p.read_bytes() for p in watched} == before


def test_apply_twice_records_each_ruling_once_and_keeps_listing_what_the_pass_must_change(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED]})
    first = run(refine, "apply", "--date", "2026-10-10")
    log_after_first = log_path(refine, "telemetry").read_bytes()

    second = run(refine, "apply", "--date", "2026-10-10")

    assert log_path(refine, "telemetry").read_bytes() == log_after_first
    assert (second["recorded"], second["written"]) == ([], [])
    assert second["already_recorded"] == [{"track": "telemetry", "ruling": 2}]
    assert [row["state"] for row in second["next_pass_must_change"]] == ["recorded"]
    assert second["next_pass_must_change"][0]["source"] == first["next_pass_must_change"][0]["source"]


def test_a_pass_entry_after_the_correction_clears_the_list_and_the_ruling_counts_as_done(tmp_path):
    refine = make_tree(tmp_path, {"kernels": [k4_answered()]})
    run(refine, "apply", "--date", "2026-10-10")
    kernels_log = log_path(refine, "kernels")
    write(kernels_log, read(kernels_log) + "\n" + example("pass-entry"))

    result = run(refine, "apply", "--date", "2026-10-11")
    states = run(refine, "list", "--all")

    assert result["next_pass_must_change"] == []
    assert result["already_recorded"] == [{"track": "kernels", "ruling": 4}]
    assert (result["recorded"], result["written"]) == ([], [])
    assert [(r["number"], r["state"]) for r in states["rulings"]] == [(4, "done")]


def test_an_answer_rewritten_after_it_was_recorded_is_recorded_again(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    run(refine, "apply", "--date", "2026-10-10")
    answer_ruling(refine, "telemetry", 2, "report only generation time (owner, 2026-10-11)")

    listed = run(refine, "list", "--all")
    result = run(refine, "apply", "--date", "2026-10-11")

    assert [(r["number"], r["state"]) for r in listed["rulings"]] == [(2, "reanswered")]
    assert result["recorded"] == [{"track": "telemetry", "ruling": 2, "state": "reanswered"}]
    corrections = [e for e in rf.parse_log(read(log_path(refine, "telemetry"))).entries if isinstance(e, rf.OwnerCorrection)]
    assert [(c.date, c.rulings) for c in corrections] == [("2026-10-10", (2,)), ("2026-10-11", (2,))]
    assert "was answered again by the owner, and this answer replaces the earlier one" in corrections[1].summary
    assert 'Answer, as written: "report only generation time (owner, 2026-10-11)"' in corrections[1].summary
    assert result["next_pass_must_change"][0]["owner_rule"] == "report only generation time (owner, 2026-10-11)"
    assert result["next_pass_must_change"][0]["state"] == "reanswered"


def test_several_rulings_of_one_track_share_one_correction(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [ruling(1, "First?", answer="yes"), ruling(2, "Second?", answer="no"), ruling(3, "Third?")]})

    result = run(refine, "apply", "--date", "2026-10-10")

    corrections = [e for e in rf.parse_log(read(log_path(refine, "telemetry"))).entries if isinstance(e, rf.OwnerCorrection)]
    assert [c.rulings for c in corrections] == [(1, 2)]
    assert [r["ruling"] for r in result["recorded"]] == [1, 2]


def test_a_dry_run_writes_nothing_and_returns_the_entry_it_would_append(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    before = log_path(refine, "telemetry").read_bytes()

    preview = run(refine, "apply", "--dry-run", "--date", "2026-10-10")
    real = run(refine, "apply", "--date", "2026-10-10")

    assert preview["dry_run"] is True
    assert preview["written"] == []
    assert preview["recorded"] == [{"track": "telemetry", "ruling": 2, "state": "new"}]
    assert real["written"] == [str(log_path(refine, "telemetry"))]
    assert read(log_path(refine, "telemetry")) == before.decode("utf-8") + "\n" + preview["preview"]["telemetry"]


def test_the_entry_is_dated_with_the_eastern_day_when_no_date_is_given(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    late = datetime(2026, 10, 11, 3, 30, tzinfo=timezone.utc)  # still Saturday evening in New York

    rulings.main(["apply", "--refine-dir", str(refine)], now=late)

    entry = rf.parse_log(read(log_path(refine, "telemetry"))).entries[-1]
    assert entry.date == "2026-10-10"


def test_a_hand_written_correction_that_carries_the_changes_marks_the_ruling_done(tmp_path):
    refine = make_tree(tmp_path, {"kernels": [k4_answered()]})
    kernels_log = log_path(refine, "kernels")
    write(kernels_log, read(kernels_log) + "\n" + example("owner-correction"))
    before = kernels_log.read_bytes()

    result = run(refine, "apply")

    assert kernels_log.read_bytes() == before
    assert (result["recorded"], result["next_pass_must_change"]) == ([], [])
    assert result["already_recorded"] == [{"track": "kernels", "ruling": 4}]


def test_a_track_whose_log_does_not_parse_is_left_alone_and_the_others_are_still_recorded(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED], "learning": [ruling(1, "Which base?", answer="the 4B student")]})
    learning_log = log_path(refine, "learning")
    write(learning_log, read(learning_log) + "\n## Something else entirely\n\ntext\n")
    broken = learning_log.read_bytes()

    result = run(refine, "apply", "--date", "2026-10-10")

    assert result["ok"] is False
    assert [e["track"] for e in result["errors"]] == ["learning"]
    assert "unrecognized log heading" in result["errors"][0]["error"]
    assert learning_log.read_bytes() == broken
    assert result["recorded"] == [{"track": "telemetry", "ruling": 2, "state": "new"}]


def test_mirror_still_posts_the_open_rulings_when_a_log_cannot_be_read(tmp_path):
    refine = tree_with_two_open(tmp_path)
    broken = log_path(refine, "kernels")
    write(broken, read(broken) + "\n## Something else entirely\n\ntext\n")
    notion = FakePages()

    result = run(refine, "mirror", client=notion)

    assert result["ok"] is True
    assert result["unanswered"] == 2
    assert [w["track"] for w in result["warnings"]] == ["kernels"]
    assert result["targets"] == {"overnight_status": {"action": "posted"}, "team_comms_fyi": {"action": "posted"}}


def test_a_track_without_a_log_reports_it_instead_of_inventing_one(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    log_path(refine, "telemetry").unlink()

    result = run(refine, "apply")

    assert result["ok"] is False
    assert "there is no log to record the answers in" in result["errors"][0]["error"]
    assert not log_path(refine, "telemetry").exists()


def test_the_correction_keeps_the_logs_windows_line_endings(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    crlf = read(log_path(refine, "telemetry")).replace("\n", "\r\n").rstrip("\r\n")  # no final newline either
    log_path(refine, "telemetry").write_bytes(crlf.encode("utf-8"))

    run(refine, "apply", "--date", "2026-10-10")

    raw = log_path(refine, "telemetry").read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n")
    entry = rf.parse_log(raw.decode("utf-8")).entries[-1]
    assert (entry.date, entry.rulings) == ("2026-10-10", (2,))


def test_apply_waits_for_the_state_lock_and_gives_up_with_a_busy_answer(tmp_path, state_dir):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    before = log_path(refine, "telemetry").read_bytes()

    with FileLock(state_dir / rulings.LOCK_NAME, timeout=1.0):
        result = rulings.apply_rulings(refine, (), today="2026-10-10", lock_timeout=0.05)

    assert result["ok"] is False
    assert result["outcome"] == "busy"
    assert log_path(refine, "telemetry").read_bytes() == before


def test_applies_running_at_once_record_a_ruling_exactly_once(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    results = []

    def go():
        results.append(rulings.apply_rulings(refine, (), today="2026-10-10"))

    threads = [threading.Thread(target=go) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    corrections = [e for e in rf.parse_log(read(log_path(refine, "telemetry"))).entries if isinstance(e, rf.OwnerCorrection)]
    assert len(results) == 4
    assert all(r["ok"] for r in results)
    assert len(corrections) == 1
    assert sum(len(r["recorded"]) for r in results) == 1


# ------------------------------------------------------------------ Notion text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain words stay", "plain words stay"),
        ("Use OWNER_RULINGS.md here", "Use `OWNER_RULINGS.md` here"),
        ("see lab/refine/kernels/REFINE_LOG.md.", "see `lab/refine/kernels/REFINE_LOG.md`."),
        ("(refine_pass.txt)", "(`refine_pass.txt`)"),
        ("lab/__main__.py", "`lab/__main__.py`"),
        ("e.g. rulings.py is fine", "e.g. `rulings.py` is fine"),
        ("tok/s and J/token stay", "tok/s and J/token stay"),
        ("a < b and c > d", "a \\< b and c \\> d"),
        ("5 * 3 [x] {y} | z ^ $5 ~ w", "5 \\* 3 \\[x\\] \\{y\\} \\| z \\^ \\$5 \\~ w"),
        ("keep `inline_code *as is*` and escape *this*", "keep `inline_code *as is*` and escape \\*this\\*"),
        ("an unpaired ` tick", "an unpaired \\` tick"),
        ("line one\nline two   spaced", "line one line two spaced"),
    ],
)
def test_notion_text_escapes_markup_and_puts_file_names_in_code(raw, expected):
    assert rulings.notion_text(raw) == expected


# ------------------------------------------------------------------ the posts


def standing(track, number, question, now, recommend, tension, origin="pass 2"):
    return rulings.Standing(track, number, question, origin, now, recommend, tension, "", "open")


A = standing(
    "telemetry",
    1,
    "Should the power sampler run at 1 Hz or 10 Hz?",
    "1 Hz, the rate the hub reports.",
    "10 Hz, because a 15-minute slot has no time for a coarse curve.",
    "RUBRIC.md TT2 against the sampler overhead in lab/README.md.",
)
B = standing(
    "kernels",
    4,
    "Must a kernel win beat the run-to-run spread, or a fixed percentage?",
    "the spread, as kernel test 3 says; E3 uses five runs per point.",
    "the spread. A fixed percentage ignores how noisy a pod is on a given day.",
    "kernel test 3 against the 15-minute slot.",
)


def test_the_posts_are_signed_name_each_question_and_carry_the_list_id():
    posts = rulings.build_posts([A, B], NOW)

    assert re.fullmatch(r"[0-9a-f]{8}", posts.list_id)
    assert posts.count == 2
    assert posts.segment == (
        f'{SIGN} Lab rulings waiting on you (2): telemetry 1 "Should the power sampler run at 1 Hz or 10 Hz?"; '
        'kernels 4 "Must a kernel win beat the run-to-run spread, or a fixed percentage?"'
        " - the FYI of the same name on your Team Comms page has each one with the plan's default and the recommendation"
        f" (list {posts.list_id})"
    )
    assert posts.block.split("\n") == [
        f"- {SIGN} Lab rulings waiting on you (2). These are questions the lab's refinement passes could not decide by themselves; "
        "until you answer one, the plan keeps the default listed with it. To answer, tell your agent the track, the number and your "
        f"decision, or write it on that ruling's Answer line in the file named. (list {posts.list_id})",
        "\t- telemetry question 1: Should the power sampler run at 1 Hz or 10 Hz? "
        "Until you answer, the plan does this: 1 Hz, the rate the hub reports. "
        "Recommendation: 10 Hz, because a 15-minute slot has no time for a coarse curve. "
        "Why it is open: `RUBRIC.md` TT2 against the sampler overhead in `lab/README.md`. "
        "File: `lab/refine/telemetry/OWNER_RULINGS.md`.",
        "\t- kernels question 4: Must a kernel win beat the run-to-run spread, or a fixed percentage? "
        "Until you answer, the plan does this: the spread, as kernel test 3 says; E3 uses five runs per point. "
        "Recommendation: the spread. A fixed percentage ignores how noisy a pod is on a given day. "
        "Why it is open: kernel test 3 against the 15-minute slot. "
        "File: `lab/refine/kernels/OWNER_RULINGS.md`.",
    ]


@pytest.mark.parametrize(
    ("moment", "stamp"),
    [
        (datetime(2026, 10, 10, 14, 30, tzinfo=timezone.utc), "Sat 10:30 AM ET:"),
        (datetime(2026, 10, 10, 16, 5, tzinfo=timezone.utc), "Sat 12:05 PM ET:"),
        (datetime(2026, 10, 11, 4, 7, tzinfo=timezone.utc), "Sun 12:07 AM ET:"),
        (datetime(2026, 12, 10, 17, 5, tzinfo=timezone.utc), "Thu 12:05 PM ET:"),
        (datetime(2026, 10, 10, 10, 30, tzinfo=timezone(timedelta(hours=-4))), "Sat 10:30 AM ET:"),
    ],
)
def test_the_signature_shows_the_day_and_eastern_time(moment, stamp):
    assert rulings.build_posts([A], moment).block.startswith(f"- {HEART} Ilan's agent {DOT} {stamp} Lab rulings waiting on you (1).")


def test_the_list_id_follows_the_wording_of_the_list_and_not_the_clock():
    base = rulings.build_posts([A, B], NOW).list_id

    assert rulings.build_posts([A, B], NOW + timedelta(hours=5)).list_id == base
    assert rulings.build_posts([A], NOW).list_id != base
    reworded = rulings.Standing(**{**A.__dict__, "recommend": "1 Hz."})
    assert rulings.build_posts([reworded, B], NOW).list_id != base


def test_a_list_of_nine_names_six_questions_and_counts_the_rest():
    nine = [standing("learning", n, f"Question number {n}?", "x.", "y.", "z.") for n in range(1, 10)]

    posts = rulings.build_posts(nine, NOW)

    assert posts.count == 9
    assert 'Lab rulings waiting on you (9): learning 1 "Question number 1?"; learning 2 "Question number 2?"; ' in posts.segment
    assert 'learning 6 "Question number 6?", and 3 more - the FYI' in posts.segment
    assert "Question number 7" not in posts.segment
    assert len(posts.block.split("\n")) == 10  # the first line and nine children


def test_a_long_title_is_cut_on_the_overnight_line_and_a_long_default_in_the_fyi_block():
    long_item = standing("plumbing", 3, "Q " + "word " * 40, "n " + "word " * 300, "r.", "t.")

    posts = rulings.build_posts([long_item], NOW)

    assert 'plumbing 3 "Q ' + "word " * 14 + 'word ..."' in posts.segment
    assert posts.segment.count("word") == 15
    child = posts.block.split("\n")[1]
    assert child.startswith("\t- plumbing question 3: Q " + "word " * 39 + "word Until you answer, the plan does this: n ")
    assert "word ... Recommendation: r. Why it is open: t. File:" in child
    assert child.count("word") == 40 + 139


def test_with_nothing_waiting_there_is_no_segment_and_the_block_is_a_closing_line():
    posts = rulings.build_posts([], NOW)

    assert posts.count == 0
    assert posts.segment is None
    assert posts.block == (
        f"- {SIGN} Lab rulings waiting on you: none. Every ruling in the lab's `OWNER_RULINGS.md` files now has an answer. (list {posts.list_id})"
    )


# ------------------------------------------------------------------ the mirror, against a fake page store


NEEDS_LINE = "- **Needs a human (Ilan):** Q2 and Q3 on Ilan \u2194 Dominick (dataset and package formats)"
OVERNIGHT = "\n".join(
    [
        '<callout icon="\U0001F319" color="purple_bg">',
        "\tEach person's coding agent rewrites **its own section** every cycle.",
        "</callout>",
        "## How agents update this page",
        "- Replace only your own section's lines. Never edit another person's section.",
        "---",
        "## Dominick",
        "- **Last update:** \U0001F916 Dominick's agent \u00b7 Sat 6:10 AM ET",
        "- **Needs a human (Dominick):** (1) who owns Sort It S1/S2",
        "## Hemdutt",
        "- **Last update:** \u2014",
        "- **Needs a human (Hemdutt):** \u2014",
        "## Ilan",
        f"- **Last update:** {HEART} Ilan's agent {DOT} Sat 6:50 AM ET",
        "- **Working on:** Ilan's lab machine, the Orchestration tasks",
        "- **Blocked on:** nothing on the lab side",
        NEEDS_LINE,
    ]
)
FYI_HEADING = "## FYI (no reply needed)"
TEAM_COMMS = "\n".join(
    [
        '<callout icon="\U0001F9EA" color="orange_bg">',
        "\tIlan owns: GPU lab setup (R2), teacher labeling (L2).",
        "</callout>",
        "## Open questions",
        "- [x] **Q1. Can the R9700 servers join Tailscale?**",
        "\t- Answer: No.",
        "- [ ] **Q2. Dataset format for `label.py`.**",
        "\t- Answer:",
        "## Answered",
        FYI_HEADING,
        f"- {HEART} Ilan's agent {DOT} Sat 6:40 AM: **Ilan's lane split into finer tasks on the board** so several agents can build in parallel.",
        "- **Start here with your coding agent:** have it read `AGENTS.md` in the repo.",
        "## Questions for Dominick",
        "-",
    ]
)


class FakePages:
    """A page store with the behavior of Notion's page-markdown endpoints: read the text, replace one exact match."""

    def __init__(self, overnight: str = OVERNIGHT, team_comms: str = TEAM_COMMS) -> None:
        self.pages = {rulings.OVERNIGHT_PAGE: overnight, rulings.TEAM_COMMS_PAGE: team_comms}
        self.edits: list[tuple[str, str, str]] = []

    @property
    def overnight(self) -> str:
        return self.pages[rulings.OVERNIGHT_PAGE]

    @property
    def team_comms(self) -> str:
        return self.pages[rulings.TEAM_COMMS_PAGE]

    def read_page(self, page_id: str) -> str:
        return self.pages[page_id]

    def replace_text(self, page_id: str, old: str, new: str) -> None:
        found = self.pages[page_id].count(old)
        if found != 1:
            raise rulings.NotionError(f"the text to replace occurs {found} times, not once")
        self.pages[page_id] = self.pages[page_id].replace(old, new)
        self.edits.append((page_id, old, new))


def mirrored(refine: Path, notion, moment=NOW):
    return rulings.mirror_rulings(open_items(refine), notion, moment)


def tree_with_two_open(tmp_path):
    return make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED], "kernels": [K4_OPEN]})


def test_the_first_mirror_adds_the_segment_to_the_needs_line_and_the_block_under_the_fyi_heading(tmp_path):
    refine = tree_with_two_open(tmp_path)
    notion = FakePages()

    result = mirrored(refine, notion)

    posts = rulings.build_posts(open_items(refine), NOW)
    assert result["ok"] is True
    assert result["targets"] == {"overnight_status": {"action": "posted"}, "team_comms_fyi": {"action": "posted"}}
    assert notion.overnight == OVERNIGHT.replace(NEEDS_LINE, NEEDS_LINE + "; " + posts.segment)
    assert notion.team_comms == TEAM_COMMS.replace(FYI_HEADING, FYI_HEADING + "\n" + posts.block)
    inserted = notion.team_comms.split("\n")[notion.team_comms.split("\n").index(FYI_HEADING) + 1]
    assert re.fullmatch(rf"- {HEART} Ilan's agent {DOT} Sat 10:30 AM ET: Lab rulings waiting on you \(2\)\..* \(list [0-9a-f]{{8}}\)", inserted)


def test_a_second_mirror_changes_nothing_even_at_another_time(tmp_path):
    refine = tree_with_two_open(tmp_path)
    notion = FakePages()
    mirrored(refine, notion)
    pages, edits = dict(notion.pages), list(notion.edits)

    again = mirrored(refine, notion, NOW + timedelta(hours=3))

    assert again["ok"] is True
    assert again["targets"] == {"overnight_status": {"action": "unchanged"}, "team_comms_fyi": {"action": "unchanged"}}
    assert notion.pages == pages
    assert notion.edits == edits


def test_a_changed_list_replaces_the_earlier_posts_in_place_instead_of_adding_new_ones(tmp_path):
    refine = tree_with_two_open(tmp_path)
    notion = FakePages()
    mirrored(refine, notion)
    first_id = rulings.build_posts(open_items(refine), NOW).list_id
    write(rulings_path(refine, "learning"), read(rulings_path(refine, "learning")) + "\n" + L1)

    result = mirrored(refine, notion, NOW + timedelta(hours=1))

    second = rulings.build_posts(open_items(refine), NOW + timedelta(hours=1))
    assert result["targets"] == {"overnight_status": {"action": "updated"}, "team_comms_fyi": {"action": "updated"}}
    assert first_id not in notion.overnight + notion.team_comms
    assert notion.overnight == OVERNIGHT.replace(NEEDS_LINE, NEEDS_LINE + "; " + second.segment)
    assert notion.team_comms == TEAM_COMMS.replace(FYI_HEADING, FYI_HEADING + "\n" + second.block)
    assert notion.team_comms.count("Lab rulings waiting on you (3).") == 1
    assert notion.overnight.count("Lab rulings waiting on you") == 1


def test_when_every_ruling_is_answered_the_segment_goes_and_the_fyi_says_none_is_waiting(tmp_path):
    refine = tree_with_two_open(tmp_path)
    notion = FakePages()
    mirrored(refine, notion)
    answer_ruling(refine, "telemetry", 1, "10 Hz (owner, 2026-10-10)")
    answer_ruling(refine, "kernels", 4, "the spread (owner, 2026-10-10)")

    result = mirrored(refine, notion, NOW + timedelta(hours=2))

    closing = rulings.build_posts([], NOW + timedelta(hours=2))
    assert result["targets"] == {"overnight_status": {"action": "removed"}, "team_comms_fyi": {"action": "closed"}}
    assert notion.overnight == OVERNIGHT
    assert notion.team_comms == TEAM_COMMS.replace(FYI_HEADING, FYI_HEADING + "\n" + closing.block)
    assert "question" not in closing.block
    again = mirrored(refine, notion, NOW + timedelta(hours=4))
    assert again["targets"] == {"overnight_status": {"action": "unchanged"}, "team_comms_fyi": {"action": "unchanged"}}


def test_an_empty_list_with_no_earlier_post_posts_nothing_at_all(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    notion = FakePages()

    result = mirrored(refine, notion)

    assert result["targets"] == {"overnight_status": {"action": "unchanged"}, "team_comms_fyi": {"action": "unchanged"}}
    assert notion.edits == []


def test_a_new_question_after_the_closing_line_replaces_the_closing_line(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    notion = FakePages()
    notion.pages[rulings.TEAM_COMMS_PAGE] = TEAM_COMMS.replace(
        FYI_HEADING, FYI_HEADING + "\n" + rulings.build_posts([], NOW - timedelta(days=1)).block
    )
    write(rulings_path(refine, "learning"), read(rulings_path(refine, "learning")) + "\n" + L1)

    result = mirrored(refine, notion)

    assert result["targets"]["team_comms_fyi"] == {"action": "updated"}
    assert "none. Every ruling" not in notion.team_comms
    assert notion.team_comms.count("Lab rulings waiting on you") == 1


def test_text_the_flow_did_not_write_is_never_replaced(tmp_path):
    refine = tree_with_two_open(tmp_path)
    human = "- Lab rulings waiting on you is what Dominick calls the lab list (list abcd1234)"
    signed_by_someone_else = f"- {HEART} Dominick's agent {DOT} Sat 6:00 AM ET: Lab rulings waiting on you (list 12345678)"
    page = TEAM_COMMS.replace(FYI_HEADING, FYI_HEADING + "\n" + human + "\n" + signed_by_someone_else)
    notion = FakePages(team_comms=page)

    result = mirrored(refine, notion)

    assert result["targets"]["team_comms_fyi"] == {"action": "posted"}
    assert human in notion.team_comms and signed_by_someone_else in notion.team_comms
    assert len(notion.edits) == 2


@pytest.mark.parametrize(
    ("body", "glue"),
    [
        ("", ""),
        (" \u2014", ""),
        (" none", ""),
        (" Q2 and Q3 on the dataset format", "; "),
        (" Q2 and Q3 on the dataset format.", " "),
        (" (1) first; (2) second;", " "),
    ],
)
def test_the_segment_goes_on_the_end_of_whatever_the_line_already_says(tmp_path, body, glue):
    refine = tree_with_two_open(tmp_path)
    line = "- **Needs a human (Ilan):**" + body
    notion = FakePages(overnight=OVERNIGHT.replace(NEEDS_LINE, line))

    mirrored(refine, notion)

    posts = rulings.build_posts(open_items(refine), NOW)
    kept = body.strip() if body.strip() not in ("", "\u2014", "none") else ""
    assert notion.overnight.split("\n")[-1] == "- **Needs a human (Ilan):** " + (kept + glue if kept else "") + posts.segment


def test_removing_the_segment_leaves_the_other_items_and_a_placeholder_when_it_was_alone(tmp_path):
    refine = tree_with_two_open(tmp_path)
    alone = FakePages(overnight=OVERNIGHT.replace(NEEDS_LINE, "- **Needs a human (Ilan):**"))
    mirrored(refine, alone)
    shared = FakePages()
    mirrored(refine, shared)
    answer_ruling(refine, "telemetry", 1, "10 Hz (owner, 2026-10-10)")
    answer_ruling(refine, "kernels", 4, "the spread (owner, 2026-10-10)")

    mirrored(refine, alone)
    mirrored(refine, shared)

    assert alone.overnight.split("\n")[-1] == "- **Needs a human (Ilan):** none"
    assert shared.overnight.split("\n")[-1] == NEEDS_LINE


def test_a_page_without_the_needs_line_fails_that_target_and_the_other_is_still_done(tmp_path):
    refine = tree_with_two_open(tmp_path)
    notion = FakePages(overnight=OVERNIGHT.replace(NEEDS_LINE, "- **Waiting on:** nothing"))

    result = mirrored(refine, notion)

    assert result["ok"] is False
    assert result["targets"]["overnight_status"]["action"] == "failed"
    assert "found 0" in result["targets"]["overnight_status"]["error"]
    assert result["targets"]["team_comms_fyi"] == {"action": "posted"}


def test_two_needs_lines_for_ilan_are_refused_rather_than_guessed(tmp_path):
    refine = tree_with_two_open(tmp_path)
    notion = FakePages(overnight=OVERNIGHT + "\n" + NEEDS_LINE)

    result = mirrored(refine, notion)

    assert result["targets"]["overnight_status"]["action"] == "failed"
    assert "found 2" in result["targets"]["overnight_status"]["error"]
    assert notion.overnight == OVERNIGHT + "\n" + NEEDS_LINE


def test_a_team_comms_page_without_an_fyi_heading_fails_that_target(tmp_path):
    refine = tree_with_two_open(tmp_path)
    notion = FakePages(team_comms=TEAM_COMMS.replace(FYI_HEADING, "## Notes"))

    result = mirrored(refine, notion)

    assert result["ok"] is False
    assert result["targets"]["team_comms_fyi"]["action"] == "failed"
    assert "FYI" in result["targets"]["team_comms_fyi"]["error"]
    assert result["targets"]["overnight_status"] == {"action": "posted"}


def test_a_write_that_does_not_show_up_on_the_page_is_reported_as_failed(tmp_path):
    class Forgetful(FakePages):
        def replace_text(self, page_id, old, new):
            pass

    refine = tree_with_two_open(tmp_path)

    result = mirrored(refine, Forgetful())

    assert result["ok"] is False
    assert {t["action"] for t in result["targets"].values()} == {"failed"}
    assert all("after the write" in t["error"] for t in result["targets"].values())


def test_a_notion_error_on_one_page_is_reported_for_that_page_only(tmp_path):
    class RefusesOvernight(FakePages):
        def replace_text(self, page_id, old, new):
            if page_id == rulings.OVERNIGHT_PAGE:
                raise rulings.NotionError("Notion answered 403 to PATCH")
            super().replace_text(page_id, old, new)

    refine = tree_with_two_open(tmp_path)
    notion = RefusesOvernight()

    result = mirrored(refine, notion)

    assert result["ok"] is False
    assert result["targets"]["overnight_status"] == {"action": "failed", "error": "Notion answered 403 to PATCH"}
    assert result["targets"]["team_comms_fyi"] == {"action": "posted"}
    assert notion.overnight == OVERNIGHT


# ------------------------------------------------------------------ the mirror verb: print mode, token, REST client


class NotionApi:
    """The page-markdown endpoints as the API reference describes them, at the transport under RestNotionClient."""

    def __init__(self, overnight: str = OVERNIGHT, team_comms: str = TEAM_COMMS) -> None:
        self.pages = {rulings.OVERNIGHT_PAGE: overnight, rulings.TEAM_COMMS_PAGE: team_comms}
        self.calls: list[tuple[str, str, dict, object]] = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), json.loads(body) if body else None))
        match = re.fullmatch(r"https://api\.notion\.com/v1/pages/([0-9a-f-]{36})/markdown", url)
        assert match, f"unexpected url {url}"
        page_id = match.group(1)
        if headers.get("Authorization") != f"Bearer {TOKEN}":
            return 401, {"object": "error", "code": "unauthorized", "message": "API token is invalid."}
        if method == "GET":
            return 200, {"object": "page_markdown", "id": page_id, "markdown": self.pages[page_id], "truncated": False, "unknown_block_ids": []}
        assert method == "PATCH"
        update = json.loads(body)
        assert update["type"] == "update_content"
        for item in update["update_content"]["content_updates"]:
            found = self.pages[page_id].count(item["old_str"])
            if found != 1:
                return 400, {"object": "error", "code": "validation_error", "message": f"old_str matched {found} times"}
            self.pages[page_id] = self.pages[page_id].replace(item["old_str"], item["new_str"])
        return 200, {"object": "page_markdown", "id": page_id, "markdown": self.pages[page_id], "truncated": False, "unknown_block_ids": []}


def forbidden_transport(*args):
    raise AssertionError("the network must not be touched")


def test_print_mode_touches_no_client_and_returns_the_exact_texts_and_where_they_go(tmp_path):
    refine = tree_with_two_open(tmp_path)

    result = run(refine, "mirror", "--print", transport=forbidden_transport, environ={"NOTION_TOKEN": TOKEN}, env_files=[])

    posts = rulings.build_posts(open_items(refine), NOW)
    assert result["ok"] is True and result["mode"] == "print"
    assert (result["unanswered"], result["list_id"]) == (2, posts.list_id)
    overnight, fyi = result["posts"]
    assert (overnight["target"], overnight["action"], overnight["page_url"]) == (
        "overnight_status",
        "add_or_replace",
        "https://app.notion.com/p/3f596ef5fec58172a2d3e0db86fc09a3",
    )
    assert overnight["text"] == posts.segment
    assert (fyi["target"], fyi["action"], fyi["page_url"]) == (
        "team_comms_fyi",
        "add_or_replace",
        "https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7",
    )
    assert fyi["text"] == posts.block
    assert overnight["skip_if_page_shows"] == fyi["skip_if_page_shows"] == f"(list {posts.list_id})"
    assert "Needs a human (Ilan)" in overnight["where"] and "## FYI" in fyi["where"]


def test_print_mode_with_nothing_waiting_only_replaces_earlier_posts(tmp_path):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})

    result = run(refine, "mirror", "--print")

    overnight, fyi = result["posts"]
    assert (overnight["action"], overnight["text"], overnight["skip_if_page_shows"]) == ("remove_if_present", "", "")
    assert fyi["action"] == "replace_if_present"
    assert fyi["text"].endswith(f"(list {result['list_id']})")


def test_the_texts_a_session_would_paste_are_the_ones_the_live_run_posts(tmp_path):
    refine = tree_with_two_open(tmp_path)
    printed = run(refine, "mirror", "--print")
    api = NotionApi()

    run(refine, "mirror", environ={"NOTION_TOKEN": TOKEN}, env_files=[], transport=api)

    overnight, fyi = printed["posts"]
    assert overnight["text"] in api.pages[rulings.OVERNIGHT_PAGE]
    assert fyi["text"] in api.pages[rulings.TEAM_COMMS_PAGE]


def test_without_a_token_nothing_is_posted_and_the_texts_come_back_with_a_failing_answer(tmp_path):
    refine = tree_with_two_open(tmp_path)

    result = run(refine, "mirror", environ={}, env_files=[], transport=forbidden_transport)

    assert result["ok"] is False
    assert result["outcome"] == "no_token"
    assert "NOTION_TOKEN is not set" in result["error"]
    assert result["mode"] == "print"
    assert [p["target"] for p in result["posts"]] == ["overnight_status", "team_comms_fyi"]


def test_a_blank_token_counts_as_missing(tmp_path):
    refine = tree_with_two_open(tmp_path)

    result = run(refine, "mirror", environ={"NOTION_TOKEN": ""}, env_files=[], transport=forbidden_transport)

    assert result["outcome"] == "no_token"


def test_the_token_is_read_from_an_env_file_when_the_environment_has_none(tmp_path):
    refine = tree_with_two_open(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER=1\nNOTION_TOKEN={TOKEN}\n", encoding="utf-8")
    api = NotionApi()

    result = run(refine, "mirror", environ={}, env_files=[env_file], transport=api)

    assert result["ok"] is True
    assert result["targets"] == {"overnight_status": {"action": "posted"}, "team_comms_fyi": {"action": "posted"}}


def test_the_rest_client_speaks_the_page_markdown_api_and_the_mirror_works_through_it(tmp_path):
    refine = tree_with_two_open(tmp_path)
    api = NotionApi()

    result = run(refine, "mirror", environ={"NOTION_TOKEN": TOKEN}, env_files=[], transport=api)

    posts = rulings.build_posts(open_items(refine), NOW)
    assert result["ok"] is True
    assert api.pages[rulings.OVERNIGHT_PAGE] == OVERNIGHT.replace(NEEDS_LINE, NEEDS_LINE + "; " + posts.segment)
    assert api.pages[rulings.TEAM_COMMS_PAGE] == TEAM_COMMS.replace(FYI_HEADING, FYI_HEADING + "\n" + posts.block)
    overnight_url = f"https://api.notion.com/v1/pages/{rulings.OVERNIGHT_PAGE}/markdown"
    assert api.calls[0][:2] == ("GET", overnight_url)
    assert api.calls[0][2] == {
        "Authorization": f"Bearer {TOKEN}",
        "Notion-Version": "2026-03-11",
        "Content-Type": "application/json",
        "User-Agent": "atlas-lab-rulings",
    }
    assert api.calls[0][3] is None
    patch = api.calls[1]
    assert patch[:2] == ("PATCH", overnight_url)
    assert patch[3] == {
        "type": "update_content",
        "update_content": {"content_updates": [{"old_str": NEEDS_LINE, "new_str": NEEDS_LINE + "; " + posts.segment}]},
    }
    assert [c[0] for c in api.calls] == ["GET", "PATCH", "GET", "GET", "PATCH", "GET"]


def test_a_second_run_through_the_rest_client_sends_no_patch(tmp_path):
    refine = tree_with_two_open(tmp_path)
    api = NotionApi()
    run(refine, "mirror", environ={"NOTION_TOKEN": TOKEN}, env_files=[], transport=api)
    api.calls.clear()

    again = run(refine, "mirror", environ={"NOTION_TOKEN": TOKEN}, env_files=[], transport=api)

    assert again["ok"] is True
    assert [c[0] for c in api.calls] == ["GET", "GET"]


def test_the_token_never_reaches_the_output_or_an_error_message(tmp_path):
    refine = tree_with_two_open(tmp_path)

    def leaky(method, url, headers, body, timeout):
        raise ConnectionError(f"could not connect with {headers['Authorization']}")

    def echoing(method, url, headers, body, timeout):
        return 401, {"object": "error", "code": "unauthorized", "message": f"{TOKEN} is not a valid token"}

    for transport in (leaky, echoing):
        result = run(refine, "mirror", environ={"NOTION_TOKEN": TOKEN}, env_files=[], transport=transport)
        assert result["ok"] is False
        assert {t["action"] for t in result["targets"].values()} == {"failed"}
        assert TOKEN not in json.dumps(result)
    assert TOKEN not in repr(rulings.RestNotionClient(TOKEN))


def test_a_page_the_api_returns_in_part_is_refused(tmp_path):
    def partial(method, url, headers, body, timeout):
        return 200, {"object": "page_markdown", "id": "x", "markdown": OVERNIGHT, "truncated": True, "unknown_block_ids": []}

    client = rulings.RestNotionClient(TOKEN, transport=partial)

    with pytest.raises(rulings.NotionError, match="only part of the page"):
        client.read_page(rulings.OVERNIGHT_PAGE)


def test_a_rest_client_needs_a_token():
    with pytest.raises(ValueError, match="token"):
        rulings.RestNotionClient("")


def test_a_track_that_cannot_be_read_stops_the_mirror_before_anything_is_posted(tmp_path):
    refine = tree_with_two_open(tmp_path)
    broken = rulings_path(refine, "learning")
    write(broken, read(broken) + "\n## 7. Half a ruling (pass 1)\n\nNow: x\n")
    api = NotionApi()

    result = run(refine, "mirror", environ={"NOTION_TOKEN": TOKEN}, env_files=[], transport=api)

    assert result["ok"] is False
    assert result["outcome"] == "format_error"
    assert result["errors"][0]["track"] == "learning"
    assert api.calls == []


# ------------------------------------------------------------------ through the dispatcher


def run_verb(capsys, *argv):
    code = dispatcher.main(["rulings", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_the_verb_is_registered_and_listed_in_help(capsys):
    assert dispatcher.VERBS["rulings"] == "lab.machine.rulings:main"
    assert dispatcher.main(["help"]) == 0
    assert re.search(r"^  rulings +Owner rulings: list the open ones", capsys.readouterr().out, flags=re.MULTILINE)


def test_the_dispatcher_runs_list_and_apply_and_answers_with_one_json_line(tmp_path, capsys):
    refine = make_tree(tmp_path, {"telemetry": [T1, T2_ANSWERED]})

    code, listed = run_verb(capsys, "list", "--refine-dir", str(refine))
    assert code == 0
    assert [(r["track"], r["number"]) for r in listed["rulings"]] == [("telemetry", 1)]

    code, applied = run_verb(capsys, "apply", "--refine-dir", str(refine), "--date", "2026-10-10")
    assert code == 0
    assert applied["recorded"] == [{"track": "telemetry", "ruling": 2, "state": "new"}]


def test_the_dispatcher_exits_one_when_mirror_has_no_token(tmp_path, capsys, monkeypatch):
    refine = tree_with_two_open(tmp_path)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.setenv("LAB_ENV_FILES", "")

    code, result = run_verb(capsys, "mirror", "--refine-dir", str(refine))

    assert code == 1
    assert result["outcome"] == "no_token"
    assert result["posts"][1]["text"].startswith(f"- {HEART} Ilan's agent {DOT} ")


def test_the_dispatcher_refuses_the_verb_while_the_stop_file_exists(tmp_path, capsys, state_dir):
    refine = make_tree(tmp_path, {"telemetry": [T2_ANSWERED]})
    state_dir.mkdir(parents=True)
    (state_dir / "STOP").write_text("test", encoding="utf-8")

    code, result = run_verb(capsys, "apply", "--refine-dir", str(refine))

    assert code == 1
    assert result["outcome"] == "stopped"
    assert read(log_path(refine, "telemetry")) == LOG_HEADER


def test_python_dash_m_lab_rulings_runs_from_the_repo_root(tmp_path, state_dir):
    refine = make_tree(tmp_path, {"telemetry": [T1]})

    done = subprocess.run(
        [sys.executable, "-m", "lab", "rulings", "list", "--refine-dir", str(refine)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "LAB_STATE_DIR": str(state_dir)},
        timeout=60,
    )

    assert done.returncode == 0, done.stderr
    assert len(done.stdout.strip().splitlines()) == 1
    assert json.loads(done.stdout)["unanswered"] == 1


def test_bad_arguments_exit_two(tmp_path):
    refine = make_tree(tmp_path, {})

    with pytest.raises(SystemExit) as raised:
        rulings.main(["apply", "--refine-dir", str(refine), "--date", "yesterday"])

    assert raised.value.code == 2


def test_the_digest_in_a_correction_is_the_sha256_prefix_of_the_answer_so_a_rewrite_is_noticed():
    assert rulings.digest("report both (owner, 2026-10-10)") == hashlib.sha256(b"report both (owner, 2026-10-10)").hexdigest()[:8]
    assert rulings.digest("a") != rulings.digest("b")
