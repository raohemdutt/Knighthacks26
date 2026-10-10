"""Tests for the metrics baseline file (lab/refine/metrics.jsonl) and its reader (lab/machine/metrics.py)."""

import datetime as dt
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from lab.machine import metrics

ROOT = Path(__file__).resolve().parents[1]
MODEL = "RedHatAI/Qwen3.8-27B-INT4"


def make_row(**overrides):
    row = {
        "metric": "teacher.decode_tok_s",
        "value": 7.7,
        "unit": "tok/s",
        "node": "pod-a",
        "conditions": {"model": MODEL, "batch": 1},
        "source": "unit test",
        "ts": "2026-10-10T10:20:00Z",
        "low_confidence": False,
    }
    row.update(overrides)
    return row


def write_lines(path, *lines):
    path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))


# --- the seeded baseline -------------------------------------------------------------------------

# (metric, node, conditions that select exactly one row, value, unit, low_confidence)
EXPECTED_SEED = [
    ("teacher.decode_tok_s", "pod-a", {"prompt": "image"}, 6.6, "tok/s", False),
    ("teacher.decode_tok_s", "pod-a", {"prompt": "text", "thinking": "off"}, 7.8, "tok/s", False),
    ("teacher.decode_tok_s", "pod-a", {"prompt": "text", "thinking": "on"}, 7.7, "tok/s", False),
    ("teacher.ttft_s", "pod-a", {"prompt": "image"}, 1.09, "s", False),
    ("teacher.ttft_s", "pod-a", {"prompt": "text", "thinking": "on"}, 0.28, "s", False),
    ("teacher.ttft_s", "pod-a", {"prompt": "text", "thinking": "off"}, 0.55, "s", False),
    ("teacher.watts_avg", "pod-a", {}, 200, "W", False),
    ("teacher.watts_peak", "pod-a", {}, 300, "W", False),
    ("teacher.engine_init_s", "pod-a", {}, 270, "s", False),
    ("teacher.vram_pct", "pod-a", {}, 84, "%", False),
    ("teacher.max_concurrency", "pod-a", {}, 10, "x", False),
    ("teacher.joules_per_token", "pod-a", {}, 25.97, "J/token", True),
    ("trainer.samples_per_s", "pod-b", {}, 0.51, "samples/s", False),
    ("trainer.peak_mem_gb", "pod-b", {}, 8.7, "GiB", False),
    ("trainer.smoke_loss_first", "pod-b", {}, 2.23, "loss", False),
    ("trainer.smoke_loss_last", "pod-b", {}, 0.16, "loss", False),
    ("pod.idle_watts", "pod-a", {}, 39.0, "W", True),
    ("pod.start_s", "pod-a", {}, 15, "s", False),
    ("pod.credits_per_min", "pod-a", {}, 4, "credits/min", False),
    ("pod.credits_per_min", "pod-b", {}, 4, "credits/min", False),
    ("pod.balance_credits", "pod-a", {}, 59960, "credits", True),
    ("pod.balance_credits", "pod-b", {}, 59960, "credits", True),
]

# The numbers the lab setup measured, by metric name.
MEASURED_NAMES = {
    "teacher.decode_tok_s", "teacher.ttft_s", "teacher.watts_avg", "teacher.watts_peak",
    "teacher.engine_init_s", "teacher.vram_pct", "trainer.samples_per_s", "trainer.peak_mem_gb",
    "pod.idle_watts", "pod.start_s", "pod.credits_per_min", "pod.balance_credits",
}


def test_seed_file_has_exactly_the_expected_rows_in_schema_order():
    rows = metrics.load()
    assert len(rows) == len(EXPECTED_SEED)
    for row in rows:
        assert list(row) == list(metrics.FIELDS)


@pytest.mark.parametrize("metric, node, conditions, value, unit, low_confidence", EXPECTED_SEED)
def test_seed_row_carries_the_traced_number(metric, node, conditions, value, unit, low_confidence):
    matches = metrics.history(metric, node=node, conditions=conditions)
    assert len(matches) == 1
    row = matches[0]
    assert row["value"] == value
    assert row["unit"] == unit
    assert row["low_confidence"] is low_confidence
    assert row["source"].strip() != ""


def test_seed_covers_every_number_the_lab_setup_measured():
    assert MEASURED_NAMES <= {row["metric"] for row in metrics.load()}


def test_unfiltered_latest_on_the_seed_is_the_headline_text_prompt():
    assert metrics.latest("teacher.decode_tok_s")["value"] == 7.7
    assert metrics.latest("teacher.ttft_s")["value"] == 0.55


def test_seed_history_keeps_every_condition_of_a_metric_in_file_order():
    assert [row["value"] for row in metrics.history("teacher.decode_tok_s")] == [6.6, 7.8, 7.7]
    assert [row["value"] for row in metrics.history("teacher.ttft_s")] == [1.09, 0.28, 0.55]


def test_derived_joules_per_token_matches_the_inputs_it_names():
    row = metrics.latest("teacher.joules_per_token", node="pod-a")
    assert row["conditions"]["derived"] is True
    inputs = row["conditions"]["inputs"]
    for name, value in inputs.items():
        assert value in [r["value"] for r in metrics.history(name, node="pod-a")], name
    assert row["value"] == round(inputs["teacher.watts_avg"] / inputs["teacher.decode_tok_s"], 2)


def test_every_low_confidence_seed_row_says_why():
    flagged = [row for row in metrics.load() if row["low_confidence"]]
    assert {row["metric"] for row in flagged} == {
        "teacher.joules_per_token", "pod.idle_watts", "pod.balance_credits"}
    for row in flagged:
        reason = row["conditions"].get("low_confidence_reason")
        assert isinstance(reason, str) and reason.strip() != "", row["metric"]


def test_seed_file_is_plain_ascii_with_unix_line_endings():
    data = metrics.DEFAULT_PATH.read_bytes()
    data.decode("ascii")
    assert b"\r" not in data
    assert data.endswith(b"\n")


# --- add -----------------------------------------------------------------------------------------

def test_add_appends_one_canonical_line_and_returns_the_row_as_written(tmp_path):
    path = tmp_path / "m.jsonl"
    row = make_row()
    written = metrics.add(row, path=path)
    assert path.read_bytes() == (
        '{"metric": "teacher.decode_tok_s", "value": 7.7, "unit": "tok/s", "node": "pod-a", '
        '"conditions": {"model": "RedHatAI/Qwen3.8-27B-INT4", "batch": 1}, "source": "unit test", '
        '"ts": "2026-10-10T10:20:00Z", "low_confidence": false}\n'
    ).encode("utf-8")
    assert written == row


def test_add_puts_fields_in_schema_order_whatever_the_caller_order(tmp_path):
    path = tmp_path / "m.jsonl"
    shuffled = dict(reversed(list(make_row().items())))
    metrics.add(shuffled, path=path)
    assert list(json.loads(path.read_text(encoding="utf-8"))) == list(metrics.FIELDS)


def test_add_keeps_earlier_lines_untouched_and_appends_in_order(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1), path=path)
    first = path.read_bytes()
    metrics.add(make_row(value=2), path=path)
    metrics.add(make_row(value=3), path=path)
    assert path.read_bytes().startswith(first)
    assert [row["value"] for row in metrics.load(path)] == [1, 2, 3]


def test_add_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "m.jsonl"
    metrics.add(make_row(), path=path)
    assert len(metrics.load(path)) == 1


def test_add_stamps_the_current_utc_time_when_ts_is_left_out(tmp_path):
    path = tmp_path / "m.jsonl"
    row = make_row()
    del row["ts"]
    before = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    written = metrics.add(row, path=path)
    after = dt.datetime.now(dt.timezone.utc)
    assert written["ts"].endswith("Z")
    stamped = dt.datetime.strptime(written["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    assert before <= stamped <= after
    assert metrics.load(path)[0]["ts"] == written["ts"]


def test_add_accepts_fractional_seconds_and_integer_values(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(ts="2026-10-10T10:20:00.5Z", value=4), path=path)
    row = metrics.load(path)[0]
    assert row["ts"] == "2026-10-10T10:20:00.5Z"
    assert row["value"] == 4 and isinstance(row["value"], int)


BAD_ROWS = [
    ("missing metric", lambda r: r.pop("metric"), "metric"),
    ("missing value", lambda r: r.pop("value"), "value"),
    ("missing conditions", lambda r: r.pop("conditions"), "conditions"),
    ("missing low_confidence", lambda r: r.pop("low_confidence"), "low_confidence"),
    ("unknown field", lambda r: r.update(extra=1), "extra"),
    ("metric without a dot", lambda r: r.update(metric="decode"), "metric"),
    ("metric in capitals", lambda r: r.update(metric="Teacher.decode"), "metric"),
    ("metric with a space", lambda r: r.update(metric="teacher.decode tok"), "metric"),
    ("value is a bool", lambda r: r.update(value=True), "value"),
    ("value is a string", lambda r: r.update(value="7.7"), "value"),
    ("value is nan", lambda r: r.update(value=float("nan")), "value"),
    ("value is infinite", lambda r: r.update(value=float("inf")), "value"),
    ("unit is empty", lambda r: r.update(unit=""), "unit"),
    ("unit is not text", lambda r: r.update(unit=3), "unit"),
    ("node in capitals", lambda r: r.update(node="Pod-A"), "node"),
    ("node is empty", lambda r: r.update(node=""), "node"),
    ("conditions is a list", lambda r: r.update(conditions=[1]), "conditions"),
    ("conditions not json", lambda r: r.update(conditions={"x": object()}), "conditions"),
    ("source is blank", lambda r: r.update(source="   "), "source"),
    ("low_confidence is an int", lambda r: r.update(low_confidence=0), "low_confidence"),
    ("ts without the Z", lambda r: r.update(ts="2026-10-10T10:20:00"), "ts"),
    ("ts with an offset", lambda r: r.update(ts="2026-10-10T10:20:00+00:00"), "ts"),
    ("ts date only", lambda r: r.update(ts="2026-10-10"), "ts"),
    ("ts impossible date", lambda r: r.update(ts="2026-13-40T10:20:00Z"), "ts"),
]


@pytest.mark.parametrize("label, damage, field", BAD_ROWS, ids=[case[0] for case in BAD_ROWS])
def test_add_rejects_a_bad_row_and_leaves_the_file_as_it_was(tmp_path, label, damage, field):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(), path=path)
    before = path.read_bytes()
    row = make_row()
    damage(row)
    with pytest.raises(metrics.MetricsFormatError, match=field):
        metrics.add(row, path=path)
    assert path.read_bytes() == before


def test_add_refuses_to_append_after_a_line_that_was_cut_short(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_bytes(json.dumps(make_row()).encode("utf-8"))  # no trailing newline
    before = path.read_bytes()
    with pytest.raises(metrics.MetricsFormatError, match="newline"):
        metrics.add(make_row(value=9), path=path)
    assert path.read_bytes() == before


# --- load, latest, history -----------------------------------------------------------------------

def test_load_of_a_missing_file_is_empty_and_creates_nothing(tmp_path):
    path = tmp_path / "absent.jsonl"
    assert metrics.load(path) == []
    assert not path.exists()


def test_load_skips_blank_lines_and_reads_crlf_endings(tmp_path):
    path = tmp_path / "m.jsonl"
    first, second = json.dumps(make_row(value=1)), json.dumps(make_row(value=2))
    path.write_bytes((first + "\r\n\r\n" + second + "\r\n").encode("utf-8"))
    assert [row["value"] for row in metrics.load(path)] == [1, 2]


def test_load_refuses_a_corrupt_line_and_names_it(tmp_path):
    path = tmp_path / "m.jsonl"
    write_lines(path, json.dumps(make_row()), '{"metric": "teacher.decode_tok_s", "value": ', json.dumps(make_row()))
    with pytest.raises(metrics.MetricsFormatError, match=r"line 2"):
        metrics.load(path)


def test_load_refuses_bytes_that_are_not_utf8(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_bytes(json.dumps(make_row()).encode("utf-8") + b"\n\xff\xfe\n")
    with pytest.raises(metrics.MetricsFormatError, match=r"line 2"):
        metrics.load(path)


def test_load_refuses_a_row_that_breaks_the_schema_and_names_line_and_field(tmp_path):
    path = tmp_path / "m.jsonl"
    write_lines(path, json.dumps(make_row()), json.dumps(make_row(value=True)))
    with pytest.raises(metrics.MetricsFormatError, match=r"line 2.*value"):
        metrics.load(path)


def test_load_refuses_a_row_with_an_unknown_field(tmp_path):
    path = tmp_path / "m.jsonl"
    write_lines(path, json.dumps({**make_row(), "seq": 1}))
    with pytest.raises(metrics.MetricsFormatError, match=r"line 1.*seq"):
        metrics.load(path)


def test_load_requires_ts_on_every_stored_row(tmp_path):
    path = tmp_path / "m.jsonl"
    row = make_row()
    del row["ts"]
    write_lines(path, json.dumps(row))
    with pytest.raises(metrics.MetricsFormatError, match=r"line 1.*ts"):
        metrics.load(path)


def test_latest_is_the_newest_timestamp_not_the_last_line(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=2, ts="2026-10-10T12:00:00Z"), path=path)
    metrics.add(make_row(value=1, ts="2026-10-10T09:00:00Z"), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 2


def test_latest_orders_fractional_seconds_correctly(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=2, ts="2026-10-10T10:20:00.5Z"), path=path)
    metrics.add(make_row(value=1, ts="2026-10-10T10:20:00Z"), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 2


def test_latest_breaks_a_timestamp_tie_with_the_later_line(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1), path=path)
    metrics.add(make_row(value=2), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 2


def test_latest_filters_by_node_and_by_condition_subset(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1, node="pod-a", conditions={"prompt": "text", "batch": 1}), path=path)
    metrics.add(make_row(value=2, node="pod-a", conditions={"prompt": "image", "batch": 1}), path=path)
    metrics.add(make_row(value=3, node="pod-b", conditions={"prompt": "text", "batch": 4}), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 3
    assert metrics.latest("teacher.decode_tok_s", node="pod-a", path=path)["value"] == 2
    assert metrics.latest("teacher.decode_tok_s", node="pod-a", conditions={"prompt": "text"}, path=path)["value"] == 1
    assert metrics.latest("teacher.decode_tok_s", conditions={"batch": 4}, path=path)["value"] == 3
    assert metrics.latest("teacher.decode_tok_s", node="pod-b", conditions={"prompt": "image"}, path=path) is None
    assert metrics.latest("teacher.decode_tok_s", node="pod-c", path=path) is None


def test_latest_of_an_unknown_metric_is_none_and_its_history_is_empty(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(), path=path)
    assert metrics.latest("teacher.ttft_s", path=path) is None
    assert metrics.history("teacher.ttft_s", path=path) == []


def test_history_is_oldest_first_for_one_metric_only(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=3, ts="2026-10-10T12:00:00Z"), path=path)
    metrics.add(make_row(metric="teacher.ttft_s", unit="s", value=0.5), path=path)
    metrics.add(make_row(value=1, ts="2026-10-10T08:00:00Z"), path=path)
    metrics.add(make_row(value=2, ts="2026-10-10T10:00:00Z"), path=path)
    assert [row["value"] for row in metrics.history("teacher.decode_tok_s", path=path)] == [1, 2, 3]


def test_history_filters_by_node(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1, node="pod-a"), path=path)
    metrics.add(make_row(value=2, node="pod-b", ts="2026-10-10T11:00:00Z"), path=path)
    assert [row["value"] for row in metrics.history("teacher.decode_tok_s", node="pod-b", path=path)] == [2]


def test_a_row_added_after_a_read_shows_up_on_the_next_read(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics.add(make_row(value=1), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 1
    metrics.add(make_row(value=2, ts="2026-10-10T11:00:00Z"), path=path)
    assert metrics.latest("teacher.decode_tok_s", path=path)["value"] == 2


# --- the file lock, across real processes --------------------------------------------------------

HOLD_THE_LOCK = """
import sys
from lab.machine import metrics

lock = metrics.FileLock(sys.argv[1])
if not lock.acquire(timeout=10):
    sys.exit("child could not take the lock")
print("locked", flush=True)
sys.stdin.read()
lock.release()
"""

WRITE_AFTER_GO = """
import os
import sys
import time
from lab.machine import metrics

path, writer, count, go = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
print("ready", flush=True)
while not os.path.exists(go):
    time.sleep(0.001)
for i in range(count):
    metrics.add({"metric": "test.counter", "value": i, "unit": "n", "node": "writer-" + writer,
                 "conditions": {"writer": writer, "i": i}, "source": "concurrency test",
                 "ts": "2026-10-10T10:20:00Z", "low_confidence": False}, path=path)
"""


def start(code, *args, errors, stdin=None):
    """Run `code` in a child python from the repo root; its stderr goes to the open file `errors`, so a child that
    writes a lot can never block on a full pipe."""
    return subprocess.Popen(
        [sys.executable, "-c", code, *map(str, args)], cwd=ROOT, text=True,
        stdin=stdin, stdout=subprocess.PIPE, stderr=errors)


def test_add_and_load_give_up_with_metrics_locked_while_another_process_holds_the_lock(tmp_path):
    path = tmp_path / "m.jsonl"
    errors_path = tmp_path / "children.err"
    with open(errors_path, "w") as errors:
        holder = start(HOLD_THE_LOCK, path, errors=errors, stdin=subprocess.PIPE)
        try:
            assert holder.stdout.readline().strip() == "locked", errors_path.read_text()
            with pytest.raises(metrics.MetricsLocked):
                metrics.add(make_row(), path=path, lock_timeout=0.3)
            with pytest.raises(metrics.MetricsLocked):
                metrics.load(path, lock_timeout=0.3)
            holder.stdin.close()
            assert holder.wait(timeout=30) == 0, errors_path.read_text()
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
    assert path.read_bytes() == b""
    metrics.add(make_row(), path=path)
    assert len(metrics.load(path)) == 1


def test_add_waits_for_the_lock_instead_of_failing_when_it_frees_in_time(tmp_path):
    path = tmp_path / "m.jsonl"
    errors_path = tmp_path / "children.err"
    outcome = {}

    def writer():
        try:
            outcome["row"] = metrics.add(make_row(), path=path, lock_timeout=30)
        except BaseException as exc:  # handed to the assertion below
            outcome["error"] = exc

    with open(errors_path, "w") as errors:
        holder = start(HOLD_THE_LOCK, path, errors=errors, stdin=subprocess.PIPE)
        try:
            assert holder.stdout.readline().strip() == "locked", errors_path.read_text()
            thread = threading.Thread(target=writer)
            thread.start()
            thread.join(timeout=0.5)
            assert thread.is_alive(), "add returned while another process still held the lock"
            holder.stdin.close()  # the holder lets go; the waiting add must now finish
            thread.join(timeout=30)
            assert not thread.is_alive()
            assert holder.wait(timeout=30) == 0, errors_path.read_text()
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
    assert "error" not in outcome, repr(outcome.get("error"))
    assert [row["metric"] for row in metrics.load(path)] == ["teacher.decode_tok_s"]
    assert outcome["row"] == make_row()


def test_concurrent_writers_lose_no_rows_and_never_interleave(tmp_path):
    path = tmp_path / "m.jsonl"
    go = tmp_path / "go"
    errors_path = tmp_path / "children.err"
    writers, per_writer = 4, 40
    with open(errors_path, "w") as errors:
        children = [start(WRITE_AFTER_GO, path, name, per_writer, go, errors=errors) for name in "abcd"]
        try:
            for child in children:
                assert child.stdout.readline().strip() == "ready", errors_path.read_text()
            go.write_text("go")
            for child in children:
                assert child.wait(timeout=120) == 0, errors_path.read_text()
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
    rows = metrics.load(path)
    assert len(rows) == writers * per_writer
    assert path.read_bytes().count(b"\n") == writers * per_writer
    assert {(row["conditions"]["writer"], row["conditions"]["i"]) for row in rows} == {
        (name, i) for name in "abcd" for i in range(per_writer)}
