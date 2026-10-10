"""lab/train.py, the parts that need no training stack: which rows a run may train on, the labels on the spans, the
base model's commit, the pod sampler's power per step, the node check, deleting the data, the pinned install, and
the entry point's refusals and ledger row.

The committed fixture fixtures/learning/train-tiny/ is a slice of the no-camera tool-calling dataset with its frozen
split (six labeled train examples, the whole held-out side) and the rows lab/format_traces.py --split wrote for it.
Hardware boundaries are replaced: torch by a small module object where the node check reads the GPU, pip by a
recorder, the sampler by sample files written here, and the tokenizer by small ones whose behaviour at a span's edge
is known. The run that trains is in tests/test_lab_train_cpu.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from contracts.events import parse_event
from contracts.telemetry import TrainStep
from lab import dataset
from lab import format_traces as ft
from lab import train
from lab.machine import ledger, stop

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY = REPO_ROOT / "fixtures" / "learning" / "train-tiny"
TINY_DATASET = TINY / "tool-calling-tiny"
TINY_ROWS = TINY / "rows.jsonl"
SCRIPT = REPO_ROOT / "lab" / "train.py"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "89abcdef0123456789abcdef0123456789abcdef"
T0 = datetime(2026, 10, 10, 20, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "opus-test")
    monkeypatch.setenv("LAB_TIER", "opus")
    monkeypatch.delenv("LAB_TASK", raising=False)


def fixture_rows() -> list[dict]:
    return [json.loads(line) for line in TINY_ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A copy of the tiny dataset (its folder named for its dataset_id) and of its rows, as a pod would hold them."""
    folder = tmp_path / "datasets" / TINY_DATASET.name
    shutil.copytree(TINY_DATASET, folder)
    rows = tmp_path / "datasets" / "rows.jsonl"
    shutil.copyfile(TINY_ROWS, rows)
    return folder, rows


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def only_row(verb: str = train.VERB) -> dict:
    rows = [row for row in ledger.read() if row["verb"] == verb]
    assert len(rows) == 1, rows
    return rows[0]


def model_folder(tmp_path: Path, commits: list[str] | None = None) -> Path:
    """A model folder the way `hf download --local-dir` leaves it: the files, and one metadata file per file whose
    first line is the commit."""
    folder = tmp_path / "models" / "Qwen3.5-4B"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    metadata = folder / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    for number, commit in enumerate(commits or []):
        (metadata / f"file{number}.json.metadata").write_text(f"{commit}\n{'e' * 40}\n1760000000.0\n",
                                                              encoding="utf-8")
    return folder


# --- the committed fixture ----------------------------------------------------------------------

def test_the_fixture_is_a_frozen_split_that_meets_the_deploy_floor_with_six_examples_to_train_on():
    split = dataset.load_split(TINY_DATASET, require_images=False)
    assert split.dataset_id == TINY_DATASET.name == "tool-calling-tiny"
    assert [example.id for example in split.train] == ["v-a640fd7e", "v-ec128c61", "v-2792e1c5", "v-7d092ba9",
                                                      "v-ca9695a2", "v-e4e03fd7"]
    assert len(split.held_out) == 58
    assert split.meets_deploy_floor
    assert all(not example.target for example in split.held_out)
    assert all(example.target for example in split.train)


def test_the_fixture_rows_are_what_format_traces_writes_for_the_train_side(tmp_path):
    done = ft.format_dataset(TINY_DATASET, ft.fixture_tool_results(), tmp_path / "rows.jsonl", split=True)
    assert (len(done.rows), len(done.rejected), len(done.failed), done.held_out) == (6, 0, 0, 58)
    assert (tmp_path / "rows.jsonl").read_bytes() == TINY_ROWS.read_bytes().replace(b"\r\n", b"\n")


# --- the rows ---------------------------------------------------------------------------------------

def test_read_rows_takes_the_formatter_rows_in_file_order_with_their_spans():
    rows = train.read_rows(TINY_ROWS)
    raw = fixture_rows()
    assert [row.id for row in rows] == [item["id"] for item in raw]
    assert [row.spans for row in rows] == [tuple(tuple(span) for span in item["spans"]) for item in raw]
    assert [row.frame_ref for row in rows] == [item["frame_ref"] for item in raw]
    assert all(row.images == () and row.text == item["text"] for row, item in zip(rows, raw))


def test_read_rows_names_every_bad_line_by_line_and_id_and_never_quotes_the_text(tmp_path):
    good = fixture_rows()[0]
    secret = good["text"]
    lines = [
        "not json",
        json.dumps({**good, "id": ""}),
        json.dumps({**good, "id": "late", "spans": [[good["spans"][1][0], good["spans"][1][1]], good["spans"][0]]}),
        json.dumps({**good, "id": "past-end", "spans": [[0, len(secret) + 1]]}),
        json.dumps({**good, "id": "no-spans", "spans": []}),
        json.dumps({**good, "id": "climbs", "images": ["../outside.jpg"]}),
        json.dumps({**good, "id": "twice"}),
        json.dumps({**good, "id": "twice"}),
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(train.TrainError) as caught:
        train.read_rows(path)
    assert caught.value.problems == [
        "rows.jsonl line 1: not JSON",
        "rows.jsonl line 2 (id ''): has no id",
        "rows.jsonl line 3 (id 'late'): span 2 is not inside the text after the span before it",
        "rows.jsonl line 4 (id 'past-end'): span 1 is not inside the text after the span before it",
        "rows.jsonl line 5 (id 'no-spans'): has no spans",
        "rows.jsonl line 6 (id 'climbs'): images is not a list of paths inside the dataset folder",
        "rows.jsonl line 8 (id 'twice'): the id repeats line 7",
    ]
    assert secret[:40] not in str(caught.value)


def test_an_empty_or_missing_rows_file_is_refused(tmp_path):
    (tmp_path / "empty.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(train.TrainError) as empty:
        train.read_rows(tmp_path / "empty.jsonl")
    with pytest.raises(train.TrainError) as missing:
        train.read_rows(tmp_path / "missing.jsonl")
    assert (empty.value.problems, missing.value.problems) == (["empty.jsonl: has no rows"],
                                                               ["missing.jsonl: missing"])


def test_the_split_decides_membership_and_every_train_row_passes():
    split = dataset.load_split(TINY_DATASET, require_images=False)
    rows = train.read_rows(TINY_ROWS)
    chosen, without_row = train.train_rows(rows, split)
    assert chosen == rows
    assert without_row == []
    fewer, missing = train.train_rows(rows[2:], split)
    assert (fewer, missing) == (rows[2:], ["v-a640fd7e", "v-ec128c61"])


def test_a_held_out_row_an_unknown_row_or_a_row_from_another_frame_refuses_the_run():
    split = dataset.load_split(TINY_DATASET, require_images=False)
    rows = train.read_rows(TINY_ROWS)
    held = split.held_out[0]
    leaks = [train.Row(held.id, held.frame_ref, rows[0].text, rows[0].spans),
             train.Row("not-there", None, rows[0].text, rows[0].spans),
             train.Row(rows[1].id, "fixture:group:somewhere-else", rows[1].text, rows[1].spans),
             train.Row(rows[2].id, rows[2].frame_ref, rows[2].text, rows[2].spans, ("images/0001.jpg",))]
    with pytest.raises(train.TrainError) as caught:
        train.train_rows([*rows[3:], *leaks], split)
    assert caught.value.problems == [
        f"row {held.id!r}: held out for the eval gate; no training run may see it",
        "row 'not-there': the dataset has no such example",
        f"row {rows[1].id!r}: its frame_ref is not its example's",
        f"row {rows[2].id!r}: its images are not its example's own image",
    ]


def test_the_examples_hash_reads_line_endings_as_lf(tmp_path):
    (tmp_path / "lf.jsonl").write_bytes(b'{"a":1}\n{"b":2}\n')
    (tmp_path / "crlf.jsonl").write_bytes(b'{"a":1}\r\n{"b":2}\r\n')
    assert train.file_sha256(tmp_path / "crlf.jsonl") == hashlib.sha256(b'{"a":1}\n{"b":2}\n').hexdigest()
    assert train.file_sha256(tmp_path / "lf.jsonl") == train.file_sha256(tmp_path / "crlf.jsonl")


# --- labels on the spans ----------------------------------------------------------------------------

class ByteTokenizer:
    """Every byte is a token and each special token of the template is one: text comes back exactly, and every
    character boundary is a token boundary."""

    SPECIAL = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
               "</tool_response>", "<|image_pad|>")

    def __init__(self) -> None:
        self.ids = {name: 256 + number for number, name in enumerate(self.SPECIAL)}
        self.names = {number: name for name, number in self.ids.items()}
        self.split = re.compile("(" + "|".join(re.escape(name) for name in self.SPECIAL) + ")")

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for part in self.split.split(text):
            out.extend([self.ids[part]] if part in self.ids else part.encode("utf-8"))
        return out

    def decode(self, ids) -> str:
        pieces, pending = [], bytearray()
        for number in ids:
            if number < 256:
                pending.append(number)
            else:
                pieces += [pending.decode("utf-8"), self.names[number]]
                pending = bytearray()
        return "".join(pieces + [pending.decode("utf-8")])


class MergingTokenizer(ByteTokenizer):
    """Like ByteTokenizer, but a blank line followed by a letter is one token: a boundary between them is not a
    token boundary."""

    MERGED = 999

    def encode(self, text: str) -> list[int]:
        ids = super().encode(text)
        out, i = [], 0
        while i < len(ids):
            if ids[i:i + 2] == [10, 10] and i + 2 < len(ids) and chr(ids[i + 2]).isalpha():
                out.append(self.MERGED)
                i += 3
                continue
            out.append(ids[i])
            i += 1
        return out


def test_only_the_tokens_of_the_spans_carry_the_loss_and_they_are_exactly_the_span_text():
    tokenizer = ByteTokenizer()
    for item in fixture_rows():
        encoded = train.span_labels(tokenizer.encode, item["text"], [tuple(s) for s in item["spans"]],
                                    tokenizer.decode)
        assert list(encoded.ids) == tokenizer.encode(item["text"])
        trained = [label for label in encoded.labels if label != train.IGNORE]
        assert tokenizer.decode(trained) == "".join(item["text"][start:end] for start, end in item["spans"])
        assert encoded.trained == len(trained)
        first = tokenizer.encode(item["text"][:item["spans"][0][0]])
        assert set(encoded.labels[:len(first)]) == {train.IGNORE}
        assert encoded.labels[len(first)] == tokenizer.ids["<tool_call>"]


def test_a_span_that_starts_or_ends_inside_a_token_or_a_text_that_does_not_come_back_is_refused():
    merging = MergingTokenizer()
    text = "<|im_start|>assistant\n<think>\n\n</think>\n\nKeys on the desk.<|im_end|>\n"
    start = text.index("Keys")
    with pytest.raises(ValueError, match="span 1 starts inside a token"):
        train.span_labels(merging.encode, text, [(start, text.index("<|im_end|>") + 10)])
    plain = ByteTokenizer()
    with pytest.raises(ValueError, match="span 1 ends inside a token"):
        train.span_labels(lambda t: merging.encode(t), text, [(start - 2, start - 1)])
    with pytest.raises(ValueError, match="does not give the text back"):
        train.span_labels(plain.encode, text, [(start, start + 4)], lambda ids: plain.decode(ids).lower())
    with pytest.raises(ValueError, match="holds no token"):   # a normalizer that drops a zero-width space
        train.span_labels(lambda t: plain.encode(t.replace("​", "")), "ab​cd", [(2, 3)])


def test_image_labels_follow_the_processors_expansion_and_image_tokens_never_carry_loss():
    pad = 264
    ids = [1, 2, pad, 3, 4, 5]
    labels = [train.IGNORE, train.IGNORE, train.IGNORE, train.IGNORE, 4, 5]
    expanded = [1, 2, pad, pad, pad, 3, 4, 5]
    assert train.expand_image_labels(ids, labels, expanded, pad) == \
        [train.IGNORE, train.IGNORE, train.IGNORE, train.IGNORE, train.IGNORE, train.IGNORE, 4, 5]
    with pytest.raises(ValueError, match="do not line up"):
        train.expand_image_labels(ids, labels, [1, 2, pad, 3, 9, 5], pad)
    with pytest.raises(ValueError, match="no image tokens"):
        train.expand_image_labels(ids, labels, [1, 2, 3, 4, 5], pad)
    with pytest.raises(ValueError, match="do not line up"):
        train.expand_image_labels(ids, labels, [*expanded, 6], pad)
    with pytest.raises(ValueError, match="inside a span"):
        train.expand_image_labels(ids, [1, 2, pad, 3, 4, 5], expanded, pad)


# --- the base model's commit -------------------------------------------------------------------------

def test_a_model_folder_records_its_hub_commit_and_the_run_takes_it(tmp_path):
    folder = model_folder(tmp_path, [COMMIT, COMMIT])
    assert train.resolve_base(str(folder)) == train.Base(str(folder), "Qwen/Qwen3.5-4B", COMMIT)
    assert train.resolve_base(str(folder), revision=COMMIT).revision == COMMIT


@pytest.mark.parametrize("commits, revision, reason", [
    ([], None, "the base model's Hub commit is unknown"),
    ([COMMIT, OTHER_COMMIT], None, "files from 2 Hub commits"),
    ([COMMIT], OTHER_COMMIT, "is not the commit the model folder was downloaded at"),
    ([], "main", "must be a 40-character Hub commit hash"),
])
def test_a_base_without_one_known_commit_refuses_the_run(tmp_path, commits, revision, reason):
    folder = model_folder(tmp_path, commits)
    with pytest.raises(train.TrainError) as caught:
        train.resolve_base(str(folder), revision=revision)
    assert caught.value.outcome == "refused"
    assert reason in caught.value.problems[0]


def test_a_folder_without_metadata_takes_the_commit_it_is_given(tmp_path):
    folder = model_folder(tmp_path, [])
    assert train.resolve_base(str(folder), "Qwen/Qwen3.5-4B", COMMIT).revision == COMMIT


# --- the pod sampler's power, per step ------------------------------------------------------------------

def sample(node: str, at: datetime, watts: float | None, **more) -> str:
    event = {"event_id": f"s{at.timestamp()}", "type": "telemetry.gpu.power", "version": 1,
             "ts": at.isoformat().replace("+00:00", "Z"), "node": node, "source": "amd-smi", "period_ms": 1000.0,
             "socket_w": watts, "gfx_mhz": None, "mem_mhz": None, "throttled": None, "power_field": "socket_power",
             "sampler": "lab/pod_power.py 1", **more}
    return json.dumps(event)


def test_the_feed_averages_the_samplers_readings_inside_the_step_and_keeps_where_they_came_from(tmp_path):
    lines = [sample("gpu-b", T0 + timedelta(seconds=s), w) for s, w in ((0, 100.0), (1, 110.0), (2, 120.0), (5, 999.0))]
    lines += [sample("gpu-a", T0 + timedelta(seconds=1), 500.0), "not json",
              json.dumps({"type": "telemetry.request", "node": "gpu-b", "ts": T0.isoformat()}),
              sample("gpu-b", T0 + timedelta(seconds=1, milliseconds=500), None)]
    (tmp_path / "samples-2026-10-10.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    feed = train.PowerFeed(tmp_path, "gpu-b")
    window = feed.window(T0 + timedelta(milliseconds=500), T0 + timedelta(seconds=2))
    assert window == train.PowerWindow(115.0, 2, "lab/pod_power.py 1", "amd-smi", "socket_power", 1000.0)
    assert feed.bad_lines == 1
    assert feed.window(T0 + timedelta(seconds=3), T0 + timedelta(seconds=4)) == train.PowerWindow(None, 0)


def test_the_feed_reads_only_whole_lines_and_picks_up_what_the_sampler_writes_later(tmp_path):
    late = datetime(2026, 10, 10, 23, 59, 58, tzinfo=timezone.utc)
    path = tmp_path / "samples-2026-10-10.jsonl"
    first = sample("gpu-b", late, 80.0)
    second = sample("gpu-b", late + timedelta(seconds=1), 90.0)
    path.write_text(first + "\n" + second[:30], encoding="utf-8")
    feed = train.PowerFeed(tmp_path, "gpu-b")
    end = late + timedelta(seconds=3)
    assert feed.window(late, end).samples == 1
    with path.open("a", encoding="utf-8") as out:
        out.write(second[30:] + "\n")
    (tmp_path / "samples-2026-10-11.jsonl").write_text(sample("gpu-b", end, 100.0) + "\n", encoding="utf-8")
    assert feed.window(late, end) == train.PowerWindow(90.0, 3, "lab/pod_power.py 1", "amd-smi", "socket_power",
                                                      1000.0)
    assert feed.bad_lines == 0


def test_the_feed_reads_only_the_files_of_the_days_a_step_spans(tmp_path):
    (tmp_path / "samples-2026-10-01.jsonl").write_text("not json\n" + sample("gpu-b", T0, 999.0) + "\n",
                                                      encoding="utf-8")
    (tmp_path / "samples-2026-10-10.jsonl").write_text(sample("gpu-b", T0, 50.0) + "\n", encoding="utf-8")
    feed = train.PowerFeed(tmp_path, "gpu-b")
    assert feed.window(T0, T0 + timedelta(seconds=1)).watts == 50.0
    assert (feed.bad_lines, len(feed.samples)) == (0, 1)


def test_a_missing_sampler_folder_gives_no_reading(tmp_path):
    assert train.PowerFeed(tmp_path / "nothing", "gpu-b").window(T0, T0 + timedelta(hours=1)) == train.PowerWindow(None, 0)


# --- one telemetry.train.step event per step ---------------------------------------------------------------

class Clock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_each_logged_step_appends_one_train_step_event_with_its_loss_time_tokens_and_watts(tmp_path):
    (tmp_path / "power").mkdir()
    (tmp_path / "power" / "samples-2026-10-10.jsonl").write_text(
        "\n".join(sample("gpu-b", T0 + timedelta(seconds=s), 100.0 + s) for s in range(0, 6)) + "\n", encoding="utf-8")
    wall, timer = Clock(T0.timestamp()), Clock(50.0)
    steps = train.StepLog(tmp_path / "telemetry.jsonl", "run-7", "gpu-b", train.PowerFeed(tmp_path / "power"),
                          clock=wall, timer=timer)
    steps.begin()
    wall.now, timer.now = wall.now + 2.5, timer.now + 2.5
    steps.end(1)
    assert steps.logged(1, {"loss": 1.25, "num_tokens": 300.0, "learning_rate": 2e-4}) is not None
    steps.begin()
    wall.now, timer.now = wall.now + 2.0, timer.now + 2.0
    steps.end(2)
    steps.logged(2, {"loss": float("nan"), "num_tokens": 520.0})
    assert steps.logged(2, {"train_runtime": 4.5, "train_loss": 1.0}) is None
    lines = (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    events = [parse_event(line) for line in lines]
    assert [type(event) for event in events] == [TrainStep, TrainStep]
    first, second = events
    assert (first.node, first.run_id, first.step, first.loss, first.step_ms, first.tokens) == \
        ("gpu-b", "run-7", 1, 1.25, 2500.0, 300)
    assert first.socket_w == pytest.approx(101.0)   # the readings at 0, 1 and 2 s of a step from 0 to 2.5 s
    assert first.ts == T0 + timedelta(seconds=2.5)
    assert (second.step, second.loss, second.step_ms, second.tokens) == (2, None, 2000.0, 220)
    assert second.socket_w == pytest.approx(103.5)  # 3 and 4 s, inside 2.5 to 4.5 s
    assert steps.tokens_total == 520
    assert [window.samples for window in steps.windows] == [3, 2]


def test_without_a_feed_the_step_has_no_watts(tmp_path):
    steps = train.StepLog(tmp_path / "telemetry.jsonl", "r", "cpu", None, clock=Clock(T0.timestamp()), timer=Clock(0))
    steps.begin()
    steps.end(1)
    event = steps.logged(1, {"loss": 2.0})
    assert (event["socket_w"], event["tokens"], event["node"]) == (None, None, "cpu")


# --- where it runs --------------------------------------------------------------------------------------

def fake_torch(hip: str | None = "7.13.0", gpu: bool = True, arch: str = "gfx1151") -> ModuleType:
    module = ModuleType("torch")
    module.version = SimpleNamespace(hip=hip)
    module.cuda = SimpleNamespace(is_available=lambda: gpu,
                                  get_device_properties=lambda index: SimpleNamespace(gcnArchName=arch))
    return module


def test_the_node_check_admits_only_a_rocm_torch_on_the_expected_chip():
    assert train.pod_gate(fake_torch()) is None
    assert train.pod_gate(fake_torch(arch="gfx1151:xnack-")) is None
    assert "not a ROCm build" in train.pod_gate(fake_torch(hip=None))
    assert "no GPU is visible" in train.pod_gate(fake_torch(gpu=False))
    assert train.pod_gate(fake_torch(arch="gfx1201")) == \
        "the GPU is gfx1201, not gfx1151: this is not the trainer pod"
    assert train.pod_gate(fake_torch(arch="gfx1201"), gfx="gfx1201") is None


def test_without_an_arch_from_torch_the_node_check_reads_rocminfo():
    assert train.pod_gate(fake_torch(arch=""), rocminfo=lambda: "  Name:  gfx1151\n") is None
    assert "not gfx1151" in train.pod_gate(fake_torch(arch=""), rocminfo=lambda: "  Name:  gfx1201\n")
    assert "unknown kind" in train.pod_gate(fake_torch(arch=""))


# --- deleting the data --------------------------------------------------------------------------------------

def test_the_dataset_copy_and_the_rows_are_deleted(tmp_path):
    folder, rows = copy_fixture(tmp_path)
    assert train.delete_data(folder, rows) == (True, "deleted")
    assert not folder.exists() and not rows.exists()
    assert (tmp_path / "datasets").is_dir()


def test_a_folder_that_is_not_a_dataset_export_is_never_deleted(tmp_path):
    keep = tmp_path / "projects"
    keep.mkdir()
    (keep / "notes.txt").write_text("mine", encoding="utf-8")
    rows = write_rows(tmp_path / "rows.jsonl", [])
    deleted, why = train.delete_data(keep, rows)
    assert (deleted, why) == (False, "the dataset path does not look like a dataset export, so it was left")
    assert (keep / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert not rows.exists()


def test_the_home_folder_is_never_deleted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shutil.copytree(TINY_DATASET, home)   # even one that looks like an export
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    deleted, why = train.delete_data(home, tmp_path / "rows.jsonl")
    assert (deleted, why) == (False, "the dataset path is the home folder or above it")
    assert (home / "manifest.json").is_file()


def test_a_checkout_and_a_rows_folder_are_left(tmp_path):
    folder, rows = copy_fixture(tmp_path)
    (folder / ".git").mkdir()
    rows.unlink()
    rows.mkdir()
    deleted, why = train.delete_data(folder, rows)
    assert not deleted
    assert why == ("the dataset path does not look like a dataset export, so it was left; "
                   "the rows path is a folder, so it was left")
    assert folder.is_dir() and rows.is_dir()


# --- the pinned requirements ----------------------------------------------------------------------------------

def test_the_requirements_pin_peft_and_trl_exactly_and_never_touch_the_pods_torch():
    pins = train.read_pins()
    assert {"peft", "trl", "transformers", "accelerate", "datasets", "pydantic", "pillow"} <= set(pins)
    assert all(re.fullmatch(r"\d+(\.\d+)+", version) for version in pins.values())
    assert not set(pins) & {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm"}


def test_only_missing_or_different_pins_are_installed():
    installed = {"peft": "0.21.2", "trl": "1.14.0"}
    assert train.pins_to_install({"peft": "0.21.2", "trl": "1.15.0", "datasets": "5.1.0"}, installed.get) == \
        ["trl==1.15.0", "datasets==5.1.0"]


def test_the_install_pins_the_torch_the_machine_has_and_writes_its_row(tmp_path):
    requirements = tmp_path / "requirements-train.txt"
    requirements.write_text("# pins\npeft==0.21.2\ntrl==1.15.0  # the trainer\n", encoding="utf-8")
    installed = {"torch": "2.9.1+rocm7.13.0", "triton": "3.5.1+rocm7.13.0", "peft": "0.21.2"}
    seen = {}

    def pip(argv):
        seen["argv"] = argv
        seen["constraints"] = Path(argv[argv.index("-c") + 1]).read_text(encoding="utf-8")
        installed["trl"] = "1.15.0"
        return 0

    answer = train.install(requirements, run=pip, version_of=installed.get)
    assert answer["outcome"] == "installed" and answer["ok"]
    assert seen["argv"][1:5] == ["-m", "pip", "install", "--disable-pip-version-check"]
    assert seen["argv"][seen["argv"].index("-r") + 1] == str(requirements)
    assert ("--user" in seen["argv"]) == (sys.prefix == sys.base_prefix)
    assert seen["constraints"].splitlines() == ["torch==2.9.1+rocm7.13.0", "triton==3.5.1+rocm7.13.0"]
    assert not Path(seen["argv"][seen["argv"].index("-c") + 1]).exists()
    row = only_row(train.INSTALL_VERB)
    assert (row["outcome"], row["refs"]["missing"], row["refs"]["constraints"]) == \
        ("installed", "trl==1.15.0", "torch==2.9.1+rocm7.13.0, triton==3.5.1+rocm7.13.0")


def test_without_torch_nothing_is_installed_and_a_failed_pip_is_reported(tmp_path):
    requirements = tmp_path / "requirements-train.txt"
    requirements.write_text("trl==1.15.0\n", encoding="utf-8")
    calls = []
    refused = train.install(requirements, run=lambda argv: calls.append(argv) or 0, version_of={}.get)
    assert (refused["outcome"], calls) == ("refused", [])
    failed = train.install(requirements, run=lambda argv: 1, version_of={"torch": "2.9.1"}.get)
    assert (failed["outcome"], failed["error"]) == ("failed", "pip exited with 1")
    still = train.install(requirements, run=lambda argv: 0, version_of={"torch": "2.9.1"}.get)
    assert still["outcome"] == "failed" and "trl==1.15.0" in still["error"]


def test_a_command_in_a_venv_installs_without_user():
    assert train.install_command("py", "req.txt", "c.txt", True) == \
        ["py", "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-r", "req.txt", "-c", "c.txt"]
    assert "--user" in train.install_command("py", "req.txt", "c.txt", False)


# --- the entry point's refusals ----------------------------------------------------------------------------------

def run_args(folder: Path, rows: Path, model: Path, out: Path, *more: str) -> list[str]:
    return ["--dataset", str(folder), "--rows", str(rows), "--model", str(model), "--out", str(out), "--workers", "0",
            *more]


def test_a_rehearsal_on_rows_that_leak_a_held_out_example_is_refused_and_recorded_and_keeps_its_data(tmp_path):
    folder, rows = copy_fixture(tmp_path)
    leaked = fixture_rows()
    split = dataset.load_split(folder, require_images=False)
    leaked[0] = {**leaked[0], "id": split.held_out[0].id, "frame_ref": split.held_out[0].frame_ref}
    write_rows(rows, leaked)
    answer = train.main(["--cpu", *run_args(folder, rows, model_folder(tmp_path, [COMMIT]), tmp_path / "out")])
    assert (answer["ok"], answer["outcome"]) == (False, "invalid")
    assert answer["problems"] == [f"row {split.held_out[0].id!r}: held out for the eval gate; no training run may "
                                  "see it"]
    assert answer["refs"]["dataset_id"] == "tool-calling-tiny"
    assert answer["refs"]["data_deleted"] == "no: a --cpu rehearsal keeps its data"
    assert folder.is_dir() and rows.is_file()
    row = only_row()
    assert (row["outcome"], row["refs"]["run_id"], row["refs"]["split_sha256"]) == \
        ("invalid", answer["refs"]["run_id"], split.sha256)
    assert row["refs"]["examples_sha256"] == train.file_sha256(folder / "examples.jsonl")


def test_a_base_with_no_known_commit_refuses_before_anything_loads(tmp_path):
    folder, rows = copy_fixture(tmp_path)
    answer = train.main(["--cpu", *run_args(folder, rows, model_folder(tmp_path), tmp_path / "out")])
    assert answer["outcome"] == "refused"
    assert "the base model's Hub commit is unknown" in answer["error"]
    assert only_row()["outcome"] == "refused"


def test_a_dataset_folder_not_named_for_its_id_and_an_out_folder_inside_it_are_refused(tmp_path):
    folder, rows = copy_fixture(tmp_path)
    renamed = folder.with_name("renamed")
    folder.rename(renamed)
    model = model_folder(tmp_path, [COMMIT])
    answer = train.main(["--cpu", *run_args(renamed, rows, model, tmp_path / "out")])
    assert answer["problems"] == ["the dataset folder is not named for its dataset_id ('tool-calling-tiny')"]
    renamed.rename(folder)
    inside = train.main(["--cpu", *run_args(folder, rows, model, folder / "out")])
    assert (inside["outcome"], inside["problems"]) == \
        ("usage", ["--out is inside the dataset folder, which is deleted after the run"])


def test_on_another_chip_the_run_is_refused_before_it_reads_or_deletes_anything(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(arch="gfx1201"))
    folder, rows = copy_fixture(tmp_path)
    answer = train.main(run_args(folder, rows, model_folder(tmp_path, [COMMIT]), tmp_path / "out"))
    assert (answer["outcome"], answer["error"]) == \
        ("refused", "the GPU is gfx1201, not gfx1151: this is not the trainer pod")
    assert "dataset_id" not in answer["refs"]
    assert folder.is_dir() and rows.is_file()
    assert only_row()["outcome"] == "refused"


def test_on_the_pod_a_refused_run_still_deletes_its_copy_of_the_data(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    folder, rows = copy_fixture(tmp_path)
    leaked = fixture_rows()
    leaked[1] = {**leaked[1], "id": "golden-02", "frame_ref": "fixture:group:last_seen-23377f1f"}
    write_rows(rows, leaked)
    answer = train.main(run_args(folder, rows, model_folder(tmp_path, [COMMIT]), tmp_path / "out"))
    assert (answer["ok"], answer["outcome"]) == (False, "invalid")
    assert answer["refs"]["data_deleted"] == "yes"
    assert (answer["refs"]["mode"], answer["refs"]["node"]) == ("pod", "pod-b")
    assert not folder.exists() and not rows.exists()
    assert only_row()["refs"]["data_deleted"] == "yes"
    assert not (tmp_path / "out").exists()   # refused before its run folder was made: the ledger row is the record


def test_on_the_pod_a_path_that_is_not_an_export_is_left_and_the_row_says_so(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    keep = tmp_path / "keep"
    keep.mkdir()
    rows = write_rows(tmp_path / "rows.jsonl", fixture_rows())
    answer = train.main(run_args(keep, rows, model_folder(tmp_path, [COMMIT]), tmp_path / "out"))
    assert (answer["ok"], answer["outcome"]) == (False, "invalid")
    assert answer["refs"]["data_deleted"] == \
        "no: the dataset path does not look like a dataset export, so it was left"
    assert keep.is_dir() and not rows.exists()


def test_the_entry_point_refuses_to_run_while_the_machine_is_stopped(tmp_path, capsys):
    state = tmp_path / "state"
    state.mkdir()
    (state / "STOP").write_text("maintenance\n", encoding="utf-8")
    folder, rows = copy_fixture(tmp_path)
    with pytest.raises(stop.Stopped):
        train.main(["--cpu", *run_args(folder, rows, model_folder(tmp_path, [COMMIT]), tmp_path / "out")])
    assert json.loads(capsys.readouterr().out)["outcome"] == "stopped"
    assert only_row()["outcome"] == "stopped"
    assert folder.is_dir()


def test_run_by_path_from_another_folder_the_script_prints_one_json_line(tmp_path):
    folder, rows = copy_fixture(tmp_path)
    leaked = fixture_rows()
    leaked[0] = {**leaked[0], "id": "golden-02", "frame_ref": "fixture:group:last_seen-23377f1f"}
    write_rows(rows, leaked)
    env = {**os.environ, "LAB_STATE_DIR": str(tmp_path / "state")}
    done = subprocess.run([sys.executable, str(SCRIPT), "--no-install", "--cpu",
                           *run_args(folder, rows, model_folder(tmp_path, [COMMIT]), tmp_path / "out")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 1, done.stderr
    lines = done.stdout.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["outcome"] == "invalid"
    assert only_row()["outcome"] == "invalid"
