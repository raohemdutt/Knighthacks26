"""Tests for the chip capacity probe (lab/capacity.py, python -m lab capacity).

Every test runs on the fake chip client (lab/capacity_fake.py), which answers with the R33 chip sheet sample
under lab/refine/research/kernels/data/ or a copy of it written to a temporary folder. No pod, no hub, no GPU and
no Notion is reached: every test gets its own state dir and no .env file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import capacity
from lab.capacity_fake import FakeChipClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "lab" / "refine" / "research" / "kernels" / "data" / "chipsheet-sample.json"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ENV_FILES", "")
    return tmp_path


def r33_sheet(tmp_path: Path, *, gpu_seconds: float = 96.4) -> Path:
    """A quick sheet in the shape lab/kernels/chipsheet/chipsheet.py writes: measurements, versions, device."""
    sheet = {
        "format": "chipsheet/1", "gfx": "gfx1151", "node": "pod-b", "mode": "quick", "gpu_seconds": gpu_seconds,
        "budget_s": 110.0, "versions": {"rocm": "7.13.0", "torch": "2.9.1+rocm7.13.0"},
        "device": {"rocminfo": {"gfx": "gfx1151", "marketing_name": "AMD Radeon 8060S"}, "hip": {"name": "gfx1151"}},
        "measurements": [
            {"id": "copy.d2d.size_1mib", "section": "copy", "unit": "GB/s", "value": 80.0, "status": "ok"},
            {"id": "copy.d2d.size_1gib", "section": "copy", "unit": "GB/s", "value": 212.5, "status": "ok"},
            {"id": "copy.d2d.size_4gib", "section": "copy", "unit": "GB/s", "value": None, "status": "skipped"},
            {"id": "matmul.fp16.decode27b.k5120_n6144.m16", "section": "matmul", "unit": "TFLOP/s", "value": 40.0,
             "status": "ok"},
            {"id": "matmul.bf16.decode27b.k5120_n6144.m1", "section": "matmul", "unit": "TFLOP/s", "value": 3.1,
             "status": "ok"},
            {"id": "matmul.bf16.decode27b.k5120_n6144.m16", "section": "matmul", "unit": "TFLOP/s", "value": 29.5,
             "status": "ok"},
        ],
    }
    path = tmp_path / "chipsheet-gfx1151-quick.json"
    path.write_text(json.dumps(sheet), encoding="utf-8")
    return path


class Clock:
    """A monotonic clock the test moves: each read advances it by step seconds."""

    def __init__(self, step: float) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


# --- the probe on the fake --------------------------------------------------------------------------------------

def test_probe_on_the_fake_reads_the_sample():
    record = capacity.probe(FakeChipClient(), node="pod-b")
    assert record["chip"]["name"] == "Radeon 8060S (Strix Halo, Ryzen AI Max+ 395)"
    assert record["chip"]["gfx"] == "gfx1151"
    assert record["free_gib"]["value"] == 58.0
    assert record["free_gib"]["kind"] == "sample"
    assert record["visible_gib"]["value"] == 64.0
    # the sample has measured neither: unknown, with the reason, never invented
    assert record["bandwidth_gb_s"]["value"] is None and record["bandwidth_gb_s"]["kind"] == "unknown"
    assert record["bf16_tflops"]["value"] is None and record["bf16_tflops"]["kind"] == "unknown"
    # no versions.rocm in the sample: taken from the torch build string, and said so
    assert record["rocm"]["value"] == "7.13.0"
    assert "torch" in record["rocm"]["source"]
    assert record["sample"] is True


def test_probe_asks_for_the_short_mode_of_the_chip_sheet():
    client = FakeChipClient()
    capacity.probe(client, node="pod-b")
    assert client.calls == [("sheet", True), ("memory",)]


def test_probe_reads_an_r33_quick_sheet(tmp_path):
    record = capacity.probe(FakeChipClient(sheet=r33_sheet(tmp_path)), node="pod-b")
    assert record["bandwidth_gb_s"]["value"] == 212.5            # the best ok copy
    assert record["bf16_tflops"]["value"] == 29.5                # the best ok bf16 matmul, never the fp16 one
    assert record["bf16_tflops"]["kind"] == "measured"
    assert record["rocm"] == {"value": "7.13.0", "source": f"{capacity.relative(r33_sheet(tmp_path))}: versions.rocm"}
    assert record["chip"]["name"] == "AMD Radeon 8060S"
    assert record["probe"]["sheet_gpu_s"] == 96.4


def test_probe_wall_clock_and_the_two_minute_budget(tmp_path):
    record = capacity.probe(FakeChipClient(), node="pod-b", clock=Clock(0.5))
    assert record["probe"]["wall_s"] == 0.5
    assert record["probe"]["budget_s"] == 120.0
    assert record["probe"]["within_budget"] is True
    # a quick sheet that used more GPU time than the budget leaves is over two minutes in all
    slow = capacity.probe(FakeChipClient(sheet=r33_sheet(tmp_path, gpu_seconds=119.0)), node="pod-b",
                          clock=Clock(2.0))
    assert slow["probe"]["wall_s"] == 2.0
    assert slow["probe"]["within_budget"] is False


# --- the record -------------------------------------------------------------------------------------------------

def test_record_shape():
    record = capacity.probe(FakeChipClient(), node="pod-b", role=None, clock=Clock(0.25))
    assert set(record) == {"format", "node", "role", "probed_at", "sample", "chip", "rocm", "free_gib", "visible_gib",
                           "bandwidth_gb_s", "bf16_tflops", "probe"}
    assert record["format"] == capacity.FORMAT
    assert record["node"] == "pod-b" and record["role"] == "trainer"
    for field in ("free_gib", "visible_gib", "bandwidth_gb_s", "bf16_tflops"):
        assert set(record[field]) == {"value", "unit", "source", "kind"}
    assert set(record["probe"]) == {"mode", "client", "wall_s", "sheet_gpu_s", "budget_s", "within_budget"}
    assert record["probe"]["mode"] == "quick"
    json.dumps(record)


def test_role_follows_the_node_unless_given():
    assert capacity.probe(FakeChipClient(), node="pod-a")["role"] == "teacher"
    assert capacity.probe(FakeChipClient(), node="friend-1")["role"] == "friend"
    assert capacity.probe(FakeChipClient(), node="pod-a", role="trainer")["role"] == "trainer"


def test_write_and_read_back(tmp_path):
    record = capacity.probe(FakeChipClient(), node="pod-b")
    path = capacity.write_record(record, tmp_path / "cap.json")
    assert capacity.read_record(path) == record


def test_read_record_refuses_another_format(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"format": "chipsheet/1"}), encoding="utf-8")
    with pytest.raises(ValueError):
        capacity.read_record(path)


# --- the verb ---------------------------------------------------------------------------------------------------

def test_verb_is_registered_and_sorted():
    assert lab_cli.VERBS["capacity"] == "lab.capacity:main"
    assert list(lab_cli.VERBS) == sorted(lab_cli.VERBS)


def test_verb_on_the_fake_writes_the_record(tmp_path, capsys):
    out = tmp_path / "pod-b.json"
    code = lab_cli.main(["capacity", "--fake", "--node", "pod-b", "--out", str(out)])
    answer = json.loads(capsys.readouterr().out)
    assert code == 0 and answer["ok"] is True and answer["outcome"] == "probed"
    assert answer["record"] == str(out)
    assert capacity.read_record(out)["free_gib"]["value"] == 58.0


def test_verb_default_out_is_the_state_dir(isolated, capsys):
    code = lab_cli.main(["capacity", "--fake", "--node", "pod-a"])
    answer = json.loads(capsys.readouterr().out)
    assert code == 0
    assert Path(answer["record"]) == isolated / "state" / "capacity" / "capacity-pod-a.json"


def test_verb_reads_files_pulled_from_a_pod(tmp_path, capsys):
    facts = tmp_path / "pod-b.facts.txt"
    facts.write_text("mem: 62 GB total, 55 GB available\ntorch_gpu: ok devices=1 mem_gib=64.0\n", encoding="utf-8")
    out = tmp_path / "cap.json"
    code = lab_cli.main(["capacity", "--sheet", str(r33_sheet(tmp_path)), "--facts", str(facts), "--node", "pod-b",
                         "--out", str(out)])
    capsys.readouterr()
    record = capacity.read_record(out)
    assert code == 0
    assert record["free_gib"]["value"] == 55.0 and record["free_gib"]["kind"] == "recorded"
    assert record["bf16_tflops"]["value"] == 29.5
    assert record["probe"]["client"].startswith("files")


def test_verb_needs_a_client(capsys):
    with pytest.raises(SystemExit) as exc:
        capacity.build_parser().parse_args(["--node", "pod-b"])
    assert exc.value.code == 2
