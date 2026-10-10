"""lab/smoke_lora.py wears the ledger hook: a run of its real main() leaves one smoke-lora row, a run that
cannot start leaves an error row, and the script still imports "lab" when a pod runs it as a bare file. The
GPU stack (torch, transformers, peft, trl, datasets) is the hardware boundary and is replaced by stand-ins;
nothing else is."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lab.machine import ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "lab" / "smoke_lora.py"
LOSSES = [2.23, 0.16]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "sonnet-5")
    monkeypatch.setenv("LAB_TIER", "sonnet")
    monkeypatch.delenv("LAB_TASK", raising=False)
    yield
    sys.modules.pop("lab.smoke_lora", None)


def module(name, **attrs):
    fake = ModuleType(name)
    for key, value in attrs.items():
        setattr(fake, key, value)
    return fake


def install_fake_gpu_stack(monkeypatch, gpu_visible=True):
    """Stand-ins for the libraries that need the card; each keeps what smoke_lora hands it."""
    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(log_history=[{"loss": loss} for loss in LOSSES])

        def train(self):
            return SimpleNamespace(training_loss=0.4)

    stack = {
        "torch": module("torch", bfloat16="bfloat16", version=SimpleNamespace(hip="7.13.0"),
                        cuda=SimpleNamespace(is_available=lambda: gpu_visible, get_device_name=lambda index: "Fake R9700",
                                             max_memory_allocated=lambda: 8 * 2**30)),
        "datasets": module("datasets", Dataset=FakeDataset),
        "peft": module("peft", LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs)),
        "transformers": module(
            "transformers",
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: "tokenizer"),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda path, **kwargs: SimpleNamespace(config=SimpleNamespace(use_cache=True)))),
        "trl": module("trl", SFTConfig=lambda **kwargs: SimpleNamespace(**kwargs), SFTTrainer=FakeTrainer),
    }
    for name, fake in stack.items():
        monkeypatch.setitem(sys.modules, name, fake)


def only_row():
    rows = ledger.read()
    assert len(rows) == 1, rows
    return rows[0]


def import_smoke():
    sys.modules.pop("lab.smoke_lora", None)
    return importlib.import_module("lab.smoke_lora")


def test_a_smoke_run_leaves_one_smoke_lora_row_and_still_prints_and_writes_its_summary(tmp_path, monkeypatch, capsys):
    install_fake_gpu_stack(monkeypatch)
    smoke = import_smoke()
    out_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["smoke_lora.py", "--model", str(tmp_path / "model"), "--out", str(out_dir),
                                      "--steps", "3", "--examples", "8"])
    assert smoke.main() is None
    printed = capsys.readouterr().out.splitlines()
    summary = json.loads(next(line for line in printed if line.startswith("SMOKE_LORA ")).split(" ", 1)[1])
    assert (summary["steps"], summary["examples_seen"], summary["first_loss"], summary["last_loss"],
            summary["peak_mem_gb"]) == (3, 12, 2.23, 0.16, 8.0)
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) == summary
    row = only_row()
    assert (row["verb"], row["outcome"], row["actor"], row["tier"], row["metrics"]) == (
        "smoke-lora", "ok", "sonnet-5", "sonnet", {})
    assert row["duration_s"] >= 0 and row["started"] <= row["ended"]


def test_a_smoke_run_without_a_card_leaves_an_error_row_and_still_fails(tmp_path, monkeypatch):
    install_fake_gpu_stack(monkeypatch, gpu_visible=False)
    smoke = import_smoke()
    monkeypatch.setattr(sys, "argv", ["smoke_lora.py", "--out", str(tmp_path / "run")])
    with pytest.raises(AssertionError, match="no HIP device visible"):
        smoke.main()
    row = only_row()
    assert (row["verb"], row["outcome"], row["note"]) == ("smoke-lora", "error", "AssertionError: no HIP device visible")


BARE_RUN = """
import runpy, sys
script = sys.argv[1]
sys.argv = [script, '--help']
runpy.run_path(script, run_name='smoke_lora_file')['main']()
"""


def test_loaded_from_another_folder_without_lab_on_the_path_the_script_finds_lab_and_records(tmp_path):
    """A pod runs the file by path, so "lab" is not on sys.path unless the script puts it there. The file is
    run_path-loaded without its __main__ block, which needs a pod's forkserver start method."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text("# stands in for the card's torch; --help never reaches it\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = dict(os.environ, PYTHONPATH=str(stubs))
    done = subprocess.run([sys.executable, "-c", BARE_RUN, str(SCRIPT)], cwd=elsewhere, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: ")
    assert done.stderr == ""
    row = only_row()
    assert (row["verb"], row["outcome"], row["actor"]) == ("smoke-lora", "ok", "sonnet-5")
