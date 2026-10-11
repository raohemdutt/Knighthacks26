"""lab.package: a trained LoRA adapter packaged for the Jetson, and the dry run that installs it.

What is real here: the packaging step's checks, file copies, sha256 and size logic, manifest.json (validated as
ModelManifest), the ledger rows, the score copy from the gate's row, and the dry-run install read back through
runtime/registry.py's model_version(). What is faked is the toolchain on pod B (PEFT, convert_hf_to_gguf.py,
llama-quantize, git): lab.package_toolchain_fake.FakeToolchain writes small placeholder files where the real tools
write GGUFs. The merge code that runs on the pod is driven once with stand-in torch, transformers and PEFT modules.
Nothing touches a GPU, a pod, a hub or the network. Each test works in its own folders and ledger."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from contracts.learning import ModelManifest
from lab import __main__ as lab_cli
from lab import package
from lab.machine import ledger
from lab.package_toolchain_fake import COMMIT, FakeToolchain
from runtime.registry import model_version

BASE = "Qwen/Qwen3.5-4B"
RUN = "run-0007"
TASK = "desk-objects"
VERSION = "atlas-desk-objects-run-0007"
GATE_METRICS = {"learning.heldout.current": 0.41, "learning.heldout.candidate": 0.78,
                "learning.heldout.candidate.bf16": 0.8, "learning.heldout.teacher": 0.86}
GATE_SCORES = {"current": 0.41, "candidate": 0.78, "candidate.bf16": 0.8, "teacher": 0.86}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def pod(tmp_path):
    """Pod B's folders as the trainer leaves them: the unquantized base, an adapter, the base projector, llama.cpp."""
    base_dir = tmp_path / "models" / "Qwen3.5-4B"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                                      "torch_dtype": "bfloat16"}))
    adapter = tmp_path / "runs" / RUN / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": str(base_dir), "r": 16, "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights " * 64)
    mmproj = tmp_path / "models" / "Qwen3.5-4B-GGUF" / "mmproj-F16.gguf"
    mmproj.parent.mkdir(parents=True)
    mmproj.write_bytes(b"GGUF" + b"base vision projector " * 100)
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    return types.SimpleNamespace(root=tmp_path, base_dir=base_dir, adapter=adapter, mmproj=mmproj, llama=llama,
                                 out=tmp_path / "data" / "packages", state=tmp_path / "state")


class Clock:
    """Reads 100.0, then 142.5: a packaging run of exactly 42.5 seconds."""

    def __init__(self, *times: float) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


def build(pod, toolchain=None, **more):
    kwargs = dict(adapter=pod.adapter, run_id=RUN, task=TASK, base=BASE, base_dir=pod.base_dir, mmproj=pod.mmproj,
                  llama_dir=pod.llama, out_root=pod.out, toolchain=toolchain or FakeToolchain(), state_dir=pod.state,
                  clock=Clock(100.0, 142.5))
    kwargs.update(more)
    return package.build(**kwargs)


def gate_row(pod, run_id=RUN, metrics=GATE_METRICS, version=VERSION):
    """What the gate (L4) writes once it has scored the package: refs.run_id names the run."""
    conditions = {"version": version} if version else {}
    return ledger.append({"actor": "gate", "tier": None, "task": "L4", "verb": "gate", "outcome": "ok",
                          "metrics": dict(metrics), "refs": {"run_id": run_id}, "conditions": conditions},
                         state_dir=pod.state)


def package_rows(pod, verb="package build"):
    return ledger.read(state_dir=pod.state, verb=verb)


# --- merge in bf16, on the unquantized base ---------------------------------------------------------------------

def test_the_three_toolchain_commands_run_in_order_merge_convert_quantize(pod):
    fake = FakeToolchain()
    result = build(pod, fake)
    merge, convert, quantize = result["commands"]
    assert [call for call in fake.calls if call[0] != "git"] == [merge, convert, quantize]
    assert merge[1:4] == ["-m", "lab.package", "merge"]
    assert merge[merge.index("--base-dir") + 1] == str(pod.base_dir)
    assert merge[merge.index("--adapter") + 1] == str(pod.adapter)
    assert Path(convert[1]) == pod.llama / "convert_hf_to_gguf.py"
    assert convert[2] == merge[merge.index("--out") + 1]                 # the merged folder is what gets converted
    assert convert[convert.index("--outtype") + 1] == "bf16"
    assert Path(quantize[0]) == pod.llama / "build" / "bin" / "llama-quantize"
    assert quantize[1] == convert[convert.index("--outfile") + 1]
    assert quantize[3] == "Q4_K_M"


def test_a_quantized_base_is_refused_before_anything_runs(pod):
    config = json.loads((pod.base_dir / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "bitsandbytes", "load_in_4bit": True}
    (pod.base_dir / "config.json").write_text(json.dumps(config))
    fake = FakeToolchain()
    with pytest.raises(package.PackageError) as caught:
        build(pod, fake)
    assert caught.value.kind == "quantized_base"
    assert fake.calls == [] and not (pod.out / VERSION).exists()


@pytest.mark.parametrize("change, kind", [
    (lambda a: (a / "adapter_config.json").unlink(), "bad_adapter"),
    (lambda a: (a / "adapter_model.safetensors").unlink(), "bad_adapter"),
    (lambda a: (a / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA",
                                                                  "base_model_name_or_path": "Qwen/Qwen3.5-9B"})),
     "wrong_base"),
    (lambda a: (a / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": BASE,
                                                                  "target_modules": ["q_proj", "visual.merger.mlp"]})),
     "vision_adapter"),
])
def test_an_adapter_that_cannot_be_packaged_this_way_is_refused(pod, change, kind):
    change(pod.adapter)
    fake = FakeToolchain()
    with pytest.raises(package.PackageError) as caught:
        build(pod, fake)
    assert caught.value.kind == kind and fake.calls == []


def test_the_merge_on_the_pod_loads_the_base_in_bf16_and_saves_merge_and_unload(tmp_path, monkeypatch):
    seen: dict = {}

    class Model:
        def __init__(self, name):
            self.name = name

        def save_pretrained(self, out, **kwargs):
            seen["saved"] = (self.name, out, kwargs)

    class PeftModel:
        @staticmethod
        def from_pretrained(model, adapter):
            seen["adapter"] = (model.name, adapter)
            peft_model = Model("peft")
            peft_model.merge_and_unload = lambda: seen.setdefault("merged", True) and Model("merged")
            return peft_model

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen["base"] = (path, kwargs)
            return Model("base")

    class Tokenizer:
        @staticmethod
        def from_pretrained(path):
            tokenizer = types.SimpleNamespace(save_pretrained=lambda out: seen.setdefault("tokenizer", out))
            return tokenizer

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16-dtype"
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = Loader
    transformers.AutoTokenizer = Tokenizer
    peft = types.ModuleType("peft")
    peft.PeftModel = PeftModel
    for name, module in {"torch": torch, "transformers": transformers, "peft": peft}.items():
        monkeypatch.setitem(sys.modules, name, module)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "config.json").write_text("{}")

    package.merge_adapter(base_dir, tmp_path / "adapter", tmp_path / "merged")

    assert seen["base"] == (str(base_dir), {"dtype": "bf16-dtype", "device_map": {"": "cpu"}})
    assert seen["adapter"] == ("base", str(tmp_path / "adapter"))
    assert seen["merged"] is True
    assert seen["saved"][0] == "merged" and seen["saved"][1] == str(tmp_path / "merged")
    assert seen["saved"][2] == {"safe_serialization": True}
    assert seen["tokenizer"] == str(tmp_path / "merged")


def test_the_merge_refuses_a_quantized_base_on_the_pod_too(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))
    with pytest.raises(package.PackageError) as caught:
        package.merge_adapter(base_dir, tmp_path / "adapter", tmp_path / "merged")
    assert caught.value.kind == "quantized_base"


# --- GGUF, Q4_K_M, the base projector kept unchanged ------------------------------------------------------------

def test_the_package_holds_the_quantized_gguf_and_the_base_mmproj_unchanged(pod):
    result = build(pod)
    folder = pod.out / VERSION
    assert Path(result["folder"]) == folder
    assert sorted(p.name for p in folder.iterdir()) == sorted([f"{VERSION}-Q4_K_M.gguf", "mmproj-F16.gguf", "manifest.json"])
    assert sha(folder / "mmproj-F16.gguf") == sha(pod.mmproj)
    assert (folder / f"{VERSION}-Q4_K_M.gguf").read_bytes().startswith(b"GGUF")
    assert not any(pod.out.glob(".work-*"))                     # the merged checkpoint and the bf16 GGUF are not kept


def test_a_projector_not_named_mmproj_is_refused(pod):
    other = pod.mmproj.with_name("projector.gguf")
    pod.mmproj.rename(other)
    with pytest.raises(package.PackageError) as caught:
        build(pod, mmproj=other)
    assert caught.value.kind == "bad_mmproj"


def test_a_package_that_already_exists_is_never_overwritten(pod):
    build(pod)
    before = (pod.out / VERSION / "manifest.json").read_text()
    with pytest.raises(package.PackageError) as caught:
        build(pod)
    assert caught.value.kind == "exists"
    assert (pod.out / VERSION / "manifest.json").read_text() == before


def test_a_failing_tool_leaves_no_package_and_a_failed_ledger_row(pod):
    with pytest.raises(package.PackageError) as caught:
        build(pod, FakeToolchain(fail_on="quantize"))
    assert caught.value.kind == "toolchain"
    assert not (pod.out / VERSION).exists() and not any(pod.out.glob(".work-*"))
    (row,) = package_rows(pod)
    assert row["outcome"] == "fail" and row["refs"]["run_id"] == RUN and "quantize" in row["note"]
    assert len(row["commands"]) == 3                              # the step that failed is named with the ones before


# --- manifest.json: ModelManifest, one model gguf, at most one mmproj, sha256 for every file ---------------------

def test_manifest_validates_as_model_manifest_with_a_sha256_for_every_file(pod):
    build(pod)
    folder = pod.out / VERSION
    manifest = ModelManifest.model_validate_json((folder / "manifest.json").read_text())
    assert (manifest.version, manifest.base, manifest.task) == (VERSION, BASE, TASK)
    assert (manifest.gguf, manifest.mmproj) == (f"{VERSION}-Q4_K_M.gguf", "mmproj-F16.gguf")
    files = [p for p in folder.iterdir() if p.name != "manifest.json"]
    assert manifest.sha256 == {p.name: sha(p) for p in files}


def test_the_manifest_follows_the_folder_rule_of_make_manifest(pod, monkeypatch):
    real = package.folder_manifest
    seen = []
    monkeypatch.setattr(package, "folder_manifest", lambda folder, **kw: seen.append(folder) or real(folder, **kw))
    build(pod)
    assert seen == [pod.out / f".work-{VERSION}" / "package"]          # built in the work folder, then renamed
    stray = pod.root / "two"
    stray.mkdir()
    (stray / "a.gguf").write_bytes(b"x")
    (stray / "b.gguf").write_bytes(b"y")
    with pytest.raises(package.PackageError) as caught:
        real(stray, base=BASE, version="two", task=None)
    assert caught.value.kind == "folder_rule"


def test_the_version_defaults_from_task_and_run_and_must_be_a_servable_name(pod):
    assert build(pod)["version"] == VERSION
    with pytest.raises(package.PackageError) as caught:
        build(pod, version="current")
    assert caught.value.kind == "bad_version"
    with pytest.raises(package.PackageError):
        build(pod, version="../escape")


# --- scores: copied from the gate's ledger row by run id, never typed ----------------------------------------------

def test_without_a_gate_row_the_package_carries_no_scores(pod):
    result = build(pod)
    assert result["scores"] == {} and result["gate_seq"] is None
    assert json.loads((pod.out / VERSION / "manifest.json").read_text())["scores"] == {}


def test_scores_are_copied_from_the_gate_row_named_by_the_run_id(pod):
    build(pod)
    gate_row(pod, run_id="run-0006", metrics={"learning.heldout.candidate": 0.1})   # another run's gate
    seq = gate_row(pod)
    before = json.loads((pod.out / VERSION / "manifest.json").read_text())

    result = package.copy_scores(pod.out / VERSION, RUN, state_dir=pod.state)

    after = json.loads((pod.out / VERSION / "manifest.json").read_text())
    assert result["scores"] == GATE_SCORES and result["gate_seq"] == seq
    assert ModelManifest.model_validate(after).scores == GATE_SCORES
    assert {k: v for k, v in after.items() if k != "scores"} == {k: v for k, v in before.items() if k != "scores"}
    (row,) = package_rows(pod, "package scores")
    assert row["refs"]["run_id"] == RUN and row["source"] == f"ledger seq {seq}"


def test_build_copies_the_scores_when_the_gate_row_already_exists(pod):
    seq = gate_row(pod)
    result = build(pod)
    assert result["scores"] == GATE_SCORES and result["gate_seq"] == seq


def test_a_gate_row_for_another_package_version_is_refused(pod):
    build(pod)
    gate_row(pod, version="atlas-desk-objects-run-0005")
    with pytest.raises(package.PackageError) as caught:
        package.copy_scores(pod.out / VERSION, RUN, state_dir=pod.state)
    assert caught.value.kind == "gate_mismatch"
    assert json.loads((pod.out / VERSION / "manifest.json").read_text())["scores"] == {}


def test_copy_scores_without_a_gate_row_is_an_error(pod):
    build(pod)
    with pytest.raises(package.PackageError) as caught:
        package.copy_scores(pod.out / VERSION, RUN, state_dir=pod.state)
    assert caught.value.kind == "no_gate_row"


def test_the_newest_gate_row_of_the_run_wins(pod):
    build(pod)
    gate_row(pod, metrics={"learning.heldout.candidate": 0.5})
    gate_row(pod)
    assert package.copy_scores(pod.out / VERSION, RUN, state_dir=pod.state)["scores"] == GATE_SCORES


# --- the run's ledger row ----------------------------------------------------------------------------------------

def test_the_ledger_row_carries_commit_commands_package_mb_and_wall_clock(pod):
    result = build(pod)
    folder = pod.out / VERSION
    size = (folder / f"{VERSION}-Q4_K_M.gguf").stat().st_size + (folder / "mmproj-F16.gguf").stat().st_size
    (row,) = package_rows(pod)
    assert row["outcome"] == "ok" and row["refs"]["run_id"] == RUN and row["node"] == "pod-b"
    assert row["metrics"] == {"learning.package_mb": round(size / 1e6, 6)}
    assert row["units"] == {"learning.package_mb": "MB"}
    assert row["conditions"]["llama_cpp_commit"] == COMMIT
    assert row["conditions"]["quant"] == "Q4_K_M" and row["conditions"]["base"] == BASE
    assert row["commands"] == result["commands"] and len(row["commands"]) == 3
    assert row["wall_seconds"] == 42.5 == result["wall_seconds"]
    assert row["version"] == VERSION and result["ledger_seq"] == row["seq"]


# --- the dry run: installed off the box, read back through runtime/registry.py -----------------------------------

def test_dry_run_installs_under_scratch_data_models_and_model_version_reads_it_back(pod):
    build(pod)
    gate_row(pod)
    package.copy_scores(pod.out / VERSION, RUN, state_dir=pod.state)
    scratch = pod.root / "scratch"

    result = package.dry_run(pod.out / VERSION, scratch)

    installed = scratch / "data" / "models" / VERSION
    assert Path(result["installed"]) == installed and (installed / "manifest.json").is_file()
    assert (scratch / "data" / "models" / "current").resolve() == installed.resolve()
    read_back = model_version({"manifest": str(scratch / "data" / "models" / "current" / "manifest.json")})
    assert (read_back.version, read_back.base, read_back.task) == (VERSION, BASE, TASK)
    assert read_back.scores == GATE_SCORES
    assert result["ok"] is True and result["read_back"]["scores"] == GATE_SCORES


def test_dry_run_refuses_a_package_whose_file_changed(pod):
    build(pod)
    model = pod.out / VERSION / f"{VERSION}-Q4_K_M.gguf"
    model.write_bytes(model.read_bytes() + b"x")
    with pytest.raises(package.PackageError) as caught:
        package.dry_run(pod.out / VERSION, pod.root / "scratch")
    assert caught.value.kind == "verify"


def test_dry_run_refuses_the_repos_own_data_models(pod):
    build(pod)
    with pytest.raises(package.PackageError) as caught:
        package.dry_run(pod.out / VERSION, package.REPO)
    assert caught.value.kind == "not_scratch"


# --- the verb ----------------------------------------------------------------------------------------------------

def test_package_is_one_line_of_the_sorted_verb_registry():
    assert lab_cli.VERBS["package"] == "lab.package:main"
    assert list(lab_cli.VERBS) == sorted(lab_cli.VERBS)


def test_python_m_lab_package_runs_build_scores_and_dry_run_on_the_fake(pod, monkeypatch, capsys):
    monkeypatch.setenv("LAB_STATE_DIR", str(pod.state))
    common = ["--run-id", RUN]
    code = lab_cli.main(["package", "build", "--toolchain", "fake", "--adapter", str(pod.adapter), "--task", TASK,
                         "--base-dir", str(pod.base_dir), "--mmproj", str(pod.mmproj), "--llama-cpp", str(pod.llama),
                         "--out", str(pod.out), *common])
    built = json.loads(capsys.readouterr().out)
    assert code == 0 and built["outcome"] == "ok" and built["version"] == VERSION
    gate_row(pod)
    assert lab_cli.main(["package", "scores", "--package", str(pod.out / VERSION), *common]) == 0
    assert json.loads(capsys.readouterr().out)["scores"] == GATE_SCORES
    assert lab_cli.main(["package", "dry-run", "--package", str(pod.out / VERSION),
                         "--scratch", str(pod.root / "scratch")]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["read_back"] == {"version": VERSION, "base": BASE, "task": TASK, "scores": GATE_SCORES}


def test_a_refused_build_answers_one_json_line_and_exit_1(pod, monkeypatch, capsys):
    monkeypatch.setenv("LAB_STATE_DIR", str(pod.state))
    code = lab_cli.main(["package", "build", "--toolchain", "fake", "--adapter", str(pod.root / "nowhere"),
                         "--task", TASK, "--base-dir", str(pod.base_dir), "--mmproj", str(pod.mmproj),
                         "--llama-cpp", str(pod.llama), "--out", str(pod.out), "--run-id", RUN])
    answer = json.loads(capsys.readouterr().out)
    assert code == 1 and answer["ok"] is False and answer["kind"] == "bad_adapter"


def test_the_real_toolchain_runs_a_command_and_reads_its_output():
    result = package.Toolchain().run([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"])
    assert (result.returncode, result.stdout.strip(), result.stderr.strip()) == (0, "out", "err")
