"""Guards on lab/quantize_teacher.py, the teacher's 3-bit and 2-bit quantization recipe for the Colab A100.

The GPU run itself cannot happen here; what can is everything around it: the plan and recipe it commits to, the
refusals that keep it off ROCm and CPU-only machines, the Hub preconditions checked before hours of work, the
promise that it ships alone (standard library at import time) and that no token ever reaches its output.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lab import quantize_teacher as qt

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "lab" / "quantize_teacher.py"
PLAN = REPO_ROOT / "fixtures" / "lab" / "quant" / "plan.json"
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def fake_torch(hip=None, cuda=True, memory_gb=80):
    props = SimpleNamespace(total_memory=memory_gb * 1e9)
    cuda_ns = SimpleNamespace(is_available=lambda: cuda, get_device_properties=lambda i: props)
    return SimpleNamespace(version=SimpleNamespace(hip=hip), cuda=cuda_ns)


def test_top_level_imports_are_standard_library_only_so_it_ships_alone():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert set(names) <= STDLIB, names


def test_dry_run_runs_alone_and_imports_no_gpu_package(tmp_path):
    copy = tmp_path / "quantize_teacher.py"
    copy.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    check = (
        "import runpy, sys\n"
        "sys.argv = ['quantize_teacher.py', '--dry-run']\n"
        "try:\n"
        f"    runpy.run_path({str(copy)!r}, run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0, e.code\n"
        "bad = [m for m in ('torch', 'transformers', 'llmcompressor', 'datasets', 'huggingface_hub') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", check], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["base_model"] == "Qwen/Qwen3.8-27B"


def test_the_committed_plan_is_what_dry_run_prints(capsys, monkeypatch):
    monkeypatch.delenv(qt.NAMESPACE_ENV, raising=False)
    assert qt.main(["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(PLAN.read_text(encoding="utf-8"))


def test_each_width_uses_its_method_scheme_and_group():
    p = qt.plan([3, 2])
    by_width = {v["width"]: v for v in p["variants"]}
    assert by_width[3]["method"] == "gptq" and by_width[3]["scheme"] == "W3A16"
    assert by_width[2]["method"] == "autoround" and by_width[2]["scheme"] == "W2A16"
    assert p["group_size"] == 128
    assert all(v["symmetric"] for v in p["variants"])
    assert "GPTQModifier" in by_width[3]["recipe"]["quant_stage"]["quant_modifiers"]
    assert "AutoRoundModifier" in by_width[2]["recipe"]["quant_stage"]["quant_modifiers"]


def test_recipe_text_is_valid_yaml_and_keeps_the_bf16_modules():
    for width in qt.WIDTHS:
        args = next(iter(yaml.safe_load(json.dumps(qt.recipe(width)))["quant_stage"]["quant_modifiers"].values()))
        ignore = " ".join(args["ignore"])
        for kept in ("visual", "lm_head", "embed_tokens", "in_proj_a", "in_proj_b"):
            assert kept in ignore
        assert args["targets"] == ["Linear"]


def test_names_are_distinct_per_width():
    assert qt.checkpoint_name(3) != qt.checkpoint_name(2)
    assert "W3A16" in qt.checkpoint_name(3) and "W2A16" in qt.checkpoint_name(2)


def test_refuses_a_rocm_torch():
    with pytest.raises(qt.QuantError, match="ROCm"):
        qt.check_torch(fake_torch(hip="7.13.0"))


def test_refuses_a_cpu_only_torch():
    with pytest.raises(qt.QuantError, match="CPU-only"):
        qt.check_torch(fake_torch(cuda=False))


def test_refuses_a_small_gpu_and_accepts_an_a100():
    with pytest.raises(qt.QuantError, match="GB"):
        qt.check_torch(fake_torch(memory_gb=24))
    qt.check_torch(fake_torch(memory_gb=40))
    qt.check_torch(fake_torch(memory_gb=80))


def test_upload_needs_the_namespace_and_the_token():
    assert qt.hub_target(3, upload=False, environ={}) is None
    with pytest.raises(qt.QuantError, match=qt.NAMESPACE_ENV):
        qt.hub_target(3, upload=True, environ={qt.TOKEN_ENV: "x"})
    with pytest.raises(qt.QuantError, match=qt.TOKEN_ENV):
        qt.hub_target(3, upload=True, environ={qt.NAMESPACE_ENV: "someone"})
    assert qt.hub_target(2, True, {qt.NAMESPACE_ENV: "someone", qt.TOKEN_ENV: "x"}) == f"someone/{qt.checkpoint_name(2)}"


def test_upload_preconditions_fail_before_any_download(monkeypatch, capsys):
    monkeypatch.delenv(qt.NAMESPACE_ENV, raising=False)
    monkeypatch.setattr(qt, "run", lambda *a, **k: pytest.fail("run must not start"))
    assert qt.main(["--width", "3", "--upload"]) == 2
    assert qt.NAMESPACE_ENV in capsys.readouterr().err


def test_missing_calibration_file_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(qt, "run", lambda *a, **k: pytest.fail("run must not start"))
    assert qt.main(["--width", "2", "--calibration", str(tmp_path / "none.jsonl")]) == 2
    assert "not found" in capsys.readouterr().err


def test_the_token_never_reaches_the_output(monkeypatch, capsys):
    secret = "hf_" + "Q" * 30  # built at run time, so this file holds no token-shaped literal
    monkeypatch.setenv(qt.TOKEN_ENV, secret)
    monkeypatch.setenv(qt.NAMESPACE_ENV, "someone")
    assert qt.main(["--dry-run"]) == 0
    out = capsys.readouterr()
    assert secret not in out.out + out.err
    assert "someone/" in out.out


def test_width_parsing():
    assert qt.parse_widths("2,3") == [3, 2]
    assert qt.parse_widths("3") == [3]
    with pytest.raises(Exception):
        qt.parse_widths("4")


def test_calibration_reader_takes_the_committed_set():
    texts = qt.read_calibration(REPO_ROOT / qt.DEFAULT_CALIBRATION)
    assert len(texts) == 184


def test_requirements_pin_the_read_commits():
    req = (REPO_ROOT / "lab" / "requirements-quantize.txt").read_text(encoding="utf-8")
    assert "llm-compressor@229758564a9a9607b858daccb65edb0c633315ec" in req
    assert "compressed-tensors@41103c0bf3db81fc42d9c90349c42ddf0ca6ff22" in req
    assert not any(line.startswith("torch") for line in req.splitlines())


def test_outside_a_kernel_the_command_line_is_used():
    assert qt.kernel_argv("/usr/bin/python3", environ={}) is None
    assert qt.kernel_argv("lab/quantize_teacher.py", environ={}) is None


def test_inside_a_kernel_the_args_file_is_read(tmp_path):
    args = tmp_path / "args.txt"
    args.write_text("# a comment\n--width 3 --out '/content/quant/out dir'\n", encoding="utf-8")
    argv0 = "/usr/local/lib/python3.12/dist-packages/ipykernel_launcher.py"
    assert qt.kernel_argv(argv0, environ={qt.ARGS_FILE_ENV: str(args)}) == ["--width", "3", "--out",
                                                                         "/content/quant/out dir"]
    with pytest.raises(qt.QuantError, match="args file"):
        qt.kernel_argv(argv0, environ={qt.ARGS_FILE_ENV: str(tmp_path / "missing.txt")})


def test_the_committed_args_files_parse_and_point_at_the_uploaded_files():
    for width in (3, 2):
        text = (REPO_ROOT / "fixtures" / "lab" / "quant" / f"args-w{width}.txt").read_text(encoding="utf-8")
        argv = qt.shlex.split(text, comments=True)
        assert argv[argv.index("--width") + 1] == str(width)
        assert argv[argv.index("--calibration") + 1] == "/content/quant/calibration.jsonl"
        assert argv[argv.index("--requirements") + 1] == "/content/quant/requirements-quantize.txt"
        assert "--install" in argv and "--upload" in argv and "--token-file" in argv


def test_the_placeholder_namespace_is_refused():
    with pytest.raises(qt.QuantError, match="namespace"):
        qt.hub_target(3, True, {qt.TOKEN_ENV: "x"}, namespace=qt.PLACEHOLDER)


def test_the_token_may_come_from_a_file(tmp_path):
    token_file = tmp_path / "hf_token"
    token_file.write_text("  abc  \n", encoding="utf-8")
    assert qt.hub_token({}, str(token_file)) == "abc"
    assert qt.hub_token({qt.TOKEN_ENV: "env"}, str(token_file)) == "env"
    assert qt.hub_target(3, True, {}, namespace="someone", token_file=str(token_file)).startswith("someone/")
    with pytest.raises(qt.QuantError, match="token"):
        qt.hub_target(3, True, {}, namespace="someone", token_file=str(tmp_path / "none"))


def test_install_refuses_a_missing_requirements_file(tmp_path):
    with pytest.raises(qt.QuantError, match="requirements"):
        qt.install(str(tmp_path / "requirements-quantize.txt"))
