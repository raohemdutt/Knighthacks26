"""E29: lab/pod_b_llamacpp.sh (llama.cpp HIP on pod B) and the rows its measurement left.

The script tests run the real script text in bash from a scratch folder laid out the way lab.controller run-script
leaves the Jupyter root, with stand-ins on PATH; no GPU, pod or network is used. The fixture test recomputes the
teacher_llamacpp.* medians in lab/refine/metrics.jsonl from the raw rows in fixtures/lab/bench/pod-b-llamacpp-2026-10-11.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import statistics
import subprocess
from pathlib import Path

import pytest

from lab.machine import metrics

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "lab" / "pod_b_llamacpp.sh"
FIXTURE = REPO / "fixtures" / "lab" / "bench" / "pod-b-llamacpp-2026-10-11"

needs_bash = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")


def _stub(bin_dir: Path, name: str, text: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + text + "\n", encoding="ascii")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def run(tmp_path: Path, *args: str, stubs: dict[str, str] | None = None,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    root, bin_dir, home = tmp_path / "root", tmp_path / "bin", tmp_path / "home"
    for d in (root, bin_dir, home):
        d.mkdir(exist_ok=True)
    shutil.copyfile(SCRIPT, root / "lab_run_t1.sh")
    for name, text in (stubs or {}).items():
        _stub(bin_dir, name, text)
    full_env = {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", **(env or {})}
    return subprocess.run(["bash", "./lab_run_t1.sh", *args], cwd=root, env=full_env, capture_output=True,
                          text=True, timeout=60)


def fake_build(tmp_path: Path) -> Path:
    build = tmp_path / "llama" / "build-hip"
    (build / "bin").mkdir(parents=True)
    for name in ("llama-server", "llama-bench", "llama-cli"):
        _stub(build / "bin", name, "echo ran $0 \"$@\"")
    return build


@needs_bash
def test_an_unknown_step_exits_2(tmp_path):
    done = run(tmp_path, "nonsense")
    assert done.returncode == 2
    assert "unknown step" in done.stdout


@needs_bash
def test_build_is_skipped_when_the_binaries_exist(tmp_path):
    build = fake_build(tmp_path)
    done = run(tmp_path, "build", env={"BUILD_DIR": str(build)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "already built" in done.stdout
    assert "### build wall_s=" in done.stdout and "rc=0" in done.stdout


@needs_bash
def test_build_refuses_a_checkout_at_another_commit(tmp_path):
    # Another task's tree is never moved: a checkout at a different commit ends the step with a non-zero code.
    tree = tmp_path / "other-tree"
    (tree / ".git").mkdir(parents=True)
    stubs = {"git": 'if [ "$3" = "rev-parse" ]; then echo 0000000000000000000000000000000000000000; fi',
             "rocminfo": "echo '  Name: gfx1151'", "cmake": "exit 0", "ninja": "exit 0"}
    done = run(tmp_path, "build", stubs=stubs, env={"LLAMA_DIR": str(tree), "BUILD_DIR": str(tmp_path / "b")})
    assert done.returncode != 0
    assert "not 23b0202" in done.stdout


@needs_bash
def test_download_fails_on_a_checksum_mismatch(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "Qwen3.8-27B-Q4_K_M.gguf").write_text("not the real file")
    (models / "mmproj-Qwen3.8-27B-f16.gguf").write_text("not the real file")
    done = run(tmp_path, "download", env={"MODEL_DIR": str(models)})
    assert done.returncode != 0
    assert "sha256 MISMATCH" in done.stdout
    assert not list(models.glob(".verified-*"))


@needs_bash
def test_serve_refuses_while_another_engine_holds_the_card(tmp_path):
    build = fake_build(tmp_path)
    stubs = {"pgrep": "echo '4242 python3 -m vllm.entrypoints.openai.api_server --port 8100'"}
    done = run(tmp_path, "serve", "16", stubs=stubs, env={"BUILD_DIR": str(build)})
    assert done.returncode != 0
    assert "the card is in use" in done.stdout


@needs_bash
def test_serve_without_a_build_fails(tmp_path):
    done = run(tmp_path, "serve", env={"BUILD_DIR": str(tmp_path / "nothing")})
    assert done.returncode != 0
    assert "run build first" in done.stdout


@needs_bash
def test_stop_with_no_server_is_fine(tmp_path):
    done = run(tmp_path, "stop", stubs={"pgrep": "exit 1"})
    assert done.returncode == 0
    assert "no llama-server on 8100" in done.stdout


def _fixture_rows() -> list[dict]:
    rows = []
    for path in sorted(FIXTURE.glob("*.rows.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def test_fixture_rows_are_the_runs_the_ledger_names():
    rows = _fixture_rows()
    assert len(rows) == 8
    assert sorted(r["batch_size"] for r in rows) == [1, 1, 1, 4, 4, 4, 8, 16]
    assert all(r["engine"] == "llamacpp" and r["node"] == "pod-b" and r["outcome"] == "ok" for r in rows)
    assert all(r["conditions"]["checkpoint"].startswith("/home/jovyan/models/") for r in rows)


@pytest.mark.parametrize("name, pick", [
    ("decode_tok_s", lambda r: r["decode_tok_s"]),
    ("decode_tok_s_total", lambda r: r["decode_tok_s_total"]),
    ("ttft_s", lambda r: r["ttft_s"]),
    ("joules_per_token", lambda r: r["joules_per_token"]),
    ("watts_avg", lambda r: r["watts"]["avg_w"]),
    ("watts_peak", lambda r: r["watts"]["peak_w"]),
])
def test_ledger_medians_match_the_fixture_rows(name, pick):
    rows = _fixture_rows()
    for batch in (1, 4, 8, 16):
        runs = [r for r in rows if r["batch_size"] == batch]
        stored = metrics.history(f"teacher_llamacpp.{name}", node="pod-b", conditions={"batch_size": batch})
        assert len(stored) == 1
        assert stored[0]["value"] == round(statistics.median(pick(r) for r in runs), 4)
        assert stored[0]["conditions"]["runs"] == [r["run_id"] for r in runs]
    assert not metrics.history(f"teacher_llamacpp.{name}", node="pod-b", conditions={"batch_size": 64})


def test_llama_bench_rows_match_its_json():
    for entry in json.loads((FIXTURE / "llama-bench-hip.json").read_text(encoding="utf-8")):
        test = f"pp{entry['n_prompt']}" if entry["n_prompt"] else f"tg{entry['n_gen']}"
        stored = metrics.history(f"teacher_llamacpp.llama_bench_{test}_tok_s", node="pod-b")
        assert [row["value"] for row in stored] == [round(entry["avg_ts"], 4)]
