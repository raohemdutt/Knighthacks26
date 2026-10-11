"""The pod scripts end with a non-zero code when a step fails, whoever starts them.

python -m lab.controller run-script uploads a script to the Jupyter root as lab_run_<id>.sh and takes ok from its exit
code, so a script that prints an error and exits 0 writes an ok ledger row for a step that never ran. These tests run
the real lab/pod_b_trainer.sh and lab/pod_a_teacher.sh text in bash from a scratch folder laid out the way run-script
leaves the Jupyter root, with stand-ins on PATH for the programs a step calls. No GPU, no pod and no network is used.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "lab"

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")


def _stub(bin_dir: Path, name: str, text: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + text + "\n", encoding="ascii")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def jupyter_root(tmp_path: Path, script: str, *, beside: tuple[str, ...] = ()) -> Path:
    """A folder like the pod's Jupyter root after run-script: the script as lab_run_t1.sh, and any --with files."""
    root = tmp_path / "root"
    root.mkdir()
    shutil.copyfile(LAB / script, root / "lab_run_t1.sh")
    for name in beside:
        shutil.copyfile(LAB / name, root / name)
    return root


def run(root: Path, tmp_path: Path, step: str, stubs: dict[str, str]) -> subprocess.CompletedProcess[str]:
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir(exist_ok=True)
    home.mkdir(exist_ok=True)
    for name, text in stubs.items():
        _stub(bin_dir, name, text)
    env = {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    return subprocess.run(["bash", "./lab_run_t1.sh", step], cwd=root, env=env, capture_output=True, text=True,
                          timeout=60)


# --- pod B: smoke ----------------------------------------------------------------------------

def test_smoke_fails_when_smoke_lora_py_is_not_beside_the_uploaded_script(tmp_path):
    # run-script uploads only the script unless --with names smoke_lora.py: python3 then cannot open the file.
    done = run(jupyter_root(tmp_path, "pod_b_trainer.sh"), tmp_path, "smoke", {})
    assert done.returncode != 0, done.stdout + done.stderr
    assert "smoke_lora.py" in done.stdout + done.stderr


def test_smoke_fails_when_the_lora_run_itself_fails(tmp_path):
    root = jupyter_root(tmp_path, "pod_b_trainer.sh", beside=("smoke_lora.py",))
    done = run(root, tmp_path, "smoke", {"python3": 'echo "RuntimeError: No HIP GPUs are available" >&2\nexit 1'})
    assert done.returncode != 0, done.stdout + done.stderr
    assert "No HIP GPUs" in done.stdout


def test_smoke_succeeds_and_prints_its_summary_when_the_run_works(tmp_path):
    root = jupyter_root(tmp_path, "pod_b_trainer.sh", beside=("smoke_lora.py",))
    done = run(root, tmp_path, "smoke", {"python3": 'echo "SMOKE_LORA {\\"steps\\": 20}"'})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "SMOKE_LORA" in done.stdout


def test_all_stops_at_the_first_step_that_fails(tmp_path):
    root = jupyter_root(tmp_path, "pod_b_trainer.sh", beside=("smoke_lora.py",))
    # The proxy check fails and its install fails: nothing after it may run, and the script says so in its code.
    done = run(root, tmp_path, "all", {"python3": "exit 1", "pip": "exit 1", "git": "touch \"$HOME/git_ran\"; exit 0"})
    assert done.returncode != 0
    assert not (tmp_path / "home" / "git_ran").exists()


# --- pod A: build and serve ----------------------------------------------------------------

def test_build_fails_when_the_vllm_patch_is_not_beside_the_uploaded_script(tmp_path):
    stubs = {"python3": "exit 1", "git": "exit 0", "pip": "exit 0", "nproc": "echo 4", "rocminfo": "echo gfx1201",
             "pkg-config": "exit 0"}
    done = run(jupyter_root(tmp_path, "pod_a_teacher.sh"), tmp_path, "build", stubs)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "vllm-gfx1201.diff" in done.stdout


def test_build_fails_when_the_patch_does_not_apply(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1201.diff",))
    (tmp_path / "home" / "vllm" / ".git").mkdir(parents=True)
    stubs = {"python3": "exit 1", "nproc": "echo 4", "rocminfo": "echo gfx1201", "pip": "exit 0", "pkg-config": "exit 0",
             "git": 'if [ "$1" = apply ]; then echo "error: patch failed" >&2; exit 1; fi\nexit 0'}
    done = run(root, tmp_path, "build", stubs)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "rc=0" not in done.stdout


def test_build_fails_when_the_pip_build_fails_behind_its_tail(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1201.diff",))
    (tmp_path / "home" / "vllm" / ".git").mkdir(parents=True)
    pip = 'case "$*" in *"-e ."*) echo "error: subprocess-exited-with-error"; exit 1 ;; esac\nexit 0'
    stubs = {"python3": "exit 1", "nproc": "echo 4", "rocminfo": "echo gfx1201", "git": "exit 0", "pip": pip,
             "pkg-config": "exit 0"}
    done = run(root, tmp_path, "build", stubs)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "### build" in done.stdout and "rc=0" not in done.stdout


def test_serve_fails_when_the_server_never_answers(tmp_path):
    # Every readiness check fails and the server process stays up: after the wait the step must not report ok.
    real_sleep = shutil.which("sleep")
    stubs = {"python3": f'if [ "$1" = "-m" ]; then exec {real_sleep} 5; fi\nexit 1', "pkill": "exit 0",
             "nohup": 'exec "$@"', "sleep": "exit 0", "seq": "echo 1; echo 2", "rocminfo": "echo gfx1201"}
    done = run(jupyter_root(tmp_path, "pod_a_teacher.sh"), tmp_path, "serve", stubs)
    assert done.returncode != 0
    assert "did not come up" in done.stdout
