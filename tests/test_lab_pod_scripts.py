"""The pod scripts end with a non-zero code when a step fails, whoever starts them.

python -m lab.controller run-script uploads a script to the Jupyter root as lab_run_<id>.sh and takes ok from its exit
code, so a script that prints an error and exits 0 writes an ok ledger row for a step that never ran. These tests run
the real lab/pod_b_trainer.sh and lab/pod_a_teacher.sh text in bash from a scratch folder laid out the way run-script
leaves the Jupyter root, with stand-ins on PATH for the programs a step calls. No GPU, no pod and no network is used.
"""
from __future__ import annotations

import os
import shlex
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


def run(root: Path, tmp_path: Path, step: str, stubs: dict[str, str], *, args: tuple[str, ...] = (),
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir(exist_ok=True)
    home.mkdir(exist_ok=True)
    for name, text in stubs.items():
        _stub(bin_dir, name, text)
    env = {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", **(env or {})}
    return subprocess.run(["bash", "./lab_run_t1.sh", step, *args], cwd=root, env=env, capture_output=True, text=True,
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

# A python3 that answers the build's torch version probe (read from stdin), runs vLLM's use_existing_torch.py and
# fails everything else (import vllm among them), as an image without vLLM would.
TORCH_PROBE = ('case "$1" in -) echo "torch==2.9.1+rocm7.13.0" ;; use_existing_torch.py) ;; *) exit 1 ;; esac\n'
               'exit 0')
# A git that logs its arguments and clones by making the folder.
GIT_LOG = ('echo "git $*" >> "$HOME/calls.log"\n'
           'if [ "$1" = clone ]; then mkdir -p "$3/.git"; fi\nexit 0')

def test_build_fails_when_the_vllm_patch_is_not_beside_the_uploaded_script(tmp_path):
    stubs = {"python3": "exit 1", "git": "exit 0", "pip": "exit 0", "nproc": "echo 4", "rocminfo": "echo gfx1201",
             "pkg-config": "exit 0"}
    done = run(jupyter_root(tmp_path, "pod_a_teacher.sh"), tmp_path, "build", stubs)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "vllm-gfx1151.diff" in done.stdout


def test_build_fails_when_the_patch_does_not_apply(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1151.diff",))
    (tmp_path / "home" / "vllm" / ".git").mkdir(parents=True)
    stubs = {"python3": TORCH_PROBE, "nproc": "echo 4", "rocminfo": "echo gfx1201", "pip": "exit 0", "pkg-config": "exit 0",
             "git": 'if [ "$1" = apply ]; then echo "error: patch failed" >&2; exit 1; fi\nexit 0'}
    done = run(root, tmp_path, "build", stubs)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "rc=0" not in done.stdout


def test_build_fails_when_the_pip_build_fails_behind_its_tail(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1151.diff",))
    (tmp_path / "home" / "vllm" / ".git").mkdir(parents=True)
    pip = 'case "$*" in *"-e ."*) echo "error: subprocess-exited-with-error"; exit 1 ;; esac\nexit 0'
    stubs = {"python3": TORCH_PROBE, "nproc": "echo 4", "rocminfo": "echo gfx1201", "git": "exit 0", "pip": pip,
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


def _calls(tmp_path: Path) -> str:
    log = tmp_path / "home" / "calls.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _build_stubs(**more: str) -> dict[str, str]:
    stubs = {"python3": TORCH_PROBE, "nproc": "echo 4", "rocminfo": "echo gfx1151", "pkg-config": "exit 0",
             "git": GIT_LOG, "pip": 'echo "pip $*" >> "$HOME/calls.log"\nexit 0'}
    stubs.update(more)
    return stubs


def test_build_applies_the_gfx1151_patch_by_default_and_stamps_the_folder(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1151.diff",))
    (tmp_path / "home" / "vllm" / ".git").mkdir(parents=True)
    done = run(root, tmp_path, "build", _build_stubs())
    assert done.returncode == 0, done.stdout + done.stderr
    calls = _calls(tmp_path)
    assert f"git apply --reject --whitespace=nowarn {root / 'vllm-gfx1151.diff'}" in calls
    assert "pip install --user --no-build-isolation -e ." in calls
    stamp = (tmp_path / "home" / "vllm" / ".atlas-build-stamp").read_text(encoding="utf-8")
    assert stamp.startswith("commit=b1388b1fbf5aaef47937fabe98931211684666a6 patch=") and stamp.strip().endswith("gfx=gfx1151")
    assert "### build" in done.stdout and "rc=0" in done.stdout


def test_build_still_takes_the_april_patch_when_asked(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1201.diff",))
    (tmp_path / "home" / "vllm" / ".git").mkdir(parents=True)
    done = run(root, tmp_path, "build", _build_stubs(), env={"PATCH": "vllm-gfx1201.diff"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"git apply --reject --whitespace=nowarn {root / 'vllm-gfx1201.diff'}" in _calls(tmp_path)


def test_build_rebuilds_a_folder_whose_stamp_names_another_patch(tmp_path):
    # The pods hold builds made before the stamp existed, or from the April patch: vllm imports there, and that alone
    # must not skip the build.
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1151.diff",))
    vllm = tmp_path / "home" / "vllm"
    (vllm / ".git").mkdir(parents=True)
    (vllm / ".atlas-build-stamp").write_text("commit=b1388b1fb patch=0000 gfx=gfx1151\n", encoding="utf-8")
    python3 = 'if [ "$1" = "-" ]; then echo "torch==2.9.1"; fi\nexit 0'   # vllm imports, and so does all else
    done = run(root, tmp_path, "build", _build_stubs(python3=python3))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "rebuilding" in done.stdout
    assert "git apply" in _calls(tmp_path) and "-e ." in _calls(tmp_path)


def test_build_skips_a_folder_already_built_from_the_same_commit_and_patch(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1151.diff",))
    vllm = tmp_path / "home" / "vllm"
    (vllm / ".git").mkdir(parents=True)
    first = run(root, tmp_path, "build", _build_stubs())
    assert first.returncode == 0, first.stdout + first.stderr
    (tmp_path / "home" / "calls.log").unlink()
    python3 = 'if [ "$1" = "-" ]; then echo "torch==2.9.1"; fi\nexit 0'
    again = run(root, tmp_path, "build", _build_stubs(python3=python3))
    assert again.returncode == 0, again.stdout + again.stderr
    assert "already built" in again.stdout
    assert _calls(tmp_path) == ""


def test_build_refuses_another_folder_without_a_venv_so_the_serving_build_is_not_replaced(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1151.diff",))
    done = run(root, tmp_path, "build", _build_stubs(), args=("--vllm-dir", str(tmp_path / "home" / "vllm-hybrid")))
    assert done.returncode != 0, done.stdout + done.stderr
    assert "--venv" in done.stdout
    assert _calls(tmp_path) == ""


def test_build_beside_the_teacher_goes_into_its_own_folder_and_venv(tmp_path):
    root = jupyter_root(tmp_path, "pod_a_teacher.sh", beside=("vllm-gfx1151.diff",))
    home = tmp_path / "home"
    (home / "vllm" / ".git").mkdir(parents=True)
    # python3 -m venv makes a venv whose python logs its calls, sees a ROCm torch and answers the version probe.
    venv_python = ('echo "venv-python $*" >> "$HOME/calls.log"\n'
                   'if [ "$1" = "-" ]; then echo "torch==2.9.1+rocm7.13.0"; fi\nexit 0')
    python3 = ('if [ "$1 $2" = "-m venv" ]; then\n'
               '  mkdir -p "$4/bin"\n'
               f"  printf '%s\\n' '#!/bin/bash' {shlex.quote(venv_python)} > \"$4/bin/python\"\n"
               '  chmod +x "$4/bin/python"; exit 0\nfi\n' + TORCH_PROBE)
    args = ("--vllm-dir", str(home / "vllm-hybrid"), "--venv", str(home / "venv-hybrid"))
    done = run(root, tmp_path, "build", _build_stubs(python3=python3), args=args)
    assert done.returncode == 0, done.stdout + done.stderr
    calls = _calls(tmp_path)
    assert f"git clone https://github.com/vllm-project/vllm.git {home / 'vllm-hybrid'}" in calls
    assert "venv-python -m pip install --no-build-isolation -e ." in calls
    assert "pip install --user --no-build-isolation" not in calls
    assert (home / "vllm-hybrid" / ".atlas-build-stamp").exists()
    assert not (home / "vllm" / ".atlas-build-stamp").exists()
    assert "vllm_build-vllm-hybrid.log" in done.stdout


def test_serve_puts_the_selected_quantized_matmul_kernel_on_its_ledger_line(tmp_path):
    real_sleep = shutil.which("sleep")
    python3 = ('if [ "$1" = "-m" ]; then echo "INFO Using RDNAHybridW4A16LinearKernel for CompressedTensorsWNA16"; '
               f'exec {real_sleep} 5; fi\nexec {real_sleep} 1')
    stubs = {"python3": python3, "pkill": "exit 0", "nohup": 'exec "$@"', "sleep": "exit 0", "seq": "echo 1",
             "rocminfo": "echo gfx1151"}
    done = run(jupyter_root(tmp_path, "pod_a_teacher.sh"), tmp_path, "serve", stubs)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "### serve" in done.stdout and "rc=0 kernel=RDNAHybridW4A16LinearKernel" in done.stdout
