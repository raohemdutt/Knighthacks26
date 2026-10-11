"""lab/pod_a_teacher.sh serve: the flags the teacher's vLLM server is started with.

The script itself runs, in bash, with stand-ins on PATH for python3, pkill, nohup and sleep: the python3 stand-in writes
the arguments it was started with to a file and answers the readiness check, so no GPU, no vLLM and no pod is involved
and nothing is started. What is checked is the command line the real script builds."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "lab" / "pod_a_teacher.sh"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")

STUBS = {
    # `python3 -m vllm.entrypoints...` is the server: write its arguments, one a line. Any other python3 call is the
    # readiness check, which says the server is up.
    "python3": '#!/bin/bash\nif [ "$1" = "-m" ]; then printf "%s\\n" "$@" > "$HOME/vllm_argv.part" && '
               'mv "$HOME/vllm_argv.part" "$HOME/vllm_argv.txt"; fi\nexit 0\n',
    "pkill": "#!/bin/bash\nexit 0\n",
    "nohup": '#!/bin/bash\nexec "$@"\n',
    "sleep": "#!/bin/bash\nexit 0\n",
}


def _runs(program: str) -> bool:
    try:
        return subprocess.run([program, "-c", "true"], capture_output=True).returncode == 0
    except OSError:
        return False


def _bash() -> str:
    # On Windows a bare "bash" can be WSL's, which neither sees this PATH nor reads a Windows path: Git Bash first.
    candidates = (str(GIT_BASH), "bash") if os.name == "nt" else ("bash",)
    found = next((b for b in candidates if _runs(b)), None)
    assert found is not None, "bash is needed to run lab/pod_a_teacher.sh (Git Bash on Windows)"
    return found


def run_serve(tmp_path: Path, gfx: str, args: tuple[str, ...] = (), env_extra: dict[str, str] | None = None
              ) -> tuple[subprocess.CompletedProcess, Path]:
    """Run `bash lab/pod_a_teacher.sh serve <args>` with the stand-ins; the process and the stand-in home folder."""
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()
    home.mkdir(exist_ok=True)
    for name, text in STUBS.items():
        path = bin_dir / name
        path.write_bytes(text.encode("ascii"))
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = {**os.environ, "HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "GFX": gfx,
           **(env_extra or {})}
    done = subprocess.run([_bash(), "lab/pod_a_teacher.sh", "serve", *args], cwd=REPO, env=env, capture_output=True,
                          timeout=120)
    return done, home


def serve_arguments(tmp_path: Path, gfx: str, args: tuple[str, ...] = (), env_extra: dict[str, str] | None = None
                    ) -> list[str]:
    """Run `bash lab/pod_a_teacher.sh serve` with the stand-ins and return the arguments vLLM was started with."""
    done, home = run_serve(tmp_path, gfx, args, env_extra)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    argv = home / "vllm_argv.txt"
    deadline = time.monotonic() + 20       # the server line runs in the background; its stand-in may finish after
    while not argv.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert argv.exists(), "the script never started the server: " + done.stdout.decode("utf-8", "replace")
    return argv.read_text(encoding="utf-8").splitlines()


def values(argv: list[str], flag: str) -> list[str]:
    """The values that follow a flag, up to the next flag."""
    assert flag in argv, f"{flag} is not on the command line: {argv}"
    found = []
    for item in argv[argv.index(flag) + 1:]:
        if item.startswith("--"):
            break
        found.append(item)
    return found


@pytest.mark.parametrize("gfx", ["gfx1151", "gfx1201"])
def test_the_teacher_answers_as_teacher_and_as_group_brain_and_schedules_by_priority(tmp_path, gfx):
    argv = serve_arguments(tmp_path, gfx)

    assert argv[:2] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert values(argv, "--served-model-name") == ["teacher", "group-brain"]
    assert values(argv, "--scheduling-policy") == ["priority"]
    assert values(argv, "--port") == ["8000"]


@pytest.mark.parametrize("gfx", ["gfx1151", "gfx1201"])
def test_the_server_keeps_no_request_and_listens_only_on_the_pod_itself(tmp_path, gfx):
    argv = serve_arguments(tmp_path, gfx)

    assert not {"--enable-log-requests", "--enable-log-outputs"} & set(argv)
    assert values(argv, "--host") == ["127.0.0.1"]


@pytest.mark.parametrize("gfx", ["gfx1151", "gfx1201"])
def test_the_server_turns_on_tool_calling_with_the_qwen3_parser(tmp_path, gfx):
    argv = serve_arguments(tmp_path, gfx)

    assert "--enable-auto-tool-choice" in argv
    assert values(argv, "--tool-call-parser") == ["qwen3_coder"]


def speculative_config(argv: list[str]) -> dict:
    found = values(argv, "--speculative-config")
    assert len(found) == 1, found
    return json.loads(found[0])


@pytest.mark.parametrize("args", [(), ("--speculative", "off"), ("--speculative=off",)])
def test_speculative_decoding_is_off_unless_asked_for(tmp_path, args):
    argv = serve_arguments(tmp_path, "gfx1151", args)

    assert "--speculative-config" not in argv


def test_ngram_speculation_uses_vllms_prompt_lookup_and_keeps_tool_calling(tmp_path):
    argv = serve_arguments(tmp_path, "gfx1151", ("--speculative", "ngram"))

    assert speculative_config(argv) == {"method": "ngram", "num_speculative_tokens": 4,
                                        "prompt_lookup_min": 1, "prompt_lookup_max": 3}
    assert "--enable-auto-tool-choice" in argv
    assert values(argv, "--tool-call-parser") == ["qwen3_coder"]
    assert values(argv, "--served-model-name") == ["teacher", "group-brain"]


def test_the_speculation_settings_come_from_the_environment(tmp_path):
    argv = serve_arguments(tmp_path, "gfx1151", ("--speculative=ngram",),
                           {"SPEC_K": "8", "SPEC_MIN": "2", "SPEC_MAX": "4"})

    assert speculative_config(argv) == {"method": "ngram", "num_speculative_tokens": 8,
                                        "prompt_lookup_min": 2, "prompt_lookup_max": 4}


def test_draft_speculation_runs_the_small_model_from_the_models_folder(tmp_path):
    draft = tmp_path / "home" / "models" / "Qwen3.5-0.8B"
    draft.mkdir(parents=True)
    (draft / "config.json").write_text("{}", encoding="utf-8")
    argv = serve_arguments(tmp_path, "gfx1151", ("--speculative", "draft"))

    assert speculative_config(argv) == {"method": "draft_model", "model": str(draft), "num_speculative_tokens": 4}
    assert "--enable-auto-tool-choice" in argv


def test_draft_speculation_without_the_draft_checkpoint_starts_nothing(tmp_path):
    done, home = run_serve(tmp_path, "gfx1151", ("--speculative", "draft"))

    assert done.returncode != 0
    assert b"draft checkpoint missing" in done.stdout
    assert not (home / "vllm_argv.txt").exists()


@pytest.mark.parametrize("args", [("--speculative", "eagle"), ("--speculative",), ("--fast",)])
def test_an_unknown_speculation_mode_or_option_starts_nothing(tmp_path, args):
    done, home = run_serve(tmp_path, "gfx1151", args)

    assert done.returncode != 0
    assert not (home / "vllm_argv.txt").exists()
