"""R12, the first kernel experiment on the Strix Halo (gfx1151): one group of tests per offline part of the Done when.

Nothing here needs a GPU, hipcc, a pod or model weights. The patch tests run `git apply --check` against a clean
vLLM checkout at the pinned commit when one is found (R12_VLLM_TREE, ~/vllm, or a `vllm` folder beside this repo)
and skip with the reason otherwise; the rebuild recipe runs with a fake build command against a throwaway git repo;
the LDS recorder and the match checker read the fixture files under fixtures/kernels/."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GPU = REPO / "gpu"
PATCHES = GPU / "patches"
BASE_DIFF = REPO / "lab" / "vllm-gfx1201.diff"
FIX = REPO / "fixtures" / "kernels"
VLLM_COMMIT = "b1388b1fbf5aaef47937fabe98931211684666a6"
KERNEL_FILE = "vllm/v1/attention/ops/triton_unified_attention.py"
LEVERS = ("stages", "block")

sys.path.insert(0, str(REPO))


def patch_path(lever: str) -> Path:
    return PATCHES / f"unified-attention-{lever}.patch"


def git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def files_in(diff_text: str) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", diff_text, flags=re.M))


def clean_tree() -> tuple[Path | None, str]:
    """A clean vLLM checkout at the pinned commit, or None and why not."""
    candidates = [os.environ.get("R12_VLLM_TREE"), str(Path.home() / "vllm"), str(REPO.parent / "vllm")]
    reasons = []
    for c in candidates:
        if not c:
            continue
        tree = Path(c)
        if not (tree / ".git").exists():
            reasons.append(f"{tree}: not a git checkout")
            continue
        head = git("rev-parse", "HEAD", cwd=tree, check=False).stdout.strip()
        if head != VLLM_COMMIT:
            reasons.append(f"{tree}: at {head[:9] or '?'}, not {VLLM_COMMIT[:9]}")
            continue
        if git("status", "--porcelain", "--untracked-files=no", cwd=tree, check=False).stdout.strip():
            reasons.append(f"{tree}: has local changes")
            continue
        return tree, ""
    return None, "no clean vLLM checkout at b1388b1 (set R12_VLLM_TREE): " + "; ".join(reasons)


def added_stages(patch_text: str) -> list[str]:
    """The num_stages values a patch adds, in file order (comment lines left out)."""
    return re.findall(r"^\+\s*num_stages=(\d+),$", patch_text, flags=re.M)


# --- part 1: the two patches ------------------------------------------------------------------------------------

@pytest.mark.parametrize("lever", LEVERS)
def test_each_patch_touches_only_the_unified_attention_kernel(lever: str) -> None:
    text = patch_path(lever).read_text()
    assert files_in(text) == {KERNEL_FILE}


def test_the_patches_are_kept_apart_from_the_build_enabling_hunks() -> None:
    """The recipe applies the diff without its kernel file, so no build-enabling hunk overlaps a lever patch."""
    build_files = files_in(BASE_DIFF.read_text()) - {KERNEL_FILE}
    assert build_files, "the base diff has build-enabling hunks"
    for lever in LEVERS:
        assert not files_in(patch_path(lever).read_text()) & build_files
    recipe = (GPU / "rebuild.sh").read_text()
    assert f'KERNEL_FILE="{KERNEL_FILE}"' in recipe and '--exclude="$KERNEL_FILE"' in recipe


def test_the_stages_patch_sets_one_stage_on_2d_and_two_on_3d() -> None:
    assert added_stages(patch_path("stages").read_text()) == ["1", "2"]
    assert "return 16" not in patch_path("stages").read_text()


def test_the_block_patch_keeps_one_stage_and_widens_the_decode_tile() -> None:
    text = patch_path("block").read_text()
    assert added_stages(text) == ["1", "1"]
    assert "-    return 16 if element_size >= 2 else 32" in text
    assert "+    return 32" in text


@pytest.mark.parametrize("lever", LEVERS)
def test_each_patch_applies_to_the_clean_tree(lever: str) -> None:
    tree, why = clean_tree()
    if tree is None:
        pytest.skip(why)
    run = git("apply", "--check", "-v", str(patch_path(lever)), cwd=tree, check=False)
    assert run.returncode == 0, run.stderr
    # and on top of the build-enabling hunks, as the recipe stacks them
    work = Path(os.environ.get("TMPDIR", "/tmp")) / f"r12-stack-{lever}-{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    try:
        git("worktree", "add", "--detach", str(work), VLLM_COMMIT, cwd=tree)
        git("apply", f"--exclude={KERNEL_FILE}", str(BASE_DIFF), cwd=work)
        run = git("apply", "--check", str(patch_path(lever)), cwd=work, check=False)
        assert run.returncode == 0, run.stderr
    finally:
        git("worktree", "remove", "--force", str(work), cwd=tree, check=False)
        shutil.rmtree(work, ignore_errors=True)


def _code_lines(path: Path) -> list[str]:
    return [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]


@pytest.mark.parametrize("lever,expected", [
    ("stages", [("-", "num_stages=1,"), ("+", "num_stages=2,")]),
    ("block", [("-", "return 16 if element_size >= 2 else 32"), ("+", "return 32")]),
])
def test_each_patch_differs_from_the_stock_kernel_by_one_lever(lever: str, expected: list, tmp_path: Path) -> None:
    """Stock is the teacher's build: the whole base diff, its num_stages=1 hunk included. Each arm's kernel file
    differs from stock's in exactly one code line (comments aside)."""
    tree, why = clean_tree()
    if tree is None:
        pytest.skip(why)
    files = {}
    for arm, patch, extra in (("stock", BASE_DIFF, [f"--include={KERNEL_FILE}"]), (lever, patch_path(lever), [])):
        repo = tmp_path / arm
        (repo / KERNEL_FILE).parent.mkdir(parents=True)
        shutil.copy(tree / KERNEL_FILE, repo / KERNEL_FILE)
        git("init", "-q", cwd=repo)
        git("apply", *extra, str(patch), cwd=repo)
        files[arm] = _code_lines(repo / KERNEL_FILE)
    import difflib
    changed = [(ln[0], ln[1:].strip()) for ln in difflib.unified_diff(files["stock"], files[lever], lineterm="", n=0)
               if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    assert changed == expected

