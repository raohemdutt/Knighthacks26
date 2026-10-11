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


# --- part 2: the rebuild recipe ---------------------------------------------------------------------------------

def _commit_all(repo: Path, message: str) -> str:
    git("add", "-A", cwd=repo)
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", message, cwd=repo)
    return git("rev-parse", "HEAD", cwd=repo).stdout.strip()


@pytest.fixture
def fake_pod(tmp_path: Path) -> dict:
    """A throwaway upstream with the kernel file and a build file, a base diff touching both (like the real one),
    a lever patch on the kernel file alone, and a fake build that counts its runs."""
    up = tmp_path / "upstream"
    (up / KERNEL_FILE).parent.mkdir(parents=True)
    (up / KERNEL_FILE).write_text("def launch():\n    stages = 2\n    tile = 16\n    return stages, tile\n")
    (up / "build.txt").write_text("torch==2.10.0\nninja\n")
    git("init", "-q", cwd=up)
    commit = _commit_all(up, "upstream")
    # the base diff: a build-enabling hunk plus the known-fix hunk on the kernel file
    (up / "build.txt").write_text("ninja\n")
    (up / KERNEL_FILE).write_text("def launch():\n    stages = 1\n    tile = 16\n    return stages, tile\n")
    base = tmp_path / "base.diff"
    base.write_text("text before the first header is ignored\n\n" + git("diff", cwd=up).stdout)
    git("checkout", "-q", "--", ".", cwd=up)
    patches = tmp_path / "patches"
    patches.mkdir()
    (up / KERNEL_FILE).write_text("def launch():\n    stages = 1\n    tile = 32\n    return stages, tile\n")
    (patches / "unified-attention-block.patch").write_text(git("diff", cwd=up).stdout)
    git("checkout", "-q", "--", ".", cwd=up)
    counter = tmp_path / "builds.txt"
    env = {**os.environ, "HOME": str(tmp_path / "home"), "VLLM_DIR": str(tmp_path / "home" / "vllm"),
           "VLLM_REPO": str(up), "VLLM_COMMIT": commit, "R12_BASE_DIFF": str(base), "R12_PATCH_DIR": str(patches),
           "R12_STATE_DIR": str(tmp_path / "state"), "R12_REBUILD_JSON": str(tmp_path / "rebuild_times.json"),
           "R12_BUILD_CMD": f"cat {KERNEL_FILE} > /dev/null && echo built >> {counter}", "GFX": "gfx1151"}
    return {"env": env, "tmp": tmp_path, "patches": patches, "counter": counter, "tree": tmp_path / "home" / "vllm"}


def rebuild(fake: dict, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(GPU / "rebuild.sh"), *args], env={**fake["env"], **env},
                          capture_output=True, text=True, timeout=120)


def builds(fake: dict) -> int:
    return len(fake["counter"].read_text().splitlines()) if fake["counter"].exists() else 0


def records(fake: dict) -> list[dict]:
    path = fake["tmp"] / "rebuild_times.json"
    return json.loads(path.read_text())["rebuilds"] if path.exists() else []


def test_rebuild_on_a_fresh_pod_clones_applies_builds_and_records_the_time(fake_pod: dict) -> None:
    run = rebuild(fake_pod, "block")
    assert run.returncode == 0, run.stdout + run.stderr
    tree = fake_pod["tree"]
    assert (tree / "build.txt").read_text() == "ninja\n"           # build-enabling hunk applied
    assert "tile = 32" in (tree / KERNEL_FILE).read_text()         # the lever
    assert "stages = 1" in (tree / KERNEL_FILE).read_text()        # the lever patch carries the known fix itself
    assert builds(fake_pod) == 1
    [rec] = records(fake_pod)
    assert rec["arm"] == "block" and rec["patch"] == "unified-attention-block.patch"
    assert rec["patch_sha256"] == __import__("hashlib").sha256(
        (fake_pod["patches"] / "unified-attention-block.patch").read_bytes()).hexdigest()
    assert isinstance(rec["wall_s"], float) and rec["wall_s"] >= 0 and rec["gfx"] == "gfx1151"
    assert "### rebuild block wall_s=" in run.stdout


def test_rebuild_skips_when_the_patch_is_unchanged_and_rebuilds_when_it_changes(fake_pod: dict) -> None:
    assert rebuild(fake_pod, "block").returncode == 0
    again = rebuild(fake_pod, "block")
    assert again.returncode == 0 and "up to date" in again.stdout
    assert builds(fake_pod) == 1 and len(records(fake_pod)) == 1
    patch = fake_pod["patches"] / "unified-attention-block.patch"
    patch.write_text(patch.read_text().replace("tile = 32", "tile = 64"))
    changed = rebuild(fake_pod, "block")
    assert changed.returncode == 0, changed.stdout
    assert builds(fake_pod) == 2 and len(records(fake_pod)) == 2
    assert "tile = 64" in (fake_pod["tree"] / KERNEL_FILE).read_text()
    assert rebuild(fake_pod, "block", "--force").returncode == 0 and builds(fake_pod) == 3


def test_rebuild_stock_applies_the_whole_base_diff(fake_pod: dict) -> None:
    run = rebuild(fake_pod, "stock")
    assert run.returncode == 0, run.stdout + run.stderr
    text = (fake_pod["tree"] / KERNEL_FILE).read_text()
    assert "stages = 1" in text and "tile = 16" in text
    assert records(fake_pod)[0]["patch_sha256"] == "none"
    # switching arms is a patch change, so it rebuilds
    assert rebuild(fake_pod, "block").returncode == 0 and builds(fake_pod) == 2


def test_rebuild_stops_on_a_rejected_hunk_and_prints_it(fake_pod: dict) -> None:
    assert rebuild(fake_pod, "block").returncode == 0
    bad = fake_pod["patches"] / "unified-attention-bad.patch"
    bad.write_text((fake_pod["patches"] / "unified-attention-block.patch").read_text()
                   .replace("-    tile = 16", "-    tile = 99").replace(" def launch():", " def nowhere():"))
    run = rebuild(fake_pod, "bad")
    assert run.returncode == 3, run.stdout + run.stderr
    assert "REJECTED" in run.stdout and "--- rejected hunk in" in run.stdout and "-    tile = 99" in run.stdout
    assert builds(fake_pod) == 1 and len(records(fake_pod)) == 1      # nothing built, nothing recorded
    tree = fake_pod["tree"]
    assert git("status", "--porcelain", cwd=tree).stdout.strip() == ""  # tree restored, no .rej left
    # the stamp still says block is built, so block is not rebuilt
    assert "up to date" in rebuild(fake_pod, "block").stdout


def test_rebuild_failed_build_exits_nonzero_and_keeps_the_old_stamp(fake_pod: dict) -> None:
    run = rebuild(fake_pod, "block", R12_BUILD_CMD="echo compiler error; exit 1")
    assert run.returncode == 4 and "BUILD FAILED" in run.stdout
    assert records(fake_pod) == []
    assert rebuild(fake_pod, "block").returncode == 0 and builds(fake_pod) == 1


def test_rebuild_refuses_an_unknown_patch(fake_pod: dict) -> None:
    run = rebuild(fake_pod, "nosuch")
    assert run.returncode == 2 and "no patch file" in run.stdout


# --- part 3: the LDS recorder -----------------------------------------------------------------------------------

def test_lds_reads_triton_metadata_shared_and_skips_group_files() -> None:
    from gpu import lds_check
    rows = lds_check.read_triton_cache(FIX / "triton_cache_stages_synthetic")
    by_name = {r["kernel"]: r for r in rows}
    assert set(by_name) == {"kernel_unified_attention_2d", "kernel_unified_attention_3d", "reduce_segments",
                            "_fwd_kernel_rotary"}
    k3 = by_name["kernel_unified_attention_3d"]
    assert k3["lds_bytes"] == 33792 and k3["num_stages"] == 2 and k3["arch"] == "gfx1151"
    assert k3["source"] == "triton.metadata.shared"


def test_lds_reads_group_segment_fixed_size_from_code_object_notes() -> None:
    from gpu import lds_check
    rows = lds_check.parse_readelf_notes((FIX / "readelf_notes_synthetic.txt").read_text(), path="x.hsaco")
    assert [(r["kernel"], r["lds_bytes"]) for r in rows] == [("kernel_unified_attention_3d", 33792),
                                                             ("wvSplitK_hf_sml_", 69632)]
    assert {r["source"] for r in rows} == {".group_segment_fixed_size"}
    assert rows[0]["arch"] == "gfx1151"


def test_lds_resource_failures_are_read_from_the_serve_log() -> None:
    from gpu import lds_check
    fails = lds_check.resource_failures((FIX / "serve_oor_synthetic.log").read_text())
    assert [(f["required_bytes"], f["limit_bytes"], f["kernel"]) for f in fails] == [
        (66560, 65536, None), (69632, 65536, "wvSplitK_hf_sml_")]


def test_lds_check_passes_the_stages_arm_and_writes_the_record(tmp_path: Path) -> None:
    out = tmp_path / "lds.json"
    run = subprocess.run([sys.executable, str(GPU / "lds_check.py"), "--arm", "stages",
                          "--triton-cache", str(FIX / "triton_cache_stages_synthetic"), "--out", str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    rec = json.loads(out.read_text())
    assert rec["limit_bytes"] == 65536 and rec["pass"] is True and rec["failures"] == 0
    assert rec["touched"] == ["kernel_unified_attention_3d"] and rec["missing"] == []
    assert all(k["fits"] for k in rec["kernels"])
    assert json.loads(run.stdout.strip().splitlines()[-1])["pass"] is True


def test_lds_check_counts_over_limit_missing_and_resource_failures_as_failures(tmp_path: Path) -> None:
    out = tmp_path / "lds.json"
    run = subprocess.run([sys.executable, str(GPU / "lds_check.py"), "--arm", "block",
                          "--triton-cache", str(FIX / "triton_cache_over_synthetic"),
                          "--code-object-notes", str(FIX / "readelf_notes_synthetic.txt"),
                          "--serve-log", str(FIX / "serve_oor_synthetic.log"), "--out", str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 1, run.stdout + run.stderr
    rec = json.loads(out.read_text())
    # 3D over the limit (triton), the HIP kernel over the limit, reduce_segments not compiled, two log failures
    assert rec["pass"] is False and rec["failures"] == 5
    assert rec["missing"] == ["reduce_segments"]
    assert sorted(k["kernel"] for k in rec["kernels"] if not k["fits"]) == ["kernel_unified_attention_3d",
                                                                            "wvSplitK_hf_sml_"]


def test_lds_check_refuses_a_missing_cache(tmp_path: Path) -> None:
    run = subprocess.run([sys.executable, str(GPU / "lds_check.py"), "--arm", "stages",
                          "--triton-cache", str(tmp_path / "nope")], capture_output=True, text=True)
    assert run.returncode == 2
