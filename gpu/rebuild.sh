#!/bin/bash
# R12: rebuild vLLM on a pod from the build-enabling hunks of lab/vllm-gfx1201.diff plus one named kernel patch.
#
#   bash gpu/rebuild.sh stages|block|stock|<path to a .patch> [--force]
#
# stages and block are gpu/patches/unified-attention-<name>.patch, one lever each. They are applied on top of the
# base diff WITHOUT its kernel file (git apply --exclude), so the build-enabling hunks and the kernel lever stay
# apart. stock applies the whole base diff, the teacher's build today (its num_stages=1 hunk included).
#
# What it does, in order:
#   1. Works out the build key: vLLM commit, sha256 of the base diff, sha256 of the patch ("none" for stock).
#      If the stamp file holds the same key, nothing is rebuilt (exit 0) unless --force.
#   2. Clones vLLM when $VLLM_DIR is missing (a fresh pod), checks out the commit, restores every tracked file.
#   3. Applies the base diff, then the patch, each with --reject. A rejected hunk stops the recipe: the hunk is
#      printed from its .rej file, the tree is restored, nothing is built and the stamp is left alone (exit 3).
#   4. Runs the build command and times it. A failed build exits 4 and leaves the stamp alone.
#   5. Writes the stamp and appends one record (arm, key, wall-clock seconds, start time, host, gfx) to the JSON file.
#
# Settings (environment, all optional):
#   VLLM_DIR          the checkout (default ~/vllm, as lab/pod_a_teacher.sh)
#   VLLM_REPO         where to clone from (default https://github.com/vllm-project/vllm.git)
#   VLLM_COMMIT       default b1388b1fbf5aaef47937fabe98931211684666a6 (tag v0.19.1, built as 0.19.2+<gfx>)
#   R12_BASE_DIFF     default lab/vllm-gfx1201.diff in this repo
#   R12_PATCH_DIR     default gpu/patches in this repo
#   R12_STATE_DIR     where the stamp lives (default ~/.cache/r12)
#   R12_REBUILD_JSON  the wall-clock record (default gpu/results/rebuild_times.json in this repo)
#   R12_BUILD_CMD     the build, run with bash -c inside $VLLM_DIR (default: the editable ROCm install below)
#   GFX               the gfx target (default: the first one rocminfo lists)
# Exit codes: 0 built or already up to date, 2 usage, 3 rejected hunk, 4 build failed, 5 build tools missing.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL_FILE="vllm/v1/attention/ops/triton_unified_attention.py"
VLLM_DIR="${VLLM_DIR:-$HOME/vllm}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-b1388b1fbf5aaef47937fabe98931211684666a6}"
BASE_DIFF="${R12_BASE_DIFF:-$REPO/lab/vllm-gfx1201.diff}"
PATCH_DIR="${R12_PATCH_DIR:-$REPO/gpu/patches}"
STATE_DIR="${R12_STATE_DIR:-$HOME/.cache/r12}"
REBUILD_JSON="${R12_REBUILD_JSON:-$REPO/gpu/results/rebuild_times.json}"
STAMP="$STATE_DIR/vllm-build.stamp"

ARM="${1:-}"
FORCE=0
[ "${2:-}" = "--force" ] && FORCE=1
case "$ARM" in
  "" | -h | --help) sed -n '2,30p' "$0"; exit 2 ;;
  stock) PATCH_FILE="" ;;
  *.patch) PATCH_FILE="$(cd "$(dirname "$ARM")" && pwd)/$(basename "$ARM")"; ARM="$(basename "$ARM" .patch)" ;;
  *) PATCH_FILE="$PATCH_DIR/unified-attention-$ARM.patch" ;;
esac
if [ -n "$PATCH_FILE" ] && [ ! -f "$PATCH_FILE" ]; then echo "no patch file $PATCH_FILE"; exit 2; fi
[ -f "$BASE_DIFF" ] || { echo "no base diff $BASE_DIFF"; exit 2; }

sha() { sha256sum "$1" | cut -d' ' -f1; }
gfx_target() {
  if [ -n "${GFX:-}" ]; then echo "$GFX"; return; fi
  rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]\+'
}

DIFF_SHA="$(sha "$BASE_DIFF")"
PATCH_SHA="none"
[ -n "$PATCH_FILE" ] && PATCH_SHA="$(sha "$PATCH_FILE")"
KEY="commit=$VLLM_COMMIT diff=$DIFF_SHA patch=$PATCH_SHA"

if [ "$FORCE" = 0 ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$KEY" ]; then
  echo "up to date: $ARM ($KEY); nothing rebuilt"
  exit 0
fi

default_build() {
  # The editable ROCm install of lab/pod_a_teacher.sh build(), against the pod's own torch stack.
  local gfx jobs constraints
  command -v cmake >/dev/null && command -v ninja >/dev/null || {
    echo "cmake or ninja missing: run 'bash lab/pod_a_teacher.sh build' once on a fresh pod (it installs the build deps)"
    return 5; }
  gfx="$(gfx_target)"
  [ -n "$gfx" ] || { echo "no gfx target from rocminfo; set GFX=<target>"; return 5; }
  jobs="$(awk '/MemTotal/ {j=int($2/1048576/3); print (j<1?1:(j>48?48:j))}' /proc/meminfo)"
  [ "$jobs" -gt "$(nproc)" ] && jobs="$(nproc)"
  constraints="$STATE_DIR/torch-constraints.txt"
  python3 -c 'from importlib import metadata as m
for n in ("torch","torchvision","torchaudio","triton","pytorch-triton-rocm"):
    try: print(f"{n}=={m.version(n)}")
    except m.PackageNotFoundError: pass' > "$constraints"
  python3 use_existing_torch.py
  SETUPTOOLS_SCM_PRETEND_VERSION="0.19.2+$gfx" VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH="$gfx" \
    MAX_JOBS="$jobs" CMAKE_BUILD_PARALLEL_LEVEL="$jobs" PIP_CONSTRAINT="$constraints" \
    pip install --user --no-build-isolation -e . > "$STATE_DIR/build.log" 2>&1
  local rc=$?
  tail -3 "$STATE_DIR/build.log"
  return $rc
}

mkdir -p "$STATE_DIR"
if [ ! -d "$VLLM_DIR/.git" ]; then
  echo "cloning $VLLM_REPO into $VLLM_DIR"
  git clone -q "$VLLM_REPO" "$VLLM_DIR" || { echo "clone failed"; exit 4; }
fi
cd "$VLLM_DIR" || exit 4
git checkout -q "$VLLM_COMMIT" 2>/dev/null || { git fetch -q origin "$VLLM_COMMIT" && git checkout -q "$VLLM_COMMIT"; } \
  || { echo "cannot check out $VLLM_COMMIT"; exit 4; }
git checkout -q -- .
find . -name '*.rej' -not -path './.git/*' -delete

apply_or_stop() {
  # git apply --reject applies what it can and writes each rejected hunk to <file>.rej; any .rej stops the recipe.
  local label="$1"; shift
  if git apply --reject --whitespace=nowarn "$@" 2>"$STATE_DIR/apply.err"; then return 0; fi
  echo "REJECTED: $label does not apply to $VLLM_COMMIT"
  cat "$STATE_DIR/apply.err"
  local rej
  while IFS= read -r rej; do echo "--- rejected hunk in ${rej%.rej}"; cat "$rej"; rm -f "$rej"; done \
    < <(find . -name '*.rej' -not -path './.git/*')
  git checkout -q -- .
  exit 3
}

if [ -z "$PATCH_FILE" ]; then
  apply_or_stop "$(basename "$BASE_DIFF")" "$BASE_DIFF"
else
  apply_or_stop "$(basename "$BASE_DIFF") (build-enabling hunks)" --exclude="$KERNEL_FILE" "$BASE_DIFF"
  apply_or_stop "$(basename "$PATCH_FILE")" "$PATCH_FILE"
fi
echo "applied: $ARM ($KEY)"

STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0="$(date +%s.%N)"
if [ -n "${R12_BUILD_CMD:-}" ]; then bash -c "$R12_BUILD_CMD"; else default_build; fi
RC=$?
WALL="$(python3 -c 'import sys; print(round(float(sys.argv[1]) - float(sys.argv[2]), 3))' "$(date +%s.%N)" "$T0")"
if [ "$RC" != 0 ]; then
  echo "BUILD FAILED: rc=$RC after ${WALL}s; stamp left as it was"
  [ "$RC" = 5 ] && exit 5
  exit 4
fi
echo "$KEY" > "$STAMP"

mkdir -p "$(dirname "$REBUILD_JSON")"
ARM="$ARM" KEY_COMMIT="$VLLM_COMMIT" DIFF_SHA="$DIFF_SHA" PATCH_SHA="$PATCH_SHA" STARTED="$STARTED" WALL="$WALL" \
PATCH_NAME="$(basename "${PATCH_FILE:-none}")" GFX_NOW="$(gfx_target)" OUT="$REBUILD_JSON" python3 - <<'PY'
import json, os, socket
path = os.environ["OUT"]
try:
    with open(path) as fh:
        data = json.load(fh)
except (OSError, ValueError):
    data = {"schema": "gpu.r12.rebuild/1", "rebuilds": []}
data["rebuilds"].append({
    "arm": os.environ["ARM"], "patch": os.environ["PATCH_NAME"], "patch_sha256": os.environ["PATCH_SHA"],
    "base_diff_sha256": os.environ["DIFF_SHA"], "vllm_commit": os.environ["KEY_COMMIT"],
    "started_utc": os.environ["STARTED"], "wall_s": round(float(os.environ["WALL"]), 3),
    "host": socket.gethostname(), "gfx": os.environ["GFX_NOW"] or None})
with open(path, "w") as fh:
    json.dump(data, fh, indent=1)
    fh.write("\n")
PY
echo "### rebuild $ARM wall_s=$WALL rc=0 (recorded in $REBUILD_JSON)"
