#!/bin/bash
# Teacher vLLM upgrade on a Strix Halo pod (gfx1151, ROCm 7.13, torch 2.9.1+rocm7.13.0): build a newer vLLM that
# carries the RDNA W4A16 decode kernel (RDNAHybridW4A16LinearKernel, vLLM #40977 and follow-ups) into its OWN tree
# (~/vllm-new) and its OWN venv (~/venv-vllm-new), so the running teacher (~/vllm, user site) is never touched while
# it builds. The audit behind the pin and the patch rump is lab/refine/research/kernels/vllm_upgrade_gfx1151.md (R59).
#
# Usage on the pod:  bash lab/vllm_upgrade.sh [build|status|verify|lds|serve|restore|models]
#   build    clone + patch + venv + compile in the background at nice 19 and MAX_JOBS<=8; returns at once.
#            Progress: ~/vllm_new_build.log, final line in ~/vllm_new_build.status ("ok" or "fail <step>").
#   status   the status line and the build log's last lines.
#   verify   import check in the new venv: version, commit, compiled extensions, platform, W4A16 kernel list.
#   lds      apply lab/patches/vllm-main-gfx1151-num-stages.diff (only when the LDS check at first start fires).
#   serve    stop the running teacher and serve the SAME model, names and flags from the new venv (port 8000).
#   restore  stop it and bring back the old teacher with lab/pod_a_teacher.sh serve.
# serve and restore stop the teacher: the caller checks the quiet window first (no client, 0 running, 0 waiting).
set -u -o pipefail
STEP="${1:-status}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-5281e4990}"        # main 2026-10-07 (#52619); v0.31.0 is the tagged fallback
VLLM_DIR="${VLLM_DIR:-$HOME/vllm-new}"
VENV="${VENV:-$HOME/venv-vllm-new}"
JOBS="${JOBS:-8}"                              # the teacher keeps the rest of the 32 threads
PATCH_DIR="${PATCH_DIR:-$HERE/patches}"
PLATFORM_PATCH="$PATCH_DIR/vllm-main-gfx1151-platform.diff"
LDS_PATCH="$PATCH_DIR/vllm-main-gfx1151-num-stages.diff"
LOG="$HOME/vllm_new_build.log"
STATUS="$HOME/vllm_new_build.status"
MODELS="$HOME/models"
TEACHER="${TEACHER:-RedHatAI/Qwen3.8-27B-INT4}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
SERVE_DIR="$HOME/serve"
SERVE_LOG="${SERVE_LOG:-$SERVE_DIR/vllm-new.log}"
# Runtime requirements left out of the no-deps install: optional on this serve line, and several pin torch or
# pull large native stacks (tilelang, mooncake, amd-quark, runai, tensorizer, fastsafetensors, gRPC, test tools).
SKIP_REQS='^(grpcio|grpcio-reflection|datasets|peft|pytest-asyncio|tensorizer|runai-model-streamer|amd-quark|tilelang|apache-tvm-ffi|fastsafetensors|mooncake-transfer-engine-rocm)\b'

build_env() {
  # The venv's python and pip first, then ~/.local/bin for the pkg-config symlink lab/pod_a_teacher.sh left there;
  # the libdrm stub it built satisfies torch's LoadHIP.cmake the same way.
  export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
  export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:/opt/rocm/lib/pkgconfig"
  export LIBRARY_PATH="$HOME/.local/lib"
}

build_inner() {
  local gfx t0 version
  t0=$(date +%s)
  gfx="${GFX:-$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]\+')}"
  echo "### start $(date -u +%FT%TZ) commit=$VLLM_COMMIT gfx=$gfx jobs=$JOBS"
  [ -n "$gfx" ] || { echo "fail rocminfo" > "$STATUS"; return 1; }
  echo "### step clone"
  [ -d "$VLLM_DIR/.git" ] || git clone -q "$VLLM_REPO" "$VLLM_DIR" || { echo "fail clone" > "$STATUS"; return 1; }
  cd "$VLLM_DIR" || return 1
  git fetch -q --tags origin 2>/dev/null
  git checkout -q -f "$VLLM_COMMIT" && git clean -fdq -e '*.so' || { echo "fail checkout" > "$STATUS"; return 1; }
  echo "vllm_commit: $(git rev-parse --short=9 HEAD) $(git log -1 --format=%cs)"
  echo "### step patch"
  # The rump of lab/vllm-gfx1201.diff on main: only the platform fallback (H5) still needs an edit; H1 to H4 are
  # what tools/use_existing_torch.py strips, H6 and H9 are upstream. The num_stages guard waits for the LDS check.
  git apply --whitespace=nowarn "$PLATFORM_PATCH" || { echo "fail patch" > "$STATUS"; return 1; }
  if [ -f tools/use_existing_torch.py ]; then python3 tools/use_existing_torch.py; else python3 use_existing_torch.py; fi \
    || { echo "fail use_existing_torch" > "$STATUS"; return 1; }
  git diff --stat | tail -1
  echo "### step venv"
  # The image has no ensurepip, so the venv is made without pip and uses the system pip module, which installs into
  # the venv because sys.prefix is the venv. --system-site-packages keeps the image's torch, Triton and numpy;
  # the user site (where the old vLLM lives) is not visible inside a venv.
  [ -x "$VENV/bin/python" ] || python3 -m venv --without-pip --system-site-packages "$VENV" || { echo "fail venv" > "$STATUS"; return 1; }
  build_env
  python -m pip --version
  python - > "$HOME/.cache/vllm-new-constraints.txt" <<'PY' || { echo "fail constraints" > "$STATUS"; return 1; }
from importlib import metadata
for name in ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm"):
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
PY
  cat "$HOME/.cache/vllm-new-constraints.txt"
  export PIP_CONSTRAINT="$HOME/.cache/vllm-new-constraints.txt"
  python -m pip install --quiet "cmake>=3.26.1,<4" ninja "packaging>=24.2" "setuptools>=78.1.1,<80" "setuptools-scm>=8" \
    "setuptools-rust>=1.9.0" wheel "jinja2>=3.1.6" regex || { echo "fail build-tools" > "$STATUS"; return 1; }
  echo "### step compile"
  version="0.31.2.dev0+g$(git rev-parse --short=9 HEAD).$gfx"
  # --no-deps: the runtime requirements go in next, filtered; a no-isolation build against the image's torch.
  SETUPTOOLS_SCM_PRETEND_VERSION="$version" VLLM_RS_BUILD_VERSION="$version" VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH="$gfx" MAX_JOBS="$JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
    nice -n 19 python -m pip install --no-build-isolation --no-deps -v -e "$VLLM_DIR" \
    || { echo "fail compile" > "$STATUS"; return 1; }
  echo "### step runtime-deps"
  { grep -v -E '^\s*(#|-r|$)' requirements/common.txt; grep -v -E '^\s*(#|-r|$)' requirements/rocm.txt; } \
    | grep -v -E "$SKIP_REQS" > "$HOME/.cache/vllm-new-reqs.txt"
  nice -n 19 python -m pip install --quiet -r "$HOME/.cache/vllm-new-reqs.txt" || { echo "fail runtime-deps" > "$STATUS"; return 1; }
  # amdsmi in the venv would load a second libamd_smi and hide the GPU from torch (lab/README.md); keep it out.
  python -m pip uninstall -y --quiet amdsmi 2>/dev/null
  python -c "import torch; assert torch.__version__ == '$(sed -n 's/^torch==//p' "$PIP_CONSTRAINT")', torch.__version__; print('torch kept:', torch.__version__)" \
    || { echo "fail torch-changed" > "$STATUS"; return 1; }
  echo "### build wall_s=$(( $(date +%s) - t0 ))"
  echo "ok $(git rev-parse --short=9 HEAD) wall_s=$(( $(date +%s) - t0 ))" > "$STATUS"
}

build() {
  if pgrep -f "vllm_upgrade.sh build-inner" >/dev/null; then echo "a build is already running"; return 0; fi
  mkdir -p "$HOME/.cache"
  echo "running" > "$STATUS"
  # setsid + nohup: the build outlives the terminal that started it.
  setsid nohup bash "$0" build-inner > "$LOG" 2>&1 < /dev/null &
  echo "build started in the background, pid $!; log $LOG, status $STATUS"
}

status() {
  echo "status: $(cat "$STATUS" 2>/dev/null || echo none)"
  grep '^### ' "$LOG" 2>/dev/null | tail -3
  tail -n "${N:-5}" "$LOG" 2>/dev/null | cut -c1-240
}

verify() {
  build_env
  mkdir -p "$SERVE_DIR" && cd "$SERVE_DIR"
  echo "vllm_commit: $(git -C "$VLLM_DIR" rev-parse --short=9 HEAD) patched: $(git -C "$VLLM_DIR" diff --name-only | tr '\n' ' ')"
  python - <<'PY'
import importlib, torch, triton, vllm
print(f"vllm: {vllm.__version__} from {vllm.__file__}")
print(f"torch: {torch.__version__} triton: {triton.__version__}")
for name in ("vllm._C", "vllm._rocm_C", "vllm._moe_C", "vllm._C_stable_libtorch"):
    try:
        importlib.import_module(name); print(f"{name}: imports")
    except Exception as exc:
        print(f"{name}: FAILS {type(exc).__name__}: {exc}")
from vllm.platforms import current_platform
print(f"platform: {type(current_platform).__name__}, torch sees {torch.cuda.device_count()} GPU")
try:
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import RDNAHybridW4A16LinearKernel
    print("RDNAHybridW4A16LinearKernel: present")
except Exception as exc:
    print(f"RDNAHybridW4A16LinearKernel: FAILS {type(exc).__name__}: {exc}")
try:
    from vllm.tool_parsers import ToolParserManager
    print("qwen3_coder parser:", "qwen3_coder" in ToolParserManager.list_registered())
except Exception as exc:
    print(f"tool parsers: {type(exc).__name__}: {exc}")
PY
}

lds() {
  cd "$VLLM_DIR" || return 1
  git apply --check "$LDS_PATCH" 2>/dev/null || { echo "num_stages guard already applied or does not apply"; git diff --stat; return 0; }
  git apply "$LDS_PATCH" && echo "num_stages guard applied (Python only, no rebuild)"
}

stop_server() {
  pkill -f "vllm.entrypoints" 2>/dev/null || { echo "no teacher server running"; return 0; }
  for i in $(seq 1 30); do
    pgrep -f "vllm.entrypoints" >/dev/null || { echo "teacher stopped"; return 0; }
    sleep 2
  done
  pkill -9 -f "vllm.entrypoints" 2>/dev/null; sleep 2
  pgrep -f "vllm.entrypoints" >/dev/null && { echo "teacher still running"; return 1; }
  echo "teacher stopped (killed)"
}

serve() {
  # The serve line of lab/pod_a_teacher.sh, unchanged except for the venv's python: two names, priority
  # scheduling, the tool flags the labeler needs, the 16k context, one image, bf16, Triton encoder attention.
  local dir="$MODELS/$(basename "$TEACHER")" pid t0
  [ -x "$VENV/bin/python" ] || { echo "no venv at $VENV: build first"; return 1; }
  build_env
  mkdir -p "$SERVE_DIR" && cd "$SERVE_DIR"
  stop_server || return 1
  t0=$(date +%s)
  date -u +%FT%TZ > "$SERVE_DIR/serve_new_started_at"
  nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$dir" --served-model-name teacher group-brain --port "$PORT" --host 127.0.0.1 \
    --scheduling-policy priority \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
    --max-model-len "$MAX_LEN" --limit-mm-per-prompt '{"image":1}' --gpu-memory-utilization "$GPU_UTIL" \
    --dtype bfloat16 --mm-encoder-attn-backend TRITON_ATTN > "$SERVE_LOG" 2>&1 &
  pid=$!
  echo "new vllm serve pid $pid (log: $SERVE_LOG)"
  for i in $(seq 1 240); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5)" 2>/dev/null \
      && { echo "teacher up on $PORT (new build) after $(( $(date +%s) - t0 )) s"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "the new server ended before it answered; see $SERVE_LOG"; tail -30 "$SERVE_LOG"; return 1; }
    sleep 5
  done
  echo "new server did not come up in 20 min; see $SERVE_LOG"; tail -30 "$SERVE_LOG"
  return 1
}

restore() {
  # The old teacher exactly as the lab serves it (its serve step stops whatever runs on the port first).
  local script="$HERE/pod_a_teacher.sh"
  [ -f "$script" ] || script="$HOME/Knighthacks26/lab/pod_a_teacher.sh"
  bash "$script" serve
}

models() {
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=10).read().decode())"
}

case "$STEP" in
  build) build ;;
  build-inner) build_inner ;;
  status) status ;;
  verify) verify ;;
  lds) lds ;;
  serve) serve ;;
  restore) restore ;;
  models) models ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
