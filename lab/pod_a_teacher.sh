#!/bin/bash
# Teacher pod: build vLLM from source for the pod's GPU (ROCm 7.13) and serve Qwen3.8-27B behind the hub proxy.
# The GPU target comes from rocminfo, so the same steps build for the R9700 (gfx1201, DUB accounts A and B) and for
# the Strix Halo (gfx1151, SJC accounts C and D); GFX=<target> overrides it.
# Idempotent and resumable: re-run after a pod relaunch; finished steps are skipped.
# Usage on the pod (JupyterLab terminal):  bash lab/pod_a_teacher.sh [facts|proxy|build|verify|download|serve|models|stop|all]
#   serve takes --speculative off|ngram|draft (see serve below).
# From the box, with nothing typed on the pod (the patch file is uploaded next to the script):
#   python -m lab.controller run-script --account D --with lab/vllm-gfx1151.diff lab/pod_a_teacher.sh build
# Beside a serving teacher, into its own folder and venv (the serving build is not touched):
#   python -m lab.controller run-script --account C --timeout 3600 --with lab/vllm-gfx1151.diff lab/pod_a_teacher.sh \
#     build --vllm-dir /home/jovyan/vllm-hybrid --venv /home/jovyan/venv-hybrid
# The same names work as environment variables (VLLM_DIR, VLLM_VENV, PATCH, BUILD_JOBS) when the script is run by hand.
# facts prints the pod's hour-0 facts and changes nothing; verify checks a finished build; models asks the local server for /v1/models; stop ends
# the server. build, download and serve end with a line "### <step> wall_s=<seconds> rc=<code>" for the ledger.
# Callers reach the server at https://<hub>/user/<account>/proxy/8000/v1/... with header
# "Authorization: token <hub API token>" (jupyter-server-proxy must be installed once: step "proxy").
# A step that fails ends the script with a non-zero code, so run-script's ok and its ledger row tell the truth.
set -u -o pipefail
export PATH="$HOME/.local/bin:$PATH"
STEP="${1:-all}"
[ $# -gt 0 ] && shift
# Options after the step (run-script passes them as arguments): --vllm-dir DIR, --venv DIR, --patch FILE, --jobs N,
# --speculative off|ngram|draft (serve only).
while [ $# -gt 0 ]; do
  case "$1" in
    --vllm-dir) VLLM_DIR="${2:-}"; shift 2 ;;
    --venv) VLLM_VENV="${2:-}"; shift 2 ;;
    --patch) PATCH="${2:-}"; shift 2 ;;
    --jobs) BUILD_JOBS="${2:-}"; shift 2 ;;
    --speculative) [ $# -ge 2 ] || { echo "--speculative needs off, ngram or draft"; exit 2; }; SPECULATIVE="$2"; shift 2 ;;
    --speculative=*) SPECULATIVE="${1#*=}"; shift ;;
    *) echo "unknown option $1 (known: --vllm-dir, --venv, --patch, --jobs, --speculative)"; exit 2 ;;
  esac
done
DEFAULT_VLLM_DIR="$HOME/vllm"
VLLM_DIR="${VLLM_DIR:-$DEFAULT_VLLM_DIR}"
# Empty: vLLM is installed into the user site (the teacher's usual build). A path: a separate venv that sees the
# image's torch (--system-site-packages) and nothing of the user site, so a second build can sit beside the first.
VLLM_VENV="${VLLM_VENV:-}"
VLLM_COMMIT="${VLLM_COMMIT:-b1388b1fbf5aaef47937fabe98931211684666a6}"   # April 2026 main; builds against torch 2.9.1
# The patch sits beside this script: in lab/ of the clone, or in the Jupyter root when run-script uploads it with
# --with. vllm-gfx1151.diff (the April patch plus the RDNA hybrid W4A16 kernel) is the default;
# PATCH=vllm-gfx1201.diff (or a full path) selects the April patch alone. The clone at ~/Knighthacks26/lab is tried last.
PATCH="${PATCH:-vllm-gfx1151.diff}"
case "$PATCH" in
  /*) ;;
  *) if [ -f "$PATCH" ]; then PATCH="$(pwd)/$PATCH"
     elif [ -f "$(cd "$(dirname "$0")" && pwd)/$(basename "$PATCH")" ]; then PATCH="$(cd "$(dirname "$0")" && pwd)/$(basename "$PATCH")"
     elif [ -f "$HOME/Knighthacks26/lab/$(basename "$PATCH")" ]; then PATCH="$HOME/Knighthacks26/lab/$(basename "$PATCH")"
     else PATCH="$(cd "$(dirname "$0")" && pwd)/$(basename "$PATCH")"
     fi ;;
esac
PY=python3
[ -n "$VLLM_VENV" ] && PY="$VLLM_VENV/bin/python"
BUILD_LOG="$HOME/vllm_build.log"
[ "$VLLM_DIR" = "$DEFAULT_VLLM_DIR" ] || BUILD_LOG="$HOME/vllm_build-$(basename "$VLLM_DIR").log"
STAMP="$VLLM_DIR/.atlas-build-stamp"
STEP_MARKS=""                                 # extra key=value fields a step adds to its "### <step>" line
MODELS="$HOME/models"
# RedHatAI/Qwen3.8-27B-INT4 (compressed-tensors W4A16, 18 GB) is what serves on this vLLM commit.
# amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 (Aug 2026 Quark export) fails in this commit's Quark weight mapper; see lab/README.md.
TEACHER="${TEACHER:-RedHatAI/Qwen3.8-27B-INT4}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
# Tool calling: without both flags vLLM returns a tool call as plain text in content and the labeler rejects it. The checkpoint's
# chat template asks for <tool_call><function=...><parameter=...> blocks (the Qwen3-Coder XML format), which is parser
# qwen3_coder at b1388b1; hermes expects JSON inside <tool_call> and would not parse them.
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
SERVE_DIR="$HOME/serve"
SERVE_LOG="$SERVE_DIR/vllm.log"

timed() {
  # Run a step and print its wall-clock, so the caller can write it to the ledger without a stopwatch.
  local t0 rc
  t0=$(date +%s)
  "$@"; rc=$?
  echo "### $1 wall_s=$(( $(date +%s) - t0 )) rc=$rc${STEP_MARKS:+ $STEP_MARKS}"
  return $rc
}

facts() {
  # Hour-0 facts, read-only: what the task pages record beside R2's and R29's tables.
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gfx: $(gpu_arch)"
  echo "gpu: $(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*Marketing Name: *//')"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //'), $(nproc) threads"
  echo "mem: $(awk '/MemTotal/ {t=$2} /MemAvailable/ {a=$2} END {printf "%.0f GB total, %.0f GB available", t/1048576, a/1048576}' /proc/meminfo)"
  echo "os: $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME"), kernel $(uname -r)"
  echo "hip: $(hipconfig --version 2>/dev/null || echo none)"
  python3 - <<'PY'
import importlib, importlib.util
for name in ("torch", "triton", "numpy", "vllm", "transformers", "huggingface_hub", "amdsmi", "jupyter_server_proxy"):
    if importlib.util.find_spec(name) is None:
        print(f"{name}: absent")
        continue
    try:
        print(f"{name}: {getattr(importlib.import_module(name), '__version__', 'present')}")
    except Exception as exc:  # a broken install is a fact too
        print(f"{name}: import fails ({type(exc).__name__})")
try:
    import torch
    print(f"torch_gpu: available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
          + (f" mem_gib={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}" if torch.cuda.is_available() else ""))
except Exception as exc:
    print(f"torch_gpu: {type(exc).__name__}")
PY
  local tool tools=""
  for tool in hipcc gcc git wget curl cmake ninja pkg-config hf amd-smi rocm-smi; do
    command -v "$tool" >/dev/null 2>&1 && tools="$tools $tool" || tools="$tools no-$tool"
  done
  echo "tools:$tools"
  echo "home: $HOME ($(df -h "$HOME" | awk 'NR==2 {print $2 " size, " $4 " free, " $1}'))"
  echo "cwd: $(pwd) ($(df -h . | awk 'NR==2 {print $4 " free, " $1}'))"
  echo "home_holds: $(ls -A "$HOME" 2>/dev/null | tr '\n' ' ')"
  # No ss or netstat in the image: try the ports the lab uses.
  echo "listeners: $(python3 -c "import socket
for port in (8000, 8099, 8790, 8888):
    s = socket.socket(); s.settimeout(1)
    print(port if s.connect_ex(('127.0.0.1', port)) == 0 else 'no-%d' % port, end=' ')
    s.close()")"
  local code host
  for host in github.com huggingface.co pypi.org repo.radeon.com; do
    code=$(python3 -c "import urllib.request,sys
try: print(urllib.request.urlopen('https://$host', timeout=10).status)
except Exception as e: print(getattr(e, 'code', type(e).__name__))" 2>/dev/null)
    echo "egress $host: $code"
  done
  # Idle watts: SOCKET_POWER is the whole APU package; ten reads one second apart, the first can be high.
  local i watts=""
  for i in 1 2 3 4 5 6 7 8 9 10; do
    watts="$watts $(amd-smi metric --power 2>/dev/null | grep -m1 -o 'SOCKET_POWER: *[0-9.]\+' | grep -o '[0-9.]\+$')"
    sleep 1
  done
  echo "idle_socket_power_w:$watts"
  echo "server_running: $(pgrep -f 'vllm.entrypoints' >/dev/null && echo yes || echo no)"
}

gpu_arch() {
  # The first gfx target rocminfo lists: gfx1201 on the R9700, gfx1151 on the Strix Halo.
  if [ -n "${GFX:-}" ]; then echo "$GFX"; return; fi
  rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]\+'
}

build_jobs() {
  # One compile job per 3 GB of available RAM, at most one per thread and at most 48: 48 on the R9700 host
  # (128 threads, 502 GB), 20 on an idle Strix Halo (32 threads, 62 GB), where more jobs run out of memory. Available,
  # not total: beside a serving teacher most of the shared 64 GB is the model's, and an out-of-memory kill could take
  # the server with it. BUILD_JOBS overrides.
  if [ -n "${BUILD_JOBS:-}" ]; then echo "$BUILD_JOBS"; return; fi
  local threads mem_gb jobs
  threads=$(nproc)
  mem_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  jobs=$((mem_gb / 3))
  [ "$jobs" -gt "$threads" ] && jobs=$threads
  [ "$jobs" -gt 48 ] && jobs=48
  [ "$jobs" -lt 1 ] && jobs=1
  echo "$jobs"
}

proxy() {
  python3 -c "import jupyter_server_proxy" 2>/dev/null && { echo "jupyter-server-proxy present"; return; }
  pip install --user --quiet jupyter-server-proxy || { echo "could not install jupyter-server-proxy"; return 1; }
  echo "jupyter-server-proxy installed: relaunch the pod once so the extension loads"
}

deps() {
  # pkgconf provides the pkg-config binary torch's LoadHIP.cmake needs (the image has none and there is no root).
  # cmake stays below 4, as vLLM's own requirements/rocm-build.txt asks; an unpinned install gets 4.x.
  pip install --user --quiet "cmake>=3.26.1,<4" ninja pkgconf packaging "setuptools>=77,<81" setuptools-scm wheel jinja2 regex huggingface_hub \
    || { echo "pip could not install the build tools"; return 1; }
  command -v pkg-config >/dev/null || ln -sf "$HOME/.local/bin/pkgconf" "$HOME/.local/bin/pkg-config"
  # Neither image ships a libdrm pkg-config entry or headers (the Strix Halo image has the runtime libdrm.so.2 only),
  # but torch's LoadHIP.cmake -> rocm_smi-config.cmake requires one, and the torch target then carries its include
  # and link flags. An empty stub satisfies both.
  if ! pkg-config --exists libdrm 2>/dev/null; then
    mkdir -p "$HOME/.local/include/libdrm" "$HOME/.local/lib/pkgconfig"
    echo 'int atlas_libdrm_stub(void){return 0;}' > /tmp/drm_stub.c
    /opt/rocm/llvm/bin/clang -shared -fPIC -o "$HOME/.local/lib/libdrm.so" /tmp/drm_stub.c
    cat > "$HOME/.local/lib/pkgconfig/libdrm.pc" <<EOF
prefix=$HOME/.local
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: libdrm
Description: stub so rocm_smi's cmake check passes on an image without libdrm
Version: 2.4.120
Libs: -L\${libdir} -ldrm
Cflags: -I\${includedir}/libdrm
EOF
  fi
  export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:/opt/rocm/lib/pkgconfig"
  export LIBRARY_PATH="$HOME/.local/lib"
  # No amdsmi Python package: torch 2.9.1+rocm7.13 loads its own libamd_smi.so.26 (from _rocm_sdk_core), and the
  # package from /opt/rocm/share/amd_smi loads a second copy into the same process, which then sees no GPU, so
  # torch.cuda.device_count() answers 0 and vLLM finds no platform (verified on the Strix Halo pod). Without it the
  # patch's torch.version.hip fallback detects ROCm and reads the gfx target through torch.
  if pip show amdsmi 2>/dev/null | grep -q "Location: $HOME/.local"; then
    pip uninstall -y --quiet amdsmi && echo "removed the user-site amdsmi package (it hides the GPU from torch)"
  fi
}

torch_constraints() {
  # The pod's own torch stack as pip constraints, so no vLLM dependency can swap in a CUDA wheel from PyPI.
  "$PY" - <<'PY'
from importlib import metadata
for name in ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm"):
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
PY
}

build_stamp() {
  # What a finished build was made from: the vLLM commit, the patch's content hash and the gfx target. A build whose
  # stamp differs (an older patch, another commit, no stamp at all) is rebuilt, not skipped because vllm imports.
  echo "commit=$VLLM_COMMIT patch=$(sha256sum "$PATCH" | cut -c1-16) gfx=$1"
}

make_venv() {
  # A venv beside the user site: it sees the image's torch 2.9.1+rocm7.13 (system site) and not the user site, so the
  # serving teacher's vLLM there neither shadows this build nor is replaced by it.
  [ -x "$PY" ] || python3 -m venv --system-site-packages "$VLLM_VENV" || { echo "could not create the venv $VLLM_VENV"; return 1; }
  "$PY" -c "import torch, sys; sys.exit(0 if torch.version.hip else 1)" 2>/dev/null \
    || { echo "the venv $VLLM_VENV does not see the image's ROCm torch"; return 1; }
  "$PY" -m pip install --quiet "cmake>=3.26.1,<4" ninja packaging "setuptools>=77,<81" setuptools-scm wheel jinja2 regex \
    || { echo "pip could not install the build tools into $VLLM_VENV"; return 1; }
}

build() {
  local gfx jobs constraints want
  gfx="$(gpu_arch)"
  [ -n "$gfx" ] || { echo "no gfx target from rocminfo; set GFX=<target>"; return 1; }
  case "$gfx" in
    gfx11*|gfx12*) ;;
    *) echo "these patches are for RDNA (gfx11, gfx12); rocminfo says $gfx"; return 1 ;;
  esac
  [ -f "$PATCH" ] || { echo "no patch at $PATCH: upload lab/$(basename "$PATCH") beside the script (run-script --with) or set PATCH"; return 2; }
  if [ "$VLLM_DIR" != "$DEFAULT_VLLM_DIR" ] && [ -z "$VLLM_VENV" ]; then
    echo "VLLM_DIR is $VLLM_DIR but no venv is given: a user-site install would replace the serving build; pass --venv DIR"
    return 2
  fi
  want="$(build_stamp "$gfx")"
  mkdir -p "$SERVE_DIR"
  if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want" ] && (cd "$SERVE_DIR" && "$PY" -c "import vllm") 2>/dev/null; then
    echo "vllm already built from $want in $VLLM_DIR"; return
  fi
  [ -f "$STAMP" ] && echo "rebuilding: $VLLM_DIR holds $(cat "$STAMP"), wanted $want"
  jobs="$(build_jobs)"
  deps || return 1
  if [ -n "$VLLM_VENV" ]; then make_venv || return 1; fi
  constraints="$HOME/.cache/vllm-torch-constraints-$(basename "$VLLM_DIR").txt"
  mkdir -p "$HOME/.cache" && torch_constraints > "$constraints" && [ -s "$constraints" ] \
    || { echo "could not read the pod's torch versions"; return 1; }
  [ -d "$VLLM_DIR/.git" ] || git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR" || { echo "could not clone vLLM"; return 1; }
  cd "$VLLM_DIR" || return 1
  rm -f "$STAMP"
  # Back to the clean commit: tracked files reset, files an earlier patch added removed (build/ is ignored and stays,
  # so a rebuild reuses what it can).
  git checkout -q -- . && git clean -q -f -d && git checkout -q "$VLLM_COMMIT" \
    || { echo "could not check out vLLM $VLLM_COMMIT"; return 1; }
  git apply --reject --whitespace=nowarn "$PATCH" || { echo "the patch $PATCH did not apply"; return 1; }
  "$PY" use_existing_torch.py || { echo "use_existing_torch.py failed"; return 1; }
  echo "building vLLM $VLLM_COMMIT + $(basename "$PATCH") for $gfx with $jobs jobs into ${VLLM_VENV:-the user site} (log: $BUILD_LOG)"
  local install=(pip install --user)
  [ -n "$VLLM_VENV" ] && install=("$PY" -m pip install)
  # SETUPTOOLS_SCM_PRETEND_VERSION: `git describe` on the NFS home times out (40 s) and kills the build otherwise.
  SETUPTOOLS_SCM_PRETEND_VERSION="0.19.2+$gfx" VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH="$gfx" \
    MAX_JOBS="$jobs" CMAKE_BUILD_PARALLEL_LEVEL="$jobs" PIP_CONSTRAINT="$constraints" \
    "${install[@]}" --no-build-isolation -e . 2>&1 | tee "$BUILD_LOG" | tail -3 || return 1
  echo "$want" > "$STAMP"
}

download() {
  mkdir -p "$MODELS"
  local dir="$MODELS/$(basename "$TEACHER")"
  if [ -f "$dir/config.json" ] && [ -z "$(find "$dir" -name '*.incomplete' 2>/dev/null)" ]; then
    echo "checkpoint present: $dir ($(du -sh "$dir" | cut -f1), $(find "$dir" -maxdepth 1 -type f | wc -l) files)"; return
  fi
  hf download "$TEACHER" --local-dir "$dir" || return 1
  echo "checkpoint: $dir ($(du -sh "$dir" | cut -f1), $(find "$dir" -maxdepth 1 -type f | wc -l) files)"
}

serve() {
  local dir="$MODELS/$(basename "$TEACHER")"
  # Speculative decoding: serve --speculative off|ngram|draft (read into $SPECULATIVE above; default off). On gfx1151 a target
  # pass that checks up to 16 tokens costs about one weight read (the chip sheet's matmul and int4 dequant times are
  # flat from 1 to 16 rows), so each accepted draft token is close to free; see
  # lab/refine/research/kernels/speculative_gfx1151.md. ngram drafts from the request's own prompt and output (vLLM's
  # prompt lookup, no second model); draft runs a small model of the same vocabulary (vLLM refuses a different size)
  # from $MODELS. SPEC_K is num_speculative_tokens; SPEC_MIN and SPEC_MAX the prompt-lookup n-gram range.
  local spec="${SPECULATIVE:-off}"
  local spec_args=()
  case "$spec" in
    off) ;;
    ngram)
      spec_args=(--speculative-config "{\"method\": \"ngram\", \"num_speculative_tokens\": ${SPEC_K:-4}, \"prompt_lookup_min\": ${SPEC_MIN:-1}, \"prompt_lookup_max\": ${SPEC_MAX:-3}}") ;;
    draft)
      local draft_dir="$MODELS/$(basename "${DRAFT:-Qwen/Qwen3.5-0.8B}")"
      [ -f "$draft_dir/config.json" ] || { echo "draft checkpoint missing: hf download ${DRAFT:-Qwen/Qwen3.5-0.8B} --local-dir $draft_dir"; return 1; }
      spec_args=(--speculative-config "{\"method\": \"draft_model\", \"model\": \"$draft_dir\", \"num_speculative_tokens\": ${SPEC_K:-4}}") ;;
    *) echo "--speculative must be off, ngram or draft, not $spec"; return 2 ;;
  esac
  echo "speculative decoding: $spec"
  # Never launch from $HOME: the ~/vllm checkout would shadow the installed package as a namespace package.
  mkdir -p "$SERVE_DIR" && cd "$SERVE_DIR"
  pkill -f "vllm.entrypoints" 2>/dev/null; sleep 2
  local vit_attn=()
  case "$(gpu_arch)" in
    gfx115*)
      # Without flash_attn, vLLM gives the vision encoder Torch SDPA. On gfx1151 torch 2.9.1 keeps its AOTriton SDPA
      # kernels off as experimental, so SDPA runs the math kernel and the encoder's profiling pass asks for a
      # 256 GiB attention matrix; switched on (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1) they start, but go wrong
      # at a 720p image's ~3.5k tokens (max error 0.08 to 0.12 against 0.001, NaN at 4096) and the caption comes
      # back as "!!!!". vLLM's own Triton attention is exact there and needs no attention matrix (fix round 3; on
      # pod C a 1280x720 caption came back as real text and the server starts in about 5 min with a warm cache).
      vit_attn=(--mm-encoder-attn-backend TRITON_ATTN) ;;
  esac
  # Two names, one model: "teacher" for labeling and captions, "group-brain" for the Circle's planning questions
  # (contexts/circle). Priority scheduling serves the lower request priority first and, when the KV cache runs out,
  # preempts the highest: planning asks at -10, labeling sends none (0), so a planning question never queues behind
  # a labeling run, and continuous batching keeps the labels decoding beside it. Requests are not logged
  # (--enable-log-requests stays off), so no prompt is written to the server log or to disk.
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$dir" --served-model-name teacher group-brain --port "$PORT" --host 127.0.0.1 \
    --scheduling-policy priority \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
    --max-model-len "$MAX_LEN" --limit-mm-per-prompt '{"image":1}' --gpu-memory-utilization "$GPU_UTIL" \
    --dtype bfloat16 ${vit_attn[@]+"${vit_attn[@]}"} ${spec_args[@]+"${spec_args[@]}"} > "$SERVE_LOG" 2>&1 &
  local pid=$!
  echo "vllm serve pid $pid (log: $SERVE_LOG)"
  # Readiness through python3, not curl: not every pod image ships curl.
  for i in $(seq 1 180); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5)" 2>/dev/null \
      && { serve_kernel; echo "teacher up on $PORT (quantized matmul kernel: ${STEP_MARKS#kernel=})"; return; }
    kill -0 "$pid" 2>/dev/null || { echo "the server process ended before it answered; see $SERVE_LOG"; tail -20 "$SERVE_LOG"; return 1; }
    sleep 5
  done
  echo "server did not come up; see $SERVE_LOG"; tail -20 "$SERVE_LOG"
  return 1
}

serve_kernel() {
  # vLLM logs the kernel it picked for the 4-bit linear layers once at load ("Using ConchLinearKernel for
  # CompressedTensorsWNA16"); it goes on the serve step's ledger line as kernel=<name>.
  local name
  name=$(grep -o 'Using [A-Za-z0-9_]* for CompressedTensorsWNA16' "$SERVE_LOG" 2>/dev/null | head -1 | awk '{print $2}')
  STEP_MARKS="kernel=${name:-unknown}"
}

verify() {
  # After build: the patch's rejected hunks (none expected), the compiled extensions, and the platform vLLM resolves.
  echo "vllm_commit: $(git -C "$VLLM_DIR" rev-parse --short=9 HEAD 2>/dev/null)"
  echo "patched_files: $(git -C "$VLLM_DIR" diff --name-only 2>/dev/null | wc -l)"
  echo "rejected_hunks: $(find "$VLLM_DIR" -name '*.rej' -not -path '*/.git/*' 2>/dev/null | tr '\n' ' ')"
  echo "build_stamp: $(cat "$STAMP" 2>/dev/null || echo none)"
  mkdir -p "$SERVE_DIR" && cd "$SERVE_DIR"   # not $HOME: the ~/vllm checkout would shadow the installed package
  "$PY" - <<'PY'
import importlib
import torch, vllm
print(f"vllm: {vllm.__version__}")
for name in ("vllm._C", "vllm._rocm_C", "vllm._moe_C"):
    try:
        importlib.import_module(name); print(f"{name}: imports")
    except Exception as exc:
        print(f"{name}: FAILS {type(exc).__name__}: {exc}")
from vllm.platforms import current_platform
print(f"platform: {type(current_platform).__name__}, torch sees {torch.cuda.device_count()} GPU")
print(f"vllm_from: {vllm.__file__}")
try:
    import vllm._rocm_C  # noqa: F401
    print(f"hybrid_w4a16_op: {'present' if hasattr(torch.ops._rocm_C, 'wvSplitK_int4_g') else 'absent'}")
except Exception as exc:
    print(f"hybrid_w4a16_op: unknown ({type(exc).__name__})")
PY
}

models() {
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=10).read().decode())"
}

stop() {
  # End the server and wait until it is gone, so the pod is left as it was found.
  pkill -f "vllm.entrypoints" 2>/dev/null || { echo "no teacher server running"; return 0; }
  for i in $(seq 1 30); do
    pgrep -f "vllm.entrypoints" >/dev/null || { echo "teacher stopped"; return 0; }
    sleep 2
  done
  pkill -9 -f "vllm.entrypoints" 2>/dev/null; sleep 2
  pgrep -f "vllm.entrypoints" >/dev/null && { echo "teacher still running"; return 1; }
  echo "teacher stopped (killed)"
}

case "$STEP" in
  facts) facts ;;
  proxy) proxy ;;
  build) timed build ;;
  verify) verify ;;
  download) timed download ;;
  serve) timed serve ;;
  models) models ;;
  stop) stop ;;
  all) proxy && timed build && timed download && timed serve ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
