#!/bin/bash
# Pod A (teacher): build vLLM for the R9700 (gfx1201, ROCm 7.13) and serve Qwen3.8-27B behind the hub proxy.
# Idempotent and resumable: re-run after a pod relaunch; finished steps are skipped.
# Usage on the pod (JupyterLab terminal):  bash lab/pod_a_teacher.sh [build|download|serve|all]
# The Jetson calls the server at https://dub.aupcloud.io/user/<account>/proxy/8000/v1/... with
# header "Authorization: token <hub API token>" (jupyter-server-proxy must be installed once: step "proxy").
set -u
export PATH="$HOME/.local/bin:$PATH"
STEP="${1:-all}"
VLLM_DIR="$HOME/vllm"
VLLM_COMMIT="b1388b1fb"                      # April 2026 main; builds against the pod's torch 2.9.1+rocm7.13
PATCH="$(cd "$(dirname "$0")" && pwd)/vllm-gfx1201.diff"
MODELS="$HOME/models"
TEACHER="${TEACHER:-amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}"   # fallbacks: RedHatAI/Qwen3.8-27B-INT4, Qwen/Qwen3.8-27B-FP8
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"

proxy() {
  python3 -c "import jupyter_server_proxy" 2>/dev/null && { echo "jupyter-server-proxy present"; return; }
  pip install --user --quiet jupyter-server-proxy
  echo "jupyter-server-proxy installed: relaunch the pod once so the extension loads"
}

deps() {
  # pkgconf provides the pkg-config binary torch's LoadHIP.cmake needs (the image has none and there is no root).
  pip install --user --quiet cmake ninja pkgconf packaging "setuptools>=77,<81" setuptools-scm wheel jinja2 regex huggingface_hub
  command -v pkg-config >/dev/null || ln -sf "$HOME/.local/bin/pkgconf" "$HOME/.local/bin/pkg-config"
  export PKG_CONFIG_PATH=/opt/rocm/lib/pkgconfig
  [ -d /opt/rocm/share/amd_smi ] && python3 -c "import amdsmi" 2>/dev/null || pip install --user --quiet /opt/rocm/share/amd_smi 2>/dev/null || true
}

build() {
  if python3 -c "import vllm" 2>/dev/null; then echo "vllm already importable: $(python3 -c 'import vllm;print(vllm.__version__)')"; return; fi
  deps
  [ -d "$VLLM_DIR/.git" ] || git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
  cd "$VLLM_DIR"
  git checkout -q "$VLLM_COMMIT"
  git checkout -- . && git apply --reject --whitespace=nowarn "$PATCH"
  python3 use_existing_torch.py
  # SETUPTOOLS_SCM_PRETEND_VERSION: `git describe` on the NFS home times out (40 s) and kills the build otherwise.
  SETUPTOOLS_SCM_PRETEND_VERSION=0.19.2+gfx1201 VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=48 CMAKE_BUILD_PARALLEL_LEVEL=48 \
    pip install --user --no-build-isolation -e . 2>&1 | tee "$HOME/vllm_build.log" | tail -3
}

download() {
  mkdir -p "$MODELS"
  local dir="$MODELS/$(basename "$TEACHER")"
  [ -f "$dir/config.json" ] && [ -z "$(find "$dir" -name '*.incomplete' 2>/dev/null)" ] && { echo "checkpoint present: $dir"; return; }
  hf download "$TEACHER" --local-dir "$dir"
}

serve() {
  local dir="$MODELS/$(basename "$TEACHER")"
  pkill -f "vllm serve" 2>/dev/null; sleep 2
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$dir" --served-model-name teacher --port "$PORT" --host 127.0.0.1 \
    --max-model-len "$MAX_LEN" --limit-mm-per-prompt '{"image":1}' --gpu-memory-utilization 0.90 \
    --dtype bfloat16 > "$HOME/vllm_serve.log" 2>&1 &
  echo "vllm serve pid $! (log: ~/vllm_serve.log)"
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && { echo "teacher up on $PORT"; return; }; sleep 5; done
  echo "server did not come up; see ~/vllm_serve.log"; tail -20 "$HOME/vllm_serve.log"
}

case "$STEP" in
  proxy) proxy ;;
  build) build ;;
  download) download ;;
  serve) serve ;;
  all) proxy; build; download; serve ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
