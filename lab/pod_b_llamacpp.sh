#!/bin/bash
# Pod B (account D, Strix Halo, gfx1151): build llama.cpp from source with the HIP backend and serve the teacher's model
# (Qwen/Qwen3.8-27B) as the public Q4_K_M GGUF with its vision projector, on port 8100 behind the hub proxy (E29).
# The GPU target comes from rocminfo; GFX=<target> overrides it.
# Idempotent and resumable: finished steps are skipped (a build whose binaries exist, a GGUF whose sha256 matches).
# Usage on the pod:  bash pod_b_llamacpp.sh [facts|build|verify|download|sampler|serve [NP] [CTX]|models|bench|stop|all]
#   serve [NP] [CTX]: NP parallel slots (default 16) sharing CTX tokens of context (default 4096 per slot); the R11
#   workload needs about 1,300 tokens per request (an image prompt is 1,137 in, 128 out), so 64 streams run as
#   "serve 64 131072" (2,048 per slot). NP and CTX may also come from the environment.
# From the box, with nothing typed on the pod (each run gets its own file in the Jupyter root):
#   python -m lab.controller run-script --account D --timeout 900 lab/pod_b_llamacpp.sh build
#   python -m lab.controller run-script --account D --with lab/pod_power.py --with lab/pod_power.sh lab/pod_b_llamacpp.sh sampler
#   python -m lab.controller run-script --account D --timeout 900 lab/pod_b_llamacpp.sh serve 16 65536   # in a slot
# Callers reach the server at https://<hub>/user/<account>/proxy/8100/v1/... with "Authorization: token <hub token>".
# build, download, serve and bench end with "### <step> wall_s=<seconds> rc=<code>" for the ledger. A failed step ends
# the script with a non-zero code. Start the server only inside a GPU Queue slot, and stop it after.
set -u -o pipefail
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"
STEP="${1:-all}"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
# llama.cpp master of 2026-10-10 (build b11552), the commit the kernels research note (graphics pipeline) read.
LLAMA_COMMIT="${LLAMA_COMMIT:-23b0202a189c44a54625aadcb37a946dd1d6278d}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/e29/llama.cpp}"   # its own tree: ~/llama.cpp and its build folders belong to other tasks
BUILD_DIR="${BUILD_DIR:-$LLAMA_DIR/build-hip}"
BIN="$BUILD_DIR/bin"
# The public Q4_K_M quant of Qwen/Qwen3.8-27B with its f16 mmproj, pinned to a revision and checked by sha256.
GGUF_REPO="${GGUF_REPO:-bartowski/Qwen3.8-27B-GGUF}"
GGUF_REV="${GGUF_REV:-0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a}"
GGUF_FILE="Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA="880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673"
MMPROJ_FILE="mmproj-Qwen3.8-27B-f16.gguf"
MMPROJ_SHA="cb0a71fe9b84af2552fc6260acf8e9773a29d115c683aba01c72a7ac537b212b"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-27B-GGUF}"
PORT="${PORT:-8100}"
NP="${NP:-16}"
CTX="${CTX:-$((NP * 4096))}"
ALIAS="${ALIAS:-teacher}"
SERVE_DIR="$HOME/serve"
SERVE_LOG="${SERVE_LOG:-$SERVE_DIR/llama-server-$PORT.log}"
PIDFILE="$SERVE_DIR/llama-server-$PORT.pid"
BENCH_OUT="${BENCH_OUT:-$HOME/e29/llama-bench.json}"

timed() {
  local t0 rc
  t0=$(date +%s)
  "$@"; rc=$?
  echo "### $1 wall_s=$(( $(date +%s) - t0 )) rc=$rc"
  return $rc
}

gpu_arch() {
  if [ -n "${GFX:-}" ]; then echo "$GFX"; return; fi
  rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]\+'
}

facts() {
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gfx: $(gpu_arch)"
  echo "cpu: $(nproc) threads; $(awk '/MemTotal/ {t=$2} /MemAvailable/ {a=$2} END {printf "%.0f GB total, %.0f GB available", t/1048576, a/1048576}' /proc/meminfo)"
  echo "rocm: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
  echo "hip_compiler: $(/opt/rocm/llvm/bin/clang++ --version 2>/dev/null | head -1)"
  echo "cmake: $(cmake --version 2>/dev/null | head -1 || echo none)"
  echo "llama_dir: $LLAMA_DIR at $(git -C "$LLAMA_DIR" rev-parse --short=7 HEAD 2>/dev/null || echo absent)"
  echo "binaries: $(ls "$BIN" 2>/dev/null | grep -E '^llama-(server|bench|cli)$' | tr '\n' ' ')"
  echo "gguf: $(ls -l "$MODEL_DIR/$GGUF_FILE" "$MODEL_DIR/$MMPROJ_FILE" 2>/dev/null | awk '{print $5, $9}' | tr '\n' ' ')"
  echo "server_running: $(pgrep -f "llama-server.*--port $PORT" >/dev/null && echo yes || echo no)"
  echo "port_$PORT: $(python3 -c "import socket; s=socket.socket(); s.settimeout(1); print('busy' if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 'free')")"
  rocm-smi --showuse --showmemuse 2>/dev/null | grep -E 'GPU use|VRAM%'
}

deps() {
  # The image has no cmake; cmake from pip is fine for llama.cpp (no upper bound needed).
  command -v cmake >/dev/null && command -v ninja >/dev/null && return
  pip install --user --quiet cmake ninja || { echo "pip could not install cmake and ninja"; return 1; }
}

source_tree() {
  # Use an existing checkout only when it is the pinned commit; never move another task's checkout.
  if [ -d "$LLAMA_DIR/.git" ]; then
    local head
    head="$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null)"
    [ "$head" = "$LLAMA_COMMIT" ] && return
    echo "$LLAMA_DIR is at $head, not $LLAMA_COMMIT: set LLAMA_DIR to a fresh folder"; return 1
  fi
  mkdir -p "$LLAMA_DIR" && cd "$LLAMA_DIR" || return 1
  git init -q && git remote add origin "$LLAMA_REPO" && git fetch -q --depth 1 origin "$LLAMA_COMMIT" \
    && git checkout -q FETCH_HEAD || { echo "could not fetch llama.cpp $LLAMA_COMMIT"; return 1; }
}

build() {
  if [ -x "$BIN/llama-server" ] && [ -x "$BIN/llama-bench" ] && [ -x "$BIN/llama-cli" ]; then
    echo "llama.cpp already built in $BUILD_DIR"; return
  fi
  local gfx jobs
  gfx="$(gpu_arch)"
  [ -n "$gfx" ] || { echo "no gfx target from rocminfo; set GFX=<target>"; return 1; }
  source_tree || return 1
  deps || return 1
  # One job per 3 GB of RAM, at most one per thread (20 on the Strix Halo), niced so a neighbour's run keeps the CPU.
  jobs=$(( $(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo) / 3 ))
  [ "$jobs" -gt "$(nproc)" ] && jobs=$(nproc)
  [ "$jobs" -lt 1 ] && jobs=1
  echo "building llama.cpp $(git -C "$LLAMA_DIR" rev-parse --short=7 HEAD) for $gfx (HIP) with $jobs jobs in $BUILD_DIR"
  # CMake's HIP language refuses the hipcc wrapper: clang++ from the ROCm LLVM is the HIP compiler.
  # GGML_HIP_ROCWMMA_FATTN stays off (no rocWMMA headers in the image); curl is off (no libcurl-dev, no root).
  nice -n 10 cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DGPU_TARGETS="$gfx" -DAMDGPU_TARGETS="$gfx" \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF > "$BUILD_DIR.configure.log" 2>&1 \
    || { echo "cmake configure failed; the end of $BUILD_DIR.configure.log:"; tail -20 "$BUILD_DIR.configure.log"; return 1; }
  nice -n 10 cmake --build "$BUILD_DIR" --target llama-server llama-bench llama-cli llama-mtmd-cli -j "$jobs" \
    > "$BUILD_DIR.build.log" 2>&1 \
    || { echo "build failed; the end of $BUILD_DIR.build.log:"; grep -E "error|Error" "$BUILD_DIR.build.log" | tail -20; return 1; }
  ls -l "$BIN" | grep -E 'llama-(server|bench|cli|mtmd-cli)$'
}

verify() {
  echo "commit: $(git -C "$LLAMA_DIR" rev-parse --short=7 HEAD)"
  echo "hip_compiler: $(grep -m1 '^CMAKE_HIP_COMPILER:' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2)"
  echo "targets: $(grep -m1 '^GPU_TARGETS:' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2)"
  "$BIN/llama-server" --version 2>&1 | grep -E '^version|^built'
  timeout 60 "$BIN/llama-server" --list-devices 2>&1 | grep -E 'ROCm|HIP|gfx|MiB' | head -4
}

check_sha() {
  local file="$1" want="$2" got
  got="$(sha256sum "$file" | cut -d' ' -f1)"
  [ "$got" = "$want" ] && { echo "sha256 ok: $(basename "$file")"; return 0; }
  echo "sha256 MISMATCH: $(basename "$file") is $got, expected $want"; return 1
}

download() {
  mkdir -p "$MODEL_DIR"
  local marker="$MODEL_DIR/.verified-$GGUF_REV"
  if [ -f "$marker" ] && [ -f "$MODEL_DIR/$GGUF_FILE" ] && [ -f "$MODEL_DIR/$MMPROJ_FILE" ]; then
    echo "GGUF present and verified: $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"; return
  fi
  if ! { [ -f "$MODEL_DIR/$GGUF_FILE" ] && [ -f "$MODEL_DIR/$MMPROJ_FILE" ]; }; then
    command -v hf >/dev/null || pip install --user --quiet huggingface_hub || return 1
    nice -n 10 hf download "$GGUF_REPO" "$GGUF_FILE" "$MMPROJ_FILE" --revision "$GGUF_REV" --local-dir "$MODEL_DIR" \
      || { echo "download of $GGUF_REPO failed"; return 1; }
  fi
  check_sha "$MODEL_DIR/$GGUF_FILE" "$GGUF_SHA" && check_sha "$MODEL_DIR/$MMPROJ_FILE" "$MMPROJ_SHA" || return 1
  touch "$marker"
  ls -l "$MODEL_DIR" | grep gguf
}

sampler() {
  # The T3/T6 power sampler on port 8790, which the box reads through the proxy; left running when found running.
  local here
  here="$(cd "$(dirname "$0")" && pwd)"
  [ -f "$here/pod_power.py" ] || [ -f "$HOME/atlas-power/pod_power.py" ] || cp "$HOME/pod_power.py" "$here/" 2>/dev/null
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8790/status', timeout=5)" 2>/dev/null; then
    echo "sampler already answering on 8790"; return
  fi
  local sh="$here/pod_power.sh"
  [ -f "$sh" ] || sh="$HOME/Knighthacks26/lab/pod_power.sh"
  [ -f "$sh" ] || { echo "pod_power.sh not found beside this script"; return 1; }
  bash "$sh" start "${SAMPLER_NODE:-gpu-b}"   # the sampler names pods gpu-a and gpu-b
}

gpu_busy() {
  # Another engine on the card (vLLM, a trainer, another llama-server) means wait: E29 measures an idle card.
  pgrep -af 'vllm.entrypoints|VLLM::|train.py|llama-server|llama-bench' | grep -v -E "pgrep|pod_b_llamacpp|defunct" | cut -c1-120
}

serve() {
  [ -x "$BIN/llama-server" ] || { echo "no llama-server in $BIN: run build first"; return 1; }
  local busy
  busy="$(gpu_busy)"
  [ -z "$busy" ] || { echo "the card is in use; not starting:"; echo "$busy"; return 1; }
  mkdir -p "$SERVE_DIR" && cd "$SERVE_DIR" || return 1
  # Every layer on the GPU, flash attention on, NP slots of CTX/NP tokens each, the chat template from the GGUF
  # (--jinja, so chat_template_kwargs.enable_thinking works), one image per request through the mmproj.
  # Requests are not logged beyond llama-server's own slot timings.
  nohup "$BIN/llama-server" -m "$MODEL_DIR/$GGUF_FILE" --mmproj "$MODEL_DIR/$MMPROJ_FILE" --alias "$ALIAS" \
    --host 127.0.0.1 --port "$PORT" -ngl 999 -fa on -c "$CTX" -np "$NP" --jinja --no-webui \
    > "$SERVE_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "llama-server pid $pid, $NP slots, ctx $CTX (log: $SERVE_LOG)"
  for i in $(seq 1 120); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null \
      && { echo "llama-server up on $PORT"; grep -E "n_slots|n_ctx|kv_cache|flash_attn|offloaded|multimodal" "$SERVE_LOG" | head -8; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "llama-server exited before it answered; the end of $SERVE_LOG:"; tail -20 "$SERVE_LOG"; return 1; }
    sleep 5
  done
  echo "llama-server did not come up in 10 minutes; see $SERVE_LOG"; tail -20 "$SERVE_LOG"
  return 1
}

models() {
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=10).read().decode())"
}

bench() {
  # The engine's own view: prompt processing (pp512) and generation (tg128) on the same GGUF, 3 repetitions each.
  # Refused while llama-server or any other engine holds the card.
  local busy
  busy="$(gpu_busy)"
  [ -z "$busy" ] || { echo "the card is in use; not running llama-bench:"; echo "$busy"; return 1; }
  mkdir -p "$(dirname "$BENCH_OUT")"
  "$BIN/llama-bench" -m "$MODEL_DIR/$GGUF_FILE" -ngl 999 -fa 1 -p 512 -n 128 -r 3 -o json > "$BENCH_OUT" \
    2> "${BENCH_OUT%.json}.err" || { echo "llama-bench failed:"; tail -10 "${BENCH_OUT%.json}.err"; return 1; }
  python3 - "$BENCH_OUT" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
    print(f"{test}: {r['avg_ts']:.2f} +- {r['stddev_ts']:.2f} tok/s  ({r['backends']}, build {r['build_commit']}, fa {r['flash_attn']})")
PY
}

alive() {
  # A process that has exited but whose parent has not reaped it (state Z) answers kill -0; it holds nothing.
  kill -0 "$1" 2>/dev/null && [ "$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)" != "Z" ]
}

stop() {
  local pid=""
  [ -f "$PIDFILE" ] && pid="$(cat "$PIDFILE")"
  if [ -z "$pid" ] || ! alive "$pid"; then
    pid="$(pgrep -f "llama-server.*--port $PORT" | head -1)"
  fi
  [ -n "$pid" ] || { echo "no llama-server on $PORT"; rm -f "$PIDFILE"; return 0; }
  kill "$pid"
  for i in $(seq 1 30); do
    alive "$pid" || { echo "llama-server stopped"; rm -f "$PIDFILE"; return 0; }
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null; sleep 2
  alive "$pid" && { echo "llama-server still running"; return 1; }
  rm -f "$PIDFILE"; echo "llama-server stopped (killed)"
}

case "$STEP" in
  facts) facts ;;
  build) timed build ;;
  verify) verify ;;
  download) timed download ;;
  sampler) sampler ;;
  serve) [ $# -ge 2 ] && { NP="$2"; CTX="${3:-$((NP * 4096))}"; }; timed serve ;;
  models) models ;;
  bench) timed bench ;;
  stop) stop ;;
  all) timed build && timed download && sampler && timed serve ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
