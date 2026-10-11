#!/bin/bash
# R65: package the two-pod adapter the way L5 packaged L16's, and serve it for the gate's candidate arm (pod D).
#   build <run id>              lab package build of ~/adapters/r65-<run id>-rank1/adapter into ~/packages (CPU only)
#   serve <run id> [port]       llama-server (E29's HIP build) on 127.0.0.1:<port> with the Jetson's serve flags
#                               (jetson/atlas-llm.sh) plus --top-k 1, the gate's greedy setting; inside a GPU Queue slot
#   stop [port]                 stop the server this script started
# Needs ~/r65/pkg-bundle.tar.gz (lab/, contracts/ and jetson/make_manifest.py at the branch), L5's llama-quantize in
# ~/llama.cpp/build and the base projector in ~/models/Qwen3.5-4B-GGUF. The scorer runs elsewhere and reaches the
# server over the tailnet (Tailscale userspace hands an inbound connection to the same port on 127.0.0.1):
#   python -m contexts.agent.eval --runtime llama --base-url http://<pod D tailnet IP>:<port> --json out.json
set -u -o pipefail
STEP="${1:-help}"
VENV="$HOME/venvs/atlas-train"
SRC="$HOME/r65/pkg-src"
LLAMA="$HOME/llama.cpp"
SERVER="${LLAMA_SERVER:-$HOME/e29/llama.cpp/build-hip/bin/llama-server}"
MMPROJ="$HOME/models/Qwen3.5-4B-GGUF/mmproj-F16.gguf"
TASK="tool-calling"
export PATH="$HOME/.local/bin:$PATH" PYTHONNOUSERSITE=1 LAB_STATE_DIR="$HOME/r65/pkg-state" LAB_TASK=R65

build() {
  local run="$1" t0 rc version="atlas-$TASK-r65-$1"
  rm -rf "$SRC"; mkdir -p "$SRC" "$LAB_STATE_DIR"
  tar -xzf "$HOME/r65/pkg-bundle.tar.gz" -C "$SRC" || { echo "no bundle"; return 1; }
  t0=$(date +%s)
  (cd "$SRC" && "$VENV/bin/python" -m lab package build --adapter "$HOME/adapters/r65-$run-rank1/adapter" \
    --run-id "r65-$run" --task "$TASK" --mmproj "$MMPROJ" --base-dir "$HOME/models/Qwen3.5-4B" --llama-cpp "$LLAMA" \
    --out "$HOME/packages") 2>&1 | tail -15
  rc=${PIPESTATUS[0]}
  echo "### build rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  ls -la "$HOME/packages/$version" 2>&1
  return "$rc"
}

serve() {
  local run="$1" port="${2:-8091}" version="atlas-$TASK-r65-$1" pkg gguf t0
  pkg="$HOME/packages/$version"
  gguf="$(ls "$pkg"/*Q4_K_M.gguf 2>/dev/null | head -1)"
  [ -n "$gguf" ] || { echo "no Q4_K_M gguf in $pkg"; return 1; }
  pgrep -af "llama-server" | grep -v -E "pgrep|defunct" && { echo "a llama-server already runs here: not ours to stop"; return 1; }
  mkdir -p "$HOME/serve"
  t0=$(date +%s)
  nohup "$SERVER" -m "$gguf" --mmproj "$MMPROJ" --jinja --host 127.0.0.1 --port "$port" -c 4096 -ngl 99 -np 1 \
    --reasoning-budget 0 --cache-ram 0 --top-k 1 > "$HOME/serve/r65-llama-$port.log" 2>&1 &
  echo $! > "$HOME/serve/r65-llama-$port.pid"
  for i in $(seq 1 90); do
    curl -s -m 2 "http://127.0.0.1:$port/health" | grep -q ok && { echo "llama-server up on $port after $(( $(date +%s) - t0 )) s ($gguf)"; break; }
    kill -0 "$(cat "$HOME/serve/r65-llama-$port.pid")" 2>/dev/null || { echo "llama-server ended"; tail -20 "$HOME/serve/r65-llama-$port.log"; return 1; }
    sleep 2
  done
  curl -s "http://127.0.0.1:$port/props" | python3 -c "import json,sys; p=json.load(sys.stdin)['default_generation_settings']; print('top_k', p.get('params', p).get('top_k'))"
}

stop() {
  local port="${1:-8091}" pid
  pid="$(cat "$HOME/serve/r65-llama-$port.pid" 2>/dev/null)"
  [ -n "$pid" ] && kill "$pid" 2>/dev/null && echo "llama-server $pid stopped"
  rm -f "$HOME/serve/r65-llama-$port.pid"
}

case "$STEP" in
  build) shift; build "$@" ;;
  serve) shift; serve "$@" ;;
  stop) shift; stop "$@" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
rc=$?
echo "### END"
exit "$rc"
