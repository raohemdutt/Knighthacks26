#!/bin/bash
# L16 retrain on tool-calling-v2, the pod D side (account D, node pod-b: the trainer pod's role). A copy of the pod
# steps of lab/l16_run.sh (train), lab/l5_run.sh (package) and the serve, check and stop steps of the L4 run script,
# with the new dataset, the chosen recipe and a version name that says where the adapter was trained.
#
# On the workstation, from the repo root (Git Bash):
#   bash lab/l16v2/pod_run.sh pack <colab out dir> <dest dir>   l16v2-bundle.tar.gz (lab/, contracts/ and
#                                         jetson/make_manifest.py at HEAD) and l16v2-inputs.tar.gz (the dataset
#                                         without its gold, the two rows files, the A100 adapter with chosen.json)
#   then: python -m lab.transfer push-file --account D <dest>/l16v2-bundle.tar.gz l16v2/l16v2-bundle.tar.gz   (and inputs)
# On the pod, through the controller, inside a Running GPU Queue row only
# (python -m lab.controller run-script --account D lab/l16v2/pod_run.sh <step> ...):
#   facts                          read-only: who is on the card, the ports, the inputs, memory
#   prep                           unpack the bundle and the inputs under ~/l16v2; check the venv imports
#   package <version> <a100 | pod:<run id>> <run id>   python -m lab package build, as the first package was built
#   serve <version>                two llama-server (HIP build) with the box's flags plus --top-k 1:
#                                  8100 = before (the base rebuilt through the package's path, checked by sha256),
#                                  8101 = the package's GGUF; then check
#   check                          /props top_k of each server, thinking off and the same answer twice
#   stop                           stop both servers; show that nothing of this run listens or holds memory
#   train <run id> [trainer flags] the chosen recipe on this pod: adapter, telemetry, validation.jsonl, run.json and
#                                  the log under ~/adapters/l16v2-<run id>/ (the trainer deletes its data copy)
#   cleanup [job ids]              remove ~/l16v2 and this task's job files; packages and adapters stay
set -u -o pipefail
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"
STEP="${1:-help}"
BASE_REPO="Qwen/Qwen3.5-4B"
BASE_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATASET_ID="tool-calling-v2"
TASK="tool-calling"
L="$HOME/l16v2"; SRC="$L/src"; VENV="$HOME/venvs/atlas-train"; LLAMA="$HOME/llama.cpp"
MODEL="$HOME/models/Qwen3.5-4B"
MMPROJ="$HOME/models/Qwen3.5-4B-GGUF/mmproj-F16.gguf"
MMPROJ_SHA="cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864"
BEFORE="$HOME/l4/Qwen3.5-4B-base-samepath-Q4_K_M.gguf"
BEFORE_SHA="84284aa2899f112244a6735d9fbb0c6f94085fb59eb719c95ec376e421eee04f"
PORT_BEFORE=8100; PORT_CAND=8101
TRAIN_TIMEOUT_S="${TRAIN_TIMEOUT_S:-3000}"   # 50 minutes: the run is interrupted (SIGINT) before a 60-minute row ends
die() { echo "l16v2 pod_run: $*" >&2; exit 1; }

# --- on the workstation ---------------------------------------------------------------------------------------

pack() {
  local colab="${1:?the folder the Colab products were pulled into}" dest="${2:?where the two tarballs go}" stage
  mkdir -p "$dest"
  git archive --format=tar.gz -o "$dest/l16v2-bundle.tar.gz" HEAD lab contracts jetson/make_manifest.py ':(exclude)lab/refine' \
    || die "git archive failed"
  stage="$dest/inputs"; rm -rf "$stage"; mkdir -p "$stage/datasets" "$stage/a100"
  tar -xzf "$colab/dataset-$DATASET_ID.tar.gz" -C "$stage/datasets" || die "no dataset tarball in $colab"
  mv "$stage/datasets/$DATASET_ID/rows-train.jsonl" "$stage/datasets/$DATASET_ID/rows-validation.jsonl" "$stage/datasets/"
  # The dataset as the trainer reads it: manifest, examples, split, images. The gold and the build records stay here.
  rm -f "$stage/datasets/$DATASET_ID"/{expected.jsonl,validation.json,build.json,dataset.json,split_check.json} \
        "$stage/datasets/$DATASET_ID/images/.gitkeep"
  tar -xzf "$colab/best-adapter.tar.gz" -C "$stage/a100" adapter chosen.json run.json || die "no best-adapter tarball in $colab"
  tar -czf "$dest/l16v2-inputs.tar.gz" -C "$stage" datasets a100 || die "tar failed"
  rm -rf "$stage"
  echo "bundle: $dest/l16v2-bundle.tar.gz at $(git rev-parse HEAD) ($(stat -c %s "$dest/l16v2-bundle.tar.gz") bytes)"
  echo "inputs: $dest/l16v2-inputs.tar.gz ($(stat -c %s "$dest/l16v2-inputs.tar.gz") bytes, sha256 $(sha256sum "$dest/l16v2-inputs.tar.gz" | cut -d' ' -f1))"
}

# --- on the pod --------------------------------------------------------------------------------------------------

server_bin() {
  local b
  for b in "$HOME/e29/llama.cpp/build-hip/bin/llama-server" "$HOME/llama.cpp/build-hip-gfx1151/bin/llama-server"; do
    [ -x "$b" ] && { echo "$b"; return 0; }
  done
  return 1
}

port_state() {
  python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); print('busy' if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 'free')" "$1"
}

card_users() {
  pgrep -af 'vllm.entrypoints|VLLM::|train.py|llama-server|llama-bench' | grep -v -E "pgrep|defunct|pod_run|lab_run" | cut -c1-140
}

facts() {
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "card users (engines):"; card_users | sed 's/^/  /'; echo "  (end)"
  echo "port $PORT_BEFORE: $(port_state $PORT_BEFORE)  port $PORT_CAND: $(port_state $PORT_CAND)"
  echo "server bin: $(server_bin || echo NONE)"
  ls -l "$BEFORE" "$MMPROJ" "$LLAMA/convert_hf_to_gguf.py" "$LLAMA/build/bin/llama-quantize" "$VENV/bin/python" \
        "$L/l16v2-bundle.tar.gz" "$L/l16v2-inputs.tar.gz" 2>&1 | awk '{print "  " $5, $NF}'
  echo "packages: $(ls "$HOME/packages" 2>/dev/null | tr '\n' ' ')"
  echo "adapters: $(ls "$HOME/adapters" 2>/dev/null | tr '\n' ' ')"
  awk '/MemTotal/ {t=$2} /MemAvailable/ {a=$2} END {printf "mem: %.1f GB total, %.1f GB available\n", t/1048576, a/1048576}' /proc/meminfo
  echo "disk: $(df -h "$HOME" | awk 'NR==2 {print $4 " free of " $2}')"
  rocm-smi --showmemuse 2>/dev/null | grep -E 'VRAM%|GPU Memory' | sed 's/^/  /'
  echo "### facts rc=0"
}

prep() {
  [ -f "$L/l16v2-bundle.tar.gz" ] && [ -f "$L/l16v2-inputs.tar.gz" ] || die "push l16v2-bundle.tar.gz and l16v2-inputs.tar.gz to ~/l16v2 first"
  rm -rf "$SRC" "$L/datasets" "$L/a100"
  mkdir -p "$SRC" "$L/state"
  tar -xzf "$L/l16v2-bundle.tar.gz" -C "$SRC" || die "could not unpack the bundle"
  python3 -c 'import sys, tarfile; print("source", tarfile.open(sys.argv[1]).pax_headers.get("comment", "unknown"))' "$L/l16v2-bundle.tar.gz" | tee "$SRC/SOURCE_COMMIT"
  tar -xzf "$L/l16v2-inputs.tar.gz" -C "$L" || die "could not unpack the inputs"
  echo "inputs sha256 $(sha256sum "$L/l16v2-inputs.tar.gz" | cut -d' ' -f1)"
  mkdir -p "$L/datasets/$DATASET_ID/images"
  [ "$(sha256sum "$MMPROJ" 2>/dev/null | cut -d' ' -f1)" = "$MMPROJ_SHA" ] || die "the mmproj sha256 is not $MMPROJ_SHA"
  [ -x "$LLAMA/build/bin/llama-quantize" ] || die "no llama-quantize in $LLAMA/build/bin"
  export PYTHONNOUSERSITE=1
  (cd "$SRC" && "$VENV/bin/python" -c "import lab.package, lab.train, peft, transformers, torch; print('imports ok: torch', torch.__version__, 'peft', peft.__version__)") || die "imports"
  (cd "$SRC" && "$VENV/bin/python" lab/dataset.py "$L/datasets/$DATASET_ID" --text-only --split) || die "the dataset does not check out"
  ls -la "$L/a100/adapter" "$L/datasets"
  wc -l "$L/datasets/rows-train.jsonl" "$L/datasets/rows-validation.jsonl"
  sha256sum "$L/datasets/rows-train.jsonl" "$L/datasets/rows-validation.jsonl" "$L/datasets/$DATASET_ID/split.json" "$L/a100/adapter/adapter_model.safetensors"
  echo "### prep rc=0"
}

package() {
  local version="${1:?version}" adapter="${2:?adapter: a100, pod:<run id> or a folder}" run_id="${3:?run id}" t0 rc
  case "$adapter" in       # run-script quotes its arguments, so a path under the pod's home is named, not written
    a100) adapter="$L/a100/adapter" ;;
    pod:*) adapter="$HOME/adapters/l16v2-${adapter#pod:}/adapter" ;;
  esac
  [ -f "$adapter/adapter_config.json" ] || die "no adapter at $adapter"
  export PYTHONNOUSERSITE=1 LAB_STATE_DIR="$L/state" LAB_TASK=L16
  [ -e "$HOME/packages/$version" ] && die "$HOME/packages/$version exists: a version is built once"
  t0=$(date +%s)
  (cd "$SRC" && "$VENV/bin/python" -m lab package build --adapter "$adapter" --run-id "$run_id" --task "$TASK" \
    --mmproj "$MMPROJ" --base-dir "$MODEL" --llama-cpp "$LLAMA" --out "$HOME/packages" --version "$version") 2>&1 | tail -30
  rc=${PIPESTATUS[0]}
  echo "### package rc=$rc wall_s=$(( $(date +%s) - t0 ))"
  cp "$L/state/ledger.jsonl" "$L/ledger.jsonl" 2>/dev/null
  ls -la "$HOME/packages/$version" 2>&1
  sha256sum "$HOME/packages/$version"/* 2>&1
  cat "$HOME/packages/$version/manifest.json" 2>&1; echo
  return "$rc"
}

start_one() {
  local name="$1" port="$2" model="$3" mmproj="$4" bin pid
  bin="$(server_bin)" || die "no HIP llama-server found"
  [ "$(port_state "$port")" = "free" ] || die "port $port is busy"
  # jetson/atlas-llm.sh's args plus --top-k 1 (greedy), exactly as the L4 run served both arms.
  nohup "$bin" -m "$model" --jinja --host 127.0.0.1 --port "$port" -c 4096 -ngl 99 -np 1 --reasoning-budget 0 \
    --cache-ram 0 --top-k 1 --mmproj "$mmproj" > "$L/llama-server-$port.log" 2>&1 &
  pid=$!
  echo "$pid" > "$L/llama-server-$port.pid"
  echo "$name: llama-server pid $pid on $port ($bin)"
}

wait_up() {
  local port="$1" pid i
  pid="$(cat "$L/llama-server-$port.pid")"
  for i in $(seq 1 90); do
    python3 -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:' + sys.argv[1] + '/health', timeout=5)" "$port" 2>/dev/null \
      && { echo "up on $port after about $((i * 2)) s"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "llama-server on $port exited; the end of its log:"; tail -15 "$L/llama-server-$port.log"; return 1; }
    sleep 2
  done
  echo "llama-server on $port did not come up in 180 s"; tail -15 "$L/llama-server-$port.log"; return 1
}

serve() {
  local version="${1:?version}" cand busy got
  cand="$HOME/packages/$version/$version-Q4_K_M.gguf"
  busy="$(card_users)"
  [ -z "$busy" ] || { echo "the card is in use; not starting:"; echo "$busy"; exit 3; }
  [ -s "$cand" ] || die "no package GGUF at $cand"
  got="$(sha256sum "$BEFORE" | cut -d' ' -f1)"
  [ "$got" = "$BEFORE_SHA" ] || die "the before GGUF sha256 is $got, not $BEFORE_SHA"
  echo "BEFORE(samepath) sha256 ok $got"
  echo "CANDIDATE sha256 $(sha256sum "$cand" | cut -d' ' -f1) bytes $(stat -c %s "$cand")"
  start_one before "$PORT_BEFORE" "$BEFORE" "$HOME/packages/$version/mmproj-F16.gguf"
  start_one candidate "$PORT_CAND" "$cand" "$HOME/packages/$version/mmproj-F16.gguf"
  wait_up "$PORT_BEFORE" || return 1
  wait_up "$PORT_CAND" || return 1
  echo "### serve rc=0"
  check
}

check() {
  python3 - "$PORT_BEFORE" "$PORT_CAND" <<'PY'
import hashlib, json, sys, urllib.request
def get(port, path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20))
def chat(port):
    body = json.dumps({"messages": [{"role": "user", "content": "Name three colours of the rainbow."}], "max_tokens": 48,
                       "temperature": 0.2}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=120))
    return out["choices"][0]["message"], out.get("timings", {})
for port in sys.argv[1:]:
    props = get(port, "/props")
    settings = props["default_generation_settings"]
    params = settings.get("params", settings)
    template = props.get("chat_template") or ""
    first, timings = chat(port)
    second, _ = chat(port)
    thinking = bool(first.get("reasoning_content")) or "<think>" in (first.get("content") or "")
    print(f"{port}: top_k={params.get('top_k')} temperature={params.get('temperature')} n_ctx={settings.get('n_ctx')} "
          f"model={(props.get('model_path') or '').split('/')[-1]} chat_template_sha256={hashlib.sha256(template.encode()).hexdigest()[:16]} "
          f"thinking_text={'YES' if thinking else 'none'} same_answer_twice={first.get('content') == second.get('content')} "
          f"tok_s={timings.get('predicted_per_second')}")
PY
  echo "### check rc=$?"
}

alive() { kill -0 "$1" 2>/dev/null && [ "$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)" != "Z" ]; }

stop_port() {
  local port="$1" pid i
  pid="$(cat "$L/llama-server-$port.pid" 2>/dev/null)"
  if [ -z "$pid" ] || ! alive "$pid"; then pid="$(pgrep -f "llama-server.*--port $port" | head -1)"; fi
  [ -n "$pid" ] || { echo "no llama-server on $port"; return 0; }
  kill "$pid" 2>/dev/null
  for i in $(seq 1 20); do alive "$pid" || break; sleep 1; done
  alive "$pid" && { kill -9 "$pid" 2>/dev/null; sleep 2; }
  alive "$pid" && echo "llama-server $pid on $port STILL RUNNING" || { echo "llama-server on $port stopped"; rm -f "$L/llama-server-$port.pid"; }
}

stop() {
  local port
  for port in "$PORT_BEFORE" "$PORT_CAND"; do stop_port "$port"; done
  echo "port $PORT_BEFORE: $(port_state $PORT_BEFORE)  port $PORT_CAND: $(port_state $PORT_CAND)"
  echo "engines left on the card:"; card_users | sed 's/^/  /'; echo "  (end)"
  awk '/MemTotal/ {t=$2} /MemAvailable/ {a=$2} END {printf "mem: %.1f GB total, %.1f GB available\n", t/1048576, a/1048576}' /proc/meminfo
  rocm-smi --showmemuse 2>/dev/null | grep -E 'VRAM%|GPU Memory' | sed 's/^/  /'
  echo "### stop rc=0"
}

train() {
  local run_id="${1:-}" out log rc started ended
  [ -n "$run_id" ] || die "usage: pod_run.sh train <run id> [trainer flags]"
  shift
  [ -d "$L/datasets/$DATASET_ID" ] && [ -f "$L/datasets/rows-train.jsonl" ] && [ -f "$L/datasets/rows-validation.jsonl" ] \
    || die "no dataset under $L/datasets: run prep (the trainer deletes its copy when it ends)"
  [ -x "$VENV/bin/python" ] && [ -f "$SRC/lab/train.py" ] || die "run prep first"
  out="$HOME/adapters/l16v2-$run_id"
  [ -e "$out" ] && die "$out exists: a run id is used once"
  mkdir -p "$out"
  log="$out/train.log"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LAB_STATE_DIR="$L/state" LAB_TASK=L16
  started="$(date +%s)"
  echo "### L16v2 run $run_id start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(cat "$SRC/SOURCE_COMMIT")" | tee -a "$log"
  set -- --dataset "$L/datasets/$DATASET_ID" --rows "$L/datasets/rows-train.jsonl" \
    --val-rows "$L/datasets/rows-validation.jsonl" --model "$MODEL" --base-repo "$BASE_REPO" \
    --base-revision "$BASE_REVISION" --out "$out" --run-id "$run_id" "$@"
  echo "### command: python lab/train.py $*" | tee -a "$log"
  (cd "$SRC" && timeout -s INT -k 120 "$TRAIN_TIMEOUT_S" "$VENV/bin/python" -u lab/train.py "$@") 2>&1 | tee -a "$log" | grep -v "^{'loss'" | tail -40
  rc=${PIPESTATUS[0]}
  ended="$(date +%s)"
  cp "$L/state/ledger.jsonl" "$out/ledger.jsonl" 2>/dev/null || echo "no ledger rows to copy" | tee -a "$log"
  rm -rf "$out/trainer"
  echo "### L16v2 run $run_id end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc wall_s=$((ended - started))" | tee -a "$log"
  ls -la "$out" "$out/adapter" 2>&1 | tee -a "$log"
  return "$rc"
}

cleanup() {
  rm -rf "$L"
  local id; for id in "$@"; do rm -f "$HOME/lab_run_$id.sh" "$HOME/lab_run_$id.out"; done
  ls "$HOME/packages" "$HOME/adapters" 2>&1
  echo "cleanup done"
}

case "$STEP" in
  pack) shift; pack "$@" ;;
  facts) facts ;;
  prep) prep ;;
  package) shift; package "$@" ;;
  serve) shift; serve "$@" ;;
  check) check ;;
  stop) stop ;;
  train) shift; train "$@" ;;
  cleanup) shift; cleanup "$@" ;;
  *) sed -n '2,24p' "$0"; [ "$STEP" = help ] || exit 2 ;;
esac
