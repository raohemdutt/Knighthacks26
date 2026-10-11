#!/bin/bash
# L10: the group fine-tune on the network GPU, pod B's role (SJC account D, Strix Halo, Radeon 8060S, gfx1151, ROCm 7.13,
# torch 2.9.1+rocm7.13.0 from the image). Runs `python -m lab group-finetune` over the Circle's pooled opt-in exports
# with the live teacher on pod C. lab/README.md "Group fine-tune (L10)" has the whole sequence, boxes included.
#
# On the workstation, from the repo root (Git Bash):
#   bash lab/pod_b_group_finetune.sh bundle          lab/ contracts/ bus/ runtime/ contexts/ config/ at HEAD, one tar.gz
# On pod D (python -m lab.controller run-script --account D lab/pod_b_group_finetune.sh <step> [args]):
#   prep                                             unpack the bundle into ~/l10/src; the verb's few packages into the
#                                                    training venv L16 made (~/venvs/atlas-train); check llama.cpp
#   roster <circle> <task> <member>[=<current version>] ...
#                                                    write ~/pool/<circle>/<task>/pool.json: every Circle member,
#                                                    uploader or not, and the model each box serves now
#   status <circle> <task>                           the plan (python -m lab group-finetune --dry-run): counts, steps
#   run <circle> <task> <run id> [verb flags]        INSIDE A GPU QUEUE SLOT (GPU_SLOT=<row id> must be set): the run
#   serve                                            the package server on 8300 (lab/package_server.py), for the boxes
#   cleanup                                          remove ~/l10 (bundle and source); packages and run outputs stay
#
# The teacher: TEACHER_URL (default the pod C proxy below) with its hub token in TEACHER_TOKEN, which the dispatcher
# exports into the job's environment; it is never written to a file or printed. The group brain (G4) shares that
# teacher and keeps priority: labeling asks one example at a time (LABEL_WORKERS=1) so a G4 question never waits
# behind a batch of ours.
#
# What stays on the pod: ~/data/packages/<version>/ (the group model, manifest.json, group_offer.json),
# ~/group-runs/<run id>/ (adapter, telemetry, run.json, gate scores, log, ledger copy). The pooled exports under
# ~/pool/<circle>/<task>/<member>/ and every copy of them are deleted by the verb on every way out; pool.json stays.
set -u -o pipefail
STEP="${1:-help}"

TEACHER_URL="${TEACHER_URL:-https://sjc.aupcloud.io/user/roscon01/proxy/8000}"
BASE_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"   # the student's pinned commit (lab/l16_run.sh)
LABEL_WORKERS="${LABEL_WORKERS:-1}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-3000}"   # 50 minutes inside a 60-minute slot
L10="$HOME/l10"
SRC="$L10/src"
VENV="$HOME/venvs/atlas-train"
MODEL="$HOME/models/Qwen3.5-4B"
MMPROJ="${MMPROJ:-$HOME/models/Qwen3.5-4B-GGUF/mmproj-F16.gguf}"
POOL="$HOME/pool"
# llama.cpp: the source tree (convert_hf_to_gguf.py, gguf-py, .git) and the build whose bin/ holds llama-server.
# On pod D today: LLAMA_SRC=~/e29/llama.cpp LLAMA_BUILD=~/e29/llama.cpp/build-hip (E29's HIP build, no llama-quantize).
# prep makes ~/l10/llama.cpp, a folder of links in the layout lab/package.py reads (<dir>/build/bin/...), and builds a
# CPU-only llama-quantize in ~/l10/llama-quant when the build has none (cmake from PATH, else pip's cmake in the
# venv). Nothing in LLAMA_SRC or LLAMA_BUILD is changed.
LLAMA_SRC="${LLAMA_SRC:-$HOME/llama.cpp}"
LLAMA_BUILD="${LLAMA_BUILD:-$LLAMA_SRC/build}"
LLAMA="$L10/llama.cpp"
# Smoke knobs: L10_STEPS caps the trainer's optimizer steps (--max-steps); GATE=fake scores with the stand-in gate
# (no model servers; recorded as gate fake, no metrics rows); MIN_HELD_OUT lowers the gate's floor to report a short
# held-out set (it then never deploys); TEACHER_EXPECT names what the teacher's checkpoint must hold.
L10_STEPS="${L10_STEPS:-}"
GATE="${GATE:-real}"
MIN_HELD_OUT="${MIN_HELD_OUT:-50}"
TEACHER_EXPECT="${TEACHER_EXPECT:-Qwen3.8-27B}"

die() { echo "pod_b_group_finetune: $*" >&2; exit 1; }

bundle() {
  local data="${L10_DATA:-../L10-data}"
  mkdir -p "$data"
  git archive --format=tar.gz -o "$data/l10-bundle.tar.gz" HEAD lab contracts bus runtime contexts config \
    ':(exclude)lab/refine/research' || die "git archive failed"
  echo "bundle: $data/l10-bundle.tar.gz at $(git rev-parse HEAD)"
  echo "push it: python -m lab.transfer push-file --account D $data/l10-bundle.tar.gz l10/l10-bundle.tar.gz"
}

prep() {
  [ -f "$L10/l10-bundle.tar.gz" ] || die "no $L10/l10-bundle.tar.gz: push it first"
  rm -rf "$SRC" && mkdir -p "$SRC" && tar -xzf "$L10/l10-bundle.tar.gz" -C "$SRC" || die "could not unpack the bundle"
  [ -x "$VENV/bin/python" ] || die "no $VENV: run lab/l16_run.sh prep first (it makes the training venv)"
  PYTHONNOUSERSITE=1 "$VENV/bin/python" -m pip install --quiet pydantic httpx jsonschema pyyaml \
    || die "pip could not install the verb's packages into $VENV"
  llama_links || die "llama.cpp tools missing under $LLAMA_SRC / $LLAMA_BUILD (set LLAMA_SRC and LLAMA_BUILD)"
  [ -f "$MODEL/config.json" ] || die "no base at $MODEL"
  [ -f "$MMPROJ" ] || die "no base projector at $MMPROJ (set MMPROJ)"
  (cd "$SRC" && PYTHONNOUSERSITE=1 "$VENV/bin/python" -m lab group-finetune -h > /dev/null) || die "the verb does not load"
  echo "prep done"
}

llama_links() {
  [ -f "$LLAMA_SRC/convert_hf_to_gguf.py" ] && [ -x "$LLAMA_BUILD/bin/llama-server" ] || return 1
  local quant="$LLAMA_BUILD/bin/llama-quantize" cmake_bin
  if [ ! -x "$quant" ]; then
    quant="$L10/llama-quant/bin/llama-quantize"
    if [ ! -x "$quant" ]; then
      cmake_bin="$(command -v cmake || true)"
      if [ -z "$cmake_bin" ]; then
        PYTHONNOUSERSITE=1 "$VENV/bin/python" -m pip install --quiet cmake || return 1
        cmake_bin="$VENV/bin/cmake"
      fi
      echo "building a CPU-only llama-quantize from $LLAMA_SRC into $L10/llama-quant"
      "$cmake_bin" -S "$LLAMA_SRC" -B "$L10/llama-quant" -DGGML_HIP=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
        > "$L10/llama-quant.log" 2>&1 \
        && "$cmake_bin" --build "$L10/llama-quant" --target llama-quantize -j 16 >> "$L10/llama-quant.log" 2>&1 \
        || { tail -5 "$L10/llama-quant.log"; return 1; }
    fi
  fi
  rm -rf "$LLAMA" && mkdir -p "$LLAMA/build/bin" || return 1
  ln -s "$LLAMA_SRC/convert_hf_to_gguf.py" "$LLAMA/convert_hf_to_gguf.py"
  ln -s "$LLAMA_SRC/gguf-py" "$LLAMA/gguf-py"
  ln -s "$LLAMA_SRC/.git" "$LLAMA/.git"
  ln -s "$LLAMA_BUILD/bin/llama-server" "$LLAMA/build/bin/llama-server"
  ln -s "$quant" "$LLAMA/build/bin/llama-quantize"
  echo "llama.cpp links: $LLAMA at $(git -C "$LLAMA" rev-parse HEAD)"
}

roster() {
  local circle="${1:-}" task="${2:-}"
  [ -n "$circle" ] && [ -n "$task" ] && [ $# -ge 3 ] || die "usage: roster <circle> <task> <member>[=<version>] ..."
  shift 2
  mkdir -p "$POOL/$circle/$task"
  PYTHONNOUSERSITE=1 "$VENV/bin/python" - "$POOL/$circle/$task/pool.json" "$circle" "$task" "$@" <<'PY'
import json, os, sys
path, circle, task, *members = sys.argv[1:]
old = {}
if os.path.exists(path):
    old = {m["member"]: m for m in json.load(open(path))["members"]}
body = {"version": 1, "circle": circle, "task": task, "members": []}
for item in members:
    name, _, version = item.partition("=")
    body["members"].append({"member": name, "current_version": version or "atlas-base",
                            "exports": old.get(name, {}).get("exports", {})})
open(path, "w").write(json.dumps(body, indent=2) + "\n")
print(json.dumps(body))
PY
}

status() {
  local circle="${1:-}" task="${2:-}"
  [ -n "$circle" ] && [ -n "$task" ] || die "usage: status <circle> <task>"
  (cd "$SRC" && PYTHONNOUSERSITE=1 LAB_STATE_DIR="$L10/state" "$VENV/bin/python" -m lab group-finetune \
    --circle "$circle" --task "$task" --pool "$POOL" --dry-run)
}

run() {
  local circle="${1:-}" task="${2:-}" run_id="${3:-}" out log rc
  [ -n "$circle" ] && [ -n "$task" ] && [ -n "$run_id" ] || die "usage: run <circle> <task> <run id> [verb flags]"
  shift 3
  [ -n "${GPU_SLOT:-}" ] || die "no GPU_SLOT: request and start a GPU Queue slot first (python -m lab slot ...)"
  [ -n "${TEACHER_TOKEN:-}" ] || die "no TEACHER_TOKEN in the environment: the dispatcher exports the pod C hub token"
  out="$HOME/group-runs/$run_id"
  [ -e "$out" ] && die "$out exists: a run id is used once"
  mkdir -p "$out"
  log="$out/run.log"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LAB_STATE_DIR="$L10/state" LAB_TASK=L10
  [ -x "$LLAMA/build/bin/llama-quantize" ] || die "no $LLAMA: run prep first"
  echo "### L10 run $run_id start $(date -u +%Y-%m-%dT%H:%M:%SZ) slot $GPU_SLOT" | tee -a "$log"
  (cd "$SRC" && timeout -s INT -k 120 "$RUN_TIMEOUT_S" "$VENV/bin/python" -m lab group-finetune \
    --circle "$circle" --task "$task" --pool "$POOL" --run-id "$run_id" --work "$out" \
    --packages "$HOME/data/packages" --teacher-url "$TEACHER_URL" --teacher-token-env TEACHER_TOKEN \
    --model "$MODEL" --mmproj "$MMPROJ" --llama-cpp "$LLAMA" --trainer-python "$VENV/bin/python" \
    --base-revision "$BASE_REVISION" --label-workers "$LABEL_WORKERS" --decode-attested \
    --gate "$GATE" --min-held-out "$MIN_HELD_OUT" --teacher-expect-model "$TEACHER_EXPECT" \
    ${L10_STEPS:+--max-steps "$L10_STEPS"} --metrics-file "$out/metrics.jsonl" "$@") 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  cp "$L10/state/ledger.jsonl" "$out/ledger.jsonl" 2>/dev/null || echo "no ledger rows to copy" | tee -a "$log"
  echo "### L10 run $run_id end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc" | tee -a "$log"
  ls -la "$POOL/$circle/$task" 2>&1 | tee -a "$log"
  return "$rc"
}

serve() {
  (cd "$SRC" && nohup python3 lab/package_server.py --root "$HOME/data/packages" --port 8300 \
    --log "$HOME/package_server.log" > /dev/null 2>&1 &)
  echo "package server on 8300 serving $HOME/data/packages"
}

cleanup() {
  rm -rf "$L10/src" "$L10/l10-bundle.tar.gz" "$LLAMA"
  echo "cleanup done; kept $L10/state, ~/group-runs and ~/data/packages"
}

case "$STEP" in
  bundle) bundle ;;
  prep) prep ;;
  roster) shift; roster "$@" ;;
  status) shift; status "$@" ;;
  run) shift; run "$@" ;;
  serve) serve ;;
  cleanup) cleanup ;;
  *) sed -n '2,26p' "$0" ;;
esac
