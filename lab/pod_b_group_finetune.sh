#!/bin/bash
# L10: the group fine-tune on the network GPU, pod B's role (SJC account D, Strix Halo, Radeon 8060S, gfx1151, ROCm 7.13,
# torch 2.9.1+rocm7.13.0 from the image). Runs `python -m lab group-finetune` over the Circle's pooled opt-in exports
# with the live teacher on pod C. lab/README.md "Group fine-tune (L10)" has the whole sequence, boxes included.
#
# On the workstation, from the repo root (Git Bash):
#   bash lab/pod_b_group_finetune.sh bundle          lab/ contracts/ bus/ runtime/ contexts/ config/ at HEAD, one tar.gz
# On pod D (python tooling/dispatcher/pod_exec.py D run lab/pod_b_group_finetune.sh <name>.sh, or the controller):
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
  [ -x "$HOME/llama.cpp/build/bin/llama-server" ] && [ -x "$HOME/llama.cpp/build/bin/llama-quantize" ] \
    || die "llama.cpp tools missing: bash lab/pod_b_trainer.sh llama"
  [ -f "$MODEL/config.json" ] || die "no base at $MODEL"
  [ -f "$MMPROJ" ] || die "no base projector at $MMPROJ (set MMPROJ)"
  (cd "$SRC" && PYTHONNOUSERSITE=1 "$VENV/bin/python" -m lab group-finetune -h > /dev/null) || die "the verb does not load"
  echo "prep done"
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
  echo "### L10 run $run_id start $(date -u +%Y-%m-%dT%H:%M:%SZ) slot $GPU_SLOT" | tee -a "$log"
  (cd "$SRC" && timeout -s INT -k 120 "$RUN_TIMEOUT_S" "$VENV/bin/python" -m lab group-finetune \
    --circle "$circle" --task "$task" --pool "$POOL" --run-id "$run_id" --work "$out" \
    --packages "$HOME/data/packages" --teacher-url "$TEACHER_URL" --teacher-token-env TEACHER_TOKEN \
    --model "$MODEL" --mmproj "$MMPROJ" --llama-cpp "$HOME/llama.cpp" --trainer-python "$VENV/bin/python" \
    --base-revision "$BASE_REVISION" --label-workers "$LABEL_WORKERS" --decode-attested \
    --metrics-file "$out/metrics.jsonl" "$@") 2>&1 | tee -a "$log"
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
  rm -rf "$L10/src" "$L10/l10-bundle.tar.gz"
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
  *) sed -n '2,25p' "$0" ;;
esac
