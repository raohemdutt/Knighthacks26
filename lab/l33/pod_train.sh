#!/bin/bash
# L33: train tool-calling-v3 (the set built from the box's own episodes) on pod D with Ilan1's v2 recipe. The pack,
# prep and train steps of lab/l16v2/pod_run.sh, cut to what a training run needs: no A100 adapter, no packaging and
# no serving (score with lab/l16v2/pod_run.sh package/serve and lab/l16v2/score_pod.sh, as v2 was).
#
# On the workstation, from the repo root (Git Bash):
#   bash lab/l33/pod_train.sh pack <dest dir>      l33-bundle.tar.gz (lab/ and contracts/ at HEAD) and l33-inputs.tar.gz
#                                                  (fixtures/learning/tool-calling-v3 without its gold, and the two
#                                                  rows files beside it)
#   then: python -m lab.transfer push-file --account D <dest>/l33-bundle.tar.gz l33/l33-bundle.tar.gz   (and inputs)
# On the pod, through the controller, inside a Running GPU Queue row only
# (python -m lab.controller run-script --account D lab/l33/pod_train.sh <step> ...):
#   prep                           unpack under ~/l33, check the venv imports and the dataset's split
#   train <run id> [trainer flags] lab/train.py on ~/l33/datasets (the trainer deletes that copy when it ends; run prep
#                                  again before another run); adapter, telemetry, validation.jsonl, run.json and the
#                                  log under ~/adapters/l33-<run id>/
#   cleanup                        remove ~/l33; adapters stay
#
# Ilan1's v2 recipe on v3 (lr 1e-4, rank 16, alpha 32, dropout 0.05, warmup 0.10, validation every 8 steps, keep-best):
#   python -m lab.controller run-script --account D --timeout 3500 lab/l33/pod_train.sh train v3-recipe \
#     --max-steps 62 --eval-steps 8 --lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 --warmup-ratio 0.10 \
#     --batch 4 --grad-accum 4 --keep-best --seed 42
set -u -o pipefail
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"
STEP="${1:-help}"
BASE_REPO="Qwen/Qwen3.5-4B"
BASE_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATASET_ID="tool-calling-v3"
L="$HOME/l33"; SRC="$L/src"; VENV="$HOME/venvs/atlas-train"
MODEL="$HOME/models/Qwen3.5-4B"
TRAIN_TIMEOUT_S="${TRAIN_TIMEOUT_S:-3300}"   # 55 minutes: the run is interrupted (SIGINT) before a 60-minute row ends
die() { echo "l33 pod_train: $*" >&2; exit 1; }

pack() {
  local dest="${1:?where the two tarballs go}" stage here
  here="$(git rev-parse --show-toplevel)" || die "run it inside the clone"
  mkdir -p "$dest"
  git -C "$here" archive --format=tar.gz -o "$dest/l33-bundle.tar.gz" HEAD lab contracts ':(exclude)lab/refine' \
    || die "git archive failed"
  stage="$dest/inputs"; rm -rf "$stage"; mkdir -p "$stage/datasets"
  cp -r "$here/fixtures/learning/$DATASET_ID" "$stage/datasets/" || die "no fixtures/learning/$DATASET_ID"
  mv "$stage/datasets/$DATASET_ID/rows-train.jsonl" "$stage/datasets/$DATASET_ID/rows-validation.jsonl" "$stage/datasets/"
  # The dataset as the trainer reads it: manifest, examples, split. The gold and the build records stay here.
  rm -f "$stage/datasets/$DATASET_ID"/{expected.jsonl,validation.json,build.json,source.json,teacher.jsonl,README.md} \
        "$stage/datasets/$DATASET_ID/images/.gitkeep"
  tar -czf "$dest/l33-inputs.tar.gz" -C "$stage" datasets || die "tar failed"
  rm -rf "$stage"
  echo "bundle: $dest/l33-bundle.tar.gz at $(git -C "$here" rev-parse HEAD) ($(stat -c %s "$dest/l33-bundle.tar.gz") bytes)"
  echo "inputs: $dest/l33-inputs.tar.gz ($(stat -c %s "$dest/l33-inputs.tar.gz") bytes, sha256 $(sha256sum "$dest/l33-inputs.tar.gz" | cut -d' ' -f1))"
}

prep() {
  [ -f "$L/l33-bundle.tar.gz" ] && [ -f "$L/l33-inputs.tar.gz" ] || die "push l33-bundle.tar.gz and l33-inputs.tar.gz to ~/l33 first"
  rm -rf "$SRC" "$L/datasets"
  mkdir -p "$SRC" "$L/state"
  tar -xzf "$L/l33-bundle.tar.gz" -C "$SRC" || die "could not unpack the bundle"
  python3 -c 'import sys, tarfile; print("source", tarfile.open(sys.argv[1]).pax_headers.get("comment", "unknown"))' "$L/l33-bundle.tar.gz" | tee "$SRC/SOURCE_COMMIT"
  tar -xzf "$L/l33-inputs.tar.gz" -C "$L" || die "could not unpack the inputs"
  echo "inputs sha256 $(sha256sum "$L/l33-inputs.tar.gz" | cut -d' ' -f1)"
  mkdir -p "$L/datasets/$DATASET_ID/images"
  export PYTHONNOUSERSITE=1
  (cd "$SRC" && "$VENV/bin/python" -c "import lab.train, peft, transformers, torch; print('imports ok: torch', torch.__version__, 'peft', peft.__version__)") || die "imports"
  (cd "$SRC" && "$VENV/bin/python" lab/dataset.py "$L/datasets/$DATASET_ID" --text-only --split) || die "the dataset does not check out"
  wc -l "$L/datasets/rows-train.jsonl" "$L/datasets/rows-validation.jsonl"
  sha256sum "$L/datasets/rows-train.jsonl" "$L/datasets/rows-validation.jsonl" "$L/datasets/$DATASET_ID/split.json"
  echo "### prep rc=0"
}

train() {
  local run_id="${1:-}" out log rc started ended
  [ -n "$run_id" ] || die "usage: pod_train.sh train <run id> [trainer flags]"
  shift
  [ -d "$L/datasets/$DATASET_ID" ] && [ -f "$L/datasets/rows-train.jsonl" ] && [ -f "$L/datasets/rows-validation.jsonl" ] \
    || die "no dataset under $L/datasets: run prep (the trainer deletes its copy when it ends)"
  [ -x "$VENV/bin/python" ] && [ -f "$SRC/lab/train.py" ] || die "run prep first"
  out="$HOME/adapters/l33-$run_id"
  [ -e "$out" ] && die "$out exists: a run id is used once"
  mkdir -p "$out"
  log="$out/train.log"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LAB_STATE_DIR="$L/state" LAB_TASK=L33
  started="$(date +%s)"
  echo "### L33 run $run_id start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(cat "$SRC/SOURCE_COMMIT")" | tee -a "$log"
  set -- --dataset "$L/datasets/$DATASET_ID" --rows "$L/datasets/rows-train.jsonl" \
    --val-rows "$L/datasets/rows-validation.jsonl" --model "$MODEL" --base-repo "$BASE_REPO" \
    --base-revision "$BASE_REVISION" --out "$out" --run-id "$run_id" "$@"
  echo "### command: python lab/train.py $*" | tee -a "$log"
  (cd "$SRC" && timeout -s INT -k 120 "$TRAIN_TIMEOUT_S" "$VENV/bin/python" -u lab/train.py "$@") 2>&1 | tee -a "$log" | grep -v "^{'loss'" | tail -40
  rc=${PIPESTATUS[0]}
  ended="$(date +%s)"
  cp "$L/state/ledger.jsonl" "$out/ledger.jsonl" 2>/dev/null || echo "no ledger rows to copy" | tee -a "$log"
  rm -rf "$out/trainer"
  echo "### L33 run $run_id end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc wall_s=$((ended - started))" | tee -a "$log"
  ls -la "$out" "$out/adapter" 2>&1 | tee -a "$log"
  return "$rc"
}

cleanup() {
  rm -rf "$L"
  ls "$HOME/adapters" 2>&1
  echo "cleanup done"
}

case "$STEP" in
  pack) shift; pack "$@" ;;
  prep) prep ;;
  train) shift; train "$@" ;;
  cleanup) cleanup ;;
  *) sed -n '2,25p' "$0"; [ "$STEP" = help ] || exit 2 ;;
esac
