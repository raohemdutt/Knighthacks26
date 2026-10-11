#!/bin/bash
# L16: the first real fine-tune of the student (Qwen/Qwen3.5-4B) on the trainer pod, pod B's role: SJC account D,
# Strix Halo (Radeon 8060S, gfx1151), ROCm 7.13, torch 2.9.1+rocm7.13.0 from the pod image. The config and the
# source of every number in it are in lab/L16_CONFIG.md; this file holds the exact commands.
#
# On the workstation, from the repo root (Git Bash):
#   bash lab/l16_run.sh dataset           the labeled train rows and their dataset folder, into $L16_DATA
#   bash lab/l16_run.sh bundle            lab/ and contracts/ at HEAD in one tar.gz for the pod, into $L16_DATA
#   bash lab/l16_run.sh dry-run           the same trainer command line on the CPU with the tiny fixture and a tiny model
# On the pod, through the controller (python -m lab.controller run-script --account D lab/l16_run.sh <step> ...):
#   prep                                  unpack the bundle; the training venv over the image's torch; the base model
#   train <run id> <dataset id> [flags]   inside the GPU Queue slot: the run; adapter, telemetry, run.json, the log
#                                         and the ledger rows under ~/adapters/l16-<run id>/ (flags go to the trainer)
#   cleanup                               remove ~/l16 (bundle, source, ledger copy) and this task's job files
#
# The pod keeps afterwards: ~/adapters/l16-<run id>/ (the product), ~/venvs/atlas-train (the pinned training stack,
# installed over the image's torch, never replacing it) and ~/models/Qwen3.5-4B (the base at its pinned commit).
# The image's own Python and the user site under ~/.local are never written: pip runs only inside the venv, with
# PYTHONNOUSERSITE=1 so the user site (where pod D keeps vLLM and its numpy) is neither read nor changed.
set -u -o pipefail
STEP="${1:-help}"

# The run's config (lab/L16_CONFIG.md says where each comes from). These are R36's pinned baseline, the trainer's
# own defaults, written out so the command line states them.
BASE_REPO="Qwen/Qwen3.5-4B"
BASE_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"   # lab/format_traces.py TEMPLATE_REVISION: rows and base agree
EPOCHS=3
LR=2e-4
RANK=16
ALPHA=32
DROPOUT=0
BATCH=4
GRAD_ACCUM=4
SEED=42
TRAIN_TIMEOUT_S="${TRAIN_TIMEOUT_S:-2400}"   # 40 minutes: with prep inside the slot, the run is interrupted (SIGINT) before the 60-minute slot ends

# Pod paths. On SJC the Jupyter root is /home/jovyan, so what the transfer pushes to l16/... lands in ~/l16/...
L16="$HOME/l16"
SRC="$L16/src"
VENV="$HOME/venvs/atlas-train"
MODEL="$HOME/models/Qwen3.5-4B"

die() { echo "l16_run: $*" >&2; exit 1; }

# --- on the workstation ---------------------------------------------------------------------------

# The Python of the clone's venv on Windows or elsewhere.
local_python() {
  if [ -x .venv/Scripts/python ]; then echo .venv/Scripts/python
  elif [ -x .venv/bin/python ]; then echo .venv/bin/python
  else command -v python3 || command -v python; fi
}

# L13's labels when they are in the repo (fixtures/learning/tool-calling-labels/, a dataset folder or one level down);
# otherwise the reference traces of the no-camera fixture's expected.jsonl, which learning ruling 12 allows. Only the
# train side of the frozen split gets a label: the held-out examples go to the pod without one.
dataset() {
  local py data labels folder id
  py="$(local_python)"
  data="${L16_DATA:-../L16-data}"
  labels="${L16_LABELS:-fixtures/learning/tool-calling-labels}"   # L16_LABELS: the folder read from L13's branch
  mkdir -p "$data"
  folder=""
  if [ -f "$labels/manifest.json" ]; then folder="$labels"
  elif [ -d "$labels" ]; then folder="$(dirname "$(ls -d "$labels"/*/manifest.json 2>/dev/null | head -n 1)")"; fi
  if [ -n "$folder" ] && [ -f "$folder/manifest.json" ]; then
    id="$("$py" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['dataset_id'])" "$folder/manifest.json")"
    echo "labels: L13's teacher labels in $folder (dataset $id)"
    rm -rf "${data:?}/$id"
    mkdir -p "$data/$id"
    # The dataset itself only; the labeler's own records (traces, its ledger and manifest, rejects) stay here.
    cp -r "$folder/manifest.json" "$folder/examples.jsonl" "$folder/split.json" "$folder/images" "$data/$id/"
    rm -f "$data/$id/images/.gitkeep"   # the contents API refuses a hidden file, and a text dataset has no images
  else
    id=tool-calling-fixture
    echo "labels: L13's labels are not in the repo; the reference traces of fixtures/learning/$id/expected.jsonl (learning ruling 12)"
    rm -rf "${data:?}/$id"
    cp -r "fixtures/learning/$id" "$data/$id"
    rm -f "$data/$id/expected.jsonl"   # the gold stays on the workstation; the dataset is manifest, examples, images
    rm -f "$data/$id/images/.gitkeep"
    "$py" - "$data/$id" "fixtures/learning/$id/expected.jsonl" <<'PY'
import json
import sys
from pathlib import Path

from lab.dataset import load_split, write_targets

folder, expected = Path(sys.argv[1]), Path(sys.argv[2])
split = load_split(folder, require_images=False)
train = {example.id for example in split.train}
labels = {}
for line in expected.read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        if row["id"] in train:
            labels[row["id"]] = json.dumps(row["label"])   # the label text as a teacher answer is stored
written = write_targets(folder, labels)
print(json.dumps({"train": len(train), "labeled": len(labels), "written": list(written)}))
PY
  fi
  "$py" lab/dataset.py "$data/$id" --text-only || die "the dataset folder does not check out"
  set -- "$data/$id" --out "$data/rows.jsonl" --split
  [ -n "${TOKENIZER:-}" ] && set -- "$@" --tokenizer "$TOKENIZER"
  "$py" lab/format_traces.py "$@" || die "format_traces wrote no rows"
  echo "dataset: $data/$id  rows: $data/rows.jsonl"
}

# lab/ and contracts/ as committed at HEAD (no lab/refine/, no runtime state), for the pod, which cannot clone the repo.
bundle() {
  local data
  data="${L16_DATA:-../L16-data}"
  mkdir -p "$data"
  git archive --format=tar.gz -o "$data/l16-bundle.tar.gz" HEAD lab contracts ':(exclude)lab/refine' \
    || die "git archive failed"
  echo "bundle: $data/l16-bundle.tar.gz at $(git rev-parse HEAD)"
}

# The trainer's command line on the CPU: the tiny fixture (six train rows) and a two-layer model of the student's own
# architecture with random weights, built the way tests/test_lab_train_cpu.py builds it. Nothing on the pod is touched.
dry_run() {
  local py data workers=()
  py="$(local_python)"
  data="${L16_DATA:-../L16-data}/dry-run"
  rm -rf "$data"
  mkdir -p "$data/datasets"
  cp -r fixtures/learning/train-tiny/tool-calling-tiny "$data/datasets/tool-calling-tiny"
  cp fixtures/learning/train-tiny/rows.jsonl "$data/datasets/rows.jsonl"
  "$py" - "$data/tiny-qwen3.5" <<'PY' || die "could not build the tiny model"
import sys
from pathlib import Path

import torch
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tests.test_lab_train_cpu import TEXT_CONFIG, downloaded, rows_text, small_tokenizer

folder = Path(sys.argv[1])
tokenizer = small_tokenizer(rows_text())
tokenizer.save_pretrained(folder)
torch.manual_seed(0)
config = Qwen3_5TextConfig(vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
                           pad_token_id=tokenizer.pad_token_id, **TEXT_CONFIG)
Qwen3_5ForCausalLM(config).save_pretrained(folder)
downloaded(folder)
print(f"tiny model: {folder}")
PY
  # Windows has no forkserver start method, so DataLoader workers are off there; the pod keeps the pinned two.
  "$py" -c "import multiprocessing as m, sys; sys.exit(0 if 'forkserver' in m.get_all_start_methods() else 1)" \
    || workers=(--workers 0)
  LAB_STATE_DIR="$data/state" LAB_TASK=L16 "$py" lab/train.py --cpu --no-install \
    --dataset "$data/datasets/tool-calling-tiny" --rows "$data/datasets/rows.jsonl" \
    --model "$data/tiny-qwen3.5" --base-repo "$BASE_REPO" --base-revision 0123456789abcdef0123456789abcdef01234567 \
    --out "$data/run" --run-id "l16-dry-run" --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH" \
    --grad-accum "$GRAD_ACCUM" --rank "$RANK" --alpha "$ALPHA" --dropout "$DROPOUT" --seed "$SEED" "${workers[@]}"
}

# --- on the pod ------------------------------------------------------------------------------------

# Unpack the bundle, make the venv over the image's torch, install the pins with the trainer's own install (a
# train-install ledger row), and download the base at its pinned commit. Safe to run again.
prep() {
  local before after where
  [ -f "$L16/l16-bundle.tar.gz" ] || die "no $L16/l16-bundle.tar.gz: push it first (python -m lab.transfer push-file)"
  rm -rf "$SRC"
  mkdir -p "$SRC" "$L16/state"
  tar -xzf "$L16/l16-bundle.tar.gz" -C "$SRC" || die "could not unpack the bundle"
  # git archive writes the commit into the tar's pax header (no git needed on the pod to read it).
  python3 -c 'import sys, tarfile; print(tarfile.open(sys.argv[1]).pax_headers.get("comment", "unknown"))'     "$L16/l16-bundle.tar.gz" > "$SRC/SOURCE_COMMIT" 2>/dev/null || echo unknown > "$SRC/SOURCE_COMMIT"
  echo "source: $(cat "$SRC/SOURCE_COMMIT")"
  before="$(PYTHONNOUSERSITE=1 python3 -c 'import torch; print(torch.__version__, torch.__file__)')" \
    || die "the image's torch does not import"
  echo "image torch: $before"
  if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv --system-site-packages "$VENV" 2>/dev/null \
      || python3 -m venv --system-site-packages --without-pip "$VENV" || die "could not make $VENV"
  fi
  export PYTHONNOUSERSITE=1
  "$VENV/bin/python" -m pip --version || die "no pip reachable from $VENV"
  (cd "$SRC" && LAB_STATE_DIR="$L16/state" LAB_TASK=L16 "$VENV/bin/python" -c \
    'import json; from lab import train; answer = train.install(); print(json.dumps(answer)); raise SystemExit(0 if answer.get("ok") else 1)') \
    || die "the pinned training stack did not install"
  after="$("$VENV/bin/python" -c 'import torch; print(torch.__version__, torch.__file__)')" || die "torch broke"
  [ "$after" = "$before" ] || die "torch changed: $before -> $after"
  where="$("$VENV/bin/python" -c 'import trl, peft, transformers; print(trl.__version__, peft.__version__, transformers.__version__, trl.__file__)')"
  echo "venv torch: $after"
  echo "venv stack: $where"
  BASE_REPO="$BASE_REPO" BASE_REVISION="$BASE_REVISION" MODEL="$MODEL" "$VENV/bin/python" - <<'PY' || die "the base model did not download"
import os
import time

from huggingface_hub import snapshot_download

started = time.monotonic()
path = snapshot_download(repo_id=os.environ["BASE_REPO"], revision=os.environ["BASE_REVISION"],
                         local_dir=os.environ["MODEL"])
print(f"base model: {path} in {time.monotonic() - started:.0f} s")
PY
  # The commit the download recorded is the one the trainer will put in its row; it must be the pinned one.
  where="$(head -qn 1 "$MODEL"/.cache/huggingface/download/*.metadata 2>/dev/null | sort -u | tr '\n' ' ')"
  [ "$where" = "$BASE_REVISION " ] || die "the model folder records commit(s) '$where', not $BASE_REVISION"
  du -sh "$MODEL"
  echo "prep done"
}

# The run. The trainer deletes the dataset copy and the rows when it ends (lab/train.py); the adapter, telemetry,
# run.json, the log and a copy of the ledger rows stay in ~/adapters/l16-<run id>/.
train() {
  local run_id="${1:-}" dataset_id="${2:-}" out log rc started ended extra=()
  [ -n "$run_id" ] && [ -n "$dataset_id" ] || die "usage: l16_run.sh train <run id> <dataset id> [trainer flags]"
  shift 2
  extra=("$@")   # e.g. --torch-loss-head if Triton cannot build TRL's loss kernel here; the row records either way
  [ -d "$L16/datasets/$dataset_id" ] && [ -f "$L16/datasets/rows.jsonl" ] || die "no dataset under $L16/datasets"
  [ -x "$VENV/bin/python" ] && [ -f "$SRC/lab/train.py" ] || die "run prep first"
  out="$HOME/adapters/l16-$run_id"
  [ -e "$out" ] && die "$out exists: a run id is used once"
  mkdir -p "$out"
  log="$out/train.log"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LAB_STATE_DIR="$L16/state" LAB_TASK=L16
  started="$(date +%s)"
  echo "### L16 run $run_id start $(date -u +%Y-%m-%dT%H:%M:%SZ) source $(cat "$SRC/SOURCE_COMMIT")" | tee -a "$log"
  set -- --dataset "$L16/datasets/$dataset_id" --rows "$L16/datasets/rows.jsonl" --model "$MODEL" \
    --base-repo "$BASE_REPO" --base-revision "$BASE_REVISION" --out "$out" --run-id "$run_id" \
    --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH" --grad-accum "$GRAD_ACCUM" --rank "$RANK" --alpha "$ALPHA" \
    --dropout "$DROPOUT" --seed "$SEED" "${extra[@]}"
  echo "### command: python lab/train.py $*" | tee -a "$log"
  (cd "$SRC" && timeout -s INT -k 120 "$TRAIN_TIMEOUT_S" "$VENV/bin/python" -u lab/train.py "$@") 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  ended="$(date +%s)"
  cp "$L16/state/ledger.jsonl" "$out/ledger.jsonl" 2>/dev/null || echo "no ledger rows to copy" | tee -a "$log"
  rm -rf "$out/trainer"   # TRL's output_dir; save_strategy is no, so it holds nothing the run needs
  echo "### L16 run $run_id end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc wall_s=$((ended - started))" | tee -a "$log"
  ls -la "$out" "$out/adapter" 2>&1 | tee -a "$log"
  return "$rc"
}

# What this task put on the pod besides the adapter, the venv and the model: the bundle, the unpacked source, the
# ledger copy (already inside the adapter folder) and the job files run-script left in the Jupyter root.
cleanup() {
  rm -rf "$L16"
  local ids="${*:-}" id
  for id in $ids; do rm -f "$HOME/lab_run_$id.sh" "$HOME/lab_run_$id.out"; done
  ls -la "$HOME/adapters" 2>&1
  echo "cleanup done"
}

case "$STEP" in
  dataset) dataset ;;
  bundle) bundle ;;
  dry-run) dry_run ;;
  prep) prep ;;
  train) shift; train "$@" ;;
  cleanup) shift; cleanup "$@" ;;
  *) sed -n '2,20p' "$0"; [ "$STEP" = help ] || exit 2 ;;
esac
