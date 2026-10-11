#!/bin/bash
# L5 on pod D: package the L16 adapter (merge in bf16, GGUF, Q4_K_M, the base mmproj, manifest.json).
#   prep    unpack the bundle (lab/, contracts/, jetson/make_manifest.py at a main commit); build llama-quantize
#           CPU-only into ~/llama.cpp/build (where lab/package.py looks); fetch the base mmproj at a pinned commit;
#           check that the venv imports everything the build runs
#   build   python -m lab package build ... --out ~/packages (the folder R10's file server serves)
#   cleanup remove ~/l5 after the ledger copy is pulled
set -u -o pipefail
STEP="${1:-help}"
export PATH="$HOME/.local/bin:$PATH"
L5="$HOME/l5"; SRC="$L5/src"; VENV="$HOME/venvs/atlas-train"; LLAMA="$HOME/llama.cpp"
RUN_ID="train-20261011T032149Z-7dd098"; TASK="tool-calling"
ADAPTER="$HOME/adapters/l16-$RUN_ID/adapter"
MMPROJ_DIR="$HOME/models/Qwen3.5-4B-GGUF"; MMPROJ="$MMPROJ_DIR/mmproj-F16.gguf"
MMPROJ_REPO="unsloth/Qwen3.5-4B-GGUF"; MMPROJ_REV="e87f176479d0855a907a41277aca2f8ee7a09523"
MMPROJ_SHA="cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864"
export PYTHONNOUSERSITE=1 LAB_STATE_DIR="$L5/state" LAB_TASK=L5
die() { echo "l5_run: $*" >&2; exit 1; }

prep() {
  local t0 t1
  [ -f "$L5/l5-bundle.tar.gz" ] || die "no bundle"
  rm -rf "$SRC"; mkdir -p "$SRC" "$L5/state"
  tar -xzf "$L5/l5-bundle.tar.gz" -C "$SRC" || die "unpack"
  python3 -c 'import sys, tarfile; print("source", tarfile.open(sys.argv[1]).pax_headers.get("comment", "unknown"))' "$L5/l5-bundle.tar.gz"
  echo "llama.cpp $(git -C "$LLAMA" rev-parse HEAD)"
  if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
    command -v cmake >/dev/null || die "no cmake on PATH"
    t0=$(date +%s)
    env -u PYTHONNOUSERSITE cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > "$L5/cmake.log" 2>&1 || { tail -20 "$L5/cmake.log"; die "cmake configure"; }
    env -u PYTHONNOUSERSITE cmake --build "$LLAMA/build" --config Release --target llama-quantize -j 16 >> "$L5/cmake.log" 2>&1 || { tail -20 "$L5/cmake.log"; die "cmake build"; }
    t1=$(date +%s); echo "llama-quantize built CPU-only in $((t1 - t0)) s"
  fi
  ls -la "$LLAMA/build/bin/llama-quantize" || die "no llama-quantize"
  if [ "$(sha256sum "$MMPROJ" 2>/dev/null | cut -d' ' -f1)" != "$MMPROJ_SHA" ]; then
    t0=$(date +%s)
    "$VENV/bin/python" -c "from huggingface_hub import hf_hub_download as d; print(d('$MMPROJ_REPO', 'mmproj-F16.gguf', revision='$MMPROJ_REV', local_dir='$MMPROJ_DIR'))" || die "mmproj download"
    t1=$(date +%s); echo "mmproj downloaded in $((t1 - t0)) s"
  fi
  [ "$(sha256sum "$MMPROJ" | cut -d' ' -f1)" = "$MMPROJ_SHA" ] || die "the mmproj sha256 is not $MMPROJ_SHA"
  echo "mmproj sha256 $MMPROJ_SHA $(stat -c %s "$MMPROJ") bytes"
  (cd "$SRC" && "$VENV/bin/python" -c "import lab.package, lab.transfer, peft, transformers, torch; print('package imports ok')") || die "package imports"
  "$VENV/bin/python" "$LLAMA/convert_hf_to_gguf.py" --help > "$L5/convert-help.txt" 2>&1 || { tail -5 "$L5/convert-help.txt"; die "convert_hf_to_gguf.py does not start in the venv"; }
  echo "convert_hf_to_gguf.py starts"
  ls -la "$ADAPTER"
  echo "prep done"
}

build() {
  local t0 t1 version="atlas-$TASK-$RUN_ID"
  t0=$(date +%s)
  (cd "$SRC" && "$VENV/bin/python" -m lab package build --adapter "$ADAPTER" --run-id "$RUN_ID" --task "$TASK" \
    --mmproj "$MMPROJ" --base-dir "$HOME/models/Qwen3.5-4B" --llama-cpp "$LLAMA" --out "$HOME/packages") 2>&1 | tail -40
  local rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  echo "### build rc=$rc wall_s=$((t1 - t0))"
  cp "$L5/state/ledger.jsonl" "$L5/ledger.jsonl" 2>/dev/null
  ls -la "$HOME/packages/$version" 2>&1
  sha256sum "$HOME/packages/$version"/* 2>&1
  return "$rc"
}

cleanup() {
  rm -rf "$L5"
  local id; for id in "$@"; do rm -f "$HOME/lab_run_$id.sh" "$HOME/lab_run_$id.out"; done
  echo "cleanup done"
}

case "$STEP" in
  prep) prep ;;
  build) build ;;
  cleanup) shift; cleanup "$@" ;;
  *) echo "usage: prep | build | cleanup [job ids]"; exit 2 ;;
esac
