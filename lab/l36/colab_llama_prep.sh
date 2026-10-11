#!/bin/bash
# L36: stage the packaging toolchain on the Colab VM while the trainings run, as lab/l5_run.sh prep does on pod D:
# llama.cpp at the commit the first package used (lab/L5_RESULT.md), llama-quantize built CPU-only into
# /content/llama.cpp/build (where lab/package.py looks), and the base mmproj at its pinned revision, sha256 checked.
# Writes /content/l36/llama/READY (or FAILED with the reason). Used only if a new adapter is chosen.
set -u
OUT=/content/l36/llama; LLAMA=/content/llama.cpp
COMMIT=23b0202a189c44a54625aadcb37a946dd1d6278d
MMPROJ_DIR=/content/models/Qwen3.5-4B-GGUF; MMPROJ_REPO=unsloth/Qwen3.5-4B-GGUF
MMPROJ_REV=e87f176479d0855a907a41277aca2f8ee7a09523
MMPROJ_SHA=cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864
mkdir -p "$OUT"; rm -f "$OUT/READY" "$OUT/FAILED"
fail() { echo "$*" > "$OUT/FAILED"; exit 1; }
t0=$(date +%s)
if [ ! -d "$LLAMA/.git" ]; then
  git init -q "$LLAMA" && git -C "$LLAMA" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$LLAMA" fetch -q --depth 1 origin "$COMMIT" > "$OUT/fetch.log" 2>&1 || fail "fetch of llama.cpp $COMMIT"
git -C "$LLAMA" checkout -q FETCH_HEAD || fail "checkout"
[ "$(git -C "$LLAMA" rev-parse HEAD)" = "$COMMIT" ] || fail "llama.cpp is not at $COMMIT"
nice cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > "$OUT/cmake.log" 2>&1 \
  || fail "cmake configure"
nice cmake --build "$LLAMA/build" --config Release --target llama-quantize -j 6 >> "$OUT/cmake.log" 2>&1 || fail "cmake build"
[ -x "$LLAMA/build/bin/llama-quantize" ] || fail "no llama-quantize"
t1=$(date +%s)
python3 -c "from huggingface_hub import hf_hub_download as d; print(d('$MMPROJ_REPO', 'mmproj-F16.gguf', revision='$MMPROJ_REV', local_dir='$MMPROJ_DIR'))" \
  > "$OUT/mmproj.log" 2>&1 || fail "mmproj download"
[ "$(sha256sum "$MMPROJ_DIR/mmproj-F16.gguf" | cut -d' ' -f1)" = "$MMPROJ_SHA" ] || fail "mmproj sha256 is not $MMPROJ_SHA"
python3 -c "import sys; sys.path.insert(0, '$LLAMA/gguf-py'); import gguf; print('gguf-py ok')" > "$OUT/gguf.log" 2>&1 || fail "gguf-py import"
echo "llama.cpp $COMMIT build_s=$((t1 - t0)) mmproj $MMPROJ_SHA total_s=$(( $(date +%s) - t0 ))" > "$OUT/READY"
