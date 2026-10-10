#!/bin/bash
# Pod B (trainer): training stack on the R9700 (ROCm 7.13), llama.cpp convert/quantize tools, student checkpoint, smoke LoRA.
# Idempotent and resumable. Usage on the pod:  bash lab/pod_b_trainer.sh [proxy|deps|llama|download|smoke|all]
set -u
export PATH="$HOME/.local/bin:$PATH"
STEP="${1:-all}"
MODELS="$HOME/models"
STUDENT="${STUDENT:-Qwen/Qwen3.5-4B}"
LAB_DIR="$(cd "$(dirname "$0")" && pwd)"

proxy() {
  python3 -c "import jupyter_server_proxy" 2>/dev/null && { echo "jupyter-server-proxy present"; return; }
  pip install --user --quiet jupyter-server-proxy
  echo "jupyter-server-proxy installed: relaunch the pod once so the extension loads"
}

deps() {
  # torch/torchvision/triton come from the pod image (2.9.1+rocm7.13.0); never reinstall them.
  pip install --user --quiet cmake ninja huggingface_hub gguf peft trl transformers accelerate datasets sentencepiece protobuf
  python3 -c "import torch, peft, trl, transformers; print('torch', torch.__version__, 'peft', peft.__version__, 'trl', trl.__version__, 'transformers', transformers.__version__)"
}

llama() {
  cd "$HOME"
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  [ -x build/bin/llama-quantize ] && { echo "llama-quantize present"; return; }
  # CMake's HIP language wants clang itself, not the hipcc wrapper.
  cmake -S . -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
  cmake --build build --config Release --target llama-quantize llama-cli llama-server -j 32
  pip install --user --quiet -r requirements/requirements-convert_hf_to_gguf.txt 2>/dev/null || true
}

download() {
  mkdir -p "$MODELS"
  local dir="$MODELS/$(basename "$STUDENT")"
  [ -f "$dir/config.json" ] && [ -z "$(find "$dir" -name '*.incomplete' 2>/dev/null)" ] && { echo "checkpoint present: $dir"; return; }
  hf download "$STUDENT" --local-dir "$dir"
}

smoke() {
  python3 "$LAB_DIR/smoke_lora.py" --model "$MODELS/$(basename "$STUDENT")" --steps "${STEPS:-20}" 2>&1 | tee "$HOME/smoke_lora.log" | grep -E "SMOKE_LORA|device|Error|error" | tail -5
}

case "$STEP" in
  proxy) proxy ;;
  deps) deps ;;
  llama) llama ;;
  download) download ;;
  smoke) smoke ;;
  all) proxy; deps; llama; download; smoke ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
