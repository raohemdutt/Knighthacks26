#!/bin/bash
# Pod B (trainer): training stack on the pod's GPU (ROCm 7.13), llama.cpp convert/quantize tools, student checkpoint, smoke LoRA.
# The GPU target comes from rocminfo, so the same steps build for the Strix Halo (gfx1151, SJC account D, the lab today)
# and for the R9700 (gfx1201, DUB account B, the other chip); GFX=<target> overrides it.
# Idempotent and resumable. Usage on the pod:  bash lab/pod_b_trainer.sh [proxy|deps|llama|download|smoke|all]
# From the box: python -m lab.controller run-script --account B --with lab/smoke_lora.py lab/pod_b_trainer.sh smoke
# A step that fails ends the script with a non-zero code, so run-script's ok and its ledger row tell the truth.
set -u -o pipefail
export PATH="$HOME/.local/bin:$PATH"
STEP="${1:-all}"
MODELS="$HOME/models"
STUDENT="${STUDENT:-Qwen/Qwen3.5-4B}"
# smoke_lora.py sits beside this script: in lab/ of the clone, or in the Jupyter root when run-script uploads it with
# --with. LAB_DIR overrides; the clone at ~/Knighthacks26/lab is tried last.
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")" && pwd)}"
[ -f "$LAB_DIR/smoke_lora.py" ] || [ ! -f "$HOME/Knighthacks26/lab/smoke_lora.py" ] || LAB_DIR="$HOME/Knighthacks26/lab"

proxy() {
  python3 -c "import jupyter_server_proxy" 2>/dev/null && { echo "jupyter-server-proxy present"; return; }
  pip install --user --quiet jupyter-server-proxy || { echo "could not install jupyter-server-proxy"; return 1; }
  echo "jupyter-server-proxy installed: relaunch the pod once so the extension loads"
}

deps() {
  # torch/torchvision/triton come from the pod image (2.9.1+rocm7.13.0); never reinstall them.
  pip install --user --quiet cmake ninja huggingface_hub gguf peft trl transformers accelerate datasets sentencepiece protobuf \
    || { echo "pip could not install the training stack"; return 1; }
  python3 -c "import torch, peft, trl, transformers; print('torch', torch.__version__, 'peft', peft.__version__, 'trl', trl.__version__, 'transformers', transformers.__version__)"
}

gpu_arch() {
  # The first gfx target rocminfo lists: gfx1151 on the Strix Halo, gfx1201 on the R9700.
  if [ -n "${GFX:-}" ]; then echo "$GFX"; return; fi
  rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]\+'
}

llama() {
  local gfx
  gfx="$(gpu_arch)"
  [ -n "$gfx" ] || { echo "no gfx target from rocminfo; set GFX=<target>"; exit 1; }
  cd "$HOME"
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp || { echo "could not clone llama.cpp"; return 1; }
  cd llama.cpp || return 1
  [ -x build/bin/llama-quantize ] && { echo "llama-quantize present"; return; }
  echo "building llama.cpp for $gfx"
  # CMake's HIP language wants clang itself, not the hipcc wrapper.
  cmake -S . -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS="$gfx" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ || return 1
  cmake --build build --config Release --target llama-quantize llama-cli llama-server -j 32 || return 1
  pip install --user --quiet -r requirements/requirements-convert_hf_to_gguf.txt 2>/dev/null || true
}

download() {
  mkdir -p "$MODELS"
  local dir="$MODELS/$(basename "$STUDENT")"
  [ -f "$dir/config.json" ] && [ -z "$(find "$dir" -name '*.incomplete' 2>/dev/null)" ] && { echo "checkpoint present: $dir"; return; }
  hf download "$STUDENT" --local-dir "$dir"
}

smoke() {
  [ -f "$LAB_DIR/smoke_lora.py" ] || {
    echo "smoke_lora.py is not in $LAB_DIR: upload it beside the script (run-script --with lab/smoke_lora.py) or set LAB_DIR"
    return 2
  }
  local rc
  python3 "$LAB_DIR/smoke_lora.py" --model "$MODELS/$(basename "$STUDENT")" --steps "${STEPS:-20}" > "$HOME/smoke_lora.log" 2>&1
  rc=$?
  # The lines worth reading, and the end of the log when the run failed, whatever words its error uses.
  grep -E "SMOKE_LORA|device|Error|error" "$HOME/smoke_lora.log" | tail -5
  [ "$rc" -eq 0 ] || { echo "smoke_lora.py failed with code $rc; the end of $HOME/smoke_lora.log:"; tail -5 "$HOME/smoke_lora.log"; }
  return "$rc"
}

case "$STEP" in
  proxy) proxy ;;
  deps) deps ;;
  llama) llama ;;
  download) download ;;
  smoke) smoke ;;
  all) proxy && deps && llama && download && smoke ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
