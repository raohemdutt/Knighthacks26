#!/bin/bash
# Serve test of a low-bit teacher (3-bit or 2-bit compressed-tensors, from lab/quantize_teacher.py) on a Strix Halo pod.
#
# Usage on the pod (the exclusive pod, account D, in a GPU Queue slot; never the teacher's pod):
#   WIDTH=3 HUB_REPO=<namespace>/Qwen3.8-27B-W3A16-G128-GPTQ-atlas bash lab/serve_lowbit.sh [all|download|patch|bench]
#   DRY_RUN=1 WIDTH=3 HUB_REPO=... bash lab/serve_lowbit.sh all     # print every command, run none (any machine)
#
#   download  hf download of the checkpoint into ~/models/<name> (HF_TOKEN from the environment for a private repo;
#             never printed).
#   patch     apply lab/vllm-lowbit.diff to the installed vLLM package of $VENV (Python-only, no rebuild); a second
#             run sees it applied and does nothing. The vLLM must be the upgraded one (lab/vllm_upgrade.sh, main
#             5281e4990, or later): b1388b1's loader does not accept 2 or 3 bits.
#   bench     lab/bench.py run: one cold spawn per repeat at batch 1 and 4, --dtype float16 (the ExLlama kernel that
#             serves these widths takes fp16 activations only), rows under lab/state/bench/<session>/rows.jsonl.
#   all       the three in order.
# Then, from the box: python lab/lowbit_report.py report --bench 3=<session dir> ... (see its docstring).
#
# Node gate: refuses to run (outside DRY_RUN) unless rocminfo reports gfx1151, and refuses aarch64 (the Jetson).
set -u -o pipefail
STEP="${1:-all}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
WIDTH="${WIDTH:-}"
HUB_REPO="${HUB_REPO:-}"
VENV="${VENV:-$HOME/venv-vllm-new}"
VLLM_DIR="${VLLM_DIR:-$HOME/vllm-new}"
MODELS="${MODELS:-$HOME/models}"
PATCH="${PATCH:-$HERE/vllm-lowbit.diff}"
NODE="${NODE:-pod-b}"                    # the ledger's name for the exclusive pod (account D)
BATCHES="${BATCHES:-1,4}"
REPEATS="${REPEATS:-3}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "serve_lowbit: $*" >&2; exit 2; }

case "$WIDTH" in
  3|2) ;;
  *) die "WIDTH must be 3 or 2" ;;
esac
[ -n "$HUB_REPO" ] || die "HUB_REPO is required (<namespace>/<checkpoint name> from lab/quantize_teacher.py)"
case "$HUB_REPO" in
  */*) ;;
  *) die "HUB_REPO must be <namespace>/<name>" ;;
esac
NAME="${HUB_REPO##*/}"
CKPT="$MODELS/$NAME"
MODEL_NAME="teacher-w$WIDTH"

run() {
  # Prints the command; runs it unless DRY_RUN=1.
  printf '+'; printf ' %q' "$@"; printf '\n'
  [ "$DRY_RUN" = "1" ] || "$@"
}

gate() {
  [ "$DRY_RUN" = "1" ] && { echo "node gate: skipped (DRY_RUN=1, nothing runs)"; return 0; }
  [ "$(uname -m)" != "aarch64" ] || die "refusing aarch64: this is the Jetson, not a pod"
  local gfx
  gfx="$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]\+')"
  [ "$gfx" = "gfx1151" ] || die "refusing node: rocminfo reports '${gfx:-nothing}', expected gfx1151 (Strix Halo pod)"
  [ -x "$VENV/bin/python" ] || die "no upgraded vLLM venv at $VENV (build it with lab/vllm_upgrade.sh build)"
  echo "node gate: $(hostname) gfx1151"
}

download() {
  run mkdir -p "$CKPT"
  run "$VENV/bin/hf" download "$HUB_REPO" --local-dir "$CKPT"
}

patch_vllm() {
  local site
  if [ "$DRY_RUN" = "1" ]; then
    site='<site-packages of $VENV>'
  else
    site="$("$VENV/bin/python" -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" \
      || die "vllm does not import in $VENV"
  fi
  if [ "$DRY_RUN" != "1" ] && (cd "$site" && git apply --check -R "$PATCH" >/dev/null 2>&1); then
    echo "patch: already applied in $site"
    return 0
  fi
  run git -C "$site" apply --check "$PATCH"
  run git -C "$site" apply "$PATCH"
}

bench() {
  local serve_cmd
  serve_cmd="[\"$VENV/bin/vllm\", \"serve\", \"{checkpoint}\", \"--served-model-name\", \"{model}\", \"--port\", \"{port}\", \"--max-model-len\", \"{max_len}\", \"--max-num-seqs\", \"{batch}\", \"--dtype\", \"float16\", \"--quantization\", \"compressed-tensors\", \"--gpu-memory-utilization\", \"$GPU_UTIL\"]"
  run "$VENV/bin/python" "$REPO_ROOT/lab/bench.py" run --engine vllm --model "$MODEL_NAME" --checkpoint "$CKPT" \
    --quantization "compressed-tensors-w${WIDTH}a16" --max-len "$MAX_LEN" --node "$NODE" --batches "$BATCHES" --no-64 \
    --repeats "$REPEATS" --python "$VENV/bin/python" --engine-dir "$VLLM_DIR" --patch "$PATCH" \
    --serve-cmd "$serve_cmd"
}

gate
case "$STEP" in
  download) download ;;
  patch) patch_vllm ;;
  bench) bench ;;
  all) download && patch_vllm && bench ;;
  *) die "unknown step '$STEP' (all, download, patch, bench)" ;;
esac
