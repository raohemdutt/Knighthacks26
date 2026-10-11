#!/bin/bash
# Pod side of the R63 head-to-head: the lookup-table GEMV (lab/kernels/lut_gemv.py) against vLLM's
# RDNAHybridW4A16LinearKernel, run in the new build's isolated venv (~/venv-vllm-new, lab/vllm_upgrade.sh build) and
# left in ~/r63_dispatch. Run through the controller inside a GPU Queue slot, the files uploaded under an r63_ prefix
# so nothing another session keeps in the Jupyter root is overwritten:
#   python -m lab.controller run-script --account D --timeout 900 \
#     --with <stage>/r63_dispatch_bench.py --with <stage>/r63_lut_gemv.py --with <stage>/r63_chipsheet.py \
#     --with <stage>/r63_chipsheet_format.py --with <stage>/r63_pod_power.py lab/kernels/pod_dispatch_bench.sh \
#     bench --node pod-b
# Arguments go to dispatch_bench.py; it refuses unless rocminfo names gfx1151. Installs nothing, stops nothing.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/venv-vllm-new}"
OUT="$HOME/r63_dispatch"
echo "host $(hostname) $(uname -m) $(date -u +%FT%TZ)"
[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV: run lab/vllm_upgrade.sh build first"; exit 2; }
# The environment lab/vllm_upgrade.sh serve_env gives the new build: the venv's ROCm 7.14 SDK, not the image's 7.13,
# and no LD_LIBRARY_PATH (with it the process maps two HIP runtimes).
ROCM="$("$VENV/bin/rocm-sdk" path --root 2>/dev/null)"
if [ -n "$ROCM" ]; then export ROCM_PATH="$ROCM" ROCM_HOME="$ROCM" HIP_PATH="$ROCM"; fi
# The venv is not put on PATH: amd-smi and rocm-smi are python3 scripts that need the image's python3 with its
# amdsmi module (the venv leaves amdsmi out), and the power reader runs them.
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"
unset LD_LIBRARY_PATH
rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || echo "rocminfo names no gfx"
echo "other GPU users:"; pgrep -af "vllm|lab bench|llama|lut_gemv|chipsheet" | grep -v pod_dispatch | cut -c1-160
mkdir -p "$OUT"
for f in dispatch_bench.py lut_gemv.py chipsheet.py chipsheet_format.py pod_power.py; do
  if [ -f "$HERE/r63_$f" ]; then cp "$HERE/r63_$f" "$OUT/$f"
  elif [ -f "$HERE/$f" ]; then cp "$HERE/$f" "$OUT/$f"
  else echo "missing $f (or r63_$f) next to this script: pass it with --with"; exit 2; fi
done
cd "$OUT" || exit 2
[ $# -gt 0 ] || set -- bench --node pod-b
"$VENV/bin/python" dispatch_bench.py "$@" --out "$OUT"
rc=$?
for f in dispatch_bench.py lut_gemv.py chipsheet.py chipsheet_format.py pod_power.py; do rm -f "$HERE/r63_$f"; done
exit $rc
