#!/bin/bash
# Pod side of the R62 lookup-table GEMV bench: measure and leave the result in ~/r62_lut.
# Run through the controller inside a GPU Queue slot, with the files uploaded under an r62_ prefix so nothing another
# session keeps in the Jupyter root (chipsheet.py, pod_power.py) is overwritten:
#   python -m lab.controller run-script --account D --timeout 900 \
#     --with <stage>/r62_lut_gemv.py --with <stage>/r62_chipsheet.py --with <stage>/r62_chipsheet_format.py \
#     --with <stage>/r62_pod_power.py lab/kernels/pod_lut_gemv.sh bench --node pod-b
# (<stage> holds copies of lab/kernels/lut_gemv.py, lab/kernels/chipsheet/chipsheet.py and chipsheet_format.py and
# lab/pod_power.py under those names.) Arguments go to lut_gemv.py; it refuses unless rocminfo names gfx1151.
# Uses the pod image's torch and triton; installs nothing.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"
echo "host $(hostname) $(uname -m)"
rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || echo "rocminfo names no gfx"
OUT="$HOME/r62_lut"
mkdir -p "$OUT"
for f in lut_gemv.py chipsheet.py chipsheet_format.py pod_power.py; do
  if [ -f "$HERE/r62_$f" ]; then cp "$HERE/r62_$f" "$OUT/$f"
  elif [ -f "$HERE/$f" ]; then cp "$HERE/$f" "$OUT/$f"
  else echo "missing $f (or r62_$f) next to this script: pass it with --with"; exit 2; fi
done
cd "$OUT" || exit 2
[ $# -gt 0 ] || set -- bench --node pod-b
python3 lut_gemv.py "$@" --out "$OUT"
rc=$?
for f in lut_gemv.py chipsheet.py chipsheet_format.py pod_power.py; do rm -f "$HERE/r62_$f"; done
exit $rc
