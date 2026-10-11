#!/bin/bash
# Pod side of the chip sheet: measure this pod's GPU and leave the sheet in ~/chipsheet.
# Run through the controller, which uploads the Python files and the probe source next to this script:
#   python -m lab.controller run-script --account D --timeout 1200 \
#     --with lab/kernels/chipsheet/chipsheet.py --with lab/kernels/chipsheet/chipsheet_format.py \
#     --with lab/kernels/chipsheet/probe.hip --with lab/pod_power.py \
#     lab/kernels/chipsheet/pod_chipsheet.sh run --node pod-b
# Arguments are passed to chipsheet.py (run --node pod-b [--quick]); chipsheet.py refuses to measure unless rocminfo
# names gfx1151. The last line is chipsheet.py's JSON answer naming the files to pull back.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"
echo "host $(hostname) $(uname -m)"
rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || echo "rocminfo names no gfx"
mkdir -p "$HOME/chipsheet"
for f in chipsheet.py chipsheet_format.py probe.hip pod_power.py; do
  [ -f "$HERE/$f" ] || { echo "missing $f next to this script: pass it with --with"; exit 2; }
  cp "$HERE/$f" "$HOME/chipsheet/$f"
done
cd "$HOME/chipsheet" || exit 2
[ $# -gt 0 ] || set -- run --node pod-b
python3 chipsheet.py "$@" --out "$HOME/chipsheet"
