#!/bin/bash
# Pod side of the wavefront microbenchmarks: measure one kernel class (one lever) and leave the sheet in ~/wavefront.
# Run through the controller, which uploads the Python files next to this script; the one-line command for a class
# is what `python -m lab.kernels.wavefront.wavefront pod-script --only <class>` prints, for example:
#   python -m lab.controller run-script --account D --timeout 1200 --with lab/kernels/wavefront/wavefront.py ... \
#     lab/kernels/wavefront/pod_wavefront.sh run --node pod-b --only gdn
# Arguments are passed to wavefront.py; it refuses to measure unless rocminfo names gfx1151. The last line is
# wavefront.py's JSON answer naming the files to pull back.
#
# The Python is the trainer's own when the pod has it: the venv L16's prep left (~/venvs/atlas-train: the image's torch
# and triton plus the pinned transformers, whose Gated DeltaNet reference the gdn class times), run as the trainer
# runs, without the user site (PYTHONNOUSERSITE=1; pod D keeps vLLM and its numpy there). Without that venv it is the
# image's python3, where a class whose package is missing is written as skipped. WAVEFRONT_PYTHON names another.
# Nothing is installed either way.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"
echo "host $(hostname) $(uname -m)"
rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || echo "rocminfo names no gfx"
mkdir -p "$HOME/wavefront"
for f in wavefront.py wavefront_format.py gdn.py conv.py launch.py tile.py lora.py chipsheet_format.py; do
  [ -f "$HERE/$f" ] || { echo "missing $f next to this script: pass it with --with"; exit 2; }
  cp "$HERE/$f" "$HOME/wavefront/$f"
done
cd "$HOME/wavefront" || exit 2
[ $# -gt 0 ] || set -- run --node pod-b --quick
PY="${WAVEFRONT_PYTHON:-$HOME/venvs/atlas-train/bin/python}"
if [ -x "$PY" ]; then export PYTHONNOUSERSITE=1; else PY=python3; fi
echo "python $PY"
"$PY" wavefront.py "$@" --out "$HOME/wavefront"
