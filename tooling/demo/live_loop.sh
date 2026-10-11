#!/bin/bash
# L29: the live learning loop demo, one command from this machine (Git Bash, from the repo root).
#   bash tooling/demo/live_loop.sh --slot <GPU Queue row>                 start both acts now
#   bash tooling/demo/live_loop.sh --slot <row> --on-approval              start when the phone approves an export
#   bash tooling/demo/live_loop.sh --check                                 the pod and the box are ready; starts nothing
# The screen: the box's Compare tab, http://<box>:8000/lab/?tab=compare (--card: the old local card at :8788)
# The run of show: tooling/demo/LIVE_LOOP_RUN_OF_SHOW.md
# Everything else is in tooling/demo/live_loop.py (the relay) and tooling/demo/live_loop_pod.sh (pod D's side).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY="${PYTHON:-python}"
[ -x .venv/Scripts/python.exe ] && PY=.venv/Scripts/python.exe
exec "$PY" -u tooling/demo/live_loop.py "$@"
