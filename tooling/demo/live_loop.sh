#!/bin/bash
# L29: the live learning loop demo, one command from this machine (Git Bash, from the repo root).
#   bash tooling/demo/live_loop.sh --slot <pod D row> --slot-c <pod C row>     start both acts now (act 2 on both pods)
#   bash tooling/demo/live_loop.sh --slot <D row> --slot-c <C row> --on-approval  start when the phone approves an export
#   bash tooling/demo/live_loop.sh --check                                      the pods and the box are ready; starts nothing
# The teacher is the 122B pair (--teacher-expect Qwen3.5-122B, the default); --teacher-expect Qwen3.8-27B is the
# fallback, --duo-pods 1 keeps act 2 on pod D. Act sizes: --solo-examples 50 --duo-examples 50 (per member).
# The screen: the box's Compare tab, http://<box>:8000/lab/?tab=compare (--card: the old local card at :8788)
# The run of show: tooling/demo/LIVE_LOOP_RUN_OF_SHOW.md
# Everything else is in tooling/demo/live_loop.py (the relay) and tooling/demo/live_loop_pod.sh (the pods' side).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY="${PYTHON:-python}"
[ -x .venv/Scripts/python.exe ] && PY=.venv/Scripts/python.exe
exec "$PY" -u tooling/demo/live_loop.py "$@"
