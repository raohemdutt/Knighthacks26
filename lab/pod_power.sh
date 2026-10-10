#!/bin/bash
# The pod power sampler (lab/pod_power.py) on an R9700 pod: start it, stop it, look at it, or read once.
# Reads only (amd-smi / rocm-smi, no root needed); serves its samples on 127.0.0.1:$POD_POWER_PORT for the hub proxy.
# Usage on the pod:  bash pod_power.sh start gpu-a|gpu-b [period_ms] | stop | status | check
# pod_power.py must sit next to this script; it is copied to $POD_POWER_DIR (default ~/atlas-power, persistent).
# From the workstation, after Ilan's go for that pod:
#   python -m lab.controller run-script --account A --with lab/pod_power.py lab/pod_power.sh start gpu-a
set -u
STEP="${1:-status}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${POD_POWER_DIR:-$HOME/atlas-power}"
PORT="${POD_POWER_PORT:-8790}"
PIDFILE="$DIR/sampler.pid"
export PATH="$HOME/.local/bin:/opt/rocm/bin:$PATH"

install_copy() {
  mkdir -p "$DIR"
  [ -f "$HERE/pod_power.py" ] && cp "$HERE/pod_power.py" "$DIR/pod_power.py"
  [ -f "$DIR/pod_power.py" ] || { echo "pod_power.py is neither next to $0 nor in $DIR"; exit 1; }
}

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

show_status() {
  python3 -c 'import sys, urllib.request; print(urllib.request.urlopen("http://127.0.0.1:%s/status" % sys.argv[1], timeout=5).read().decode())' "$PORT"
}

case "$STEP" in
  start)
    NODE="${2:?usage: bash pod_power.sh start gpu-a|gpu-b [period_ms]}"
    PERIOD="${3:-1000}"
    install_copy
    if running; then echo "already running (pid $(cat "$PIDFILE"))"; show_status; exit 0; fi
    nohup python3 "$DIR/pod_power.py" serve --node "$NODE" --period-ms "$PERIOD" --port "$PORT" --dir "$DIR" \
      >> "$DIR/sampler.log" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 15  # past the 10 s start-up energy check, so the status below shows its verdict
    if running; then show_status; else echo "the sampler exited:"; tail -n 20 "$DIR/sampler.log"; exit 1; fi
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")"; rm -f "$PIDFILE"; echo "stopped"; else echo "not running"; fi
    ;;
  status)
    if running; then show_status; else echo "not running"; exit 1; fi
    ;;
  check)
    install_copy
    python3 "$DIR/pod_power.py" check --save "$DIR/raw-$(date -u +%Y%m%dT%H%M%SZ)"
    ;;
  *)
    echo "usage: bash pod_power.sh start gpu-a|gpu-b [period_ms] | stop | status | check"
    exit 2
    ;;
esac
