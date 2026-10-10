#!/bin/bash
# The pod power sampler (lab/pod_power.py) on a Strix Halo pod (SJC hub, accounts C and D): start it, stop it,
# look at it, or read once.
# Reads only (amd-smi / rocm-smi, no root needed); serves its samples on 127.0.0.1:$POD_POWER_PORT for the hub proxy.
# Usage on the pod:  bash pod_power.sh start gpu-a|gpu-b [period_ms] | stop | status | check
# pod_power.py must sit next to this script; it is copied to $POD_POWER_DIR (default ~/atlas-power, persistent).
# From the workstation, after Ilan's go for that pod:
#   python -m lab.controller run-script --account C --with lab/pod_power.py lab/pod_power.sh start gpu-a
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

# Is the pid file's process the sampler? The pid file is on the persistent home and outlives a relaunch, so its
# number may belong to any process in the new container; pod_power.py alive answers from /proc/<pid>/cmdline only.
# Returns 0 with SAMPLER_PID set (a running pod_power.py serve), 1 (none), 2 (cannot tell; the reason is on stderr).
SAMPLER_PID=""
running() {
  SAMPLER_PID=""
  local script="$HERE/pod_power.py" out code=0
  [ -f "$script" ] || script="$DIR/pod_power.py"
  if [ ! -f "$script" ]; then
    echo "cannot check $PIDFILE: pod_power.py is neither next to $0 nor in $DIR" >&2
    return 2
  fi
  out="$(python3 "$script" alive --pidfile "$PIDFILE")" || code=$?
  case "$code" in
    0) SAMPLER_PID="$out"; return 0 ;;
    1) return 1 ;;
    *) echo "cannot check $PIDFILE: pod_power.py alive exited $code" >&2; return 2 ;;
  esac
}

# A pid file that names no running sampler is left over from a pod that was stopped or culled: remove it.
drop_stale_pidfile() {
  if [ -f "$PIDFILE" ]; then
    echo "removing $PIDFILE: pid $(head -c 32 "$PIDFILE" | tr -d '[:space:]') is not a running pod_power.py serve"
    rm -f "$PIDFILE"
  fi
}

show_status() {
  python3 -c 'import sys, urllib.request; print(urllib.request.urlopen("http://127.0.0.1:%s/status" % sys.argv[1], timeout=5).read().decode())' "$PORT"
}

case "$STEP" in
  start)
    NODE="${2:?usage: bash pod_power.sh start gpu-a|gpu-b [period_ms]}"
    PERIOD="${3:-1000}"
    install_copy
    running; STATE=$?
    if [ "$STATE" -eq 2 ]; then exit 1; fi
    if [ "$STATE" -eq 0 ]; then
      echo "already running (pid $SAMPLER_PID)"
      if show_status; then exit 0; fi
      echo "pid $SAMPLER_PID is pod_power.py serve but nothing answered on 127.0.0.1:$PORT"
      exit 1
    fi
    drop_stale_pidfile
    nohup python3 "$DIR/pod_power.py" serve --node "$NODE" --period-ms "$PERIOD" --port "$PORT" --dir "$DIR" \
      >> "$DIR/sampler.log" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 15  # past the 10 s start-up energy check, so the status below shows its verdict
    running; STATE=$?
    if [ "$STATE" -eq 0 ]; then show_status; else echo "the sampler exited:"; tail -n 20 "$DIR/sampler.log"; exit 1; fi
    ;;
  stop)
    running; STATE=$?
    case "$STATE" in
      0) kill "$SAMPLER_PID"; rm -f "$PIDFILE"; echo "stopped" ;;
      1) drop_stale_pidfile; echo "not running" ;;
      *) exit 1 ;;
    esac
    ;;
  status)
    running; STATE=$?
    case "$STATE" in
      0) show_status ;;
      1) echo "not running"; exit 1 ;;
      *) exit 1 ;;
    esac
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
