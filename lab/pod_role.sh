#!/bin/bash
# R65: give a Strix Halo pod a role (teacher, trainer or both), or let it choose one by looking for a friend pod.
# Both SJC pods are symmetric since R65: each has the teacher's checkpoint and a vLLM v0.31 build with the RDNA hybrid
# W4A16 kernel (~/venv-vllm-new), the training venv (~/venvs/atlas-train) and the student base (~/models/Qwen3.5-4B).
#
# On the pod, through the controller (scripts it calls go up next to it with --with):
#   python -m lab.controller run-script --account D --timeout 900 \
#       --with lab/vllm_upgrade.sh --with lab/pod_a_teacher.sh lab/pod_role.sh teacher D [--apply]
#
#   status  <C|D>            what this pod runs now and what it holds (read-only)
#   teacher <C|D> [--apply]  serve the teacher here: the 27B, same script, flags and port as pod C (vllm_upgrade.sh
#                            serve from ~/venv-vllm-new, else pod_a_teacher.sh serve); refused if a trainer runs here
#   trainer <C|D> [--apply]  the trainer here: checks the venv and the base and prints the l16_run.sh train line;
#                            refused while a teacher serves here (stopping a teacher is the owner's call, not this script's)
#   both    <C|D> [--apply]  the teacher at a 0.60 memory share (placement plan, PL1) so a LoRA job fits beside it
#   beacon  <C|D> [seconds]  answer friend checks on port $FRIEND_PORT: GET /status says what this pod holds (CPU only)
#   auto    <C|D> [addr] [--apply]  the friend check, then a plain decision (below); --apply starts the teacher alone
#
# Without --apply nothing is started or stopped: the script prints its inputs and what it would run.
# A role switch also needs the workstation's LAB_POD_ROLES line, which the script prints at the end (the box and the
# lab verbs find the teacher through it: the account in the pod-a role).
#
# auto's decision, written out:
#   addr     = the argument, else $FRIEND_ADDR, else the first word of ~/.atlas_friend, else the tailnet address of the
#              peer named pod-c or pod-d in `tailscale status` (lab/pod_tailscale.sh)
#   friend   = the other pod's beacon (pod_role.sh beacon) answers GET http://<addr>:$FRIEND_PORT/status within 5 s,
#              through tailscaled's SOCKS5 proxy for a tailnet address; this pod's own address is never a friend
#   big_here = $BIG_MODEL_DIR holds a config.json and safetensors here; big_there = the friend's /status says so
#   serve:  friend and big_here and big_there and POD_TO_POD=yes -> the big model, pipeline-parallel 2 across the pods
#           otherwise                                            -> the 27B teacher alone on this pod
#   train:  friend   -> two ranks, one per pod (lab/r65_rank.sh)
#           otherwise -> one rank on this pod
# R65 measured that the SJC hub refuses every pod-to-pod connection and that Tailscale joins the pods only through a
# DERP relay with no interface, so POD_TO_POD stays no there: a friend found over Tailscale gets the two-rank LoRA
# (control over the tailnet, data over the hub proxy), never a split model.
set -u -o pipefail
ROLE="${1:-help}"
LETTER="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
APPLY=0
for a in "$@"; do [ "$a" = "--apply" ] && APPLY=1; done
PORT="${PORT:-8000}"
FRIEND_PORT="${FRIEND_PORT:-29555}"
TEACHER_DIR="${TEACHER_DIR:-$HOME/models/Qwen3.8-27B-INT4}"
BIG_MODEL_DIR="${BIG_MODEL_DIR:-$HOME/models/big-w4a16}"
VLLM_VENV="${VLLM_VENV:-$HOME/venv-vllm-new}"
TRAIN_VENV="${TRAIN_VENV:-$HOME/venvs/atlas-train}"
STUDENT_DIR="${STUDENT_DIR:-$HOME/models/Qwen3.5-4B}"
TS_BIN="${TS_BIN:-$HOME/ts/bin/tailscale}"
TS_SOCK="${TS_SOCK:-$HOME/ts/tailscaled.sock}"

die() { echo "pod_role: $*"; echo "### END"; exit 1; }
case "$LETTER" in C) OTHER=D ;; D) OTHER=C ;; *) [ "$ROLE" = help ] || die "usage: pod_role.sh <role> <C|D> [--apply]" ;; esac

teacher_up() { python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=3)" 2>/dev/null; }
vllm_running() { pgrep -f "vllm.entrypoints" >/dev/null 2>&1; }
trainer_running() { pgrep -f "lab/train.py" >/dev/null 2>&1; }
has_model() { [ -f "$1/config.json" ] && ls "$1"/*.safetensors >/dev/null 2>&1; }
yn() { if "$@"; then echo yes; else echo no; fi; }

roles_line() {
  # The workstation's setting after the switch: the teacher's account fills pod-a, the trainer's pod-b.
  case "$1" in
    teacher) echo "LAB_POD_ROLES=$LETTER:pod-a,$OTHER:pod-b" ;;
    trainer) echo "LAB_POD_ROLES=$OTHER:pod-a,$LETTER:pod-b" ;;
    both)    echo "LAB_POD_ROLES=$LETTER:pod-a,$OTHER:pod-b   (this pod also trains; the ledger names it by its teacher role)" ;;
  esac
}

status() {
  echo "pod $LETTER: $(hostname) $(hostname -I 2>/dev/null)"
  echo "teacher answering on $PORT: $(yn teacher_up)   vllm process: $(yn vllm_running)   trainer process: $(yn trainer_running)"
  echo "teacher checkpoint $TEACHER_DIR: $(yn has_model "$TEACHER_DIR")"
  echo "big checkpoint $BIG_MODEL_DIR: $(yn has_model "$BIG_MODEL_DIR")"
  echo "vLLM venv $VLLM_VENV: $(ls -d "$VLLM_VENV"/lib/python3*/site-packages/vllm-*.dist-info 2>/dev/null | sed 's/.*vllm-//; s/.dist-info//' || echo missing)"
  echo "training venv $TRAIN_VENV: $(yn test -x "$TRAIN_VENV/bin/python")   student base $STUDENT_DIR: $(yn has_model "$STUDENT_DIR")"
  free -g | awk '/Mem/{print "memory: " $2 " GB total, " $7 " GB available"}'
}

start_teacher() {
  local util="$1" t0 rc
  if teacher_up; then echo "a teacher already answers on $PORT here: nothing to do"; return 0; fi
  vllm_running && { echo "a vLLM process runs here but does not answer on $PORT: not ours to stop; refused"; return 1; }
  trainer_running && [ "$util" = "0.90" ] && { echo "a trainer runs here: the teacher at 0.90 would not fit; use 'both' or stop it first"; return 1; }
  has_model "$TEACHER_DIR" || { echo "no teacher checkpoint at $TEACHER_DIR"; return 1; }
  if [ -x "$VLLM_VENV/bin/python" ] && [ -f "$HERE/vllm_upgrade.sh" ]; then
    echo "would run: GPU_UTIL=$util PORT=$PORT bash vllm_upgrade.sh serve   (venv $VLLM_VENV)"
    [ "$APPLY" = 1 ] || return 0
    t0=$(date +%s); GPU_UTIL="$util" PORT="$PORT" VENV="$VLLM_VENV" bash "$HERE/vllm_upgrade.sh" serve; rc=$?
  elif [ -f "$HERE/pod_a_teacher.sh" ]; then
    echo "would run: GPU_UTIL=$util PORT=$PORT bash pod_a_teacher.sh serve   (the April build; no $VLLM_VENV)"
    [ "$APPLY" = 1 ] || return 0
    t0=$(date +%s); GPU_UTIL="$util" PORT="$PORT" bash "$HERE/pod_a_teacher.sh" serve; rc=$?
  else
    echo "neither vllm_upgrade.sh nor pod_a_teacher.sh next to this script: upload them with --with"; return 1
  fi
  echo "### role teacher start wall_s=$(( $(date +%s) - t0 )) rc=$rc util=$util"
  return "$rc"
}

trainer() {
  if teacher_up || vllm_running; then
    echo "a teacher serves here: stopping it is the owner's call (bash ~/r60/pod_a_teacher.sh stop), not this script's; refused"
    return 1
  fi
  [ -x "$TRAIN_VENV/bin/python" ] || { echo "no training venv at $TRAIN_VENV: bash lab/l16_run.sh prep"; return 1; }
  has_model "$STUDENT_DIR" || { echo "no student base at $STUDENT_DIR: bash lab/l16_run.sh prep"; return 1; }
  echo "trainer ready: bash lab/l16_run.sh train <run id> <dataset id>   (inside a GPU Queue slot)"
}

beacon() {
  local secs="${1:-900}"
  BIG="$(yn has_model "$BIG_MODEL_DIR")" TEACH="$(yn has_model "$TEACHER_DIR")" TRAINV="$(yn test -x "$TRAIN_VENV/bin/python")" \
  nohup python3 - "$FRIEND_PORT" "$secs" > "$HOME/.atlas_beacon.log" 2>&1 <<'PY' &
import json, os, socket, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port, secs = int(sys.argv[1]), float(sys.argv[2])
body = json.dumps({"host": socket.gethostname(), "big_model": os.environ["BIG"] == "yes",
                   "teacher_model": os.environ["TEACH"] == "yes", "train_venv": os.environ["TRAINV"] == "yes"}).encode()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
srv = ThreadingHTTPServer(("0.0.0.0", port), H)
threading.Timer(secs, srv.shutdown).start()
srv.serve_forever()
PY
  echo "$!" > "$HOME/.atlas_beacon.pid"
  echo "beacon on $(hostname -I 2>/dev/null | awk '{print $1}'):$FRIEND_PORT for ${secs}s (pid $(cat "$HOME/.atlas_beacon.pid"))"
}

auto() {
  local addr="${FRIEND_ADDR:-}" via="the argument" answer friend=no big_here big_there=no serve train curl_via=()
  if [ -z "$addr" ] && [ -f "$HOME/.atlas_friend" ]; then addr="$(awk '{print $1; exit}' "$HOME/.atlas_friend")"; via="~/.atlas_friend"; fi
  if [ -z "$addr" ] && [ -S "$TS_SOCK" ]; then
    # the friend on the tailnet: the peer named pod-<other letter, lowercase> in `tailscale status` (lab/pod_tailscale.sh)
    addr="$("$TS_BIN" --socket="$TS_SOCK" status 2>/dev/null | awk -v n="pod-$(echo "$OTHER" | tr CD cd)" '$2 == n {print $1; exit}')"
    via="tailscale status"
  fi
  big_here="$(yn has_model "$BIG_MODEL_DIR")"
  case "$addr" in 100.*) curl_via=(--socks5-hostname 127.0.0.1:1055) ;; esac   # userspace tailnet: through tailscaled
  if [ -n "$addr" ] && hostname -I 2>/dev/null | tr ' ' '
' | grep -qx "$addr"; then
    answer="$addr is this pod's own address, not a friend"
  elif [ -n "$addr" ]; then
    answer="$(curl -s -m 5 ${curl_via[@]+"${curl_via[@]}"} "http://$addr:$FRIEND_PORT/status" 2>&1 | tail -1)"
    [ -n "$answer" ] || answer="no answer in 5 s"
    case "$answer" in \{*) friend=yes; echo "$answer" | grep -q '"big_model": true' && big_there=yes ;; esac
  else
    answer="no friend address (argument, FRIEND_ADDR, ~/.atlas_friend or a tailnet peer named pod-c or pod-d)"
  fi
  echo "input pod: $LETTER ($(hostname -I 2>/dev/null | awk '{print $1}')), friend pod: $OTHER at ${addr:-unknown}:$FRIEND_PORT (from $via${curl_via[0]:+, through the tailscaled SOCKS5 proxy})"
  echo "input friend answers its beacon: $friend   ($answer)"
  echo "input big checkpoint here ($BIG_MODEL_DIR): $big_here   on the friend: $big_there"
  if [ "$friend" = yes ] && [ "$big_here" = yes ] && [ "$big_there" = yes ] && [ "${POD_TO_POD:-no}" = yes ]; then
    serve="big model, pipeline-parallel 2 across pods $LETTER and $OTHER"
  else
    serve="the 27B teacher alone on pod $LETTER"
    [ "$friend" = yes ] && serve="$serve (a split model needs a pod-to-pod network, POD_TO_POD=yes; R65 found none)"
  fi
  if [ "$friend" = yes ]; then train="two ranks, one on each pod (lab/r65_rank.sh: local SGD, control over the tailnet, data over the hub proxy)"; else train="one rank on pod $LETTER"; fi
  echo "decision serve: $serve"
  echo "decision train: $train"
  if [ "$APPLY" = 1 ]; then
    start_teacher 0.90
  fi
}

t_start=$(date +%s)
case "$ROLE" in
  status) status ;;
  teacher) start_teacher 0.90; rc=$?; [ "$rc" = 0 ] && roles_line teacher ;;
  trainer) trainer; rc=$?; [ "$rc" = 0 ] && roles_line trainer ;;
  both) start_teacher 0.60; rc=$?; [ "$rc" = 0 ] && roles_line both ;;
  beacon) shift 2; beacon "${1:-900}" ;;
  auto) [ "${3:-}" != "--apply" ] && [ -n "${3:-}" ] && FRIEND_ADDR="$3"; auto; rc=$? ;;
  *) sed -n '2,35p' "$0"; [ "$ROLE" = help ] || exit 2 ;;
esac
echo "### pod_role $ROLE $LETTER apply=$APPLY wall_s=$(( $(date +%s) - t_start )) rc=${rc:-0}"
echo "### END"
exit "${rc:-0}"
