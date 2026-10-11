#!/bin/bash
# L29: the pod D side of the live learning-loop demo (tooling/demo/live_loop.sh runs it; see LIVE_LOOP_RUN_OF_SHOW.md).
# Two acts of the L10 group fine-tune (lab/pod_b_group_finetune.sh run, the same verb and flags), one after the other:
#   act 1 "solo": the ilan member alone, 100 examples;  act 2 "duo": ilan and dom pooled, 200 examples.
# Each act: pool, label at the pod C teacher (LABEL_WORKERS streams), train a LoRA (STEPS short steps, one loss row per
# step), package, stand-in gate, offer. Pod C labels act 2 while pod D trains act 1; act 2 then trains on the GPU while
# act 1 packages on the CPU (pod D alone, serial, is 240 s of work: two trainings near 56 s and two packagings near
# 64 s), so the overlapping stretch of pod D's power is split evenly between the two stages (T4's split rule).
#
#   bash live_loop_pod.sh check                         what the run needs is on the pod; disk; the sampler; any job
#   bash live_loop_pod.sh start <stamp> <slot> [steps]  reads the teacher token from ./l29_tt (uploaded beside this
#                                                       script with run-script --with, deleted at once), starts the two
#                                                       acts in the background and returns
#
# What it writes: ~/l29/<stamp>/stages.jsonl, one line per stage change on this pod's clock (the relay reads it),
# ~/l29/<stamp>/driver.log, and per act ~/group-runs/<run id>/ and ~/data/packages/atlas-group-desk-objects-<run id>/
# as the L10 verb writes them. The pooled copies are deleted by the verb on every way out, as in L10.
set -u -o pipefail
STEP="${1:-help}"
L10="$HOME/l10"
VENV="$HOME/venvs/atlas-train"
BIGPOOL="${BIGPOOL:-$L10/bigpool.tar.gz}"   # the L10 smoke exports: ilan/desk-objects-ilan-smoke, dom/... (100 each)
TASK="desk-objects"
SOLO="l29-solo"
DUO="l29-duo"

now() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }

check() {
  local ok=1
  [ -x "$VENV/bin/python" ] || { echo "missing: $VENV"; ok=0; }
  [ -f "$L10/src/lab/pod_b_group_finetune.sh" ] || { echo "missing: $L10/src (lab/pod_b_group_finetune.sh bundle + prep)"; ok=0; }
  [ -x "$L10/llama.cpp/build/bin/llama-quantize" ] || { echo "missing: $L10/llama.cpp links (prep)"; ok=0; }
  [ -f "$BIGPOOL" ] || { echo "missing: $BIGPOOL"; ok=0; }
  echo "pool members in the bundle: $(tar -tzf "$BIGPOOL" 2>/dev/null | grep -c 'examples.jsonl$') exports"
  echo "disk free in home: $(df -h "$HOME" | awk 'NR==2{print $4}')"
  echo "sampler: $(cat "$HOME/atlas-power/sampler.pid" 2>/dev/null) $(ps -p "$(cat "$HOME/atlas-power/sampler.pid" 2>/dev/null)" -o etime= 2>/dev/null)"
  echo "jobs: $(pgrep -af 'lab/train.py|group-finetune|llama-server|vllm' | grep -v defunct | cut -c1-120 | tr '\n' ';')"
  [ "$ok" = 1 ] && echo "check ok"
}

# The trainer wrapper the verb runs lab/train.py under (--trainer-python): it marks the end of labeling, waits for
# the file in L29_WAIT (act 2 waits for act 1 to finish packaging), marks the start and the end of training.
write_wrapper() {
  cat > "$1" <<'SH'
#!/bin/bash
mark() { printf '{"act":"%s","stage":"%s","ts":"%s"}\n' "$L29_ACT" "$1" "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" >> "$L29_STAGES"; }
mark labeled
if [ -n "${L29_WAIT:-}" ]; then while [ ! -e "$L29_WAIT" ]; do sleep 0.2; done; fi
mark train
if [ -n "${L29_MIN_HELD_OUT:-}" ] && [ "${1:-}" = "-u" ] && [ "${2:-}" = "lab/train.py" ]; then
  # act 1's 100 examples freeze 25 held out; lab/train.py wants the 50 a deploying gate needs. This act never
  # deploys (stand-in gate), so its trainer reads the split with the act's own floor; nothing else changes.
  shift 2
  "$L29_PY" -u -c 'import sys; import lab.dataset as d; d.load_split.__kwdefaults__["min_held_out"] = int(sys.argv[1]); sys.argv = ["lab/train.py"] + sys.argv[2:] + ["--no-install"]; import lab.train as t; sys.exit(t._start())' "$L29_MIN_HELD_OUT" "$@"
else
  "$L29_PY" "$@"
fi
rc=$?
mark trained
# act 1 packages on the CPU while act 2 trains: the act 1 verb (this script's parent) and the packaging steps it
# starts next run at low priority, so act 2's trainer, which is on the critical path, keeps the cores it needs.
[ -n "${L29_NICE_AFTER:-}" ] && renice -n "$L29_NICE_AFTER" -p "$PPID" > /dev/null 2>&1
exit $rc
SH
  chmod +x "$1"
}

roster() {   # roster <circle> <member> ...: the pool folder, emptied of old exports, with its pool.json
  local circle="$1"; shift
  local dir="$HOME/pool/$circle/$TASK"
  rm -rf "$dir" && mkdir -p "$dir"
  {
    printf '{"version": 1, "circle": "%s", "task": "%s", "members": [' "$circle" "$TASK"
    local first=1 m
    for m in "$@"; do
      [ $first = 1 ] || printf ', '
      printf '{"member": "%s", "current_version": "atlas-base", "exports": {}}' "$m"
      first=0
    done
    printf ']}\n'
  } > "$dir/pool.json"
  tar -xzf "$BIGPOOL" -C "$dir" "${@/%//}" || return 1
}

act() {   # act <name> <circle> <run id> [wait file]: one L10 run, with its stage lines
  local name="$1" circle="$2" run_id="$3" wait="${4:-}"
  printf '{"act":"%s","stage":"start","ts":"%s","run_id":"%s","circle":"%s"}\n' "$name" "$(now)" "$run_id" "$circle" >> "$STAGES"
  local floor=""
  local nice_after=""
  [ "$name" = solo ] && floor=25 && nice_after=15
  L29_ACT="$name" L29_WAIT="$wait" L29_STAGES="$STAGES" L29_PY="$VENV/bin/python" L29_MIN_HELD_OUT="$floor" L29_NICE_AFTER="$nice_after" \
    bash "$L10/src/lab/pod_b_group_finetune.sh" run "$circle" "$TASK" "$run_id" \
    --grad-accum 1 --trainer-python "$RUN/trainer.sh" >> "$RUN/$name.log" 2>&1
  local rc=$?
  printf '{"act":"%s","stage":"done","ts":"%s","rc":%d}\n' "$name" "$(now)" "$rc" >> "$STAGES"
  echo "$rc" > "$RUN/$name.exit"
}

drive() {
  local stamp="$1"
  RUN="$HOME/l29/$stamp"
  STAGES="$RUN/stages.jsonl"
  export RUN STAGES
  local a1="live-$stamp-solo" a2="live-$stamp-duo"
  printf '{"act":"all","stage":"go","ts":"%s","solo":"%s","duo":"%s","steps":%s,"workers":%s}\n' \
    "$(now)" "$a1" "$a2" "$L10_STEPS" "$LABEL_WORKERS" >> "$STAGES"
  write_wrapper "$RUN/trainer.sh"
  roster "$SOLO" ilan || { echo '{"act":"all","stage":"failed","why":"pool solo"}' >> "$STAGES"; return 1; }
  roster "$DUO" ilan dom || { echo '{"act":"all","stage":"failed","why":"pool duo"}' >> "$STAGES"; return 1; }
  act solo "$SOLO" "$a1" &
  local p1=$!
  # act 2 starts labeling the moment act 1 hands its labels to the trainer (or ends)
  while ! grep -q '"act":"solo","stage":"labeled"' "$STAGES" && [ ! -e "$RUN/solo.exit" ]; do sleep 0.2; done
  # act 2 trains as soon as its labels are in, while act 1 packages on the CPU (L29_SERIAL=1: wait for act 1 to end)
  local wait=""
  [ "${L29_SERIAL:-0}" = 1 ] && wait="$RUN/solo.exit"
  act duo "$DUO" "$a2" "$wait" &
  local p2=$!
  wait $p1 $p2
  printf '{"act":"all","stage":"end","ts":"%s"}\n' "$(now)" >> "$STAGES"
}

start() {
  local stamp="${1:-}" slot="${2:-}" steps="${3:-8}"
  [ -n "$stamp" ] && [ -n "$slot" ] || { echo "usage: start <stamp> <slot row> [steps]"; return 2; }
  umask 077
  local tt="./l29_tt"
  [ -s "$tt" ] || { echo "no token file beside the script"; return 1; }
  TEACHER_TOKEN="$(cat "$tt")"
  rm -f "$tt"
  local run="$HOME/l29/$stamp"
  [ -e "$run" ] && { echo "$run exists: a stamp is used once"; return 1; }
  mkdir -p "$run"
  export TEACHER_TOKEN GPU_SLOT="$slot" L10_STEPS="$steps" LABEL_WORKERS="${LABEL_WORKERS:-16}" GATE=fake \
    MIN_HELD_OUT=1 RUN_TIMEOUT_S=900
  cp "$0" "$run/pod.sh"
  setsid nohup bash "$run/pod.sh" drive "$stamp" > "$run/driver.log" 2>&1 < /dev/null &
  unset TEACHER_TOKEN
  echo "started $stamp: ~/l29/$stamp/stages.jsonl"
}

case "$STEP" in
  check) check ;;
  start) shift; start "$@" ;;
  drive) shift; drive "$@" ;;
  *) sed -n '2,17p' "$0" ;;
esac
echo "### END"
