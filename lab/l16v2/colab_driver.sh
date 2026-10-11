#!/bin/bash
# The workstation side of the L16 retrain on a Colab A100 (a copy of tooling/dispatcher/a100_eval_rerun.sh, widened):
# upload the branch as a git bundle with the job script, launch the job detached on the VM, then poll and pull every
# product the moment its marker appears, so the 20-minute idle reclaim cannot take a result with it.
#
#   bash lab/l16v2/colab_driver.sh <session> <branch> [sweep budget s]        # upload, launch, poll, pull, stop
#   bash lab/l16v2/colab_driver.sh <session> <branch> --attach                # a job is already running: poll and pull
#
# A launch on a VM that already holds the dataset keeps it: the job rebuilds it from the new commit, compares the
# sha256 of every file, and reuses the frozen dataset and its tarball when they are the same bytes.
#
# Run it detached with its own log (nohup ... > driver.log 2>&1 &). The session must exist and be READY, with the base
# model in /content/models (lab/l16v2/colab_setup.sh). It needs the workspace's Colab wrapper (L16V2_COLAB, default
# tooling/dispatcher/colab.sh under L16V2_WORKSPACE) and writes under <workspace>/work/L16v2-out/:
#
#   progress.log      one line per event, each starting with the Eastern time: a status line at every poll in the
#                     job's own key=value form (stage, run i/n, lr, rank, drop, step, loss, val, best, eta_s), a line
#                     for every stage change, finished run (RESULT), download with its verified size and sha256,
#                     retry and error (PROBLEM), and a heartbeat when three minutes pass with nothing new
#   colab/            every product, checked against the size and sha256 the job wrote beside it; downloads.txt
#                     lists them
#
# Every read of the Colab CLI goes through clean(): ANSI codes stripped, ASCII only, so no output can break a match.
# A status read that fails is retried, never taken as a dead poll. The session is stopped only after the last
# download is verified, and the usage rate is read back afterwards. lab/l16v2/colab_keepalive.sh polls the session
# until the driver's first good status read, and again if the driver ends early (a trap on exit), so the VM always
# has something polling it.
set -u
export MSYS_NO_PATHCONV=1
S="${1:?session}"; BRANCH="${2:?branch}"; MODE="${3:-2400}"
W="${L16V2_WORKSPACE:-/c/Users/hackathon/Code/Knighthacks26}"
C="bash ${L16V2_COLAB:-$W/tooling/dispatcher/colab.sh}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$W/work/L16v2-out/colab"
WIN() { cygpath -m "$1"; }                    # C:/... for upload and exec -f (the wrapper maps an existing path)
WSL() { echo "/mnt/$(echo "$1" | sed -e 's|^/\([a-zA-Z]\)/|\L\1/|')"; }   # /mnt/c/... for a download target
POLL_S=75
mkdir -p "$OUT/launch"
plog() { echo "$(date -u -d '-4 hours' '+%-I:%M %p ET') $*" >> "$W/work/L16v2-out/progress.log"; LAST_LINE=$(date +%s); }
LAST_LINE=$(date +%s)
clean() { sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' | tr -cd '\11\12\15\40-\176'; }
failed() { grep -q "Connection was lost\|Traceback\|Error\|No such\|not found" <<<"$1"; }

up() { # up <local> <remote>: up to 4 tries
  local i o
  for i in 1 2 3 4; do
    o=$(timeout 900 $C upload -s "$S" "$(WIN "$1")" "$2" 2>&1 | clean)
    if grep -q "Uploaded" <<<"$o" && ! failed "$o"; then plog "upload done $2 bytes=$(stat -c %s "$1")"; return 0; fi
    plog "PROBLEM upload try $i failed for $2: $(echo "$o" | grep -v '^ *$' | tail -1 | cut -c1-140)"; sleep 10
  done; return 1
}
ex() { # ex <local py> <word the answer must hold>: up to 5 tries; a failed or unreadable answer is a retry
  local i o
  for i in 1 2 3 4 5; do
    o=$(timeout 150 $C exec -s "$S" --timeout 90 -f "$(WIN "$1")" 2>&1 | clean)
    if grep -q "$2" <<<"$o" && ! grep -q "Connection was lost\|Traceback" <<<"$o"; then echo "$o"; return 0; fi
    plog "PROBLEM exec try $i of $(basename "$1") failed, retrying: $(echo "$o" | grep -v '^ *$' | tail -1 | cut -c1-140)"; sleep 6
  done; return 1
}
raw() { # raw <remote path> <local name>: a log, pulled without a checksum (the job did not write one)
  timeout 600 $C download -s "$S" "$1" "$(WSL "$OUT")/$2" > /dev/null 2>&1 \
    && plog "download $2 bytes=$(stat -c %s "$OUT/$2" 2>/dev/null) unchecked=log" || plog "PROBLEM could not pull $1"
}
get() { # get <product name>: the .sha256 first, then the file, checked; up to 4 tries
  local name="$1" i want size have
  for i in 1 2 3 4; do
    rm -f "$OUT/$name.sha256.part" "$OUT/$name.part"
    timeout 300 $C download -s "$S" "/content/l16v2/products/$name.sha256" "$(WSL "$OUT")/$name.sha256.part" > /dev/null 2>&1
    timeout 1500 $C download -s "$S" "/content/l16v2/products/$name" "$(WSL "$OUT")/$name.part" > /dev/null 2>&1
    want=$(cut -d' ' -f1 "$OUT/$name.sha256.part" 2>/dev/null); size=$(cut -d' ' -f2 "$OUT/$name.sha256.part" 2>/dev/null)
    have=$(sha256sum "$OUT/$name.part" 2>/dev/null | cut -d' ' -f1)
    if [ -n "$want" ] && [ "$want" = "$have" ] && [ "$size" = "$(stat -c %s "$OUT/$name.part")" ]; then
      mv "$OUT/$name.part" "$OUT/$name"; mv "$OUT/$name.sha256.part" "$OUT/$name.sha256"
      echo "$name $size $have verified" >> "$OUT/downloads.txt"
      plog "download $name bytes=$size sha256=$have verified=yes"; return 0
    fi
    plog "PROBLEM download try $i of $name did not verify (want ${want:-none} size ${size:-none}, have ${have:-none}); retrying"; sleep 8
  done
  plog "PROBLEM download of $name failed 4 times"; return 1
}

# If the driver ends for any reason before its last verified download, say so and hand the session back to the
# keepalive, so the VM is never left with nothing polling it. (The keepalive is one copy at a time: starting it
# while it still runs does nothing.)
DONE_OK=0
finish() {
  local rc=$?
  if [ "$DONE_OK" != 1 ]; then
    plog "PROBLEM driver exited with code $rc before the last verified download; the session is left up and the keepalive takes over"
    rm -f "$OUT/keepalive.stop"
    (nohup bash "$HERE/colab_keepalive.sh" "$S" > "$OUT/keepalive.log" 2>&1 &)
  fi
}
trap finish EXIT

if [ "$MODE" != "--attach" ]; then
  # git is a Windows program and MSYS_NO_PATHCONV=1 (set for the Colab wrapper) stops /c/... arguments from being
  # converted for it: run it from inside the clone and hand it Windows-form paths.
  COMMIT=$(cd "$REPO" && git rev-parse --short "$BRANCH" 2>/dev/null)
  [ -n "$COMMIT" ] || { plog "PROBLEM driver refuses to start: no commit for branch $BRANCH in $REPO"; exit 1; }
  [ -z "$(cd "$REPO" && git status --porcelain -- lab contexts contracts runtime fixtures config 2>/dev/null)" ] \
    || plog "note: the clone has uncommitted changes under lab/ or the agent's folders; the bundle holds commit $COMMIT only"
  plog "driver start session=$S branch=$BRANCH commit=$COMMIT budget_s=$MODE"
  rm -f "$OUT/l16v2.bundle"
  (cd "$REPO" && git bundle create "$(WIN "$OUT/l16v2.bundle")" "$BRANCH") > "$OUT/bundle.log" 2>&1
  [ -s "$OUT/l16v2.bundle" ] || { plog "PROBLEM git bundle failed: $(tail -1 "$OUT/bundle.log" | cut -c1-140)"; exit 1; }
  plog "bundle made commit=$COMMIT bytes=$(stat -c %s "$OUT/l16v2.bundle")"
  cat > "$OUT/launch/launch_job.py" <<PY
import subprocess
subprocess.run("pkill -f '[c]olab_job.sh'; pkill -f '[l]ab/train.py'; pkill -f '[l]16v2/sweep.py'; rm -rf /content/l16v2/runs /content/l16v2/finished.txt /content/l16v2/status.txt; find /content/l16v2/markers -type f ! -name DATASET_DONE -delete 2>/dev/null; find /content/l16v2/products -type f ! -name 'dataset-*' -delete 2>/dev/null", shell=True)
subprocess.Popen("setsid nohup bash /content/colab_job.sh $BRANCH $MODE > /content/colab_job.log 2>&1 &", shell=True)
print("launched job")
PY
  up "$OUT/l16v2.bundle" /content/l16v2.bundle || { plog "PROBLEM upload of the bundle failed; driver stops, session left up"; exit 1; }
  up "$HERE/colab_job.sh" /content/colab_job.sh || { plog "PROBLEM upload of the job failed; driver stops, session left up"; exit 1; }
  ex "$OUT/launch/launch_job.py" "launched job" > /dev/null || { plog "PROBLEM launch failed; driver stops, session left up"; exit 1; }
  plog "job launched detached on the VM; polling every $POLL_S s"
else
  plog "driver attached to the running job session=$S"
fi

declare -A HANDLED
STAGE=""; SEEN_FINISHED=0; DEAD=0; FAILS=0; ALL_OK=1
for poll in $(seq 1 400); do
  if ! st=$(ex "$HERE/colab_status.py" "UPTIME"); then
    FAILS=$((FAILS + 1)); plog "PROBLEM status read failed 5 times in a row (round $FAILS); next try in 30 s"
    [ $FAILS -ge 8 ] && { plog "PROBLEM no status for 8 rounds; driver stops, session left up"; exit 4; }
    sleep 30; continue
  fi
  FAILS=0
  [ -e "$OUT/keepalive.stop" ] || touch "$OUT/keepalive.stop"   # the driver's own polls keep the session busy from here
  line=$(echo "$st" | grep "^STATUS " | head -1 | cut -c8-)
  plog "${line:-stage=unknown} vm_job_alive=$(echo "$st" | grep '^ALIVE' | cut -d' ' -f2)"
  now_stage=$(echo "$line" | grep -oE "stage=[a-z_]+" | head -1)
  if [ -n "$now_stage" ] && [ "$now_stage" != "$STAGE" ]; then plog "stage change ${STAGE:-none} -> $now_stage"; STAGE="$now_stage"; fi
  n=0
  while IFS= read -r fin; do
    n=$((n + 1)); [ $n -le $SEEN_FINISHED ] && continue
    plog "RESULT $fin"
  done < <(echo "$st" | sed -n '/^FINISHED_BEGIN/,/^FINISHED_END/p' | grep -v "^FINISHED_")
  SEEN_FINISHED=$n
  while read -r _ marker product; do
    [ -n "${marker:-}" ] || continue
    [ -n "${HANDLED[$marker]:-}" ] && continue
    case "$marker" in
      SMOKE_DONE) plog "stage smoke passed on the VM"; HANDLED[$marker]=1 ;;
      JOB_DONE) : ;;
      FAILED_*)
        plog "PROBLEM the job wrote $marker; pulling the logs"
        raw /content/colab_job.log colab_job.log; raw /content/l16v2/sweep.log sweep.log; raw /content/l16v2/smoke.log smoke.log
        raw /content/l16v2/dataset.err dataset.err
        [ "$marker" = "FAILED_smoke" ] && get "$product"
        HANDLED[$marker]=1 ;;
      *) if get "$product"; then HANDLED[$marker]=1; else ALL_OK=0; HANDLED[$marker]=failed; fi ;;
    esac
  done < <(echo "$st" | grep "^MARK ")
  if echo "$st" | grep -q "^MARK JOB_DONE"; then
    raw /content/colab_job.log colab_job.log
    for marker in "${!HANDLED[@]}"; do   # one more pass over anything that did not verify
      if [ "${HANDLED[$marker]}" = "failed" ]; then
        product=$(echo "$st" | grep "^MARK $marker " | cut -d' ' -f3)
        get "$product" && HANDLED[$marker]=1
      fi
    done
    left=0; for marker in "${!HANDLED[@]}"; do [ "${HANDLED[$marker]}" = "failed" ] && left=$((left + 1)); done
    if [ $left -gt 0 ]; then
      plog "PROBLEM $left products did not verify after JOB_DONE; the session is LEFT UP so they can be pulled by hand"
      exit 5
    fi
    plog "RESULT job done; every product is local and verified: $(cut -d' ' -f1 "$OUT/downloads.txt" | tr '\n' ' ')"
    timeout 180 $C stop -s "$S" 2>&1 | clean | tail -1
    use=$(timeout 120 $C usage 2>&1 | clean | tr '\n' ' ')
    plog "RESULT session $S stopped; $use"
    DONE_OK=1
    exit 0
  fi
  if [ "$(echo "$st" | grep '^ALIVE' | cut -d' ' -f2)" = "0" ]; then
    DEAD=$((DEAD + 1))
    if [ $DEAD -ge 3 ]; then
      plog "PROBLEM the job is not running on the VM and wrote no JOB_DONE; pulling the logs; the session is LEFT UP"
      raw /content/colab_job.log colab_job.log; raw /content/l16v2/sweep.log sweep.log; raw /content/l16v2/smoke.log smoke.log
      exit 6
    fi
  else
    DEAD=0
  fi
  while [ $(( $(date +%s) - LAST_LINE )) -lt $POLL_S ]; do sleep 5; done
  [ $(( $(date +%s) - LAST_LINE )) -ge 180 ] && plog "heartbeat driver alive, nothing new for 3 minutes"
done
plog "PROBLEM no JOB_DONE after 400 polls; the session is LEFT UP"; exit 3
