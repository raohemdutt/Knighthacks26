#!/bin/bash
# Keeps a Colab session from idling (Colab reclaims an idle VM after about 20 minutes) whenever the driver is not
# polling it: one tiny status exec every 75 s and one line in progress.log for each. One copy at a time (a pid file);
# it ends when <workspace>/work/L16v2-out/colab/keepalive.stop exists, which the driver creates after its own first
# good status read. The driver starts it again if it exits before its last verified download.
#   usage: bash lab/l16v2/colab_keepalive.sh <session>
export MSYS_NO_PATHCONV=1
S="${1:?session}"
W="${L16V2_WORKSPACE:-/c/Users/hackathon/Code/Knighthacks26}"
C="bash ${L16V2_COLAB:-$W/tooling/dispatcher/colab.sh}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$W/work/L16v2-out/colab"
PIDF="$OUT/keepalive.pid"
mkdir -p "$OUT"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then exit 0; fi
echo $$ > "$PIDF"
rm -f "$OUT/keepalive.stop"
plog() { echo "$(date -u -d '-4 hours' '+%-I:%M %p ET') $*" >> "$W/work/L16v2-out/progress.log"; }
clean() { sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' | tr -cd '\11\12\15\40-\176'; }   # no ANSI codes, ASCII only
plog "keepalive started session=$S pid=$$"
while [ ! -e "$OUT/keepalive.stop" ]; do
  ok=0
  for i in 1 2 3 4; do      # a failed or unreadable answer is a retry, never a dead poll
    o=$(timeout 150 $C exec -s "$S" --timeout 90 -f "$(cygpath -m "$HERE/colab_status.py")" 2>&1 | clean)
    if grep -q "UPTIME" <<<"$o" && ! grep -q "Connection was lost\|Traceback" <<<"$o"; then ok=1; break; fi
    plog "PROBLEM stage=keepalive status exec try $i failed, retrying: $(echo "$o" | grep -v '^ *$' | tail -1 | cut -c1-140)"
    sleep 5
  done
  if [ $ok = 1 ]; then
    plog "status stage=keepalive session=$S exec=ok vm_uptime_s=$(echo "$o" | grep -oE 'UPTIME [0-9.]+' | cut -d' ' -f2) vm_job_alive=$(echo "$o" | grep '^ALIVE' | cut -d' ' -f2) job=\"$(echo "$o" | grep '^STATUS' | cut -c8-100)\""
  else
    plog "PROBLEM stage=keepalive status exec failed 4 times in a row; next try in 75 s"
  fi
  for i in $(seq 1 15); do [ -e "$OUT/keepalive.stop" ] && break; sleep 5; done
done
rm -f "$PIDF"
plog "keepalive stopped session=$S (the driver is polling)"
