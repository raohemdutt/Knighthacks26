#!/bin/bash
# L16 retrain: one pod D row from start to done for one adapter: package it there, serve before and after, score the
# two agreement sets and the gate from the workstation, stop the servers, close the row. Run from the repo root
# (Git Bash) with the row already Queued and the two tarballs already pushed to ~/l16v2 on the pod.
#
#   bash lab/l16v2/pod_step4.sh <tag: a100|podd> <row id> <version> <adapter: a100 | pod:<run id>> <run id> <v2 dataset dir>
#
# Everything lands in <workspace>/work/L16v2-out/pod/<tag>/ and every step writes its line to progress.log.
# The row is extended once (15 minutes) after the first agreement run when the rest would not fit; a step that fails
# stops the servers and closes the row with what was measured so far.
set -u
TAG="${1:?tag}"; ROW="${2:?row}"; VERSION="${3:?version}"; ADAPTER="${4:?adapter}"; RUN_ID="${5:?run id}"; V2="${6:?v2 dataset dir}"
W="${L16V2_WORKSPACE:-/c/Users/hackathon/Code/Knighthacks26}"
PY="${L16V2_PYTHON:-$W/atlas/.venv/Scripts/python}"
POD="$W/work/L16v2-out/pod"; OUT="$POD/$TAG"
mkdir -p "$OUT" "$POD/state"
export LAB_TASK=L16 LAB_ACTOR="Ilan1 agent" LAB_TIER=opus PYTHONIOENCODING=utf-8 LAB_STATE_DIR="$POD/state"
plog() { echo "$(date -u -d '-4 hours' '+%-I:%M %p ET') $*" >> "$W/work/L16v2-out/progress.log"; }
T0=$(date +%s)
mins() { echo $(( ($(date +%s) - T0) / 60 )); }
RESULT=""
ctl() { # ctl <name> <timeout> <pod_run.sh args...>: one run-script call, its JSON kept; 0 when the script exited 0
  local name="$1" timeout="$2"; shift 2
  "$PY" -m lab.controller run-script --account D --timeout "$timeout" lab/l16v2/pod_run.sh "$@" > "$OUT/$name.json" 2> "$OUT/$name.err"
  local rc=$?
  "$PY" -c "import json,sys; a=json.load(open(sys.argv[1])); print(a.get('outcome'), 'exit', a.get('exit_code'), 'seconds', a.get('seconds'), 'run', a.get('run_id')); print(a.get('output_tail','')[-1500:])" "$OUT/$name.json" > "$OUT/$name.txt" 2>&1
  return $rc
}
arm() { # arm <v2|old58>: "before=a/n after=b/n" from that arm's own answer, or "failed"
  "$PY" -c "
import json,sys
try:
    s=json.load(open(sys.argv[1])); n=s['n_items']
    print('before=%s/%d after=%s/%d' % ('/'.join(map(str, s['noise_runs'])), n, '/'.join(map(str, s['after_runs'])), n))
except Exception:
    print('failed')" "$OUT/agreement-$1/stdout.json"
}
gate_said() {
  "$PY" -c "
import json,sys
try:
    g=json.load(open(sys.argv[1])); s=g.get('scores', {})
    print('decision=%s current=%s candidate=%s teacher=%s on %s items' % (g.get('decision'), s.get('current'), s.get('candidate'), s.get('teacher'), g.get('n_items')))
except Exception:
    print('failed')" "$OUT/gate/gate.stdout.json"
}
close_row() { # close_row <result text>
  ctl stop 180 stop; plog "pod D servers stopped: $(grep -m1 'port 8100' "$OUT/stop.txt" | cut -c1-80)"
  bash lab/l16v2/score_pod.sh unforward
  "$PY" -m lab slot done --row "$ROW" --result "$1" > "$OUT/slot-done.json" 2>&1
  plog "pod D row done row=$ROW tag=$TAG after $(mins) min: $1"
}

"$PY" -m lab slot start --row "$ROW" > "$OUT/slot-start.json" 2>&1 || { plog "PROBLEM pod D row $ROW did not start: $(cut -c1-300 "$OUT/slot-start.json")"; exit 1; }
T0=$(date +%s)
plog "pod D row started row=$ROW tag=$TAG version=$VERSION"

ctl prep 420 prep || { plog "PROBLEM pod D prep failed: $(tail -n 3 "$OUT/prep.txt" | tr '\n' ' ' | cut -c1-300)"; close_row "prep failed; nothing scored"; exit 2; }
plog "pod D prep done: $(grep -m1 'imports ok' "$OUT/prep.txt" | cut -c1-120); $(grep -m1 '^source' "$OUT/prep.txt" | cut -c1-80)"
ctl package 900 package "$VERSION" "$ADAPTER" "$RUN_ID" || { plog "PROBLEM pod D packaging failed: $(tail -n 4 "$OUT/package.txt" | tr '\n' ' ' | cut -c1-400)"; close_row "packaging failed; nothing scored"; exit 3; }
AFTER_SHA=$(grep -E "Q4_K_M.gguf$" "$OUT/package.txt" | grep -oE "^[0-9a-f]{64}" | head -n 1)
export AFTER_SHA
plog "pod D packaging done version=$VERSION gguf_sha256=$AFTER_SHA $(grep -m1 '### package' "$OUT/package.txt")"
ctl serve 420 serve "$VERSION" || { plog "PROBLEM pod D serve failed: $(tail -n 4 "$OUT/serve.txt" | tr '\n' ' ' | cut -c1-400)"; close_row "serve failed; nothing scored"; exit 4; }
plog "pod D servers up (8100 before, 8101 after): $(grep -E '^810[01]:' "$OUT/serve.txt" | cut -c1-150 | tr '\n' ' ')"
"$PY" -m lab.transfer pull-file --account D "packages/$VERSION/manifest.json" "$OUT/manifest.json" > "$OUT/pull-manifest.json" 2>&1 \
  && plog "transfer pulled packages/$VERSION/manifest.json bytes=$(stat -c %s "$OUT/manifest.json")" \
  || plog "PROBLEM could not pull the package manifest: $(cut -c1-200 "$OUT/pull-manifest.json")"
bash lab/l16v2/score_pod.sh forward > "$OUT/forward.txt" 2>&1 || { close_row "forwarders failed; nothing scored"; exit 5; }

bash lab/l16v2/score_pod.sh v2 "$TAG" "$RUN_ID" "$V2" 1; RC_V2=$?
RESULT="v2 held-out $(arm v2)"
if [ "$(mins)" -ge 6 ]; then
  "$PY" -m lab slot extend --row "$ROW" --note "packaged $VERSION, both servers up and checked, v2 held-out agreement scored ($RESULT); the old fixture's 58 and the gate remain, about 10 minutes" > "$OUT/slot-extend.json" 2>&1 \
    && plog "pod D row extended once (15 min) at minute $(mins)" || plog "PROBLEM the row was not extended: $(cut -c1-200 "$OUT/slot-extend.json")"
fi
bash lab/l16v2/score_pod.sh old "$TAG" "$RUN_ID" 1; RC_OLD=$?
RESULT="$RESULT; old 58 $(arm old58)"
if [ -s "$OUT/manifest.json" ]; then
  bash lab/l16v2/score_pod.sh gate "$TAG" "$RUN_ID" "$OUT/manifest.json"; RC_GATE=$?
  RESULT="$RESULT; gate $(gate_said)"
else
  RC_GATE=9; RESULT="$RESULT; gate not run (no manifest)"
fi
close_row "$VERSION packaged (Q4_K_M sha256 ${AFTER_SHA:0:12}); tool-call agreement and gate, before = base samepath GGUF: $RESULT"
[ $RC_V2 -eq 0 ] && [ $RC_OLD -eq 0 ] && [ $RC_GATE -eq 0 ]
