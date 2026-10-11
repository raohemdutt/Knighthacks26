#!/bin/bash
# L16 retrain: score before against after from the workstation, against the two llama-server that
# lab/l16v2/pod_run.sh serve started on pod D (a copy of the L4 run's run_gate.sh, with the two agreement runs added).
# Run from the repo root (Git Bash), inside the Running GPU Queue row that holds the servers.
#
#   bash lab/l16v2/score_pod.sh forward                              the three local doors: 8100 before, 8101 after,
#                                                                    8200 the teacher (account C, greedy, thinking off)
#   bash lab/l16v2/score_pod.sh v2 <tag> <run id> <v2 dataset dir> [before runs]     tool-call agreement, v2 held out
#   bash lab/l16v2/score_pod.sh old <tag> <run id> [before runs]                     the first fixture's 58 held out
#   bash lab/l16v2/score_pod.sh gate <tag> <run id> <package manifest.json>          python -m lab eval-gate, as tonight
#   bash lab/l16v2/score_pod.sh unforward
#
# <tag> names the arm (a100 or podd); everything lands in <workspace>/work/L16v2-out/pod/<tag>/ and every step
# writes its line to progress.log. The scoring rules are lab/L16V2_RULES.md; nothing here changes them.
set -u
W="${L16V2_WORKSPACE:-/c/Users/hackathon/Code/Knighthacks26}"
PY="${L16V2_PYTHON:-$W/atlas/.venv/Scripts/python}"
FORWARD="${L16V2_FORWARD:-$W/tooling/dispatcher/pod_forward.py}"
POD="$W/work/L16v2-out/pod"
STEP="${1:-help}"
plog() { echo "$(date -u -d '-4 hours' '+%-I:%M %p ET') $*" >> "$W/work/L16v2-out/progress.log"; }
export LAB_TASK=L16 LAB_ACTOR="Ilan1 agent" LAB_TIER=opus PYTHONIOENCODING=utf-8 LAB_STATE_DIR="$POD/state"
mkdir -p "$POD/state"

forward() {
  local port account podport
  for spec in "8100 D 8100" "8101 D 8101" "8200 C 8000"; do
    set -- $spec; port=$1; account=$2; podport=$3
    if [ "$port" = 8200 ]; then
      (nohup "$PY" "$FORWARD" --account "$account" --pod-port "$podport" --listen "$port" \
        --set-json '{"chat_template_kwargs": {"enable_thinking": false}, "top_k": 1}' > "$POD/fwd-$port.log" 2>&1 &)
    else
      (nohup "$PY" "$FORWARD" --account "$account" --pod-port "$podport" --listen "$port" > "$POD/fwd-$port.log" 2>&1 &)
    fi
  done
  sleep 3
  for port in 8100 8101; do
    "$PY" -c "import json,sys,urllib.request as u; p=json.load(u.urlopen('http://127.0.0.1:%s/props' % sys.argv[1], timeout=30)); s=p['default_generation_settings']; print(sys.argv[1], 'top_k', s.get('params', s).get('top_k'), (p.get('model_path') or '').split('/')[-1])" "$port" \
      || { plog "PROBLEM forwarder $port does not answer /props"; return 1; }
  done
  "$PY" -c "import json,urllib.request as u; print(8200, [m['id'] for m in json.load(u.urlopen('http://127.0.0.1:8200/v1/models', timeout=30))['data']])" \
    || plog "PROBLEM forwarder 8200 (teacher) does not answer /v1/models"
  plog "forwarders up: 8100 before, 8101 after (account D), 8200 teacher (account C)"
}

unforward() {
  local pid
  for port in 8100 8101 8200; do
    pid=$(head -n 1 "$POD/fwd-$port.log" 2>/dev/null | grep -oE '[0-9]+')
    [ -n "$pid" ] && taskkill //PID "$pid" //F > /dev/null 2>&1
  done
  plog "forwarders stopped"
}

agreement() { # agreement <label> <tag> <run id> <dataset dir> <before runs>
  local label="$1" tag="$2" run_id="$3" dataset="$4" before_runs="${5:-1}" out started
  out="$POD/$tag/agreement-$label"; mkdir -p "$out"
  started=$(date +%s)
  plog "scoring arm start: tool-call agreement $label held_out tag=$tag before_runs=$before_runs after_runs=1"
  "$PY" -m lab tool-call-agreement --dataset "$dataset" --run-id "$run_id" --before-url http://127.0.0.1:8100 \
    --after-url http://127.0.0.1:8101 --out "$out" --side held_out --before-runs "$before_runs" --after-runs 1 \
    --before-sha256 84284aa2899f112244a6735d9fbb0c6f94085fb59eb719c95ec376e421eee04f ${AFTER_SHA:+--after-sha256 "$AFTER_SHA"} \
    > "$out/stdout.json" 2> "$out/stderr.log"
  local rc=$?
  if [ $rc -ne 0 ]; then plog "PROBLEM agreement $label tag=$tag failed rc=$rc: $(tail -n 1 "$out/stderr.log" | cut -c1-160) $(cut -c1-160 "$out/stdout.json")"; return $rc; fi
  plog "RESULT agreement $label tag=$tag $("$PY" -c "
import json,sys
s=json.load(open(sys.argv[1]))
m=s['mcnemar']
print('n=%d groups=%d before=%s after=%s gained_0_to_1=%d lost_1_to_0=%d p_mcnemar=%.4g p_sign_flip=%.4g noise=%.3f' % (s['n_items'], s['n_groups'], '/'.join(map(str, s['noise_runs'])), '/'.join(map(str, s['after_runs'])), m['gained_0_to_1'], m['lost_1_to_0'], m['p_value'], s['p_value'], s['noise']))" "$out/stdout.json") seconds=$(( $(date +%s) - started ))"
}

gate() { # gate <tag> <run id> <manifest.json>
  local tag="$1" run_id="$2" manifest="$3" out started rc
  out="$POD/$tag/gate"; mkdir -p "$out"
  started=$(date +%s)
  plog "scoring arm start: eval gate tag=$tag (current twice, candidate, teacher) on tool-calling-fixture"
  "$PY" -m lab eval-gate --dataset fixtures/learning/tool-calling-fixture --run-id "$run_id" \
    --current-url http://127.0.0.1:8100 --candidate-url http://127.0.0.1:8101 --teacher-url http://127.0.0.1:8200 \
    --candidate-manifest "$manifest" --decode-attested > "$out/gate.stdout.json" 2> "$out/gate.stderr.log"
  rc=$?
  echo "gate exit $rc" >> "$out/gate.stderr.log"
  cp -r "$POD/state/eval-gate/$run_id" "$out/files" 2>/dev/null
  plog "RESULT gate tag=$tag $("$PY" -c "
import json,sys
try:
    g=json.load(open(sys.argv[1]))
    s=g.get('scores', {})
    print('decision=%s current=%s candidate=%s teacher=%s n_items=%s n_groups=%s p=%s noise=%s reasons=%s' % (g.get('decision'), s.get('current'), s.get('candidate'), s.get('teacher'), g.get('n_items'), g.get('n_groups'), g.get('p_value'), g.get('noise'), ' | '.join(g.get('reasons', []))[:300]))
except Exception as exc:
    print('unreadable gate output: %s' % type(exc).__name__)" "$out/gate.stdout.json") rc=$rc seconds=$(( $(date +%s) - started ))"
  return $rc
}

case "$STEP" in
  forward) forward ;;
  unforward) unforward ;;
  v2) agreement v2 "$2" "$3" "$4" "${5:-1}" ;;
  old) agreement old58 "$2" "$3" fixtures/learning/tool-calling-fixture "${4:-1}" ;;
  gate) gate "$2" "$3" "$4" ;;
  *) sed -n '2,16p' "$0"; [ "$STEP" = help ] || exit 2 ;;
esac
