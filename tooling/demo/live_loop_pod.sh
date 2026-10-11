#!/bin/bash
# L29: the pods' side of the live learning-loop demo (tooling/demo/live_loop.sh runs it; see LIVE_LOOP_RUN_OF_SHOW.md).
# Two acts of the L10 group fine-tune (lab/pod_b_group_finetune.sh run, the same verb and flags), one after the other:
#   act 1 "solo": the ilan member alone;  act 2 "duo": ilan and dom pooled (SOLO_EXAMPLES / DUO_EXAMPLES per member).
# Each act: pool, label at the teacher (TEACHER_URL, pod C's port 8000: the 122B pair served across pods C and D, or
# the 27B on pod C; TEACHER_EXPECT names the checkpoint), train a LoRA (STEPS short steps, one loss row per step),
# package, stand-in gate, offer. Act 2 starts labeling when act 1's labels are in, and trains while act 1 packages.
# With DUO_PODS=2 act 2 trains on both pods at once (R65's two-pod LoRA, lab/r65_local_sgd.py): rank 0 on pod D in
# the verb's place, rank 1 on pod C, each on half the rows, deltas swapped through the hub proxy every 4 steps.
#
#   bash live_loop_pod.sh check                         (pod D or C) what the run needs is here; disk; sampler; jobs
#   bash live_loop_pod.sh start <stamp> <slot> [steps]  (pod D) reads the teacher token from ./l29_tt and the settings
#                                                       from ./l29_env (both uploaded beside this script with
#                                                       run-script --with; the token file is deleted at once), starts
#                                                       the two acts in the background and returns
#   bash live_loop_pod.sh rank1 <stamp> <slot> <pod D's hub user>
#                                                       (pod C, DUO_PODS=2) reads pod D's hub token from ./l29_td
#                                                       (deleted at once), waits in the background for act 2's
#                                                       handoff, then trains rank 1 and returns at once
#
# What it writes. Pod D: ~/l29/<stamp>/stages.jsonl, one line per stage change on pod D's clock (the relay reads it),
# ~/l29/<stamp>/driver.log, and per act ~/group-runs/<run id>/ and ~/data/packages/atlas-group-desk-objects-<run id>/
# as the L10 verb writes them. Pod C (rank 1): ~/l29/<stamp>/stages-c.jsonl (rank1_train, rank1_trained, on pod C's
# clock) and ~/l29/<stamp>/rank1/ (telemetry.jsonl, run.json, local_sgd.json; the adapter is not kept).
#
# The pooled data. The verb deletes the pooled copies on every way out, as in L10. For the two-pod act the trainer's
# copy of the dataset and rank 1's half of the rows go to pod C once, through pod D's handoff server on 127.0.0.1
# (HANDOFF_PORT) behind the hub proxy (only pod D's own account token opens it); lab/train.py deletes rank 1's copy on
# pod C when it ends, the waiter deletes what is left on every way out, and pod D deletes the handoff when rank 0 ends.
set -u -o pipefail
STEP="${1:-help}"
HERE="$(cd "$(dirname "$0")" && pwd)"
L10="$HOME/l10"
VENV="$HOME/venvs/atlas-train"
BIGPOOL="${BIGPOOL:-$L10/bigpool.tar.gz}"   # the L10 smoke exports: ilan/desk-objects-ilan-smoke, dom/... (100 each)
TASK="desk-objects"
SOLO="l29-solo"
DUO="l29-duo"
HANDOFF_PORT="${HANDOFF_PORT:-29562}"   # below 29600-29727, which the 122B pair's network shim pins (R66)
PROXY_HOST="${PROXY_HOST:-proxy-public.jupyterhub.svc.cluster.local}"   # the hub proxy inside the cluster (R65)
TS_BIN="${TS_BIN:-$HOME/ts/bin/tailscale}"
TS_SOCK="${TS_SOCK:-$HOME/ts/tailscaled.sock}"

now() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }

peer_ip() {   # peer_ip <pod-c|pod-d>: the other pod's tailnet address (PEER_IP overrides), as lab/pod_role.sh finds it
  if [ -n "${PEER_IP:-}" ]; then echo "$PEER_IP"; return; fi
  "$TS_BIN" --socket "$TS_SOCK" status 2>/dev/null | awk -v n="$1" '$2 == n {print $1; exit}'
}

two_pod_ready() {   # lab/r65_local_sgd.py and the lab/train.py that takes its callback (R65, PR 410) in this source tree
  [ -f "$1/lab/r65_local_sgd.py" ] && grep -q "no-relay-timing" "$1/lab/r65_local_sgd.py" && grep -q "callbacks" "$1/lab/train.py"
}

check() {
  local ok=1
  [ -x "$VENV/bin/python" ] || { echo "missing: $VENV"; ok=0; }
  [ -f "$L10/src/lab/pod_b_group_finetune.sh" ] || { echo "missing: $L10/src (lab/pod_b_group_finetune.sh bundle + prep)"; ok=0; }
  [ -x "$L10/llama.cpp/build/bin/llama-quantize" ] || echo "no $L10/llama.cpp links (prep): needed on pod D only"
  [ -f "$BIGPOOL" ] || echo "no $BIGPOOL: needed on pod D only"
  echo "pool members in the bundle: $(tar -tzf "$BIGPOOL" 2>/dev/null | grep -c 'examples.jsonl$') exports"
  echo "two-pod LoRA in $L10/src: $(two_pod_ready "$L10/src" && echo yes || echo 'no (re-bundle and prep from main at or after PR 410 and this change)')"
  local pc pd
  pc="$(peer_ip pod-c)"; pd="$(peer_ip pod-d)"
  echo "tailnet peers: pod-c ${pc:-none} pod-d ${pd:-none}"
  # the two-pod act needs both names on the tailnet; printing them is not a test (R66: both daemons were dead and the
  # check still passed, then rank 1 refused at the start)
  [ -n "$pc" ] && [ -n "$pd" ] || { echo "missing: a tailnet peer (lab/pod_tailscale.sh up pod-c and up pod-d)"; ok=0; }
  echo "disk free in home: $(df -h "$HOME" | awk 'NR==2{print $4}')"
  echo "memory available: $(awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
  echo "sampler: $(cat "$HOME/atlas-power/sampler.pid" 2>/dev/null) $(ps -p "$(cat "$HOME/atlas-power/sampler.pid" 2>/dev/null)" -o etime= 2>/dev/null)"
  echo "jobs: $(pgrep -af 'lab/train.py|r65_local_sgd|group-finetune|llama-server|vllm' | grep -v defunct | cut -c1-120 | tr '\n' ';')"
  [ "$ok" = 1 ] && echo "check ok"
}

# The helpers the trainer wrapper and rank 1 run: the pool trim, the two-pod handoff, its fetch, rank 1's arguments.
write_helpers() {
  cat > "$1/l29_split.py" <<'PY'
"""L29's helpers on the pods: trim a member's export, split the rows for two ranks, fetch the handoff."""
import hashlib, json, os, shutil, sys, tarfile, time, urllib.request
from pathlib import Path

def flag(args, name):
    return args[args.index(name) + 1] if name in args else None

def setflag(args, name, value):
    if name in args:
        args[args.index(name) + 1] = value
    else:
        args += [name, value]
    return args

def trim(folder, n):
    """Keep the first n examples of every export under folder (examples.jsonl and the manifest's count); the verb
    freezes its own split, so an export's split.json, if any, goes."""
    n = int(n)
    for ex in Path(folder).rglob("examples.jsonl"):
        lines = [x for x in ex.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(lines) <= n:
            continue
        ex.write_text("".join(x + "\n" for x in lines[:n]), encoding="utf-8")
        man = ex.with_name("manifest.json")
        if man.is_file():
            m = json.loads(man.read_text(encoding="utf-8"))
            m["examples"] = n
            man.write_text(json.dumps(m), encoding="utf-8")
        ex.with_name("split.json").unlink(missing_ok=True)
        print(f"trimmed {ex.parent.name} to {n} examples")

def handoff(run, *args):
    """Split the trainer's rows in two (even lines rank 0, odd lines rank 1), write rank 0's arguments, and pack the
    dataset copy, rank 1's rows and the arguments for pod C into <run>/handoff/handoff.tar with its sha256 beside."""
    run, args = Path(run), list(args)
    rows = Path(flag(args, "--rows"))
    lines = [x for x in rows.read_text(encoding="utf-8").splitlines() if x.strip()]
    (run / "rows.rank0.jsonl").write_text("".join(x + "\n" for x in lines[0::2]), encoding="utf-8")
    (run / "rows.rank1.jsonl").write_text("".join(x + "\n" for x in lines[1::2]), encoding="utf-8")
    rank0 = setflag(list(args), "--rows", str(run / "rows.rank0.jsonl"))
    (run / "rank0.args").write_text("\n".join(rank0) + "\n", encoding="utf-8")
    box = run / "handoff"
    box.mkdir(exist_ok=True)
    dataset = Path(flag(args, "--dataset"))
    with tarfile.open(box / "handoff.tar", "w") as tar:
        tar.add(dataset, arcname=f"dataset/{dataset.name}")
        tar.add(run / "rows.rank1.jsonl", arcname="rows.rank1.jsonl")
        info = json.dumps({"args": args, "home": os.environ.get("HOME", ""), "dataset": dataset.name}).encode()
        t = tarfile.TarInfo("args.json"); t.size = len(info)
        import io; tar.addfile(t, io.BytesIO(info))
    (run / "rows.rank1.jsonl").unlink()
    digest = hashlib.sha256((box / "handoff.tar").read_bytes()).hexdigest()
    (box / "ready").write_text(digest + "\n", encoding="utf-8")
    print(f"handoff {len(lines)} rows: {len(lines[0::2])} here, {len(lines[1::2])} for pod C")

def fetch(user, port, token_file, out, wait_s):
    """Wait for pod D's handoff through the hub proxy, fetch it, check its sha256, unpack it into out."""
    token = Path(token_file).read_text().strip()
    base = f"http://{os.environ.get('PROXY_HOST', 'proxy-public.jupyterhub.svc.cluster.local')}/user/{user}/proxy/{port}"
    def get(name):
        req = urllib.request.Request(f"{base}/{name}", headers={"Authorization": f"token {token}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    deadline = time.time() + float(wait_s)
    while True:
        try:
            digest = get("ready").decode().strip()
            break
        except Exception:
            if time.time() > deadline:
                print("no handoff from pod D in time"); return 1
            time.sleep(1.0)
    blob = get("handoff.tar")
    if hashlib.sha256(blob).hexdigest() != digest:
        print("handoff sha256 does not match"); return 1
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "handoff.tar").write_bytes(blob)
    with tarfile.open(out / "handoff.tar") as tar:
        try:
            tar.extractall(out, filter="data")   # no links out of the folder, no special files
        except TypeError:   # a Python without extraction filters: the tar is pod D's own, made just above
            tar.extractall(out)
    (out / "handoff.tar").unlink()
    print(f"handoff fetched: {len(blob)} bytes"); return 0

def rank1_args(data, out, run_id, power_node):
    """Rank 1's lab/train.py arguments: pod D's, with this pod's dataset, rows, out, run id and sampler node."""
    data = Path(data)
    meta = json.loads((data / "args.json").read_text(encoding="utf-8"))
    args = [a.replace(meta["home"], os.environ["HOME"], 1) if meta["home"] and a.startswith(meta["home"]) else a
            for a in meta["args"]]
    for name, value in (("--dataset", str(data / "dataset" / meta["dataset"])), ("--rows", str(data / "rows.rank1.jsonl")),
                        ("--out", out), ("--run-id", run_id), ("--power-node", power_node)):
        args = setflag(args, name, value)
    print("\n".join(args))

if __name__ == "__main__":
    step, rest = sys.argv[1], sys.argv[2:]
    sys.exit({"trim": trim, "handoff": handoff, "fetch": fetch, "rank1_args": rank1_args}[step](*rest) or 0)
PY
  # one rank of the two-pod LoRA with the act's held-out floor (the stand-in gate never deploys)
  cat > "$1/l29_rank.py" <<'PY'
import runpy, sys
import lab.dataset as d
d.load_split.__kwdefaults__["min_held_out"] = int(sys.argv[1])
sys.argv = ["lab/r65_local_sgd.py"] + sys.argv[2:]
runpy.run_path("lab/r65_local_sgd.py", run_name="__main__")
PY
}

# The trainer wrapper the verb runs lab/train.py under (--trainer-python): it marks the end of labeling, waits for
# the file in L29_WAIT (with --serial act 2 waits for act 1 to end), marks the start and the end of training. With
# L29_SPLIT=1 it trains rank 0 of the two-pod LoRA in lab/train.py's place and hands rank 1's half to pod C.
write_wrapper() {
  cat > "$1" <<'SH'
#!/bin/bash
mark() { printf '{"act":"%s","stage":"%s","ts":"%s"}\n' "$L29_ACT" "$1" "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" >> "$L29_STAGES"; }
mark labeled
if [ -n "${L29_WAIT:-}" ]; then while [ ! -e "$L29_WAIT" ]; do sleep 0.2; done; fi
mark train
if [ "${L29_SPLIT:-0}" = 1 ] && [ "${1:-}" = "-u" ] && [ "${2:-}" = "lab/train.py" ]; then
  shift 2
  "$L29_PY" "$L29_RUN/l29_split.py" handoff "$L29_RUN" "$@" >> "$L29_RUN/duo.log" 2>&1 || { mark failed; exit 1; }
  (cd "$L29_RUN/handoff" && exec python3 -m http.server "$L29_HANDOFF_PORT" --bind 127.0.0.1) > "$L29_RUN/handoff.log" 2>&1 &
  hs=$!
  ( umask 077; printf '%s' "$TEACHER_TOKEN" > "$L29_RUN/tokpeer0" )   # pod C's account token; r65 reads and deletes it
  readarray -t a0 < "$L29_RUN/rank0.args"
  "$L29_PY" -u "$L29_RUN/l29_rank.py" "${L29_MIN_HELD_OUT:-1}" --rank 0 --peer-tailnet "$L29_PEER_C" \
    --peer-user "$L29_C_USER" --token-file "$L29_RUN/tokpeer0" --k 4 --no-relay-timing --ctl-port 29560 --box-port 29561 --wait-s 180 -- "${a0[@]}" --no-install
  rc=$?
  kill "$hs" 2>/dev/null
  rm -rf "$L29_RUN/handoff" "$L29_RUN/tokpeer0" "$L29_RUN/rows.rank0.jsonl"
elif [ -n "${L29_MIN_HELD_OUT:-}" ] && [ "${1:-}" = "-u" ] && [ "${2:-}" = "lab/train.py" ]; then
  # a small act freezes fewer held-out examples than the 50 a deploying gate needs; this act never deploys (stand-in
  # gate), so its trainer reads the split with the act's own floor; nothing else changes.
  shift 2
  "$L29_PY" -u -c 'import sys; import lab.dataset as d; d.load_split.__kwdefaults__["min_held_out"] = int(sys.argv[1]); sys.argv = ["lab/train.py"] + sys.argv[2:] + ["--no-install"]; import lab.train as t; sys.exit(t._start())' "$L29_MIN_HELD_OUT" "$@"
  rc=$?
else
  "$L29_PY" "$@"
  rc=$?
fi
mark trained
# act 1 packages on the CPU while act 2 trains: the act 1 verb (this script's parent) and the packaging steps it
# starts next run at low priority, so act 2's trainer, which is on the critical path, keeps the cores it needs.
[ -n "${L29_NICE_AFTER:-}" ] && renice -n "$L29_NICE_AFTER" -p "$PPID" > /dev/null 2>&1
exit $rc
SH
  chmod +x "$1"
}

roster() {   # roster <circle> <examples per member> <member> ...: the pool folder, emptied of old exports, with its pool.json
  local circle="$1" per="$2"; shift 2
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
  "$VENV/bin/python" "$RUN/l29_split.py" trim "$dir" "$per" >> "$RUN/driver.log" 2>&1
}

act() {   # act <name> <circle> <run id> <members> [wait file]: one L10 run, with its stage lines
  local name="$1" circle="$2" run_id="$3" members="$4" wait="${5:-}"
  local split=0 pods='["pod-d"]' parts='{}'
  if [ "$name" = duo ] && [ "${DUO_PODS:-1}" = 2 ]; then
    split=1 pods='["pod-d","pod-c"]' parts="{\"pod-c\":\"$run_id.pod-c\"}"
  fi
  printf '{"act":"%s","stage":"start","ts":"%s","run_id":"%s","circle":"%s","members":%s,"pods":%s,"parts":%s}\n' \
    "$name" "$(now)" "$run_id" "$circle" "$members" "$pods" "$parts" >> "$STAGES"
  local nice_after=""
  [ "$name" = solo ] && nice_after=15
  L29_ACT="$name" L29_WAIT="$wait" L29_STAGES="$STAGES" L29_PY="$VENV/bin/python" L29_MIN_HELD_OUT=1 \
    L29_NICE_AFTER="$nice_after" L29_SPLIT="$split" L29_RUN="$RUN" L29_HANDOFF_PORT="$HANDOFF_PORT" \
    L29_PEER_C="$(peer_ip pod-c)" L29_C_USER="${C_USER:-}" \
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
  printf '{"act":"all","stage":"go","ts":"%s","solo":"%s","duo":"%s","steps":%s,"workers":%s,"solo_examples":%s,"duo_examples":%s,"duo_pods":%s,"teacher_expect":"%s"}\n' \
    "$(now)" "$a1" "$a2" "$L10_STEPS" "$LABEL_WORKERS" "$SOLO_EXAMPLES" "$DUO_EXAMPLES" "$DUO_PODS" "$TEACHER_EXPECT" >> "$STAGES"
  write_helpers "$RUN"
  write_wrapper "$RUN/trainer.sh"
  roster "$SOLO" "$SOLO_EXAMPLES" ilan || { echo '{"act":"all","stage":"failed","why":"pool solo"}' >> "$STAGES"; return 1; }
  roster "$DUO" "$DUO_EXAMPLES" ilan dom || { echo '{"act":"all","stage":"failed","why":"pool duo"}' >> "$STAGES"; return 1; }
  act solo "$SOLO" "$a1" 1 &
  local p1=$!
  # act 2 starts labeling the moment act 1 hands its labels to the trainer (or ends)
  while ! grep -q '"act":"solo","stage":"labeled"' "$STAGES" && [ ! -e "$RUN/solo.exit" ]; do sleep 0.2; done
  # act 2 trains as soon as its labels are in, while act 1 packages on the CPU (L29_SERIAL=1: wait for act 1 to end)
  local wait=""
  [ "${L29_SERIAL:-0}" = 1 ] && wait="$RUN/solo.exit"
  act duo "$DUO" "$a2" 2 "$wait" &
  local p2=$!
  wait $p1 $p2
  printf '{"act":"all","stage":"end","ts":"%s"}\n' "$(now)" >> "$STAGES"
}

read_env() {   # the relay's settings from ./l29_env: KEY=VALUE lines, only these keys, nothing else is sourced
  local key value
  [ -f "$HERE/l29_env" ] || return 0
  while IFS='=' read -r key value; do
    case "$key" in
      SOLO_EXAMPLES|DUO_EXAMPLES|DUO_PODS) [[ "$value" =~ ^[0-9]+$ ]] && export "$key=$value" ;;
      TEACHER_EXPECT) [[ "$value" =~ ^[A-Za-z0-9._|-]*$ ]] && export "$key=$value" ;;
      C_USER|D_USER) [[ "$value" =~ ^[A-Za-z0-9._-]+$ ]] && export "$key=$value" ;;
      L29_SERIAL) [ "$value" = 1 ] && export L29_SERIAL=1 ;;
    esac
  done < "$HERE/l29_env"
  rm -f "$HERE/l29_env"
}

start() {
  local stamp="${1:-}" slot="${2:-}" steps="${3:-8}"
  [ -n "$stamp" ] && [ -n "$slot" ] || { echo "usage: start <stamp> <slot row> [steps]"; return 2; }
  umask 077
  local tt="$HERE/l29_tt"
  [ -s "$tt" ] || { echo "no token file beside the script"; return 1; }
  TEACHER_TOKEN="$(cat "$tt")"
  rm -f "$tt"
  read_env
  local run="$HOME/l29/$stamp"
  [ -e "$run" ] && { echo "$run exists: a stamp is used once"; return 1; }
  if [ "${DUO_PODS:-1}" = 2 ]; then
    two_pod_ready "$L10/src" || { echo "act 2 on two pods needs lab/r65_local_sgd.py --no-relay-timing in $L10/src: re-bundle and prep"; return 1; }
    [ -n "$(peer_ip pod-c)" ] || { echo "act 2 on two pods needs pod C on the tailnet (lab/pod_tailscale.sh up pod-c)"; return 1; }
    [ -n "${C_USER:-}" ] || { echo "act 2 on two pods needs pod C's hub user in l29_env"; return 1; }
  fi
  mkdir -p "$run"
  export TEACHER_TOKEN GPU_SLOT="$slot" L10_STEPS="$steps" LABEL_WORKERS="${LABEL_WORKERS:-16}" GATE=fake \
    MIN_HELD_OUT=1 RUN_TIMEOUT_S=900 SOLO_EXAMPLES="${SOLO_EXAMPLES:-100}" DUO_EXAMPLES="${DUO_EXAMPLES:-100}" \
    DUO_PODS="${DUO_PODS:-1}" TEACHER_EXPECT="${TEACHER_EXPECT:-Qwen3.8-27B}" C_USER="${C_USER:-}" HANDOFF_PORT
  cp "$0" "$run/pod.sh"
  setsid nohup bash "$run/pod.sh" drive "$stamp" > "$run/driver.log" 2>&1 < /dev/null &
  unset TEACHER_TOKEN
  echo "started $stamp: ~/l29/$stamp/stages.jsonl (act 2 on ${DUO_PODS:-1} pod(s), teacher $TEACHER_EXPECT)"
}

# --- pod C: rank 1 of act 2 ---------------------------------------------------------------------------------------

rank1() {
  local stamp="${1:-}" slot="${2:-}" d_user="${3:-}"
  [ -n "$stamp" ] && [ -n "$slot" ] && [ -n "$d_user" ] || { echo "usage: rank1 <stamp> <slot row> <pod D's hub user>"; return 2; }
  umask 077
  local run="$HOME/l29/$stamp"
  [ -e "$run" ] && { echo "$run exists: a stamp is used once"; rm -f "$HERE/l29_td"; return 1; }
  [ -s "$HERE/l29_td" ] || { echo "no token file beside the script"; return 1; }
  two_pod_ready "$L10/src" || { rm -f "$HERE/l29_td"; echo "pod C needs lab/r65_local_sgd.py --no-relay-timing in $L10/src: bundle and prep there too"; return 1; }
  [ -n "$(peer_ip pod-d)" ] || { rm -f "$HERE/l29_td"; echo "pod C needs pod D on the tailnet (lab/pod_tailscale.sh up pod-d)"; return 1; }
  mkdir -p "$run"
  mv "$HERE/l29_td" "$run/tokd"
  cp "$0" "$run/pod.sh"
  GPU_SLOT="$slot" setsid nohup bash "$run/pod.sh" rank1_wait "$stamp" "$d_user" > "$run/rank1.log" 2>&1 < /dev/null &
  echo "rank 1 waiting for act 2's handoff: ~/l29/$stamp/stages-c.jsonl"
}

rank1_wait() {
  local stamp="$1" d_user="$2"
  local run="$HOME/l29/$stamp" act2="live-$stamp-duo"
  local stages="$run/stages-c.jsonl" data="$run/data" out="$run/rank1"
  trap 'rm -rf "$data" "$run/tokd" "$run/tokpeer1" "$out/adapter"' EXIT
  mark1() { printf '{"act":"duo","stage":"%s","ts":"%s","node":"pod-c","run_id":"%s"%s}\n' "$1" "$(now)" "$act2.pod-c" "${2:-}" >> "$stages"; }
  write_helpers "$run"
  PROXY_HOST="$PROXY_HOST" "$VENV/bin/python" "$run/l29_split.py" fetch "$d_user" "$HANDOFF_PORT" "$run/tokd" "$data" "${RANK1_WAIT_S:-600}" \
    || { mark1 rank1_failed ',"why":"no handoff"'; return 1; }
  readarray -t a1 < <("$VENV/bin/python" "$run/l29_split.py" rank1_args "$data" "$out" "$act2.pod-c" "${POWER_NODE_C:-gpu-a}")
  cp "$run/tokd" "$run/tokpeer1"   # r65 reads and deletes it; tokd goes on exit
  mark1 rank1_train
  (cd "$L10/src" && PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LAB_STATE_DIR="$run/state" LAB_TASK=L29 \
    "$VENV/bin/python" -u "$run/l29_rank.py" 1 --rank 1 --peer-tailnet "$(peer_ip pod-d)" --peer-user "$d_user" \
    --token-file "$run/tokpeer1" --k 4 --no-relay-timing --ctl-port 29560 --box-port 29561 --wait-s 180 -- "${a1[@]}" --no-install) >> "$run/rank1.log" 2>&1
  local rc=$?
  mark1 rank1_trained ",\"rc\":$rc"
}

case "$STEP" in
  check) check ;;
  start) shift; start "$@" ;;
  drive) shift; drive "$@" ;;
  rank1) shift; rank1 "$@" ;;
  rank1_wait) shift; rank1_wait "$@" ;;
  *) sed -n '2,27p' "$0" ;;
esac
echo "### END"
