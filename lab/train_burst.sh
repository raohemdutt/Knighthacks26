#!/bin/bash
# A short, real LoRA training burst on this pod, for the web app's "Train for a minute" button (L37).
#
#   bash lab/train_burst.sh POD [SECONDS]        POD is C or D; SECONDS defaults to 60
#
# Run through the lab's controller (python -m lab.controller run-script --account D lab/train_burst.sh D 60), as the
# gateway's POST /api/learning/lab/train-burst does. It trains Qwen3.5-4B with the R66 proof's LoRA settings (rank 16,
# alpha 32, batch 4 x grad accumulation 4, lr 2e-4, seed 42, ATLAS_TRAIN_WORKERS=0) for enough optimizer steps to fill
# about SECONDS: one step took about 64 s beside the serving 122B pair in the R66 proof (129 s for 2 steps), so 60 s
# is 1 step and 120 s is 2. The model load adds about 15 s on top.
#
# lab/train.py deletes its dataset folder and rows file when it exits, so it never sees the only copy: the persistent
# copy lives in ~/burst-data (tool-calling-fixture/ and rows.jsonl, pushed once with lab.transfer push-dataset
# --to burst-data and push-file), and every burst trains on a fresh copy under ~/burst-runs/<id>/data.
#
# It never touches the pair's processes (vLLM, its tunnels, relays, samplers): it only starts its own trainer, under
# a lock so two bursts never run at once on one pod, and stops it with a timeout. The last line before the
# controller's end marker is the result, one JSON object:
#
#   BURST_RESULT {"pod", "ok", "rc", "steps", "seconds", "train_s", "loss", "peak_gb", "vram_before_gb",
#                 "vram_peak_gb", "started", "ended", "error"}
#
# `seconds` is the trainer's wall time including the model load, `train_s` the trainer's own train_runtime, `peak_gb`
# the GPU memory the burst added at its peak (peak minus before, read with amd-smi every 2 s), `loss` the trainer's
# train_loss. Exit codes: 0 trained, 1 the trainer failed, 2 usage or missing data, 3 a burst already runs here.
set -u
POD="${1:-}"; WANT_S="${2:-60}"
case "$POD" in C|D) ;; *) echo "usage: train_burst.sh C|D [SECONDS]"; exit 2 ;; esac
case "$WANT_S" in ''|*[!0-9]*) echo "SECONDS must be a whole number"; exit 2 ;; esac
[ "$WANT_S" -ge 30 ] && [ "$WANT_S" -le 600 ] || { echo "SECONDS must be between 30 and 600"; exit 2; }
STEP_S="${BURST_STEP_S:-64}"
STEPS=$(( (WANT_S + STEP_S / 2) / STEP_S )); [ "$STEPS" -ge 1 ] || STEPS=1

VENV="$HOME/venvs/atlas-train"; MODEL="$HOME/models/Qwen3.5-4B"; KEEP="$HOME/burst-data"; RUNS="$HOME/burst-runs"
SRC="$HOME/l10/src"; [ -f "$SRC/lab/train.py" ] || SRC="$HOME/r66/src"
ID="burst-$POD-$(date -u +%Y%m%dT%H%M%SZ)"; WORK="$RUNS/$ID"; OUT="$HOME/adapters/$ID"

result() {  # result OK RC ERROR: the one JSON line the gateway reads
  python3 - "$@" <<'PY'
import json, os, sys
ok, rc, error = sys.argv[1] == "1", int(sys.argv[2]), sys.argv[3] or None
env = os.environ
def num(key, cast=float):
    try:
        return cast(env.get(key, ""))
    except ValueError:
        return None
before, peak = num("B_BEFORE_MB"), num("B_PEAK_MB")
print("BURST_RESULT " + json.dumps({
    "pod": env.get("B_POD"), "ok": ok, "rc": rc, "steps": num("B_STEPS", int), "seconds": num("B_WALL"),
    "train_s": num("B_TRAIN_S"), "loss": num("B_LOSS"),
    "peak_gb": round((peak - before) / 1024, 1) if peak is not None and before is not None else None,
    "vram_before_gb": round(before / 1024, 1) if before is not None else None,
    "vram_peak_gb": round(peak / 1024, 1) if peak is not None else None,
    "started": env.get("B_STARTED") or None, "ended": env.get("B_ENDED") or None, "error": error}))
PY
}
export B_POD="$POD" B_STEPS="" B_WALL="" B_TRAIN_S="" B_LOSS="" B_BEFORE_MB="" B_PEAK_MB="" B_STARTED="" B_ENDED=""

mkdir -p "$RUNS"
exec 9>"$RUNS/lock"
if ! flock -n 9; then result 0 3 "a training burst already runs on pod $POD"; exit 3; fi
for need in "$VENV/bin/python" "$MODEL/config.json" "$SRC/lab/train.py" "$KEEP/rows.jsonl" \
            "$KEEP/tool-calling-fixture/manifest.json" "$KEEP/tool-calling-fixture/examples.jsonl"; do
  [ -e "$need" ] || { result 0 2 "missing on pod $POD: ${need/#$HOME/~}"; exit 2; }
done

# a fresh copy of the data for this burst (train.py deletes it on exit)
mkdir -p "$WORK/data" "$OUT"
cp -r "$KEEP/tool-calling-fixture" "$WORK/data/" && cp "$KEEP/rows.jsonl" "$WORK/data/rows.jsonl" \
  || { result 0 2 "could not copy the training data"; exit 2; }

vram() { amd-smi metric --mem-usage 2>/dev/null | grep -E "USED_VRAM" | head -1 | awk '{print $2}'; }
B_BEFORE_MB="$(vram)"; export B_BEFORE_MB
( while sleep 2; do vram; done ) > "$WORK/vram.txt" &
MON=$!
echo "burst $ID: $STEPS step(s) for about $WANT_S s, vram before ${B_BEFORE_MB:-?} MB, code $SRC"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LAB_STATE_DIR="$RUNS/state" LAB_TASK=L37 ATLAS_TRAIN_WORKERS=0
mkdir -p "$RUNS/state"
B_STARTED="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"; export B_STARTED
t0=$(date +%s.%N)
(cd "$SRC" && timeout -s INT -k 30 $(( WANT_S * 4 + 240 )) "$VENV/bin/python" -u lab/train.py \
  --dataset "$WORK/data/tool-calling-fixture" --rows "$WORK/data/rows.jsonl" --model "$MODEL" \
  --base-repo Qwen/Qwen3.5-4B --base-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out "$OUT" --run-id "$ID" \
  --epochs 1 --max-steps "$STEPS" --lr 2e-4 --batch 4 --grad-accum 4 --rank 16 --alpha 32 --dropout 0 --seed 42 \
  --no-install) > "$WORK/train.log" 2>&1
rc=$?
B_ENDED="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"; export B_ENDED
kill $MON 2>/dev/null
B_WALL="$(python3 -c "import time; print(round(time.time() - $t0, 1))")"; export B_WALL
B_PEAK_MB="$(sort -n "$WORK/vram.txt" | tail -1)"; [ -n "$B_PEAK_MB" ] || B_PEAK_MB="$B_BEFORE_MB"; export B_PEAK_MB
B_TRAIN_S="$(grep -aoE "'train_runtime': '[0-9.]+'" "$WORK/train.log" | tail -1 | grep -oE '[0-9.]+')"; export B_TRAIN_S
B_LOSS="$(grep -aoE "'train_loss': '[0-9.]+'" "$WORK/train.log" | tail -1 | grep -oE '[0-9.]+')"; export B_LOSS
B_STEPS="$STEPS"; export B_STEPS
grep -aE "^\{'loss'|train_runtime|Error|error" "$WORK/train.log" | cut -c1-240 | tail -6
cp "$OUT/run.json" "$WORK/run.json" 2>/dev/null
rm -rf "$OUT" "$WORK/data"   # the adapter of a demo burst is not kept; the log and run.json stay in ~/burst-runs
if [ "$rc" -eq 0 ]; then
  result 1 0 ""
  exit 0
fi
result 0 1 "the trainer exited with code $rc ($(grep -aE "Error|error" "$WORK/train.log" | tail -1 | cut -c1-160))"
exit 1
