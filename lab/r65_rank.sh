#!/bin/bash
# R65: one rank of the two-pod local-SGD LoRA (lab/r65_local_sgd.py) on this pod, inside its GPU Queue slot.
#   bash r65_rank.sh <rank 0|1> <run id> [peer tailnet IP]
# Needs ~/r65/bundle.tar.gz (lab/ and contracts/ at the branch), ~/r65/data/<dataset id>/ and ~/r65/data/rows.jsonl
# (this rank's half of the L16 rows), the L16 venv and base (bash lab/l16_run.sh prep), and for rank 1 Tailscale up
# (lab/pod_tailscale.sh). Rank 0 listens on 127.0.0.1:$PORT; rank 1 reaches it through tailscaled's SOCKS5 proxy.
# The adapter, run.json, telemetry, local_sgd.json and the log go to ~/adapters/r65-<run id>-rank<rank>/.
set -u -o pipefail
RANK="${1:?rank}"; RUN="${2:?run id}"; PEER="${3:-}"
PORT="${PORT:-29600}"
TIMEOUT_S="${TIMEOUT_S:-1500}"   # a 15-minute slot plus its one extension
VENV="$HOME/venvs/atlas-train"
SRC="$HOME/r65/src"
OUT="$HOME/adapters/r65-$RUN-rank$RANK"
rm -rf "$SRC"; mkdir -p "$SRC" "$HOME/r65/state"
BUNDLE="${BUNDLE:-$(ls -t "$HOME"/r65/bundle*.tar.gz 2>/dev/null | head -1)}"
echo "bundle: $BUNDLE"
tar -xzf "$BUNDLE" -C "$SRC" || { echo "no bundle"; echo "### END"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists"; echo "### END"; exit 1; }
mkdir -p "$OUT"
extra=(--port "$PORT")
[ "$RANK" = 1 ] && extra+=(--peer "$PEER" --socks 127.0.0.1:1055)
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LAB_STATE_DIR="$HOME/r65/state" LAB_TASK=R65
t0=$(date +%s)
echo "### R65 rank $RANK run $RUN start $(date -u +%FT%TZ) host $(hostname) mem_avail_gib $(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)"
(cd "$SRC" && timeout -s INT -k 60 "$TIMEOUT_S" "$VENV/bin/python" -u lab/r65_local_sgd.py --rank "$RANK" "${extra[@]}" --wait-s 600 -- \
  --dataset "$HOME/r65/data/tool-calling-fixture" --rows "$HOME/r65/data/rows.jsonl" --model "$HOME/models/Qwen3.5-4B" \
  --base-repo Qwen/Qwen3.5-4B --base-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out "$OUT" --run-id "r65-$RUN-rank$RANK" \
  --epochs 3 --lr 2e-4 --batch 4 --grad-accum 4 --rank 16 --alpha 32 --dropout 0 --seed 42) > "$OUT/train.log" 2>&1
rc=$?
cp "$HOME/r65/state/ledger.jsonl" "$OUT/ledger.jsonl" 2>/dev/null
rm -rf "$OUT/trainer"
echo "### R65 rank $RANK end $(date -u +%FT%TZ) rc=$rc wall_s=$(( $(date +%s) - t0 ))"
grep -E "^### local-sgd|^\{'loss'|train_runtime|^\{\"ok\"|Error|error" "$OUT/train.log" | cut -c1-400 | tail -40
ls "$OUT" "$OUT/adapter" 2>&1 | tr '\n' ' '
echo
echo "### END"
