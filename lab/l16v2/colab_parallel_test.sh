#!/bin/bash
# TEST A on a Colab worker: do several trainers on one card add throughput? 6 optimizer steps of the search trials'
# configuration with 1, then 2, then 3 (then more, if asked) trainers at the same time, each its own folder and seed.
# One line per case in /content/l16v2/parallel.txt: seconds per step per process (first step left out), total
# examples per second across processes, peak memory on the card. PARALLEL_DONE marks the end.
#   usage: colab_parallel_test.sh [counts, default "1 2 3"]
set -u
OUT=/content/l16v2; MODEL=/content/models/Qwen3.5-4B; REPO=/content/Knighthacks26; DS=$OUT/data/tool-calling-v2
COUNTS="${1:-1 2 3}"
rm -f $OUT/markers/PARALLEL_DONE; : > $OUT/parallel.txt
cd $REPO
for N in $COUNTS; do
  echo "stage=parallel_test trainers=$N elapsed_s=$SECONDS" > $OUT/status.txt
  ( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done > $OUT/par-mem-$N.txt ) & MON=$!
  PIDS=""
  for K in $(seq 1 $N); do
    rm -rf $OUT/par-$N-$K
    LAB_STATE_DIR=$OUT/state-par-$K python3 lab/train.py --cuda --no-install --dataset $DS --rows $DS/rows-train.jsonl \
      --model $MODEL --base-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out $OUT/par-$N-$K --run-id par-$N-$K \
      --max-steps 7 --lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 --warmup-ratio 0.1 --batch 4 --grad-accum 4 \
      --seed $((41 + K)) > $OUT/par-$N-$K.json 2> $OUT/par-$N-$K.log &
    PIDS="$PIDS $!"
  done
  wait $PIDS
  kill $MON 2>/dev/null
  python3 - "$N" >> $OUT/parallel.txt <<'PY'
import json, os, sys
n = int(sys.argv[1]); out = "/content/l16v2"
per, outcomes = [], []
for k in range(1, n + 1):
    try:
        a = json.loads(open(f"{out}/par-{n}-{k}.json").read().strip().splitlines()[-1])
    except Exception:
        a = {}
    outcomes.append(str(a.get("outcome")))
    t = [json.loads(l)["step_ms"] for l in open(f"{out}/par-{n}-{k}/telemetry.jsonl")] if os.path.exists(f"{out}/par-{n}-{k}/telemetry.jsonl") else []
    if len(t) > 1:
        per.append(sum(t[1:]) / len(t[1:]) / 1000.0)
mem = max([int(x) for x in open(f"{out}/par-mem-{n}.txt").read().split() if x.isdigit()] or [0])
total = sum(16.0 / s for s in per)
print(f"parallel trainers={n} outcomes={'/'.join(outcomes)} step_s_per_process={'/'.join(f'{s:.2f}' for s in per)} "
      f"total_examples_per_s={total:.2f} peak_card_mem_mib={mem}")
PY
  rm -rf $OUT/par-$N-*/adapter $OUT/par-$N-*/trainer
done
echo "stage=parallel_test_done elapsed_s=$SECONDS" > $OUT/status.txt
echo parallel > $OUT/markers/PARALLEL_DONE
