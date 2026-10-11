#!/bin/bash
# L16 retrain on the Colab A100: time a few optimizer steps each way before the sweep, so the sweep runs on the fast
# path and the difference from the pod is written down. Runs detached; SPEED_DONE marks the end.
#   timing rows go to /content/l16v2/speed.txt: label, outcome, step_ms of each step, examples/s of steps 2..n, peak GB
set -x
OUT=/content/l16v2; MODEL=/content/models/Qwen3.5-4B; REPO=/content/Knighthacks26; DS=$OUT/data/tool-calling-v2
pkill -f '[c]olab_job.sh'; pkill -f '[l]16v2/sweep.py'; pkill -f '[l]ab/train.py'; sleep 3
rm -f $OUT/markers/SPEED_DONE; : > $OUT/speed.txt
cd $REPO
export LAB_STATE_DIR=$OUT/state-speed LAB_TASK=L16 LAB_ACTOR="Ilan1 agent" LAB_TIER=opus
timeit() { # timeit <label> <trainer flags...>
  local label="$1"; shift
  echo "stage=speed run=$label elapsed_s=$SECONDS" > $OUT/status.txt
  rm -rf $OUT/speed-$label
  python3 lab/train.py --cuda --no-install --dataset $DS --rows $DS/rows-train.jsonl --model $MODEL \
    --base-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out $OUT/speed-$label --run-id speed-$label \
    --max-steps 5 --lr 1e-4 --rank 16 --alpha 32 --dropout 0.05 --seed 42 "$@" > $OUT/speed-$label.json 2> $OUT/speed-$label.log
  python3 - "$label" $OUT/speed-$label $OUT/speed-$label.json >> $OUT/speed.txt <<'PY'
import json, sys, os
label, out, answer = sys.argv[1:4]
try:
    a = json.loads(open(answer).read().strip().splitlines()[-1])
except Exception:
    a = {}
t = [json.loads(l) for l in open(out + "/telemetry.jsonl")] if os.path.exists(out + "/telemetry.jsonl") else []
ms = [round(e["step_ms"]) for e in t]
m = a.get("metrics", {})
per = m.get("trainer.batch", 0) * m.get("trainer.grad_accum", 0)
rate = per * 1000 * len(ms[1:]) / sum(ms[1:]) if len(ms) > 1 and per else 0
s = json.loads(a.get("refs", {}).get("settings", "{}") or "{}")
print(f"speed label={label} outcome={a.get('outcome')} step_ms={'/'.join(map(str, ms))} examples_per_s={rate:.2f} "
      f"peak_mem_gb={m.get('trainer.peak_mem_gb', 0):.1f} gdn_kernel={s.get('gdn_kernel')} conv_kernel={s.get('causal_conv1d_kernel')} "
      f"loss_head={a.get('refs', {}).get('loss_head')} error={str(a.get('error'))[:140]}")
PY
  rm -rf $OUT/speed-$label/adapter $OUT/speed-$label/trainer
}
timeit torch-b4x4-ckpt --batch 4 --grad-accum 4
timeit torch-b16x1-nockpt --batch 16 --grad-accum 1 --no-gradient-checkpointing
timeit torch-b16x1-ckpt --batch 16 --grad-accum 1
echo "stage=speed run=pip-install elapsed_s=$SECONDS" > $OUT/status.txt
( timeout 420 pip install flash-linear-attention 2>&1 | tail -n 3 ) > $OUT/speed-pip-fla.log 2>&1
echo "pip fla: $(tail -n 1 $OUT/speed-pip-fla.log | cut -c1-200)" >> $OUT/speed.txt
( timeout 420 pip install causal-conv1d 2>&1 | tail -n 3 ) > $OUT/speed-pip-conv.log 2>&1
echo "pip causal-conv1d: $(tail -n 1 $OUT/speed-pip-conv.log | cut -c1-200)" >> $OUT/speed.txt
python3 -c "
import importlib.util as u, importlib.metadata as m
for n, d in (('fla', 'flash-linear-attention'), ('causal_conv1d', 'causal-conv1d'), ('torchao', 'torchao')):
    try: v = m.version(d)
    except Exception: v = 'not installed'
    print('kernel package', d, v, 'importable' if u.find_spec(n) else 'not importable')
import torch, peft, transformers
print('still ok torch', torch.__version__, 'peft', peft.__version__, 'transformers', transformers.__version__)
" >> $OUT/speed.txt 2>&1
timeit fla-b16x1-nockpt --batch 16 --grad-accum 1 --no-gradient-checkpointing
timeit fla-b16x1-ckpt --batch 16 --grad-accum 1
grep -a -i -E "falling back|fallback" $OUT/speed-fla-b16x1-nockpt.log | cut -c1-200 | sort -u >> $OUT/speed.txt
echo "stage=speed_done elapsed_s=$SECONDS" > $OUT/status.txt
echo speed > $OUT/markers/SPEED_DONE
