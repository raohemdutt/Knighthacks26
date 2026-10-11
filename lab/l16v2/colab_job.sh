#!/bin/bash
# The L16 retrain on the Colab A100 (a copy of work/colab-a100/eval_job.sh's shape): clone the branch from its git
# bundle, build tool-calling-v2 on the VM, sweep, choose by validation, then score the held-out side in bf16.
# Runs detached (launched with setsid nohup). Every stage writes a marker under /content/l16v2/markers when its
# product and the product's .sha256 are whole; a failed stage writes FAILED_<stage> and the job goes on where it can.
#   usage: bash colab_job.sh <branch> [sweep budget seconds] [kernels: torch|fla] [sweep.py flags ...]
# kernels says which Gated DeltaNet and causal conv path the model takes: transformers' torch fallback (what the pod
# has) or flash-linear-attention and causal-conv1d (installed or removed here to match). The sweep.py flags carry the
# A100's split of the effective batch and --no-gradient-checkpointing; every run records all of it in its row.
set -x
BRANCH="${1:?branch}"; BUDGET="${2:-2400}"; KERNELS="${3:-torch}"
shift 3 2>/dev/null || shift $#
SWEEP_FLAGS=("$@")
OUT=/content/l16v2
MODEL=/content/models/Qwen3.5-4B
REPO=/content/Knighthacks26
mkdir -p $OUT/markers $OUT/products
stamp() { echo "$1 $(date -u +%FT%TZ)" >> $OUT/times.txt; }
status() { echo "stage=$1 elapsed_s=$SECONDS" > $OUT/status.txt; }
product() { # product <file in products/> <marker>: the .sha256 first, then the marker
  ( cd $OUT/products && echo "$(sha256sum "$1" | cut -d' ' -f1) $(stat -c %s "$1") $1" > "$1.sha256" ) && echo "$1" > "$OUT/markers/$2"
}
stamp job_start; status clone
# Colab ships torchao 0.10.0; PEFT 0.21.2 refuses any torchao below 0.16 the moment it puts LoRA on a model, and the
# trainer uses none. Taken out here as well as in colab_setup.sh, so a session set up before this line still runs.
echo "torchao: $(pip uninstall -y torchao 2>&1 | tail -n 1)" >> $OUT/versions.txt; python3 -c "import importlib.util as u; print('torchao', 'present' if u.find_spec('torchao') else 'absent')" >> $OUT/versions.txt
rm -rf $REPO
git clone -q -b "$BRANCH" /content/l16v2.bundle $REPO || { echo clone > $OUT/markers/FAILED_clone; exit 1; }
git -C $REPO log --format='repo %H %s' -1 >> $OUT/versions.txt
cd $REPO
export LAB_STATE_DIR=$OUT/state LAB_TASK=L16 LAB_ACTOR="Ilan1 agent" LAB_TIER=opus PYTHONIOENCODING=utf-8
mkdir -p $LAB_STATE_DIR

if [ "$KERNELS" = fla ]; then
  ( timeout 420 pip install flash-linear-attention causal-conv1d 2>&1 | tail -n 1 ) >> $OUT/versions.txt
else
  pip uninstall -y flash-linear-attention fla-core causal-conv1d > /dev/null 2>&1
fi
python3 -c "
import importlib.metadata as m
for d in ('flash-linear-attention', 'fla-core', 'causal-conv1d'):
    try: print('kernel package', d, m.version(d))
    except Exception: print('kernel package', d, 'not installed')
print('kernels asked:', '$KERNELS', '| sweep flags:', '${SWEEP_FLAGS[*]}')" >> $OUT/versions.txt

stamp dataset_start; status dataset
DS=$OUT/data/tool-calling-v2
REUSED=no
if [ -e $OUT/markers/DATASET_DONE ] && [ -s $DS/build.json ] && [ -s $OUT/products/dataset-tool-calling-v2.tar.gz ]; then
  # A dataset an earlier launch built on this VM is reused, same frozen split and same tarball, but only when this
  # commit builds the same seven files byte for byte (the build without the tokenizer check: the bytes do not depend on it).
  rm -rf $OUT/data-check
  python3 lab/fixture_dataset_v2.py --out $OUT/data-check > $OUT/dataset-check.json 2>> $OUT/dataset.err
  python3 -c "import json,sys; a,b=(json.load(open(p))['sha256'] for p in sys.argv[1:3]); sys.exit(0 if a==b else 1)" \
    $DS/build.json $OUT/data-check/tool-calling-v2/build.json && REUSED=yes
  rm -rf $OUT/data-check
fi
if [ $REUSED = no ]; then
  rm -rf $OUT/data $OUT/markers/DATASET_DONE
  python3 lab/fixture_dataset_v2.py --out $OUT/data --tokenizer $MODEL > $OUT/dataset.json 2> $OUT/dataset.err
  if [ $? -ne 0 ] || [ ! -s $DS/rows-train.jsonl ]; then
    tail -n 20 $OUT/dataset.err; echo dataset > $OUT/markers/FAILED_dataset; stamp dataset_failed; exit 1
  fi
  python3 lab/dataset.py $DS --text-only --split > $OUT/split_check.json 2>> $OUT/dataset.err
  cp $OUT/dataset.json $OUT/split_check.json $DS/ 2>/dev/null
  ( cd $OUT/data && tar -czf $OUT/products/dataset-tool-calling-v2.tar.gz tool-calling-v2 )
  product dataset-tool-calling-v2.tar.gz DATASET_DONE
fi
echo "dataset reused=$REUSED tarball $(cat $OUT/products/dataset-tool-calling-v2.tar.gz.sha256)" >> $OUT/versions.txt
echo "dataset reused=$REUSED sha256_rows_train=$(sha256sum $DS/rows-train.jsonl | cut -d' ' -f1) sha256_split=$(sha256sum $DS/split.json | cut -d' ' -f1)" >> $OUT/finished.txt
stamp "dataset_done_reused_$REUSED"

if [ -e $OUT/markers/SMOKE_DONE ]; then
  stamp smoke_skipped_passed_earlier_on_this_vm
else
  stamp smoke_start; status smoke
  python3 lab/l16v2/sweep.py --dataset $OUT/data/tool-calling-v2 --model $MODEL --out $OUT --only smoke "${SWEEP_FLAGS[@]}" > $OUT/smoke.log 2>&1
  if [ $? -ne 0 ]; then
    ( cd $OUT && tar -czf products/failed-smoke.tar.gz smoke.log runs/smoke/train.log runs/smoke/stdout.json 2>/dev/null )
    product failed-smoke.tar.gz FAILED_smoke; stamp smoke_failed; exit 1
  fi
  stamp smoke_done; echo smoke > $OUT/markers/SMOKE_DONE
fi

stamp sweep_start
python3 lab/l16v2/sweep.py --dataset $OUT/data/tool-calling-v2 --model $MODEL --out $OUT --budget-s "$BUDGET" "${SWEEP_FLAGS[@]}" > $OUT/sweep.log 2>&1
RC=$?
stamp "sweep_script_end_rc_$RC"
status bundle
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader >> $OUT/versions.txt
( cd $OUT && tar -czf products/final-bundle.tar.gz --exclude='runs/*/adapter' --exclude='runs/*/trainer' --exclude='products' --exclude='data' \
    times.txt versions.txt status.txt finished.txt choice_rule.txt dataset.json split_check.json sweep.json chosen.json \
    smoke.log sweep.log state runs validation-agreement held-out-bf16 2>/dev/null )
product final-bundle.tar.gz BUNDLE_DONE
if [ $RC -ne 0 ] || [ ! -e $OUT/markers/BEST_DONE ]; then echo sweep > $OUT/markers/FAILED_sweep; fi
stamp job_end
echo "stage=done elapsed_s=$SECONDS" > $OUT/status.txt
echo JOB_DONE > $OUT/markers/JOB_DONE
