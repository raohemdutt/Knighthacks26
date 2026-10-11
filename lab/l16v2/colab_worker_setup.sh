#!/bin/bash
# Bring one Colab session to a commit of the branch with the frozen dataset, so it can run trials for the search
# that is driven from the workstation (lab/l16v2/bo_local.py). Safe to run again: the base model and the pinned
# stack are set up once (lab/l16v2/colab_setup.sh), the clone is replaced, the dataset is built only if it is
# missing. The marker /content/l16v2/WORKER_READY holds the commit and the sha256 of the rows and of the split,
# which the workstation checks against the frozen dataset's.
#   usage: colab_worker_setup.sh <branch>
set -x
BRANCH="${1:?branch}"
OUT=/content/l16v2; MODEL=/content/models/Qwen3.5-4B; REPO=/content/Knighthacks26; DS=$OUT/data/tool-calling-v2
mkdir -p $OUT/markers $OUT/products $OUT/runs $OUT/scores
rm -f $OUT/WORKER_READY
if [ ! -e $OUT/SETUP_DONE ] || [ ! -e $MODEL/config.json ]; then
  bash /content/colab_setup.sh > /content/colab_setup.log 2>&1
fi
[ -e $MODEL/config.json ] || { echo "WORKER_FAILED no base model" > $OUT/WORKER_READY; exit 1; }
pip uninstall -y torchao > /dev/null 2>&1
pip uninstall -y flash-linear-attention fla-core causal-conv1d > /dev/null 2>&1
rm -rf $REPO
git clone -q -b "$BRANCH" /content/l16v2.bundle $REPO || { echo "WORKER_FAILED clone" > $OUT/WORKER_READY; exit 1; }
cd $REPO
export LAB_STATE_DIR=$OUT/state LAB_TASK=L16 LAB_ACTOR="Ilan1 agent" LAB_TIER=opus PYTHONIOENCODING=utf-8
if [ ! -s $DS/rows-train.jsonl ]; then
  rm -rf $OUT/data
  python3 lab/fixture_dataset_v2.py --out $OUT/data --tokenizer $MODEL > $OUT/dataset.json 2> $OUT/dataset.err \
    || { echo "WORKER_FAILED dataset $(tail -n 1 $OUT/dataset.err | cut -c1-120)" > $OUT/WORKER_READY; exit 1; }
fi
python3 -c "import torch, peft, transformers, trl; print('stack', torch.__version__, peft.__version__, transformers.__version__, trl.__version__, torch.cuda.get_device_name(0))" > $OUT/worker_stack.txt 2>&1
echo "$(git rev-parse HEAD) $(sha256sum $DS/rows-train.jsonl | cut -d' ' -f1) $(sha256sum $DS/split.json | cut -d' ' -f1) $(tail -n 1 $OUT/worker_stack.txt | tr ' ' '_')" > $OUT/WORKER_READY
