#!/bin/bash
# R55 2-bit: wait for the 3-bit eval to finish, then quantize (AutoRound W2A16), repack to GPTQ, and evaluate.
set -uo pipefail
R=/content/r55; REPO=/content/Knighthacks26; Q=/content/quant
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/times.txt; }
while ! grep -q "EVAL_DONE w3" $R/times.txt; do sleep 20; done
NAME=$(python3 -c "import sys;sys.path.insert(0,'$REPO/lab');import quantize_teacher as q;print(q.checkpoint_name(2))")
cd $REPO
python3 -c "import torch,llmcompressor,compressed_tensors,auto_round;print(torch.__version__,llmcompressor.__version__,auto_round.__version__)"
stamp "start quant w2"
python3 lab/quantize_teacher.py --width 2 --calibration fixtures/lab/quant/calibration-mixed-512.jsonl --out $Q || { stamp "FAILED quant w2"; exit 1; }
cp $Q/$NAME/atlas_quant.json $R/atlas_quant_w2.json; stamp "done quant w2"
python3 lab/ct_to_gptq.py --src $Q/$NAME --dst $Q/$NAME-gptq || { stamp "FAILED convert w2"; exit 1; }
stamp "done convert w2"
bash /content/r55_eval.sh 2
