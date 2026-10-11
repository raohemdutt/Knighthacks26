#!/bin/bash
# R55 evaluation after quantize + convert: perplexities under vLLM 0.31 (system) and 0.19.1 (venv, the pods' source).
# Usage: bash r55_eval.sh <width>
set -uo pipefail
W=${1:-3}; R=/content/r55; REPO=/content/Knighthacks26; Q=/content/quant
NAME=$(python3 -c "import sys;sys.path.insert(0,'$REPO/lab');import quantize_teacher as q;print(q.checkpoint_name($W))")
V19=/content/v19/bin/python
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/times.txt; }
cd $REPO
if [ ! -x $V19 ]; then
  ( uv venv -q --python python3 /content/v19 && uv pip install -q --python $V19 "vllm==0.19.1" datasets \
    && $V19 -c "import vllm,torch;print('v19', vllm.__version__, torch.__version__)" ) > $R/v19_install.log 2>&1 &
  INST=$!
else INST=""; fi
stamp "start eval w$W"
[ -f $R/ppl-int4-v031.json ] || python3 lab/lowbit_ppl.py --model /content/models/Qwen3.8-27B-INT4 --label int4-v031 --out $R/ppl-int4-v031.json > $R/ppl-int4-v031.log 2>&1
stamp "int4-v031 rc=$?"
python3 lab/lowbit_ppl.py --model $Q/$NAME --label w$W-ct-v031 --out $R/ppl-w$W-ct-v031.json > $R/ppl-w$W-ct-v031.log 2>&1
stamp "w$W-ct-v031 rc=$?"
[ -n "$INST" ] && wait $INST; tail -2 $R/v19_install.log
$V19 /content/kcheck.py $Q/$NAME-gptq $W > $R/kcheck-w$W.log 2>&1; stamp "kcheck w$W rc=$?"; tail -8 $R/kcheck-w$W.log
$V19 lab/lowbit_ppl.py --model $Q/$NAME-gptq --dtype float16 --label w$W-gptq-v019 --out $R/ppl-w$W-gptq-v019.json > $R/ppl-w$W-gptq-v019.log 2>&1
stamp "w$W-gptq-v019 rc=$?"
[ -f $R/ppl-int4-v019.json ] || $V19 lab/lowbit_ppl.py --model /content/models/Qwen3.8-27B-INT4 --label int4-v019 --out $R/ppl-int4-v019.json > $R/ppl-int4-v019.log 2>&1
stamp "int4-v019 rc=$?"
[ -f $R/ppl-int4-fp16-v019.json ] || $V19 lab/lowbit_ppl.py --model /content/models/Qwen3.8-27B-INT4 --dtype float16 --label int4-fp16-v019 --out $R/ppl-int4-fp16-v019.json > $R/ppl-int4-fp16-v019.log 2>&1
stamp "int4-fp16-v019 rc=$?"
for f in $R/ppl-*.json; do cat $f; done
stamp "EVAL_DONE w$W"
