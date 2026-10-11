#!/bin/bash
# R55 on the Colab A100: low-bit teacher weights. Usage: bash r55_run.sh <width> [first_stage]
# Stages: setup calib fetch quant convert vllm check ppl pack. Stamps go to /content/r55/times.txt.
set -uo pipefail
W=${1:-3}; FIRST=${2:-setup}
R=/content/r55; REPO=/content/Knighthacks26; Q=/content/quant
mkdir -p $R $Q
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
NAME=$(python3 -c "import sys;sys.path.insert(0,'$REPO/lab');import quantize_teacher as q;print(q.checkpoint_name($W))" 2>/dev/null)
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/times.txt; }
fail(){ stamp "FAILED $*"; touch $R/FAILED_$W; exit 1; }
go=0
run(){ [ "$1" = "$FIRST" ] && go=1; [ $go = 1 ] || return 0; stamp "start $1 w$W"; "st_$1" || fail "$1"; stamp "done $1 w$W"; }

st_setup(){
  nvidia-smi; df -h /content /root; free -g; nproc
  pip install -q uv || return 1
  # The pinned compressed-tensors commit reports a version below what the pinned llm-compressor asks for, so the
  # resolver refuses the pair: install llm-compressor first, then put the pinned compressed-tensors over it without deps.
  CT=$(grep '^compressed-tensors' $REPO/lab/requirements-quantize.txt); LC=$(grep '^llmcompressor' $REPO/lab/requirements-quantize.txt)
  uv pip install --system -q "$LC" "huggingface_hub>=1.0" datasets 2>&1 | tail -5
  uv pip install --system -q --no-deps "$CT" 2>&1 | tail -5
  python3 -c "import torch,transformers,llmcompressor,compressed_tensors;print(torch.__version__,torch.version.cuda,transformers.__version__,llmcompressor.__version__,compressed_tensors.__version__)"
}
st_calib(){
  cd $REPO && timeout 900 python3 fixtures/lab/quant/make_calibration_mixed.py; cp fixtures/lab/quant/calibration-mixed-512.jsonl $R/
}
st_fetch(){
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import time
t=time.time(); p=snapshot_download("Qwen/Qwen3.8-27B"); print("base", p, round(time.time()-t))
t=time.time(); p=snapshot_download("RedHatAI/Qwen3.8-27B-INT4", local_dir="/content/models/Qwen3.8-27B-INT4"); print("int4", p, round(time.time()-t))
PY
  df -h /content
}
st_quant(){
  cd $REPO && python3 lab/quantize_teacher.py --width $W --calibration fixtures/lab/quant/calibration-mixed-512.jsonl --out $Q
  cp $Q/$NAME/atlas_quant.json $R/atlas_quant_w$W.json
}
st_convert(){
  cd $REPO && python3 lab/ct_to_gptq.py --src $Q/$NAME --dst $Q/$NAME-gptq && du -sh $Q/$NAME $Q/$NAME-gptq
}
st_vllm(){
  python3 -c "import vllm" 2>/dev/null && { python3 -c "import vllm;print(vllm.__version__)"; return 0; }
  uv pip install --system -q vllm 2>&1 | tail -3; python3 -c "import vllm,torch;print(vllm.__version__,torch.__version__)"
}
st_check(){
  cd $REPO && python3 /content/kcheck.py $Q/$NAME-gptq $W
}
st_ppl(){
  cd $REPO
  [ -f $R/ppl-int4.json ] || python3 lab/lowbit_ppl.py --model /content/models/Qwen3.8-27B-INT4 --label int4 --out $R/ppl-int4.json || return 1
  [ -f $R/ppl-int4-fp16.json ] || python3 lab/lowbit_ppl.py --model /content/models/Qwen3.8-27B-INT4 --dtype float16 --label int4-fp16 --out $R/ppl-int4-fp16.json || true
  python3 lab/lowbit_ppl.py --model $Q/$NAME-gptq --dtype float16 --label w${W}a16-gptq --out $R/ppl-w$W.json || return 1
  python3 lab/lowbit_ppl.py --model $Q/$NAME --label w${W}a16-ct --out $R/ppl-w$W-ct.json > $R/ppl-w$W-ct.log 2>&1 || echo "compressed-tensors w$W does not load in this vllm (see ppl-w$W-ct.log)"
  tail -3 $R/ppl-w$W-ct.log
}
st_pack(){
  cd $Q && rm -rf parts-w$W && mkdir parts-w$W && tar cf - $NAME-gptq | split -b 1000M -d -a 3 - parts-w$W/$NAME-gptq.tar.
  (cd parts-w$W && sha256sum * > SHA256SUMS && ls -la)
  cp $R/times.txt $R/times-w$W.txt; tar czf /content/r55-w$W.tar.gz -C /content r55
}
for s in setup calib fetch quant convert vllm check ppl pack; do run $s; done
stamp "JOB_DONE w$W"; touch $R/JOB_DONE_$W
