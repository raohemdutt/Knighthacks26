#!/bin/bash
# R57: R21's correctness record on the A100 (one more spawn of the same stock vLLM serve line).
set -x
rm -rf /content/kh2 && mkdir -p /content/kh2 && tar -xzf /content/repo2.tar.gz -C /content/kh2
cd /content/kh2
OUT=/content/r57/correctness; mkdir -p $OUT
python3 -m vllm.entrypoints.openai.api_server --model /content/models/Qwen3.8-27B-INT4 --served-model-name teacher \
    --port 8100 --host 127.0.0.1 --max-model-len 16384 --limit-mm-per-prompt '{"image":1}' \
    --gpu-memory-utilization 0.90 --dtype bfloat16 > $OUT/serve.log 2>&1 &
PID=$!
for i in $(seq 1 450); do curl -sf http://127.0.0.1:8100/v1/models > /dev/null && break; sleep 2; done
python3 -m lab correctness record --url http://127.0.0.1:8100 --engine vllm --engine-commit pypi-wheel-0.31.0 \
    --checkpoint RedHatAI/Qwen3.8-27B-INT4 --quantization int4 --node a100 --chip a100 --torch 2.13.0+cu130 \
    --triton 3.7.1 --set vllm_version=0.31.0 --set cuda=13.0 --set driver=580.82.07 \
    --set gpu=NVIDIA-A100-SXM4-40GB --set hub=colab --set dtype=bfloat16 --set enforce_eager=false \
    --set other_chip=true --out-dir $OUT --date 2026-10-10 > $OUT/record_result.json 2> $OUT/record_stderr.log
echo "record exit $?" >> $OUT/record_stderr.log
kill -TERM -- -$PID 2>/dev/null; kill $PID; sleep 15; pkill -f vllm.entrypoints; sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $OUT/record_stderr.log
echo CORRECT_DONE > $OUT/CORRECT_DONE
