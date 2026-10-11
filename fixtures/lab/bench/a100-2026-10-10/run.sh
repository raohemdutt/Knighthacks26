#!/bin/bash
# R57 job 1 on the Colab A100: the R11 harness (python -m lab bench run) against stock vLLM, three cold spawns per
# batch size, with nvidia-smi power.draw sampled once a second beside it; then the greedy outputs of the prompt set.
set -x
OUT=/content/r57
mkdir -p $OUT
cd /content/Knighthacks26
date -u +%FT%TZ > $OUT/start_utc.txt
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu --format=csv -l 1 > $OUT/nvidia_smi_power.csv 2>&1 &
SMI=$!
# The GPU idle with no server: 60 s before the first spawn.
echo "idle_start $(date -u +%FT%T.%3NZ)" > $OUT/idle_window.txt
sleep 60
echo "idle_end $(date -u +%FT%T.%3NZ)" >> $OUT/idle_window.txt
nvidia-smi -q > $OUT/nvidia_smi_q.txt 2>&1
python3 -m lab bench run --engine vllm --checkpoint /content/models/Qwen3.8-27B-INT4 --model teacher --max-len 16384 \
    --node a100 --repeats 3 --thinking off --hub colab --python python3 --out $OUT/bench \
    --metrics-file $OUT/metrics_from_harness.jsonl > $OUT/bench_result.json 2> $OUT/bench_stderr.log
echo "bench exit $?" >> $OUT/bench_stderr.log
date -u +%FT%TZ > $OUT/bench_end_utc.txt
python3 /content/greedy.py > $OUT/greedy_stdout.log 2>&1
echo "greedy exit $?" >> $OUT/greedy_stdout.log
sleep 5
kill $SMI
date -u +%FT%TZ > $OUT/end_utc.txt
echo RUN_DONE > $OUT/RUN_DONE
