#!/bin/bash
# R66 vLLM across the two pods, inside a GPU slot. One instance per pod.
#   r66_vllm.sh RANK MODE          RANK 0 on pod C (10.99.0.1), RANK 1 on pod D (10.99.0.2); MODE pp|tp
# Uses the same build, model and tool flags as the teacher, a SECOND port 8010, through the shim + chisel tunnel.
# The node-0 process is the head (serves the OpenAI API on 8010); node-1 runs --headless.
# Never stops the teacher on 8000. The shim .so and shim.env sit next to this script.
H="$(cd "$(dirname "$0")" && pwd)"; W=~/r66; mkdir -p $W
[ -f "$H/shim.so" ] && cp -f "$H/shim.so" $W/shim.so
[ -f "$H/shim.env" ] && cp -f "$H/shim.env" $W/shim.env
cp -f "$H/vllm_upgrade.sh" "$H/pod_a_teacher.sh" $W/ 2>/dev/null
RANK="$1"; MODE="${2:-pp}"
VENV=~/venv-vllm-new
MODEL=~/models/Qwen3.8-27B-INT4
PORT=8010
MASTER=10.99.0.1; MPORT=29600

set -a; [ -f $W/shim.env ] && . $W/shim.env; set +a
export R66_SELF=10.99.0.$((RANK + 1))
# the shim gives the engine the pods0 interface; vLLM advertises VLLM_HOST_IP on it
export VLLM_HOST_IP=$R66_SELF
export GLOO_SOCKET_IFNAME=pods0 NCCL_SOCKET_IFNAME=pods0
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export NCCL_NSOCKS_PERTHREAD=1 NCCL_SOCKET_NTHREADS=1 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1
export NCCL_DEBUG=WARN
unset LD_LIBRARY_PATH
# the serve env from vllm_upgrade.sh (ROCm 7.14 paths), without re-exporting LD_LIBRARY_PATH
ROCM="$($VENV/bin/rocm-sdk path --root 2>/dev/null)"
[ -n "$ROCM" ] && export ROCM_PATH="$ROCM" HIP_PATH="$ROCM" PATH="$VENV/bin:$ROCM/bin:$HOME/.local/bin:$PATH"

if [ "$MODE" = tp ]; then PP=1; TP=2; else PP=2; TP=1; fi

# multi-node PP/TP across the two pods: --nnodes 2, --node-rank 0|1, --master-addr/--master-port on the fake iface,
# node 1 --headless (no API server). Same model, tool flags and context as the teacher; a second port on node 0.
common=("$MODEL" --served-model-name group-brain --dtype bfloat16
        --enable-auto-tool-choice --tool-call-parser qwen3_coder
        --max-model-len 16384 --limit-mm-per-prompt '{"image":1}'
        --gpu-memory-utilization 0.45 --mm-encoder-attn-backend TRITON_ATTN
        --distributed-executor-backend mp
        --pipeline-parallel-size "$PP" --tensor-parallel-size "$TP"
        --nnodes 2 --node-rank "$RANK" --master-addr "$MASTER" --master-port "$MPORT")

pgrep -f "vllm.entrypoints.*8000" >/dev/null && echo "teacher on 8000 present (good, not touching it)"
echo "rank $RANK mode $MODE pp $PP tp $TP self $R66_SELF shim $(sha256sum $W/shim.so 2>/dev/null | cut -c1-12)"
pgrep -f "chisel (client|server)" >/dev/null || echo "WARNING: no chisel process on this pod"

LOG=$W/vllm-r66-rank$RANK.log
if [ "$RANK" = 0 ]; then
  LD_PRELOAD=$W/shim.so nohup "$VENV/bin/vllm" serve \
    "${common[@]}" --host 0.0.0.0 --port "$PORT" \
    > "$LOG" 2>&1 < /dev/null &
else
  LD_PRELOAD=$W/shim.so nohup "$VENV/bin/vllm" serve \
    "${common[@]}" --headless \
    > "$LOG" 2>&1 < /dev/null &
fi
echo $! > $W/vllm-r66.pid
echo "started pid $(cat $W/vllm-r66.pid), log $LOG"
# tail until it serves or dies (rank 0 only answers the API)
for i in $(seq 1 120); do
  kill -0 "$(cat $W/vllm-r66.pid)" 2>/dev/null || { echo "process ended early"; tail -40 "$LOG"; exit 1; }
  if [ "$RANK" = 0 ] && curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
    echo "group-brain up on $PORT after $((i*10)) s"; break
  fi
  sleep 10
done
grep -iE "Using .* for CompressedTensors|KV cache|pipeline|Loading weights took|World size|rank|init engine|EngineCore" "$LOG" | tail -20 | cut -c1-200
echo "### END"
