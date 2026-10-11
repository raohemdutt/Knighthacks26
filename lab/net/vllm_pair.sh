#!/bin/bash
# R66 vLLM across the two pods, inside a GPU slot. One instance per pod.
#   vllm_pair.sh RANK MODE          RANK is the node rank (0 = head with the API, 1 = --headless); MODE pp|tp; then KEY=VALUE overrides (MASTER, MODEL, GPU_UTIL, PP_PARTITION, TEXT_ONLY, TOOL_PARSER, MPORT)
# Uses the same build, model and tool flags as the teacher, a SECOND port 8010, through the shim + chisel tunnel.
# The node-0 process is the head (serves the OpenAI API on 8010); node-1 runs --headless.
# Never stops the teacher on 8000. The shim .so and shim.env sit next to this script.
H="$(cd "$(dirname "$0")" && pwd)"; W=~/r66; mkdir -p $W
[ -f "$H/shim.so" ] && cp -f "$H/shim.so" $W/shim.so
[ -f "$H/shim.env" ] && cp -f "$H/shim.env" $W/shim.env
cp -f "$H/vllm_upgrade.sh" "$H/pod_a_teacher.sh" $W/ 2>/dev/null
RANK="$1"; MODE="${2:-pp}"
for kv in "${@:3}"; do export "$kv"; done   # KEY=VALUE overrides: MASTER, GPU_UTIL, PP_PARTITION, MPORT
VENV=~/venv-vllm-new
MODEL=${MODEL:-~/models/Qwen3.8-27B-INT4}
PORT=8010
MASTER=${MASTER:-10.99.0.1}; MPORT=${MPORT:-29727}   # top of the pinned range: port-0 binds fill it from 29600 up

set -a; [ -f $W/shim.env ] && . $W/shim.env; set +a
# R66_SELF comes from shim.env (10.99.0.1 on pod C, 10.99.0.2 on pod D)
# the shim gives the engine the pods0 interface; vLLM advertises VLLM_HOST_IP on it
export VLLM_HOST_IP=$R66_SELF
export GLOO_SOCKET_IFNAME=pods0 NCCL_SOCKET_IFNAME=pods0
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export NCCL_NSOCKS_PERTHREAD=1 NCCL_SOCKET_NTHREADS=1 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1
export NCCL_DEBUG=WARN
# pod C keeps the teacher's memory (about 6 GB free): give it the few layers named here, e.g. 60,4
[ -n "$PP_PARTITION" ] && export VLLM_PP_LAYER_PARTITION=$PP_PARTITION
# the pods' /dev/shm is 64 MB; vLLM's message queue wants 10 chunks of 16 MB (160 MB) by default
export VLLM_MQ_MAX_CHUNK_BYTES_MB=${VLLM_MQ_MAX_CHUNK_BYTES_MB:-2}
unset LD_LIBRARY_PATH
# the serve env from vllm_upgrade.sh (ROCm 7.14 paths), without re-exporting LD_LIBRARY_PATH
ROCM="$($VENV/bin/rocm-sdk path --root 2>/dev/null)"
[ -n "$ROCM" ] && export ROCM_PATH="$ROCM" HIP_PATH="$ROCM" PATH="$VENV/bin:$ROCM/bin:$HOME/.local/bin:$PATH"

# TEXT_ONLY=1: no rank loads the vision tower (pod C's stage cannot fit it beside the teacher)
MODEL=$(eval echo $MODEL)
if [ "${TEXT_ONLY:-0}" = 1 ]; then
  if grep -q '"--language-model-only"' ~/vllm-new/vllm/engine/arg_utils.py 2>/dev/null; then MMFLAG="--language-model-only"
  else MMFLAG="--limit-mm-per-prompt {\"image\":0,\"video\":0}"; fi
elif [ "${TEXT_ONLY:-0}" = model ]; then MMFLAG=""   # a text-only checkpoint: no multimodal flag at all
else MMFLAG="--limit-mm-per-prompt {\"image\":1}"; fi
echo "multimodal flag: $MMFLAG"
if [ "$MODE" = tp ]; then PP=1; TP=2; else PP=2; TP=1; fi

# multi-node PP/TP across the two pods: --nnodes 2, --node-rank 0|1, --master-addr/--master-port on the fake iface,
# node 1 --headless (no API server). Same model, tool flags and context as the teacher; a second port on node 0.
common=("$MODEL" --served-model-name group-brain --dtype bfloat16
        --enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER:-qwen3_coder}
        --max-model-len 16384 $MMFLAG
        --gpu-memory-utilization ${GPU_UTIL:-0.33} ${MMFLAG:+--mm-encoder-attn-backend TRITON_ATTN}
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
[ "$RANK" = 1 ] && N_WAIT=9 || N_WAIT=120
for i in $(seq 1 $N_WAIT); do
  ps -o stat= -p "$(cat $W/vllm-r66.pid)" 2>/dev/null | grep -qv Z || { echo "process ended early"; tail -40 "$LOG"; exit 1; }
  if [ "$RANK" = 0 ] && curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
    echo "group-brain up on $PORT after $((i*10)) s"; break
  fi
  sleep 10
done
grep -iE "Using .* for CompressedTensors|KV cache|pipeline|Loading weights took|World size|rank|init engine|EngineCore" "$LOG" | tail -20 | cut -c1-200
echo "### END"
