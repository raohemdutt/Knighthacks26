#!/bin/bash
# R66 gate on one pod: r66_gate.sh RANK [gloo|nccl]
# Uses the shim .so and its env next to this script (shim.so, shim.env), copied to ~/r66 first.
H="$(cd "$(dirname "$0")" && pwd)"
W=~/r66; mkdir -p $W
[ -f "$H/shim.so" ] && cp -f "$H/shim.so" $W/shim.so
[ -f "$H/shim.env" ] && cp -f "$H/shim.env" $W/shim.env
cp -f "$H/r66_gloo.py" $W/
RANK="$1"; BACKEND="${2:-gloo}"
SELF=10.99.0.$((RANK + 1))
set -a; [ -f $W/shim.env ] && . $W/shim.env; set +a
export R66_SELF=${R66_SELF:-$SELF}
export GLOO_SOCKET_IFNAME=pods0 NCCL_SOCKET_IFNAME=pods0 NCCL_IB_DISABLE=1
unset LD_LIBRARY_PATH
PY=~/venv-vllm-new/bin/python
echo "rank $RANK self $SELF backend $BACKEND shim $(sha256sum $W/shim.so 2>/dev/null | cut -c1-12)"
pgrep -f "chisel (client|server)" >/dev/null && echo "tunnel process up" || echo "NO TUNNEL PROCESS on this pod"
LD_PRELOAD=$W/shim.so timeout 240 $PY $W/r66_gloo.py "$RANK" 10.99.0.1 29600 "$BACKEND" 2>&1 | tail -25
echo "### END"
