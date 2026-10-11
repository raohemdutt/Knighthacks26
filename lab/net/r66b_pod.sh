#!/bin/bash
# R66b: prove lab/net/podshim.so on one pod, CPU only, two local processes (no GPU, no slot, never port 8000).
# Run through the controller with the sources uploaded next to it:
#   python -m lab.controller run-script --account C --timeout 600 lab/net/r66b_pod.sh STEP... \
#     --with lab/net/podshim.c --with lab/net/podshim_test.c --with lab/net/gloo_pair.py
# Steps: build  ctest  gloo  control  baseline  rccl  pytest  install <self> <peer>
# The local ranges (40000-40127 for 10.99.0.1, 41000-41127 for 10.99.0.2) stay clear of R66's tunnel (29600-29927).
set -u
HERE="$(pwd)"
DIR="$HOME/r66b"
mkdir -p "$DIR"
for f in podshim.c podshim_test.c gloo_pair.py test_lab_podshim.py; do [ -f "$HERE/$f" ] && cp "$HERE/$f" "$DIR/$f"; done
cd "$DIR"
PY=""
for p in "$HOME/venvs/atlas-train/bin/python" "$HOME/venv-vllm-new/bin/python"; do [ -x "$p" ] && { PY="$p"; break; }; done
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
now() { date +%s.%N; }
since() { awk -v a="$1" -v b="$(now)" 'BEGIN{printf "%.3f", b-a}'; }

rank_env() { # rank 0 is 10.99.0.1 on 40000-40127, rank 1 is 10.99.0.2 on 41000-41127; the "tunnel" is the identity
  if [ "$1" = 0 ]; then echo "PODSHIM_SELF=10.99.0.1 PODSHIM_PEER=10.99.0.2 PODSHIM_PORT_LOW=40000 PODSHIM_PORT_HIGH=40127 PODSHIM_PEER_PORT_LOW=41000 PODSHIM_PEER_PORT_BASE=41000"
  else echo "PODSHIM_SELF=10.99.0.2 PODSHIM_PEER=10.99.0.1 PODSHIM_PORT_LOW=41000 PODSHIM_PORT_HIGH=41127 PODSHIM_PEER_PORT_LOW=40000 PODSHIM_PEER_PORT_BASE=40000"; fi
}

pair() { # pair <label> <preload 0|1> <ifname> <master addr> <master port>
  local label="$1" pre="$2" ifn="$3" addr="$4" port="$5" t0 pids=() r
  rm -f "$DIR/$label".*.json "$DIR/$label".shim.log
  t0=$(now)
  for r in 0 1; do
    local extra=""
    [ "$pre" = 1 ] && extra="LD_PRELOAD=$DIR/podshim.so PODSHIM_LOG=$DIR/$label.shim.log $(rank_env $r)"
    env $extra GLOO_SOCKET_IFNAME="$ifn" MASTER_ADDR="$addr" MASTER_PORT="$port" \
      timeout 150 "$PY" gloo_pair.py "$r" --timeout 60 --out "$DIR/$label.$r.json" > "$DIR/$label.$r.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "--- $label: wall $(since "$t0") s"
  for r in 0 1; do
    if [ -s "$DIR/$label.$r.json" ]; then cat "$DIR/$label.$r.json"; else echo "rank $r: no result"; tail -5 "$DIR/$label.$r.log"; fi
  done
  if [ -f "$DIR/$label.shim.log" ]; then
    echo "shim trace: $(grep -c ' bind ' "$DIR/$label.shim.log") binds, $(grep -c ' connect ' "$DIR/$label.shim.log") connects"
    grep ' bind ' "$DIR/$label.shim.log" | head -4; grep ' connect ' "$DIR/$label.shim.log" | head -6
  fi
}

step() {
  case "$1" in
    build)
      echo "### STEP build"; gcc --version | head -1; ls /usr/include/ifaddrs.h
      t0=$(now)
      gcc -O2 -Wall -Wextra -shared -fPIC -o podshim.so podshim.c -ldl -lpthread && \
        gcc -O2 -Wall -o podshim_test podshim_test.c -lpthread && echo "build ok in $(since "$t0") s"
      nm -D --defined-only podshim.so | awk '{print $3}' | tr '\n' ' '; echo
      echo "listeners in 40000-42127 before the tests:"
      for f in /proc/net/tcp /proc/net/tcp6; do awk 'NR>1 && $4=="0A"{n=split($2,a,":"); print a[n]}' "$f"; done |
        while read -r h; do p=$((16#$h)); [ "$p" -ge 40000 ] && [ "$p" -le 42127 ] && echo "$p"; done | sort -u | tr '\n' ' '; echo
      ;;
    ctest)
      echo "### STEP ctest"
      t0=$(now)
      env LD_PRELOAD="$DIR/podshim.so" PODSHIM_LOG="$DIR/ctest.shim.log" PODSHIM_SELF=10.99.0.1 PODSHIM_PEER=10.99.0.2 \
        PODSHIM_PORT_LOW=40000 PODSHIM_PORT_HIGH=40127 PODSHIM_PEER_PORT_LOW=41000 PODSHIM_PEER_PORT_BASE=42000 ./podshim_test
      echo "ctest rc $? in $(since "$t0") s"
      echo "without the shim:"; env PODSHIM_SELF=10.99.0.1 PODSHIM_PEER=10.99.0.2 ./podshim_test 2>&1 | grep -E '^(ok|FAIL)' | head -3
      ;;
    gloo) echo "### STEP gloo"; echo "python $PY"; pair shim 1 pods0 10.99.0.1 40000 ;;
    control) echo "### STEP control"; pair control 0 pods0 10.99.0.1 40000 ;;
    baseline) echo "### STEP baseline"; pair base 0 lo 127.0.0.1 40000 ;;
    rccl)
      echo "### STEP rccl"
      lib=$(ls "$($PY -c 'import torch,os;print(os.path.dirname(torch.__file__))')"/lib/librccl.so* /opt/rocm/lib/librccl.so* 2>/dev/null | head -1)
      echo "librccl: ${lib:-none}"
      [ -n "$lib" ] || return
      cat > rccl_uid.py <<'PY'
import ctypes, socket, struct, sys
lib = ctypes.CDLL(sys.argv[1])
uid = ctypes.create_string_buffer(128)
rc = lib.ncclGetUniqueId(uid)
raw = uid.raw
fam = struct.unpack_from("<H", raw, 8)[0]
port = struct.unpack_from(">H", raw, 10)[0]
print("ncclGetUniqueId rc", rc, "family", fam, "addr", socket.inet_ntoa(raw[12:16]) if fam == 2 else "-", "port", port)
PY
      t0=$(now)
      env LD_PRELOAD="$DIR/podshim.so" PODSHIM_LOG="$DIR/rccl.shim.log" $(rank_env 0) NCCL_SOCKET_IFNAME=pods0 \
        NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,BOOTSTRAP ROCR_VISIBLE_DEVICES=-1 timeout 60 "$PY" rccl_uid.py "$lib" 2>&1 | tail -15
      echo "rccl probe in $(since "$t0") s"; cat "$DIR/rccl.shim.log" 2>/dev/null | head -5
      ;;
    pytest)
      echo "### STEP pytest"
      rm -rf repo; mkdir -p repo/lab/net repo/tests
      cp podshim.c podshim_test.c gloo_pair.py repo/lab/net/; cp test_lab_podshim.py repo/tests/
      "$PY" -m pytest --version 2>&1 | head -1
      t0=$(now); (cd repo && "$PY" -m pytest -q -p no:cacheprovider tests/test_lab_podshim.py 2>&1 | tail -5)
      echo "pytest in $(since "$t0") s"
      ;;
    install)
      echo "### STEP install $2 $3"
      mkdir -p "$HOME/r66"
      gcc -O2 -Wall -shared -fPIC -o "$HOME/r66/shim.so" podshim.c -ldl -lpthread && echo "built ~/r66/shim.so"
      cat > "$HOME/r66/shim.env" <<EOF
# podshim settings for R66 (lab/net/podshim.c). Source this, then set LD_PRELOAD=\$HOME/r66/shim.so.
export R66_SELF=$2
export R66_PEER=$3
export R66_BASE=29600
export R66_N=128
export R66_OFF=200
export PODSHIM_IFNAME=pods0
export PODSHIM_LOG=\$HOME/r66/shim.log
export PODSHIM_SELF=\$R66_SELF
export PODSHIM_PEER=\$R66_PEER
export PODSHIM_PORT_LOW=\$R66_BASE
export PODSHIM_PORT_HIGH=\$((R66_BASE + R66_N - 1))
export PODSHIM_PEER_PORT_LOW=\$R66_BASE
export PODSHIM_PEER_PORT_BASE=\$((R66_BASE + R66_OFF))
export PODSHIM_PEER_HOST=127.0.0.1
EOF
      cat "$HOME/r66/shim.env"; ls -l "$HOME/r66/shim.so"; sha256sum "$HOME/r66/shim.so" podshim.c
      ( . "$HOME/r66/shim.env"; env LD_PRELOAD="$HOME/r66/shim.so" PODSHIM_LOG= "$PY" -c \
        'import socket;print("if_nametoindex(pods0) =", socket.if_nametoindex("pods0"))' 2>&1 | tail -1 )
      return 0
      ;;
  esac
}

while [ $# -gt 0 ]; do
  if [ "$1" = install ]; then step install "$2" "$3"; shift 3; else step "$1"; shift; fi
done
echo "### END"
