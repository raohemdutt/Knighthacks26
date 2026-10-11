#!/bin/bash
# R66 tunnel control on one pod.
#   chisel_tunnel.sh server            start the chisel server (pod D) on 127.0.0.1:29999 behind Jupyter's /proxy/29999/
#   chisel_tunnel.sh client PEERUSER   start the chisel client (pod C) with the peer account token from ./tokpeer.txt
#   chisel_tunnel.sh echo SECONDS      start the probe echo server on 29600 for SECONDS
#   chisel_tunnel.sh probe HOST PORT MB
#   chisel_tunnel.sh stop              stop what this script started (pid files only)
#   chisel_tunnel.sh facts
HERE="$(cd "$(dirname "$0")" && pwd)"
W=~/r66; mkdir -p $W; cp -f "$HERE"/probe.py $W/ 2>/dev/null
BASE=${R66_BASE:-29600}; N=${R66_N:-64}; OFF=${R66_OFF:-100}
getchisel() {
  if [ ! -x $W/chisel ]; then
    curl -sSL -m 120 -o $W/chisel.gz https://github.com/jpillora/chisel/releases/download/v1.12.0/chisel_1.12.0_linux_amd64.gz \
      && gunzip -f $W/chisel.gz && chmod +x $W/chisel
  fi
  $W/chisel --version 2>&1 | head -1; sha256sum $W/chisel | cut -c1-16
}
case "$1" in
facts)
  echo "host $(hostname) $(hostname -I 2>/dev/null)"
  which gcc cc; gcc --version 2>&1 | head -1
  for py in python3 ~/venv-vllm-new/bin/python; do
    echo "$py: $($py -c 'import jupyter_server_proxy as j; print("jupyter_server_proxy", getattr(j,"__version__","?"))' 2>&1 | tail -1)"
  done
  pgrep -f jupyterhub-singleuser >/dev/null && echo "singleuser up" ; pgrep -af "jupyter" | cut -c1-160 | head -3
  ls /usr/include/ifaddrs.h 2>&1
  ;;
server)
  getchisel
  if [ -f $W/srv.pid ] && kill -0 "$(cat $W/srv.pid)" 2>/dev/null; then echo "server already up pid $(cat $W/srv.pid)"; else
    setsid nohup $W/chisel server --host 127.0.0.1 --port 29999 --reverse --keepalive 25s > $W/srv.log 2>&1 < /dev/null &
    echo $! > $W/srv.pid; sleep 1
  fi
  echo "health: $(curl -s -m 3 http://127.0.0.1:29999/health)"; tail -3 $W/srv.log
  ;;
client)
  TOK="$(cat "$HERE/tokpeer.txt" 2>/dev/null)"; rm -f "$HERE/tokpeer.txt"
  [ -n "$TOK" ] || { echo "no token"; exit 1; }
  getchisel
  U="http://proxy-public.jupyterhub.svc.cluster.local/user/$2/proxy/${R66_PPORT:-29999}"
  echo "health through hub proxy: $(curl -s -m 5 -H "Host: sjc.aupcloud.io" -H "Authorization: token $TOK" -o /dev/null -w '%{http_code} %{time_total}s' $U/health)"
  REM=""
  for i in $(seq 0 $((N-1))); do
    REM="$REM 127.0.0.1:$((BASE+OFF+i)):127.0.0.1:$((BASE+i)) R:127.0.0.1:$((BASE+OFF+i)):127.0.0.1:$((BASE+i))"
  done
  if [ -f $W/cli.pid ] && kill -0 "$(cat $W/cli.pid)" 2>/dev/null; then echo "client already up"; else
    printf '%s' "Authorization: token $TOK" > $W/.h; chmod 600 $W/.h
    # the header is passed from a file read by bash so it is not in this script's text
    setsid nohup bash -c 'exec '"$W"'/chisel client --keepalive 25s --hostname sjc.aupcloud.io --header "$(cat '"$W"'/.h)" "$0" "$@"' "$U/" $REM > $W/cli.log 2>&1 < /dev/null &
    echo $! > $W/cli.pid; sleep 3; rm -f $W/.h
  fi
  grep -v -i "token" $W/cli.log | tail -4
  ;;
echo)
  setsid nohup python3 $W/probe.py serve $BASE "$2" > $W/echo.log 2>&1 < /dev/null &
  echo $! > $W/echo.pid; sleep 1; echo "echo on $BASE for $2 s"
  ;;
probe)
  python3 $W/probe.py client "$2" "$3" "${4:-64}"
  ;;
stop)
  for p in cli srv echo; do [ -f $W/$p.pid ] && kill "$(cat $W/$p.pid)" 2>/dev/null && echo "stopped $p"; rm -f $W/$p.pid; done
  rm -f $W/.h "$HERE/tokpeer.txt"
  ;;
esac
echo "### END"
