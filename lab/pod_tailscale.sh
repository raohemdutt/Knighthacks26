#!/bin/bash
# R65: Tailscale in userspace mode on a Strix Halo pod (no root, no /dev/net/tun), so two pods on networks that cannot
# reach each other directly still get a TCP path. CPU only.
#   install                   the static tarball from pkgs.tailscale.com into ~/ts/bin
#   up <hostname>             tailscaled --tun=userspace-networking (SOCKS5 on 127.0.0.1:1055, HTTP proxy on :1056),
#                             then tailscale up with the auth key in ./tsauth.txt next to this script; the key file is
#                             deleted whatever happens
#   status                    tailscale status, ip, netcheck (DERP region and whether UDP gets out)
#   ping <peer>               tailscale ping <peer> (direct or via DERP, and the round trip)
#   stop                      tailscale logout and stop this pod's tailscaled
# In userspace mode the pod has no tailnet interface: an outbound connection to a peer goes through the SOCKS5 or HTTP
# proxy (or `tailscale nc`), and an inbound connection to this pod's tailnet address is handed to the same port on
# 127.0.0.1. lab/r65_link.sh probe-ts measures that path.
set -u
STEP="${1:-status}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VERSION="${TS_VERSION:-1.104.1}"
TS="$HOME/ts"
SOCK="$TS/tailscaled.sock"
export PATH="$TS/bin:$PATH"
mkdir -p "$TS/bin" "$TS/state"

install() {
  local t0=$(date +%s)
  if [ ! -x "$TS/bin/tailscaled" ] || ! "$TS/bin/tailscale" version 2>/dev/null | head -1 | grep -qx "$VERSION"; then
    curl -fsSL -o "$TS/ts.tgz" "https://pkgs.tailscale.com/stable/tailscale_${VERSION}_amd64.tgz" || { echo "download failed"; return 1; }
    tar -xzf "$TS/ts.tgz" -C "$TS" && cp "$TS/tailscale_${VERSION}_amd64/tailscale" "$TS/tailscale_${VERSION}_amd64/tailscaled" "$TS/bin/"
    rm -rf "$TS/ts.tgz" "$TS/tailscale_${VERSION}_amd64"
  fi
  echo "tailscale $("$TS/bin/tailscale" version | head -1) in $TS/bin ($(( $(date +%s) - t0 )) s)"
}

up() {
  local name="${1:?hostname}" rc
  # The key never goes on a command line: tailscale reads it from a file, which is deleted right after.
  [ -s "$HERE/tsauth.txt" ] || { echo "no auth key next to the script"; return 1; }
  umask 077; mv "$HERE/tsauth.txt" "$TS/authkey"
  if ! "$TS/bin/tailscale" --socket="$SOCK" status >/dev/null 2>&1; then
    [ -f "$TS/tailscaled.pid" ] && kill "$(cat "$TS/tailscaled.pid")" 2>/dev/null
    nohup "$TS/bin/tailscaled" --tun=userspace-networking --socket="$SOCK" --statedir="$TS/state"       --socks5-server=127.0.0.1:1055 --outbound-http-proxy-listen=127.0.0.1:1056 > "$TS/tailscaled.log" 2>&1 &
    echo $! > "$TS/tailscaled.pid"
    sleep 3
  fi
  "$TS/bin/tailscale" --socket="$SOCK" up --auth-key="file:$TS/authkey" --hostname="$name" --accept-dns=false --timeout=60s 2>&1
  rc=$?
  rm -f "$TS/authkey"
  echo "tailscale up rc=$rc (key file removed: $([ -e "$TS/authkey" ] && echo no || echo yes))"
  "$TS/bin/tailscale" --socket="$SOCK" ip -4 2>&1
}

status() {
  "$TS/bin/tailscale" --socket="$SOCK" status 2>&1 | head -10
  echo "ip: $("$TS/bin/tailscale" --socket="$SOCK" ip -4 2>&1)"
  timeout 40 "$TS/bin/tailscale" --socket="$SOCK" netcheck 2>&1 | head -25
}

ping() {
  timeout 60 "$TS/bin/tailscale" --socket="$SOCK" ping -c 10 "$1" 2>&1 | tail -12
}

stop() {
  "$TS/bin/tailscale" --socket="$SOCK" logout 2>&1 | tail -2
  [ -f "$TS/tailscaled.pid" ] && kill "$(cat "$TS/tailscaled.pid")" 2>/dev/null && echo "tailscaled stopped"
  rm -f "$TS/tailscaled.pid"
}

case "$STEP" in
  install) install ;;
  up) shift; install && up "$@" ;;
  status) status ;;
  ping) shift; ping "$@" ;;
  stop) stop ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
ls "$HERE/tsauth.txt" >/dev/null 2>&1 && rm -f "$HERE/tsauth.txt"
echo "### END"
