#!/bin/bash
# R65: the direct link between the two SJC Strix Halo pods (CPU only, no GPU, no slot needed).
#   facts                     this pod's hostname, addresses, DNS search path and which network tools exist
#   listen <port> <seconds>   a TCP sink and echo server on 0.0.0.0:<port> for <seconds> (python, nohup)
#   probe <host> <port> [socks] [MB]  connect to the other pod's listener: TCP connect time, a TCP ping loop, 1 GB each way;
#                             with socks (host:port, e.g. 127.0.0.1:1055, tailscaled's SOCKS5) every connection goes
#                             through that proxy: the Tailscale userspace path (lab/pod_tailscale.sh)
#   stop                      stop this task's listener
set -u
STEP="${1:-facts}"
DIR="$HOME/r65"
mkdir -p "$DIR"

facts() {
  echo "hostname: $(hostname)"
  echo "hostname -f: $(hostname -f 2>/dev/null)"
  echo "hostname -I: $(hostname -I 2>/dev/null)"
  echo "--- /etc/hosts"; cat /etc/hosts
  echo "--- /etc/resolv.conf"; cat /etc/resolv.conf
  echo "--- ip"; (ip -4 addr 2>/dev/null || cat /proc/net/fib_trie 2>/dev/null | head -40)
  echo "--- route"; (ip route 2>/dev/null || cat /proc/net/route)
  echo "--- tools"
  for t in iperf3 iperf nc ncat socat ping curl ray; do printf '%s: %s\n' "$t" "$(command -v $t 2>/dev/null || echo none)"; done
  echo "--- env"
  env | grep -E '^(KUBERNETES_SERVICE_HOST|JUPYTERHUB_USER|HOSTNAME|JUPYTERHUB_SERVICE_PREFIX)=' | sed 's/=.*/=<set>/'
  echo "JUPYTERHUB_USER=${JUPYTERHUB_USER:-}"
  echo "--- interfaces"; cat /proc/net/dev | awk 'NR>2{print $1, $2, $10}'
  echo "--- mtu"; for i in /sys/class/net/*; do echo "$(basename $i) mtu $(cat $i/mtu 2>/dev/null) speed $(cat $i/speed 2>/dev/null)"; done
  echo "--- nproc $(nproc)  mem $(free -g | awk '/Mem/{print $2" GB total, "$7" GB avail"}')"
}

listen() {
  local port="${1:-29555}" secs="${2:-900}"
  cat > "$DIR/sink.py" <<'PY'
import socket, sys, threading, time
port, secs = int(sys.argv[1]), float(sys.argv[2])
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", port)); srv.listen(8); srv.settimeout(1.0)
def handle(c, a):
    # first byte picks the mode: E = echo (ping), S = sink (bandwidth), R = source (send N bytes back)
    try:
        mode = c.recv(1)
        if mode == b"E":
            while True:
                d = c.recv(64)
                if not d: break
                c.sendall(d)
        elif mode == b"S":
            n = 0; t0 = time.monotonic()
            while True:
                d = c.recv(1 << 20)
                if not d: break
                n += len(d)
            c.sendall(f"{n} {time.monotonic()-t0:.6f}".encode())
        elif mode == b"R":
            n = int(c.recv(32).decode().strip()); buf = b"\0" * (1 << 20); sent = 0
            while sent < n:
                c.sendall(buf); sent += len(buf)
        print(f"{time.strftime('%H:%M:%S')} {a[0]} mode {mode}", flush=True)
    finally:
        c.close()
end = time.monotonic() + secs
while time.monotonic() < end:
    try:
        c, a = srv.accept()
    except socket.timeout:
        continue
    threading.Thread(target=handle, args=(c, a), daemon=True).start()
PY
  nohup python3 "$DIR/sink.py" "$port" "$secs" > "$DIR/sink.log" 2>&1 &
  echo $! > "$DIR/sink.pid"
  sleep 1
  echo "listener pid $(cat "$DIR/sink.pid") port $port for ${secs}s on $(hostname) $(hostname -I 2>/dev/null)"
  (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -E ":$port\b" || true
}

probe() {
  local host="$1" port="${2:-29555}"
  cat > "$DIR/probe.py" <<'PY'
import socket, statistics, struct, sys, time
host, port = sys.argv[1], int(sys.argv[2])
socks = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
size = int(sys.argv[4]) << 20 if len(sys.argv) > 4 and sys.argv[4] else 1 << 30   # bytes for the timed sends
_plain = socket.create_connection
def connect(addr, timeout=30):
    if not socks:
        return _plain(addr, timeout=timeout)
    ph, pp = socks.rsplit(":", 1)
    s = _plain((ph, int(pp)), timeout=timeout)
    s.sendall(bytes([5, 1, 0]))
    if s.recv(2) != bytes([5, 0]):
        raise OSError("SOCKS5 refused the no-auth method")
    s.sendall(bytes([5, 1, 0, 1]) + socket.inet_aton(addr[0]) + struct.pack(">H", addr[1]))
    rep = s.recv(10)
    if len(rep) < 2 or rep[1] != 0:
        raise OSError(f"SOCKS5 connect failed, reply {rep!r}")
    return s
socket.create_connection = connect
print("path:", f"SOCKS5 {socks}" if socks else "direct")
t = time.monotonic()
try:
    s = socket.create_connection((host, port), timeout=5)
except Exception as e:
    print(f"connect {host}:{port} FAILED {type(e).__name__}: {e}"); sys.exit(3)
print(f"connect {host}:{port} ok in {1000*(time.monotonic()-t):.2f} ms, peer {s.getpeername()}")
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.sendall(b"E")
rtts = []
for i in range(200):
    t = time.perf_counter(); s.sendall(b"x" * 8); got = b""
    while len(got) < 8: got += s.recv(8 - len(got))
    rtts.append((time.perf_counter() - t) * 1e6)
s.close()
rtts.sort()
print(f"tcp ping 200: min {rtts[0]:.0f} us, median {statistics.median(rtts):.0f} us, p95 {rtts[189]:.0f} us, max {rtts[-1]:.0f} us")
for label, total in (("warm-up 64 MB", 64 << 20), (f"send {size >> 20} MB", size), (f"send {size >> 20} MB again", size)):
    s = socket.create_connection((host, port), timeout=30)
    s.sendall(b"S"); buf = b"\0" * (1 << 20); sent = 0; t = time.monotonic()
    while sent < total:
        s.sendall(buf); sent += len(buf)
    s.shutdown(socket.SHUT_WR)
    reply = s.recv(64).decode(); el = time.monotonic() - t; s.close()
    n, rx_s = reply.split()
    print(f"{label}: {sent} bytes in {el:.3f} s = {sent/el/1e9*8:.2f} Gbit/s ({sent/el/1e6:.0f} MB/s); receiver counted {n} in {float(rx_s):.3f} s")
# the other direction over one connection this pod opened: the listener sends 1 GB back
s = socket.create_connection((host, port), timeout=30)
s.sendall(b"R"); s.sendall(str(size).ljust(32).encode()); got = 0; t = time.monotonic()
while got < size:
    d = s.recv(1 << 20)
    if not d: break
    got += len(d)
el = time.monotonic() - t; s.close()
print(f"receive {size >> 20} MB: {got} bytes in {el:.3f} s = {got/el/1e9*8:.2f} Gbit/s ({got/el/1e6:.0f} MB/s)")
PY
  echo "probe from $(hostname) $(hostname -I 2>/dev/null) to $host:$port"
  getent hosts "$host" || echo "getent: $host does not resolve"
  timeout 900 python3 "$DIR/probe.py" "$host" "$port" "${3:-}" "${4:-}"
  if command -v ping >/dev/null; then timeout 20 ping -c 10 -i 0.2 "$host" 2>&1 | tail -3; fi
}

stop() {
  [ -f "$DIR/sink.pid" ] && kill "$(cat "$DIR/sink.pid")" 2>/dev/null && echo "listener stopped"
  rm -f "$DIR/sink.pid"; tail -20 "$DIR/sink.log" 2>/dev/null
}

case "$STEP" in
  facts) facts ;;
  listen) shift; listen "$@" ;;
  probe) shift; probe "$@" ;;
  stop) stop ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
echo "### END"
