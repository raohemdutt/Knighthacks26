"""R66 probe: an echo/sink server and a client that measures round trip and throughput over a TCP port.

  python3 probe.py serve PORT SECONDS
  python3 probe.py client HOST PORT [MB]
"""
import socket, statistics, sys, threading, time


def serve(port, secs):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port)); s.listen(16); s.settimeout(1.0)
    end = time.time() + secs

    def handle(c):
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = c.makefile("rb")
        while True:
            line = f.readline()
            if not line:
                break
            cmd = line.split()
            if cmd[0] == b"who":
                c.sendall(socket.gethostname().encode()[:64].ljust(64))
            elif cmd[0] == b"ping":
                c.sendall(b"p")
            elif cmd[0] == b"sink":
                n = int(cmd[1]); got = 0
                while got < n:
                    d = f.read1(min(1 << 20, n - got)) if hasattr(f, "read1") else f.read(min(1 << 20, n - got))
                    if not d:
                        break
                    got += len(d)
                c.sendall(b"ok")
            elif cmd[0] == b"blob":
                n = int(cmd[1]); buf = b"\0" * (1 << 20); sent = 0
                while sent < n:
                    k = min(len(buf), n - sent); c.sendall(buf[:k]); sent += k
        c.close()

    while time.time() < end:
        try:
            c, _ = s.accept()
        except socket.timeout:
            continue
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def client(host, port, mb):
    t = time.perf_counter()
    c = socket.create_connection((host, port), timeout=60)
    conn_ms = (time.perf_counter() - t) * 1000
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    c.sendall(b"who\n"); who = b""
    while len(who) < 64:
        who += c.recv(64 - len(who))
    print(f"{host}:{port} answered by {who.decode().strip()}")
    rt = []
    for _ in range(200):
        t = time.perf_counter(); c.sendall(b"ping\n"); c.recv(1); rt.append((time.perf_counter() - t) * 1000)
    rt.sort()
    print(f"{host}:{port} connect {conn_ms:.2f} ms; 200 pings min {rt[0]:.2f} median {statistics.median(rt):.2f} p95 {rt[189]:.2f} ms")
    n = mb << 20
    buf = b"\0" * (1 << 20)
    t = time.monotonic(); c.sendall(f"sink {n}\n".encode()); sent = 0
    while sent < n:
        c.sendall(buf); sent += len(buf)
    c.recv(2); el = time.monotonic() - t
    print(f"send {mb} MB in {el:.2f} s = {n/el/1e6:.1f} MB/s")
    t = time.monotonic(); c.sendall(f"blob {n}\n".encode()); got = 0
    while got < n:
        d = c.recv(1 << 20)
        if not d:
            break
        got += len(d)
    el = time.monotonic() - t
    print(f"receive {got >> 20} MB in {el:.2f} s = {got/el/1e6:.1f} MB/s")
    c.close()


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(int(sys.argv[2]), float(sys.argv[3]))
    else:
        client(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 64)
