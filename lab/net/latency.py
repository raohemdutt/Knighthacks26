"""R66 latency check through the tunnel: one write per round trip against two writes per round trip.

  python3 latency.py serve PORT SECONDS     a server that reads N bytes and answers 1 byte
  python3 latency.py client HOST PORT       200 round trips each way of writing
"""
import socket, statistics, sys, threading, time


def serve(port, secs):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port)); s.listen(4); s.settimeout(1.0); end = time.time() + secs

    def handle(c):
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            d = b""
            while len(d) < 64:
                x = c.recv(64 - len(d))
                if not x:
                    return
                d += x
            c.sendall(b"k")

    while time.time() < end:
        try:
            c, _ = s.accept()
        except socket.timeout:
            continue
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def run(c, split):
    rt = []
    for _ in range(200):
        t = time.perf_counter()
        if split:
            c.sendall(b"a" * 16); c.sendall(b"b" * 48)
        else:
            c.sendall(b"a" * 64)
        c.recv(1); rt.append((time.perf_counter() - t) * 1000)
    rt.sort()
    return f"min {rt[0]:.2f} median {statistics.median(rt):.2f} p95 {rt[189]:.2f} ms"


def client(host, port):
    for nodelay in (1, 0):
        c = socket.create_connection((host, port), timeout=30)
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, nodelay)
        print(f"{host}:{port} nodelay={nodelay} one write: {run(c, False)}")
        print(f"{host}:{port} nodelay={nodelay} two writes: {run(c, True)}")
        c.close()


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(int(sys.argv[2]), float(sys.argv[3]))
    else:
        client(sys.argv[2], int(sys.argv[3]))
