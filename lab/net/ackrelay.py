"""R66 quick-ACK relay: sits between the Jupyter proxy and the chisel server on the same pod.

  python3 ackrelay.py LISTEN_PORT TARGET_PORT [TARGET_HOST]

Every socket gets TCP_NODELAY, and both sockets re-arm TCP_QUICKACK after each read, so
the proxy's Nagle-held second segment is released by an immediate ACK instead of the 40 ms delayed one.
"""
import socket, sys, threading

QUICKACK = getattr(socket, "TCP_QUICKACK", 12)


def pump(src, dst, quick):
    try:
        while True:
            if quick:
                src.setsockopt(socket.IPPROTO_TCP, QUICKACK, 1)
            d = src.recv(1 << 18)
            if quick:
                src.setsockopt(socket.IPPROTO_TCP, QUICKACK, 1)
            if not d:
                break
            dst.sendall(d)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def main(listen, target, host="127.0.0.1"):
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", listen)); ls.listen(64)
    while True:
        a, _ = ls.accept()
        b = socket.create_connection((host, target))
        for s in (a, b):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=pump, args=(a, b, True), daemon=True).start()
        threading.Thread(target=pump, args=(b, a, True), daemon=True).start()   # both reads re-arm quickack


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1")
