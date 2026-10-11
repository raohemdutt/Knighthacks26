"""Tests for lab/net: the probe's client/server talk over loopback, and the shell scripts are well formed.

These run anywhere (no pod, no GPU): the probe test binds a real socket on 127.0.0.1, and the script tests only
parse the shell with `bash -n`. The cross-pod gate (gloo through the shim and the tunnel) is measured on the pods
and recorded in lab/refine/research/plumbing/network-shim.md, not here.
"""
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

NET = Path(__file__).resolve().parent.parent


def test_probe_roundtrip_over_loopback():
    sys.path.insert(0, str(NET))
    import probe

    port = 29771
    t = threading.Thread(target=probe.serve, args=(port, 6), daemon=True)
    t.start()
    time.sleep(0.5)
    import socket

    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    c.sendall(b"who\n")
    who = b""
    while len(who) < 64:
        who += c.recv(64 - len(who))
    assert who.strip()  # the server named a host
    c.sendall(b"ping\n")
    assert c.recv(1) == b"p"
    c.sendall(b"sink 1048576\n")
    c.sendall(b"\0" * 1048576)
    assert c.recv(2) == b"ok"
    c.close()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", ["chisel_tunnel.sh", "gate.sh", "vllm_pair.sh"])
def test_script_parses(script):
    # the bytes go in on stdin, so neither a Windows path nor a newline translation reaches bash
    r = subprocess.run(["bash", "-n"], input=(NET / script).read_bytes(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_tunnel_pins_chisel_version():
    # the version and its download URL must agree, so a bump cannot leave a stale URL
    text = (NET / "chisel_tunnel.sh").read_text()
    assert "v1.12.0" in text and "chisel_1.12.0_linux_amd64.gz" in text


def test_ackrelay_passes_bytes_both_ways():
    # the quick-ACK relay in front of a loopback echo: bytes go through unchanged in both directions
    import socket

    sys.path.insert(0, str(NET))
    import ackrelay

    echo = socket.socket()
    echo.bind(("127.0.0.1", 0)); echo.listen(1)
    target = echo.getsockname()[1]

    def serve_echo():
        c, _ = echo.accept()
        while True:
            d = c.recv(65536)
            if not d:
                break
            c.sendall(d)

    threading.Thread(target=serve_echo, daemon=True).start()
    probe_sock = socket.socket(); probe_sock.bind(("127.0.0.1", 0)); listen = probe_sock.getsockname()[1]; probe_sock.close()
    threading.Thread(target=ackrelay.main, args=(listen, target), daemon=True).start()
    time.sleep(0.3)
    c = socket.create_connection(("127.0.0.1", listen), timeout=5)
    payload = bytes(range(256)) * 64
    c.sendall(payload)
    got = b""
    while len(got) < len(payload):
        got += c.recv(65536)
    assert got == payload
    c.close()
