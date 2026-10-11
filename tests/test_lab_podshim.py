"""lab/net/podshim.c: the LD_PRELOAD network shim (R66), built with gcc and run for real on Linux.

The C test (lab/net/podshim_test.c) checks the fake interface, the pinned binds, the PEER mapping into a local
"tunnel" listener, the IPv4-mapped IPv6 connect, getsockname/getpeername and 400 concurrent connects. With torch
installed, two gloo ranks in two processes form a process group through the shim (lab/net/gloo_pair.py). Everything
skips cleanly where there is no gcc or no LD_PRELOAD (Windows, macOS).
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

NET = Path(__file__).resolve().parents[1] / "lab" / "net"

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("gcc") is None,
                                reason="needs Linux and gcc")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("podshim")
    so, exe = out / "podshim.so", out / "podshim_test"
    subprocess.run(["gcc", "-O2", "-Wall", "-shared", "-fPIC", "-o", str(so), str(NET / "podshim.c"), "-ldl",
                    "-lpthread"], check=True, capture_output=True, text=True)
    subprocess.run(["gcc", "-O2", "-Wall", "-o", str(exe), str(NET / "podshim_test.c"), "-lpthread"], check=True,
                   capture_output=True, text=True)
    return so, exe


def _ranges() -> tuple[int, int, int]:
    """Three disjoint 128-port blocks in an unprivileged, rarely used band: own range, peer range, tunnel end."""
    base = random.randrange(45000, 52000, 512)
    return base, base + 128, base + 256


def _shim_env(so: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PODSHIM_")}
    env.update(LD_PRELOAD=str(so), **extra)
    return env


def test_c_program_passes_every_check(built):
    so, exe = built
    low, peer_low, tunnel = _ranges()
    env = _shim_env(so, PODSHIM_SELF="10.99.0.1", PODSHIM_PEER="10.99.0.2", PODSHIM_PORT_LOW=str(low),
                    PODSHIM_PORT_HIGH=str(low + 127), PODSHIM_PEER_PORT_LOW=str(peer_low),
                    PODSHIM_PEER_PORT_BASE=str(tunnel))
    run = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "PASSED: 0 failure(s)" in run.stdout
    assert "FAIL" not in run.stdout


def test_c_program_fails_without_the_shim(built):
    _, exe = built
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    env.update(PODSHIM_SELF="10.99.0.1", PODSHIM_PEER="10.99.0.2")
    run = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode != 0
    assert "FAIL getifaddrs lists pods0" in run.stdout


def test_unconfigured_shim_changes_nothing(built):
    so, _ = built
    code = "import socket\ntry:\n    socket.if_nametoindex('pods0')\n    print('found')\nexcept OSError:\n    print('absent')"
    run = subprocess.run([sys.executable, "-c", code], env=_shim_env(so), capture_output=True, text=True, timeout=60)
    assert run.stdout.strip() == "absent", run.stderr


def test_gloo_forms_through_the_shim(built, tmp_path):
    pytest.importorskip("torch.distributed")
    so, _ = built
    low0, low1, _ = _ranges()
    ranks = []
    for rank, (self_ip, peer_ip, own, peer) in enumerate([("10.99.0.1", "10.99.0.2", low0, low1),
                                                          ("10.99.0.2", "10.99.0.1", low1, low0)]):
        env = _shim_env(so, PODSHIM_SELF=self_ip, PODSHIM_PEER=peer_ip, PODSHIM_PORT_LOW=str(own),
                        PODSHIM_PORT_HIGH=str(own + 127), PODSHIM_PEER_PORT_LOW=str(peer),
                        PODSHIM_PEER_PORT_BASE=str(peer), PODSHIM_LOG=str(tmp_path / "shim.log"),
                        GLOO_SOCKET_IFNAME="pods0", MASTER_ADDR="10.99.0.1", MASTER_PORT=str(low0),
                        CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
        ranks.append(subprocess.Popen([sys.executable, str(NET / "gloo_pair.py"), str(rank), "--timeout", "30"],
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outputs = [p.communicate(timeout=120)[0] for p in ranks]
    assert all(p.returncode == 0 for p in ranks), outputs
    trace = (tmp_path / "shim.log").read_text()
    assert "(pinned)" in trace and f"10.99.0.1:{low0} -> 127.0.0.1:{low0}" in trace
