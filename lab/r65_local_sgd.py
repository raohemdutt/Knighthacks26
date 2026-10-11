"""R65: the L16 LoRA trained by two ranks on two pods that share no network, averaged once per epoch (local SGD).

The two SJC Strix Halo pods cannot open a connection to each other (the hub's network policy refuses pod-to-pod
traffic) and their outbound UDP is blocked, so the only path between them is Tailscale in userspace mode relayed
through a DERP server: one TCP connection at about 2.7 MB/s with a 10 ms round trip (lab/pod_tailscale.sh,
lab/refine/research/plumbing/symmetric_pods.md). torch.distributed cannot form over that path: gloo and RCCL each
advertise an interface address the peer must reach, and a userspace tailnet gives the pod no interface. So the two
ranks exchange their LoRA weights themselves, over one TCP connection, at the end of every epoch:

- Each rank runs lab/train.py unchanged (the L16 recipe: r 16, alpha 32, lr 2e-4, batch 4 by accumulation 4, seed 42)
  on its half of the training rows; the same seed gives both ranks the same LoRA initialisation.
- At every epoch end (the trainer's on_epoch_end, before the adapter is saved) both ranks send their trainable weights
  as bfloat16 at the same time and both set them to the mean of the two bfloat16 copies, so both hold identical weights
  afterwards (checked with a sha256 of the result). Optimizer states stay local, as in local SGD.
- Rank 0 listens on 127.0.0.1:<port> (Tailscale userspace hands an inbound tailnet connection to the same localhost
  port); rank 1 connects through tailscaled's SOCKS5 proxy to rank 0's tailnet address.

    python lab/r65_local_sgd.py --rank 0 --port 29600 -- <lab/train.py arguments>
    python lab/r65_local_sgd.py --rank 1 --peer 100.x.y.z --port 29600 --socks 127.0.0.1:1055 -- <lab/train.py arguments>

Each rank writes <out>/local_sgd.json: every exchange's bytes, seconds (pack, transfer, apply) and the sha256 of the
averaged weights, and a memory guard's lowest MemAvailable reading. The guard stops this rank (never another process)
when the pod's MemAvailable falls below --min-free-gib, so a rank sharing a pod with the teacher can never push the
teacher out of memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HEADER = struct.Struct(">Q")


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1 << 20)
    return float("nan")


def socks5_connect(proxy: str, host: str, port: int, timeout: float) -> socket.socket:
    proxy_host, proxy_port = proxy.rsplit(":", 1)
    sock = socket.create_connection((proxy_host, int(proxy_port)), timeout=timeout)
    sock.sendall(bytes([5, 1, 0]))
    if sock.recv(2) != bytes([5, 0]):
        raise OSError("the SOCKS5 proxy refused the no-auth method")
    sock.sendall(bytes([5, 1, 0, 1]) + socket.inet_aton(host) + struct.pack(">H", port))
    reply = sock.recv(10)
    if len(reply) < 2 or reply[1] != 0:
        raise OSError(f"SOCKS5 connect to {host}:{port} failed: {reply!r}")
    return sock


def recv_exact(sock: socket.socket, size: int) -> bytes:
    view = bytearray(size)
    got = 0
    while got < size:
        n = sock.recv_into(memoryview(view)[got:], min(size - got, 1 << 20))
        if n == 0:
            raise ConnectionError("the peer closed the connection")
        got += n
    return bytes(view)


def send_blob(sock: socket.socket, blob: bytes) -> None:
    sock.sendall(HEADER.pack(len(blob)) + blob)


def recv_blob(sock: socket.socket) -> bytes:
    (size,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    return recv_exact(sock, size)


def swap(sock: socket.socket, blob: bytes) -> bytes:
    """Send blob and receive the peer's at the same time (both ranks call this at once)."""
    errors: list[BaseException] = []

    def sender() -> None:
        try:
            send_blob(sock, blob)
        except BaseException as exc:   # surfaced below
            errors.append(exc)

    thread = threading.Thread(target=sender, daemon=True)
    thread.start()
    theirs = recv_blob(sock)
    thread.join()
    if errors:
        raise errors[0]
    return theirs


class Link:
    def __init__(self, rank: int, peer: str | None, port: int, socks: str | None, wait_s: float) -> None:
        self.rank, self.peer, self.port, self.socks, self.wait_s = rank, peer, port, socks, wait_s
        self.sock: socket.socket | None = None
        self.connect_s: float | None = None

    def open(self) -> None:
        started = time.monotonic()
        if self.rank == 0:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", self.port))
            server.listen(1)
            server.settimeout(self.wait_s)
            self.sock, _ = server.accept()
            server.close()
        else:
            last: Exception | None = None
            while time.monotonic() - started < self.wait_s:
                try:
                    if self.socks:
                        self.sock = socks5_connect(self.socks, self.peer or "", self.port, timeout=10)
                    else:
                        self.sock = socket.create_connection((self.peer, self.port), timeout=10)
                    # a SOCKS connect succeeds before the far side accepts: prove the peer is there
                    self.sock.settimeout(30)
                    send_blob(self.sock, b"hello")
                    if recv_blob(self.sock) != b"hello":
                        raise OSError("bad handshake")
                    break
                except (OSError, ConnectionError) as exc:
                    last = exc
                    self.sock = None
                    time.sleep(3)
            if self.sock is None:
                raise ConnectionError(f"no peer at {self.peer}:{self.port} after {self.wait_s:.0f} s: {last}")
        if self.rank == 0:
            self.sock.settimeout(30)
            if recv_blob(self.sock) != b"hello":
                raise OSError("bad handshake")
            send_blob(self.sock, b"hello")
        self.sock.settimeout(1800)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connect_s = time.monotonic() - started


def make_callback(link: Link, record: dict[str, Any]) -> Any:
    import torch
    from transformers import TrainerCallback

    def trainable(model: Any) -> list[tuple[str, Any]]:
        return sorted(((n, p) for n, p in model.named_parameters() if p.requires_grad), key=lambda item: item[0])

    class LocalSGD(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            if link.sock is None:
                link.open()
            params = trainable(model)
            names = "\n".join(f"{n}:{tuple(p.shape)}" for n, p in params).encode()
            theirs = swap(link.sock, names)
            if theirs != names:
                raise RuntimeError("the two ranks do not train the same LoRA parameters")
            record.update(connect_s=round(link.connect_s or 0.0, 3), tensors=len(params),
                          parameters=sum(p.numel() for _, p in params))
            # identical by seed: checked with a digest, not a full exchange (one exchange is about 65 MB over the relay)
            mine = torch.cat([p.detach().to(torch.bfloat16).flatten() for _, p in params]).cpu()
            digest = hashlib.sha256(mine.view(torch.int16).numpy().tobytes()).hexdigest()
            record["init_identical"] = swap(link.sock, digest.encode()).decode() == digest

        def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            self.round(model, state, label=f"epoch {round(state.epoch or 0)}")

        def round(self, model: Any, state: Any, label: str) -> None:
            params = trainable(model)
            t0 = time.monotonic()
            mine = torch.cat([p.detach().to(torch.bfloat16).flatten() for _, p in params]).cpu()
            blob = mine.view(torch.int16).numpy().tobytes()
            t1 = time.monotonic()
            theirs_blob = swap(link.sock, blob)
            t2 = time.monotonic()
            theirs = torch.frombuffer(bytearray(theirs_blob), dtype=torch.int16).view(torch.bfloat16)
            mean = (mine.float() + theirs.float()) / 2
            offset = 0
            with torch.no_grad():
                for _, p in params:
                    n = p.numel()
                    p.copy_(mean[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype))
                    offset += n
            digest = hashlib.sha256(mean.numpy().tobytes()).hexdigest()
            t3 = time.monotonic()
            other = swap(link.sock, digest.encode()).decode()
            drift = float((mine.float() - theirs.float()).abs().mean())
            row = {"round": label, "step": state.global_step, "bytes_sent": len(blob), "bytes_received": len(theirs_blob),
                   "pack_s": round(t1 - t0, 3), "transfer_s": round(t2 - t1, 3), "apply_s": round(t3 - t2, 3),
                   "mb_s_each_way": round(len(blob) / max(t2 - t1, 1e-9) / 1e6, 3),
                   "mean_abs_diff_before": drift, "sha256": digest, "peer_matches": other == digest,
                   "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            record["rounds"].append(row)
            print(f"### local-sgd {json.dumps(row)}", file=sys.stderr, flush=True)
            if other != digest:
                raise RuntimeError(f"the averaged weights differ between the ranks after {label}")

    return LocalSGD()


def memory_guard(min_free_gib: float, record: dict[str, Any], stop: threading.Event) -> None:
    lowest = float("inf")
    while not stop.wait(1.0):
        free = mem_available_gib()
        lowest = min(lowest, free)
        record["lowest_mem_available_gib"] = round(lowest, 2)
        if free < min_free_gib:
            record["guard"] = f"stopped: MemAvailable {free:.1f} GiB below {min_free_gib} GiB"
            print(f"### memory guard: {record['guard']}", file=sys.stderr, flush=True)
            os.kill(os.getpid(), signal.SIGINT)
            return


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: r65_local_sgd.py --rank N [--peer IP] --port P [--socks HOST:PORT] -- <train.py args>")
    cut = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--peer")
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--socks")
    parser.add_argument("--wait-s", type=float, default=900.0)
    parser.add_argument("--min-free-gib", type=float, default=3.0)
    ours = parser.parse_args(argv[:cut])
    train_argv = argv[cut + 1:]
    if ours.rank == 1 and not ours.peer:
        raise SystemExit("rank 1 needs --peer")
    out = Path(train_argv[train_argv.index("--out") + 1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"rank": ours.rank, "peer": ours.peer, "port": ours.port, "socks": ours.socks,
                              "host": socket.gethostname(), "rounds": [], "started": time.time()}
    stop = threading.Event()
    threading.Thread(target=memory_guard, args=(ours.min_free_gib, record, stop), daemon=True).start()
    link = Link(ours.rank, ours.peer, ours.port, ours.socks, ours.wait_s)
    from lab import train   # the pinned environment is applied by train.main before torch is imported
    answer: dict[str, Any] = {}
    try:
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        answer = train.main(train_argv, callbacks=[make_callback_proxy(link, record)])
    finally:
        stop.set()
        record["ended"] = time.time()
        record["wall_s"] = round(record["ended"] - record["started"], 1)
        record["train_answer"] = {k: answer.get(k) for k in ("ok", "outcome", "error", "metrics")} if answer else None
        (out / "local_sgd.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if link.sock is not None:
            link.sock.close()
    print(json.dumps({"ok": answer.get("ok"), "outcome": answer.get("outcome"), "rank": ours.rank,
                      "rounds": len(record["rounds"]), "out": str(out)}))
    return 0 if answer.get("ok") else 1


def make_callback_proxy(link: Link, record: dict[str, Any]) -> Any:
    """The TrainerCallback, built once train.main has imported torch under its pinned environment. TRL takes callback
    instances, so the class is made here (importing transformers imports torch: train.main has already applied the
    environment by the time callbacks are used, but not when this list is built, so the env is applied first)."""
    from lab import train
    train.apply_env(train.run_env())
    return make_callback(link, record)


if __name__ == "__main__":
    sys.exit(main())
