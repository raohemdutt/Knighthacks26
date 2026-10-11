"""R65: the L16 LoRA trained by two ranks on two pods that share no network (local SGD with compressed deltas).

The two SJC Strix Halo pods cannot open a connection to each other (the hub's network policy refuses pod-to-pod
traffic) and their outbound UDP is blocked, so torch.distributed cannot form between them
(lab/refine/research/plumbing/symmetric-pods.md). Two paths do exist, and this module uses each for what it is good at:

- Control over Tailscale (userspace mode, relayed through DERP: about 10 ms and 2.7 MB/s). Rank 0 listens on
  127.0.0.1:<ctl port> (Tailscale userspace hands an inbound tailnet connection to the same localhost port); rank 1
  dials rank 0's tailnet address through tailscaled's SOCKS5 proxy. The channel carries only small messages: the
  handshake (same LoRA tensors, same starting weights), and at every round each rank's announcement that its delta is
  published, which is also the round's barrier.
- Data over the hub's in-cluster proxy (proxy-public:80, about 90 MB/s a connection and 4 ms). Each rank serves its
  own deltas from a small HTTP mailbox on 127.0.0.1:<box port>; the peer fetches them through
  /user/<this pod's account>/proxy/<box port>/ with this account's hub token (read from a file that is deleted at once,
  never printed), in chunks well under the proxy's ~100 MB response limit, on parallel connections.

The algorithm, every --k optimizer steps:
1. delta = (my weights - the shared base) + my error-feedback residual; quantize it to int8 in blocks of 4096 values
   with one float32 scale each; the residual keeps what the int8 code lost (error feedback, kept locally).
2. Publish the code in the mailbox, swap the announcement over the control channel (the barrier), and start fetching
   the peer's code in the background. Training goes on.
3. One step later, merge: the new shared base = base + (my dequantized delta + the peer's) / 2, the same on both ranks;
   my weights move by (new base - my weights at publish time), so the step taken meanwhile is kept (applied a step late).
At the end of training (before the adapter is saved) a last synchronous round leaves both ranks on the same base, which
both check with a sha256, and the final code is also fetched once over the Tailscale relay to time that path.

    python lab/r65_local_sgd.py --rank 0 --peer-tailnet <rank 1 IP> --peer-user <rank 1's account> \
        --token-file <file with rank 1's hub token> -- <lab/train.py arguments>
    python lab/r65_local_sgd.py --rank 1 --peer-tailnet <rank 0 IP> --peer-user <rank 0's account> \
        --token-file <file with rank 0's hub token> -- <lab/train.py arguments>

Each rank writes <out>/local_sgd.json: per round the bytes, the pack, publish-to-fetched and merge seconds, the
reconstruction error of the int8 code, the drift between the ranks, and the relay timing; and the memory guard's lowest
MemAvailable reading. The guard stops this rank (never another process) when MemAvailable falls below --min-free-gib,
so a rank sharing a pod with the teacher can never push the teacher out of memory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HEADER = struct.Struct(">Q")
BLOCK = 4096
PROXY_HOST = "proxy-public.jupyterhub.svc.cluster.local"
HUB_HOST = "sjc.aupcloud.io"


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


def swap_json(sock: socket.socket, message: dict[str, Any]) -> dict[str, Any]:
    return json.loads(swap(sock, json.dumps(message, sort_keys=True).encode()).decode())


# --- the mailbox: this rank's published codes, served on localhost --------------------------------

class Mailbox:
    def __init__(self, port: int) -> None:
        self.items: dict[str, bytes] = {}
        box = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                key = self.path.rstrip("/").rsplit("/get/", 1)[-1] if "/get/" in self.path else ""
                body = box.items.get(key)
                if body is None:
                    body, code = b"not yet", 404
                else:
                    code = 200
                self.send_response(code)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def put(self, key: str, body: bytes) -> None:
        self.items[key] = body

    def close(self) -> None:
        self.server.shutdown()


class Fetcher:
    """GETs the peer's chunks: through the hub's in-cluster proxy (token) or the Tailscale relay (SOCKS5)."""

    def __init__(self, path: str, peer_user: str, box_port: int, token: str, peer_tailnet: str, socks: str,
                 parallel: int) -> None:
        self.path, self.peer_user, self.box_port, self.token = path, peer_user, box_port, token
        self.peer_tailnet, self.socks, self.parallel = peer_tailnet, socks, parallel

    def _connection(self) -> tuple[http.client.HTTPConnection, str, dict[str, str]]:
        if self.path == "proxy":
            conn = http.client.HTTPConnection(PROXY_HOST, 80, timeout=120)
            prefix = f"/user/{self.peer_user}/proxy/{self.box_port}"
            return conn, prefix, {"Authorization": f"token {self.token}", "Host": HUB_HOST}
        conn = http.client.HTTPConnection(self.peer_tailnet, self.box_port, timeout=300)
        if self.socks:   # empty only in a rehearsal on one machine
            conn.sock = socks5_connect(self.socks, self.peer_tailnet, self.box_port, timeout=30)
        return conn, "", {}

    def _one(self, key: str, deadline: float) -> bytes:
        while True:
            conn, prefix, headers = self._connection()
            try:
                conn.request("GET", f"{prefix}/get/{key}", headers=headers)
                answer = conn.getresponse()
                body = answer.read()
            finally:
                conn.close()
            if answer.status == 200:
                return body
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.path}: {key} answered {answer.status}: {body[:120]!r}")
            time.sleep(0.5)

    def fetch(self, keys: list[str], timeout: float = 900.0) -> list[bytes]:
        deadline = time.monotonic() + timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as pool:
            return list(pool.map(lambda key: self._one(key, deadline), keys))


# --- the int8 code with error feedback --------------------------------------------------------------

def quantize(torch: Any, x: Any) -> tuple[bytes, Any]:
    """x: 1-D float32. Returns (scales then int8 codes as bytes, the dequantized tensor)."""
    n = x.numel()
    pad = (-n) % BLOCK
    blocks = torch.nn.functional.pad(x, (0, pad)).view(-1, BLOCK)
    scale = (blocks.abs().amax(dim=1) / 127.0).clamp_min(1e-30)
    codes = torch.round(blocks / scale[:, None]).clamp(-127, 127).to(torch.int8)
    back = (codes.float() * scale[:, None]).flatten()[:n]
    return scale.numpy().tobytes() + codes.flatten()[:n].numpy().tobytes(), back


def dequantize(torch: Any, blob: bytes, n: int) -> Any:
    nblocks = -(-n // BLOCK)
    scale = torch.frombuffer(bytearray(blob[:4 * nblocks]), dtype=torch.float32)
    codes = torch.frombuffer(bytearray(blob[4 * nblocks:]), dtype=torch.int8).float()
    codes = torch.nn.functional.pad(codes, (0, nblocks * BLOCK - n)).view(nblocks, BLOCK)
    return (codes * scale[:, None]).flatten()[:n]


def make_callback(cfg: argparse.Namespace, link: "Control", box: Mailbox, proxy: Fetcher, relay: Fetcher,
                  record: dict[str, Any]) -> Any:
    import torch
    from transformers import TrainerCallback

    def named(model: Any) -> list[tuple[str, Any]]:
        return sorted(((n, p) for n, p in model.named_parameters() if p.requires_grad), key=lambda item: item[0])

    def flat(params: list[Any]) -> Any:
        return torch.cat([p.detach().float().flatten().cpu() for p in params])

    def add_to(params: list[Any], vector: Any) -> None:
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.add_(vector[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype))
                offset += n

    chunk = cfg.chunk_mb << 20

    class LocalSGD(TrainerCallback):
        def __init__(self) -> None:
            self.base = None
            self.residual = None
            self.pending: dict[str, Any] | None = None
            self.round = 0

        def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            if link.sock is None:
                link.open()
            pairs = named(model)
            names = hashlib.sha256("\n".join(f"{n}:{tuple(p.shape)}" for n, p in pairs).encode()).hexdigest()
            self.base = flat([p for _, p in pairs])
            self.residual = torch.zeros_like(self.base)
            digest = hashlib.sha256(self.base.numpy().tobytes()).hexdigest()
            peer = swap_json(link.sock, {"names": names, "init": digest})
            if peer["names"] != names:
                raise RuntimeError("the two ranks do not train the same LoRA parameters")
            record.update(connect_s=round(link.connect_s or 0.0, 3), tensors=len(pairs),
                          parameters=int(self.base.numel()), init_identical=peer["init"] == digest,
                          k=cfg.k, chunk_mb=cfg.chunk_mb, parallel=cfg.parallel)

        def publish(self, model: Any, state: Any, label: str) -> dict[str, Any]:
            t0 = time.monotonic()
            now = flat([p for _, p in named(model)])
            delta = now - self.base + self.residual
            blob, back = quantize(torch, delta)
            self.residual = delta - back
            err = float((delta - back).norm() / delta.norm().clamp_min(1e-30))
            self.round += 1
            keys = []
            for i in range(0, len(blob), chunk):
                key = f"{self.round}/{i // chunk}"
                box.put(key, blob[i:i + chunk])
                keys.append(key)
            t1 = time.monotonic()
            peer = swap_json(link.sock, {"round": self.round, "keys": keys, "bytes": len(blob),
                                         "sha": hashlib.sha256(blob).hexdigest(), "step": state.global_step})
            t2 = time.monotonic()
            row = {"round": self.round, "label": label, "step": state.global_step, "peer_step": peer["step"],
                   "bytes_published": len(blob), "chunks": len(keys), "pack_s": round(t1 - t0, 3),
                   "barrier_s": round(t2 - t1, 3), "reconstruction_rel_err": err}
            result: dict[str, Any] = {"row": row, "now": now, "back": back, "peer": peer}

            def fetch() -> None:
                try:
                    result["blob"] = b"".join(proxy.fetch(peer["keys"]))
                    result["fetch_s"] = time.monotonic() - t2
                except BaseException as exc:
                    result["error"] = exc

            result["thread"] = threading.Thread(target=fetch, daemon=True)
            result["thread"].start()
            return result

        def merge(self, model: Any, pending: dict[str, Any]) -> None:
            t0 = time.monotonic()
            pending["thread"].join()
            waited = time.monotonic() - t0
            if "error" in pending:
                raise pending["error"]
            blob = pending["blob"]
            if hashlib.sha256(blob).hexdigest() != pending["peer"]["sha"]:
                raise RuntimeError("the peer's code arrived damaged")
            theirs = dequantize(torch, blob, self.base.numel())
            new_base = self.base + (pending["back"] + theirs) / 2
            add_to([p for _, p in named(model)], new_base - pending["now"])
            drift = float((pending["back"] - theirs).norm() / pending["back"].norm().clamp_min(1e-30))
            self.base = new_base
            row = pending["row"]
            row.update(bytes_fetched=len(blob), fetch_s=round(pending["fetch_s"], 3),
                       mb_s=round(len(blob) / max(pending["fetch_s"], 1e-9) / 1e6, 2),
                       waited_for_fetch_s=round(waited, 3), merge_s=round(time.monotonic() - t0, 3),
                       rank_delta_rel_diff=drift, base_sha=hashlib.sha256(new_base.numpy().tobytes()).hexdigest(),
                       at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            record["rounds"].append(row)
            print(f"### local-sgd {json.dumps(row)}", file=sys.stderr, flush=True)

        def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            if self.pending is not None:      # the round published one step ago: merge it now (a step late)
                self.merge(model, self.pending)
                self.pending = None
            if state.global_step % cfg.k == 0 and state.global_step < state.max_steps:
                self.pending = self.publish(model, state, f"every {cfg.k} steps")

        def on_train_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            if self.pending is not None:
                self.merge(model, self.pending)
                self.pending = None
            final = self.publish(model, state, "final, synchronous")
            self.merge(model, final)           # nothing trained in between: both ranks now hold the same base
            mine = record["rounds"][-1]["base_sha"]
            theirs = swap_json(link.sock, {"base_sha": mine})["base_sha"]
            record["final_identical"] = theirs == mine
            # the same final code once more over the Tailscale relay, to time that path against the proxy
            t0 = time.monotonic()
            try:
                blob = b"".join(relay.fetch(final["peer"]["keys"], timeout=600))
                seconds = time.monotonic() - t0
                record["relay_exchange"] = {"bytes": len(blob), "seconds": round(seconds, 3),
                                            "mb_s": round(len(blob) / max(seconds, 1e-9) / 1e6, 3),
                                            "same_bytes": hashlib.sha256(blob).hexdigest() == final["peer"]["sha"]}
            except BaseException as exc:
                record["relay_exchange"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            swap_json(link.sock, {"done": True})   # neither rank closes its mailbox before the peer has read it
            print(f"### local-sgd final identical {record['final_identical']} relay {json.dumps(record['relay_exchange'])}",
                  file=sys.stderr, flush=True)

    return LocalSGD()


class Control:
    def __init__(self, rank: int, peer: str, port: int, socks: str, wait_s: float) -> None:
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
            self.sock.settimeout(60)
            if recv_blob(self.sock) != b"hello":
                raise OSError("bad handshake")
            send_blob(self.sock, b"hello")
        else:
            last: Exception | None = None
            while time.monotonic() - started < self.wait_s:
                try:
                    if self.socks:
                        self.sock = socks5_connect(self.socks, self.peer, self.port, timeout=15)
                    else:
                        self.sock = socket.create_connection((self.peer, self.port), timeout=15)
                    self.sock.settimeout(60)
                    send_blob(self.sock, b"hello")   # a SOCKS connect succeeds before the far side accepts
                    if recv_blob(self.sock) != b"hello":
                        raise OSError("bad handshake")
                    break
                except (OSError, ConnectionError) as exc:
                    last = exc
                    self.sock = None
                    time.sleep(3)
            if self.sock is None:
                raise ConnectionError(f"no peer at {self.peer}:{self.port} after {self.wait_s:.0f} s: {last}")
        self.sock.settimeout(1800)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connect_s = time.monotonic() - started


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
        raise SystemExit("usage: r65_local_sgd.py --rank N --peer-tailnet IP --peer-user NAME --token-file F -- <train.py args>")
    cut = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--peer-tailnet", required=True, help="the other pod's Tailscale address")
    parser.add_argument("--peer-user", required=True, help="the hub user name of the other pod's account")
    parser.add_argument("--token-file", required=True, help="the other account's hub token; deleted once read")
    parser.add_argument("--ctl-port", type=int, default=29600)
    parser.add_argument("--box-port", type=int, default=29601)
    parser.add_argument("--peer-box-port", type=int, help="the peer's mailbox port (default --box-port)")
    parser.add_argument("--socks", default="127.0.0.1:1055", help="tailscaled's SOCKS5; empty for a direct test")
    parser.add_argument("--k", type=int, default=4, help="optimizer steps between averaging rounds")
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--path", choices=("proxy", "relay"), default="proxy", help="the data path for the rounds")
    parser.add_argument("--wait-s", type=float, default=900.0)
    parser.add_argument("--min-free-gib", type=float, default=3.0)
    cfg = parser.parse_args(argv[:cut])
    token_file = Path(cfg.token_file).expanduser()
    token = token_file.read_text().strip() if token_file.exists() else ""
    token_file.unlink(missing_ok=True)
    train_argv = argv[cut + 1:]
    out = Path(train_argv[train_argv.index("--out") + 1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"rank": cfg.rank, "peer_tailnet": cfg.peer_tailnet, "data_path": cfg.path,
                              "token_read": bool(token), "token_file_left": token_file.exists(),
                              "host": socket.gethostname(), "rounds": [], "started": time.time()}
    stop = threading.Event()
    threading.Thread(target=memory_guard, args=(cfg.min_free_gib, record, stop), daemon=True).start()
    link = Control(cfg.rank, cfg.peer_tailnet, cfg.ctl_port, cfg.socks, cfg.wait_s)
    box = Mailbox(cfg.box_port)
    peer_box = cfg.peer_box_port or cfg.box_port
    proxy = Fetcher(cfg.path, cfg.peer_user, peer_box, token, cfg.peer_tailnet, cfg.socks, cfg.parallel)
    relay = Fetcher("relay", cfg.peer_user, peer_box, token, cfg.peer_tailnet, cfg.socks, cfg.parallel)
    from lab import train
    answer: dict[str, Any] = {}
    try:
        train.apply_env(train.run_env())   # before transformers imports torch, as train.main would
        callback = make_callback(cfg, link, box, proxy, relay, record)
        answer = train.main(train_argv, callbacks=[callback])
    finally:
        stop.set()
        token = ""
        proxy.token = relay.token = ""
        record["ended"] = time.time()
        record["wall_s"] = round(record["ended"] - record["started"], 1)
        record["train_answer"] = {k: answer.get(k) for k in ("ok", "outcome", "error", "metrics")} if answer else None
        (out / "local_sgd.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        box.close()
        if link.sock is not None:
            link.sock.close()
    print(json.dumps({"ok": answer.get("ok"), "outcome": answer.get("outcome"), "rank": cfg.rank,
                      "rounds": len(record["rounds"]), "out": str(out)}))
    return 0 if answer.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
