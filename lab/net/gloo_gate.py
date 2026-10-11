"""R66 gate: one gloo rank per pod through the network shim and the chisel tunnel.

  python gloo_gate.py RANK MASTER_ADDR MASTER_PORT [nccl]
Prints one JSON line with the init time, the all-reduce result and timings.
"""
import datetime, json, os, socket, statistics, sys, time

rank, master, port = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
backend = sys.argv[4] if len(sys.argv) > 4 else "gloo"
t_imp = time.perf_counter()
import torch
import torch.distributed as dist
out = {"rank": rank, "backend": backend, "host": socket.gethostname()[-8:], "torch": torch.__version__,
       "import_s": round(time.perf_counter() - t_imp, 2),
       "env": {k: os.environ.get(k) for k in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME") if os.environ.get(k)},
       "preload": os.path.basename(os.environ.get("LD_PRELOAD", ""))}
dev = "cpu"
if backend == "nccl":
    torch.cuda.set_device(0); dev = "cuda"
t = time.perf_counter()
try:
    dist.init_process_group(backend, init_method=f"tcp://{master}:{port}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=150))
except Exception as e:  # report where it failed and stop
    out.update(ok=False, stage="init", error=f"{type(e).__name__}: {str(e)[:600]}",
               init_s=round(time.perf_counter() - t, 2))
    print("R66JSON " + json.dumps(out)); sys.exit(1)
out["init_s"] = round(time.perf_counter() - t, 3)
x = torch.full((4,), float(rank + 1), device=dev)
t = time.perf_counter(); dist.all_reduce(x)
if dev == "cuda":
    torch.cuda.synchronize()
out["first_allreduce_ms"] = round((time.perf_counter() - t) * 1000, 2)
out["allreduce_result"] = x.tolist(); out["ok"] = x.tolist() == [3.0] * 4
lat = []
for _ in range(100):
    y = torch.ones(4, device=dev); t = time.perf_counter(); dist.all_reduce(y)
    if dev == "cuda":
        torch.cuda.synchronize()
    lat.append((time.perf_counter() - t) * 1000)
lat.sort()
out["small_allreduce_ms"] = {"min": round(lat[0], 3), "median": round(statistics.median(lat), 3), "p95": round(lat[94], 3)}
for mb in (4, 64):
    z = torch.ones((mb << 20) // 4, device=dev); dist.barrier(); t = time.perf_counter(); dist.all_reduce(z)
    if dev == "cuda":
        torch.cuda.synchronize()
    el = time.perf_counter() - t
    out[f"allreduce_{mb}MB_s"] = round(el, 3); out[f"allreduce_{mb}MB_MBps"] = round(mb * 1.048576 / el, 1)
# a point-to-point send like pipeline parallel's activations (rank 0 to rank 1 and back), 4 MB and 16 KB
for nbytes in (16 << 10, 4 << 20):
    buf = torch.zeros(nbytes // 4, device=dev); dist.barrier(); t = time.perf_counter(); reps = 20
    for _ in range(reps):
        if rank == 0:
            dist.send(buf, 1); dist.recv(buf, 1)
        else:
            dist.recv(buf, 0); dist.send(buf, 0)
    if dev == "cuda":
        torch.cuda.synchronize()
    out[f"p2p_roundtrip_{nbytes >> 10}KB_ms"] = round((time.perf_counter() - t) * 1000 / reps, 2)
dist.barrier(); dist.destroy_process_group()
print("R66JSON " + json.dumps(out))
