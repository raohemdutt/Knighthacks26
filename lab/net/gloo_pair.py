"""One rank of a two-rank torch.distributed gloo check, CPU only (R66).

    python lab/net/gloo_pair.py RANK [--out FILE]

MASTER_ADDR, MASTER_PORT and GLOO_SOCKET_IFNAME come from the environment, so the same file runs with the podshim
preloaded (MASTER_ADDR 10.99.0.1, GLOO_SOCKET_IFNAME pods0), without it (the negative control) and over plain
loopback (the baseline). Prints one JSON line: the init time, a 1 MB and a 32 MB all-reduce with their results
checked, and the median round trip of a one-element all-reduce.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist

    torch.set_num_threads(4)
    rank, world = args.rank, 2
    result: dict = {"rank": rank, "torch": torch.__version__, "ifname": os.environ.get("GLOO_SOCKET_IFNAME"),
                    "master": f"{os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
                    "preload": bool(os.environ.get("LD_PRELOAD"))}
    started = time.perf_counter()
    try:
        dist.init_process_group("gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                                rank=rank, world_size=world, timeout=timedelta(seconds=args.timeout))
    except Exception as exc:  # the negative control lands here
        result.update(ok=False, init_s=round(time.perf_counter() - started, 3), error=f"{type(exc).__name__}: {exc}"[:400])
        return finish(result, args.out)
    result["init_s"] = round(time.perf_counter() - started, 3)

    def timed_all_reduce(n_floats: int, reps: int) -> tuple[list[float], bool]:
        x = torch.empty(n_floats)
        times = []
        for _ in range(reps):
            x.fill_(float(rank + 1))
            dist.barrier()
            t = time.perf_counter()
            dist.all_reduce(x)
            times.append(time.perf_counter() - t)
        return times, bool(torch.all(x == 3.0).item())  # rank 0 adds 1, rank 1 adds 2

    one_mb, ok1 = timed_all_reduce(262_144, 20)
    big, ok32 = timed_all_reduce(8_388_608, 5)
    rtt, okrt = timed_all_reduce(1, 200)
    result.update(
        ok=ok1 and ok32 and okrt,
        allreduce_1mb_ms_median=round(statistics.median(one_mb) * 1e3, 3),
        allreduce_1mb_mb_per_s=round(1 / statistics.median(one_mb), 1),
        allreduce_32mb_s=[round(t, 4) for t in big],
        allreduce_32mb_mb_per_s_median=round(32 / statistics.median(big), 1),
        roundtrip_1elem_us_median=round(statistics.median(rtt) * 1e6, 1),
        roundtrip_1elem_us_p90=round(sorted(rtt)[int(len(rtt) * 0.9)] * 1e6, 1),
        results_checked={"1mb": ok1, "32mb": ok32, "1elem": okrt},
    )
    dist.destroy_process_group()
    return finish(result, args.out)


def finish(result: dict, out: str | None) -> int:
    line = json.dumps(result)
    print(line, flush=True)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
