# R66b: the preload network shim (a section for the R66 note)

Measured on pod C on 2026-10-11 between about 5:55 and 6:05 AM ET, CPU only, two local processes, no GPU Queue slot;
the teacher on port 8000 was not touched. The raw outputs are in `fixtures/lab/r66b-2026-10-11/`.

## Why a shim

RCCL, gloo and vLLM's process groups pick an interface by name, advertise its address to the peer and bind to it
(`lab/refine/research/plumbing/symmetric-pods.md`). The pods have no interface that reaches the other pod, only a TCP
tunnel whose local end is 127.0.0.1. `lab/net/podshim.c` (299 lines, `make podshim`) is an LD_PRELOAD library that
gives each process the interface and the peer it expects:

- `getifaddrs` gains one AF_INET entry, `PODSHIM_IFNAME` (pods0) with `PODSHIM_SELF`/24, flags UP RUNNING; `freeifaddrs`
  unlinks and frees it (2000 rounds in the test). `if_nametoindex` and `if_indextoname` know the name (index 7919).
- `bind` to SELF becomes the wildcard on the same port; a bind to SELF:0 is pinned to the next free port of
  `PODSHIM_PORT_LOW` to `PODSHIM_PORT_HIGH`, so the advertised port is one the tunnel carries.
- `connect` to PEER:p goes to `PODSHIM_PEER_HOST`:(`PODSHIM_PEER_PORT_BASE` + p - `PODSHIM_PEER_PORT_LOW`); a connect to
  SELF:p goes to 127.0.0.1:p. The defaults are R66's layout: ports 29600 to 29727 and the tunnel at P + 200.
- `getsockname` and `getpeername` report SELF and PEER back, so a library that re-reads its bound address advertises
  the fake one.
- IPv4-mapped IPv6 (`::ffff:a.b.c.d`) is handled in all four socket hooks: torch's TCPStore client connects to
  MASTER_ADDR through an AF_INET6 socket, and the first run without this never reached the store.
- Real functions come from `dlsym(RTLD_NEXT)` under `pthread_once`. Per-fd state is one atomic word per descriptor (a
  stale entry from a reused fd is detected by checking the real peer). Nothing allocates except `getifaddrs`.
  `PODSHIM_LOG` writes one line per rewrite.

On both pods `~/r66/shim.so` (sha256 1f975c69..., gcc 13.3) and `~/r66/shim.env` are installed. The env file sets
`R66_SELF`, `R66_PEER`, `R66_BASE=29600`, `R66_N=128` and `R66_OFF=200` and maps them onto the `PODSHIM_` variables.
Pod C is 10.99.0.1 and pod D is 10.99.0.2.

## What it proved

- **[measured]** The C test (`lab/net/podshim_test.c`) passes all 16 checks on pod C: pods0 is listed with
  10.99.0.1/24; a SELF:0 bind is pinned to 40000 and reported as 10.99.0.1:40000; 127 of 128 range ports were pinned,
  all distinct, then refused (one was in TIME_WAIT from the run before); PEER:41005 reached a local listener on
  127.0.0.1:42005 and getpeername said 10.99.0.2:41005; the same worked through an IPv6 socket to
  `::ffff:10.99.0.2`; a one-byte round trip through the mapping took 4.4 to 6.3 us; 8 threads made 400 concurrent
  connects with no error; a bind to 127.0.0.1 was left alone. Without the shim the interface checks fail.
  Source: `fixtures/lab/r66b-2026-10-11/ctest.txt`
- **[measured]** torch.distributed gloo forms through the shim (torch 2.9.1+rocm7.13.0, CPU tensors,
  `GLOO_SOCKET_IFNAME=pods0`, `MASTER_ADDR=10.99.0.1`): rank 0 as 10.99.0.1 on ports 40000 to 40127, rank 1 as
  10.99.0.2 on 41000 to 41127, the tunnel simulated by the shim's mapping onto the other rank's real listener.
  Over four runs: init_process_group 0.010 to 0.024 s, 1 MB all-reduce 0.16 to 0.25 ms median, 32 MB all-reduce
  4.8 to 7.7 ms (4.6 to 6.3 GB/s median), one-element all-reduce 22 to 52 us median, results checked on both ranks.
  The trace shows gloo's listener bind on 10.99.0.1:0 pinned to 40001, both store clients' connects
  redirected from 10.99.0.1:40000 to 127.0.0.1:40000, and rank 1's gloo connect from 10.99.0.1:40001 to 127.0.0.1:40001.
  Source: `fixtures/lab/r66b-2026-10-11/gloo.txt`
- **[measured]** The shim costs nothing measurable here: the same pair over plain loopback (`lo`, no preload) took
  0.17 to 0.26 ms for 1 MB, 4.8 to 5.1 GB/s for 32 MB and 21 to 31 us a round trip. Without the preload, the same
  pods0 run fails after the 60 s store timeout: "client socket has timed out ... connect to (10.99.0.1, 40000)".
  Source: `fixtures/lab/r66b-2026-10-11/baseline.txt`, `fixtures/lab/r66b-2026-10-11/control.txt`
  - Implication: on the real tunnel (103 to 112 MB/s, 0.33 ms round trip), the link sets gloo's speed, not the shim:
    a 32 MB all-reduce should take about 0.3 s and a small one about 1 ms.
- **[measured]** RCCL cannot be tried without a GPU: with no device visible (`ROCR_VISIBLE_DEVICES=-1`),
  `ncclGetUniqueId` from `/opt/rocm/lib/librccl.so` fails in init.cc:224 with "HIP failure: no ROCm-capable device
  is detected" before the bootstrap picks an interface. RCCL through pods0 needs a GPU Queue slot on both pods.
  Source: `fixtures/lab/r66b-2026-10-11/rccl.txt`
- **[measured]** `tests/test_lab_podshim.py` builds the shim and the C test with gcc and runs the C test, the
  no-preload control, an unconfigured preload and the two-rank gloo pair: 4 passed in 3.3 s on pod C; on a machine
  without gcc or LD_PRELOAD it skips.
  Source: `fixtures/lab/r66b-2026-10-11/pytest.txt`

## How to run it

```
. ~/r66/shim.env
LD_PRELOAD=$HOME/r66/shim.so GLOO_SOCKET_IFNAME=pods0 MASTER_ADDR=10.99.0.1 MASTER_PORT=29600 python ...
```

To repeat the single-pod proof:
`python -m lab.controller run-script --account C --timeout 600 --with lab/net/podshim.c --with lab/net/podshim_test.c --with lab/net/gloo_pair.py --with tests/test_lab_podshim.py lab/net/r66b_pod.sh build ctest gloo baseline pytest`.
The `--with` options go before the script path, because everything after the script is passed to it as arguments.
