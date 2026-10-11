# One model across the two Strix Halo pods: a proxy tunnel, a network shim, and what vLLM did over them

R66 (2026-10-11, about 5:40 to 8:00 AM ET) gave the two SJC pods a network between them, which the hub does not
provide (R65: every pod-to-pod connection is refused), and tried vLLM pipeline parallel across it. The tunnel and
the measurements are here; the preload shim that gives each pod a fake interface was built by R66b (branch
`ctx/runtime/network-shim-lib`) and is used here as delivered. Raw outputs are in `fixtures/lab/r66-2026-10-11/`.
Kernel speed and model quality are out of scope.

## The tunnel through the hub proxy

- **[measured]** chisel 1.12.0 (a static binary, no root) carries TCP between the pods through the hub's in-cluster
  proxy: the server on pod D listens on 127.0.0.1:29999 with `--reverse`, and the client on pod C dials
  `http://proxy-public/user/<pod D user>/proxy/29999/` with pod D's hub token in an `Authorization` header and the
  hub's host name as `Host`. Jupyter's server proxy passes the websocket (health 200 in 10 to 13 ms, client
  connected in 2.7 to 47 ms). One client connection carries 128 forward remotes (pod C's 127.0.0.1:P+200 reaches pod
  D's 127.0.0.1:P for P in 29600 to 29727) and 128 reverse remotes (the same the other way), so only one token
  moves between the pods. Through it, 256 MB went from C to D at 101.5 and 108.9 MB/s, came back at 109.6 and 96.5
  MB/s, and went from D to C at 106.0 MB/s and back at 92.3 MB/s; a one-byte request and answer took 0.33 to 0.35 ms
  median (0.28 to 0.31 minimum, p95 0.42 to 0.58), and a connection opened in 0.6 to 0.9 ms. The far end's host name
  answered each probe, so the bytes crossed the pods.
  Source: `fixtures/lab/r66-2026-10-11/tunnel-chisel-probe.txt`; chisel's README for the reverse remotes and the
  header and host options [chisel README](https://github.com/jpillora/chisel)
  - Implication: the pods now have a link of about 100 MB/s each way at a third of a millisecond, against R65's
    3.9 ms for one HTTP request through the same proxy: once the websocket is up, the proxy relays bytes without a
    per-request token check. It is about 0.8 Gbit/s, an eighth of the 10 Gbit/s the pod interface reports.
  - Candidate experiment: run 2 and 4 chisel clients side by side and split the remotes between them. Expected
    movement: aggregate throughput above 100 MB/s if the single SSH stream is the limit, flat if the proxy is.
    Minimum sample size: 3 transfers of 256 MB per setting.
- **[measured]** wstunnel, gost, a hand-written Python websocket tunnel and the Tailscale SOCKS path were not tried:
  the order was to stop at the first candidate that passed the four checks (websocket through the proxy, forward and
  reverse remotes, over 50 MB/s, round trip), and chisel passed them first. R65 measured the Tailscale floor at
  2.7 MB/s and 10 ms.
  Source: `fixtures/lab/r66-2026-10-11/tunnel-chisel-probe.txt`, `symmetric-pods.md#tailscale-between-the-pods`
  - Implication: chisel is a choice by order, not a comparison; another tool could be faster.
  - Candidate experiment: the same 256 MB and ping probe through wstunnel with a path prefix. Expected movement:
    within 10 percent of chisel if the proxy is the limit. Minimum sample size: 3 transfers.

## Forty milliseconds per small message, and the relay that removes it

- **[measured]** Through the plain tunnel, a request written as one 64-byte write took 0.36 ms median, but the same
  request written as two writes (16 then 48 bytes) took 1.67 to 2.98 ms median and 43 to 44 ms at p95 with the
  client's TCP_NODELAY on, and 44 to 45 ms median with it off. With a relay on pod D between Jupyter's proxy and the
  chisel server, which sets TCP_NODELAY on both of its sockets and re-arms TCP_QUICKACK after every read on the socket
  facing the proxy, the two-write request took 0.36 ms median and 0.57 ms at p95. The two-write pattern is gloo's and
  RCCL's (a header, then a payload).
  Source: `fixtures/lab/r66-2026-10-11/latency-two-writes.txt`, `lab/net/ackrelay.py`; Linux tcp(7) on both
  options ("disable the Nagle algorithm"; quickack "is not permanent, it only enables a switch to or from quickack
  mode") [tcp(7)](https://linux.die.net/man/7/tcp)
  - Implication: a socket inside the proxy chain that we cannot configure holds a second small segment until the
    first is acknowledged (Nagle), and the receiver's delayed acknowledgement makes that wait about 40 ms. Making
    the receiving end of that hop acknowledge at once removes it in the C to D direction. Every application socket
    must also have TCP_NODELAY (the last case shows 44 ms without it); gloo and RCCL set it themselves.
  - Candidate experiment: a second chisel tunnel the other way (client on pod D, server and relay on pod C) for the
    D to C remotes, so each direction's traffic enters a chisel server through a relay. Expected movement: the D to C
    two-write p95 (next finding) falls from about 43 ms toward 1 ms. Minimum sample size: 200 round trips each way.
- **[measured]** The other direction keeps the stall: from pod D to pod C through the reverse remotes, the two-write
  request took 0.62 ms median and 42.5 ms p95 with the relay on pod D only, 0.87 ms and 43.3 ms with a second relay on
  pod C between the chisel client and the in-cluster proxy, and 0.90 ms and 43.4 ms when that relay re-armed
  TCP_QUICKACK on both of its sockets.
  Source: `fixtures/lab/r66-2026-10-11/latency-two-writes.txt`; [tcp(7)](https://linux.die.net/man/7/tcp)
  - Implication: in the D to C direction the socket that waits for an acknowledgement sends into a hop whose receiver
    we do not run (inside the hub's proxy chain, between Jupyter's server on pod D and the hub's proxy), so no relay
    on either pod reaches it. About one small exchange in twenty from D to C pays 40 ms, which is what the gloo p95
    below shows.
  - Candidate experiment: the second tunnel in the other direction named in the finding above. Expected movement:
    D to C two-write p95 from about 43 ms to under 2 ms. Minimum sample size: 200 round trips.

## torch.distributed across the pods

- **[measured]** With R66b's preload shim (sha256 1f975c69, gcc 13.3: `getifaddrs` adds `pods0` with 10.99.0.1 on pod
  C and 10.99.0.2 on pod D, a bind on the fake address or to port 0 lands on 0.0.0.0 inside 29600 to 29727, a connect
  to the peer's fake address goes to the tunnel port P+200, `getsockname` and `getpeername` report the fake
  addresses) and `GLOO_SOCKET_IFNAME=pods0`, two gloo ranks of torch 2.12.0+rocm7.14.0 (the vLLM venv), rank 0 on pod
  C and rank 1 on pod D, master 10.99.0.1:29600, formed a process group and all-reduced correctly on both ranks
  ([3, 3, 3, 3]). On the plain tunnel: init 0.16 s, a small all-reduce 44.7 ms median (1.5 ms minimum, 92 ms p95), 64
  MB all-reduced at 41 MB/s, a 16 KB send and return 45.6 ms. With the relay: init 0.02 s, the small all-reduce 1.35
  to 1.38 ms median (1.06 ms minimum, 46 to 52 ms p95), 64 MB at 48 MB/s, 16 KB send and return 1.1 ms, 4 MB 75 to 77
  ms.
  Source: `fixtures/lab/r66-2026-10-11/gloo-gate-before-relay.txt`, `fixtures/lab/r66-2026-10-11/gloo-gate-after-relay.txt`,
  `lab/net/gloo_gate.py`
  - Implication: this is the first process group ever formed across these two pods; R65 could not form one. A
    collective costs about 1.4 ms in the median, so a stage-to-stage handoff of a few kilobytes per token is cheap,
    while the p95 still shows the 40 ms stall in the direction the relay does not cover. An all-reduce moves about
    48 MB/s, so tensor parallel (two all-reduces of a hidden vector per layer) pays about 1.4 ms per all-reduce, or
    about 0.2 s per token for a 64-layer model: pipeline parallel is the mode that fits this link.
  - Candidate experiment: the same gate with the relay on both pods. Expected movement: p95 from about 50 ms to under
    5 ms. Minimum sample size: 100 all-reduces per run, 3 runs.
- **[measured]** RCCL (the build's nccl 2.30.4) also formed across the pods through the shim inside vLLM's start, with
  `NCCL_SOCKET_IFNAME=pods0`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1` and one socket, one
  thread and one channel: the pod C stage's log shows its connects to pod D's pinned ports redirected through the
  tunnel and vLLM's profiling run across both stages completing in 78 s.
  Source: `fixtures/lab/r66-2026-10-11/pair-27b-attempts.txt`
  - Implication: vLLM's device communicator does not need a real interface either; the fake one is enough.
  - Candidate experiment: the gate script's `nccl` mode on both GPUs in one slot. Expected movement: a correct
    all-reduce with timings near gloo's. Minimum sample size: 100 all-reduces.

## vLLM pipeline parallel with the 27B beside the teacher (four attempts, none served)

- **[measured]** vLLM v0.31 (pod D's build at 5281e4990, pod C's at db9527a46) started across the pods with
  `--nnodes 2 --node-rank 0|1 --master-addr <fake address> --master-port 29727 --distributed-executor-backend mp
  --pipeline-parallel-size 2`, node 1 `--headless`, and `VLLM_HOST_IP` set to the pod's fake address. Four things
  stopped it before a token, each in one attempt: (1) the pods' `/dev/shm` is 64 MB and vLLM's message queue asks for
  160 MB (ten 16 MB chunks), fixed with `VLLM_MQ_MAX_CHUNK_BYTES_MB=2`; (2) the master port 29600 collided with the
  shim's first pinned port, fixed by moving it to 29727; (3) the teacher's 0.90 reservation leaves pod C 6.35 GiB of
  free device memory, and pod C's stage loaded 6.49 GiB with 4 layers and 6.11 GiB with 2 (its KV cache came out at
  -0.84 and -0.27 GiB); (4) a stage holding only layer 63, a full-attention layer, stopped with "no mamba layers in
  the model": vLLM's hybrid state for Qwen3.5-family models needs a linear-attention layer on every stage.
  Source: `fixtures/lab/r66-2026-10-11/pair-27b-attempts.txt`; vLLM's multi-node multiprocessing example
  [vLLM docs, Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/); the hybrid
  stage rule in a related form [vLLM issue 54710](https://github.com/vllm-project/vllm/issues/54710)
  - Implication: the network was never the blocker for the 27B; pod C's memory beside the teacher was. Every stage
    pays a fixed 5.7 GiB or so (the output layer and the embedding, about 2.4 GiB each, and the rest) before its
    layers, so a stage cannot be made small enough to fit in 6 GiB. Issue 54710 reports the mirror case (a stage with
    only linear-attention layers fails a KV-layout check), so with a hybrid model a stage needs both kinds of layer.
  - Candidate experiment: none for the 27B beside the teacher; the pair is worth it only for a model that one pod
    cannot hold, which is the next section. Expected movement: n/a.

## A model one pod cannot hold, split across both

- **[measured]** PENDING: the Qwen3.5-122B-A10B GPTQ int4 pair.
  Source: PENDING
  - Implication: PENDING
  - Candidate experiment: PENDING

## Open questions

- Would a second tunnel in the other direction (server and relay on pod C) remove the D to C stall?
- Would AMD give the pods a pod-to-pod rule, which would make the tunnel unnecessary (R65's candidate experiment)?

## Search log

- 2026-10-11, about 25 minutes in pieces inside the work: tunnel tools by GitHub stars (chisel, wstunnel, frp, bore,
  gost, proxychains-ng, graftcp, redsocks), reports of NCCL or gloo over SOCKS or tunnels (none found), vLLM's
  multi-node docs and issues on hybrid models under pipeline parallel, Linux tcp(7). Stopped when chisel passed the
  checks and the four start failures each had a cause in the logs.
