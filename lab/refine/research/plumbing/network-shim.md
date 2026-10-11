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

- **[measured]** Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 (Qwen's own int4 build of the student's generation: a hybrid MoE,
  48 layers with full attention every fourth, 256 experts with 8 active, about 10B active parameters; GPTQ 4-bit,
  group 128, on the routed experts only, the rest bf16) is 78.9 GB on disk, 39 shards. It downloaded to both pods at
  50 to 55 MB/s each over plain HTTP (`HF_HUB_DISABLE_XET=1`; hf's default xet transfer ran at 15 MB/s on pod C and
  stalled at 1 MB/s on pod D), about 25 minutes. Served text-only (`--language-model-only`) as pipeline parallel 2
  with the default even split (24 and 24 layers, so each stage holds 6 full-attention and 18 linear-attention
  layers), head on pod C, the stages loaded 33.72 GiB (pod C) and 35.14 GiB (pod D) in 35 to 90 s, and at
  `--gpu-memory-utilization 0.65` (about 22 GB left free on each pod) pod C's KV cache came out at 3.71 GiB (97,133
  tokens) and pod D's at 1.99 GiB (83 linear-attention state blocks). Two things stopped the first two starts:
  vLLM's fused-MoE config lookup asks amdsmi for the device name and the venv cannot import amdsmi (fixed by putting
  the venv ROCm SDK's own `share/amd_smi` on `PYTHONPATH`), and pod D's 83 state blocks are fewer than the default
  256 sequences (fixed with `--max-num-seqs 32`). The experts run on vLLM's MoE WNA16 Triton kernels; GPTQ MoE Marlin
  is not available on this platform.
  Source: `fixtures/lab/r66-2026-10-11/pair-122b.txt`; the checkpoint's `config.json` at
  [Qwen/Qwen3.5-122B-A10B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4)
  - Implication: at 69 GiB of weights the model cannot be served from one 64 GB pod at all; split, each pod holds a
    little over half and keeps about 22 GB free. KV room is small on pod D (the stage with the output layer), so the
    pair serves a few long requests or a few dozen short ones, not R60's hundreds.
  - Candidate experiment: the same pair at utilization 0.75 (about 16 GB free per pod). Expected movement: pod D's
    KV from 1.99 GiB to about 8 GiB and its state blocks from 83 to over 300. Minimum sample size: one start.
- **[measured]** The pair served text: the first request's first token came 0.29 s after it was sent and 128 tokens
  decoded at 10.02 tok/s; a tools request came back as structured `tool_calls`. Benched with R26's harness in attach
  mode through the hub proxy (bench-v1's two text prompts, 4 requests a stream, 128 tokens, thinking off, one repeat;
  pod C's T6 sampler in the harness and pod D's over the same window): 1 stream 10.30 tok/s (10.00 total), TTFT 0.96
  s, 108.4 W on pod C plus 95.7 W on pod D, 20.4 J per token for the pair; 4 streams 5.99 tok/s each, 21.57 total, TTFT
  1.67 s, 109.7 plus 106.2 W, 10.0 J per token; 16 streams 5.60 tok/s each, 82.50 total, TTFT 3.57 s, 110.1 plus 109.6
  W, 2.66 J per token. R60's 27B on one pod with the same harness: 12.33 and 10.96 tok/s at 1 stream (8.82 J), 29.54
  total at 4 (3.27 J), 51.09 total at 16 with a 10.45 s TTFT (1.88 J).
  Source: `fixtures/lab/r66-2026-10-11/bench-pp122b-1_4-rows.jsonl`, `fixtures/lab/r66-2026-10-11/bench-pp122b-16-rows.jsonl`,
  `fixtures/lab/r66-2026-10-11/pair-122b.txt`, `lab/refine/metrics.jsonl` (prefix `pair_pp122b`), R60's rows
  `lab/refine/metrics.jsonl:187-210`
  - Implication: one stream of a 122B MoE (10B active) across two pods decodes about as fast as the 27B on one, and
    the tunnel does not show in the first token. Per-stream speed holds near 6 tok/s from 4 to 16 streams because the
    two-stage pipeline fills, so at 16 streams the pair delivers 1.6 times the 27B's tokens with a third of its first-
    token wait; it costs 1.4 times the energy per token, since two pods draw.
  - Candidate experiment: the same bench at 32 streams (the pair's sequence cap). Expected movement: total tok/s rises
    above 82 while TTFT grows; per-stream near 5. Minimum sample size: 3 repeats per setting.
- **[measured]** The live loop's labeler (`lab/label.py` through the hub proxy to pod C's port 8000, model "teacher",
  `--expect-model Qwen3.5-122B`, the text tool-calling fixture) labeled 8 examples with 4 workers in 66.3 s (8.3 s of
  wall time an example, all 8 accepted) and 32 with 16 workers in 75.0 s (2.34 s an example, all 32 accepted, about
  1,170 tokens an example, 1.9 requests an example); during the 16-worker run pod C drew 108.2 W and pod D 107.4 W,
  about 505 J per labeled example for the pair.
  Source: `fixtures/lab/r66-2026-10-11/labeling-8-4.json`, `fixtures/lab/r66-2026-10-11/labeling-32-16.json`,
  `fixtures/lab/r66-2026-10-11/pair-122b.txt`
  - Implication: 16 workers fit well inside the pair's 32-sequence cap and its KV (about 1.2k tokens a request), and
    label 3.5 times faster than 4; the live loop's act sizes follow from 2.34 s a label.
  - Candidate experiment: 32 workers on 64 examples. Expected movement: seconds per example below 2.34 if the pipeline
    still has room, flat if pod D's 83 state blocks bind. Minimum sample size: 2 runs.
- **[measured]** The L16 LoRA (4B, r 16, batch 4 by accumulation 4, 2 optimizer steps) ran beside the serving pair on
  each pod: pod C 129.1 s of trainer runtime with GPU memory peaking at 54.9 GB (41.9 GB before it), pod D 127.9 s
  peaking at 56.9 GB (43.4 GB before), against 126.5 s and 13.8 GB on pod D alone; the two runs were one after the
  other, not at once. The pair still answered after pod C's run (first token 0.28 s, 9.0 tok/s, a structured tool
  call).
  Source: `fixtures/lab/r66-2026-10-11/pair-122b.txt`
  - Implication: at utilization 0.65 each pod serves its half of the 122B and trains a 4B LoRA at the same time with
    7 to 9 GB to spare, and the step is not slower beside the pair.
  - Candidate experiment: both pods' LoRA at once while the pair labels at 16 workers. Expected movement: step time and
    labeling time each up by the share of the GPU the other takes; memory as above. Minimum sample size: 1 run.
- **[measured]** With the vision tower (no `--language-model-only`, `--limit-mm-per-prompt {"image":1}` and
  `--mm-encoder-attn-backend TRITON_ATTN` as the 27B teacher had), the stages loaded 34.58 GiB (pod C) and 36.0 GiB
  (pod D) and the KV caches came out at 3.31 GiB (157,988 tokens on pod C) and 3.26 GiB, with about 23 and 21 GB still
  free. A drawn picture (a red circle, a blue square, the word ATLAS; 170 prompt tokens) was described correctly with
  the first token at 0.83 s and 9.27 tok/s; text decoded at 10.10 tok/s with a 0.32 s first token; a tools request
  came back structured. The live loop's labeler then labeled 12 of 12 desk-objects examples with pictures through
  the hub proxy. A LoRA beside this pair failed in its DataLoader with a bus error: the pods' 64 MB `/dev/shm` was
  held to 60 MB by vLLM's message queues (`VLLM_MQ_MAX_CHUNK_BYTES_MB=2`) and by segments leaked from killed starts.
  Source: `fixtures/lab/r66-2026-10-11/pair-122b.txt`
  - Implication: the pair is a full replacement for the 27B teacher (pictures, text, tools) at about the 27B's
    single-stream speed; on these pods `/dev/shm` is the scarce resource when serving and training share a pod, so
    the queue chunks go to 1 MB and leaked `psm_*` segments are removed after any killed start.
  - Candidate experiment: the LoRA beside the pair with 1 MB queue chunks. Expected movement: the DataLoader starts
    and the step time stays near 128 s. Minimum sample size: 1 run per pod.

- **[measured]** The live loop (L29, main at 955cd8e plus PRs 435 and 437) ran end to end against the vision pair:
  the solo act (12 examples) and the duo act (24 examples, its LoRA trained as two ranks, rank 0 on pod D and rank 1
  on pod C, both beside the pair) labeled with pictures on the pair in 3.8 and 7.4 s, trained 8 steps in 109.0 and
  104.3 s, packaged in 105.7 and 84.1 s, and posted to the box in 228.4 s from the start; the two ranks swapped their
  int8 deltas through the hub proxy (32.5 MB in 0.62 s) and ended with identical weights. The pods spent 833 J an
  example on the solo act and 502 J on the duo act. While both acts packaged, the pair still decoded at 7.95 tok/s.
  Source: `fixtures/lab/r66-2026-10-11/live-loop-run5.txt`
  - Implication: one pair of pods now serves the 122B teacher and trains two LoRAs at once; the loop needed three
    fixes to get there (ports below the shim's pinned range, the trainer's DataLoader in-process because RCCL holds
    most of `/dev/shm`, and the rank runner's import path), none of them in the network itself.
  - Candidate experiment: the same loop with 15 examples a member, three runs. Expected movement: act times near
    233 s; duo energy per example stays near 0.6 of solo. Minimum sample size: 3 runs.

## Open questions

- Would a second tunnel in the other direction (server and relay on pod C) remove the D to C stall?
- Would AMD give the pods a pod-to-pod rule, which would make the tunnel unnecessary (R65's candidate experiment)?

## Search log

- 2026-10-11, about 25 minutes in pieces inside the work: tunnel tools by GitHub stars (chisel, wstunnel, frp, bore,
  gost, proxychains-ng, graftcp, redsocks), reports of NCCL or gloo over SOCKS or tunnels (none found), vLLM's
  multi-node docs and issues on hybrid models under pipeline parallel, Linux tcp(7). Stopped when chisel passed the
  checks and the four start failures each had a cause in the logs.
