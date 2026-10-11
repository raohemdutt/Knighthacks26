# Two Strix Halo pods as a pair: the link between them, the role switch, and what a second pod adds

R65 (2026-10-11, 3:15 to about 5 AM ET) made the two SJC pods symmetric, so either can be the teacher or the trainer,
and measured what a second pod can add: a model split across the two (a bigger brain) and a training job split
across the two (a bigger lesson). Everything here was measured on the live pods through `lab/controller.py`; the raw
outputs are in `fixtures/lab/r65-2026-10-11/`. The teacher on pod C was never stopped or restarted. Kernel speed and
the teacher's quality are out of scope.

## The direct link between the pods

- **[measured]** Pod C (10.42.6.28) and pod D (10.42.1.131) sit on different Kubernetes nodes on a flannel overlay
  (eth0, MTU 1450, 10 Gbit/s reported), and every TCP connection from one to the other is refused within 0.3 ms, in
  both directions, on every port tried (a test listener on 29555, 8888, 8000, 8790, 22), while the same listener
  answers on the pod's own address. The pod hostnames do not resolve in cluster DNS, the pods have no ping, iperf3, nc
  or Kubernetes API access, and both reach the hub (8081) and the in-cluster proxy (80) in about 1 ms.
  Source: `fixtures/lab/r65-2026-10-11/reach-from-pod-C.txt`, `fixtures/lab/r65-2026-10-11/reach-from-pod-D.txt`,
  `fixtures/lab/r65-2026-10-11/facts-pod-C.txt`
  - Implication: the refusal is the hub's network policy, not distance: Zero to JupyterHub's chart lets a user pod
    accept connections only from pods labelled `hub.jupyter.org/network-access-singleuser=true` on the notebook port
    [S69]. No pod-to-pod mode (vLLM tensor or pipeline parallel, torch DDP, Ray) can form on the cluster network, and
    `link.failed_call_rate` for a direct call between the pods is 1.0.
  - Candidate experiment: ask AMD for a pod-to-pod allow rule (or the label on both pods), then rerun
    `lab/r65_link.sh probe <ip> 29555`. Expected movement: connect succeeds and a 1 GB send runs near the 10 Gbit/s
    the interface reports, against 0 today.
- **[measured]** The only path between the pods on the cluster is the hub's own proxy with the target account's hub
  token: through the in-cluster proxy (proxy-public:80) an HTTP round trip from pod C to pod D took 3.9 ms median (1.8
  minimum), 64 MB moved at 84 to 89 MB/s down and 101 to 106 MB/s up per connection (about 0.8 Gbit/s), and one response
  above about 100 MB failed (status 599 at 256 MB and 1 GB); the public hub URL took 23 ms and 73 to 88 MB/s.
  Source: `fixtures/lab/r65-2026-10-11/relay-and-tailscale-transcribed.txt`
  - Implication: good enough to copy files between the pods in pieces (a 19 GB checkpoint in about 4 minutes), but it
    is HTTP to one port through jupyter-server-proxy, so it cannot carry RCCL, gloo or Ray, which open arbitrary
    ports in both directions.
  - Candidate experiment: copy the 19 GB teacher checkpoint from pod C to a scratch folder on pod D in 64 MB parts
    over the in-cluster proxy. Expected movement: about 220 s at 88 MB/s against 281 s for D's Hugging Face download
    (R30).

## Tailscale between the pods

- **[measured]** Tailscale 1.104.1 in userspace mode (`lab/pod_tailscale.sh`, no root, as `pod-c` 100.114.98.66 and
  `pod-d` 100.65.16.109) joins both pods to the tailnet, but outbound UDP is blocked on the pods (netcheck: UDP false;
  STUN to DERP and Google times out), so no direct path forms and every packet is relayed through the DERP server in
  San Francisco: `tailscale ping` 10 to 12 ms, "direct connection not established". A TCP connection through
  tailscaled's SOCKS5 proxy to a listener on the other pod (which userspace mode hands to the same port on
  127.0.0.1) connected in 13 to 22 ms with a 10.2 to 10.3 ms round trip, and moved 2.7 MB/s (about 22 Mbit/s): 1 GB
  from C to D in 398 s, 128 MB from D to C in 48 to 70 s and back in 50 s.
  Source: `fixtures/lab/r65-2026-10-11/tailscale-probe-C-to-D.txt`,
  `fixtures/lab/r65-2026-10-11/relay-and-tailscale-transcribed.txt`
  - Implication: Tailscale routes around the hub's network policy, not around the blocked UDP: the pair gets a link
    of a few MB/s, about 35 times slower than the hub proxy and 400 times below the interface. Tailscale's own docs
    call DERP the last resort, used when a direct or peer-relay path is impossible, and say peer relays perform
    better than DERP [S71]. A copy of the 19 GB checkpoint would take about two hours. R65 therefore uses Tailscale
    only to find the friend and as the control channel (who is rank 0, when a round starts) and moves the data
    through the hub proxy.
  - Candidate experiment: with UDP egress allowed on the pods (one rule for 41641/udp and STUN 3478/udp), rerun the
    same probe. Expected movement: `tailscale ping` reports a direct path and the 1 GB send rises from 2.7 MB/s by an
    order of magnitude or more.
- **[practitioner]** Tailscale's userspace-networking mode is for containers without `/dev/net/tun`: tailscaled
  acts as a SOCKS5 or HTTP proxy that other processes connect through; the HTTP proxy carries only HTTP and HTTPS
  [S70].
  - Implication: in this mode a pod has no tailnet interface and no tailnet address of its own. RCCL and gloo bind to
    an interface and advertise that interface's address to the peer, which here is either the 10.42 pod address the
    policy refuses or 127.0.0.1, so torch.distributed and vLLM's multi-node executor cannot form over it, whatever the
    speed. Only a program that dials one known port through the proxy works, which is what the two-rank LoRA below does.
  - Candidate experiment: none on these pods without a tun device (next finding). Expected movement: n/a.
- **[measured]** Neither pod can get a tun device: `/dev/net/tun` does not exist on either pod, so opening it fails
  in the pod's own namespace and inside `unshare -Urn`, although user and network namespaces work there
  (`unshare -Urn true` succeeds, max_user_namespaces 255358). The pods have no `ip`, `slirp4netns` or `pasta`; the
  process runs with no effective capabilities and NoNewPrivs 1.
  Source: `fixtures/lab/r65-2026-10-11/tun-userns-pod-C.txt`, `fixtures/lab/r65-2026-10-11/tun-userns-pod-D.txt`
  - Implication: kernel-mode Tailscale (a real interface, and with it the standard RCCL or gloo setup over the
    tailnet) needs the hub to expose `/dev/net/tun` to the pods; a user namespace cannot make the device node.
  - Candidate experiment: if AMD exposes the device, start tailscaled with a tun inside `unshare -Urn` and run
    `torch.distributed` gloo between the pods over the tailnet addresses. Expected movement: the process group forms
    (today it cannot); its bandwidth stays at the DERP rate until UDP is allowed.

## The model split across the pods (dropped)

- **[measured]** Both builds have vLLM's multi-node flags without Ray: `--nnodes`, `--node-rank`, `--master-addr`,
  `--master-port` (arg_utils.py lines 1117 to 1120 at db9527a46 on pod C, 1130 to 1133 at 5281e4990 on pod D),
  `--headless` for the second node, and `VLLM_HOST_IP` for the address a node advertises; their CPU process groups
  use gloo (`vllm/distributed/parallel_state.py`).
  Source: `vllm/engine/arg_utils.py`, `vllm/entrypoints/cli/serve.py` and `vllm/envs.py` in `~/vllm-new` on each
  pod (vllm-project/vllm at db9527a46 and 5281e4990, the R60 builds), read there with grep on 2026-10-11
  - Implication: the engine side is ready for pipeline parallel 2 across the pods
    (`--pipeline-parallel-size 2 --nnodes 2 --node-rank 0|1 --master-addr <peer> --distributed-executor-backend mp`,
    node 1 `--headless`); what is missing is a network on which every rank can reach every other rank's advertised
    address. R65 dropped the split model on Ilan's call once the link was known.
  - Candidate experiment: with a pod-to-pod allow rule, serve a 70B-class W4A16 checkpoint pipeline-parallel 2 from
    the R60 build and bench it at 1 and 16 streams with both samplers. Expected movement: a model that one 64 GB pod
    cannot hold with KV room serves at all; per-token speed near one pod's for its half of the layers.

## The role switch

- **[measured]** Pod D can be the teacher with pod C's script and flags: `lab/pod_role.sh teacher D --apply` started
  `vllm_upgrade.sh serve` from pod D's `~/venv-vllm-new` (vLLM 0.31.2.dev0+g5281e4990, main, where pod C serves the
  v0.31.0 tag db9527a46; both carry the RDNA hybrid W4A16 kernel and the platform patch) on the same 19 GB checkpoint
  (identical file sizes and config hashes), and the teacher answered after 451 s (weights 26.7 s, KV cache 593,679
  tokens, `RDNAHybridW4A16LinearKernel`); a chat answer decoded 35 tokens in 2.81 s end to end (12.47 tok/s) and a
  tools request came back as structured `tool_calls`. Stopping it took 60 s. Pod C now has pod D's training venv, the
  same 34 packages at the same versions (pod D's also holds cmake), over the same image torch 2.9.1+rocm7.13.0,
  and the student base at 851bf6e; the install and the 8.8 GB download took 94 s.
  Source: `fixtures/lab/r65-2026-10-11/role-switch-pod-D.txt`, GPU Queue row of R65 (07:25 to 07:36 UTC)
  - Implication: a role switch costs about 8 minutes of no teacher (60 s to stop one, 451 s to start the other) plus
    the workstation's `LAB_POD_ROLES` line, which the box's teacher link and the lab verbs read; the trainer side
    costs nothing once both pods hold the venv and the base.
  - Candidate experiment: time `pod_role.sh teacher C --apply` after a stop on pod C at a quiet moment, three times.
    Expected movement: about the 467 s R60 measured on pod C, so the switch time is the engine start either way.

## The training split across the pods

- **[measured]** Two ranks, one on each pod, trained the L16 LoRA (Qwen3.5-4B, r 16, alpha 32, lr 2e-4, batch 4 by
  accumulation 4, 3 epochs, seed 42) on 42 rows each, alternate rows of L16's 84, with `lab/r65_local_sgd.py`
  wrapping `lab/train.py` unchanged. Every 4 optimizer steps each rank published its LoRA delta as an int8 code with
  error feedback, fetched the other's through the hub's in-cluster proxy while training went on, and merged it one
  step later; a final synchronous round left both ranks on the same weights (sha256 equal). Rank 0 sat on pod C beside
  the live teacher, rank 1 on pod D. Each rank ran 9 steps in 430.1 to 430.8 s (`trainer.train_s`, which includes the
  final round and a timed relay fetch of about 15 s; the trainer's own runtime was 412.9 to 414.4 s): 0.334
  samples/s a rank and 0.669 for the pair by the trainer's measure, against L16's single pod at 0.367 (18 steps,
  784.5 s). The same 252 examples took 1.82 times less wall-clock (1.89 times on the trainer's runtime). Peak memory
  was 11.0 to 11.1 GB a rank, as L16's 11.1, and pod C never fell below 24.8 GiB available.
  Source: `fixtures/lab/r65-2026-10-11/lsgd1/rank0-run.json`, `fixtures/lab/r65-2026-10-11/lsgd1/rank1-run.json`,
  `fixtures/demo/l16-run.json`, `lab/refine/metrics.jsonl:1225`
  - Implication: a second pod roughly doubles the rate a LoRA job takes in data, so in a fixed slot the pair trains
    on twice the examples one pod would, with twice the batch per step (two ranks of 16) at the same memory a rank.
    The largest batch a pod takes was not measured; at L16's 11.1 GB a rank, both pods have room for more.
  - Candidate experiment: the same run with each rank on all 84 rows (twice L16's data in the pair's 7 minutes), and
    once each with k 1 and k 9. Expected movement: `trainer.split.samples_per_s` for the pair stays near 0.67; the loss at
    equal examples moves toward L16's as k falls.
- **[measured]** The training loss did not reach the single pod's: the last step's loss was 0.028 on rank 0 and 0.023
  on rank 1 against L16's 0.0095, and the mean over the run 0.064 and 0.071 against 0.054. Each rank took 9 optimizer
  steps where L16 took 18, so the pair made half as many updates, each from twice the examples once averaged.
  Source: `fixtures/lab/r65-2026-10-11/lsgd1/rank0-run.json`, `fixtures/lab/r65-2026-10-11/lsgd1/rank1-run.json`,
  `fixtures/demo/l16-run.json`
  - Implication: the speed-up is in examples per second, not in loss per second. Whether the averaged adapter scores
    as well on held-out examples is not measured (L17's scorer), and a last-step loss is one batch's, so it is a weak
    comparison either way.
  - Candidate experiment: score `~/adapters/r65-lsgd1-rank1/adapter` on pod D and L16's adapter on the 58 held-out
    examples with L17's scorer. Expected movement: within L16's run-to-run noise if half the updates at twice the
    batch cost nothing at this data size; lower otherwise.
- **[measured]** One round's int8 code is 32,496,600 bytes (32,464,896 values and one float32 scale per 4,096). Fetched
  through the hub proxy in 4 chunks on 4 connections during training, it took 0.62 s a round (52 MB/s, 6 rounds over
  both ranks) and no rank ever waited for it; the same code over the Tailscale relay took 13.2 to 14.9 s (2.2 to 2.5
  MB/s). On idle CPUs the proxy moved it in 0.33 s with 4 connections and 0.36 s with 1, the relay in 11.6 s. Packing
  the delta (copy to the CPU, code, residual) took 3.7 to 4.1 s and blocks training; the int8 code lost 0.37 to 0.61
  percent of the delta's norm before error feedback; the faster rank waited up to 12.7 s at a round's barrier.
  Source: `fixtures/lab/r65-2026-10-11/lsgd1/rank0-local_sgd.json`,
  `fixtures/lab/r65-2026-10-11/lsgd1/rank1-local_sgd.json`,
  `fixtures/lab/r65-2026-10-11/relay-and-tailscale-transcribed.txt`
  - Implication: on this hardware the exchange is not the cost. The proxy carries a round in under a second, so the
    rounds could come every step; what costs is the CPU packing and the barrier between two ranks that do not run at
    the same speed. Over the relay alone a round would take about 14 s, a third of a step, and averaging once an
    epoch would be the right rhythm there.
  - Candidate experiment: pack the delta on the GPU (code there, copy only the int8 bytes) and rerun with k 1.
    Expected movement: pack time from about 3.8 s to under 0.5 s a round, and the pair's samples per second up by
    the time saved.
- **[measured]** The friend check finds the other pod through Tailscale: on pod C, `lab/pod_role.sh auto C` read pod
  D's tailnet address from `tailscale status` (the peer named `pod-d`), asked pod D's beacon through tailscaled's
  SOCKS5 proxy, got its answer (teacher checkpoint and training venv present, no big checkpoint), and printed its
  decision: serve the 27B alone (no pod-to-pod network for a split model), train with two ranks.
  Source: `fixtures/lab/r65-2026-10-11/pod-role-auto-C.txt`
  - Implication: the decision is a plain shell test of three inputs (the beacon answers, the big checkpoint on both
    pods, a pod-to-pod network), printed before anything acts; with Tailscale down it falls back to one pod.
  - Candidate experiment: stop tailscaled on pod D and rerun `auto C`. Expected movement: the friend reads no and the
    decision is one rank.

## Which mode pays on this hardware

- **[measured]** On these two pods the training split pays and the model split cannot be tried. The pair trained the
  same examples in 1.82 times less wall-clock with an exchange that takes under a second through the hub proxy,
  while the model split needs a network on which every rank reaches every other rank's address, which neither the
  cluster (its policy) nor Tailscale in userspace mode (no interface, DERP only) gives.
  Source: `lab/refine/metrics.jsonl:1225`, `fixtures/lab/r65-2026-10-11/reach-from-pod-C.txt`,
  `fixtures/lab/r65-2026-10-11/tun-userns-pod-C.txt`
  - Implication: data-parallel training with compressed, infrequent exchanges tolerates a slow or roundabout link,
    because a LoRA delta is 32 MB against a 45 s step; a split model sends activations at every stage for every token
    and needs a real network both ways. So "a box that finds a friend gets a bigger brain and a bigger lesson" holds
    tonight for the lesson only: a pod that finds its friend trains on twice the data per hour, with identical
    weights at the end. The bigger brain waits for a tun device, UDP egress or a pod-to-pod rule.
  - Candidate experiment: if AMD allows pod-to-pod traffic, serve a 70B-class W4A16 checkpoint pipeline-parallel 2
    and rerun this run with k 1 over the direct link. Expected movement: the split model serves where one pod cannot
    hold it with KV room; the training exchange falls from 0.62 s to under 0.1 s a round.

## Open questions

- Would AMD allow pod-to-pod traffic between one team's pods, or UDP egress, for the rest of the weekend?
- Does DERP cap a single client's throughput, and at what rate? Tailscale's page gives no number [S71].

## Search log

- 2026-10-11, about 30 minutes of reading inside the measurement work: Tailscale's userspace-networking and DERP
  pages, Zero to JupyterHub's security page; vLLM's argument parser read on the pods. Stopped when the network policy,
  the userspace mode and the missing tun device each explained one failure.
