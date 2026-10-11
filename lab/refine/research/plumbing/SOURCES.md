# Sources for the plumbing research notes

Index of every source cited in the notes in this folder: `hub-api-lifecycle.md` (hub), `proxy-path.md` (proxy),
`transfers.md` (transfers), `idle-credits-recovery.md` (idle), `placement.md` (placement), `symmetric-pods.md` (symmetric-pods) and the measurement record
`measurements-2026-10-10.md`. Each source was opened on the date shown (the page, file or thread itself, not a
search snippet) before it was cited. Keys are stable: a note cites `[S20]`, and a pass that relies on a finding
cites the note's file and heading. Source code is cited at the tag or commit that matches what the live hub and
pods run (JupyterHub 5.4.4, jupyter_server 2.19.0, jupyter-server-proxy 4.6.0, KubeSpawner 7.0.0,
jupyterhub-idle-culler 1.4.0, CHP 5.2.0; see `hub-api-lifecycle.md` section 1), and AUP Learning Cloud at
9b4af19 with each cited block dated against the live build (M13). Keys S18 and S19 are unused.

The search ran on 2026-10-10 from about 6:45 to 7:25 AM ET, well inside the two-hour bound, plus read-only probes
of the live hub at 11:00 to 11:10 UTC (7:00 to 7:10 AM ET). It stopped when the last sources opened (terminal
culling, JupyterLab's chunk size, the token-mint handler) only refined findings already made.

Plumbing pass 1 (2026-10-10, 11:58 AM to 12:12 PM ET) added S44 to S48 and re-opened S15, S25, S27 and S36
before a change rested on them; its search log is at the end of `hub-api-lifecycle.md`.

Plumbing pass 2 (2026-10-10, 2:57 to 3:12 PM ET) added S49 to S59, re-opened S44, and wrote `contention.md`; its search
log is at the end of `hub-api-lifecycle.md`.

Plumbing pass 3 (2026-10-10, 8:30 to 9:10 PM ET) added S63 to S68 and re-opened S15 and S61; its search log is at the
end of `hub-api-lifecycle.md`.

## Vendor and project documentation

| Key | Source | Section | Type | URL | Opened | Cited in |
|---|---|---|---|---|---|---|
| S01 | JupyterHub docs, "Starting servers with the JupyterHub API" | Starting servers; Waiting for a server to start; Stopping servers | vendor doc (stable) | https://jupyterhub.readthedocs.io/en/stable/tutorial/server-api.html | 2026-10-10 | hub |
| S02 | JupyterHub docs, "Using JupyterHub's REST API" | Make an API request | vendor doc (stable) | https://jupyterhub.readthedocs.io/en/stable/howto/rest.html | 2026-10-10 | hub |
| S03 | JupyterHub docs, "Scopes in JupyterHub" | Scope conventions; Metascopes; Available scopes | vendor doc (stable) | https://jupyterhub.readthedocs.io/en/stable/rbac/scopes.html | 2026-10-10 | hub |
| S04 | jupyter-server-proxy docs, "Accessing Arbitrary Ports or Hosts" | whole page | project doc (latest) | https://jupyter-server-proxy.readthedocs.io/en/latest/arbitrary-ports-hosts.html | 2026-10-10 | proxy |
| S05 | Jupyter Server docs, "The REST API" | PUT /api/contents/{path}; /api/terminals | project doc (latest) | https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html | 2026-10-10 | transfers |
| S06 | jupyterhub-idle-culler README, tag 1.4.0 | Caveats; How it works | project doc | https://github.com/jupyterhub/jupyterhub-idle-culler/blob/1.4.0/README.md | 2026-10-10 | idle |
| S08 | ROCm docs, "AMD SMI CLI tool usage" | Commands: amd-smi metric, amd-smi process | vendor doc (ROCm) | https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html | 2026-10-10 | idle |
| S09 | ROCm k8s-device-plugin README | Example workload; Health per GPU | vendor doc (ROCm) | https://github.com/ROCm/k8s-device-plugin | 2026-10-10 | hub |
| S10 | vLLM docs, "Production Metrics" | General Metrics | project doc (latest) | https://docs.vllm.ai/en/latest/usage/metrics.html | 2026-10-10 | proxy, idle |
| S11 | vLLM docs, "torch.compile integration" | Compilation Cache; Cudagraph Capture | project doc (latest) | https://docs.vllm.ai/en/latest/design/torch_compile.html | 2026-10-10 | idle |
| S12 | Hugging Face docs, "Download files from the Hub" | Download file(s) to a local folder; Dry-run mode; Faster downloads | vendor doc | https://huggingface.co/docs/huggingface_hub/guides/download | 2026-10-10 | transfers |
| S13 | Transformers v5.19 docs, "Trainer" | Trainer.train; TrainingArguments, Checkpointing & Saving | vendor doc | https://huggingface.co/docs/transformers/en/main_classes/trainer | 2026-10-10 | idle |
| S14 | Kubernetes docs, "Pod Lifecycle" | Termination of Pods; Pod Termination Flow | vendor doc | https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination | 2026-10-10 | hub |
| S15 | Cloudflare docs, "Connection limits" | limits table | vendor doc | https://developers.cloudflare.com/fundamentals/reference/connection-limits/ | 2026-10-10 | proxy |
| S16 | Cloudflare docs, "Error 413" | Max upload size by plan | vendor doc | https://developers.cloudflare.com/support/troubleshooting/http-status-codes/4xx-client-error/error-413/ | 2026-10-10 | transfers |
| S17 | Cloudflare docs, "Error 524" | cause and resolutions | vendor doc | https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/ | 2026-10-10 | proxy |
| S44 | Cloudflare docs, "Error 502 or 504" | 502/504 from Cloudflare; 502/504 from your origin web server | vendor doc | https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-502-504/ | 2026-10-10 | hub |
| S50 | Cloudflare docs, "Error responses" | Content negotiation; Response fields; Error categories; Retry-After header | vendor doc | https://developers.cloudflare.com/fundamentals/reference/error-responses/ | 2026-10-10 | hub |
| S69 | Zero to JupyterHub docs, "Security" | Kubernetes Network Policies: the singleuser policy's ingress (only pods labelled `hub.jupyter.org/network-access-singleuser=true`, notebook port) and egress defaults | vendor doc (stable) | https://z2jh.jupyter.org/en/stable/administrator/security.html | 2026-10-11 | symmetric-pods |
| S70 | Tailscale docs, "Userspace networking mode (for containers)" | outbound through the SOCKS5 or HTTP proxy; the HTTP proxy only carries HTTP and HTTPS; meant for containers without /dev/net/tun | vendor doc | https://tailscale.com/kb/1112/userspace-networking | 2026-10-11 | symmetric-pods |
| S71 | Tailscale docs, "DERP servers" | DERP is the last resort when no direct or peer-relay path forms; peer relays offer lower latency and better performance than DERP; no throughput figure given | vendor doc | https://tailscale.com/kb/1232/derp-servers | 2026-10-11 | symmetric-pods |

## Source code at pinned versions

| Key | Source | Files and functions | URL | Opened | Cited in |
|---|---|---|---|---|---|
| S20 | JupyterHub 5.4.4 | `jupyterhub/apihandlers/users.py`: `UserServerAPIHandler.post` and `.delete`, `SpawnProgressAPIHandler`, `UserTokenListAPIHandler.post` | https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/apihandlers/users.py | 2026-10-10 | hub |
| S21 | JupyterHub 5.4.4 | `jupyterhub/handlers/base.py`: `UserUrlHandler`, `_fail_api_request` | https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/handlers/base.py | 2026-10-10 | proxy |
| S22 | JupyterHub 5.4.4 | `jupyterhub/services/auth.py`: `HubAuth.cache_max_age`, `_api_request`, `get_token`, `get_user`, `check_xsrf_cookie` | https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/services/auth.py | 2026-10-10 | hub, transfers |
| S23 | JupyterHub 5.4.4 | `jupyterhub/app.py`: `cleanup_servers`, `activity_resolution`, `last_activity_interval`, `update_last_activity`, `token_expires_in_max_seconds` | https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/app.py | 2026-10-10 | hub, idle |
| S24 | JupyterHub 5.4.4 | `jupyterhub/singleuser/extension.py`: `hub_activity_interval`; `jupyterhub/spawner.py`: `poll_interval`, `start_timeout` | https://github.com/jupyterhub/jupyterhub/blob/5.4.4/jupyterhub/spawner.py | 2026-10-10 | proxy, idle |
| S25 | jupyter-server-proxy v4.6.0 | `jupyter_server_proxy/handlers.py`: `ProxyHandler.prepare`, `_record_activity`, `proxy`, `_proxy_progressive`, `_proxy_buffered`, `proxy_request_headers`, `proxy_request_options` | https://github.com/jupyterhub/jupyter-server-proxy/blob/v4.6.0/jupyter_server_proxy/handlers.py | 2026-10-10 | proxy, idle |
| S26 | jupyter_server v2.19.0 | `serverapp.py`: `last_activity`, `max_body_size`, `max_buffer_size`, `contents_manager_class`; `base/handlers.py`: `JupyterHandler.prepare`, `check_xsrf_cookie`, `update_api_activity`, `AuthenticatedFileHandler`; `services/contents/largefilemanager.py` | https://github.com/jupyter-server/jupyter_server/tree/v2.19.0/jupyter_server | 2026-10-10 | proxy, transfers, idle |
| S27 | Tornado v6.5.10 (defaults also checked at 6.4.0 and 6.5.5) | `tornado/simple_httpclient.py`: `SimpleAsyncHTTPClient`; `tornado/web.py`: `StaticFileHandler` (Range, symlink rule) | https://github.com/tornadoweb/tornado/tree/v6.5.10/tornado | 2026-10-10 | proxy, transfers |
| S28 | KubeSpawner 7.0.0 | `kubespawner/spawner.py`: `delete_grace_period`, `stop` | https://github.com/jupyterhub/kubespawner/blob/7.0.0/kubespawner/spawner.py | 2026-10-10 | hub |
| S29 | jupyterhub-idle-culler 1.4.0 | `jupyterhub_idle_culler/__init__.py`: `cull_idle` | https://github.com/jupyterhub/jupyterhub-idle-culler/blob/1.4.0/jupyterhub_idle_culler/__init__.py | 2026-10-10 | idle |
| S30 | configurable-http-proxy 5.2.0 | `lib/configproxy.js`: `handleProxy`, `handleProxyError`, `updateLastActivity` | https://github.com/jupyterhub/configurable-http-proxy/blob/5.2.0/lib/configproxy.js | 2026-10-10 | proxy, idle |
| S31 | JupyterLab v4.5.6 | `packages/filebrowser/src/model.ts`: `CHUNK_SIZE`, `LARGE_FILE_SIZE`, chunked upload loop | https://github.com/jupyterlab/jupyterlab/blob/v4.5.6/packages/filebrowser/src/model.ts | 2026-10-10 | transfers |
| S32 | jupyter_server_terminals v0.5.4 | `jupyter_server_terminals/terminalmanager.py`: `cull_inactive_timeout`, `cull_interval` | https://github.com/jupyter-server/jupyter_server_terminals/blob/v0.5.4/jupyter_server_terminals/terminalmanager.py | 2026-10-10 | idle |
| S33 | Zero to JupyterHub 4.3.3 | `images/hub/requirements.txt` (jupyterhub, kubespawner, idle-culler, tornado pins) | https://github.com/jupyterhub/zero-to-jupyterhub-k8s/blob/4.3.3/images/hub/requirements.txt | 2026-10-10 | hub |
| S34 | AMD AUP Learning Cloud at 9b4af19 | `runtime/hub/core/spawner/kubernetes.py`: `RemoteLabKubeSpawner.start`, `.stop`, `.check_timeout`, `_build_runtime_metadata_env`, `_apply_target_path_mapping`, `_with_ephemeral_clone_cleanup` | https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/spawner/kubernetes.py | 2026-10-10 | hub, transfers, idle |
| S35 | AMD AUP Learning Cloud at 9b4af19 | `runtime/hub/core/quota/manager.py`: `can_start_container`, `end_usage_session`, `cleanup_stale_sessions` | https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/quota/manager.py | 2026-10-10 | hub, idle |
| S36 | AMD AUP Learning Cloud at 9b4af19 | `runtime/hub/core/setup.py` (stale-session cleanup at startup); `runtime/hub/core/jupyterhub_config.py` (`cleanup_servers`, culler wiring) | https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/hub/core/jupyterhub_config.py | 2026-10-10 | hub, idle |
| S37 | AMD AUP Learning Cloud at 9b4af19 | `runtime/chart/values.yaml` (`cull`, `singleuser.startTimeout`, `proxy.chp.image`); `runtime/chart/templates/proxy/deployment.yaml` | https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/runtime/chart/values.yaml | 2026-10-10 | hub, proxy, idle |
| S38 | AMD AUP Learning Cloud at 9b4af19 | `dockerfiles/Base/Dockerfile.rocm` (pins, `WORKDIR`, `CMD`); `dockerfiles/Hub/Dockerfile` (`HUB_IMAGE`) | https://github.com/AMDResearch/aup-learning-cloud/blob/9b4af19/dockerfiles/Base/Dockerfile.rocm | 2026-10-10 | hub, transfers |
| S39 | AMD AUP Learning Cloud git history | 29b463a (2026-02-06, quota and timer code already present), 6e09fad (2026-06-25, `repo_persist`), b5b9b7f8 and 351cba34 (2026-07-06, `WORKDIR` and landing paths), cff8981 (2026-07-28, `restartPolicy: Never`) | https://github.com/AMDResearch/aup-learning-cloud/commits/main | 2026-10-10 | hub, transfers |
| S45 | CPython v3.12.11 (also read at v3.13.7 and main) | `Lib/http/server.py`: `SimpleHTTPRequestHandler.send_head` (no Range handling, no 206 anywhere in the module) | https://github.com/python/cpython/blob/v3.12.11/Lib/http/server.py | 2026-10-10 | transfers |
| S46 | Zero to JupyterHub 4.3.3 | `jupyterhub/files/hub/jupyterhub_config.py`: `hub_bind_url`, `hub_connect_url` | https://github.com/jupyterhub/zero-to-jupyterhub-k8s/blob/4.3.3/jupyterhub/files/hub/jupyterhub_config.py | 2026-10-10 | idle |
| S51 | vLLM b1388b1 | `vllm/config/cache.py`: `CacheConfig.gpu_memory_utilization` (L41-L48: fraction of GPU memory, default 0.9, a per-instance limit) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/cache.py | 2026-10-10 | contention, placement |
| S60 | vLLM b1388b1 | `vllm/v1/worker/utils.py`: `request_memory` (L403-L423: asks for the device's total memory times `gpu_memory_utilization` at start and raises when the free memory is less) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/utils.py | 2026-10-10 | placement |
| S68 | jupyter_server v2.19.0 and v2.20.0 | `jupyter_server/services/contents/largefilemanager.py` (identical at both tags) and, at v2.20.0, `jupyter_server/serverapp.py` (`max_body_size` 512 MiB, L1237-L1248; `contents_manager_class` default `AsyncLargeFileManager`, L1617-L1621) | https://github.com/jupyter-server/jupyter_server/blob/v2.20.0/jupyter_server/serverapp.py#L1237-L1248 | 2026-10-10 | transfers |

## Issues and pull requests

| Key | Source | State | URL | Opened | Cited in |
|---|---|---|---|---|---|
| S40 | jupyter-server-proxy PR #479, "feat: support stream api", and changelog v4.4.0 | merged 2024-08-29 | https://github.com/jupyterhub/jupyter-server-proxy/pull/479 | 2026-10-10 | proxy |
| S41 | jupyter-server-proxy issue #502, "Responsive Proxy for other Stream Types" | open | https://github.com/jupyterhub/jupyter-server-proxy/issues/502 | 2026-10-10 | proxy |
| S42 | jupyter-server-proxy issue #281, "Connection lost (HTTP ERROR 599) when downloading large files through proxy" | open since 2021, last comment 2026-05-11 | https://github.com/jupyterhub/jupyter-server-proxy/issues/281 | 2026-10-10 | proxy |
| S43 | jupyter-server-proxy PR #282, "Add proxy_client_options" | closed unmerged 2024-06-12 | https://github.com/jupyterhub/jupyter-server-proxy/pull/282 | 2026-10-10 | proxy |

## Our own records

| Key | Source | URL or path | Opened | Cited in |
|---|---|---|---|---|
| R1 | Read-only probes of the live hub, 2026-10-10 11:00 to 11:10 UTC (M1 to M15) | `lab/refine/research/plumbing/measurements-2026-10-10.md` | 2026-10-10 | hub, proxy, transfers, idle |
| R2 | `lab/README.md` from R2 (pod facts, launch, teacher numbers) at bb351e6 | https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/README.md | 2026-10-10 | hub, transfers, idle |
| R3 | PR #37, "R2: GPU lab setup (teacher on GPU A, trainer on GPU B)" (merged): smoke LoRA 0.51 samples/s, 8.7 GB peak | https://github.com/raohemdutt/Knighthacks26/pull/37 | 2026-10-10 | idle |
| R4 | R9 task page, "Verified today" line (Sat 6:35 AM) | https://app.notion.com/p/3f596ef5fec581579ba5c7a9c6d19230 | 2026-10-10 | hub |
| R5 | Ilan's Team Comms page, Q1 answer (Sat 5:25 AM): Tailscale blocked, egress, proxy path, persistence, power reads | https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 | 2026-10-10 | transfers, idle |
| R6 | `runtime/llama.py` (`LlamaServerRuntime`) at bf482f4 | https://github.com/raohemdutt/Knighthacks26/blob/bf482f4/runtime/llama.py | 2026-10-10 | proxy |
| R7 | `lab/refine/metrics.jsonl` line 13 (`trainer.samples_per_s` = 0.51, pod B) at 7c22c81 | https://github.com/raohemdutt/Knighthacks26/blob/7c22c81/lab/refine/metrics.jsonl#L13 | 2026-10-10 | idle |
| R8 | O3 plumbing rubric, `lab/refine/plumbing/RUBRIC.md` (metric set and tests T1 to T8), merged as 936fb69 | https://github.com/raohemdutt/Knighthacks26/blob/936fb69/lab/refine/plumbing/RUBRIC.md | 2026-10-10 | hub, proxy, transfers, idle |
| S47 | Read-only probes by plumbing pass 1, 2026-10-10 15:58 to 16:04 UTC (M16 front door, M17 clock) | `lab/refine/research/plumbing/measurements-2026-10-10.md` | 2026-10-10 | hub, idle |
| S48 | R9 task page, lock line of Sat 11:24 AM ET (the hub answering 502) | https://app.notion.com/p/3f596ef5fec581579ba5c7a9c6d19230 | 2026-10-10 | hub |
| S49 | Read-only probes by plumbing pass 2, 2026-10-10 18:59 to 19:31 UTC (M18 to M24: SJC front door, accounts C and D before and after their relaunch as Strix Halo, pod C's Jupyter server, DUB's structured 502, clock, `python -m lab ready`) | `lab/refine/research/plumbing/measurements-2026-10-10.md` | 2026-10-10 | hub, transfers |
| S52 | `lab/machine/ready.py` (`ACCOUNTS`, `REQUIRED_ENV`, `HUB_URL`, L37-L41) and the sitting prompts `lab/machine/prompts/opus.md`, `fable.md`, `sonnet.md` (L12), main at 97428e4 | https://github.com/raohemdutt/Knighthacks26/blob/97428e4/lab/machine/ready.py | 2026-10-10 | hub |
| S53 | `lab/refine/metrics.jsonl` lines 9, 10, 13 and 14 (teacher engine init 270 s, teacher 84 percent of VRAM at 0.90, trainer 0.51 samples/s and 8.7 GiB peak), main at 97428e4 | https://github.com/raohemdutt/Knighthacks26/blob/97428e4/lab/refine/metrics.jsonl | 2026-10-10 | contention |
| S54 | `lab/hub.py` (L55-L62 accounts A and B and the retry pauses, L191 one hub address, L199-L200 the Accept header, L268), `lab/controller.py` (module docstring, `credit_figures`, `keep_alive`) and `jetson/egress.py` L50 (`HUB_HOSTS`), main at 97428e4 | https://github.com/raohemdutt/Knighthacks26/blob/97428e4/lab/hub.py | 2026-10-10 | hub, contention |
| S55 | `lab/README.md` L3 (one R9700, 32 GB, per pod) and L104 (engine init about 270 s, 84 percent of VRAM) and `lab/pod_a_teacher.sh` L79 (`--gpu-memory-utilization 0.90`), main at 97428e4 | https://github.com/raohemdutt/Knighthacks26/blob/97428e4/lab/pod_a_teacher.sh | 2026-10-10 | contention |
| S56 | O54 task page (accounts C and D on SJC, verified read-only at 1:52 PM ET; the Done when covers `lab/hub.py` and the controller) | https://app.notion.com/p/3f596ef5fec5813e8975ed859ff4dc01 | 2026-10-10 | pass 2 log |
| S57 | R29 task page (SJC pod C brought up from an empty home; rows on node pod-a; pods C and D up since 2:54 PM ET for 40 hours) | https://app.notion.com/p/3f596ef5fec581459f14c5afc37c7819 | 2026-10-10 | pass 2 log, placement |
| S58 | Ilan1 and Ilan2 dispatcher page (only Ilan1 touches pods; the sjc host missing from `jetson/egress.py`, a review finding on PR #142; lines of Sat 3:46, 4:00 and 4:27 PM ET by that page's clock: SJC ignores the REST launch, the pods relaunched through the web form as Strix Halo, and Ilan's decision to move the lab to them) | https://app.notion.com/p/3f596ef5fec58106826dcbd9d3d17561 | 2026-10-10 | pass 2 log |
| S59 | Ilan's Team Comms page, FYI of Sat 2:47 PM ET (the pods moved to SJC; quota off there; code must take the hub from a setting) | https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 | 2026-10-10 | pass 2 log |
| S61 | R30 task page (pod D's hour-0 facts at 21:35 UTC: 64.0 GiB visible to torch, 62 GB RAM with 58 GB available, idle 9 W; the vLLM build and checkpoint left in pod D's home; engine ready 523 s after a cold start; nothing listening on port 8000 after 22:21 UTC) | https://app.notion.com/p/3f596ef5fec58118b653e4f3ae2d3024 | 2026-10-10 | placement |
| S62 | `lab/refine/metrics.jsonl` lines 14, 24, 29, 34, 35 and 36 (LoRA peak 8.7 GiB on the R9700; on pod C the teacher's 2.88 tok/s, 120.3 W decoding, 297 s warm start, 10.2 W idle loaded, 106.8 W in request windows), main at 7208c59 | https://github.com/raohemdutt/Knighthacks26/blob/7208c59/lab/refine/metrics.jsonl | 2026-10-10 | placement |
| S63 | Read-only probes by plumbing pass 3, 2026-10-11 00:30 to 00:39 UTC (M25 SJC accounts C and D, their servers, quota and Jupyter roots; M26 clock; M27 `python -m lab ready`) | `lab/refine/research/plumbing/measurements-2026-10-10.md` | 2026-10-10 | hub, transfers |
| S64 | `lab/credits.py` (module docstring, `_assess_unmetered`), `lab/package_pull.py` and `lab/package_server.py` (module docstrings), `lab/transfer.py` (module docstring, `push_file`, the `--via` option, default `files`), `lab/placement.py` and `jetson/egress.py` (L56-L61, L119-L120), main at fba248a | https://github.com/raohemdutt/Knighthacks26/blob/fba248a/lab/transfer.py | 2026-10-10 | hub, transfers, contention |
| S65 | `lab/refine/metrics.jsonl` lines 38 to 41 (R13 on pod D: 5.55 s to ready by form launch, first request 11.7 s, keep-alive 70 min, 0 credits per hour unmetered) and 49 to 58 (R18 on pod D: the proxy's answers by case, 294.1 s restart to ready, streamed and buffered first byte, 5 MB body, 20 concurrent), main at fba248a | https://github.com/raohemdutt/Knighthacks26/blob/fba248a/lab/refine/metrics.jsonl | 2026-10-10 | hub, proxy, contention |
| S66 | `lab/README.md` at fba248a: "Controlling the pods: `lab/controller.py`" (R13's measured paragraph), "Failure-mode map of the proxy path on SJC (R18)" and "The same build on pod D, from scripts alone (R30)" | https://github.com/raohemdutt/Knighthacks26/blob/fba248a/lab/README.md | 2026-10-10 | hub, proxy |
| S67 | Task pages R10, R13, R14, R15, R17, R18, R31 and R43 as fetched by plumbing pass 3 (R18's open stopped-pod clause, Sat 7:06 PM; R43's placement rule from Ilan, signed Sat 9:08 PM ET by the Ilan1 agent's clock; R14 and R15 handed to Hemdutt's agents, Sat 5:23 PM ET) | https://app.notion.com/p/3f596ef5fec5813082b348fe147d739e | 2026-10-10 | proxy, contention |

## Opened but not cited

| Source | URL | Opened | Why not cited |
|---|---|---|---|
| Zero to JupyterHub docs, "Culling user pods" | https://z2jh.jupyter.org/en/stable/jupyterhub/customizing/user-management.html | 2026-10-10 | only says idle kernels keep servers active; the culler README and source (S06, S29) cover the rest |
| Web search results on Cloudflare and SSE buffering (a blog post and two community threads, seen only as snippets) | n/a | 2026-10-10 | not opened as pages, so not citable; the gap is recorded in `proxy-path.md` |
| The `type` URL in Cloudflare's JSON 502 body, "error-502" | https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-502/ | 2026-10-10 | answered 404 when opened (M21) |
| Web search results on Cloudflare's structured 5xx responses (snippets only) | n/a | 2026-10-10 | not opened as pages, so not citable; the reference page they pointed to is S50 |
