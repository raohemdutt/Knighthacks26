# Sources for the plumbing research notes

Index of every source cited in the notes in this folder: `hub-api-lifecycle.md` (hub), `proxy-path.md` (proxy),
`transfers.md` (transfers), `idle-credits-recovery.md` (idle) and the measurement record
`measurements-2026-10-10.md`. Each source was opened on the date shown (the page, file or thread itself, not a
search snippet) before it was cited. Keys are stable: a note cites `[S20]`, and a pass that relies on a finding
cites the note's file and heading. Source code is cited at the tag or commit that matches what the live hub and
pods run (JupyterHub 5.4.4, jupyter_server 2.19.0, jupyter-server-proxy 4.6.0, KubeSpawner 7.0.0,
jupyterhub-idle-culler 1.4.0, CHP 5.2.0; see `hub-api-lifecycle.md` section 1), and AUP Learning Cloud at
9b4af19 with each cited block dated against the live build (M13). Keys S18 and S19 are unused.

The search ran on 2026-10-10 from about 6:45 to 7:25 AM ET, well inside the two-hour bound, plus read-only probes
of the live hub at 11:00 to 11:10 UTC (7:00 to 7:10 AM ET). It stopped when the last sources opened (terminal
culling, JupyterLab's chunk size, the token-mint handler) only refined findings already made.

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
| R1 | Read-only probes of the live hub, 2026-10-10 11:00 to 11:10 UTC (M1 to M14) | `lab/refine/research/plumbing/measurements-2026-10-10.md` | 2026-10-10 | hub, proxy, transfers, idle |
| R2 | `lab/README.md` from R2 (pod facts, launch, teacher numbers) at bb351e6 | https://github.com/raohemdutt/Knighthacks26/blob/bb351e6/lab/README.md | 2026-10-10 | hub, transfers, idle |
| R3 | PR #37, "R2: GPU lab setup (teacher on GPU A, trainer on GPU B)" (merged): smoke LoRA 0.51 samples/s, 8.7 GB peak | https://github.com/raohemdutt/Knighthacks26/pull/37 | 2026-10-10 | idle |
| R4 | R9 task page, "Verified today" line (Sat 6:35 AM) | https://app.notion.com/p/3f596ef5fec581579ba5c7a9c6d19230 | 2026-10-10 | hub |
| R5 | Ilan's Team Comms page, Q1 answer (Sat 5:25 AM): Tailscale blocked, egress, proxy path, persistence, power reads | https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 | 2026-10-10 | transfers, idle |
| R6 | `runtime/llama.py` (`LlamaServerRuntime`) at bf482f4 | https://github.com/raohemdutt/Knighthacks26/blob/bf482f4/runtime/llama.py | 2026-10-10 | proxy |
| R7 | `lab/refine/metrics.jsonl` line 13 (`trainer.samples_per_s` = 0.51, pod B) at 7c22c81 | https://github.com/raohemdutt/Knighthacks26/blob/7c22c81/lab/refine/metrics.jsonl#L13 | 2026-10-10 | idle |
| R8 | O3 plumbing rubric, `lab/refine/plumbing/RUBRIC.md` (metric set and tests T1 to T8), merged as 936fb69 | https://github.com/raohemdutt/Knighthacks26/blob/936fb69/lab/refine/plumbing/RUBRIC.md | 2026-10-10 | hub, proxy, transfers, idle |

## Opened but not cited

| Source | URL | Opened | Why not cited |
|---|---|---|---|
| Zero to JupyterHub docs, "Culling user pods" | https://z2jh.jupyter.org/en/stable/jupyterhub/customizing/user-management.html | 2026-10-10 | only says idle kernels keep servers active; the culler README and source (S06, S29) cover the rest |
| Web search results on Cloudflare and SSE buffering (a blog post and two community threads, seen only as snippets) | n/a | 2026-10-10 | not opened as pages, so not citable; the gap is recorded in `proxy-path.md` |
